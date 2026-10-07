"""Internalize multiplication CoT with the paper's token-removal curriculum.

Starting from an explicit-CoT checkpoint, remove eight leftmost reasoning tokens
per epoch, smooth the boundary with an exponential random offset, and reset
AdamW whenever the scheduled removal count advances. Evaluation always starts
from the multiplication question alone, so exact-answer accuracy measures what
the model can still solve after its supervised reasoning tokens disappear.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tiktoken
import torch

from benchmarks.multiplication.cot import (
    compute_removal_distribution,
    cot_token_count,
    encode_explicit_example,
    read_examples,
    remove_cot_prefix_batch,
)
from benchmarks.multiplication.evaluate import evaluate_size
from benchmarks.multiplication.train_explicit_cot import (
    load_base,
    load_or_build_token_cache,
)


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)


def save_checkpoint_atomic(
    path: Path,
    model: torch.nn.Module,
    source_payload: dict,
    *,
    epoch: int,
    global_step: int,
    scheduled_removal: int,
    cot_tokens: int,
    train_args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "internalized_cot",
            "epoch": epoch,
            "global_step": global_step,
            "scheduled_removal": scheduled_removal,
            "cot_tokens": cot_tokens,
            "model_config": model.config,
            "model": model.state_dict(),
            "explicit_cot_checkpoint": str(train_args["checkpoint"]),
            "explicit_cot_epoch": source_payload.get("epoch"),
            "base_step": source_payload.get("base_step"),
            "train_args": train_args,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--remove-per-epoch", type=float, default=8.0)
    parser.add_argument("--remove-start-from", type=int, default=0)
    parser.add_argument("--removal-smoothing-lambda", type=float, default=4.0)
    parser.add_argument("--target-validation-accuracy", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="train in BF16; omit to match the paper's GPT-2 multiplication run",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning")
    if args.batch_size * args.accumulate != 32:
        raise ValueError("paper reproduction requires effective batch size 32")
    if args.remove_per_epoch <= 0:
        raise ValueError("remove-per-epoch must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    parameter_dtype = torch.bfloat16 if args.bf16 else torch.float32
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    token_rows = load_or_build_token_cache(args.train_path, args.cache_dir, tokenizer)
    first_ids, _ = encode_explicit_example(
        tokenizer, read_examples(args.train_path, 1)[0]
    )
    cot_tokens = cot_token_count(first_ids, tokenizer.eot_token)
    if any(
        cot_token_count(row, tokenizer.eot_token) != cot_tokens
        for row in np.asarray(token_rows[: min(1024, len(token_rows))])
    ):
        raise ValueError("training examples do not have a uniform CoT token count")

    model, source_payload = load_base(args.checkpoint, device, parameter_dtype)
    train_model = torch.compile(model) if args.compile else model
    optimizer = make_optimizer(model, args.lr)
    removal_distribution = compute_removal_distribution(
        args.removal_smoothing_lambda
    ).to(device=device, dtype=torch.float32)

    validation_lines = read_examples(args.validation_path, args.eval_examples)
    test_lines = read_examples(args.test_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(train_args, indent=2) + "\n", encoding="utf-8"
    )

    steps_per_epoch = len(token_rows) // args.batch_size
    steps_per_removed_token = int(round(steps_per_epoch / args.remove_per_epoch))
    scheduled_removal = args.remove_start_from
    remove_step_counter = 0
    global_step = 0
    stop = False
    metrics: list[dict] = []
    print(
        f"examples={len(token_rows)} steps_per_epoch={steps_per_epoch} "
        f"cot_tokens={cot_tokens} steps_per_removed_token={steps_per_removed_token}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(token_rows))
        interval_started = time.perf_counter()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                continue

            previous_removal = scheduled_removal
            if remove_step_counter == steps_per_removed_token or steps_per_removed_token == 0:
                scheduled_removal += 1
                remove_step_counter = 0
            remove_step_counter += 1
            if scheduled_removal > previous_removal and previous_removal < cot_tokens:
                optimizer.zero_grad(set_to_none=True)
                optimizer = make_optimizer(model, args.lr)
                print(
                    f"epoch={epoch} step={global_step} scheduled_removal="
                    f"{scheduled_removal} optimizer_reset=true",
                    flush=True,
                )

            encoded = torch.from_numpy(
                np.asarray(token_rows[indices], dtype=np.int64)
            ).to(device)
            offsets = torch.multinomial(
                removal_distribution, len(indices), replacement=True
            )
            requested_removals = offsets + scheduled_removal
            inputs, labels, _ = remove_cot_prefix_batch(
                encoded, requested_removals, eot_token=tokenizer.eot_token
            )
            with autocast_context():
                _, loss = train_model(inputs, labels)
                scaled_loss = loss / args.accumulate
            scaled_loss.backward()
            running_loss += loss.detach().item()
            global_step += 1
            if global_step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - interval_started
                print(
                    f"epoch={epoch} step={global_step} "
                    f"scheduled_removal={min(scheduled_removal, cot_tokens)}/{cot_tokens} "
                    f"loss={running_loss / args.log_every:.6f} "
                    f"steps_per_second={args.log_every / elapsed:.2f}",
                    flush=True,
                )
                running_loss = 0.0
                interval_started = time.perf_counter()
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        model.eval()
        validation = evaluate_size(
            model,
            tokenizer,
            validation_lines,
            digits=args.digits,
            protocol="explicit",
            batch_size=args.eval_batch_size,
            do_sample=False,
        )
        validation.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "scheduled_removal": min(scheduled_removal, cot_tokens),
                "cot_tokens": cot_tokens,
                "fully_internalized": scheduled_removal >= cot_tokens,
            }
        )
        metrics.append(validation)
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        save_checkpoint_atomic(
            args.output_dir / "latest.pt",
            model,
            source_payload,
            epoch=epoch,
            global_step=global_step,
            scheduled_removal=min(scheduled_removal, cot_tokens),
            cot_tokens=cot_tokens,
            train_args=train_args,
        )
        print(
            f"validation epoch={epoch} removed={min(scheduled_removal, cot_tokens)}/"
            f"{cot_tokens}: {validation['correct']}/{validation['examples']} "
            f"({validation['accuracy']:.4%})",
            flush=True,
        )
        if (
            scheduled_removal >= cot_tokens
            and validation["accuracy"] >= args.target_validation_accuracy
        ):
            stop = True
        if stop:
            break

    model.eval()
    test = evaluate_size(
        model,
        tokenizer,
        test_lines,
        digits=args.digits,
        protocol="explicit",
        batch_size=args.eval_batch_size,
        do_sample=False,
    )
    test.update(
        {
            "global_step": global_step,
            "scheduled_removal": min(scheduled_removal, cot_tokens),
            "cot_tokens": cot_tokens,
            "fully_internalized": scheduled_removal >= cot_tokens,
        }
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"test removed={min(scheduled_removal, cot_tokens)}/{cot_tokens}: "
        f"{test['correct']}/{test['examples']} ({test['accuracy']:.4%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
