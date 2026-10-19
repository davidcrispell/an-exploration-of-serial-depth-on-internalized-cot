"""Train the frozen-base Shallow 1.1 recurrent adapter, then internalize CoT.

Shallow 1's six transformer blocks execute twice. The first pass is frozen; on
the feedback pass, two learned 1024x1024 matrices fuse the preceding top-layer
hidden state with the current token embedding. The original 162.2M parameters
remain frozen for both recurrent warmup and the paper-matched CoT-removal
curriculum.
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
    shifted_inputs_and_labels,
)
from benchmarks.multiplication.evaluate import evaluate_size
from benchmarks.multiplication.recurrent import load_recurrent_from_explicit
from benchmarks.multiplication.train_explicit_cot import load_or_build_token_cache


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_optimizer(model, lr: float):
    return torch.optim.AdamW(trainable_parameters(model), lr=lr, fused=True)


def save_checkpoint_atomic(
    path: Path,
    model,
    source_payload: dict,
    *,
    stage: str,
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
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "scheduled_removal": scheduled_removal,
            "cot_tokens": cot_tokens,
            "model_config": model.config,
            "model": model.state_dict(),
            "prefix_mixin": model.prefix_mixin,
            "hidden_jitter": model.hidden_jitter,
            "parameter_counts": model.parameter_counts(),
            "explicit_cot_checkpoint": str(train_args["checkpoint"]),
            "explicit_cot_epoch": source_payload.get("epoch"),
            "base_step": source_payload.get("base_step"),
            "train_args": train_args,
        },
        temporary,
    )
    os.replace(temporary, path)


def evaluate(model, tokenizer, lines, args, *, stage, epoch, global_step, removed, total):
    model.eval()
    result = evaluate_size(
        model,
        tokenizer,
        lines,
        digits=args.digits,
        protocol="explicit",
        batch_size=args.eval_batch_size,
        do_sample=False,
    )
    result.update(
        {
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "scheduled_removal": removed,
            "cot_tokens": total,
            "fully_internalized": removed >= total,
        }
    )
    print(
        f"validation stage={stage} epoch={epoch} removed={removed}/{total}: "
        f"{result['correct']}/{result['examples']} ({result['accuracy']:.4%})",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--remove-per-epoch", type=float, default=8.0)
    parser.add_argument("--removal-smoothing-lambda", type=float, default=4.0)
    parser.add_argument("--target-validation-accuracy", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--hidden-jitter", type=float, default=0.02)
    parser.add_argument("--bf16", action="store_true")
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
    first_line = read_examples(args.train_path, 1)[0]
    first_ids, prompt_length = encode_explicit_example(tokenizer, first_line)
    cot_tokens = cot_token_count(first_ids, tokenizer.eot_token)
    model, source_payload = load_recurrent_from_explicit(
        args.checkpoint,
        device,
        parameter_dtype,
        prefix_mixin=True,
        hidden_jitter=args.hidden_jitter,
    )
    counts = model.parameter_counts()
    if counts["recurrent"] != 2 * model.config.n_embd**2:
        raise RuntimeError(f"unexpected recurrent parameter count: {counts}")

    validation_lines = read_examples(args.validation_path, args.eval_examples)
    test_lines = read_examples(args.test_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps({**train_args, "parameter_counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"examples={len(token_rows)} sequence={len(first_ids)} cot_tokens={cot_tokens} "
        f"parameters={counts}",
        flush=True,
    )

    global_step = 0
    metrics: list[dict] = []
    stop = False

    # First teach the two recurrent matrices to preserve the already-perfect
    # explicit-CoT behavior before removing any reasoning tokens.
    optimizer = make_optimizer(model, args.lr)
    for epoch in range(args.warmup_epochs):
        model.train()
        permutation = np.random.permutation(len(token_rows))
        interval_started = time.perf_counter()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                continue
            encoded = torch.from_numpy(
                np.asarray(token_rows[indices], dtype=np.int64)
            ).to(device)
            inputs, labels = shifted_inputs_and_labels(encoded, prompt_length)
            with autocast_context():
                _, loss = model(inputs, labels)
                scaled_loss = loss / args.accumulate
            scaled_loss.backward()
            running_loss += loss.detach().item()
            global_step += 1
            if global_step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - interval_started
                print(
                    f"stage=recurrent_warmup epoch={epoch} step={global_step} "
                    f"loss={running_loss / args.log_every:.6f} "
                    f"steps_per_second={args.log_every / elapsed:.2f}",
                    flush=True,
                )
                running_loss = 0.0
                interval_started = time.perf_counter()
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        result = evaluate(
            model, tokenizer, validation_lines, args,
            stage="recurrent_warmup", epoch=epoch, global_step=global_step,
            removed=0, total=cot_tokens,
        )
        metrics.append(result)
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        save_checkpoint_atomic(
            args.output_dir / "recurrent_warmup_latest.pt",
            model,
            source_payload,
            stage="recurrent_warmup",
            epoch=epoch,
            global_step=global_step,
            scheduled_removal=0,
            cot_tokens=cot_tokens,
            train_args=train_args,
        )
        if stop:
            break

    if not stop:
        steps_per_epoch = len(token_rows) // args.batch_size
        steps_per_removed_token = int(round(steps_per_epoch / args.remove_per_epoch))
        scheduled_removal = 0
        remove_step_counter = 0
        optimizer = make_optimizer(model, args.lr)
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
                if remove_step_counter == steps_per_removed_token:
                    scheduled_removal += 1
                    remove_step_counter = 0
                remove_step_counter += 1
                if scheduled_removal > previous_removal and previous_removal < cot_tokens:
                    optimizer = make_optimizer(model, args.lr)
                    print(
                        f"stage=internalization epoch={epoch} step={global_step} "
                        f"scheduled_removal={scheduled_removal} optimizer_reset=true",
                        flush=True,
                    )

                encoded = torch.from_numpy(
                    np.asarray(token_rows[indices], dtype=np.int64)
                ).to(device)
                offsets = torch.multinomial(
                    compute_removal_distribution(args.removal_smoothing_lambda).to(device),
                    len(indices),
                    replacement=True,
                )
                inputs, labels, _ = remove_cot_prefix_batch(
                    encoded,
                    offsets + scheduled_removal,
                    eot_token=tokenizer.eot_token,
                )
                with autocast_context():
                    _, loss = model(inputs, labels)
                    scaled_loss = loss / args.accumulate
                scaled_loss.backward()
                running_loss += loss.detach().item()
                global_step += 1
                if global_step % args.accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(model), args.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - interval_started
                    print(
                        f"stage=internalization epoch={epoch} step={global_step} "
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

            result = evaluate(
                model, tokenizer, validation_lines, args,
                stage="internalization", epoch=epoch, global_step=global_step,
                removed=min(scheduled_removal, cot_tokens), total=cot_tokens,
            )
            metrics.append(result)
            (args.output_dir / "validation_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            save_checkpoint_atomic(
                args.output_dir / "latest.pt",
                model,
                source_payload,
                stage="recurrent_internalized_cot",
                epoch=epoch,
                global_step=global_step,
                scheduled_removal=min(scheduled_removal, cot_tokens),
                cot_tokens=cot_tokens,
                train_args=train_args,
            )
            if (
                scheduled_removal >= cot_tokens
                and result["accuracy"] >= args.target_validation_accuracy
            ):
                stop = True
            if stop:
                break
    else:
        scheduled_removal = 0

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
