"""Train paper-style recurrent Shallow 1.1, then internalize 5x5 CoT.

The fixed-depth Shallow 1 checkpoint is converted into a (1, 4, 1)
prelude/core/coda architecture following Geiping et al. (2025).  All model
parameters train.  Core depth is sampled per batch with the paper's
log-normal--Poisson distribution and only the final recurrent iterations
retain an autograd graph.  After an explicit-CoT gate, the ordinary
paper-matched left-to-right CoT-removal curriculum is applied unchanged.
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
from benchmarks.multiplication.paper_recurrent import (
    load_paper_recurrent_checkpoint,
    load_paper_recurrent_from_explicit,
)
from benchmarks.multiplication.train_explicit_cot import load_or_build_token_cache


def make_optimizer(model, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)


def save_checkpoint_atomic(
    path: Path,
    model,
    source_payload: dict,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    scheduled_removal: int,
    remove_step_counter: int,
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
            "remove_step_counter": remove_step_counter,
            "cot_tokens": cot_tokens,
            "model_config": model.config,
            "model": model.state_dict(),
            "recurrent_split": [1, 4, 1],
            "mean_recurrence": model.mean_recurrence,
            "mean_backprop_depth": model.mean_backprop_depth,
            "eval_recurrence": model.eval_recurrence,
            "recurrence_sigma": model.recurrence_sigma,
            "random_state_during_training": model.random_state_during_training,
            "parameter_counts": model.parameter_counts(),
            "explicit_cot_checkpoint": str(train_args["checkpoint"]),
            "explicit_cot_epoch": source_payload.get("epoch"),
            "base_step": source_payload.get("base_step"),
            "train_args": train_args,
        },
        temporary,
    )
    os.replace(temporary, path)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def evaluate_sweep(
    model,
    tokenizer,
    lines,
    args,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    removed: int,
    total: int,
) -> list[dict]:
    model.eval()
    original_recurrence = model.eval_recurrence
    results = []
    try:
        for recurrence in args.eval_recurrences:
            model.set_eval_recurrence(recurrence)
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
                    "eval_recurrence": recurrence,
                    "effective_layers": 1 + 4 * recurrence + 1,
                }
            )
            results.append(result)
            print(
                f"validation stage={stage} epoch={epoch} removed={removed}/{total} "
                f"recurrence={recurrence} effective_layers={1 + 4 * recurrence + 1}: "
                f"{result['correct']}/{result['examples']} ({result['accuracy']:.4%})",
                flush=True,
            )
    finally:
        model.set_eval_recurrence(original_recurrence)
    return results


def primary_result(results: list[dict], recurrence: int) -> dict:
    return next(result for result in results if result["eval_recurrence"] == recurrence)


def train_epoch(
    *,
    model,
    optimizer,
    token_rows,
    prompt_length: int,
    args,
    autocast_context,
    global_step: int,
    stage: str,
    epoch: int,
    cot_tokens: int,
    scheduled_removal: int = 0,
    remove_step_counter: int = 0,
    steps_per_removed_token: int | None = None,
) -> tuple[int, int, int, bool]:
    model.train()
    permutation = np.random.permutation(len(token_rows))
    interval_started = time.perf_counter()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for start in range(0, len(permutation), args.batch_size):
        indices = permutation[start : start + args.batch_size]
        if len(indices) < args.batch_size:
            continue

        if stage == "internalization":
            previous_removal = scheduled_removal
            if remove_step_counter == steps_per_removed_token:
                scheduled_removal += 1
                remove_step_counter = 0
            remove_step_counter += 1
            if scheduled_removal > previous_removal and previous_removal < cot_tokens:
                optimizer = make_optimizer(model, args.lr)
                print(
                    f"stage={stage} epoch={epoch} step={global_step} "
                    f"scheduled_removal={scheduled_removal} optimizer_reset=true",
                    flush=True,
                )

        encoded = torch.from_numpy(
            np.asarray(token_rows[indices], dtype=np.int64)
        ).to("cuda")
        if stage == "internalization":
            offsets = torch.multinomial(
                compute_removal_distribution(args.removal_smoothing_lambda).to("cuda"),
                len(indices),
                replacement=True,
            )
            inputs, labels, _ = remove_cot_prefix_batch(
                encoded,
                offsets + scheduled_removal,
                eot_token=args.eot_token,
            )
        else:
            inputs, labels = shifted_inputs_and_labels(encoded, prompt_length)

        with autocast_context():
            _, loss = model(inputs, labels)
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
                f"stage={stage} epoch={epoch} step={global_step} "
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
    return global_step, scheduled_removal, remove_step_counter, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--warmup-max-epochs", type=int, default=3)
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
    parser.add_argument("--mean-recurrence", type=int, default=4)
    parser.add_argument("--mean-backprop-depth", type=int, default=4)
    parser.add_argument("--eval-recurrences", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--primary-eval-recurrence", type=int, default=4)
    parser.add_argument("--recurrence-sigma", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning")
    if args.batch_size * args.accumulate != 32:
        raise ValueError("multiplication comparison requires effective batch size 32")
    if args.primary_eval_recurrence not in args.eval_recurrences:
        raise ValueError("primary eval recurrence must appear in eval recurrences")
    if args.remove_per_epoch <= 0:
        raise ValueError("remove-per-epoch must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parameter_dtype = torch.bfloat16 if args.bf16 else torch.float32
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if args.bf16
        else nullcontext
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    args.eot_token = tokenizer.eot_token
    token_rows = load_or_build_token_cache(args.train_path, args.cache_dir, tokenizer)
    first_line = read_examples(args.train_path, 1)[0]
    first_ids, prompt_length = encode_explicit_example(tokenizer, first_line)
    cot_tokens = cot_token_count(first_ids, tokenizer.eot_token)

    resume_payload = None
    if args.resume_checkpoint:
        model, resume_payload = load_paper_recurrent_checkpoint(
            args.resume_checkpoint, torch.device("cuda"), parameter_dtype
        )
        source_payload = {
            "epoch": resume_payload.get("explicit_cot_epoch"),
            "base_step": resume_payload.get("base_step"),
        }
    else:
        model, source_payload = load_paper_recurrent_from_explicit(
            args.checkpoint,
            torch.device("cuda"),
            parameter_dtype,
            mean_recurrence=args.mean_recurrence,
            mean_backprop_depth=args.mean_backprop_depth,
            eval_recurrence=args.primary_eval_recurrence,
            recurrence_sigma=args.recurrence_sigma,
        )
    counts = model.parameter_counts()

    validation_lines = read_examples(args.validation_path, args.eval_examples)
    test_lines = read_examples(args.test_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "eot_token"
    }
    write_json(
        args.output_dir / "training_config.json",
        {
            **train_args,
            "paper": "https://arxiv.org/abs/2502.05171",
            "recurrent_split": [1, 4, 1],
            "parameter_counts": counts,
        },
    )
    print(
        f"examples={len(token_rows)} sequence={len(first_ids)} cot_tokens={cot_tokens} "
        f"parameters={counts} recurrent_split=(1,4,1) "
        f"mean_recurrence={model.mean_recurrence} "
        f"mean_backprop_depth={model.mean_backprop_depth}",
        flush=True,
    )

    metrics_path = args.output_dir / "validation_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if resume_payload is not None and metrics_path.exists()
        else []
    )
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0
    stop = False
    warmup_passed = resume_payload is not None and resume_payload.get("stage") != "recurrent_warmup"
    if resume_payload is not None and resume_payload.get("stage") == "recurrent_warmup":
        warmup_epoch = int(resume_payload.get("epoch", -1))
        prior = [
            result
            for result in metrics
            if result.get("stage") == "recurrent_warmup"
            and result.get("epoch") == warmup_epoch
            and result.get("eval_recurrence") == args.primary_eval_recurrence
        ]
        warmup_passed = bool(
            prior and prior[-1]["accuracy"] >= args.target_validation_accuracy
        )

    if resume_payload is None:
        optimizer = make_optimizer(model, args.lr)
        for epoch in range(args.warmup_max_epochs):
            global_step, _, _, stop = train_epoch(
                model=model,
                optimizer=optimizer,
                token_rows=token_rows,
                prompt_length=prompt_length,
                args=args,
                autocast_context=autocast_context,
                global_step=global_step,
                stage="recurrent_warmup",
                epoch=epoch,
                cot_tokens=cot_tokens,
            )
            epoch_results = evaluate_sweep(
                model,
                tokenizer,
                validation_lines,
                args,
                stage="recurrent_warmup",
                epoch=epoch,
                global_step=global_step,
                removed=0,
                total=cot_tokens,
            )
            metrics.extend(epoch_results)
            write_json(metrics_path, metrics)
            save_checkpoint_atomic(
                args.output_dir / "recurrent_warmup_latest.pt",
                model,
                source_payload,
                stage="recurrent_warmup",
                epoch=epoch,
                global_step=global_step,
                scheduled_removal=0,
                remove_step_counter=0,
                cot_tokens=cot_tokens,
                train_args=train_args,
            )
            warmup_passed = (
                primary_result(epoch_results, args.primary_eval_recurrence)["accuracy"]
                >= args.target_validation_accuracy
            )
            if warmup_passed or stop:
                break

    if not warmup_passed and not stop:
        raise RuntimeError(
            "explicit-CoT recurrent warmup failed the validation gate; "
            "refusing to begin internalization"
        )

    scheduled_removal = int(resume_payload.get("scheduled_removal", 0)) if resume_payload else 0
    remove_step_counter = int(resume_payload.get("remove_step_counter", 0)) if resume_payload else 0
    if not stop:
        steps_per_epoch = len(token_rows) // args.batch_size
        steps_per_removed_token = int(round(steps_per_epoch / args.remove_per_epoch))
        optimizer = make_optimizer(model, args.lr)
        start_epoch = (
            int(resume_payload.get("epoch", -1)) + 1
            if resume_payload and resume_payload.get("stage") == "recurrent_internalized_cot"
            else 0
        )
        for epoch in range(start_epoch, args.epochs):
            global_step, scheduled_removal, remove_step_counter, stop = train_epoch(
                model=model,
                optimizer=optimizer,
                token_rows=token_rows,
                prompt_length=prompt_length,
                args=args,
                autocast_context=autocast_context,
                global_step=global_step,
                stage="internalization",
                epoch=epoch,
                cot_tokens=cot_tokens,
                scheduled_removal=scheduled_removal,
                remove_step_counter=remove_step_counter,
                steps_per_removed_token=steps_per_removed_token,
            )
            epoch_results = evaluate_sweep(
                model,
                tokenizer,
                validation_lines,
                args,
                stage="internalization",
                epoch=epoch,
                global_step=global_step,
                removed=min(scheduled_removal, cot_tokens),
                total=cot_tokens,
            )
            metrics.extend(epoch_results)
            write_json(metrics_path, metrics)
            save_checkpoint_atomic(
                args.output_dir / "latest.pt",
                model,
                source_payload,
                stage="recurrent_internalized_cot",
                epoch=epoch,
                global_step=global_step,
                scheduled_removal=min(scheduled_removal, cot_tokens),
                remove_step_counter=remove_step_counter,
                cot_tokens=cot_tokens,
                train_args=train_args,
            )
            if (
                scheduled_removal >= cot_tokens
                and primary_result(epoch_results, args.primary_eval_recurrence)["accuracy"]
                >= args.target_validation_accuracy
            ):
                stop = True
            if stop:
                break

    test_results = []
    model.eval()
    for recurrence in args.eval_recurrences:
        model.set_eval_recurrence(recurrence)
        result = evaluate_size(
            model,
            tokenizer,
            test_lines,
            digits=args.digits,
            protocol="explicit",
            batch_size=args.eval_batch_size,
            do_sample=False,
        )
        result.update(
            {
                "global_step": global_step,
                "scheduled_removal": min(scheduled_removal, cot_tokens),
                "cot_tokens": cot_tokens,
                "fully_internalized": scheduled_removal >= cot_tokens,
                "eval_recurrence": recurrence,
                "effective_layers": 1 + 4 * recurrence + 1,
            }
        )
        test_results.append(result)
        print(
            f"test removed={min(scheduled_removal, cot_tokens)}/{cot_tokens} "
            f"recurrence={recurrence}: {result['correct']}/{result['examples']} "
            f"({result['accuracy']:.4%})",
            flush=True,
        )
    write_json(args.output_dir / "test_metrics.json", test_results)


if __name__ == "__main__":
    main()
