"""Internalize GSM8K reasoning with the paper's token-removal curriculum.

This stage starts from an explicit-CoT checkpoint.  It removes leftmost CoT
tokens, smooths each boundary with an exponential offset, resets AdamW when
the scheduled removal advances, and switches to full removal at 39 scheduled
tokens.  Validation is always question-only greedy exact-answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tiktoken
import torch

from benchmarks.gsm8k.data import (
    file_sha256,
    load_or_build_token_cache,
    read_examples,
    remove_cot_prefix_rows,
)
from benchmarks.gsm8k.evaluate import evaluate_lines, load_checkpoint
from benchmarks.gsm8k.training import (
    append_jsonl,
    load_model,
    make_optimizer,
    save_checkpoint_atomic,
    seed_everything,
    serializable_args,
    write_json,
)


def smoothing_distribution(rate: float, truncate_length: int = 100) -> torch.Tensor:
    """Match the released truncated exponential offset distribution."""
    if rate <= 0 and not math.isinf(rate):
        raise ValueError("removal-smoothing-lambda must be positive")
    if math.isinf(rate):
        result = torch.zeros(truncate_length, dtype=torch.float32)
        result[0] = 1.0
        return result
    positions = torch.arange(truncate_length, dtype=torch.float64)
    result = (1.0 - math.exp(-rate)) * torch.exp(-rate * positions)
    result[-1] += 1.0 - result.sum()
    return result.float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument(
        "--remove-per-epoch",
        type=float,
        default=1.0,
        help="GPT-2 GSM8K paper setting; Mistral README example used 8",
    )
    parser.add_argument("--remove-start-from", type=int, default=0)
    parser.add_argument("--remove-all-when-remove-beyond", type=int, default=39)
    parser.add_argument("--removal-smoothing-lambda", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-new-tokens", type=int, default=800)
    parser.add_argument(
        "--eval-removals",
        type=str,
        help=(
            "comma-separated scheduled removals to validate before full removal; "
            "full-removal epochs are always validated"
        ),
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="do not inspect the held-out test split (useful for validation-only pilots)",
    )
    parser.add_argument("--target-validation-accuracy", type=float)
    parser.add_argument("--full-removal-patience", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-runtime-hours", type=float, default=24.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use BF16 parameters/autocast; FP32 is the matched GPT-2 default",
    )
    args = parser.parse_args()

    eval_removals = None
    if args.eval_removals is not None:
        eval_removals = {
            int(value.strip())
            for value in args.eval_removals.split(",")
            if value.strip()
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GSM8K fine-tuning")
    if args.batch_size * args.accumulate != 32:
        raise ValueError("the paper protocol requires effective batch size 32")
    if args.remove_per_epoch <= 0:
        raise ValueError("remove-per-epoch must be positive")
    if args.remove_all_when_remove_beyond <= 0:
        raise ValueError("remove-all threshold must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda")
    parameter_dtype = torch.bfloat16 if args.bf16 else torch.float32
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if args.bf16
        else nullcontext
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    cache = load_or_build_token_cache(
        args.train_path, args.cache_dir, tokenizer, max_length=args.max_length
    )
    validation_lines = read_examples(args.validation_path, args.eval_examples)
    test_lines = read_examples(args.test_path)
    model, source_payload = load_model(args.checkpoint, device, parameter_dtype)
    train_model = torch.compile(model) if args.compile else model
    optimizer = make_optimizer(model, args.lr)
    distribution = smoothing_distribution(args.removal_smoothing_lambda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = serializable_args(args)
    train_args.update(
        {
            "source_checkpoint_sha256": file_sha256(args.checkpoint),
            "train_sha256": file_sha256(args.train_path),
            "validation_sha256": file_sha256(args.validation_path),
            "test_sha256": file_sha256(args.test_path),
            "kept_training_examples": len(cache),
            "skipped_over_length": cache.skipped_over_length,
        }
    )
    write_json(args.output_dir / "training_config.json", train_args)
    metrics_path = args.output_dir / "training_metrics.jsonl"

    steps_per_epoch = len(cache) // args.batch_size
    if steps_per_epoch == 0:
        raise ValueError("training set is smaller than one minibatch")
    steps_per_removed_token = round(steps_per_epoch / args.remove_per_epoch)
    scheduled_removal = args.remove_start_from
    remove_step_counter = 0
    remove_all = scheduled_removal >= args.remove_all_when_remove_beyond
    global_step = 0
    optimizer_step = 0
    best_full_accuracy = float("-inf")
    best_full_epoch: int | None = None
    full_removal_epoch: int | None = None
    epochs_without_full_improvement = 0
    validation_history: list[dict] = []
    started = time.perf_counter()
    stop = False
    print(
        f"examples={len(cache)} steps_per_epoch={steps_per_epoch} "
        f"steps_per_removed_token={steps_per_removed_token} "
        f"remove_all_at={args.remove_all_when_remove_beyond}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(cache))
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_microbatches = 0
        interval_started = time.perf_counter()

        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                continue

            previous_removal = scheduled_removal
            was_remove_all = remove_all
            if steps_per_removed_token == 0 or remove_step_counter == steps_per_removed_token:
                scheduled_removal += 1
                remove_step_counter = 0
                remove_all = scheduled_removal >= args.remove_all_when_remove_beyond
            if scheduled_removal != previous_removal and not was_remove_all:
                optimizer.zero_grad(set_to_none=True)
                optimizer = make_optimizer(model, args.lr)
                print(
                    f"curriculum step={global_step}: scheduled_removal="
                    f"{'all' if remove_all else scheduled_removal}; optimizer reset",
                    flush=True,
                )

            if remove_all:
                offsets = np.zeros(len(indices), dtype=np.int64)
                removals = offsets
            else:
                offsets = torch.multinomial(
                    distribution, len(indices), replacement=True
                ).numpy()
                removals = offsets + scheduled_removal
            inputs, labels, actual = remove_cot_prefix_rows(
                cache,
                indices,
                removals,
                eot_token=tokenizer.eot_token,
                remove_all=remove_all,
            )
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context():
                _, loss = train_model(inputs, labels)
                (loss / args.accumulate).backward()
            global_step += 1
            remove_step_counter += 1
            running_loss += float(loss.detach())
            running_microbatches += 1

            if global_step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - interval_started
                record = {
                    "event": "train",
                    "stage": "internalized_cot",
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "scheduled_removal": "all" if remove_all else scheduled_removal,
                    "mean_actual_removal": float(actual.float().mean()),
                    "mean_smoothing_offset": float(np.mean(offsets)),
                    "loss": running_loss / running_microbatches,
                    "microbatches_per_second": running_microbatches / elapsed,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                append_jsonl(metrics_path, record)
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running_loss = 0.0
                running_microbatches = 0
                interval_started = time.perf_counter()

            runtime_exceeded = (
                args.max_runtime_hours is not None
                and time.perf_counter() - started >= args.max_runtime_hours * 3600
            )
            if (args.max_steps is not None and global_step >= args.max_steps) or runtime_exceeded:
                stop = True
                break

        if remove_all and full_removal_epoch is None:
            full_removal_epoch = epoch
        removal_label: int | str = "all" if remove_all else scheduled_removal
        save_checkpoint_atomic(
            args.output_dir / "latest.pt",
            model,
            stage="gsm8k_internalized_cot",
            epoch=epoch,
            global_step=global_step,
            train_args=train_args,
            source_payload=source_payload,
            extra={
                "optimizer_step": optimizer_step,
                "scheduled_removal": removal_label,
                "fully_internalized": remove_all,
            },
        )
        should_validate = (
            stop
            or remove_all
            or eval_removals is None
            or scheduled_removal in eval_removals
        )
        if not should_validate:
            print(
                f"validation skipped epoch={epoch} removal={removal_label}",
                flush=True,
            )
            continue
        model.eval()
        validation = evaluate_lines(
            model,
            tokenizer,
            validation_lines,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
        )
        validation.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "scheduled_removal": removal_label,
                "fully_internalized": remove_all,
            }
        )
        validation_history.append(validation)
        write_json(args.output_dir / "validation_metrics.json", validation_history)
        append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "stage": "internalized_cot",
                "epoch": epoch,
                "global_step": global_step,
                "scheduled_removal": removal_label,
                "fully_internalized": remove_all,
                "accuracy": validation["accuracy"],
                "correct": validation["correct"],
                "examples": validation["examples"],
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"validation epoch={epoch} removal={removal_label}: "
            f"{validation['correct']}/{validation['examples']} "
            f"({validation['accuracy']:.2%})",
            flush=True,
        )

        if remove_all:
            if validation["accuracy"] > best_full_accuracy:
                best_full_accuracy = validation["accuracy"]
                best_full_epoch = epoch
                epochs_without_full_improvement = 0
                save_checkpoint_atomic(
                    args.output_dir / "best_fully_internalized.pt",
                    model,
                    stage="gsm8k_internalized_cot",
                    epoch=epoch,
                    global_step=global_step,
                    train_args=train_args,
                    source_payload=source_payload,
                    extra={
                        "optimizer_step": optimizer_step,
                        "scheduled_removal": "all",
                        "fully_internalized": True,
                        "selection_metric": "validation_exact_answer_accuracy",
                        "selection_accuracy": best_full_accuracy,
                    },
                )
            else:
                epochs_without_full_improvement += 1
            if (
                args.target_validation_accuracy is not None
                and validation["accuracy"] >= args.target_validation_accuracy
            ):
                stop = True
            if (
                args.full_removal_patience is not None
                and epochs_without_full_improvement >= args.full_removal_patience
            ):
                stop = True
        if stop:
            break

    selected = (
        args.output_dir / "best_fully_internalized.pt"
        if best_full_epoch is not None
        else args.output_dir / "latest.pt"
    )
    if args.skip_test:
        write_json(
            args.output_dir / "run_summary.json",
            {
                "test_evaluated": False,
                "selected_checkpoint": str(selected),
                "selected_epoch": best_full_epoch,
                "selected_fully_internalized": best_full_epoch is not None,
                "selection_validation_accuracy": (
                    best_full_accuracy if best_full_epoch is not None else None
                ),
                "first_full_removal_epoch": full_removal_epoch,
                "total_runtime_seconds": time.perf_counter() - started,
            },
        )
        print("test skipped by request", flush=True)
    else:
        selected_model, _ = load_checkpoint(selected, device, parameter_dtype)
        test = evaluate_lines(
            selected_model,
            tokenizer,
            test_lines,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
        )
        test.update(
            {
                "selected_checkpoint": str(selected),
                "selected_epoch": best_full_epoch,
                "selected_fully_internalized": best_full_epoch is not None,
                "selection_validation_accuracy": (
                    best_full_accuracy if best_full_epoch is not None else None
                ),
                "first_full_removal_epoch": full_removal_epoch,
                "total_runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(args.output_dir / "test_metrics.json", test)
        print(
            f"test ({selected.name}): {test['correct']}/{test['examples']} "
            f"({test['accuracy']:.2%})",
            flush=True,
        )


if __name__ == "__main__":
    main()
