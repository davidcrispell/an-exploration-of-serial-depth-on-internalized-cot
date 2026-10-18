"""Fine-tune a depth-controlled GPT checkpoint on explicit GSM8K CoT.

Records must use the released ``question||reasoning #### answer`` format.  The
trainer masks the question, right-pads variable-length rows, and optimizes the
trace plus final answer with the paper's GPT-2 settings (AdamW 5e-5, effective
batch 32, FP32 by default, maximum training length 150).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tiktoken
import torch

from benchmarks.gsm8k.data import (
    collate_explicit_rows,
    file_sha256,
    load_or_build_token_cache,
    read_examples,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-new-tokens", type=int, default=800)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="do not inspect the held-out test split (useful for validation-only pilots)",
    )
    parser.add_argument("--target-validation-accuracy", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-runtime-hours", type=float)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use BF16 parameters/autocast; FP32 is the matched GPT-2 default",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GSM8K fine-tuning")
    if args.batch_size * args.accumulate != 32:
        raise ValueError("the paper protocol requires effective batch size 32")
    if args.epochs <= 0 or args.accumulate <= 0:
        raise ValueError("epochs and accumulate must be positive")

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
    validation_history: list[dict] = []
    started = time.perf_counter()
    global_step = 0
    optimizer_step = 0
    best_accuracy = float("-inf")
    best_epoch: int | None = None
    stop = False

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
            inputs, labels = collate_explicit_rows(
                cache, indices, eot_token=tokenizer.eot_token
            )
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context():
                _, loss = train_model(inputs, labels)
                (loss / args.accumulate).backward()
            global_step += 1
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
                    "stage": "explicit_cot",
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
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

        latest = args.output_dir / "latest.pt"
        save_checkpoint_atomic(
            latest,
            model,
            stage="gsm8k_explicit_cot",
            epoch=epoch,
            global_step=global_step,
            train_args=train_args,
            source_payload=source_payload,
            extra={"optimizer_step": optimizer_step},
        )
        model.eval()
        validation = evaluate_lines(
            model,
            tokenizer,
            validation_lines,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
        )
        validation.update(
            {"epoch": epoch, "global_step": global_step, "optimizer_step": optimizer_step}
        )
        validation_history.append(validation)
        write_json(args.output_dir / "validation_metrics.json", validation_history)
        append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "stage": "explicit_cot",
                "epoch": epoch,
                "global_step": global_step,
                "accuracy": validation["accuracy"],
                "correct": validation["correct"],
                "examples": validation["examples"],
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"validation epoch={epoch}: {validation['correct']}/{validation['examples']} "
            f"({validation['accuracy']:.2%})",
            flush=True,
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            save_checkpoint_atomic(
                args.output_dir / "best.pt",
                model,
                stage="gsm8k_explicit_cot",
                epoch=epoch,
                global_step=global_step,
                train_args=train_args,
                source_payload=source_payload,
                extra={
                    "optimizer_step": optimizer_step,
                    "selection_metric": "validation_exact_answer_accuracy",
                    "selection_accuracy": best_accuracy,
                },
            )
        if (
            args.target_validation_accuracy is not None
            and validation["accuracy"] >= args.target_validation_accuracy
        ):
            stop = True
        if stop:
            break

    if args.skip_test:
        write_json(
            args.output_dir / "run_summary.json",
            {
                "test_evaluated": False,
                "selected_epoch": best_epoch,
                "selection_validation_accuracy": best_accuracy,
                "total_runtime_seconds": time.perf_counter() - started,
            },
        )
        print("test skipped by request", flush=True)
    else:
        best_model, _ = load_checkpoint(
            args.output_dir / "best.pt", device, parameter_dtype
        )
        test = evaluate_lines(
            best_model,
            tokenizer,
            test_lines,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
        )
        test.update(
            {
                "selected_epoch": best_epoch,
                "selection_validation_accuracy": best_accuracy,
                "total_runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(args.output_dir / "test_metrics.json", test)
        print(
            f"test (best epoch {best_epoch}): {test['correct']}/{test['examples']} "
            f"({test['accuracy']:.2%})",
            flush=True,
        )


if __name__ == "__main__":
    main()
