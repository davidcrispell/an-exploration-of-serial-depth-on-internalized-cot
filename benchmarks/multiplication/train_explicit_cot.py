"""Fine-tune the configurable-depth GPT model on explicit multiplication CoT.

This reproduces the explicit stage underlying the paper's stepwise curriculum:
GPT-2 tokenization, `input EOS CoT EOS #### answer EOS`, input-loss masking,
AdamW at 5e-5, effective batch size 32, and gradient clipping at 1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tiktoken
import torch

from model import GPT
from benchmarks.multiplication.cot import (
    encode_explicit_example,
    read_examples,
    shifted_inputs_and_labels,
    strip_compile_prefix,
)
from benchmarks.multiplication.evaluate import evaluate_size


def load_base(
    path: Path, device: torch.device, dtype: torch.dtype
) -> tuple[GPT, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(payload["model_config"])
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype), payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_cache_path(data_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{data_path.stem}-{file_sha256(data_path)[:16]}.npy"


def load_or_build_token_cache(data_path: Path, cache_dir: Path, tokenizer) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = token_cache_path(data_path, cache_dir)
    if cache_path.exists():
        print(f"loading token cache {cache_path}", flush=True)
        return np.load(cache_path, mmap_mode="r")

    lines = read_examples(data_path)
    first_ids, prompt_length = encode_explicit_example(tokenizer, lines[0])
    tokens = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint16, shape=(len(lines), len(first_ids))
    )
    for index, line in enumerate(lines):
        ids, current_prompt_length = encode_explicit_example(tokenizer, line)
        if len(ids) != len(first_ids) or current_prompt_length != prompt_length:
            raise ValueError(
                f"non-uniform tokenization at line {index + 1}: "
                f"sequence={len(ids)}, prompt={current_prompt_length}"
            )
        tokens[index] = ids
        if (index + 1) % 50_000 == 0:
            print(f"tokenized {index + 1}/{len(lines)}", flush=True)
    tokens.flush()
    print(
        f"built {cache_path}: examples={len(lines)}, sequence={len(first_ids)}, "
        f"prompt={prompt_length}",
        flush=True,
    )
    return np.load(cache_path, mmap_mode="r")


def save_checkpoint(
    path: Path,
    model: GPT,
    base_payload: dict,
    *,
    epoch: int,
    global_step: int,
    train_args: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "explicit_cot",
            "epoch": epoch,
            "global_step": global_step,
            "model_config": model.config,
            "model": model.state_dict(),
            "base_step": base_payload.get("step"),
            "train_args": train_args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="train in BF16; omitted by default to match the paper's GPT-2 multiplication command",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuning")
    if args.batch_size * args.accumulate != 32:
        raise ValueError("paper reproduction requires effective batch size 32")

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
    _, prompt_length = encode_explicit_example(tokenizer, read_examples(args.train_path, 1)[0])
    model, base_payload = load_base(args.checkpoint, device, parameter_dtype)
    train_model = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    validation_lines = read_examples(args.validation_path, args.eval_examples)
    test_lines = read_examples(args.test_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = vars(args).copy()
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in train_args.items()
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(train_args, indent=2) + "\n", encoding="utf-8"
    )

    global_step = 0
    stop = False
    metrics: list[dict] = []
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
            encoded = torch.from_numpy(np.asarray(token_rows[indices], dtype=np.int64)).to(device)
            inputs, labels = shifted_inputs_and_labels(encoded, prompt_length)
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
                    f"loss={running_loss / args.log_every:.6f} "
                    f"steps_per_second={args.log_every / elapsed:.2f}",
                    flush=True,
                )
                running_loss = 0.0
                interval_started = time.perf_counter()
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        checkpoint_path = args.output_dir / f"explicit_cot_epoch_{epoch:03d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            base_payload,
            epoch=epoch,
            global_step=global_step,
            train_args=train_args,
        )
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
        validation["epoch"] = epoch
        validation["global_step"] = global_step
        metrics.append(validation)
        (args.output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"validation epoch={epoch}: {validation['correct']}/{validation['examples']} "
            f"({validation['accuracy']:.4%})",
            flush=True,
        )
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
    test["global_step"] = global_step
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"test: {test['correct']}/{test['examples']} ({test['accuracy']:.4%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
