"""Evaluate a modded-nanoGPT checkpoint on held-out multiplication data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import tiktoken
import torch

from model import GPT
from benchmarks.multiplication.cot import (
    answer_tokens_from_explicit_generation,
    batched,
    direct_answer_prompt,
    direct_answer_target,
    explicit_text,
    input_prompt,
    parse_example,
    read_examples,
    strip_compile_prefix,
)


def load_checkpoint(path: Path, device: torch.device, dtype: torch.dtype) -> GPT:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(payload["model_config"])
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def generate(
    model: GPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    do_sample: bool,
) -> torch.Tensor:
    lengths = {len(prompt) for prompt in prompts}
    if len(lengths) != 1:
        raise ValueError(f"all prompts in a batch must have equal length, got {lengths}")
    device = next(model.parameters()).device
    sequences = torch.tensor(prompts, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(sequences)
        next_logits = logits[:, -1, :]
        if do_sample:
            next_token = torch.multinomial(next_logits.softmax(dim=-1), 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        sequences = torch.cat((sequences, next_token), dim=1)
    return sequences[:, next(iter(lengths)) :].cpu()


def safe_decode(tokenizer, ids: list[int]) -> str:
    pieces: list[str] = []
    valid: list[int] = []
    for token in ids:
        if token <= tokenizer.eot_token:
            valid.append(token)
        else:
            if valid:
                pieces.append(tokenizer.decode(valid))
                valid = []
            pieces.append(f"<extra_{token}>")
    if valid:
        pieces.append(tokenizer.decode(valid))
    return "".join(pieces)


def answer_matches(tokenizer, prediction: list[int] | None, expected: str) -> bool:
    """Compare the numeric field while ignoring format-required boundary spaces."""
    return prediction is not None and safe_decode(tokenizer, prediction).strip() == expected.strip()


def evaluate_size(
    model: GPT,
    tokenizer,
    lines: list[str],
    *,
    digits: int,
    protocol: str,
    batch_size: int,
    do_sample: bool,
) -> dict:
    examples = [parse_example(line) for line in lines]
    eot = tokenizer.decode([tokenizer.eot_token])
    allowed_special = {eot}

    if protocol == "direct":
        prompts = [
            tokenizer.encode(direct_answer_prompt(example, eot), allowed_special=allowed_special)
            for example in examples
        ]
        targets = [
            tokenizer.encode(direct_answer_target(example, eot), allowed_special=allowed_special)
            for example in examples
        ]
        max_new_tokens = len(targets[0])
        if any(len(target) != max_new_tokens for target in targets):
            raise ValueError("direct-answer targets have inconsistent lengths")
        marker = None
    else:
        prompts = [
            tokenizer.encode(input_prompt(example, eot), allowed_special=allowed_special)
            for example in examples
        ]
        full_targets = [
            tokenizer.encode(explicit_text(example, eot), allowed_special=allowed_special)
            for example in examples
        ]
        max_new_tokens = len(full_targets[0]) - len(prompts[0])
        if any(len(full) - len(prompt) != max_new_tokens for full, prompt in zip(full_targets, prompts)):
            raise ValueError("explicit-CoT continuations have inconsistent lengths")
        targets = [tokenizer.encode(f" {example.answer}") for example in examples]
        marker = tokenizer.encode(" ####")

    correct = 0
    samples: list[dict] = []
    started = time.perf_counter()
    offset = 0
    for prompt_batch in batched(prompts, batch_size):
        generated_batch = generate(
            model,
            list(prompt_batch),
            max_new_tokens,
            do_sample=do_sample,
        )
        for generated_tensor in generated_batch:
            generated = generated_tensor.tolist()
            target = targets[offset]
            if protocol == "direct":
                prediction = generated
                is_correct = prediction == target
            else:
                prediction = answer_tokens_from_explicit_generation(
                    generated, marker, tokenizer.eot_token
                )
                is_correct = answer_matches(tokenizer, prediction, examples[offset].answer)
            correct += int(is_correct)
            if len(samples) < 10:
                samples.append(
                    {
                        "source": examples[offset].source,
                        "target": examples[offset].answer,
                        "prediction": None if prediction is None else safe_decode(tokenizer, prediction),
                        "raw_generation": safe_decode(tokenizer, generated),
                        "correct": is_correct,
                    }
                )
            offset += 1

    elapsed = time.perf_counter() - started
    return {
        "digits": digits,
        "protocol": protocol,
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "max_new_tokens": max_new_tokens,
        "seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "samples": samples,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--digits", type=int, nargs="+", default=[4, 5, 7, 9, 11])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--protocol", choices=["direct", "explicit"], default="direct")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark generation")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_checkpoint(args.checkpoint, device, torch.bfloat16)

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "do_sample": args.do_sample,
        "results": [],
    }
    for digits in args.digits:
        path = args.data_root / f"{digits}x{digits}" / f"{args.split}.txt"
        lines = read_examples(path, args.limit)
        result = evaluate_size(
            model,
            tokenizer,
            lines,
            digits=digits,
            protocol=args.protocol,
            batch_size=args.batch_size,
            do_sample=args.do_sample,
        )
        results["results"].append(result)
        print(
            f"{digits}x{digits}: {result['correct']}/{result['examples']} "
            f"({result['accuracy']:.4%}), {result['examples_per_second']:.1f} examples/s",
            flush=True,
        )

    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
