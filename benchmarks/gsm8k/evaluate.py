"""Question-only greedy evaluation for the GSM8K CoT curriculum.

The evaluator never supplies a ground-truth reasoning trace.  It prompts with
only ``question EOS`` and lets the model produce either a visible trace or an
empty trace, followed by ``EOS #### answer EOS``.  This makes the same metric
valid before, during, and after CoT-token removal.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch

from benchmarks.gsm8k.data import (
    file_sha256,
    input_prompt,
    normalize_answer,
    parse_example,
    read_examples,
)
from benchmarks.multiplication.cot import find_subsequence, strip_compile_prefix
from model import GPT


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def load_checkpoint(
    path: Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[GPT, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(payload["model_config"])
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype).eval(), payload


def safe_decode(tokenizer, ids: Sequence[int]) -> str:
    """Decode valid GPT-2 ids without crashing on enlarged-vocabulary ids."""
    pieces: list[str] = []
    valid: list[int] = []
    for token in ids:
        if int(token) <= tokenizer.eot_token:
            valid.append(int(token))
        else:
            if valid:
                pieces.append(tokenizer.decode(valid))
                valid = []
            pieces.append(f"<extra_{int(token)}>")
    if valid:
        pieces.append(tokenizer.decode(valid))
    return "".join(pieces)


def answer_tokens(
    generated: Sequence[int], marker: Sequence[int], eot_token: int
) -> list[int] | None:
    try:
        trace_end = list(generated).index(eot_token)
    except ValueError:
        return None
    answer_region = generated[trace_end + 1 :]
    marker_start = find_subsequence(answer_region, marker)
    if marker_start is None:
        return None
    result: list[int] = []
    for token in answer_region[marker_start + len(marker) :]:
        if int(token) == eot_token:
            break
        result.append(int(token))
    return result


@torch.inference_mode()
def generate_until_two_eos(
    model: GPT,
    prompts: list[list[int]],
    *,
    eot_token: int,
    max_new_tokens: int,
) -> list[list[int]]:
    """Greedily generate until every row has emitted two continuation EOSs."""
    lengths = {len(prompt) for prompt in prompts}
    if len(lengths) != 1:
        raise ValueError(f"prompt lengths must match within a batch, got {lengths}")
    device = next(model.parameters()).device
    parameter_dtype = next(model.parameters()).dtype
    sequences = torch.tensor(prompts, dtype=torch.long, device=device)
    eos_counts = torch.zeros(len(prompts), dtype=torch.long, device=device)
    continuations: list[list[int]] = [[] for _ in prompts]
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and parameter_dtype == torch.bfloat16
            else nullcontext()
        )
        with autocast:
            logits, _ = model(sequences)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(
            finished, torch.full_like(next_token, eot_token), next_token
        )
        token_values = next_token.tolist()
        was_finished = finished.tolist()
        for row, token in enumerate(token_values):
            if not was_finished[row]:
                continuations[row].append(token)
        eos_counts += (~finished & next_token.eq(eot_token)).long()
        finished |= eos_counts.ge(2)
        sequences = torch.cat((sequences, next_token[:, None]), dim=1)
        if bool(finished.all()):
            break
    return continuations


@torch.inference_mode()
def evaluate_lines(
    model: GPT,
    tokenizer,
    lines: list[str],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 800,
) -> dict:
    """Evaluate exact final-answer accuracy while preserving source order."""
    if not lines:
        raise ValueError("cannot evaluate an empty dataset")
    examples = [parse_example(line) for line in lines]
    eot = tokenizer.decode([tokenizer.eot_token])
    prompts = [
        tokenizer.encode(input_prompt(example, eot), allowed_special={eot})
        for example in examples
    ]
    marker = tokenizer.encode(" ####")
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt)].append(index)

    generated_by_index: list[list[int] | None] = [None] * len(examples)
    started = time.perf_counter()
    for indices in by_length.values():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            generated = generate_until_two_eos(
                model,
                [prompts[index] for index in batch_indices],
                eot_token=tokenizer.eot_token,
                max_new_tokens=max_new_tokens,
            )
            for index, continuation in zip(batch_indices, generated):
                generated_by_index[index] = continuation

    correct = 0
    correct_by_example: list[bool] = []
    samples: list[dict] = []
    failures: list[dict] = []
    truncated = 0
    empty_traces = 0
    visible_reasoning_tokens: list[int] = []
    for example, generated in zip(examples, generated_by_index):
        assert generated is not None
        prediction_tokens = answer_tokens(
            generated, marker=marker, eot_token=tokenizer.eot_token
        )
        prediction = (
            None if prediction_tokens is None else safe_decode(tokenizer, prediction_tokens)
        )
        is_correct = prediction is not None and normalize_answer(
            prediction
        ) == normalize_answer(example.answer)
        correct += int(is_correct)
        correct_by_example.append(is_correct)
        truncated += int(generated.count(tokenizer.eot_token) < 2)
        try:
            trace_tokens = generated.index(tokenizer.eot_token)
        except ValueError:
            trace_tokens = len(generated)
        visible_reasoning_tokens.append(trace_tokens)
        empty_traces += int(trace_tokens == 0)
        record = {
            "question": example.question,
            "target": example.answer,
            "prediction": prediction,
            "raw_generation": safe_decode(tokenizer, generated),
            "visible_reasoning_tokens": trace_tokens,
            "correct": is_correct,
        }
        if len(samples) < 10:
            samples.append(record)
        if not is_correct and len(failures) < 10:
            failures.append(record)

    elapsed = time.perf_counter() - started
    return {
        "protocol": "question_only_greedy",
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "empty_trace_count": empty_traces,
        "empty_trace_rate": empty_traces / len(examples),
        "mean_visible_reasoning_tokens": sum(visible_reasoning_tokens)
        / len(visible_reasoning_tokens),
        "truncated_generations": truncated,
        "max_new_tokens": max_new_tokens,
        "seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "correct_by_example": correct_by_example,
        "samples": samples,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    model, payload = load_checkpoint(args.checkpoint, device, dtype)
    tokenizer = tiktoken.get_encoding("gpt2")
    result = evaluate_lines(
        model,
        tokenizer,
        read_examples(args.data_path, args.limit),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_stage": payload.get("stage"),
            "data_path": str(args.data_path),
            "data_sha256": file_sha256(args.data_path),
        }
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
