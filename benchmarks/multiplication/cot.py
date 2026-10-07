"""Shared data and evaluation helpers for multiplication CoT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


IGNORE_INDEX = -1


@dataclass(frozen=True)
class MultiplicationExample:
    source: str
    cot: str
    answer: str


def parse_example(line: str) -> MultiplicationExample:
    source, target = line.strip().split("||", maxsplit=1)
    cot, answer = target.split(" #### ", maxsplit=1)
    return MultiplicationExample(source=source, cot=cot, answer=answer)


def explicit_text(example: MultiplicationExample, eot: str) -> str:
    """Match the paper's `input EOS CoT EOS #### answer EOS` format."""
    return (
        f" {example.source} {eot} {example.cot} "
        f"{eot} #### {example.answer} {eot}"
    )


def input_prompt(example: MultiplicationExample, eot: str) -> str:
    return f" {example.source} {eot}"


def direct_answer_prompt(example: MultiplicationExample, eot: str) -> str:
    """Prompt the untreated model at the start of the answer field."""
    return f" {example.source} {eot} ####"


def direct_answer_target(example: MultiplicationExample, eot: str) -> str:
    return f" {example.answer} {eot}"


def encode_explicit_example(tokenizer, line: str) -> tuple[list[int], int]:
    example = parse_example(line)
    eot = tokenizer.decode([tokenizer.eot_token])
    allowed_special = {eot}
    ids = tokenizer.encode(explicit_text(example, eot), allowed_special=allowed_special)
    prompt_ids = tokenizer.encode(input_prompt(example, eot), allowed_special=allowed_special)
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("explicit example does not begin with its input prompt")
    return ids, len(prompt_ids)


def shifted_inputs_and_labels(
    encoded: torch.Tensor, prompt_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the model's already-shifted targets and mask the input prefix."""
    if encoded.ndim != 2 or encoded.shape[1] < 2:
        raise ValueError("encoded examples must have shape [batch, sequence>=2]")
    inputs = encoded[:, :-1]
    labels = encoded[:, 1:].clone()
    labels[:, : prompt_length - 1] = IGNORE_INDEX
    return inputs, labels


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int | None:
    if not pattern:
        return 0
    last = len(sequence) - len(pattern)
    for start in range(last + 1):
        if list(sequence[start : start + len(pattern)]) == list(pattern):
            return start
    return None


def answer_tokens_from_explicit_generation(
    generated: Sequence[int], marker: Sequence[int], eot_token: int
) -> list[int] | None:
    marker_start = find_subsequence(generated, marker)
    if marker_start is None:
        return None
    answer_start = marker_start + len(marker)
    answer: list[int] = []
    for token in generated[answer_start:]:
        if token == eot_token:
            break
        answer.append(int(token))
    return answer


def read_examples(path: Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line.rstrip("\n"))
                if limit is not None and len(lines) >= limit:
                    break
    return lines


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
