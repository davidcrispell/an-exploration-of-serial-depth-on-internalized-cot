"""Shared data and evaluation helpers for multiplication CoT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


def compute_removal_distribution(
    smoothing_lambda: float, truncate_length: int = 100
) -> torch.Tensor:
    """Return the paper's truncated exponential distribution over extra removals."""
    if truncate_length < 1:
        raise ValueError("truncate_length must be positive")
    probabilities = torch.zeros(truncate_length, dtype=torch.float64)
    if math.isinf(smoothing_lambda):
        probabilities[0] = 1.0
        return probabilities
    if smoothing_lambda <= 0:
        raise ValueError("smoothing_lambda must be positive")
    positions = torch.arange(truncate_length, dtype=torch.float64)
    probabilities = (1 - math.exp(-smoothing_lambda)) * torch.exp(
        -smoothing_lambda * positions
    )
    probabilities[-1] += 1.0 - probabilities.sum()
    return probabilities


def cot_token_count(encoded: Sequence[int], eot_token: int) -> int:
    """Count tokens strictly between the first and second EOS separators."""
    separators = [index for index, token in enumerate(encoded) if token == eot_token]
    if len(separators) != 3:
        raise ValueError(f"expected exactly three EOS tokens, found {len(separators)}")
    return separators[1] - separators[0] - 1


def remove_cot_prefix_batch(
    encoded: torch.Tensor,
    removals: torch.Tensor,
    *,
    eot_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove each row's leftmost CoT tokens and prepare padded shifted targets.

    The first EOS (after the question), second EOS (after the CoT), answer marker,
    answer, and final EOS are retained. Padding is ignored by the loss.
    """
    if encoded.ndim != 2:
        raise ValueError("encoded examples must have shape [batch, sequence]")
    if removals.ndim != 1 or removals.shape[0] != encoded.shape[0]:
        raise ValueError("removals must have one value per encoded example")

    shortened: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    actual_removals: list[int] = []
    for row, requested in zip(encoded, removals):
        separators = torch.where(row == eot_token)[0]
        if separators.numel() != 3:
            raise ValueError(
                f"expected exactly three EOS tokens, found {separators.numel()}"
            )
        first = int(separators[0].item())
        second = int(separators[1].item())
        available = second - first - 1
        amount = min(max(int(requested.item()), 0), available)
        shortened.append(torch.cat((row[: first + 1], row[first + 1 + amount :])))
        prompt_lengths.append(first + 1)
        actual_removals.append(amount)

    max_length = max(row.numel() for row in shortened)
    inputs = encoded.new_full((len(shortened), max_length - 1), eot_token)
    labels = encoded.new_full((len(shortened), max_length - 1), IGNORE_INDEX)
    for index, (row, prompt_length) in enumerate(zip(shortened, prompt_lengths)):
        length = row.numel() - 1
        inputs[index, :length] = row[:-1]
        labels[index, :length] = row[1:]
        labels[index, : prompt_length - 1] = IGNORE_INDEX
    return inputs, labels, encoded.new_tensor(actual_removals)


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
