"""Data handling for the paper's GPT-4-augmented GSM8K experiment.

The released files use one record per line::

    question||reasoning trace #### final answer

Training follows the paper's GPT-2 serialization:
``question EOS trace EOS #### answer EOS``.  The input prefix is masked from
the language-model loss.  Unlike the multiplication benchmark, GSM8K records
have variable lengths, so this module stores a compact packed token cache and
right-pads individual minibatches with loss-masked positions.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IGNORE_INDEX = -1


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    cot: str
    answer: str


def parse_example(line: str) -> GSM8KExample:
    """Parse one released GSM8K-Aug record and reject ambiguous separators."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("empty GSM8K record")
    try:
        question, target = stripped.split("||", maxsplit=1)
        cot, answer = target.rsplit(" #### ", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            "expected `question||reasoning #### answer` GSM8K record"
        ) from error
    if not question.strip() or not answer.strip():
        raise ValueError("question and answer must be non-empty")
    return GSM8KExample(
        question=question.strip(), cot=cot.strip(), answer=answer.strip()
    )


def explicit_text(example: GSM8KExample, eot: str) -> str:
    return (
        f" {example.question} {eot} {example.cot} "
        f"{eot} #### {example.answer} {eot}"
    )


def input_prompt(example: GSM8KExample, eot: str) -> str:
    return f" {example.question} {eot}"


def encode_explicit_example(tokenizer, line: str) -> tuple[list[int], int, int]:
    """Return full ids, input-prefix length, and number of CoT tokens."""
    example = parse_example(line)
    eot = tokenizer.decode([tokenizer.eot_token])
    allowed_special = {eot}
    ids = tokenizer.encode(
        explicit_text(example, eot), allowed_special=allowed_special
    )
    prompt_ids = tokenizer.encode(
        input_prompt(example, eot), allowed_special=allowed_special
    )
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("explicit example does not begin with its input prompt")
    separators = [index for index, token in enumerate(ids) if token == tokenizer.eot_token]
    if len(separators) != 3:
        raise ValueError(f"expected exactly three EOS tokens, found {len(separators)}")
    cot_tokens = separators[1] - separators[0] - 1
    return ids, len(prompt_ids), cot_tokens


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedTokenCache:
    tokens: np.ndarray
    offsets: np.ndarray
    prompt_lengths: np.ndarray
    cot_lengths: np.ndarray
    source_sha256: str
    skipped_over_length: int
    max_length: int

    def __len__(self) -> int:
        return len(self.prompt_lengths)

    def row(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.tokens[start:end]


def _cache_stem(data_path: Path, max_length: int) -> str:
    digest = file_sha256(data_path)
    return f"{data_path.stem}-{digest[:16]}-max{max_length}"


def _cache_paths(data_path: Path, cache_dir: Path, max_length: int) -> dict[str, Path]:
    root = cache_dir / _cache_stem(data_path, max_length)
    return {
        "root": root,
        "tokens": root / "tokens.npy",
        "offsets": root / "offsets.npy",
        "prompt_lengths": root / "prompt_lengths.npy",
        "cot_lengths": root / "cot_lengths.npy",
        "metadata": root / "metadata.json",
    }


def _load_cache(paths: dict[str, Path]) -> PackedTokenCache:
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return PackedTokenCache(
        tokens=np.load(paths["tokens"], mmap_mode="r"),
        offsets=np.load(paths["offsets"], mmap_mode="r"),
        prompt_lengths=np.load(paths["prompt_lengths"], mmap_mode="r"),
        cot_lengths=np.load(paths["cot_lengths"], mmap_mode="r"),
        source_sha256=metadata["source_sha256"],
        skipped_over_length=metadata["skipped_over_length"],
        max_length=metadata["max_length"],
    )


def load_or_build_token_cache(
    data_path: Path,
    cache_dir: Path,
    tokenizer,
    *,
    max_length: int = 150,
) -> PackedTokenCache:
    """Load or build a memory-mapped, packed token cache.

    The paper says GSM8K training considers sequences of at most 150 tokens.
    We enforce that per example, rather than dropping an entire randomly padded
    minibatch when one row exceeds the limit.
    """
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    paths = _cache_paths(data_path, cache_dir, max_length)
    required = [
        paths["tokens"],
        paths["offsets"],
        paths["prompt_lengths"],
        paths["cot_lengths"],
        paths["metadata"],
    ]
    if all(path.exists() for path in required):
        print(f"loading token cache {paths['root']}", flush=True)
        return _load_cache(paths)

    paths["root"].mkdir(parents=True, exist_ok=True)
    packed = array("H")
    offsets = array("Q", [0])
    prompt_lengths = array("I")
    cot_lengths = array("I")
    skipped = 0
    seen = 0
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            seen += 1
            ids, prompt_length, cot_length = encode_explicit_example(tokenizer, line)
            if len(ids) > max_length:
                skipped += 1
                continue
            if max(ids) > np.iinfo(np.uint16).max:
                raise ValueError(f"token id exceeds uint16 at line {line_number}")
            packed.extend(ids)
            offsets.append(len(packed))
            prompt_lengths.append(prompt_length)
            cot_lengths.append(cot_length)
            if seen % 25_000 == 0:
                print(
                    f"tokenized {seen} rows; kept={len(prompt_lengths)} skipped={skipped}",
                    flush=True,
                )
    if not prompt_lengths:
        raise ValueError("no training examples remain after token-length filtering")

    np.save(paths["tokens"], np.frombuffer(packed, dtype=np.uint16))
    np.save(paths["offsets"], np.frombuffer(offsets, dtype=np.uint64))
    np.save(
        paths["prompt_lengths"], np.frombuffer(prompt_lengths, dtype=np.uint32)
    )
    np.save(paths["cot_lengths"], np.frombuffer(cot_lengths, dtype=np.uint32))
    metadata = {
        "source": str(data_path),
        "source_sha256": file_sha256(data_path),
        "max_length": max_length,
        "source_examples": seen,
        "kept_examples": len(prompt_lengths),
        "skipped_over_length": skipped,
        "tokens": len(packed),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"built token cache {paths['root']}: {metadata}", flush=True)
    return _load_cache(paths)


def collate_explicit_rows(
    cache: PackedTokenCache,
    indices: Sequence[int],
    *,
    eot_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad explicit examples and mask input/padding labels."""
    rows = [torch.from_numpy(np.asarray(cache.row(int(index)), dtype=np.int64)) for index in indices]
    max_length = max(row.numel() for row in rows)
    inputs = torch.full((len(rows), max_length - 1), eot_token, dtype=torch.long)
    labels = torch.full((len(rows), max_length - 1), IGNORE_INDEX, dtype=torch.long)
    for batch_index, (row, source_index) in enumerate(zip(rows, indices)):
        length = row.numel() - 1
        prompt_length = int(cache.prompt_lengths[int(source_index)])
        inputs[batch_index, :length] = row[:-1]
        labels[batch_index, :length] = row[1:]
        labels[batch_index, : prompt_length - 1] = IGNORE_INDEX
    return inputs, labels


def remove_cot_prefix_rows(
    cache: PackedTokenCache,
    indices: Sequence[int],
    removals: Sequence[int],
    *,
    eot_token: int,
    remove_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove a variable number of leftmost CoT tokens from variable rows."""
    if len(indices) != len(removals):
        raise ValueError("indices and removals must have the same length")
    shortened: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    actual: list[int] = []
    for source_index, requested in zip(indices, removals):
        row = torch.from_numpy(
            np.asarray(cache.row(int(source_index)), dtype=np.int64)
        )
        separators = torch.where(row == eot_token)[0]
        if separators.numel() != 3:
            raise ValueError(
                f"expected exactly three EOS tokens, found {separators.numel()}"
            )
        first = int(separators[0])
        second = int(separators[1])
        available = second - first - 1
        amount = available if remove_all else min(max(int(requested), 0), available)
        shortened.append(torch.cat((row[: first + 1], row[first + 1 + amount :])))
        prompt_lengths.append(first + 1)
        actual.append(amount)

    max_length = max(row.numel() for row in shortened)
    inputs = torch.full(
        (len(shortened), max_length - 1), eot_token, dtype=torch.long
    )
    labels = torch.full(
        (len(shortened), max_length - 1), IGNORE_INDEX, dtype=torch.long
    )
    for batch_index, (row, prompt_length) in enumerate(
        zip(shortened, prompt_lengths)
    ):
        length = row.numel() - 1
        inputs[batch_index, :length] = row[:-1]
        labels[batch_index, :length] = row[1:]
        labels[batch_index, : prompt_length - 1] = IGNORE_INDEX
    return inputs, labels, torch.tensor(actual, dtype=torch.long)


def read_examples(path: Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line.rstrip("\n"))
                if limit is not None and len(lines) >= limit:
                    break
    return lines


def batches(items: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_answer(answer: str) -> str:
    """Match the released evaluator's comma-insensitive exact comparison."""
    return answer.strip().replace(",", "")

