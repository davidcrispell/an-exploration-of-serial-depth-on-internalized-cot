"""Generate deterministic long-multiplication CoT datasets.

The released ICoT-SI benchmark starts at 4x4 multiplication.  This generator
extends the same least-significant-digit-first representation to other fixed
operand widths.  Held-out pairs are sampled without commutative duplicates,
and both operand orders are excluded from training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def reversed_digits(value: int, width: int) -> str:
    return " ".join(reversed(str(value).zfill(width)))


def multiplication_record(left: int, right: int, digits: int) -> str:
    lower = 10 ** (digits - 1)
    upper = 10**digits
    if not (lower <= left < upper and lower <= right < upper):
        raise ValueError(f"operands must both have exactly {digits} digits")

    right_digits = [int(character) for character in reversed(str(right))]
    partials = [left * digit * (10**index) for index, digit in enumerate(right_digits)]
    trace = [reversed_digits(partials[0], digits + 1)]
    cumulative = partials[0]
    for index, partial in enumerate(partials[1:], start=1):
        trace.extend(("+", reversed_digits(partial, digits + 1 + index)))
        cumulative += partial
        if index < digits - 1:
            trace.extend(("(", reversed_digits(cumulative, digits + 1 + index), ")"))

    source = f"{reversed_digits(left, digits)} * {reversed_digits(right, digits)}"
    answer = reversed_digits(left * right, 2 * digits)
    return f"{source}||{' '.join(trace)} #### {answer}"


def heldout_pairs(digits: int, count: int, seed: int) -> list[tuple[int, int]]:
    lower = 10 ** (digits - 1)
    upper = 10**digits
    candidates = [
        (left, right)
        for left in range(lower, upper)
        for right in range(left + 1, upper)
    ]
    if count > len(candidates):
        raise ValueError(f"requested {count} held-out pairs from {len(candidates)} candidates")
    return random.Random(seed).sample(candidates, count)


def _write_lines(path: Path, lines) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for line in lines:
            encoded = (line + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def generate_dataset(
    *,
    digits: int,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    heldout_per_split: int = 1_000,
    seed: int = 3456,
) -> dict:
    heldout = heldout_pairs(digits, heldout_per_split * 2, seed)
    validation_pairs = heldout[:heldout_per_split]
    test_pairs = heldout[heldout_per_split:]
    excluded = {tuple(sorted(pair)) for pair in heldout}
    lower = 10 ** (digits - 1)
    upper = 10**digits

    def train_lines():
        for left in range(lower, upper):
            for right in range(lower, upper):
                if tuple(sorted((left, right))) not in excluded:
                    yield multiplication_record(left, right, digits)

    orientation_rng = random.Random(seed + 1)

    def evaluation_lines(pairs):
        for left, right in pairs:
            if orientation_rng.getrandbits(1):
                left, right = right, left
            yield multiplication_record(left, right, digits)

    train_count, train_sha = _write_lines(train_path, train_lines())
    validation_count, validation_sha = _write_lines(
        validation_path, evaluation_lines(validation_pairs)
    )
    test_count, test_sha = _write_lines(test_path, evaluation_lines(test_pairs))
    return {
        "digits": digits,
        "seed": seed,
        "heldout_per_split": heldout_per_split,
        "train": {"path": str(train_path), "examples": train_count, "sha256": train_sha},
        "validation": {
            "path": str(validation_path),
            "examples": validation_count,
            "sha256": validation_sha,
        },
        "test": {"path": str(test_path), "examples": test_count, "sha256": test_sha},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--heldout-per-split", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = generate_dataset(
        digits=args.digits,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        heldout_per_split=args.heldout_per_split,
        seed=args.seed,
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
