"""Validate the multiplication benchmark splits released with ICoT-SI.

The paper represents every number least-significant digit first. A record is:

    reversed_a * reversed_b||long_multiplication_trace #### reversed_product

This validator checks the input/output format, exact arithmetic result, each
partial product, each cumulative sum in the trace, uniqueness, and leakage
between validation and test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


DIGIT_SIZES = (4, 5, 7, 9, 11)


class BenchmarkFormatError(ValueError):
    """Raised when a benchmark record is malformed or arithmetically wrong."""


@dataclass(frozen=True)
class Record:
    left: int
    right: int
    input_tokens: int
    cot_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SplitSummary:
    digits: int
    split: str
    examples: int
    median_input_tokens: float
    median_cot_tokens: float
    median_output_tokens: float
    sha256: str


def _decode_reversed_digits(tokens: list[str], *, field: str) -> int:
    if not tokens or any(len(token) != 1 or not token.isdigit() for token in tokens):
        raise BenchmarkFormatError(f"{field} must contain space-separated digits")
    return int("".join(reversed(tokens)))


def _take_digits(tokens: list[str], position: int, *, field: str) -> tuple[list[str], int]:
    start = position
    while position < len(tokens) and len(tokens[position]) == 1 and tokens[position].isdigit():
        position += 1
    if position == start:
        raise BenchmarkFormatError(f"missing digit sequence for {field}")
    return tokens[start:position], position


def _expect(tokens: list[str], position: int, expected: str) -> int:
    if position >= len(tokens) or tokens[position] != expected:
        actual = "<end>" if position >= len(tokens) else repr(tokens[position])
        raise BenchmarkFormatError(f"expected {expected!r}, found {actual}")
    return position + 1


def _validate_trace(cot_tokens: list[str], left: int, right_reversed: list[str]) -> None:
    digit_count = len(right_reversed)
    position = 0
    partial_tokens, position = _take_digits(cot_tokens, position, field="partial product 0")
    partials = [_decode_reversed_digits(partial_tokens, field="partial product 0")]

    expected = left * int(right_reversed[0])
    if partials[0] != expected:
        raise BenchmarkFormatError(
            f"partial product 0 is {partials[0]}, expected {expected}"
        )

    for index in range(1, digit_count):
        position = _expect(cot_tokens, position, "+")
        digits, position = _take_digits(
            cot_tokens, position, field=f"partial product {index}"
        )
        partial = _decode_reversed_digits(digits, field=f"partial product {index}")
        expected = left * int(right_reversed[index]) * (10**index)
        if partial != expected:
            raise BenchmarkFormatError(
                f"partial product {index} is {partial}, expected {expected}"
            )
        partials.append(partial)

        if index < digit_count - 1:
            position = _expect(cot_tokens, position, "(")
            digits, position = _take_digits(
                cot_tokens, position, field=f"cumulative sum {index}"
            )
            cumulative = _decode_reversed_digits(
                digits, field=f"cumulative sum {index}"
            )
            expected_sum = sum(partials)
            if cumulative != expected_sum:
                raise BenchmarkFormatError(
                    f"cumulative sum {index} is {cumulative}, expected {expected_sum}"
                )
            position = _expect(cot_tokens, position, ")")

    if position != len(cot_tokens):
        raise BenchmarkFormatError(f"unexpected trace token {cot_tokens[position]!r}")


def parse_record(line: str, digits: int) -> Record:
    try:
        input_text, target_text = line.strip().split("||", maxsplit=1)
        cot_text, output_text = target_text.split(" #### ", maxsplit=1)
    except ValueError as exc:
        raise BenchmarkFormatError("record must contain '||' and ' #### '") from exc

    input_tokens = input_text.split()
    if input_tokens.count("*") != 1:
        raise BenchmarkFormatError("input must contain exactly one '*' token")
    star = input_tokens.index("*")
    left_reversed = input_tokens[:star]
    right_reversed = input_tokens[star + 1 :]
    if len(left_reversed) != digits or len(right_reversed) != digits:
        raise BenchmarkFormatError(
            f"expected two {digits}-digit operands, found "
            f"{len(left_reversed)} and {len(right_reversed)} digits"
        )

    left = _decode_reversed_digits(left_reversed, field="left operand")
    right = _decode_reversed_digits(right_reversed, field="right operand")
    lower = 10 ** (digits - 1)
    upper = 10**digits
    if not (lower <= left < upper and lower <= right < upper):
        raise BenchmarkFormatError("operands must be exactly the requested digit length")

    cot_tokens = cot_text.split()
    _validate_trace(cot_tokens, left, right_reversed)

    output_tokens = output_text.split()
    output = _decode_reversed_digits(output_tokens, field="output")
    expected_output_tokens = list(reversed(str(left * right).zfill(2 * digits)))
    if output != left * right:
        raise BenchmarkFormatError(f"output is {output}, expected {left * right}")
    if output_tokens != expected_output_tokens:
        raise BenchmarkFormatError(
            "output must be the reversed product padded to exactly "
            f"{2 * digits} digits"
        )

    return Record(
        left=left,
        right=right,
        input_tokens=len(input_tokens),
        cot_tokens=len(cot_tokens),
        output_tokens=len(output_tokens),
    )


def validate_split(path: Path, digits: int, split: str) -> tuple[SplitSummary, set[tuple[int, int]]]:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if len(lines) != 1_000:
        raise BenchmarkFormatError(f"{path}: expected 1000 records, found {len(lines)}")

    records: list[Record] = []
    ordered_pairs: set[tuple[int, int]] = set()
    unordered_pairs: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = parse_record(line, digits)
        except BenchmarkFormatError as exc:
            raise BenchmarkFormatError(f"{path}:{line_number}: {exc}") from exc
        pair = (record.left, record.right)
        unordered = tuple(sorted(pair))
        if pair in ordered_pairs:
            raise BenchmarkFormatError(f"{path}:{line_number}: duplicate ordered operand pair {pair}")
        if unordered in unordered_pairs:
            raise BenchmarkFormatError(
                f"{path}:{line_number}: duplicate commutative operand pair {unordered}"
            )
        ordered_pairs.add(pair)
        unordered_pairs.add(unordered)
        records.append(record)

    summary = SplitSummary(
        digits=digits,
        split=split,
        examples=len(records),
        median_input_tokens=median(record.input_tokens for record in records),
        median_cot_tokens=median(record.cot_tokens for record in records),
        median_output_tokens=median(record.output_tokens for record in records),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return summary, unordered_pairs


def validate_all(data_root: Path) -> list[SplitSummary]:
    summaries: list[SplitSummary] = []
    for digits in DIGIT_SIZES:
        digit_root = data_root / f"{digits}x{digits}"
        validation, validation_pairs = validate_split(
            digit_root / "validation.txt", digits, "validation"
        )
        test, test_pairs = validate_split(digit_root / "test.txt", digits, "test")
        overlap = validation_pairs & test_pairs
        if overlap:
            sample = sorted(overlap)[0]
            raise BenchmarkFormatError(
                f"{digits}x{digits}: validation/test leakage for operand pair {sample}"
            )
        summaries.extend((validation, test))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="directory containing the NxN split directories",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    summaries = validate_all(args.data_root)
    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
        return

    for summary in summaries:
        print(
            f"{summary.digits}x{summary.digits} {summary.split}: "
            f"{summary.examples} examples, median whitespace tokens "
            f"input={summary.median_input_tokens:g} "
            f"cot={summary.median_cot_tokens:g} "
            f"output={summary.median_output_tokens:g}, "
            f"sha256={summary.sha256}"
        )


if __name__ == "__main__":
    main()
