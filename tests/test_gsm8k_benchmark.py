import re
from pathlib import Path

import numpy as np

from benchmarks.gsm8k.data import (
    IGNORE_INDEX,
    collate_explicit_rows,
    encode_explicit_example,
    load_or_build_token_cache,
    normalize_answer,
    parse_example,
    remove_cot_prefix_rows,
)
from benchmarks.gsm8k.evaluate import answer_tokens
from benchmarks.gsm8k.train_internalized_cot import smoothing_distribution


class TinyTokenizer:
    eot_token = 99

    def __init__(self):
        self._ids = {"<eot>": self.eot_token}
        self._pieces = {self.eot_token: "<eot>"}

    def encode(self, text: str, allowed_special=None) -> list[int]:
        pieces = re.findall(r"<eot>|####|[A-Za-z0-9,+]+|[^\s]", text)
        result = []
        for piece in pieces:
            if piece not in self._ids:
                token = len(self._ids)
                if token == self.eot_token:
                    token += 1
                self._ids[piece] = token
                self._pieces[token] = piece
            result.append(self._ids[piece])
        return result

    def decode(self, ids: list[int]) -> str:
        if ids == [self.eot_token]:
            return "<eot>"
        return " ".join(self._pieces[token] for token in ids)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_and_encode_released_format():
    tokenizer = TinyTokenizer()
    line = "Jan has 2 apples and buys 3. How many?||2 + 3 = 5. #### 5"
    example = parse_example(line)
    assert example.question.startswith("Jan")
    assert example.cot == "2 + 3 = 5."
    assert example.answer == "5"

    ids, prompt_length, cot_length = encode_explicit_example(tokenizer, line)
    assert ids.count(tokenizer.eot_token) == 3
    assert prompt_length > 1
    assert cot_length > 0

    direct = "What is one plus one?|| #### 2"
    direct_example = parse_example(direct)
    assert direct_example.cot == ""
    direct_ids, _, direct_cot_length = encode_explicit_example(tokenizer, direct)
    assert direct_ids.count(tokenizer.eot_token) == 3
    assert direct_cot_length == 0


def test_variable_length_cache_collation_and_full_removal(tmp_path: Path):
    tokenizer = TinyTokenizer()
    lines = [
        "What is 2+3?||Add two and three. #### 5",
        "A box has 10 balls and loses 4. How many remain?||Start with ten. Subtract four to obtain six. #### 6",
    ]
    source = _write(tmp_path / "train.txt", lines)
    cache = load_or_build_token_cache(
        source, tmp_path / "cache", tokenizer, max_length=150
    )
    assert len(cache) == 2
    assert len(cache.row(0)) != len(cache.row(1))

    inputs, labels = collate_explicit_rows(
        cache, [0, 1], eot_token=tokenizer.eot_token
    )
    assert inputs.shape == labels.shape
    assert inputs.shape[0] == 2
    for row in range(2):
        prompt_length = int(cache.prompt_lengths[row])
        assert np.all(labels[row, : prompt_length - 1].numpy() == IGNORE_INDEX)
        assert labels[row, prompt_length - 1].item() != IGNORE_INDEX

    shortened_inputs, shortened_labels, actual = remove_cot_prefix_rows(
        cache,
        [0, 1],
        [0, 0],
        eot_token=tokenizer.eot_token,
        remove_all=True,
    )
    assert shortened_inputs.shape == shortened_labels.shape
    assert actual.tolist() == cache.cot_lengths.tolist()
    for row, prompt_length in enumerate(cache.prompt_lengths.tolist()):
        assert shortened_inputs[row, prompt_length - 1].item() == tokenizer.eot_token
        assert shortened_labels[row, prompt_length - 1].item() == tokenizer.eot_token


def test_answer_extraction_and_normalization():
    tokenizer = TinyTokenizer()
    eot = tokenizer.eot_token
    generated = tokenizer.encode(" some reasoning") + [eot]
    generated += tokenizer.encode(" #### 1,234") + [eot]
    extracted = answer_tokens(generated, tokenizer.encode(" ####"), eot)
    assert extracted is not None
    assert normalize_answer(tokenizer.decode(extracted)) == "1234"


def test_smoothing_distribution_matches_geometric_shape():
    distribution = smoothing_distribution(4.0)
    assert distribution.shape == (100,)
    assert abs(float(distribution.sum()) - 1.0) < 1e-6
    assert distribution[0] > distribution[1] > distribution[2]

