import unittest
from pathlib import Path

import tiktoken
import torch

from benchmarks.multiplication.cot import (
    answer_tokens_from_explicit_generation,
    encode_explicit_example,
    shifted_inputs_and_labels,
    strip_compile_prefix,
)
from benchmarks.multiplication.evaluate import answer_matches

from benchmarks.multiplication.validate import (
    BenchmarkFormatError,
    DIGIT_SIZES,
    parse_record,
    validate_all,
)


DATA_ROOT = Path(__file__).parents[1] / "benchmarks" / "multiplication" / "data"


class MultiplicationBenchmarkTests(unittest.TestCase):
    def test_all_paper_splits_are_valid(self):
        summaries = validate_all(DATA_ROOT)
        self.assertEqual(len(summaries), 2 * len(DIGIT_SIZES))
        self.assertTrue(all(summary.examples == 1_000 for summary in summaries))

    def test_known_record_decodes_to_correct_product(self):
        line = (DATA_ROOT / "4x4" / "test.txt").read_text().splitlines()[0]
        record = parse_record(line, digits=4)
        self.assertEqual(record.left, 2967)
        self.assertEqual(record.right, 6361)
        self.assertEqual(record.left * record.right, 18_873_087)

    def test_incorrect_answer_is_rejected(self):
        line = (DATA_ROOT / "4x4" / "test.txt").read_text().splitlines()[0]
        prefix, _ = line.rsplit(" #### ", maxsplit=1)
        with self.assertRaises(BenchmarkFormatError):
            parse_record(prefix + " #### 0 0 0 0 0 0 0 0", digits=4)


class CoTFormattingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.line = (
            "7 6 9 2 * 1 6 3 6||7 6 9 2 0 + 0 2 0 8 7 1 "
            "( 7 8 9 0 8 1 ) + 0 0 1 0 9 8 0 ( 7 8 0 1 7 0 1 ) + "
            "0 0 0 2 0 8 7 1 #### 7 8 0 3 7 8 8 1"
        )

    def test_explicit_encoding_and_input_mask(self):
        ids, prompt_length = encode_explicit_example(self.tokenizer, self.line)
        self.assertEqual(len(ids), 71)
        self.assertEqual(prompt_length, 11)
        inputs, labels = shifted_inputs_and_labels(torch.tensor([ids]), prompt_length)
        self.assertEqual(inputs.shape, labels.shape)
        self.assertTrue(torch.all(labels[:, : prompt_length - 1] == -1))
        self.assertEqual(labels[0, prompt_length - 1].item(), ids[prompt_length])

    def test_extract_answer_after_marker(self):
        marker = self.tokenizer.encode(" ####")
        answer = self.tokenizer.encode(" 7 8 0 3 ")
        generated = self.tokenizer.encode(" trace") + marker + answer + [self.tokenizer.eot_token]
        self.assertEqual(
            answer_tokens_from_explicit_generation(
                generated, marker, self.tokenizer.eot_token
            ),
            answer,
        )
        self.assertTrue(answer_matches(self.tokenizer, answer, "7 8 0 3"))

    def test_strip_compile_prefix(self):
        tensor = torch.tensor([1])
        result = strip_compile_prefix({"_orig_mod.layer": tensor, "other": tensor})
        self.assertEqual(set(result), {"layer", "other"})


if __name__ == "__main__":
    unittest.main()
