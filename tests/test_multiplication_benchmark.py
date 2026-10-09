import tempfile
import unittest
from pathlib import Path

import tiktoken
import torch

from benchmarks.multiplication.cot import (
    answer_tokens_from_explicit_generation,
    compute_removal_distribution,
    cot_token_count,
    encode_explicit_example,
    explicit_text,
    parse_example,
    provided_cot_prompt,
    provided_cot_target,
    remove_cot_prefix_batch,
    shifted_inputs_and_labels,
    strip_compile_prefix,
)
from benchmarks.multiplication.generate import (
    generate_dataset,
    multiplication_record,
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
    def test_generated_record_matches_released_format(self) -> None:
        line = multiplication_record(2365, 4347, 4)
        self.assertEqual(
            line,
            "5 6 3 2 * 7 4 3 4||5 5 5 6 1 + 0 0 6 4 9 0 "
            "( 5 5 1 1 1 1 ) + 0 0 5 9 0 7 0 ( 5 5 6 0 2 8 0 ) "
            "+ 0 0 0 0 6 4 9 0 #### 5 5 6 0 8 2 0 1",
        )
        record = parse_record(line, 4)
        self.assertEqual((record.left, record.right), (2365, 4347))

    def test_dataset_excludes_heldout_pairs_in_both_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = generate_dataset(
                digits=2,
                train_path=root / "train.txt",
                validation_path=root / "validation.txt",
                test_path=root / "test.txt",
                heldout_per_split=3,
                seed=7,
            )
            heldout = {
                tuple(sorted((record.left, record.right)))
                for path in (root / "validation.txt", root / "test.txt")
                for record in (parse_record(line, 2) for line in path.read_text().splitlines())
            }
            train_pairs = {
                (record.left, record.right)
                for record in (
                    parse_record(line, 2)
                    for line in (root / "train.txt").read_text().splitlines()
                )
            }
            self.assertEqual(summary["train"]["examples"], 90 * 90 - 2 * 6)
            for left, right in heldout:
                self.assertNotIn((left, right), train_pairs)
                self.assertNotIn((right, left), train_pairs)

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

    def test_provided_cot_prompt_reconstructs_explicit_example(self):
        example = parse_example(self.line)
        eot = self.tokenizer.decode([self.tokenizer.eot_token])
        self.assertEqual(
            provided_cot_prompt(example, eot) + provided_cot_target(example, eot),
            explicit_text(example, eot),
        )

    def test_strip_compile_prefix(self):
        tensor = torch.tensor([1])
        result = strip_compile_prefix({"_orig_mod.layer": tensor, "other": tensor})
        self.assertEqual(set(result), {"layer", "other"})

    def test_paper_removal_smoothing_distribution(self):
        probabilities = compute_removal_distribution(4.0)
        self.assertAlmostEqual(probabilities.sum().item(), 1.0)
        self.assertAlmostEqual(
            probabilities[0].item(), 1 - torch.exp(torch.tensor(-4.0)).item()
        )
        self.assertGreater(probabilities[1].item(), probabilities[2].item())

    def test_left_removal_preserves_separators_and_masks_padding(self):
        ids, prompt_length = encode_explicit_example(self.tokenizer, self.line)
        cot_tokens = cot_token_count(ids, self.tokenizer.eot_token)
        encoded = torch.tensor([ids, ids])
        inputs, labels, removed = remove_cot_prefix_batch(
            encoded,
            torch.tensor([1, cot_tokens + 20]),
            eot_token=self.tokenizer.eot_token,
        )
        self.assertEqual(removed.tolist(), [1, cot_tokens])
        self.assertEqual(inputs.shape, labels.shape)
        self.assertTrue(torch.all(labels[:, : prompt_length - 1] == -1))
        self.assertEqual(
            int((inputs[0] == self.tokenizer.eot_token).sum().item()), 2
        )
        self.assertEqual(labels[0, len(ids) - 3].item(), self.tokenizer.eot_token)
        second_length = len(ids) - cot_tokens - 1
        self.assertTrue(torch.all(labels[1, second_length:] == -1))


if __name__ == "__main__":
    unittest.main()
