import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
