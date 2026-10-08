import unittest

from training_utils import linear_warmup_warmdown_factor, strip_compiled_prefix


class LearningRateScheduleTests(unittest.TestCase):
    def test_warmup_plateau_and_warmdown(self):
        kwargs = dict(num_iterations=100, warmup_iters=10, warmdown_iters=20)
        self.assertAlmostEqual(linear_warmup_warmdown_factor(0, **kwargs), 0.1)
        self.assertEqual(linear_warmup_warmdown_factor(9, **kwargs), 1.0)
        self.assertEqual(linear_warmup_warmdown_factor(50, **kwargs), 1.0)
        self.assertEqual(linear_warmup_warmdown_factor(80, **kwargs), 1.0)
        self.assertAlmostEqual(linear_warmup_warmdown_factor(90, **kwargs), 0.5)
        self.assertEqual(linear_warmup_warmdown_factor(100, **kwargs), 0.0)

    def test_zero_length_warmdown(self):
        factor = linear_warmup_warmdown_factor(
            100, num_iterations=100, warmup_iters=0, warmdown_iters=0
        )
        self.assertEqual(factor, 1.0)

    def test_invalid_overlapping_schedule(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            linear_warmup_warmdown_factor(
                0, num_iterations=10, warmup_iters=6, warmdown_iters=5
            )


class CheckpointStateTests(unittest.TestCase):
    def test_compiled_prefix_is_removed(self):
        state = {"_orig_mod.layer.weight": 1, "plain": 2}
        self.assertEqual(
            strip_compiled_prefix(state), {"layer.weight": 1, "plain": 2}
        )


if __name__ == "__main__":
    unittest.main()
