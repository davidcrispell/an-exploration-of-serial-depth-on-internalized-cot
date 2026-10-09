import unittest

import torch

from training_utils import (
    linear_warmup_warmdown_factor,
    restore_optimizer_states,
    strip_compiled_prefix,
)


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

    def test_optimizer_moments_restore_with_new_learning_rate(self):
        source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        source = torch.optim.Adam([source_parameter], lr=0.25)
        source_parameter.grad = torch.tensor([2.0])
        source.step()

        target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        target = torch.optim.Adam([target_parameter], lr=0.01)
        restore_optimizer_states([target], [source.state_dict()], [0.125])

        self.assertEqual(target.param_groups[0]["lr"], 0.125)
        self.assertEqual(target.param_groups[0]["initial_lr"], 0.125)
        self.assertEqual(target.state[target_parameter]["step"].item(), 1)

    def test_optimizer_restore_rejects_mismatched_counts(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        with self.assertRaisesRegex(ValueError, "counts must match"):
            restore_optimizer_states([optimizer], [], [0.1])


if __name__ == "__main__":
    unittest.main()
