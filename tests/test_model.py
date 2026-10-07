import argparse
import unittest

import torch

from model import GPT, GPTConfig, add_architecture_arguments, config_from_args


class GPTConfigTests(unittest.TestCase):
    def test_default_architecture(self):
        config = GPTConfig()
        self.assertEqual(config.n_layer, 12)
        self.assertEqual(config.n_embd, 768)
        self.assertEqual(config.n_head, 6)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.n_ff, 3072)

    def test_command_line_defaults(self):
        parser = add_architecture_arguments(argparse.ArgumentParser())
        config = config_from_args(parser.parse_args([]))
        self.assertEqual(config, GPTConfig())

    def test_command_line_overrides_and_aliases(self):
        parser = add_architecture_arguments(argparse.ArgumentParser())
        args = parser.parse_args(
            ["--layers", "6", "--width", "1024", "--heads", "8", "--mlp-width", "2768"]
        )
        config = config_from_args(args)
        self.assertEqual(config.n_layer, 6)
        self.assertEqual(config.n_embd, 1024)
        self.assertEqual(config.n_head, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.n_ff, 2768)

    def test_invalid_head_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            GPTConfig(n_embd=10, n_head=3)

    def test_odd_head_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even"):
            GPTConfig(n_embd=18, n_head=2)

    def test_parameter_matched_depth_pair(self):
        deep = GPTConfig(n_layer=12, n_embd=768, n_head=6, n_ff=3072)
        shallow = GPTConfig(n_layer=6, n_embd=1024, n_head=8, n_ff=2768)
        self.assertEqual(deep.expected_parameter_count(), 162_201_600)
        self.assertEqual(shallow.expected_parameter_count(), 162_201_600)


class GPTModelTests(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(vocab_size=32, n_layer=2, n_head=2, n_embd=16, n_ff=24)
        self.model = GPT(self.config)

    def test_model_shapes_follow_config(self):
        self.assertEqual(len(self.model.transformer.h), 2)
        self.assertEqual(self.model.transformer.h[0].attn.head_dim, 8)
        self.assertEqual(self.model.transformer.h[0].mlp.c_fc.out_features, 24)

    def test_analytic_parameter_count_matches_model(self):
        actual = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(actual, self.config.expected_parameter_count())

    def test_training_forward(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 5))
        logits, loss = self.model(tokens, targets=tokens)
        self.assertEqual(logits.shape, (2, 5, self.config.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_inference_only_returns_last_position(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 5))
        logits, loss = self.model(tokens)
        self.assertEqual(logits.shape, (2, 1, self.config.vocab_size))
        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main()
