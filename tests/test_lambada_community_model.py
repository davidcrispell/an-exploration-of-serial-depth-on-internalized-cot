import unittest

import torch

from benchmarks.lambada.community_model import CommunityDeepGPT
from model import GPTConfig


class CommunityDeepModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = GPTConfig(
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            n_ff=64,
        )
        self.model = CommunityDeepGPT(self.config).eval()

    def test_forward_shapes_and_softcap(self):
        tokens = torch.randint(0, self.config.vocab_size, (3, 9))
        hidden = self.model.hidden(tokens)
        logits = self.model.logits_from_hidden(hidden)
        self.assertEqual(hidden.shape, (3, 9, self.config.n_embd))
        self.assertEqual(logits.shape, (3, 9, self.config.vocab_size))
        self.assertTrue(torch.all(logits.abs() <= 30))

    def test_checkpoint_specific_parameters_are_present(self):
        state = self.model.state_dict()
        self.assertIn("transformer.h.0.lambdas", state)
        self.assertIn("transformer.h.0.attn.lamb", state)


if __name__ == "__main__":
    unittest.main()
