import torch

from model import GPT, GPTConfig
from benchmarks.multiplication.paper_recurrent import PaperRecurrentGPT


def tiny_model() -> PaperRecurrentGPT:
    base = GPT(GPTConfig(vocab_size=64, n_layer=6, n_head=2, n_embd=8, n_ff=16))
    return PaperRecurrentGPT(
        base,
        mean_recurrence=2,
        mean_backprop_depth=2,
        eval_recurrence=2,
        recurrence_sigma=0.0,
    )


def test_partition_and_parameter_counts():
    model = tiny_model()
    counts = model.parameter_counts()
    assert len(model.prelude) == 1
    assert len(model.core) == 4
    assert len(model.coda) == 1
    assert counts["recurrent"] == 2 * 8 * 8 + 8 + 1
    assert counts["trainable"] == counts["total"]


def test_adapter_initially_selects_prelude_input():
    model = tiny_model()
    state = torch.randn(2, 5, 8)
    embedded = torch.randn_like(state)
    fused = model.adapter(torch.cat((state, embedded), dim=-1))
    torch.testing.assert_close(fused, embedded)


def test_one_recurrence_exactly_preserves_source_model():
    torch.manual_seed(7)
    model = tiny_model()
    model.eval()
    tokens = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        base_logits, _ = model.base(tokens)
        recurrent_logits, _ = model(tokens, num_recurrences=1)
    torch.testing.assert_close(recurrent_logits, base_logits)


def test_fixed_recurrence_and_full_gradient_flow():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))
    hidden, stats = model.forward_hidden(tokens, num_recurrences=3)
    assert hidden.shape == (2, 6, 8)
    assert stats["total_recurrences"] == 3
    _, loss = model(tokens, targets, num_recurrences=2)
    loss.backward()
    assert model.adapter.weight.grad is not None
    assert model.base.transformer.h[0].attn.c_q.weight.grad is not None
    assert model.base.transformer.h[2].attn.c_q.weight.grad is not None
    assert model.base.transformer.h[5].attn.c_q.weight.grad is not None


def test_zero_sigma_sampler_uses_requested_recurrence():
    model = tiny_model()
    no_grad, with_grad = model.randomized_iteration_sampler()
    assert (no_grad, with_grad) == (0, 2)
