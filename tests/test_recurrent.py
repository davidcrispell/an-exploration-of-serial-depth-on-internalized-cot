import torch

from model import GPT, GPTConfig
from benchmarks.multiplication.recurrent import RecurrentGPT


def tiny_model() -> RecurrentGPT:
    base = GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=8, n_ff=16))
    return RecurrentGPT(base, prefix_mixin=False, hidden_jitter=0.0)


def test_only_two_full_width_matrices_are_trainable():
    model = tiny_model()
    counts = model.parameter_counts()
    assert counts["recurrent"] == 2 * 8 * 8
    assert counts["trainable"] == counts["recurrent"]
    assert counts["total"] == counts["base"] + counts["recurrent"]
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_feedback_keeps_plain_prefix():
    model = tiny_model()
    embeddings = torch.randn(2, 5, 8)
    hidden = torch.randn_like(embeddings)
    feedback = model._feedback_inputs(
        embeddings, hidden, plain_prefix_length=3
    )
    torch.testing.assert_close(feedback[:, :3], embeddings[:, :3])
    assert feedback.shape == embeddings.shape


def test_loss_only_backpropagates_into_recurrent_adapter():
    model = tiny_model()
    tokens = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))
    _, loss = model(tokens, targets, plain_prefix_length=1)
    loss.backward()
    assert model.hidden_projection.weight.grad is not None
    assert model.gate_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.base.parameters())
