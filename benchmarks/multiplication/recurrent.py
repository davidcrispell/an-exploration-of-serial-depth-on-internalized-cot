"""Frozen-base full-bandwidth recurrence for the Shallow 1.1 pilot.

The six-layer Shallow 1 stack runs twice.  On the second pass, the hidden state
at position t-1 is projected and gated by the ordinary token embedding at t:

    u_t = W_h h_(t-1) * sigmoid(W_g RMSNorm(e_t))

The first token, and a random plain-token prefix during training, retain their
ordinary embeddings.  Only W_h and W_g are trainable; all original Shallow 1
weights stay frozen.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model import GPT
from benchmarks.multiplication.cot import strip_compile_prefix


class RecurrentGPT(nn.Module):
    def __init__(
        self,
        base: GPT,
        *,
        prefix_mixin: bool = True,
        hidden_jitter: float = 0.02,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = base.config
        self.prefix_mixin = prefix_mixin
        self.hidden_jitter = hidden_jitter
        width = self.config.n_embd
        self.hidden_projection = nn.Linear(width, width, bias=False)
        self.gate_projection = nn.Linear(width, width, bias=False)

        # A normalized identity-projected hidden stream is a stable starting
        # point; the zero gate gives a uniform 0.5 scale that RMSNorm removes.
        nn.init.eye_(self.hidden_projection.weight)
        nn.init.zeros_(self.gate_projection.weight)
        self.freeze_base()

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "base": sum(parameter.numel() for parameter in self.base.parameters()),
            "recurrent": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if not name.startswith("base.")
            ),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }

    def _feedback_inputs(
        self,
        embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
        *,
        plain_prefix_length: int | None,
    ) -> torch.Tensor:
        if embeddings.shape != previous_hidden.shape:
            raise ValueError("embeddings and previous_hidden must have matching shapes")
        shifted = torch.zeros_like(previous_hidden)
        shifted[:, 1:] = previous_hidden[:, :-1]
        if self.training and self.hidden_jitter > 0:
            shifted = shifted + torch.empty_like(shifted).uniform_(
                -self.hidden_jitter, self.hidden_jitter
            )
        normalized_embeddings = F.rms_norm(embeddings, (embeddings.size(-1),))
        fused = self.hidden_projection(shifted) * torch.sigmoid(
            self.gate_projection(normalized_embeddings)
        )
        fused = F.rms_norm(fused, (fused.size(-1),))

        if plain_prefix_length is None:
            if self.training and self.prefix_mixin:
                # Position zero must always be plain. Sampling the boundary
                # per batch matches the paper's prefix-mixin training recipe.
                plain_prefix_length = int(
                    torch.randint(1, embeddings.size(1), (), device=embeddings.device).item()
                )
            else:
                plain_prefix_length = 1
        if not 1 <= plain_prefix_length <= embeddings.size(1):
            raise ValueError("plain_prefix_length must be in [1, sequence_length]")
        fused[:, :plain_prefix_length] = embeddings[:, :plain_prefix_length]
        return fused

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        *,
        plain_prefix_length: int | None = None,
    ):
        embeddings = self.base.transformer.wte(idx)
        # The first pass is a frozen teacher/context pass. Gradients are only
        # needed through the feedback pass into the two recurrent matrices.
        with torch.no_grad():
            first_hidden = self.base.forward_hidden_from_embeddings(embeddings)
        feedback = self._feedback_inputs(
            embeddings, first_hidden, plain_prefix_length=plain_prefix_length
        )
        hidden = self.base.forward_hidden_from_embeddings(feedback)

        if targets is not None:
            logits = self.base.lm_head(hidden).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.base.lm_head(hidden[:, [-1], :]).float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss


def load_recurrent_from_explicit(
    checkpoint,
    device: torch.device,
    dtype: torch.dtype,
    *,
    prefix_mixin: bool = True,
    hidden_jitter: float = 0.02,
) -> tuple[RecurrentGPT, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = GPT(payload["model_config"])
    base.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    model = RecurrentGPT(
        base, prefix_mixin=prefix_mixin, hidden_jitter=hidden_jitter
    )
    return model.to(device=device, dtype=dtype), payload


def load_recurrent_checkpoint(
    checkpoint,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[RecurrentGPT, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = GPT(payload["model_config"])
    model = RecurrentGPT(
        base,
        prefix_mixin=payload.get("prefix_mixin", True),
        hidden_jitter=payload.get("hidden_jitter", 0.02),
    )
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype), payload
