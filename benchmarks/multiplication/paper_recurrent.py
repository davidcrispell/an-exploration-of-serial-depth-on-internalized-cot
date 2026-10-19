"""Depth-recurrent Shallow 1 model adapted from Geiping et al. (2025).

The six pretrained Shallow 1 blocks are partitioned as (1, 4, 1): one
prelude block, a four-block recurrent core, and one coda block.  Each core
iteration receives both the previous latent state and the fixed prelude
representation through a learned 2h -> h adapter.  Training samples a
heavy-tailed recurrence count and backpropagates only through the final k
iterations, matching the central training construction in arXiv:2502.05171.

This is deliberately an adaptation rather than a byte-for-byte copy of
Huginn: it retains Shallow 1's tokenizer, attention/MLP blocks, untied output
head, and pretrained weights so that the recurrence intervention can be
compared with the existing fixed-depth multiplication runs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from model import GPT
from benchmarks.multiplication.cot import strip_compile_prefix


class PaperRecurrentGPT(nn.Module):
    """A (1, 4, 1) depth-recurrent view of a six-layer Shallow 1 GPT."""

    def __init__(
        self,
        base: GPT,
        *,
        mean_recurrence: int = 4,
        mean_backprop_depth: int = 4,
        eval_recurrence: int = 4,
        recurrence_sigma: float = 0.5,
        random_state_during_training: bool = True,
    ) -> None:
        super().__init__()
        if base.config.n_layer != 6:
            raise ValueError("paper recurrent split requires a six-layer base model")
        if mean_recurrence < 1:
            raise ValueError("mean_recurrence must be positive")
        if not 1 <= mean_backprop_depth <= mean_recurrence:
            raise ValueError("mean_backprop_depth must be in [1, mean_recurrence]")
        if eval_recurrence < 1:
            raise ValueError("eval_recurrence must be positive")
        if recurrence_sigma < 0:
            raise ValueError("recurrence_sigma must be non-negative")

        self.base = base
        self.config = base.config
        self.mean_recurrence = int(mean_recurrence)
        self.mean_backprop_depth = int(mean_backprop_depth)
        self.eval_recurrence = int(eval_recurrence)
        self.recurrence_sigma = float(recurrence_sigma)
        self.random_state_during_training = bool(random_state_during_training)

        width = self.config.n_embd
        self.adapter = nn.Linear(2 * width, width, bias=False)
        self.core_norm_weight = nn.Parameter(torch.ones(width))
        self.core_norm_mix = nn.Parameter(torch.zeros(()))

        # At one recurrence the converted checkpoint begins close to the
        # original feed-forward model: ignore s_0 and pass the prelude output.
        # Training is free to learn both halves immediately.
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, width:].copy_(torch.eye(width))

    @property
    def prelude(self):
        return self.base.transformer.h[:1]

    @property
    def core(self):
        return self.base.transformer.h[1:5]

    @property
    def coda(self):
        return self.base.transformer.h[5:]

    def parameter_counts(self) -> dict[str, int]:
        base = sum(parameter.numel() for parameter in self.base.parameters())
        recurrent = (
            self.adapter.weight.numel()
            + self.core_norm_weight.numel()
            + self.core_norm_mix.numel()
        )
        return {
            "base": base,
            "recurrent": recurrent,
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }

    def set_eval_recurrence(self, recurrence: int) -> None:
        if recurrence < 1:
            raise ValueError("eval recurrence must be positive")
        self.eval_recurrence = int(recurrence)

    def randomized_iteration_sampler(self) -> tuple[int, int]:
        """Return (no-grad, with-grad) iterations using the paper's sampler."""
        target = self.mean_recurrence
        sigma = self.recurrence_sigma
        if sigma == 0:
            sampled = target
        else:
            mu = math.log(target) - 0.5 * sigma**2
            rate = torch.empty((), device="cpu").log_normal_(mean=mu, std=sigma)
            sampled = int(torch.poisson(rate).item()) + 1
        with_grad = min(self.mean_backprop_depth, sampled)
        without_grad = max(sampled - with_grad, 0)
        return without_grad, with_grad

    def _initial_state(self, like: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.random_state_during_training:
            return torch.zeros_like(like)
        # The paper uses variance 2/5 after embedding-scale compensation.
        std = math.sqrt(2.0 / 5.0)
        state = torch.empty_like(like)
        return nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def _core_iteration(self, state: torch.Tensor, embedded_input: torch.Tensor) -> torch.Tensor:
        x = self.adapter(torch.cat((state, embedded_input), dim=-1))
        for block in self.base.transformer.h[1:5]:
            x = block(x)
        normalized = F.rms_norm(x, (x.size(-1),)) * self.core_norm_weight
        # A zero-initialized interpolation keeps r=1 exactly equivalent to the
        # source Shallow 1 checkpoint.  It can smoothly learn the paper's
        # recurrent-core output normalization instead of destroying the
        # already-perfect explicit-CoT solution at conversion time.
        mix = torch.tanh(self.core_norm_mix)
        return x + mix * (normalized - x)

    def forward_hidden(
        self,
        idx: torch.Tensor,
        *,
        num_recurrences: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        x = F.rms_norm(self.base.transformer.wte(idx), (self.config.n_embd,))
        embedded_input = self.base.transformer.h[0](x)
        state = self._initial_state(embedded_input)

        if num_recurrences is not None:
            if num_recurrences < 1:
                raise ValueError("num_recurrences must be positive")
            no_grad_steps, grad_steps = 0, int(num_recurrences)
        elif self.training:
            no_grad_steps, grad_steps = self.randomized_iteration_sampler()
        else:
            no_grad_steps, grad_steps = 0, self.eval_recurrence

        if no_grad_steps:
            with torch.no_grad():
                for _ in range(no_grad_steps):
                    state = self._core_iteration(state, embedded_input)
            state = state.detach()
        for _ in range(grad_steps):
            state = self._core_iteration(state, embedded_input)

        hidden = self.base.transformer.h[5](state)
        hidden = F.rms_norm(hidden, (hidden.size(-1),))
        return hidden, {
            "no_grad_recurrences": no_grad_steps,
            "grad_recurrences": grad_steps,
            "total_recurrences": no_grad_steps + grad_steps,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        *,
        num_recurrences: int | None = None,
    ):
        hidden, _ = self.forward_hidden(idx, num_recurrences=num_recurrences)
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


def load_paper_recurrent_from_explicit(
    checkpoint,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs,
) -> tuple[PaperRecurrentGPT, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = GPT(payload["model_config"])
    base.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    model = PaperRecurrentGPT(base, **kwargs)
    return model.to(device=device, dtype=dtype), payload


def load_paper_recurrent_checkpoint(
    checkpoint,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[PaperRecurrentGPT, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = GPT(payload["model_config"])
    model = PaperRecurrentGPT(
        base,
        mean_recurrence=payload["mean_recurrence"],
        mean_backprop_depth=payload["mean_backprop_depth"],
        eval_recurrence=payload["eval_recurrence"],
        recurrence_sigma=payload["recurrence_sigma"],
        random_state_during_training=payload.get("random_state_during_training", True),
    )
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype), payload
