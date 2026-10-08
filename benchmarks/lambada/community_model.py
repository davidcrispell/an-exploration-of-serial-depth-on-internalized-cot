"""Architecture used by the community-uploaded November 2024 checkpoint."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model import GPTConfig, Rotary, apply_rotary_emb


class CommunityCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, v1: torch.Tensor | None):
        batch_size, seq_len, _ = x.shape
        shape = (batch_size, seq_len, self.n_head, self.head_dim)
        q = self.c_q(x).view(shape)
        k = self.c_k(x).view(shape)
        v = self.c_v(x).view(shape)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        cos, sin = self.rotary(q)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y), v1


class CommunityBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CommunityCausalSelfAttention(config)
        self.mlp = CommunityMLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor, v1: torch.Tensor | None, x0: torch.Tensor):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attended, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + attended
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1


class CommunityMLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_ff, bias=False)
        self.c_proj = nn.Linear(config.n_ff, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class CommunityDeepGPT(nn.Module):
    """Modded-nanoGPT variant with value and residual connections across layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [CommunityBlock(config) for _ in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(self.transformer.wte(idx), (self.config.n_embd,))
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        return F.rms_norm(x, (self.config.n_embd,))

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        return 30 * torch.tanh(logits / 30)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden(self.hidden(idx))


class SoftcappedLMHead(nn.Module):
    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        return 30 * torch.tanh(logits / 30)
