from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    n_ff: int = 3072

    def __post_init__(self):
        positive_fields = ("vocab_size", "n_layer", "n_head", "n_embd", "n_ff")
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def expected_parameter_count(self) -> int:
        """Exact count for this bias-free model with untied embedding and head."""
        embeddings_and_head = 2 * self.vocab_size * self.n_embd
        attention_per_layer = 4 * self.n_embd * self.n_embd
        mlp_per_layer = 2 * self.n_embd * self.n_ff
        return embeddings_and_head + self.n_layer * (attention_per_layer + mlp_per_layer)


def add_architecture_arguments(parser: ArgumentParser) -> ArgumentParser:
    defaults = GPTConfig()
    group = parser.add_argument_group("model architecture")
    group.add_argument(
        "--n-layer", "--layers", dest="n_layer", type=int, default=defaults.n_layer,
        help=f"number of transformer blocks (default: {defaults.n_layer})",
    )
    group.add_argument(
        "--n-head", "--heads", dest="n_head", type=int, default=defaults.n_head,
        help=f"number of attention heads (default: {defaults.n_head})",
    )
    group.add_argument(
        "--n-embd", "--width", dest="n_embd", type=int, default=defaults.n_embd,
        help=f"residual/model width (default: {defaults.n_embd})",
    )
    group.add_argument(
        "--n-ff", "--mlp-width", dest="n_ff", type=int, default=defaults.n_ff,
        help=f"MLP hidden width (default: {defaults.n_ff})",
    )
    return parser


def config_from_args(args: Namespace, *, vocab_size: int = 50304) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_ff=args.n_ff,
    )


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        shape = (batch_size, seq_len, self.n_head, self.head_dim)
        q = self.c_q(x).view(shape)
        k = self.c_k(x).view(shape)
        v = self.c_v(x).view(shape)
        cos, sin = self.rotary(q)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_ff, bias=False)
        self.c_proj = nn.Linear(config.n_ff, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        return x + self.mlp(F.rms_norm(x, (x.size(-1),)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss
