#!/usr/bin/env python3
"""Measure our GPT variants with google-deepmind/serial_depth.

This is an operation-faithful JAX transcription of the inference paths in
``model.py``, ``benchmarks/multiplication/recurrent.py``, and
``benchmarks/multiplication/paper_recurrent.py``.  It traces abstract arrays,
so no model checkpoint or parameter allocation is required.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Architecture:
    name: str
    layers: int
    width: int
    heads: int
    mlp_width: int
    kind: str = "standard"
    recurrences: int = 1

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def parameter_count(self) -> int:
        base = (
            2 * 50304 * self.width
            + self.layers
            * (4 * self.width**2 + 2 * self.width * self.mlp_width)
        )
        if self.kind == "frozen_pilot":
            return base + 2 * self.width**2
        if self.kind == "paper":
            return base + 2 * self.width**2 + self.width + 1
        return base


ARCHITECTURES = (
    Architecture("shallow1", 6, 1024, 8, 2768),
    Architecture("deep1", 12, 768, 6, 3072),
    Architecture("deeper1", 24, 600, 5, 2336),
    Architecture("shallow1_1_frozen_pilot", 6, 1024, 8, 2768, "frozen_pilot", 2),
    Architecture("shallow1_1_paper_r1", 6, 1024, 8, 2768, "paper", 1),
    Architecture("shallow1_1_paper_r2", 6, 1024, 8, 2768, "paper", 2),
    Architecture("shallow1_1_paper_r4", 6, 1024, 8, 2768, "paper", 4),
    Architecture("shallow1_1_paper_r8", 6, 1024, 8, 2768, "paper", 8),
)


def rms_norm(x: jax.Array) -> jax.Array:
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean_square + jnp.asarray(1e-6, x.dtype))


def linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.matmul(x, jnp.swapaxes(weight, -1, -2))


def block(
    x: jax.Array,
    weights: dict[str, jax.Array],
    *,
    heads: int,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
) -> jax.Array:
    batch, sequence, width = x.shape
    head_dim = width // heads
    normalized = rms_norm(x)
    q = linear(normalized, weights["q"]).reshape(batch, sequence, heads, head_dim)
    k = linear(normalized, weights["k"]).reshape(batch, sequence, heads, head_dim)
    v = linear(normalized, weights["v"]).reshape(batch, sequence, heads, head_dim)
    q, k = rms_norm(q), rms_norm(k)

    def rotary(tensor: jax.Array) -> jax.Array:
        half = tensor.shape[-1] // 2
        first, second = tensor[..., :half], tensor[..., half:]
        return jnp.concatenate(
            (first * cos + second * sin, first * (-sin) + second * cos), axis=-1
        )

    q, k = rotary(q), rotary(k)
    scores = jnp.einsum("bthd,bshd->bhts", q, k)
    scores = scores * jnp.asarray(head_dim**-0.5, scores.dtype)
    scores = jnp.where(causal_mask, scores, jnp.asarray(-jnp.inf, scores.dtype))
    probabilities = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhts,bshd->bthd", probabilities, v)
    attended = attended.reshape(batch, sequence, width)
    x = x + linear(attended, weights["o"])

    hidden = linear(rms_norm(x), weights["up"])
    hidden = jnp.square(jax.nn.relu(hidden))
    return x + linear(hidden, weights["down"])


def run_blocks(
    x: jax.Array,
    blocks: list[dict[str, jax.Array]],
    architecture: Architecture,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
) -> jax.Array:
    for weights in blocks:
        x = block(
            x,
            weights,
            heads=architecture.heads,
            cos=cos,
            sin=sin,
            causal_mask=causal_mask,
        )
    return x


def standard_hidden(
    embeddings: jax.Array,
    blocks: list[dict[str, jax.Array]],
    architecture: Architecture,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
) -> jax.Array:
    x = rms_norm(embeddings)
    x = run_blocks(x, blocks, architecture, cos, sin, causal_mask)
    return rms_norm(x)


def forward_standard(
    parameters: dict[str, object],
    tokens: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
    architecture: Architecture,
) -> jax.Array:
    embeddings = parameters["embedding"][tokens]
    hidden = standard_hidden(
        embeddings, parameters["blocks"], architecture, cos, sin, causal_mask
    )
    return linear(hidden[:, -1:, :], parameters["head"])


def forward_frozen_pilot(
    parameters: dict[str, object],
    tokens: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
    architecture: Architecture,
) -> jax.Array:
    embeddings = parameters["embedding"][tokens]
    first_hidden = standard_hidden(
        embeddings, parameters["blocks"], architecture, cos, sin, causal_mask
    )
    shifted = jnp.concatenate((jnp.zeros_like(first_hidden[:, :1]), first_hidden[:, :-1]), axis=1)
    projected = linear(shifted, parameters["hidden_projection"])
    gate = jax.nn.sigmoid(linear(rms_norm(embeddings), parameters["gate_projection"]))
    feedback = rms_norm(projected * gate)
    feedback = feedback.at[:, :1, :].set(embeddings[:, :1, :])
    hidden = standard_hidden(
        feedback, parameters["blocks"], architecture, cos, sin, causal_mask
    )
    return linear(hidden[:, -1:, :], parameters["head"])


def forward_paper_recurrent(
    parameters: dict[str, object],
    tokens: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    causal_mask: jax.Array,
    architecture: Architecture,
) -> jax.Array:
    embeddings = rms_norm(parameters["embedding"][tokens])
    embedded_input = run_blocks(
        embeddings, parameters["blocks"][:1], architecture, cos, sin, causal_mask
    )
    state = jnp.zeros_like(embedded_input)
    for _ in range(architecture.recurrences):
        state = linear(
            jnp.concatenate((state, embedded_input), axis=-1), parameters["adapter"]
        )
        state = run_blocks(
            state, parameters["blocks"][1:5], architecture, cos, sin, causal_mask
        )
        normalized = rms_norm(state) * parameters["core_norm_weight"]
        mix = jnp.tanh(parameters["core_norm_mix"])
        state = state + mix * (normalized - state)
    hidden = run_blocks(
        state, parameters["blocks"][5:], architecture, cos, sin, causal_mask
    )
    hidden = rms_norm(hidden)
    return linear(hidden[:, -1:, :], parameters["head"])


def parameter_spec(architecture: Architecture) -> dict[str, object]:
    shape = jax.ShapeDtypeStruct
    f32 = jnp.float32
    width = architecture.width
    parameters: dict[str, object] = {
        "embedding": shape((50304, width), f32),
        "head": shape((50304, width), f32),
        "blocks": [
            {
                "q": shape((width, width), f32),
                "k": shape((width, width), f32),
                "v": shape((width, width), f32),
                "o": shape((width, width), f32),
                "up": shape((architecture.mlp_width, width), f32),
                "down": shape((width, architecture.mlp_width), f32),
            }
            for _ in range(architecture.layers)
        ],
    }
    if architecture.kind == "frozen_pilot":
        parameters["hidden_projection"] = shape((width, width), f32)
        parameters["gate_projection"] = shape((width, width), f32)
    elif architecture.kind == "paper":
        parameters["adapter"] = shape((width, 2 * width), f32)
        parameters["core_norm_weight"] = shape((width,), f32)
        parameters["core_norm_mix"] = shape((), f32)
    return parameters


def measure(calculator_module, architecture: Architecture, sequence_length: int) -> dict[str, object]:
    tokens = jax.ShapeDtypeStruct((1, sequence_length), jnp.int32)
    half_head = architecture.head_dim // 2
    cos = jax.ShapeDtypeStruct((1, sequence_length, 1, half_head), jnp.float32)
    sin = jax.ShapeDtypeStruct((1, sequence_length, 1, half_head), jnp.float32)
    causal_mask = jax.ShapeDtypeStruct(
        (1, 1, sequence_length, sequence_length), jnp.bool_
    )
    function = {
        "standard": forward_standard,
        "frozen_pilot": forward_frozen_pilot,
        "paper": forward_paper_recurrent,
    }[architecture.kind]

    def traced(parameters, token_ids, rotary_cos, rotary_sin, mask):
        return function(
            parameters,
            token_ids,
            rotary_cos,
            rotary_sin,
            mask,
            architecture,
        )

    jaxpr = jax.make_jaxpr(traced)(
        parameter_spec(architecture), tokens, cos, sin, causal_mask
    ).jaxpr
    config = calculator_module.SerialDepthConfig(
        handle_unsupported_as_depth_0=True, verbose=False
    )
    result = calculator_module.SerialDepthCalculator(config).compute(jaxpr)
    return {
        "model": architecture.name,
        "kind": architecture.kind,
        "layers": architecture.layers,
        "width": architecture.width,
        "heads": architecture.heads,
        "head_dim": architecture.head_dim,
        "mlp_width": architecture.mlp_width,
        "parameters": architecture.parameter_count,
        "recurrences": architecture.recurrences,
        "effective_transformer_blocks": (
            1 + 4 * architecture.recurrences + 1
            if architecture.kind == "paper"
            else architecture.layers * architecture.recurrences
        ),
        "sequence_length": sequence_length,
        "serial_depth": result.depth,
        "jaxpr_operations": result.total_operations,
        "unsupported_primitives": sorted(result.unsupported_primitives),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-depth-repo", type=Path, required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[13, 103, 1024])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.serial_depth_repo))
    from serial_depth import depth as calculator_module  # pylint: disable=import-outside-toplevel

    commit = "unknown"
    git_head = args.serial_depth_repo / ".git" / "HEAD"
    if git_head.exists():
        head = git_head.read_text().strip()
        if head.startswith("ref: "):
            ref = args.serial_depth_repo / ".git" / head[5:]
            if ref.exists():
                commit = ref.read_text().strip()
        else:
            commit = head

    rows = [
        measure(calculator_module, architecture, sequence_length)
        for sequence_length in args.sequence_lengths
        for architecture in ARCHITECTURES
    ]
    payload = {
        "calculator": "google-deepmind/serial_depth",
        "calculator_commit": commit,
        "jax_version": jax.__version__,
        "note": "Forward-pass serial depth of an operation-faithful JAX transcription; unsupported primitives must be empty for a valid result.",
        "measurements": rows,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
