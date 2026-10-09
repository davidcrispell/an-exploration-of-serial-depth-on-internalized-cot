"""Small, import-safe helpers shared by pretraining and continuation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def linear_warmup_warmdown_factor(
    step: int,
    *,
    num_iterations: int,
    warmup_iters: int,
    warmdown_iters: int,
) -> float:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if warmup_iters < 0 or warmdown_iters < 0:
        raise ValueError("warmup_iters and warmdown_iters must be nonnegative")
    if warmup_iters + warmdown_iters > num_iterations:
        raise ValueError("warmup_iters + warmdown_iters cannot exceed num_iterations")
    if not 0 <= step <= num_iterations:
        raise ValueError("step must be between zero and num_iterations")
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    if step < num_iterations - warmdown_iters:
        return 1.0
    if warmdown_iters == 0:
        return 1.0
    return (num_iterations - step) / warmdown_iters


def strip_compiled_prefix(state: Mapping[str, object]) -> dict[str, object]:
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def restore_optimizer_states(
    optimizers: Sequence[Any],
    state_dicts: Sequence[Mapping[str, object]],
    learning_rates: Sequence[float],
) -> None:
    """Restore optimizer moments while starting a new constant-LR schedule."""
    if not (len(optimizers) == len(state_dicts) == len(learning_rates)):
        raise ValueError("optimizer, state, and learning-rate counts must match")
    for optimizer, state_dict, learning_rate in zip(
        optimizers, state_dicts, learning_rates
    ):
        optimizer.load_state_dict(state_dict)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
