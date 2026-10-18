"""Shared training utilities for the GSM8K experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from benchmarks.multiplication.cot import strip_compile_prefix
from model import GPT


def load_model(
    path: Path, device: torch.device, dtype: torch.dtype
) -> tuple[GPT, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(payload["model_config"])
    model.load_state_dict(strip_compile_prefix(payload["model"]), strict=True)
    return model.to(device=device, dtype=dtype), payload


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.AdamW:
    kwargs = {"fused": True} if next(model.parameters()).device.type == "cuda" else {}
    return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)


def serializable_args(namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")
        handle.flush()


def save_checkpoint_atomic(
    path: Path,
    model: GPT,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    train_args: dict,
    source_payload: dict,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model.config,
        "model": model.state_dict(),
        "base_step": source_payload.get("base_step", source_payload.get("step")),
        "source_checkpoint_stage": source_payload.get("stage"),
        "train_args": train_args,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
