"""Evaluate project checkpoints or Hugging Face GPT-2 Small on LAMBADA OpenAI.

This follows lm-evaluation-harness task version 1.0: the context is every
space-separated token except the last, the target is the final token with its
leading space, accuracy is greedy teacher-forced exact match, and perplexity is
aggregated over target-word log likelihoods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT  # noqa: E402
from model import GPTConfig  # noqa: E402
from benchmarks.lambada.community_model import (  # noqa: E402
    CommunityDeepGPT,
    SoftcappedLMHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("shallow-1", "community-deep-3242", "gpt2-small"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts/checkpoints/shallow-fineweb/state_step004578.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shallow(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GPT(payload["model_config"])
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in payload["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    def hidden(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask  # Right padding cannot affect earlier causal positions.
        x = model.transformer.wte(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in model.transformer.h:
            x = block(x)
        return F.rms_norm(x, (x.size(-1),))

    metadata = {
        "name": "Shallow 1",
        "architecture": {
            "layers": model.config.n_layer,
            "width": model.config.n_embd,
            "heads": model.config.n_head,
            "mlp_width": model.config.n_ff,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
    }
    return model, hidden, model.lm_head, metadata


def load_gpt2(device: torch.device):
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval().to(device)

    def hidden(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    metadata = {
        "name": "OpenAI GPT-2 Small",
        "hub_model": "openai-community/gpt2",
        "hub_revision": getattr(model.config, "_commit_hash", None),
        "architecture": {
            "layers": model.config.n_layer,
            "width": model.config.n_embd,
            "heads": model.config.n_head,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    return model, hidden, model.lm_head, metadata


def load_community_deep(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("step") != 3242:
        raise ValueError(f"expected checkpoint step 3242, found {payload.get('step')}")
    config = GPTConfig(
        vocab_size=50304,
        n_layer=12,
        n_head=6,
        n_embd=768,
        n_ff=3072,
    )
    model = CommunityDeepGPT(config)
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in payload["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    def hidden(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return model.hidden(input_ids)

    metadata = {
        "name": "Community Deep 3242",
        "architecture": {
            "layers": config.n_layer,
            "width": config.n_embd,
            "heads": config.n_head,
            "mlp_width": config.n_ff,
            "parameters": sum(p.numel() for p in model.parameters()),
            "features": [
                "cross-layer value sharing",
                "learned residual mixing",
                "logit soft-cap 30",
            ],
        },
        "training_step": payload["step"],
        "reported_fineweb_validation_loss": 3.2766,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "source": {
            "hub_repo": "Fizzarolli/modded-nanogpt-logs",
            "hub_revision": "fc7790cb0e5eeca2716186aa627842baa3399044",
            "hub_path": "abba5381-1376-415f-a331-a869506e243d/state_step003242.pt",
            "provenance": "community upload; not the official 10.8-minute checkpoint",
        },
    }
    return model, hidden, SoftcappedLMHead(model.lm_head), metadata


def tokenize_examples(tokenizer, rows: list[str], max_length: int = 1024):
    examples = []
    truncated = 0
    for index, text in enumerate(rows):
        pieces = text.split(" ")
        context = " ".join(pieces[:-1])
        target = " " + pieces[-1]
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        max_context = max_length - len(target_ids)
        if len(context_ids) > max_context:
            context_ids = context_ids[-max_context:]
            truncated += 1
        if not context_ids or not target_ids:
            raise ValueError(f"empty context or target at example {index}")
        examples.append(
            {
                "index": index,
                "input_ids": context_ids + target_ids[:-1],
                "prediction_start": len(context_ids) - 1,
                "target_ids": target_ids,
                "target_text": target,
            }
        )
    return examples, truncated


def score(
    examples: list[dict],
    hidden_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lm_head: torch.nn.Module,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
) -> dict:
    correct_by_example: list[bool] = []
    loglikelihood_by_example: list[float] = []
    target_tokens = 0
    logical_input_tokens = 0

    def run_batch(batch: list[dict], record: bool) -> None:
        nonlocal target_tokens, logical_input_tokens
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), max_len), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (len(batch), max_len), dtype=torch.long, device=device
        )
        for row, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row, :length] = torch.tensor(item["input_ids"], device=device)
            attention_mask[row, :length] = 1

        states = hidden_fn(input_ids, attention_mask)
        selected_states = []
        targets = []
        spans = []
        offset = 0
        for row, item in enumerate(batch):
            count = len(item["target_ids"])
            start = item["prediction_start"]
            selected_states.append(states[row, start : start + count])
            targets.extend(item["target_ids"])
            spans.append((offset, offset + count))
            offset += count
        logits = lm_head(torch.cat(selected_states, dim=0)).float()
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        token_logprobs = F.log_softmax(logits, dim=-1).gather(
            1, target_tensor[:, None]
        )[:, 0]
        predictions = logits.argmax(dim=-1)

        if record:
            logical_input_tokens += sum(len(item["input_ids"]) for item in batch)
            target_tokens += len(targets)
            for start, end in spans:
                correct_by_example.append(
                    bool(torch.equal(predictions[start:end], target_tensor[start:end]))
                )
                loglikelihood_by_example.append(
                    float(token_logprobs[start:end].sum().item())
                )

    with torch.inference_mode():
        run_batch(examples[: min(batch_size, len(examples))], record=False)
        synchronize(device)
        started = time.perf_counter()
        for start in range(0, len(examples), batch_size):
            run_batch(examples[start : start + batch_size], record=True)
        synchronize(device)
        elapsed = time.perf_counter() - started

    accuracy = sum(correct_by_example) / len(correct_by_example)
    mean_word_loglikelihood = sum(loglikelihood_by_example) / len(
        loglikelihood_by_example
    )
    return {
        "examples": len(correct_by_example),
        "correct": sum(correct_by_example),
        "accuracy": accuracy,
        "perplexity": math.exp(-mean_word_loglikelihood),
        "target_token_perplexity": math.exp(
            -sum(loglikelihood_by_example) / target_tokens
        ),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(correct_by_example) / elapsed,
        "logical_input_tokens": logical_input_tokens,
        "logical_input_tokens_per_second": logical_input_tokens / elapsed,
        "target_tokens": target_tokens,
        "correct_by_example": correct_by_example,
        "loglikelihood_by_example": loglikelihood_by_example,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    rows = dataset["text"]
    if args.limit is not None:
        rows = rows[: args.limit]
    examples, truncated = tokenize_examples(tokenizer, rows)

    if args.model == "shallow-1":
        model, hidden_fn, lm_head, model_metadata = load_shallow(
            args.checkpoint, device
        )
    elif args.model == "community-deep-3242":
        model, hidden_fn, lm_head, model_metadata = load_community_deep(
            args.checkpoint, device
        )
    else:
        model, hidden_fn, lm_head, model_metadata = load_gpt2(device)

    metrics = score(
        examples,
        hidden_fn,
        lm_head,
        device,
        tokenizer.pad_token_id,
        args.batch_size,
    )
    result = {
        "benchmark": "lambada_openai",
        "task_version": "1.0",
        "dataset": "EleutherAI/lambada_openai",
        "dataset_fingerprint": dataset._fingerprint,
        "split": "test",
        "scoring": "lm-evaluation-harness-compatible greedy exact match and target-word perplexity",
        "model": model_metadata,
        "runtime": {
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "batch_size": args.batch_size,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "truncated_examples": truncated,
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "model": model_metadata["name"],
        **{key: value for key, value in metrics.items() if not key.endswith("_by_example")},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
