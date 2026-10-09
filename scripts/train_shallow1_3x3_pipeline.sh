#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/modded-nanogpt-depth}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/checkpoints/shallow1-matched/latest.pt}"
TRAIN_PATH="${TRAIN_PATH:-/root/multiplication-3x3/train.txt}"
RUN_ROOT="${RUN_ROOT:-/root/experiment-results}"
CACHE_DIR="${CACHE_DIR:-/root/token-cache/3x3}"
EXPLICIT_DIR="$RUN_ROOT/shallow1-3x3-explicit-cot-fp32"
INTERNAL_DIR="$RUN_ROOT/shallow1-3x3-internalized-cot-fp32"
VALIDATION_PATH="$REPO/benchmarks/multiplication/data/3x3/validation.txt"
TEST_PATH="$REPO/benchmarks/multiplication/data/3x3/test.txt"

cd "$REPO"
export PYTHONPATH="$REPO"

mkdir -p "$EXPLICIT_DIR" "$INTERNAL_DIR" "$CACHE_DIR"

python -m benchmarks.multiplication.train_explicit_cot \
  --checkpoint "$BASE_CHECKPOINT" \
  --train-path "$TRAIN_PATH" \
  --validation-path "$VALIDATION_PATH" \
  --test-path "$TEST_PATH" \
  --output-dir "$EXPLICIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --digits 3 \
  --epochs 1 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$EXPLICIT_DIR/train.log"

python - "$EXPLICIT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
validation = json.loads((root / "validation_metrics.json").read_text())[-1]
test = json.loads((root / "test_metrics.json").read_text())
if validation["accuracy"] < 0.99 or test["accuracy"] < 0.99:
    raise SystemExit(
        "explicit-CoT gate failed: "
        f"validation={validation['accuracy']:.4%}, test={test['accuracy']:.4%}"
    )
print(
    "explicit-CoT gate passed: "
    f"validation={validation['accuracy']:.4%}, test={test['accuracy']:.4%}",
    flush=True,
)
PY

python -m benchmarks.multiplication.train_internalized_cot \
  --checkpoint "$EXPLICIT_DIR/explicit_cot_epoch_000.pt" \
  --train-path "$TRAIN_PATH" \
  --validation-path "$VALIDATION_PATH" \
  --test-path "$TEST_PATH" \
  --output-dir "$INTERNAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --digits 3 \
  --epochs 8 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --remove-per-epoch 8 \
  --removal-smoothing-lambda 4 \
  --target-validation-accuracy 0.99 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$INTERNAL_DIR/train.log"
