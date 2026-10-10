#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/modded-nanogpt-depth}"
RUN_ROOT="${RUN_ROOT:-/root/experiment-results}"
DEEP_BASE="${DEEP_BASE:-/root/checkpoints/deep1-fineweb-matched.pt}"
TRAIN_5X5="${TRAIN_5X5:-/root/paper-data/5_by_5_mult/train.txt}"
CACHE_5X5="${CACHE_5X5:-/root/token-cache/5x5}"

DEEP_5X5_EXPLICIT="$RUN_ROOT/deep1-5x5-explicit-cot-fp32"
DEEP_5X5_INTERNAL="$RUN_ROOT/deep1-5x5-internalized-cot-fp32"

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$DEEP_5X5_EXPLICIT" "$DEEP_5X5_INTERNAL" "$CACHE_5X5"

for required in "$DEEP_BASE" "$TRAIN_5X5"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

echo "base_sha256=$(sha256sum "$DEEP_BASE" | awk '{print $1}')"
echo "train_sha256=$(sha256sum "$TRAIN_5X5" | awk '{print $1}')"

echo "stage=deep1_5x5_explicit_cot started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m benchmarks.multiplication.train_explicit_cot \
  --checkpoint "$DEEP_BASE" \
  --train-path "$TRAIN_5X5" \
  --validation-path "$REPO/benchmarks/multiplication/data/5x5/validation.txt" \
  --test-path "$REPO/benchmarks/multiplication/data/5x5/test.txt" \
  --output-dir "$DEEP_5X5_EXPLICIT" \
  --cache-dir "$CACHE_5X5" \
  --digits 5 \
  --epochs 1 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$DEEP_5X5_EXPLICIT/train.log"

python - "$DEEP_5X5_EXPLICIT" <<'PY'
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
echo "stage=deep1_5x5_explicit_cot finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "stage=deep1_5x5_internalization started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m benchmarks.multiplication.train_internalized_cot \
  --checkpoint "$DEEP_5X5_EXPLICIT/explicit_cot_epoch_000.pt" \
  --train-path "$TRAIN_5X5" \
  --validation-path "$REPO/benchmarks/multiplication/data/5x5/validation.txt" \
  --test-path "$REPO/benchmarks/multiplication/data/5x5/test.txt" \
  --output-dir "$DEEP_5X5_INTERNAL" \
  --cache-dir "$CACHE_5X5" \
  --digits 5 \
  --epochs 15 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --remove-per-epoch 8 \
  --removal-smoothing-lambda 4 \
  --target-validation-accuracy 0.99 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$DEEP_5X5_INTERNAL/train.log"
echo "stage=deep1_5x5_internalization finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
