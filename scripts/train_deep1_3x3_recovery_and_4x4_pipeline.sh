#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/modded-nanogpt-depth}"
RUN_ROOT="${RUN_ROOT:-/root/experiment-results}"
DEEP_BASE="${DEEP_BASE:-/root/checkpoints/deep1-fineweb-matched.pt}"
DEEP_3X3_EXPLICIT="${DEEP_3X3_EXPLICIT:-/root/checkpoints/deep1-3x3-explicit-cot.pt}"
TRAIN_3X3="${TRAIN_3X3:-/root/multiplication-3x3/train.txt}"
TRAIN_4X4="${TRAIN_4X4:-/workspace/paper-data/4_by_4_mult/train.txt}"
CACHE_3X3="${CACHE_3X3:-/root/token-cache/3x3}"
CACHE_4X4="${CACHE_4X4:-/workspace/token-cache}"

DEEP_3X3_INTERNAL="$RUN_ROOT/deep1-3x3-internalized-cot-fp32-recovery"
DEEP_4X4_EXPLICIT="$RUN_ROOT/deep1-4x4-explicit-cot-fp32"
DEEP_4X4_INTERNAL="$RUN_ROOT/deep1-4x4-internalized-cot-fp32"

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p \
  "$DEEP_3X3_INTERNAL" \
  "$DEEP_4X4_EXPLICIT" \
  "$DEEP_4X4_INTERNAL" \
  "$CACHE_3X3"

for required in "$DEEP_BASE" "$DEEP_3X3_EXPLICIT" "$TRAIN_3X3" "$TRAIN_4X4"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

echo "stage=deep1_3x3_internalization_recovery started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m benchmarks.multiplication.train_internalized_cot \
  --checkpoint "$DEEP_3X3_EXPLICIT" \
  --train-path "$TRAIN_3X3" \
  --validation-path "$REPO/benchmarks/multiplication/data/3x3/validation.txt" \
  --test-path "$REPO/benchmarks/multiplication/data/3x3/test.txt" \
  --output-dir "$DEEP_3X3_INTERNAL" \
  --cache-dir "$CACHE_3X3" \
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
  2>&1 | tee "$DEEP_3X3_INTERNAL/train.log"
echo "stage=deep1_3x3_internalization_recovery finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "stage=deep1_4x4_explicit_cot started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m benchmarks.multiplication.train_explicit_cot \
  --checkpoint "$DEEP_BASE" \
  --train-path "$TRAIN_4X4" \
  --validation-path "$REPO/benchmarks/multiplication/data/4x4/validation.txt" \
  --test-path "$REPO/benchmarks/multiplication/data/4x4/test.txt" \
  --output-dir "$DEEP_4X4_EXPLICIT" \
  --cache-dir "$CACHE_4X4" \
  --digits 4 \
  --epochs 1 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$DEEP_4X4_EXPLICIT/train.log"

python - "$DEEP_4X4_EXPLICIT" <<'PY'
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
echo "stage=deep1_4x4_explicit_cot finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "stage=deep1_4x4_internalization started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m benchmarks.multiplication.train_internalized_cot \
  --checkpoint "$DEEP_4X4_EXPLICIT/explicit_cot_epoch_000.pt" \
  --train-path "$TRAIN_4X4" \
  --validation-path "$REPO/benchmarks/multiplication/data/4x4/validation.txt" \
  --test-path "$REPO/benchmarks/multiplication/data/4x4/test.txt" \
  --output-dir "$DEEP_4X4_INTERNAL" \
  --cache-dir "$CACHE_4X4" \
  --digits 4 \
  --epochs 8 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --remove-per-epoch 8 \
  --removal-smoothing-lambda 4 \
  --target-validation-accuracy 0.99 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  2>&1 | tee "$DEEP_4X4_INTERNAL/train.log"
echo "stage=deep1_4x4_internalization finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
