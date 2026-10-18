#!/usr/bin/env bash
set -euo pipefail

# General continuation of the paper-format Deep 1 multiplication experiment.
# The default is the next unrun size (6x6). Override DIGITS and paths for later
# sizes while keeping the optimizer and token-removal recipe unchanged.

DIGITS="${DIGITS:-6}"
REPO="${REPO:-/workspace/modded-nanogpt-depth}"
RUN_ROOT="${RUN_ROOT:-/workspace/experiment-results}"
DEEP_BASE="${DEEP_BASE:-/workspace/checkpoints/deep1-fineweb-matched.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/paper-data/${DIGITS}_by_${DIGITS}_mult}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_ROOT/train.txt}"
VALIDATION_PATH="${VALIDATION_PATH:-$DATA_ROOT/valid.txt}"
if [[ -f "$DATA_ROOT/test_bigbench.txt" ]]; then
  DEFAULT_TEST_PATH="$DATA_ROOT/test_bigbench.txt"
else
  # The paper's additional 6x6/8x8/etc. Drive folders call this split
  # src1_test.txt, while the vendored benchmark sizes use test_bigbench.txt.
  DEFAULT_TEST_PATH="$DATA_ROOT/src1_test.txt"
fi
TEST_PATH="${TEST_PATH:-$DEFAULT_TEST_PATH}"
CACHE_DIR="${CACHE_DIR:-/workspace/token-cache/${DIGITS}x${DIGITS}}"

EXPLICIT_DIR="$RUN_ROOT/deep1-${DIGITS}x${DIGITS}-explicit-cot-fp32"
INTERNAL_DIR="$RUN_ROOT/deep1-${DIGITS}x${DIGITS}-internalized-cot-fp32"
REMOVE_PER_EPOCH="${REMOVE_PER_EPOCH:-8}"
ADAPTATION_EPOCHS="${ADAPTATION_EPOCHS:-5}"
# Uniform paper traces follow 3*n^2+n-5 GPT-2 CoT tokens (25, 47, 75,
# 109, 149, ...). Allow an override so the trainer remains the final check.
COT_TOKENS="${COT_TOKENS:-$((3 * DIGITS * DIGITS + DIGITS - 5))}"
INTERNAL_EPOCHS="${INTERNAL_EPOCHS:-$(((COT_TOKENS + REMOVE_PER_EPOCH - 1) / REMOVE_PER_EPOCH + ADAPTATION_EPOCHS))}"

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$EXPLICIT_DIR" "$INTERNAL_DIR" "$CACHE_DIR"

for required in "$DEEP_BASE" "$TRAIN_PATH" "$VALIDATION_PATH" "$TEST_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

echo "digits=$DIGITS cot_tokens_expected=$COT_TOKENS internal_epochs=$INTERNAL_EPOCHS"
echo "base_sha256=$(sha256sum "$DEEP_BASE" | awk '{print $1}')"
echo "train_sha256=$(sha256sum "$TRAIN_PATH" | awk '{print $1}')"
echo "validation_sha256=$(sha256sum "$VALIDATION_PATH" | awk '{print $1}')"
echo "test_sha256=$(sha256sum "$TEST_PATH" | awk '{print $1}')"

if [[ -f "$EXPLICIT_DIR/explicit_cot_epoch_000.pt" ]]; then
  echo "stage=deep1_${DIGITS}x${DIGITS}_explicit_cot already_complete=true"
else
  echo "stage=deep1_${DIGITS}x${DIGITS}_explicit_cot started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -m benchmarks.multiplication.train_explicit_cot \
    --checkpoint "$DEEP_BASE" \
    --train-path "$TRAIN_PATH" \
    --validation-path "$VALIDATION_PATH" \
    --test-path "$TEST_PATH" \
    --output-dir "$EXPLICIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --digits "$DIGITS" \
    --epochs 1 \
    --batch-size 32 \
    --accumulate 1 \
    --lr 5e-5 \
    --eval-examples 1000 \
    --eval-batch-size 32 \
    2>&1 | tee "$EXPLICIT_DIR/train.log"
  echo "stage=deep1_${DIGITS}x${DIGITS}_explicit_cot finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

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

if [[ -f "$INTERNAL_DIR/test_metrics.json" ]]; then
  echo "stage=deep1_${DIGITS}x${DIGITS}_internalization already_complete=true"
elif [[ -f "$INTERNAL_DIR/latest.pt" ]]; then
  echo "refusing to restart interrupted internalization; latest.pt exists at $INTERNAL_DIR" >&2
  exit 2
else
  echo "stage=deep1_${DIGITS}x${DIGITS}_internalization started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -m benchmarks.multiplication.train_internalized_cot \
    --checkpoint "$EXPLICIT_DIR/explicit_cot_epoch_000.pt" \
    --train-path "$TRAIN_PATH" \
    --validation-path "$VALIDATION_PATH" \
    --test-path "$TEST_PATH" \
    --output-dir "$INTERNAL_DIR" \
    --cache-dir "$CACHE_DIR" \
    --digits "$DIGITS" \
    --epochs "$INTERNAL_EPOCHS" \
    --batch-size 32 \
    --accumulate 1 \
    --lr 5e-5 \
    --remove-per-epoch "$REMOVE_PER_EPOCH" \
    --removal-smoothing-lambda 4 \
    --target-validation-accuracy 0.99 \
    --eval-examples 1000 \
    --eval-batch-size 32 \
    2>&1 | tee "$INTERNAL_DIR/train.log"
  echo "stage=deep1_${DIGITS}x${DIGITS}_internalization finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
