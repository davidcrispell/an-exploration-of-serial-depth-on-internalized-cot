#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/modded-nanogpt-depth}"
INPUTS="${INPUTS:-/workspace/experiment-inputs}"
RUN_ROOT="${RUN_ROOT:-/workspace/experiment-results}"
OUTPUT="${OUTPUT:-$RUN_ROOT/shallow1-1-paper-recurrent-5x5-internalized-cot-fp32}"
CACHE="${CACHE:-/workspace/token-cache/5x5-paper-recurrent}"
CHECKPOINT="${CHECKPOINT:-$INPUTS/shallow1-5x5-explicit.pt}"
TRAIN="${TRAIN:-$INPUTS/train.txt}"
VALIDATION="${VALIDATION:-$INPUTS/validation.txt}"
TEST="${TEST:-$INPUTS/test.txt}"
PYTHON="${PYTHON:-python3}"

EXPECTED_CHECKPOINT_SHA="c9a95971305960d4d8152b257088df723f8447c7064bcea35ea90b2a01d585ee"
EXPECTED_TRAIN_SHA="e41642536746292598fb5fa53bf8ebb01358a161a6ba78892d0579ccc1489253"
EXPECTED_VALIDATION_SHA="d3b95bdf0f74863a88bc777fee3fe23854e6ad458b8e7b847b65dac25e59182c"
EXPECTED_TEST_SHA="661ab1db2ea0d33fc3e6e398c75920a7a8471f64ebad481bcaf5c8a8ff2b9c4b"

verify() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path: expected=$expected actual=$actual" >&2
    exit 1
  fi
  echo "verified_sha256=$actual file=$path"
}

verify "$EXPECTED_CHECKPOINT_SHA" "$CHECKPOINT"
verify "$EXPECTED_TRAIN_SHA" "$TRAIN"
verify "$EXPECTED_VALIDATION_SHA" "$VALIDATION"
verify "$EXPECTED_TEST_SHA" "$TEST"

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$OUTPUT" "$CACHE"
echo "stage=shallow1_1_paper_recurrent_5x5 started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" -m benchmarks.multiplication.train_paper_recurrent_internalized_cot \
  --checkpoint "$CHECKPOINT" \
  --train-path "$TRAIN" \
  --validation-path "$VALIDATION" \
  --test-path "$TEST" \
  --output-dir "$OUTPUT" \
  --cache-dir "$CACHE" \
  --digits 5 \
  --warmup-max-epochs 3 \
  --epochs 15 \
  --batch-size 32 \
  --accumulate 1 \
  --lr 5e-5 \
  --remove-per-epoch 8 \
  --removal-smoothing-lambda 4 \
  --target-validation-accuracy 0.99 \
  --seed 3456 \
  --log-every 100 \
  --eval-examples 1000 \
  --eval-batch-size 32 \
  --mean-recurrence 4 \
  --mean-backprop-depth 4 \
  --eval-recurrences 1 2 4 8 \
  --primary-eval-recurrence 4 \
  --recurrence-sigma 0.5 \
  2>&1 | tee "$OUTPUT/train.log"

echo "stage=shallow1_1_paper_recurrent_5x5 finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
