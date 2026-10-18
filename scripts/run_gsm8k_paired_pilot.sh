#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/workspace/gsm8k-depth}
REPO="$ROOT/repo"
DATA="$ROOT/data"
BASE="$ROOT/base"
RUNS="$ROOT/runs"
CACHE="$ROOT/cache"
LOG="$RUNS/paired-pilot.log"

mkdir -p "$RUNS" "$CACHE"
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false

run_stage() {
  local name=$1
  shift
  echo "[$(date -u +%FT%TZ)] START $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "[$(date -u +%FT%TZ)] DONE $name" | tee -a "$LOG"
}

COMMON_EXPLICIT=(
  --train-path "$DATA/train.txt"
  --validation-path "$DATA/valid.txt"
  --test-path "$DATA/test.txt"
  --cache-dir "$CACHE"
  --epochs 1
  --batch-size 32
  --accumulate 1
  --lr 5e-5
  --max-length 150
  --seed 3456
  --log-every 100
  --eval-batch-size 16
  --eval-max-new-tokens 800
  --skip-test
)

COMMON_INTERNAL=(
  --train-path "$DATA/train.txt"
  --validation-path "$DATA/valid.txt"
  --test-path "$DATA/test.txt"
  --cache-dir "$CACHE"
  --epochs 43
  --batch-size 32
  --accumulate 1
  --lr 5e-5
  --max-length 150
  --remove-per-epoch 1
  --remove-all-when-remove-beyond 39
  --removal-smoothing-lambda 4
  --seed 3456
  --log-every 100
  --eval-removals 0,8,16,24,32
  --eval-batch-size 16
  --eval-max-new-tokens 800
  --max-runtime-hours 24
  --skip-test
)

run_stage deep1-explicit \
  python3 -u -m benchmarks.gsm8k.train_explicit_cot \
  --checkpoint "$BASE/deep1-fineweb-slim.pt" \
  --output-dir "$RUNS/deep1-explicit" \
  "${COMMON_EXPLICIT[@]}"

run_stage shallow1-explicit \
  python3 -u -m benchmarks.gsm8k.train_explicit_cot \
  --checkpoint "$BASE/shallow1-fineweb-slim.pt" \
  --output-dir "$RUNS/shallow1-explicit" \
  "${COMMON_EXPLICIT[@]}"

run_stage deep1-internalized \
  python3 -u -m benchmarks.gsm8k.train_internalized_cot \
  --checkpoint "$RUNS/deep1-explicit/best.pt" \
  --output-dir "$RUNS/deep1-internalized" \
  "${COMMON_INTERNAL[@]}"

run_stage shallow1-internalized \
  python3 -u -m benchmarks.gsm8k.train_internalized_cot \
  --checkpoint "$RUNS/shallow1-explicit/best.pt" \
  --output-dir "$RUNS/shallow1-internalized" \
  "${COMMON_INTERNAL[@]}"

date -u +%FT%TZ > "$RUNS/PILOT_COMPLETE"
echo "[$(date -u +%FT%TZ)] PILOT COMPLETE" | tee -a "$LOG"
