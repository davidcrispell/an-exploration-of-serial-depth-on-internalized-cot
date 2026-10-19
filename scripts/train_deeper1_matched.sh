#!/usr/bin/env bash
set -euo pipefail

repo_dir=${REPO_DIR:-/workspace/modded-nanogpt-depth-continuation}
output_dir=${OUTPUT_DIR:-/root/deeper1-fineweb-matched}
target_loss=${TARGET_LOSS:-3.3074311275482178}

mkdir -p "$output_dir"
cd "$repo_dir"

# Deeper 1: exactly parameter-matched to Shallow 1 and Deep 1.
# Five heads give a kernel-compatible 120-dimensional head. Attention parameter
# count is independent of the number of heads.
# Parameter count with vocab_size=50,304:
#   2*50,304*600 + 24*(4*600^2 + 2*600*2,336) = 162,201,600
exec torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --n-layer 24 \
  --n-head 5 \
  --n-embd 600 \
  --n-ff 2336 \
  --num-iterations 4578 \
  --warmup-iters 0 \
  --warmdown-iters 1308 \
  --lr-scale 1.0 \
  --target-val-loss "$target_loss" \
  --val-loss-every 25 \
  --save-every 25 \
  --batch-size 512 \
  --device-batch-size 64 \
  --sequence-length 1024 \
  --val-tokens 10485760 \
  --input-bin "$repo_dir/data/fineweb10B/fineweb_train_*.bin" \
  --input-val-bin "$repo_dir/data/fineweb10B/fineweb_val_*.bin" \
  --output-dir "$output_dir"
