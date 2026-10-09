#!/usr/bin/env bash
set -euo pipefail

repo_dir=${REPO_DIR:-/workspace/modded-nanogpt-depth-continuation}
output_dir=${OUTPUT_DIR:-/root/deep1-fineweb-matched}
target_loss=${TARGET_LOSS:-3.3074311275482178}

mkdir -p "$output_dir"
cd "$repo_dir"

exec torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --n-layer 12 \
  --n-head 6 \
  --n-embd 768 \
  --n-ff 3072 \
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
