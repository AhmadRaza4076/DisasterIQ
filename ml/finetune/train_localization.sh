#!/usr/bin/env bash
# Stage 1: building localization on hackathon train subset
set -euo pipefail
cd "$(dirname "$0")/../pytorch-xview2"

DATA_DIR="${DATA_DIR:-/data/train_subset}"
RESULTS_DIR="${RESULTS_DIR:-/results/loc}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ENCODER="${ENCODER:-resnet50}"

python main.py \
  --exec_mode train \
  --type pre \
  --data "$DATA_DIR" \
  --results "$RESULTS_DIR" \
  --encoder "$ENCODER" \
  --loss_str ce+dice \
  --deep_supervision \
  --gpus 1 \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Localization training done -> $RESULTS_DIR"
