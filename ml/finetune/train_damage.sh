#!/usr/bin/env bash
# Stage 2: damage classification fine-tune (siamese U-Net)
set -euo pipefail
cd "$(dirname "$0")/../pytorch-xview2"

DATA_DIR="${DATA_DIR:-/data/train_subset}"
RESULTS_DIR="${RESULTS_DIR:-/results/dmg}"
CKPT_PRE="${CKPT_PRE:-/results/loc/checkpoints/best.ckpt}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ENCODER="${ENCODER:-resnet50}"

if [[ ! -f "$CKPT_PRE" ]]; then
  echo "Missing localization checkpoint: $CKPT_PRE"
  echo "Run train_localization.sh first."
  exit 1
fi

python main.py \
  --exec_mode train \
  --type post \
  --dmg_model siamese \
  --data "$DATA_DIR" \
  --results "$RESULTS_DIR" \
  --encoder "$ENCODER" \
  --loss_str focal+dice \
  --ckpt_pre "$CKPT_PRE" \
  --attention \
  --deep_supervision \
  --gpus 1 \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Damage fine-tune done -> $RESULTS_DIR"
