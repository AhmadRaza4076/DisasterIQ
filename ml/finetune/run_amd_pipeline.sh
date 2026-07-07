#!/usr/bin/env bash
# Bootstrap AMD GPU instance and run full fine-tune pipeline
set -euo pipefail

echo "=== ROCm GPU check ==="
rocm-smi || true
python3 -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

REPO_ROOT="${REPO_ROOT:-$HOME/DisasterIQ}"
cd "$REPO_ROOT/ml/pytorch-xview2"

export DATA_DIR="${DATA_DIR:-/data/train_subset}"
export RESULTS_ROOT="${RESULTS_ROOT:-/results}"
mkdir -p "$RESULTS_ROOT"

echo "=== Stage 1: localization ==="
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/loc" \
  bash ../finetune/train_localization.sh

echo "=== Stage 2: damage fine-tune ==="
DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/dmg" \
  CKPT_PRE="$RESULTS_ROOT/loc/checkpoints/best.ckpt" \
  bash ../finetune/train_damage.sh

echo "=== Eval on test set ==="
python main.py --exec_mode eval --type post \
  --ckpt "$RESULTS_ROOT/dmg/checkpoints/best.ckpt" \
  --data /data/test --results "$RESULTS_ROOT/eval" \
  --gpus 1 --val_batch_size 4

echo "Done. Checkpoints:"
ls -la "$RESULTS_ROOT/dmg/checkpoints/" || true
