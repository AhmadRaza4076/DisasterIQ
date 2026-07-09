#!/usr/bin/env bash
# DisasterIQ fine-tune pipeline for Kaggle Notebooks (CUDA GPU)
# Usage (from repo root on Kaggle):
#   bash ml/finetune/run_kaggle_pipeline.sh
#   bash ml/finetune/run_kaggle_pipeline.sh --stage loc   # localization only
#   bash ml/finetune/run_kaggle_pipeline.sh --stage dmg   # damage only (needs loc ckpt)
set -euo pipefail

STAGE="all"
if [[ "${1:-}" == "--stage" ]]; then
  STAGE="${2:-all}"
  shift 2 || true
elif [[ "${1:-}" == "--prep-only" ]]; then
  STAGE="prep"
  shift
elif [[ "${1:-}" == "--train-only" ]]; then
  STAGE="train"
  shift
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
FINETUNE_DIR="$REPO_ROOT/ml/finetune"
WORKING="${KAGGLE_WORKING:-/kaggle/working}"
INPUT_ROOT="${KAGGLE_INPUT:-/kaggle/input}"

export FINETUNE_CONFIG="${FINETUNE_CONFIG:-$FINETUNE_DIR/config_subset_kaggle.yaml}"

require_gpu() {
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: CUDA not available. Kaggle: Settings → Accelerator → GPU, restart, re-run." >&2
    exit 1
  }
}

find_train_subset() {
  local candidate
  for candidate in \
    "$WORKING/data/train_subset" \
    "$INPUT_ROOT/disasteriq-train-subset" \
    "$INPUT_ROOT"/disasteriq-train-subset/train_subset \
    "$INPUT_ROOT"/*/train_subset \
    "$INPUT_ROOT"/*/*/*/train_subset \
    "$INPUT_ROOT"/*/
  do
    [[ -d "$candidate/images" && -d "$candidate/targets" ]] || continue
    echo "$candidate"
    return 0
  done
  # Fallback: walk /kaggle/input for any train_subset layout
  python3 - <<'PY'
from pathlib import Path
root = Path("/kaggle/input")
for images in root.rglob("images"):
    parent = images.parent
    if (parent / "targets").is_dir():
        print(parent)
        break
PY
}

echo "=== CUDA GPU check ==="
if [[ "$STAGE" != "prep" ]]; then
  python3 -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  require_gpu
else
  python3 -c "import torch; print('cuda (prep-only):', torch.cuda.is_available())" || true
fi

if [[ ! -d "$REPO_ROOT/ml/pytorch-xview2" ]]; then
  echo "Missing ml/pytorch-xview2 — clone michal2409/xView2 first."
  exit 1
fi

SRC_SUBSET="$(find_train_subset)"
if [[ -z "$SRC_SUBSET" || ! -d "$SRC_SUBSET/images" ]]; then
  echo "ERROR: Could not find train_subset under $INPUT_ROOT" >&2
  exit 1
fi
DEST_SUBSET="$WORKING/data/train_subset"
if [[ "$SRC_SUBSET" != "$DEST_SUBSET" ]]; then
  echo "=== Staging train_subset: $SRC_SUBSET -> $DEST_SUBSET ==="
  mkdir -p "$WORKING/data"
  rm -rf "$DEST_SUBSET"
  cp -a "$SRC_SUBSET" "$DEST_SUBSET"
fi
export DATA_DIR="$DEST_SUBSET"

echo "=== Patch upstream xView2 for subset training ==="
python3 "$FINETUNE_DIR/patch_pytorch_xview2.py"

INDEX_OUT="$REPO_ROOT/ml/pytorch-xview2/utils/index.csv"
export XVIEW2_INDEX_CSV="$INDEX_OUT"

echo "=== Generate index.csv for $DATA_DIR ==="
if [[ -f "$INDEX_OUT" ]] && [[ "$(wc -l < "$INDEX_OUT")" -gt 1 ]]; then
  echo "index.csv already exists ($(wc -l < "$INDEX_OUT") lines) — skipping regeneration"
else
  python3 "$REPO_ROOT/scripts/generate_subset_index.py" \
    --data-dir "$DATA_DIR" \
    --out "$INDEX_OUT"
fi

echo "=== CPU dataset smoke test ==="
python3 "$REPO_ROOT/scripts/test_pytorch_dataset.py" --data-dir "$DATA_DIR"

eval "$(python3 "$FINETUNE_DIR/load_config.py" --config "$FINETUNE_CONFIG" data)"
mkdir -p "$RESULTS_ROOT"

if [[ "$STAGE" == "prep" ]]; then
  echo "Prep complete (no training)."
  exit 0
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "loc" ]]; then
  echo "=== Stage 1: localization ==="
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/loc" \
    FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_localization.sh"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "dmg" ]]; then
  echo "=== Stage 2: damage fine-tune ==="
  DATA_DIR="$DATA_DIR" RESULTS_DIR="$RESULTS_ROOT/dmg" \
    CKPT_PRE="$RESULTS_ROOT/loc/checkpoints/best.ckpt" \
    FINETUNE_CONFIG="$FINETUNE_CONFIG" \
    bash "$FINETUNE_DIR/train_damage.sh"
fi

CKPT_DMG="$RESULTS_ROOT/dmg/checkpoints/best.ckpt"
if [[ -f "$CKPT_DMG" ]]; then
  cp -f "$CKPT_DMG" "$WORKING/damage_best.ckpt"
  echo "Exported checkpoint -> $WORKING/damage_best.ckpt"
  ls -la "$WORKING/damage_best.ckpt"
else
  echo "WARN: No damage checkpoint at $CKPT_DMG"
fi

echo "Done."
