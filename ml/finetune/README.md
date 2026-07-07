# Fine-tuning on AMD GPU — PyTorch path

**Status:** Data prep complete on laptop. Training awaits AMD GPU access.

## What's ready (CPU, done)

| Step | Command | Result |
|------|---------|--------|
| Verify Kaggle archive | `.\scripts\verify-kaggle-archive.ps1` | `D:\archive.zip` OK |
| Extract + flatten train | `.\scripts\extract-kaggle-archive.ps1` | `data/train/` flat layout |
| Generate target masks | `python scripts\generate_train_targets.py` | 5598 PNG masks |
| Build hackathon subset | `python scripts\prepare_train_subset.py ...` | ~1449 pairs in `data/train_subset/` |
| Vendor PyTorch repo | `ml/pytorch-xview2/` | `michal2409/xView2` clone |

Counts (verified):

```
data/train/images   : 5598
data/train/targets  : 5598
data/train_subset/* : 2898 per folder (~1449 pairs)
```

## AMD GPU verification

```bash
rocm-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `False`, install ROCm PyTorch per AMD docs before training.

## Training pipeline

### Sync data to GPU instance

```bash
bash scripts/rsync_to_amd.sh root@YOUR_DROPLET_IP
```

### Run full pipeline

```bash
cd DisasterIQ
bash ml/finetune/run_amd_pipeline.sh
```

Stages:
1. **Localization** (`--type pre`) — 10 epochs, ResNet50
2. **Damage** (`--type post`, siamese) — 20 epochs, loads loc checkpoint
3. **Eval** on `data/test/`

Config: `ml/finetune/config_subset.yaml`

### ROCm Docker alternative

```bash
cd ml/pytorch-xview2
docker build -f Dockerfile.rocm -t darknem-xview2-train .
docker run --rm --device=/dev/kfd --device=/dev/dri \
  -v /data/train_subset:/data/train_subset \
  -v /data/test:/data/test \
  -v /results:/results \
  darknem-xview2-train \
  bash -c "cd /workspace/xview2 && bash ../finetune/run_amd_pipeline.sh"
```

**Time box:** 4–6 hours. Stop and ship TF baseline if not converging.

## Integrating checkpoint

After training, copy best damage checkpoint:

```
/results/dmg/checkpoints/best.ckpt  →  ml/checkpoints/damage_best.ckpt
```

`.env`:

```
INFERENCE_MODE=pytorch
PYTORCH_CHECKPOINT_PATH=ml/checkpoints/damage_best.ckpt
```

Restart backend. Compare IoU:

```powershell
.\backend\.venv\Scripts\python.exe scripts\compare_models.py --modes docker pytorch
```

## Do not attempt

- TF 1.15 / Keras 2.2.5 training on ROCm — no practical path in hackathon window
