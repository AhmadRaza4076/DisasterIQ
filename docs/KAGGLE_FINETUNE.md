# Kaggle GPU fine-tuning — DisasterIQ

Fine-tune the PyTorch xView2 damage model on **Kaggle's free NVIDIA GPU** (T4/P100) using your pre-built `data/train_subset/`. No AMD GPU required.

**Time box:** ~3–6 hours GPU for reduced epochs (5 loc + 8 damage). Full AMD config is 10+20 epochs.

## What you need from me (agent) vs you

| Step | Who |
|------|-----|
| Zip `train_subset`, upload Kaggle dataset | **You** |
| Create notebook, enable GPU, Run All | **You** |
| Download `damage_best.ckpt` | **You** |
| Notebook + scripts in repo | Done (this doc + `notebooks/kaggle_finetune.ipynb`) |

## 1. Zip and upload dataset

On your laptop (PowerShell):

```powershell
cd D:\AMD
.\scripts\zip_train_subset.ps1
```

This creates `D:\AMD\disasteriq-train-subset.zip` with `images/`, `labels/`, `targets/` at the zip root.

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Upload `disasteriq-train-subset.zip`
3. **Title:** `disasteriq-train-subset` (slug must match notebook default, or edit `DATASET_SLUG` in cell 2)
4. Visibility: **Private**
5. Wait until processing completes

## 2. Create Kaggle notebook

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **File → Import Notebook** → upload `notebooks/kaggle_finetune.ipynb` from this repo  
   **Or** copy the notebook from GitHub after you push.
3. **Input** (right sidebar) → **Add Input** → add dataset `disasteriq-train-subset`
4. **Settings** → **Accelerator** → **GPU T4 x2** (recommended). P100 works only after cell 2 installs cu118 PyTorch (default Kaggle torch does not support P100 sm_60).
5. Verify phone is verified (required for GPU)

## 3. Run the notebook

Recommended order:

1. Run cells 1–4 on **CPU** first (install, clone, prep/index) — saves GPU quota
2. Enable **GPU** in Settings if not already
3. Run cell 5 (GPU check) — must print `CUDA: True`
4. Run cell 6 — full training via `kaggle_train.py --stage all`
5. Run cell 7 — confirms `damage_best.ckpt` in Output

**Session limits:** ~9–12 hours per run, ~30 GPU hours/week. Checkpoints are copied to `/kaggle/working/damage_best.ckpt` during training.

### Resume after disconnect

If **localization finished** but **damage failed** (most common after long runs):

```bash
cd /kaggle/working/DisasterIQ
python ml/finetune/kaggle_train.py --stage dmg --skip-deps --skip-smoke-test
```

**Do not use `git pull` on Kaggle** — prior runs edit scripts locally and pull will abort. Instead fetch latest scripts:

```python
import urllib.request
from pathlib import Path
RAW = "https://raw.githubusercontent.com/AhmadRaza4076/DisasterIQ/main/ml/finetune"
fin = Path("/kaggle/working/DisasterIQ/ml/finetune")
for name in ["kaggle_train.py", "patch_pytorch_xview2.py", "load_config.py", "config_subset_kaggle.yaml"]:
    (fin / name).write_bytes(urllib.request.urlopen(f"{RAW}/{name}", timeout=60).read())
!python ml/finetune/kaggle_train.py --stage dmg --skip-deps --skip-smoke-test
```

Or run **notebook cell 6b** (same commands). Training invokes `main.py` directly so Python tracebacks appear in the notebook output (not hidden behind `train_damage.sh`).

If you need to re-run only localization:

```bash
python ml/finetune/kaggle_train.py --stage loc --skip-deps
```

Legacy shell pipeline (still supported):

```bash
bash ml/finetune/run_kaggle_pipeline.sh --stage dmg
```

## 4. Download checkpoint

1. After successful run: right sidebar → **Output**
2. Download **`damage_best.ckpt`**
3. On laptop:

```powershell
mkdir D:\AMD\ml\checkpoints -Force
copy D:\Downloads\damage_best.ckpt D:\AMD\ml\checkpoints\damage_best.ckpt
```

4. In `.env`:

```
INFERENCE_MODE=pytorch
PYTORCH_CHECKPOINT_PATH=ml/checkpoints/damage_best.ckpt
```

5. Restart backend and compare:

```powershell
.\scripts\start-backend.ps1
.\backend\.venv\Scripts\python.exe scripts\compare_models.py --modes docker pytorch
```

## 5. Config reference

| File | Purpose |
|------|---------|
| `ml/finetune/config_subset_kaggle.yaml` | 5 loc + 8 damage epochs, Kaggle paths, `num_workers: 4` |
| `ml/finetune/kaggle_train.py` | Unified bootstrap + staged training (recommended) |
| `ml/finetune/requirements_kaggle.txt` | Pinned deps for xView2 on modern Kaggle |
| `ml/finetune/patch_pytorch_xview2.py` | All xView2 compatibility patches (idempotent) |
| `ml/finetune/run_kaggle_pipeline.sh` | Legacy full pipeline (auto-finds dataset, exports ckpt) |
| `ml/finetune/config_subset.yaml` | Full AMD run (10+20 epochs) |

To train longer on Kaggle, edit `config_subset_kaggle.yaml` epochs before running (watch session time).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GPU option greyed out | Verify phone in Kaggle account settings |
| `CUDA: False` | Settings → GPU → save; restart notebook |
| P100 `sm_60 not compatible` warnings | Use **GPU T4 x2**, or re-run cell 2 (`cu118` torch) then restart session |
| `Could not find train_subset` | Check dataset slug; ensure zip has `images/` at root |
| OOM during damage stage | Lower `damage.batch_size` to 2 in `config_subset_kaggle.yaml` |
| `Missing ml/pytorch-xview2` | Re-run clone cell (cell 3) |
| `No module named 'apex'` | Run `python ml/finetune/patch_pytorch_xview2.py` |
| `No module named 'dllogger'` or wrong dllogger | `pip install git+https://github.com/NVIDIA/dllogger.git#egg=dllogger` |
| `pytorch_lightning.metrics` not found | Pin PL 1.9.5: `pip install pytorch-lightning==1.9.5 torchmetrics` |
| PL 2.x / `validation_epoch_end` errors | Same — must use **pytorch-lightning==1.9.5** |
| `AssertionError` in `load_data` (0 images) | Run patches + data layout: `kaggle_train.py --stage prep` or symlinks under `train_subset/train/` |
| `FileNotFoundError` for `logs.json` | Results dirs created automatically by `kaggle_train.py` |
| `Missing localization checkpoint: best.ckpt` | Use `kaggle_train.py --stage dmg` — auto-resolves `last.ckpt` |
| `ground truth has different shape` (64447×1 vs 64447×4) | Fetch `overrides/model/loss.py` + re-run patches (uses pure PyTorch for damage loss) |
| `subprocess.CalledProcessError` on damage with no traceback | Use `kaggle_train.py` (not inline `main.py`); smoke test runs first and prints real errors |
| `torch.load` / `weights_only` error | Fetch `overrides/main.py` + re-run patches |
| DataLoader worker warning | `num_workers: 4` in config (default) |

## Honest judge narrative

If AMD access is delayed:

> "We fine-tuned the PyTorch xView2 damage head on our curated train subset (~1449 pairs) using NVIDIA CUDA on Kaggle while MI300 access was pending. The same checkpoint integrates into our production inference path with per-zone confidence."

After AMD access, you can re-run `ml/finetune/run_amd_pipeline.sh` for the ROCm story.

## See also

- [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md) — MI300 path when credits arrive
- [ml/finetune/README.md](../ml/finetune/README.md) — training overview
- [DATA.md](DATA.md) — how `train_subset` was built
