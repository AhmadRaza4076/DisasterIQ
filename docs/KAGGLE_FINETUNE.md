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
4. **Settings** → **Accelerator** → **GPU** (T4 or P100)
5. Verify phone is verified (required for GPU)

## 3. Run the notebook

Recommended order:

1. Run cells 1–5 on **CPU** first (install, clone, stage data, patch, index) — saves GPU quota
2. Enable **GPU** in Settings if not already
3. Run cell 6 (GPU check) — must print `CUDA: True`
4. Run cell 7 — full training (`run_kaggle_pipeline.sh`)
5. Run cell 8 — confirms `damage_best.ckpt` in Output

**Session limits:** ~9–12 hours per run, ~30 GPU hours/week. Checkpoints are copied to `/kaggle/working/damage_best.ckpt` during training.

### Resume after disconnect

If localization finished but damage did not:

```bash
cd /kaggle/working/DisasterIQ
bash ml/finetune/run_kaggle_pipeline.sh --stage dmg
```

If you need to re-run only localization:

```bash
bash ml/finetune/run_kaggle_pipeline.sh --stage loc
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
| `ml/finetune/config_subset_kaggle.yaml` | 5 loc + 8 damage epochs, Kaggle paths |
| `ml/finetune/run_kaggle_pipeline.sh` | Full pipeline (auto-finds dataset, exports ckpt) |
| `ml/finetune/config_subset.yaml` | Full AMD run (10+20 epochs) |

To train longer on Kaggle, edit `config_subset_kaggle.yaml` epochs before running (watch session time).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GPU option greyed out | Verify phone in Kaggle account settings |
| `CUDA: False` | Settings → GPU → save; restart notebook |
| `Could not find train_subset` | Check dataset slug; ensure zip has `images/` at root |
| OOM during damage stage | Lower `damage.batch_size` to 2 in `config_subset_kaggle.yaml` |
| `Missing ml/pytorch-xview2` | Re-run clone cell; repo is gitignored locally but cloned on Kaggle |
| Weekly GPU quota exhausted | Wait for reset (~Sunday UTC) or use Colab/RunPod |

## Honest judge narrative

If AMD access is delayed:

> "We fine-tuned the PyTorch xView2 damage head on our curated train subset (~1449 pairs) using NVIDIA CUDA on Kaggle while MI300 access was pending. The same checkpoint integrates into our production inference path with per-zone confidence."

After AMD access, you can re-run `ml/finetune/run_amd_pipeline.sh` for the ROCm story.

## See also

- [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md) — MI300 path when credits arrive
- [ml/finetune/README.md](../ml/finetune/README.md) — training overview
- [DATA.md](DATA.md) — how `train_subset` was built
