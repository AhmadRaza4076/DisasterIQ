# Place fine-tuned PyTorch checkpoints here (gitignored)

## Localization (from Kaggle tier3 run, ~epoch 7, F1 ~84+)
- `loc_best.ckpt` — use this as `--ckpt_pre` / upload for damage stage
- `loc_last.ckpt`, `loc_step.ckpt` — backups

## Damage (for app inference)
After Kaggle damage training, copy:
`/results/dmg/checkpoints/best.ckpt` → `damage_best.ckpt`

Then set in `.env`:
```
INFERENCE_MODE=pytorch
PYTORCH_CHECKPOINT_PATH=ml/checkpoints/damage_best.ckpt
```

Note: existing `damage_best.ckpt` is the older weak run — replace it after the new damage stage finishes.
