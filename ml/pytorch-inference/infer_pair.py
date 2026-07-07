"""Run PyTorch xView2 damage inference on a single pre/post pair.

Requires a trained post-disaster checkpoint (siamese, focal+dice).
Creates a temporary holdout layout and shells out to michal2409/xView2 main.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"
MAIN_PY = XVIEW2_ROOT / "main.py"


def _dummy_mask(path: Path) -> None:
    Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8)).save(path)


def infer_pair(
    pre_path: Path,
    post_path: Path,
    checkpoint: Path,
    out_mask: Path,
    work_dir: Path | None = None,
) -> Path:
    if not MAIN_PY.exists():
        raise FileNotFoundError(f"Missing {MAIN_PY} — clone michal2409/xView2 into ml/pytorch-xview2")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    job_id = uuid.uuid4().hex[:8]
    work = work_dir or (REPO_ROOT / "backend" / "app" / "outputs" / f"pytorch_{job_id}")
    data_root = work / "data"
    holdout = data_root / "holdout"
    img_dir = holdout / "images"
    tgt_dir = holdout / "targets"
    results = work / "results"
    for d in (img_dir, tgt_dir, results):
        d.mkdir(parents=True, exist_ok=True)

    stem = post_path.stem.replace("_post_disaster", "")
    pre_name = f"{stem}_pre_disaster.png"
    post_name = f"{stem}_post_disaster.png"
    shutil.copy2(pre_path, img_dir / pre_name)
    shutil.copy2(post_path, img_dir / post_name)
    _dummy_mask(tgt_dir / pre_name)
    _dummy_mask(tgt_dir / post_name)

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--exec_mode",
        "eval",
        "--type",
        "post",
        "--dmg_model",
        "siamese",
        "--encoder",
        "resnet50",
        "--loss_str",
        "focal+dice",
        "--data",
        str(data_root),
        "--results",
        str(results),
        "--ckpt",
        str(checkpoint),
        "--gpus",
        "0",
        "--val_batch_size",
        "1",
        "--num_workers",
        "0",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(XVIEW2_ROOT),
    }
    index_src = XVIEW2_ROOT / "utils" / "index.csv"
    if index_src.exists():
        env["XVIEW2_INDEX_CSV"] = str(index_src.resolve())

    subprocess.run(cmd, check=True, cwd=str(XVIEW2_ROOT), env=env)

    probs_dir = results / "probs"
    npy_files = sorted(probs_dir.glob("test_damage_*.npy"))
    if not npy_files:
        raise RuntimeError(f"No damage predictions in {probs_dir}")

    arr = np.load(npy_files[0])
    if arr.ndim == 3:
        mask = np.argmax(arr, axis=0).astype(np.uint8)
    else:
        mask = np.round(arr).astype(np.uint8)
    mask = np.clip(mask, 0, 4)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_mask)
    return out_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = infer_pair(args.pre, args.post, args.checkpoint, args.out)
    print(f"Wrote mask: {out}")


if __name__ == "__main__":
    main()
