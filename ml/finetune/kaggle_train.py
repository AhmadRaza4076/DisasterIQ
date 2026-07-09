#!/usr/bin/env python3
"""Unified Kaggle fine-tune bootstrap + staged training for DisasterIQ xView2.

Usage (from repo root on Kaggle):
  python ml/finetune/kaggle_train.py --stage prep
  python ml/finetune/kaggle_train.py --stage loc
  python ml/finetune/kaggle_train.py --stage dmg    # resume after loc done
  python ml/finetune/kaggle_train.py --stage all    # full pipeline
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO_ROOT / "ml" / "finetune"
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"
WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
DEFAULT_CONFIG = FINETUNE_DIR / "config_subset_kaggle.yaml"
DEFAULT_DATA = WORKING / "data" / "train_subset"
INDEX_CSV = XVIEW2_ROOT / "utils" / "index.csv"


def pip_install(*packages: str, force: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if force:
        cmd.append("--force-reinstall")
    cmd.extend(packages)
    subprocess.run(cmd, check=True)


def purge_lightning_modules() -> None:
    for key in list(sys.modules):
        if key.startswith(("pytorch_lightning", "lightning", "torchmetrics")):
            del sys.modules[key]


def ensure_pytorch_lightning_19() -> None:
    purge_lightning_modules()
    try:
        import pytorch_lightning as pl  # noqa: F401

        if pl.__version__.startswith("1.9"):
            print(f"pytorch-lightning {pl.__version__} OK")
            return
        found = pl.__version__
    except Exception:
        found = None

    print(f"Installing pytorch-lightning 1.9.5 (found {found})...")
    pip_install("pytorch-lightning==1.9.5", "torchmetrics", force=True)
    purge_lightning_modules()
    import pytorch_lightning as pl

    if not pl.__version__.startswith("1.9"):
        raise RuntimeError(
            f"Still on pytorch-lightning {pl.__version__}. "
            "Restart kernel, run install cell, then re-run this script."
        )
    print(f"pytorch-lightning {pl.__version__} OK")


def ensure_runtime_deps() -> None:
    req_file = FINETUNE_DIR / "requirements_kaggle.txt"
    if req_file.is_file():
        pip_install("-r", str(req_file))
    else:
        pip_install(
            "pytorch-lightning==1.9.5",
            "torchmetrics",
            "torch-optimizer",
            "timm",
            "segmentation-models-pytorch",
            "albumentations",
            "monai>=1.3,<2",
            "shapely",
            "fire",
            "pyyaml",
        )

    try:
        from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity  # noqa: F401
    except Exception:
        print("Installing NVIDIA DLLogger from GitHub...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "dllogger"],
            check=False,
        )
        pip_install("git+https://github.com/NVIDIA/dllogger.git#egg=dllogger")


def run_patches() -> None:
    patch_script = FINETUNE_DIR / "patch_pytorch_xview2.py"
    subprocess.run([sys.executable, str(patch_script)], check=True, cwd=str(REPO_ROOT))


def ensure_data_layout(data_root: Path) -> None:
    if not (data_root / "images").is_dir():
        raise FileNotFoundError(f"Missing images/ under {data_root}")
    if (data_root / "train" / "images").exists():
        return
    for split in ("train", "test"):
        split_dir = data_root / split
        split_dir.mkdir(exist_ok=True)
        for sub in ("images", "targets", "labels"):
            src = data_root / sub
            dst = split_dir / sub
            if src.is_dir() and not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=True)
    print(f"xView2 data layout OK under {data_root}")


def ensure_results_dirs(results_root: Path) -> None:
    for sub in ("loc", "dmg"):
        d = results_root / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"results dirs OK under {results_root}")


def resolve_checkpoint(preferred: Path) -> Path:
    """Resolve best.ckpt, last.ckpt, or newest .ckpt in the same directory."""
    if preferred.is_file():
        return preferred
    ckpt_dir = preferred.parent
    for name in ("best.ckpt", "last.ckpt"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")


def export_damage_checkpoint(results_root: Path, export_path: Path) -> None:
    dmg_dir = results_root / "dmg" / "checkpoints"
    if not dmg_dir.is_dir():
        print(f"WARN: no damage checkpoints under {dmg_dir}")
        return
    for candidate in [
        dmg_dir / "best.ckpt",
        dmg_dir / "last.ckpt",
        *sorted(dmg_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]:
        if candidate.is_file():
            shutil.copy2(candidate, export_path)
            print(f"Exported {export_path} from {candidate.name}")
            return
    print(f"WARN: no .ckpt files in {dmg_dir}")


def run_shell(script: str, **env: str) -> None:
    merged = os.environ.copy()
    merged.update({k: str(v) for k, v in env.items()})
    subprocess.run(["bash", script], check=True, cwd=str(REPO_ROOT), env=merged)


def prep_stage(data_dir: Path, config_path: Path) -> None:
    if not XVIEW2_ROOT.is_dir():
        raise FileNotFoundError(f"Missing {XVIEW2_ROOT} — clone xView2 first")

    ensure_data_layout(data_dir)
    run_patches()

    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX_CSV)
    if INDEX_CSV.is_file() and INDEX_CSV.stat().st_size > 10:
        print(f"index.csv OK ({sum(1 for _ in INDEX_CSV.open())} lines)")
    else:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_subset_index.py"),
                "--data-dir",
                str(data_dir),
                "--out",
                str(INDEX_CSV),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_pytorch_dataset.py"),
            "--data-dir",
            str(data_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def train_loc(data_dir: Path, results_root: Path, config_path: Path) -> None:
    ensure_results_dirs(results_root)
    run_shell(
        str(FINETUNE_DIR / "train_localization.sh"),
        FINETUNE_CONFIG=str(config_path),
        DATA_DIR=str(data_dir),
        RESULTS_DIR=str(results_root / "loc"),
        XVIEW2_INDEX_CSV=str(INDEX_CSV),
    )


def train_dmg(
    data_dir: Path,
    results_root: Path,
    config_path: Path,
    ckpt_pre: Path | None = None,
) -> None:
    ensure_results_dirs(results_root)
    loc_ckpt_dir = results_root / "loc" / "checkpoints"
    preferred = ckpt_pre or (loc_ckpt_dir / "best.ckpt")
    resolved = resolve_checkpoint(preferred)
    print(f"Using localization checkpoint: {resolved}")

    run_shell(
        str(FINETUNE_DIR / "train_damage.sh"),
        FINETUNE_CONFIG=str(config_path),
        DATA_DIR=str(data_dir),
        RESULTS_DIR=str(results_root / "dmg"),
        CKPT_PRE=str(resolved),
        XVIEW2_INDEX_CSV=str(INDEX_CSV),
    )


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Kaggle: Settings → Accelerator → GPU T4 x2, save, re-run."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DisasterIQ Kaggle fine-tune")
    parser.add_argument(
        "--stage",
        choices=["prep", "loc", "dmg", "all"],
        default="all",
        help="prep=patch+index; loc=localization; dmg=damage only; all=loc+dmg",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=WORKING / "results")
    parser.add_argument("--export", type=Path, default=WORKING / "damage_best.ckpt")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install checks")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.environ.setdefault("FINETUNE_CONFIG", str(args.config.resolve()))

    if not args.skip_deps:
        ensure_runtime_deps()
    ensure_pytorch_lightning_19()

    if args.stage in ("prep", "all", "loc", "dmg"):
        prep_stage(args.data_dir, args.config)

    if args.stage == "prep":
        print("Prep complete.")
        return

    require_gpu()

    if args.stage in ("loc", "all"):
        train_loc(args.data_dir, args.results_root, args.config)

    if args.stage in ("dmg", "all"):
        train_dmg(args.data_dir, args.results_root, args.config)

    if args.stage in ("dmg", "all"):
        export_damage_checkpoint(args.results_root, args.export)

    print("Done.")


if __name__ == "__main__":
    main()
