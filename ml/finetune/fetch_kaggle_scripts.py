#!/usr/bin/env python3
"""Fetch latest ml/finetune scripts from GitHub main (no git pull needed).

Use on Kaggle when local edits block `git pull` but you need updated training scripts.

  python ml/finetune/fetch_kaggle_scripts.py
  python ml/finetune/kaggle_train.py --stage dmg --skip-deps --skip-smoke-test
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO_ROOT / "ml" / "finetune"
RAW_BASE = "https://raw.githubusercontent.com/AhmadRaza4076/DisasterIQ/main/ml/finetune"

FILES = [
    "kaggle_train.py",
    "kaggle_dmg_resume.py",
    "patch_pytorch_xview2.py",
    "smoke_damage_step.py",
    "fetch_kaggle_scripts.py",
    "load_config.py",
    "config_subset_kaggle.yaml",
    "requirements_kaggle.txt",
    "overrides/main.py",
    "overrides/model/loss.py",
]


def fetch_file(name: str) -> None:
    url = f"{RAW_BASE}/{name}"
    dest = FINETUNE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to fetch {url}: {exc}") from exc
    dest.write_bytes(data)
    print(f"Fetched {name}")


def main() -> None:
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch_file(name)
    print("Done — run: python ml/finetune/kaggle_train.py --stage dmg --skip-deps")


if __name__ == "__main__":
    main()
