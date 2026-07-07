"""Patch michal2409/xView2 for DisasterIQ subset training (idempotent).

- pytorch_loader.py: read index.csv from XVIEW2_INDEX_CSV or repo-relative utils/index.csv
- utils/generate_idx.py: use CLI args instead of hardcoded /data/train paths

Run after cloning into ml/pytorch-xview2/:
  python ml/finetune/patch_pytorch_xview2.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"

LOADER = XVIEW2_ROOT / "data_loading" / "pytorch_loader.py"
GENERATE_IDX = XVIEW2_ROOT / "utils" / "generate_idx.py"

HARDCODED_LINE = '        data_frame = pd.read_csv("/workspace/xview2/utils/index.csv")'
LOADER_HELPER = '''
# DisasterIQ: configurable index.csv via _load_index_csv()
def _load_index_csv():
    index_csv = os.environ.get(
        "XVIEW2_INDEX_CSV",
        os.path.join(os.path.dirname(__file__), "..", "utils", "index.csv"),
    )
    return pd.read_csv(index_csv)
'''
LOADER_REPLACE = "        data_frame = _load_index_csv()"

MARKER = "# DisasterIQ: configurable index.csv via _load_index_csv()"


def patch_loader() -> None:
    if not LOADER.exists():
        raise SystemExit(f"Missing {LOADER} — clone michal2409/xView2 into ml/pytorch-xview2")
    text = LOADER.read_text(encoding="utf-8")
    if "_load_index_csv" in text:
        print(f"Already patched: {LOADER}")
        return
    if text.count(HARDCODED_LINE) != 2:
        raise SystemExit(
            f"Expected 2 occurrences of hardcoded index path in {LOADER}, "
            f"found {text.count(HARDCODED_LINE)}"
        )
    if "_load_index_csv" not in text:
        text = text.replace(
            "from data_loading.autoaugment import ImageNetPolicy\n",
            f"from data_loading.autoaugment import ImageNetPolicy\n{LOADER_HELPER}\n",
            1,
        )
    text = text.replace(HARDCODED_LINE, LOADER_REPLACE)
    LOADER.write_text(text, encoding="utf-8")
    print(f"Patched {LOADER}")


def patch_generate_idx() -> None:
    if not GENERATE_IDX.exists():
        print(f"Skip (no file): {GENERATE_IDX}")
        return
    text = GENERATE_IDX.read_text(encoding="utf-8")
    if "DisasterIQ: argparse" in text:
        print(f"Already patched: {GENERATE_IDX}")
        return
    new_header = '''"""Generate index.csv — patched for DisasterIQ (DisasterIQ: argparse)."""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "/data/train_subset")))
_parser.add_argument("--out", type=Path, default=Path(os.environ.get("XVIEW2_INDEX_CSV", _REPO / "utils" / "index.csv")))
_args, _ = _parser.parse_known_args()
PATH = str(_args.data_dir)
_out = _args.out
'''
    # Replace from first import through PATH assignment
    text = re.sub(
        r"^import glob.*?^PATH = .*$",
        new_header.strip(),
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    text = text.replace(
        'df.to_csv("/workspace/xview2/utils/index.csv", index=False)',
        "df.to_csv(_out, index=False)",
    )
    if 'exclude_idx = json.load(open("exclude.txt"' in text:
        text = text.replace(
            'exclude_idx = json.load(open("exclude.txt", "r"))',
            '_exclude = Path(__file__).resolve().parent / "exclude.txt"\n'
            'exclude_idx = json.load(open(_exclude, "r")) if _exclude.exists() else []',
        )
    GENERATE_IDX.write_text(text, encoding="utf-8")
    print(f"Patched {GENERATE_IDX}")


def main() -> None:
    patch_loader()
    patch_generate_idx()
    print("Done.")


if __name__ == "__main__":
    main()
