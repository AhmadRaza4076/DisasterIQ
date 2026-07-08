"""Patch michal2409/xView2 for DisasterIQ subset training (idempotent).

- pytorch_loader.py: read index.csv from XVIEW2_INDEX_CSV or repo-relative utils/index.csv

Index generation uses scripts/generate_subset_index.py (not upstream generate_idx.py).

Run after cloning into ml/pytorch-xview2/:
  python ml/finetune/patch_pytorch_xview2.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
XVIEW2_ROOT = REPO_ROOT / "ml" / "pytorch-xview2"

LOADER = XVIEW2_ROOT / "data_loading" / "pytorch_loader.py"

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


def main() -> None:
    patch_loader()
    # generate_idx.py is intentionally left unpatched/unused — superseded by
    # scripts/generate_subset_index.py, which generates index.csv scoped to
    # our actual train_subset instead of the full original xView2 dataset.
    print("Done.")


if __name__ == "__main__":
    main()
