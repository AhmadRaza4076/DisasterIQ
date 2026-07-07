"""Emit shell export statements from ml/finetune/config_subset.yaml.

Usage (bash):
  eval "$(python ml/finetune/load_config.py localization)"
  eval "$(python ml/finetune/load_config.py damage)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    raise

CONFIG = Path(__file__).resolve().parent / "config_subset.yaml"

SECTION_KEYS = {
    "localization": {
        "EPOCHS": "epochs",
        "BATCH_SIZE": "batch_size",
        "ENCODER": "encoder",
        "RESULTS_DIR": "results_dir",
    },
    "damage": {
        "EPOCHS": "epochs",
        "BATCH_SIZE": "batch_size",
        "ENCODER": "encoder",
        "RESULTS_DIR": "results_dir",
        "CKPT_PRE": "ckpt_pre",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=["localization", "damage", "data"])
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.section == "data":
        data = cfg.get("data", {})
        print(f'export DATA_DIR="{data.get("train_dir", "/data/train_subset")}"')
        print(f'export TEST_DIR="{data.get("test_dir", "/data/test")}"')
        print(f'export RESULTS_ROOT="{data.get("results_root", "/results")}"')
        return

    section = cfg.get(args.section, {})
    for env_key, yaml_key in SECTION_KEYS[args.section].items():
        val = section.get(yaml_key)
        if val is not None:
            print(f'export {env_key}="{val}"')


if __name__ == "__main__":
    main()
