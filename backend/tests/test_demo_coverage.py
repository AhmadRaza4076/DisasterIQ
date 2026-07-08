"""Regression tests for curated demo dataset damage-class coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings


def _class_pixel_counts(target_path: Path) -> dict[int, int]:
    arr = np.array(Image.open(target_path).convert("L"))
    building = arr > 0
    if not building.any():
        return {}
    vals, cnts = np.unique(arr[building], return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))


def test_demo_manifest_lists_ten_pairs() -> None:
    manifest = json.loads((settings.demo_data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["pairs"]) == 10


def test_demo_targets_include_major_damage_class() -> None:
    targets_dir = settings.demo_data_dir / "targets"
    major_pixels = 0
    for target in sorted(targets_dir.glob("*_post_disaster_target.png")):
        counts = _class_pixel_counts(target)
        major_pixels += counts.get(3, 0)
    assert major_pixels > 0, "Demo set must include class-3 (major) pixels for full legend coverage"


def test_mexico_earthquake_00000076_has_major_and_destroyed() -> None:
    target = (
        settings.demo_data_dir / "targets" / "mexico-earthquake_00000076_post_disaster_target.png"
    )
    assert target.exists(), "Expected demo pair mexico-earthquake_00000076"
    counts = _class_pixel_counts(target)
    assert counts.get(3, 0) > 100
    assert counts.get(4, 0) > 0
