"""Unit tests for deterministic zone scoring."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.schemas import BuildingCounts, DamageCounts
from app.services.scoring import (
    building_counts_for_region,
    confidence_for_region,
    counts_for_region,
    priority_score,
    score_mask,
)


def test_priority_score_ignores_undamaged_pixels() -> None:
    undamaged_only = BuildingCounts(none=1000, minor=0, major=0, destroyed=0)
    mixed = BuildingCounts(none=900, minor=0, major=0, destroyed=100)
    assert priority_score(undamaged_only) == 0.0
    assert priority_score(mixed) == 10.0


def test_priority_score_weighted_damaged_classes() -> None:
    counts = BuildingCounts(none=0, minor=10, major=10, destroyed=10)
    # (10*2 + 10*3.5 + 10*5) / (30*5) * 100 = 70
    assert priority_score(counts) == pytest.approx(70.0, rel=1e-3)


def test_counts_for_region_empty_mask() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    counts = counts_for_region(mask)
    assert counts.none == counts.minor == counts.major == counts.destroyed == 0


def test_building_counts_two_separate_blobs() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:7, 2:7] = 4
    mask[12:17, 12:17] = 4
    counts = building_counts_for_region(mask)
    assert counts.destroyed == 2


def test_score_mask_ranks_destroyed_zone_highest() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    # Top-left cell: mostly destroyed
    mask[0:10, 0:10] = 4
    # Bottom-right cell: minor only
    mask[30:40, 30:40] = 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        Image.fromarray(mask, mode="L").save(path)
        result = score_mask(path, grid_rows=4, grid_cols=4)

    assert len(result.zones) >= 2
    assert result.zones[0].priority_score >= result.zones[1].priority_score
    assert result.zones[0].building_counts.destroyed > 0


def test_score_mask_handles_empty_building_mask() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        Image.fromarray(mask, mode="L").save(path)
        result = score_mask(path, grid_rows=4, grid_cols=4)

    assert result.zones == []
    assert result.summary.total_building_pixels == 0
    assert result.summary.total_buildings == 0


def test_confidence_for_region_mean_over_building_pixels() -> None:
    mask_region = np.zeros((4, 4), dtype=np.uint8)
    mask_region[1:3, 1:3] = 4
    confidence_region = np.full((4, 4), 0.1, dtype=np.float32)
    confidence_region[1:3, 1:3] = 0.9
    assert confidence_for_region(confidence_region, mask_region) == pytest.approx(0.9)


def test_score_mask_populates_zone_confidence_from_npy() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 4
    confidence = np.full((8, 8), 0.2, dtype=np.float32)
    confidence[0:4, 0:4] = 0.8

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        conf_path = Path(tmp) / "mask_confidence.npy"
        Image.fromarray(mask, mode="L").save(path)
        np.save(conf_path, confidence)
        result = score_mask(path, grid_rows=2, grid_cols=2, confidence_path=conf_path)

    assert len(result.zones) == 1
    assert result.zones[0].confidence == pytest.approx(0.8)


def test_score_mask_mask_path_is_basename_only() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "job_mask.png"
        path.parent.mkdir(parents=True)
        Image.fromarray(mask, mode="L").save(path)
        result = score_mask(path, grid_rows=2, grid_cols=2)

    assert result.mask_path == "job_mask.png"
    assert "/" not in (result.mask_path or "")
    assert "\\" not in (result.mask_path or "")


def test_building_counts_diagonal_pixels_one_component_with_8_connectivity() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 4
    mask[2, 2] = 4
    counts = building_counts_for_region(mask)
    assert counts.destroyed == 1
