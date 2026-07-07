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
