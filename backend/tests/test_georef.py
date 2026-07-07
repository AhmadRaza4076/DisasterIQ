"""Tests for xBD affine georeferencing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.schemas import BuildingCounts, DamageCounts, Zone
from app.services.georef import (
    enrich_zones_with_geo,
    fit_affine_transform,
    load_correspondences,
    parse_wkt_centroid,
    pixel_to_geo,
)

LABEL_PATH = settings.demo_data_dir / "labels" / "midwest-flooding_00000000_post_disaster.json"


def test_parse_wkt_centroid_square() -> None:
    wkt = "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"
    lng, lat = parse_wkt_centroid(wkt)
    assert lng == pytest.approx(1.0)
    assert lat == pytest.approx(1.0)


def test_load_correspondences_demo_pair() -> None:
    if not LABEL_PATH.exists():
        pytest.skip("Demo label not present")
    pixels, geo = load_correspondences(LABEL_PATH)
    assert pixels.shape[0] >= 3
    assert geo.shape == (pixels.shape[0], 2)


def test_affine_round_trip_error_small() -> None:
    if not LABEL_PATH.exists():
        pytest.skip("Demo label not present")
    pixels, geo = load_correspondences(LABEL_PATH)
    lng_c, lat_c = fit_affine_transform(pixels, geo)
    errors: list[float] = []
    for (px, py), (lng, lat) in zip(pixels, geo, strict=True):
        pred_lng, pred_lat = pixel_to_geo(px, py, lng_c, lat_c)
        errors.append(abs(pred_lng - lng) + abs(pred_lat - lat))
    assert max(errors) < 0.01


def test_enrich_zones_plausible_arkansas_coords() -> None:
    if not LABEL_PATH.exists():
        pytest.skip("Demo label not present")
    zones = [
        Zone(
            rank=1,
            bbox=[100, 100, 200, 200],
            damage_counts=DamageCounts(destroyed=1),
            building_counts=BuildingCounts(destroyed=1),
            priority_score=50.0,
        )
    ]
    ok = enrich_zones_with_geo(zones, LABEL_PATH)
    assert ok is True
    assert zones[0].centroid_lat is not None
    assert zones[0].centroid_lng is not None
    # Midwest flooding demo is near Little Rock, Arkansas
    assert 33.0 < zones[0].centroid_lat < 36.0
    assert -94.0 < zones[0].centroid_lng < -90.0
