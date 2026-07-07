"""Affine georeferencing from xBD label correspondence points."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from app.schemas import Zone

logger = logging.getLogger(__name__)

MIN_CORRESPONDENCES = 3
WARN_CORRESPONDENCES = 10

_WKT_RING_RE = re.compile(r"POLYGON\s*\(\(([^)]+)\)\)", re.IGNORECASE)


def parse_wkt_centroid(wkt: str) -> tuple[float, float]:
    """Return (lng, lat) centroid of the first ring in a WKT POLYGON."""
    match = _WKT_RING_RE.search(wkt)
    if not match:
        raise ValueError(f"Unsupported WKT: {wkt[:80]}")
    coords: list[tuple[float, float]] = []
    for pair in match.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        lng, lat = float(parts[0]), float(parts[1])
        coords.append((lng, lat))
    if len(coords) < 3:
        raise ValueError("Polygon needs at least 3 vertices")
    # Shoelace centroid (lng=x, lat=y)
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        lngs = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return sum(lngs) / len(lngs), sum(lats) / len(lats)
    cx /= 6.0 * area
    cy /= 6.0 * area
    return cx, cy


def _feature_centroid(feature: dict) -> tuple[float, float]:
    return parse_wkt_centroid(feature["wkt"])


def load_correspondences(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load matched pixel (x,y) and geo (lng,lat) centroids from xBD label JSON."""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    xy_feats = {f["properties"]["uid"]: f for f in data["features"]["xy"]}
    geo_feats = {f["properties"]["uid"]: f for f in data["features"]["lng_lat"]}
    common = sorted(set(xy_feats) & set(geo_feats))
    if not common:
        raise ValueError("No matching building uids between xy and lng_lat")
    pixels: list[tuple[float, float]] = []
    geo: list[tuple[float, float]] = []
    for uid in common:
        px, py = _feature_centroid(xy_feats[uid])
        lng, lat = _feature_centroid(geo_feats[uid])
        pixels.append((px, py))
        geo.append((lng, lat))
    return np.array(pixels, dtype=np.float64), np.array(geo, dtype=np.float64)


def fit_affine_transform(
    pixels: np.ndarray, geo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit separate affine models: lng = A @ [x,y,1], lat = B @ [x,y,1]."""
    ones = np.ones((pixels.shape[0], 1), dtype=np.float64)
    design = np.hstack([pixels, ones])
    lng_coeffs, _, _, _ = np.linalg.lstsq(design, geo[:, 0], rcond=None)
    lat_coeffs, _, _, _ = np.linalg.lstsq(design, geo[:, 1], rcond=None)
    return lng_coeffs, lat_coeffs


def pixel_to_geo(
    x: float, y: float, lng_coeffs: np.ndarray, lat_coeffs: np.ndarray
) -> tuple[float, float]:
    vec = np.array([x, y, 1.0], dtype=np.float64)
    lng = float(vec @ lng_coeffs)
    lat = float(vec @ lat_coeffs)
    return lng, lat


def enrich_zones_with_geo(zones: list[Zone], label_path: Path) -> bool:
    """Add centroid_lat/lng to zones. Returns True if georef succeeded."""
    if not label_path.exists():
        return False
    try:
        pixels, geo = load_correspondences(label_path)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Georef failed loading %s: %s", label_path, exc)
        return False

    if len(pixels) < MIN_CORRESPONDENCES:
        logger.warning("Georef skipped: only %d correspondence points", len(pixels))
        return False
    if len(pixels) < WARN_CORRESPONDENCES:
        logger.warning(
            "Georef using sparse correspondences (%d points) for %s",
            len(pixels),
            label_path.name,
        )

    lng_coeffs, lat_coeffs = fit_affine_transform(pixels, geo)

    for zone in zones:
        x, y, w, h = zone.bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        lng, lat = pixel_to_geo(cx, cy, lng_coeffs, lat_coeffs)
        zone.centroid_lng = round(lng, 6)
        zone.centroid_lat = round(lat, 6)

    return True
