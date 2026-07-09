"""Deterministic zone scoring from xView2 damage masks (pixel values 0-4)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from app.schemas import (
    AnalysisResult,
    AnalysisSummary,
    BuildingCounts,
    DamageCounts,
    Zone,
)

logger = logging.getLogger(__name__)

# Overlay colors (RGBA) for frontend legend
OVERLAY_COLORS = {
    0: (0, 0, 0, 0),
    1: (34, 197, 94, 120),    # green - no damage
    2: (59, 130, 246, 140),   # blue - minor
    3: (249, 115, 22, 160),   # orange - major
    4: (239, 68, 68, 180),    # red - destroyed
}

WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0}

_CLASS_TO_FIELD = {
    1: "none",
    2: "minor",
    3: "major",
    4: "destroyed",
}


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def load_confidence(path: Path, mask_shape: tuple[int, ...]) -> np.ndarray:
    confidence = np.load(path)
    if confidence.shape != mask_shape:
        raise ValueError(
            f"Confidence shape {confidence.shape} does not match mask shape {mask_shape}"
        )
    return confidence


def counts_for_region(mask: np.ndarray) -> DamageCounts:
    building = mask > 0
    if not building.any():
        return DamageCounts()
    vals, cnts = np.unique(mask[building], return_counts=True)
    mapping = dict(zip(vals.tolist(), cnts.tolist()))
    return DamageCounts(
        none=int(mapping.get(1, 0)),
        minor=int(mapping.get(2, 0)),
        major=int(mapping.get(3, 0)),
        destroyed=int(mapping.get(4, 0)),
    )


_CONNECTIVITY = np.ones((3, 3), dtype=int)


def building_counts_for_region(mask: np.ndarray) -> BuildingCounts:
    """Count distinct connected components (buildings) per damage class."""
    counts = BuildingCounts()
    for cls, field in _CLASS_TO_FIELD.items():
        _, num = label(mask == cls, structure=_CONNECTIVITY)
        setattr(counts, field, int(num))
    return counts


def confidence_for_region(confidence: np.ndarray, mask_region: np.ndarray) -> float | None:
    """Mean predicted-class probability over building pixels in a zone."""
    building = mask_region > 0
    if not building.any():
        return None
    return round(float(confidence[building].mean()), 4)


def priority_score(counts: BuildingCounts) -> float:
    total_building = counts.none + counts.minor + counts.major + counts.destroyed
    if total_building == 0:
        return 0.0
    weighted = (
        counts.minor * WEIGHTS[2]
        + counts.major * WEIGHTS[3]
        + counts.destroyed * WEIGHTS[4]
    )
    return round((weighted / (total_building * WEIGHTS[4])) * 100, 2)


def _summary_from_buildings(all_buildings: BuildingCounts, all_pixels: DamageCounts) -> AnalysisSummary:
    total_buildings = (
        all_buildings.none + all_buildings.minor + all_buildings.major + all_buildings.destroyed
    )
    total_pixels = all_pixels.none + all_pixels.minor + all_pixels.major + all_pixels.destroyed
    return AnalysisSummary(
        total_building_pixels=total_pixels,
        total_buildings=total_buildings,
        destroyed_pct=round(all_buildings.destroyed / total_buildings * 100, 2) if total_buildings else 0.0,
        major_pct=round(all_buildings.major / total_buildings * 100, 2) if total_buildings else 0.0,
        minor_pct=round(all_buildings.minor / total_buildings * 100, 2) if total_buildings else 0.0,
    )


def score_mask(
    mask_path: Path,
    grid_rows: int = 4,
    grid_cols: int = 4,
    confidence_path: Path | None = None,
) -> AnalysisResult:
    mask = load_mask(mask_path)
    confidence: np.ndarray | None = None
    if confidence_path is not None and confidence_path.exists():
        try:
            confidence = load_confidence(confidence_path, mask.shape)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to load confidence from %s: %s", confidence_path, exc)
            confidence = None
    h, w = mask.shape
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)

    zones: list[Zone] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = h if row == grid_rows - 1 else (row + 1) * cell_h
            x1 = w if col == grid_cols - 1 else (col + 1) * cell_w
            region = mask[y0:y1, x0:x1]
            pixel_counts = counts_for_region(region)
            building_counts = building_counts_for_region(region)
            total_buildings = (
                building_counts.none
                + building_counts.minor
                + building_counts.major
                + building_counts.destroyed
            )
            if total_buildings == 0:
                continue
            zone_confidence = None
            if confidence is not None:
                zone_confidence = confidence_for_region(confidence[y0:y1, x0:x1], region)
            zones.append(
                Zone(
                    rank=0,
                    bbox=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    damage_counts=pixel_counts,
                    building_counts=building_counts,
                    priority_score=priority_score(building_counts),
                    confidence=zone_confidence,
                )
            )

    zones.sort(key=lambda z: z.priority_score, reverse=True)
    for i, zone in enumerate(zones, start=1):
        zone.rank = i

    all_pixels = counts_for_region(mask)
    all_buildings = building_counts_for_region(mask)
    summary = _summary_from_buildings(all_buildings, all_pixels)

    overlay = _build_overlay(mask)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return AnalysisResult(
        zones=zones,
        summary=summary,
        mask_path=mask_path.name,
        mask_base64=mask_b64,
        inference_mode="scoring",
    )


def _build_overlay(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in OVERLAY_COLORS.items():
        if cls == 0:
            continue
        rgba[mask == cls] = color
    return Image.fromarray(rgba, mode="RGBA")
