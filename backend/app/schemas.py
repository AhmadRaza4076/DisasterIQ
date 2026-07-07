from typing import Any

from pydantic import BaseModel, Field


class DamageCounts(BaseModel):
    """Pixel counts per damage class (internal / debug)."""

    none: int = 0
    minor: int = 0
    major: int = 0
    destroyed: int = 0


class BuildingCounts(BaseModel):
    """Connected-component building instance counts per damage class."""

    none: int = 0
    minor: int = 0
    major: int = 0
    destroyed: int = 0


class Zone(BaseModel):
    rank: int
    bbox: list[int] = Field(description="x, y, width, height in pixels")
    damage_counts: DamageCounts
    building_counts: BuildingCounts
    priority_score: float
    centroid_lat: float | None = None
    centroid_lng: float | None = None


class AnalysisSummary(BaseModel):
    total_building_pixels: int
    total_buildings: int
    destroyed_pct: float
    major_pct: float
    minor_pct: float


class AnalysisResult(BaseModel):
    zones: list[Zone]
    summary: AnalysisSummary
    mask_path: str | None = None
    mask_base64: str | None = None
    pair_id: str | None = None
    inference_mode: str
    geo_available: bool = False


class ReportRequest(BaseModel):
    analysis: dict[str, Any]
    brief: str


class BriefRequest(BaseModel):
    analysis: dict[str, Any]
    context: str | None = None


class BriefResponse(BaseModel):
    brief: str
    source: str  # fireworks | stub


class DemoPair(BaseModel):
    id: str
    disaster_type: str
    pre_image: str
    post_image: str
