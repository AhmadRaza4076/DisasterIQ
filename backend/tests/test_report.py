"""Tests for PDF field report generation."""

from __future__ import annotations

from app.services.report import build_field_report_pdf


def test_build_field_report_pdf_non_empty() -> None:
    analysis = {
        "geo_available": True,
        "summary": {
            "total_buildings": 42,
            "destroyed_pct": 10.0,
            "major_pct": 20.0,
            "minor_pct": 30.0,
        },
        "zones": [
            {
                "rank": 1,
                "priority_score": 85.5,
                "building_counts": {"none": 1, "minor": 2, "major": 3, "destroyed": 4},
                "centroid_lat": 34.7465,
                "centroid_lng": -92.2896,
            }
        ],
    }
    brief = "Zone #1 requires immediate assessment.\nDeploy teams to highest-priority areas first."
    pdf = build_field_report_pdf(analysis, brief, pair_id="midwest-flooding_00000000")
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 500


def test_pdf_minimal_analysis() -> None:
    analysis = {
        "geo_available": False,
        "summary": {"total_buildings": 1, "destroyed_pct": 0, "major_pct": 0, "minor_pct": 0},
        "zones": [{"rank": 1, "priority_score": 10, "building_counts": {"destroyed": 1}}],
    }
    pdf = build_field_report_pdf(analysis, "Test brief", pair_id="test")
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 500
