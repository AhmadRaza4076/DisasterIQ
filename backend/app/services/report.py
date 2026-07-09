"""Generate downloadable PDF field reports for offline coordination."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#1e293b"),
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#334155"),
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
        ),
        "brief": ParagraphStyle(
            "ReportBrief",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
        ),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_field_report_pdf(
    analysis: dict[str, Any],
    brief: str,
    pair_id: str | None = None,
) -> bytes:
    """Render analysis + brief into a PDF byte stream."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = _styles()
    story: list[Any] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("DisasterIQ Field Report", styles["title"]))
    story.append(Paragraph(f"Generated: {now}", styles["body"]))
    if pair_id:
        story.append(Paragraph(f"Imagery pair: <b>{_escape(pair_id)}</b>", styles["body"]))
    story.append(Spacer(1, 0.2 * inch))

    summary = analysis.get("summary", {})
    story.append(Paragraph("Executive Summary", styles["heading"]))
    summary_lines = [
        f"Total buildings assessed: <b>{summary.get('total_buildings', 0)}</b>",
        f"Destroyed: <b>{summary.get('destroyed_pct', 0)}%</b> &nbsp;|&nbsp; "
        f"Major: <b>{summary.get('major_pct', 0)}%</b> &nbsp;|&nbsp; "
        f"Minor: <b>{summary.get('minor_pct', 0)}%</b>",
    ]
    if analysis.get("geo_available"):
        summary_lines.append("Geographic coordinates: <b>available</b>")
    else:
        summary_lines.append("Geographic coordinates: <b>unavailable</b> (no xBD metadata)")
    for line in summary_lines:
        story.append(Paragraph(line, styles["body"]))
    story.append(Spacer(1, 0.15 * inch))

    zones = analysis.get("zones", [])
    story.append(Paragraph("Priority Zones (ML-ranked)", styles["heading"]))
    if zones:
        table_data = [
            ["Rank", "Score", "Destroyed", "Major", "Minor", "OK", "Coordinates"],
        ]
        for z in zones[:12]:
            bc = z.get("building_counts", {})
            lat, lng = z.get("centroid_lat"), z.get("centroid_lng")
            if lat is not None and lng is not None:
                coord = f"{lat:.5f}, {lng:.5f}"
            else:
                coord = "N/A"
            table_data.append(
                [
                    f"#{z.get('rank', '')}",
                    str(z.get("priority_score", "")),
                    str(bc.get("destroyed", 0)),
                    str(bc.get("major", 0)),
                    str(bc.get("minor", 0)),
                    str(bc.get("none", 0)),
                    coord,
                ]
            )
        table = Table(table_data, colWidths=[0.55 * inch, 0.65 * inch, 0.75 * inch, 0.6 * inch, 0.6 * inch, 0.5 * inch, 1.6 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
        if len(zones) > 12:
            story.append(Spacer(1, 0.08 * inch))
            story.append(
                Paragraph(
                    f"<i>Showing top 12 of {len(zones)} ranked zones.</i>",
                    styles["body"],
                )
            )
    else:
        story.append(Paragraph("No damage zones detected.", styles["body"]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Situation Brief", styles["heading"]))
    for para in brief.strip().split("\n"):
        if para.strip():
            story.append(Paragraph(_escape(para.strip()), styles["brief"]))
        else:
            story.append(Spacer(1, 0.08 * inch))

    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "<i>DisasterIQ — Team DarkNem. ML scores are deterministic; brief narrates ranked zones only.</i>",
            styles["body"],
        )
    )

    doc.build(story)
    return buf.getvalue()
