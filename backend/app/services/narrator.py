"""Fireworks LLM narrator — narrates ranked JSON only, never re-ranks."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import settings
from app.schemas import BriefResponse

SYSTEM_PROMPT = """You are a disaster response analyst. You receive a JSON object with
deterministic damage zone scores from satellite imagery analysis.

Rules:
- Narrate the situation in plain language for emergency coordinators.
- Reference zone ranks and building_counts (number of structures per damage class) exactly as provided.
- Do NOT use pixel counts (damage_counts) — only building_counts and total_buildings.
- Do NOT change rankings, scores, or invent additional damage data.
- Keep the brief under 250 words.
- Mention priority zones first and recommend where to deploy resources first.
- If centroid_lat/centroid_lng are present on a zone, mention coordinates for top zones.
"""


def _stub_brief(analysis: dict[str, Any], context: str | None) -> str:
    summary = analysis.get("summary", {})
    zones = analysis.get("zones", [])[:3]
    lines = [
        "SITUATION BRIEF (stub — set FIREWORKS_API_KEY for live narration)",
        "",
    ]
    if context:
        lines.append(f"Context: {context}")
        lines.append("")
    lines.append(
        f"Overall: {summary.get('total_buildings', 0)} buildings assessed. "
        f"Destroyed: {summary.get('destroyed_pct', 0)}%, "
        f"Major: {summary.get('major_pct', 0)}%, "
        f"Minor: {summary.get('minor_pct', 0)}%."
    )
    lines.append("")
    lines.append("Priority zones (pre-ranked by ML):")
    for z in zones:
        bc = z.get("building_counts", {})
        coord = ""
        lat, lng = z.get("centroid_lat"), z.get("centroid_lng")
        if lat is not None and lng is not None:
            coord = f" @ {lat:.5f}, {lng:.5f}"
        lines.append(
            f"  Zone #{z.get('rank')}: score {z.get('priority_score')}{coord} — "
            f"destroyed={bc.get('destroyed', 0)}, major={bc.get('major', 0)}, "
            f"minor={bc.get('minor', 0)}, undamaged={bc.get('none', 0)} buildings"
        )
    lines.append("")
    lines.append(
        "Recommendation: Deploy assessment teams to highest-scored zones first "
        "while verifying access routes and secondary hazards."
    )
    return "\n".join(lines)


async def generate_brief(analysis: dict[str, Any], context: str | None = None) -> BriefResponse:
    if not settings.fireworks_api_key:
        return BriefResponse(brief=_stub_brief(analysis, context), source="stub")

    user_content = json.dumps({"analysis": analysis, "context": context}, indent=2)
    payload = {
        "model": settings.fireworks_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {settings.fireworks_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return BriefResponse(brief=content.strip(), source="fireworks")
    except httpx.HTTPError:
        return BriefResponse(
            brief=_stub_brief(analysis, context),
            source="fireworks-fallback",
        )
