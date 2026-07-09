"""Integration tests for /analyze, /brief, and /report/pdf routes (stub mode)."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app

client = TestClient(app)


def _png_bytes(size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _first_pair_id() -> str:
    pairs = client.get("/demo/pairs").json()
    assert pairs, "no demo pairs available"
    return pairs[0]["id"]


def test_analyze_demo_pair_happy_path() -> None:
    resp = client.post("/analyze", data={"demo_pair_id": _first_pair_id()})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["zones"], list)
    assert "summary" in data
    assert data["inference_mode"].startswith("stub")


def test_analyze_invalid_demo_pair_id_rejected() -> None:
    resp = client.post("/analyze", data={"demo_pair_id": "../../etc/passwd"})
    assert resp.status_code == 400


def test_analyze_nonexistent_demo_pair_id() -> None:
    resp = client.post("/analyze", data={"demo_pair_id": "does_not_exist_123"})
    assert resp.status_code == 404


def test_analyze_traversal_filename_neutralized() -> None:
    png = _png_bytes()
    escape_target = settings.upload_dir.parent / "evil.png"
    existed_before = escape_target.exists()
    resp = client.post(
        "/analyze",
        files={
            "pre_image": ("../../evil.png", png, "image/png"),
            "post_image": ("../../evil.png", png, "image/png"),
        },
    )
    assert resp.status_code == 200
    if not existed_before:
        assert not escape_target.exists(), "traversal filename escaped the job directory"


def test_brief_stub_path() -> None:
    analysis = client.post("/analyze", data={"demo_pair_id": _first_pair_id()}).json()
    resp = client.post("/brief", json={"analysis": analysis, "context": "test context"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["brief"]
    assert data["source"] in {"stub", "fireworks", "fireworks-fallback"}


def test_report_pdf_returns_pdf() -> None:
    analysis = client.post("/analyze", data={"demo_pair_id": _first_pair_id()}).json()
    resp = client.post("/report/pdf", json={"analysis": analysis, "brief": "Test brief."})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"
