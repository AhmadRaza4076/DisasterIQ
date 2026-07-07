"""Smoke tests for FastAPI routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "inference_mode" in data
    assert data["demo_pairs"] >= 1


def test_demo_pairs() -> None:
    resp = client.get("/demo/pairs")
    assert resp.status_code == 200
    pairs = resp.json()
    assert isinstance(pairs, list)
    assert len(pairs) >= 1
    assert "id" in pairs[0]


def test_demo_image_rejects_path_traversal() -> None:
    resp = client.get("/demo/images/../manifest.json")
    assert resp.status_code in (400, 404)
