"""Tests for inference helpers."""

from pathlib import Path

from app.services.inference import list_demo_pairs, resolve_demo_target


def test_list_demo_pairs_returns_curated_ids() -> None:
    pairs = list_demo_pairs()
    assert len(pairs) >= 10
    ids = {p["id"] for p in pairs}
    assert "mexico-earthquake_00000005" in ids or any("mexico-earthquake" in i for i in ids)


def test_resolve_demo_target_for_known_pair() -> None:
    from app.config import settings

    post = settings.demo_data_dir / "images" / "mexico-earthquake_00000005_post_disaster.png"
    if not post.exists():
        return
    target = resolve_demo_target(post)
    assert target is not None
    assert target.exists()
