"""Tests for stale job directory cleanup."""

from __future__ import annotations

import time
from pathlib import Path

from app.services.cleanup import cleanup_old_jobs


def test_cleanup_old_jobs_removes_stale_directories(tmp_path: Path) -> None:
    old_dir = tmp_path / "oldjob"
    new_dir = tmp_path / "newjob"
    old_dir.mkdir()
    new_dir.mkdir()
    old_ts = time.time() - 48 * 3600
    Path(old_dir / "file.txt").write_text("x", encoding="utf-8")
    import os

    os.utime(old_dir, (old_ts, old_ts))

    removed = cleanup_old_jobs(tmp_path, max_age_hours=24)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()


def test_cleanup_old_jobs_skips_when_max_age_zero(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    stale.mkdir()
    assert cleanup_old_jobs(tmp_path, max_age_hours=0) == 0
    assert stale.exists()
