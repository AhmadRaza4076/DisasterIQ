"""Sweep stale per-job upload and output directories."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_old_jobs(
    *roots: Path,
    max_age_hours: int = 24,
) -> int:
    """Delete subdirectories older than max_age_hours under each root. Returns count removed."""
    if max_age_hours <= 0:
        return 0

    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
                logger.info("Removed stale job dir: %s", child)
    return removed
