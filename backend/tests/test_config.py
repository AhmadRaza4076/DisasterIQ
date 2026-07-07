"""Unit tests for settings path resolution."""

from __future__ import annotations

from app.config import _REPO_ROOT, Settings


def test_relative_checkpoint_resolves_to_repo_root(monkeypatch, tmp_path) -> None:
    """A relative PYTORCH_CHECKPOINT_PATH must resolve against the repo root,
    not the current working directory (backend/ when started via the script)."""
    monkeypatch.setenv("PYTORCH_CHECKPOINT_PATH", "ml/checkpoints/damage_best.ckpt")
    # Simulate the backend being started with cwd=backend/ (or anywhere else).
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    expected = (_REPO_ROOT / "ml" / "checkpoints" / "damage_best.ckpt").resolve()
    assert settings.pytorch_checkpoint_path == expected
    assert settings.pytorch_checkpoint_path.is_absolute()


def test_relative_demo_data_dir_resolves_to_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEMO_DATA_DIR", "data/demo")
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    expected = (_REPO_ROOT / "data" / "demo").resolve()
    assert settings.demo_data_dir == expected


def test_absolute_checkpoint_path_is_left_unchanged(monkeypatch, tmp_path) -> None:
    abs_path = tmp_path / "custom" / "model.ckpt"
    monkeypatch.setenv("PYTORCH_CHECKPOINT_PATH", str(abs_path))

    settings = Settings(_env_file=None)

    assert settings.pytorch_checkpoint_path == abs_path
