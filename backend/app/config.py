from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


def _resolve_against_repo_root(value: Path) -> Path:
    """Make relative paths (e.g. from .env) resolve against the repo root.

    The backend can be started with cwd=backend/, so a relative path like
    ``ml/checkpoints/damage_best.ckpt`` from .env must not be resolved
    against the current working directory.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fireworks_api_key: str = ""
    fireworks_model: str = "accounts/fireworks/models/glm-5p2"
    demo_data_dir: Path = _REPO_ROOT / "data" / "demo"
    test_data_dir: Path = _REPO_ROOT / "data" / "test"
    inference_mode: str = "stub"  # stub | docker | pytorch
    xview2_docker_image: str = "darknem-xview2-inference"
    pytorch_checkpoint_path: Path = _REPO_ROOT / "ml" / "checkpoints" / "damage_best.ckpt"
    pytorch_docker_image: str = "darknem-xview2-pytorch"
    pytorch_repo_dir: Path = _REPO_ROOT / "ml" / "pytorch-xview2"
    pytorch_inference_dir: Path = _REPO_ROOT / "ml" / "pytorch-inference"
    upload_dir: Path = Path(__file__).resolve().parents[1] / "uploads"
    output_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    grid_rows: int = 4
    grid_cols: int = 4
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        self.demo_data_dir = _resolve_against_repo_root(self.demo_data_dir)
        self.test_data_dir = _resolve_against_repo_root(self.test_data_dir)
        self.pytorch_checkpoint_path = _resolve_against_repo_root(self.pytorch_checkpoint_path)
        self.pytorch_repo_dir = _resolve_against_repo_root(self.pytorch_repo_dir)
        self.pytorch_inference_dir = _resolve_against_repo_root(self.pytorch_inference_dir)
        return self

    @model_validator(mode="after")
    def _validate_inference_mode(self) -> "Settings":
        allowed = {"stub", "docker", "pytorch"}
        if self.inference_mode.lower() not in allowed:
            raise ValueError(
                f"Invalid INFERENCE_MODE '{self.inference_mode}'. Must be one of {sorted(allowed)}."
            )
        return self


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
