from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


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
    upload_dir: Path = Path(__file__).resolve().parents[1] / "uploads"
    output_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    grid_rows: int = 4
    grid_cols: int = 4
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
