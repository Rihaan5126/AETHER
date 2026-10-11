"""Environment-variable configuration (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = f"sqlite:///{REPO_ROOT / 'aether.db'}"
    redis_url: str = "redis://localhost:6379/0"
    job_backend: Literal["inline", "celery"] = "inline"
    data_dir: Path = REPO_ROOT / "datasets" / "store"
    artifact_dir: Path = REPO_ROOT / "artifacts"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Write protection: when set, every mutating request needs header X-API-Key.
    api_key: str | None = None
    write_rate_limit_per_minute: int = Field(30, ge=1)

    # Resource limits (conservative laptop defaults).
    max_concurrent_jobs: int = Field(1, ge=1, le=8)
    max_pending_jobs: int = Field(5, ge=1, le=100)  # queued + running, across all clients
    max_workers_per_job: int = Field(1, ge=1, le=8)
    max_job_seconds: int = Field(3 * 3600, ge=60)
    max_population: int = Field(24, ge=2, le=64)
    max_generations: int = Field(10, ge=1, le=20)
    max_optuna_trials: int = Field(50, ge=1, le=100)
    max_upload_mb: int = Field(50, ge=1, le=500)
    enable_yahoo: bool = False

    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
