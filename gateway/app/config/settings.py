"""
Enterprise AI Platform Configuration Module.
Loads environment variables from .env via Pydantic BaseSettings (with robust dotenv fallback)
and provides Pathlib runtime paths.
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GATEWAY_DIR = PROJECT_ROOT / "gateway"
APP_DIR = GATEWAY_DIR / "app"
STATIC_DIR = APP_DIR / "static"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

def ensure_directories() -> None:
    """Safely bootstrap runtime directories without import-time side-effects."""
    for runtime_dir in [ARTIFACTS_DIR, LOGS_DIR, STATIC_DIR]:
        runtime_dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Centralized System Settings loaded from environment variables."""

    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", alias="VLLM_MODEL_NAME")
    hugging_face_hub_token: str | None = Field(default=None, alias="HUGGING_FACE_HUB_TOKEN")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    semantic_cache_threshold: float = Field(default=0.95, alias="SEMANTIC_CACHE_THRESHOLD")
    semantic_cache_ttl_seconds: int = Field(default=86400, alias="SEMANTIC_CACHE_TTL_SECONDS")

    gateway_host: str = Field(default="0.0.0.0", alias="GATEWAY_HOST")
    gateway_port: int = Field(default=8080, alias="GATEWAY_PORT")
    gateway_api_key: str = Field(default="secret_enterprise_ai_key_2026", alias="GATEWAY_API_KEY")

    neon_database_url: str = Field(default="", alias="NEON_DATABASE_URL")
    prometheus_metrics_path: str = "/metrics"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton cached instance of Settings."""
    return Settings()


# Singleton instance for direct imports
settings = get_settings()
