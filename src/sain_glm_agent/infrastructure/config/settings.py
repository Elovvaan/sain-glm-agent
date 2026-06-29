"""Application settings powered by pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for SAIN GLM Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    glm_api_key: str | None = Field(default=None, alias="GLM_API_KEY")
    glm_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", alias="GLM_BASE_URL")
    glm_model: str = Field(default="glm-4-flash", alias="GLM_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    github_base_url: str = Field(default="https://api.github.com", alias="GITHUB_BASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    active_provider: str = Field(default="glm", alias="ACTIVE_PROVIDER")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


def validate_settings(settings: Settings | None = None) -> Settings:
    """Validate settings and raise actionable errors when required values are missing."""
    resolved = settings or get_settings()
    provider = resolved.active_provider.strip().lower()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

    if provider == "glm" and not resolved.glm_api_key:
        raise ValueError(
            "GLM_API_KEY is required when ACTIVE_PROVIDER=glm. "
            "Set it in your environment or .env file."
        )
    if not resolved.github_base_url.startswith("http"):
        raise ValueError("GITHUB_BASE_URL must be a valid HTTP or HTTPS URL.")
    if resolved.log_level.upper() not in valid_levels:
        raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(valid_levels))}.")
    return resolved
