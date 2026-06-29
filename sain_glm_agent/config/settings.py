"""Centralised, environment-driven configuration for SAIN GLM Agent.

All settings can be provided via environment variables or a ``.env`` file.
The ``SAIN_`` prefix is used for project-specific variables; third-party
API keys keep their conventional names (e.g. ``ZHIPUAI_API_KEY``).

Example::

    from sain_glm_agent.config import get_settings

    cfg = get_settings()
    print(cfg.provider)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderName(str, Enum):
    """Supported model-provider identifiers."""

    GLM = "glm"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    LOCAL = "local"


class LogFormat(str, Enum):
    """Supported log-output formats."""

    RICH = "rich"
    JSON = "json"
    PLAIN = "plain"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / ``.env`` file.

    Attributes:
        provider: Which model provider to use (default: ``glm``).

        zhipuai_api_key: API key for ZhipuAI / GLM.
        glm_model: GLM model name.
        glm_base_url: Base URL for the GLM API.

        openai_api_key: API key for OpenAI (optional).
        openai_model: OpenAI model name.

        anthropic_api_key: API key for Anthropic Claude (optional).
        claude_model: Claude model name.

        google_api_key: API key for Google Gemini (optional).
        gemini_model: Gemini model name.

        deepseek_api_key: API key for DeepSeek (optional).
        deepseek_model: DeepSeek model name.

        dashscope_api_key: API key for Qwen / DashScope (optional).
        qwen_model: Qwen model name.

        github_token: GitHub personal-access token for repo operations.

        max_tokens: Maximum tokens per model response.
        temperature: Sampling temperature (0.0 – 2.0).
        max_iterations: Maximum agent reasoning iterations.
        memory_max_messages: Sliding-window size for conversation history.

        log_level: Log verbosity (DEBUG, INFO, WARNING, ERROR).
        log_format: Log output style.
        log_file: Optional path to a log file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SAIN_",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Provider ─────────────────────────────────────────────────────────────
    provider: ProviderName = Field(ProviderName.GLM, alias="SAIN_PROVIDER")

    # ── GLM / ZhipuAI ────────────────────────────────────────────────────────
    zhipuai_api_key: str | None = Field(None, alias="ZHIPUAI_API_KEY")
    glm_model: str = Field("glm-4-flash", alias="SAIN_GLM_MODEL")
    glm_base_url: str = Field(
        "https://open.bigmodel.cn/api/paas/v4",
        alias="SAIN_GLM_BASE_URL",
    )

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="SAIN_OPENAI_MODEL")

    # ── Anthropic / Claude ───────────────────────────────────────────────────
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(
        "claude-3-5-sonnet-20241022", alias="SAIN_CLAUDE_MODEL"
    )

    # ── Google / Gemini ──────────────────────────────────────────────────────
    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-1.5-pro", alias="SAIN_GEMINI_MODEL")

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    deepseek_api_key: str | None = Field(None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field("deepseek-coder", alias="SAIN_DEEPSEEK_MODEL")

    # ── Qwen / DashScope ─────────────────────────────────────────────────────
    dashscope_api_key: str | None = Field(None, alias="DASHSCOPE_API_KEY")
    qwen_model: str = Field("qwen-max", alias="SAIN_QWEN_MODEL")

    # ── GitHub ───────────────────────────────────────────────────────────────
    github_token: str | None = Field(None, alias="GITHUB_TOKEN")

    # ── Agent behaviour ──────────────────────────────────────────────────────
    max_tokens: int = Field(4096, alias="SAIN_MAX_TOKENS", ge=1, le=32768)
    temperature: float = Field(0.1, alias="SAIN_TEMPERATURE", ge=0.0, le=2.0)
    max_iterations: int = Field(10, alias="SAIN_MAX_ITERATIONS", ge=1, le=100)
    memory_max_messages: int = Field(
        50, alias="SAIN_MEMORY_MAX_MESSAGES", ge=1, le=1000
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="SAIN_LOG_LEVEL")
    log_format: LogFormat = Field(LogFormat.RICH, alias="SAIN_LOG_FORMAT")
    log_file: Path | None = Field(None, alias="SAIN_LOG_FILE")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    def api_key_for(self, provider: ProviderName) -> str | None:
        """Return the API key for the given provider, or ``None`` if unset."""
        mapping: dict[ProviderName, str | None] = {
            ProviderName.GLM: self.zhipuai_api_key,
            ProviderName.OPENAI: self.openai_api_key,
            ProviderName.CLAUDE: self.anthropic_api_key,
            ProviderName.GEMINI: self.google_api_key,
            ProviderName.DEEPSEEK: self.deepseek_api_key,
            ProviderName.QWEN: self.dashscope_api_key,
            ProviderName.LOCAL: None,
        }
        return mapping.get(provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) application settings singleton.

    The settings object is created once per process from the environment /
    ``.env`` file.  Call ``get_settings.cache_clear()`` in tests to reset it.
    """
    return Settings()
