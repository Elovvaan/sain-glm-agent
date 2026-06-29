"""Configuration management backed by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


DEFAULT_ALLOWED_COMMANDS = ("git", "python", "pytest", "ruff", "mypy")


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the agent."""

    provider: str = "glm"
    model: str = "glm-5.2"
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    timeout_seconds: int = 60
    max_tokens: int = 3000
    temperature: float = 0.2
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".sain_glm_agent")
    memory_file: Path | None = None
    max_context_files: int = 6
    max_file_bytes: int = 16_000
    allowed_commands: tuple[str, ...] = DEFAULT_ALLOWED_COMMANDS

    def __post_init__(self) -> None:
        if self.memory_file is None:
            self.memory_file = self.data_dir / "memory.json"
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""

        provider = os.getenv("SAIN_PROVIDER", "glm")
        model = os.getenv("SAIN_MODEL", "glm-5.2")
        api_key = os.getenv("SAIN_API_KEY") or os.getenv("GLM_API_KEY")
        base_url = os.getenv("SAIN_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        timeout_seconds = int(os.getenv("SAIN_TIMEOUT_SECONDS", "60"))
        max_tokens = int(os.getenv("SAIN_MAX_TOKENS", "3000"))
        temperature = float(os.getenv("SAIN_TEMPERATURE", "0.2"))
        log_level = os.getenv("SAIN_LOG_LEVEL", "INFO")
        data_dir = Path(os.getenv("SAIN_DATA_DIR", str(Path.home() / ".sain_glm_agent"))).expanduser()
        memory_value = os.getenv("SAIN_MEMORY_FILE")
        memory_file = Path(memory_value).expanduser() if memory_value else None
        max_context_files = int(os.getenv("SAIN_MAX_CONTEXT_FILES", "6"))
        max_file_bytes = int(os.getenv("SAIN_MAX_FILE_BYTES", "16000"))
        allowed_raw = os.getenv("SAIN_ALLOWED_COMMANDS")
        allowed_commands = (
            tuple(part.strip() for part in allowed_raw.split(",") if part.strip())
            if allowed_raw
            else DEFAULT_ALLOWED_COMMANDS
        )
        settings = cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            log_level=log_level,
            data_dir=data_dir,
            memory_file=memory_file,
            max_context_files=max_context_files,
            max_file_bytes=max_file_bytes,
            allowed_commands=allowed_commands,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate settings that must always be safe and sane."""

        if self.timeout_seconds <= 0:
            raise ConfigurationError("SAIN_TIMEOUT_SECONDS must be greater than zero.")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("SAIN_TEMPERATURE must be between 0 and 2.")
        if self.max_tokens <= 0:
            raise ConfigurationError("SAIN_MAX_TOKENS must be greater than zero.")
        if self.max_context_files <= 0:
            raise ConfigurationError("SAIN_MAX_CONTEXT_FILES must be greater than zero.")
        if self.max_file_bytes <= 0:
            raise ConfigurationError("SAIN_MAX_FILE_BYTES must be greater than zero.")
        if not self.allowed_commands:
            raise ConfigurationError("At least one allowed command must be configured.")

    def require_api_key(self) -> None:
        """Ensure remote providers have credentials before inference."""

        if not self.api_key:
            raise ConfigurationError(
                "Missing API key. Set SAIN_API_KEY or GLM_API_KEY before calling a remote model provider."
            )

    def ensure_directories(self) -> None:
        """Create the configured data directory if it does not exist."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_file is not None:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    def to_redacted_dict(self) -> dict[str, object]:
        """Serialize settings without exposing secrets."""

        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": "***redacted***" if self.api_key else None,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "memory_file": str(self.memory_file) if self.memory_file else None,
            "max_context_files": self.max_context_files,
            "max_file_bytes": self.max_file_bytes,
            "allowed_commands": list(self.allowed_commands),
        }
