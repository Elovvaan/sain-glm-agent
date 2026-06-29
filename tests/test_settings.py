from __future__ import annotations

import pytest

from sain_glm_agent.infrastructure.config.settings import Settings, get_settings, validate_settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM_API_KEY", "secret")
    settings = Settings()
    assert settings.glm_api_key == "secret"
    assert settings.glm_base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert settings.glm_model == "glm-4-flash"
    assert settings.github_base_url == "https://api.github.com"
    assert settings.log_level == "INFO"
    assert settings.active_provider == "glm"


def test_validate_settings_requires_glm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    settings = Settings()
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        validate_settings(settings)


def test_validate_settings_accepts_non_glm_provider() -> None:
    settings = Settings(ACTIVE_PROVIDER="custom", LOG_LEVEL="DEBUG")
    validated = validate_settings(settings)
    assert validated.active_provider == "custom"
    assert validated.log_level == "DEBUG"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("GLM_API_KEY", "abc")
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
