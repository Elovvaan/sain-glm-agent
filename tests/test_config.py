"""Tests for sain_glm_agent.config.settings."""


import pytest

from sain_glm_agent.config.settings import LogFormat, ProviderName, Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure the lru_cache is reset between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestProviderName:
    def test_all_values_accessible(self):
        assert ProviderName.GLM == "glm"
        assert ProviderName.OPENAI == "openai"
        assert ProviderName.CLAUDE == "claude"
        assert ProviderName.GEMINI == "gemini"
        assert ProviderName.DEEPSEEK == "deepseek"
        assert ProviderName.QWEN == "qwen"
        assert ProviderName.LOCAL == "local"


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SAIN_PROVIDER", raising=False)
        monkeypatch.delenv("SAIN_GLM_MODEL", raising=False)
        cfg = Settings()
        assert cfg.provider == ProviderName.GLM
        assert cfg.glm_model == "glm-4-flash"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.1
        assert cfg.max_iterations == 10
        assert cfg.log_level == "INFO"
        assert cfg.log_format == LogFormat.RICH

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SAIN_PROVIDER", "openai")
        monkeypatch.setenv("SAIN_MAX_TOKENS", "2048")
        monkeypatch.setenv("SAIN_TEMPERATURE", "0.7")
        monkeypatch.setenv("SAIN_LOG_LEVEL", "DEBUG")
        cfg = Settings()
        assert cfg.provider == ProviderName.OPENAI
        assert cfg.max_tokens == 2048
        assert cfg.temperature == pytest.approx(0.7)
        assert cfg.log_level == "DEBUG"

    def test_log_level_validation(self, monkeypatch):
        monkeypatch.setenv("SAIN_LOG_LEVEL", "invalid")
        with pytest.raises(ValueError):
            Settings()

    def test_log_level_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SAIN_LOG_LEVEL", "warning")
        cfg = Settings()
        assert cfg.log_level == "WARNING"

    def test_api_key_for_glm(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key-abc")
        cfg = Settings()
        assert cfg.api_key_for(ProviderName.GLM) == "test-key-abc"

    def test_api_key_for_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = Settings()
        assert cfg.api_key_for(ProviderName.OPENAI) is None

    def test_api_key_for_local(self):
        cfg = Settings()
        assert cfg.api_key_for(ProviderName.LOCAL) is None


class TestGetSettings:
    def test_returns_settings_instance(self):
        cfg = get_settings()
        assert isinstance(cfg, Settings)

    def test_cached(self):
        cfg1 = get_settings()
        cfg2 = get_settings()
        assert cfg1 is cfg2

    def test_cache_clears(self, monkeypatch):
        get_settings.cache_clear()
        cfg1 = get_settings()
        get_settings.cache_clear()
        monkeypatch.setenv("SAIN_MAX_TOKENS", "1111")
        cfg2 = get_settings()
        assert cfg1 is not cfg2
        assert cfg2.max_tokens == 1111
