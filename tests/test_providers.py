"""Tests for sain_glm_agent.providers.*"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sain_glm_agent.providers.base import (
    BaseProvider,
    Message,
    MessageRole,
    ModelResponse,
)
from sain_glm_agent.providers.factory import ProviderFactory
from sain_glm_agent.providers.stubs import (
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    LocalProvider,
    OpenAIProvider,
    QwenProvider,
)

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_dict_user(self):
        m = Message(MessageRole.USER, "Hello!")
        d = m.to_dict()
        assert d == {"role": "user", "content": "Hello!"}

    def test_to_dict_with_name(self):
        m = Message(MessageRole.TOOL, "result", name="my_tool", tool_call_id="abc")
        d = m.to_dict()
        assert d["name"] == "my_tool"
        assert d["tool_call_id"] == "abc"

    def test_to_dict_no_optional(self):
        m = Message(MessageRole.ASSISTANT, "Hi")
        d = m.to_dict()
        assert "name" not in d
        assert "tool_call_id" not in d


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_total_tokens(self):
        r = ModelResponse(
            content="x", model="m", provider="p", input_tokens=10, output_tokens=20
        )
        assert r.total_tokens == 30

    def test_defaults(self):
        r = ModelResponse(content="", model="m", provider="p")
        assert r.finish_reason == "stop"
        assert r.raw == {}
        assert r.total_tokens == 0


# ---------------------------------------------------------------------------
# BaseProvider — concrete subclass for testing
# ---------------------------------------------------------------------------


class _DummyProvider(BaseProvider):
    PROVIDER_NAME = "dummy"

    def chat(self, messages, **kwargs):
        return ModelResponse(
            content="dummy", model="dummy-model", provider=self.PROVIDER_NAME
        )

    def stream_chat(self, messages, **kwargs):
        yield "chunk1"
        yield "chunk2"


class TestBaseProvider:
    def test_repr(self):
        p = _DummyProvider(model="test-model")
        assert "test-model" in repr(p)

    def test_provider_name(self):
        p = _DummyProvider(model="x")
        assert p.provider_name == "dummy"

    def test_chat(self):
        p = _DummyProvider(model="x")
        resp = p.chat([Message(MessageRole.USER, "hi")])
        assert resp.content == "dummy"

    def test_stream_chat(self):
        p = _DummyProvider(model="x")
        chunks = list(p.stream_chat([Message(MessageRole.USER, "hi")]))
        assert chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# Stubs — they all raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,kwargs",
    [
        (OpenAIProvider, {"api_key": "k"}),
        (ClaudeProvider, {"api_key": "k"}),
        (GeminiProvider, {"api_key": "k"}),
        (DeepSeekProvider, {"api_key": "k"}),
        (QwenProvider, {"api_key": "k"}),
        (LocalProvider, {}),
    ],
)
class TestStubProviders:
    def test_chat_raises(self, provider_cls, kwargs):
        p = provider_cls(**kwargs)
        with pytest.raises(NotImplementedError):
            p.chat([])

    def test_stream_raises(self, provider_cls, kwargs):
        p = provider_cls(**kwargs)
        with pytest.raises(NotImplementedError):
            list(p.stream_chat([]))


# ---------------------------------------------------------------------------
# ProviderFactory
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def test_missing_api_key_raises(self, monkeypatch):
        from sain_glm_agent.config.settings import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        cfg = Settings()
        with pytest.raises(ValueError, match="API key"):
            ProviderFactory.create(cfg)

    def test_unknown_provider_raises(self):
        from sain_glm_agent.config.settings import Settings

        cfg = Settings()
        # Monkeypatch provider field to an unknown value
        object.__setattr__(cfg, "provider", "nonexistent")  # type: ignore[arg-type]
        with pytest.raises((ValueError, AttributeError)):
            ProviderFactory.create(cfg)

    def test_creates_glm_provider(self, monkeypatch):
        """Factory should return a GLMProvider when provider=glm and key is set."""
        from sain_glm_agent.config.settings import Settings, get_settings
        from sain_glm_agent.providers.glm import GLMProvider

        get_settings.cache_clear()
        monkeypatch.setenv("SAIN_PROVIDER", "glm")
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")

        cfg = Settings()
        # Mock the ZhipuAI SDK so we don't need a real connection
        _mock_target = "sain_glm_agent.providers.glm.GLMProvider._build_client"
        with patch(_mock_target, return_value=MagicMock()):
            provider = ProviderFactory.create(cfg)
        assert isinstance(provider, GLMProvider)
