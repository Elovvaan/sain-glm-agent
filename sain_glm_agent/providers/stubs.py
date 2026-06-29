"""Stub provider implementations for OpenAI, Claude, Gemini, DeepSeek, Qwen.

Each stub extends :class:`BaseProvider` and follows the same interface as the
GLM provider.  They raise :class:`NotImplementedError` with a clear message so
developers know exactly what to implement.

Adding a real implementation:
1. Install the relevant SDK.
2. Replace the ``raise NotImplementedError`` body with actual API calls,
   mirroring the pattern in :mod:`sain_glm_agent.providers.glm`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sain_glm_agent.providers.base import BaseProvider, Message, ModelResponse


def _not_implemented(provider: str) -> NotImplementedError:
    return NotImplementedError(
        f"The {provider} provider is not yet implemented. "
        f"See sain_glm_agent/providers/stubs.py for guidance."
    )


class OpenAIProvider(BaseProvider):
    """Stub for the OpenAI chat-completion API (gpt-4o, gpt-4-turbo …)."""

    PROVIDER_NAME = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("OpenAI")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("OpenAI")
        yield  # make this a generator


class ClaudeProvider(BaseProvider):
    """Stub for the Anthropic Claude API."""

    PROVIDER_NAME = "claude"

    def __init__(
        self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs: Any
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("Claude")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("Claude")
        yield


class GeminiProvider(BaseProvider):
    """Stub for the Google Gemini API."""

    PROVIDER_NAME = "gemini"

    def __init__(
        self, api_key: str, model: str = "gemini-1.5-pro", **kwargs: Any
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("Gemini")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("Gemini")
        yield


class DeepSeekProvider(BaseProvider):
    """Stub for the DeepSeek API."""

    PROVIDER_NAME = "deepseek"

    def __init__(
        self, api_key: str, model: str = "deepseek-coder", **kwargs: Any
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("DeepSeek")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("DeepSeek")
        yield


class QwenProvider(BaseProvider):
    """Stub for the Alibaba Qwen / DashScope API."""

    PROVIDER_NAME = "qwen"

    def __init__(
        self, api_key: str, model: str = "qwen-max", **kwargs: Any
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = api_key

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("Qwen")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("Qwen")
        yield


class LocalProvider(BaseProvider):
    """Stub for a locally-hosted model (e.g. Ollama, LM Studio, vLLM)."""

    PROVIDER_NAME = "local"

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._base_url = base_url

    def chat(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raise _not_implemented("Local")

    def stream_chat(
        self, messages: list[Message], **kwargs: Any
    ) -> Generator[str, None, None]:
        raise _not_implemented("Local")
        yield
