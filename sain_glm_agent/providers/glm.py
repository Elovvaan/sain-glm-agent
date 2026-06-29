"""GLM provider — wraps the ZhipuAI SDK to implement :class:`BaseProvider`.

The ZhipuAI SDK (``zhipuai``) follows the OpenAI-compatible chat-completion
interface.  The provider constructs the request, handles errors, and maps the
SDK response to the framework's :class:`ModelResponse`.

Supported models (non-exhaustive):
    * ``glm-4-flash`` — fast, cost-effective
    * ``glm-4-air``   — balanced
    * ``glm-4``       — flagship reasoning model
    * ``glm-4-0520``  — enhanced version

Usage::

    from sain_glm_agent.providers.glm import GLMProvider
    from sain_glm_agent.providers.base import Message, MessageRole

    provider = GLMProvider(api_key="...", model="glm-4-flash")
    response = provider.chat([Message(MessageRole.USER, "Hello!")])
    print(response.content)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sain_glm_agent.providers.base import (
    BaseProvider,
    Message,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class GLMProvider(BaseProvider):
    """Provider implementation for ZhipuAI GLM models.

    Args:
        api_key: ZhipuAI API key (``ZHIPUAI_API_KEY``).
        model: GLM model name (default: ``glm-4-flash``).
        base_url: Override the default API base URL.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.
    """

    PROVIDER_NAME = "glm"

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **extra_kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra_kwargs,
        )
        self._api_key = api_key
        self._base_url = base_url
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        """Instantiate and return the ZhipuAI client."""
        try:
            from zhipuai import ZhipuAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'zhipuai' package is required for GLMProvider. "
                "Install it with: pip install zhipuai"
            ) from exc

        return ZhipuAI(api_key=self._api_key)

    @staticmethod
    def _messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert framework :class:`Message` objects to SDK wire format."""
        return [m.to_dict() for m in messages]

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> ModelResponse:
        """Send messages to the GLM API and return the complete response.

        Args:
            messages: Full conversation history.
            **kwargs: Per-call overrides (``temperature``, ``max_tokens``).

        Returns:
            :class:`ModelResponse` containing the assistant reply.

        Raises:
            RuntimeError: If the API call fails.
        """
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": self._messages_to_dicts(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        logger.debug(
            "GLM chat request | model=%s | messages=%d",
            payload["model"],
            len(messages),
        )

        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:
            logger.error("GLM API error: %s", exc)
            raise RuntimeError(f"GLM API call failed: {exc}") from exc

        choice = response.choices[0]
        content: str = choice.message.content or ""
        usage = response.usage or {}

        return ModelResponse(
            content=content,
            model=response.model or self.model,
            provider=self.PROVIDER_NAME,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            finish_reason=choice.finish_reason or "stop",
            raw={"id": response.id},
        )

    def stream_chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream GLM response tokens as they arrive.

        Args:
            messages: Full conversation history.
            **kwargs: Per-call overrides.

        Yields:
            Successive text fragments.
        """
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": self._messages_to_dicts(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        logger.debug(
            "GLM stream request | model=%s | messages=%d",
            payload["model"],
            len(messages),
        )

        try:
            stream = self._client.chat.completions.create(**payload)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("GLM stream error: %s", exc)
            raise RuntimeError(f"GLM streaming call failed: {exc}") from exc
