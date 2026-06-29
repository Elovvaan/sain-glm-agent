"""Abstract base classes for all model providers.

Every concrete provider must subclass :class:`BaseProvider` and implement the
:meth:`~BaseProvider.chat` and :meth:`~BaseProvider.stream_chat` methods.

Example::

    class MyProvider(BaseProvider):
        def chat(self, messages, **kwargs):
            ...
        def stream_chat(self, messages, **kwargs):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Standard chat-message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single chat message exchanged with the model.

    Attributes:
        role: Who authored the message.
        content: Text content of the message.
        name: Optional name tag (used for ``tool`` role messages).
        tool_call_id: Identifier linking a tool result to a call request.
    """

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the OpenAI-compatible wire format."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ModelResponse:
    """Structured response returned by a provider's chat method.

    Attributes:
        content: The model's text reply.
        model: Model identifier reported by the API.
        provider: Provider name string.
        input_tokens: Prompt-token count (if available).
        output_tokens: Completion-token count (if available).
        finish_reason: Reason the generation stopped (``stop``, ``length`` …).
        raw: The original response payload for provider-specific inspection.
    """

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output token counts."""
        return self.input_tokens + self.output_tokens


class BaseProvider(ABC):
    """Abstract base class for all model providers.

    Subclasses must implement :meth:`chat` (and optionally :meth:`stream_chat`)
    to integrate a new model backend without touching the agent core.

    Args:
        model: Model name / identifier to pass to the API.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.
        extra_kwargs: Provider-specific keyword arguments forwarded verbatim.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **extra_kwargs: Any,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a list of messages and return a complete response.

        Args:
            messages: Conversation history including the latest user turn.
            **kwargs: Override per-call settings (temperature, max_tokens …).

        Returns:
            A :class:`ModelResponse` with the model's reply.
        """

    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yield response text chunks as they arrive (streaming).

        Args:
            messages: Conversation history including the latest user turn.
            **kwargs: Override per-call settings.

        Yields:
            Successive text fragments of the model's reply.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        # Use class-level PROVIDER_NAME constant when available
        if hasattr(self.__class__, "PROVIDER_NAME"):
            return str(self.__class__.PROVIDER_NAME)
        return self.__class__.__name__.replace("Provider", "").lower().lstrip("_") or "unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
