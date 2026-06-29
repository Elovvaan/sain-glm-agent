"""Sliding-window conversation memory for SAIN GLM Agent.

Keeps the last *N* messages in a typed list that can be passed directly to a
provider's :meth:`~sain_glm_agent.providers.base.BaseProvider.chat` method.

Design decisions:
* The system prompt (if any) is stored separately and is always prepended so
  it never falls out of the window.
* Token counting is approximate (character / 4) unless an exact tokeniser is
  plugged in via :attr:`ConversationMemory.token_counter`.

Usage::

    from sain_glm_agent.memory import ConversationMemory
    from sain_glm_agent.providers.base import MessageRole

    mem = ConversationMemory(max_messages=20)
    mem.set_system("You are a helpful coding assistant.")
    mem.add_user("Explain list comprehensions in Python.")
    mem.add_assistant("A list comprehension ...")

    messages = mem.get_messages()   # ready to pass to provider.chat()
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sain_glm_agent.providers.base import Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """Thread-safe sliding-window conversation history.

    Attributes:
        max_messages: Maximum number of *non-system* messages to retain.
        token_counter: Optional callable that returns a token count for a
            string.  Falls back to ``len(text) // 4`` when not provided.
    """

    max_messages: int = 50
    token_counter: Callable[[str], int] | None = None

    _system_message: Message | None = field(default=None, init=False, repr=False)
    _messages: list[Message] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_system(self, content: str) -> None:
        """Set or replace the system prompt.

        The system message is always the first item returned by
        :meth:`get_messages` and is never evicted by the sliding window.

        Args:
            content: System prompt text.
        """
        self._system_message = Message(role=MessageRole.SYSTEM, content=content)
        logger.debug("System prompt updated (%d chars)", len(content))

    def add_user(self, content: str) -> None:
        """Append a user message and prune old messages if necessary.

        Args:
            content: User's message text.
        """
        self._append(Message(role=MessageRole.USER, content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant (model) message.

        Args:
            content: Model's reply text.
        """
        self._append(Message(role=MessageRole.ASSISTANT, content=content))

    def add_tool_result(self, content: str, tool_call_id: str) -> None:
        """Append a tool-execution result.

        Args:
            content: Tool output text.
            tool_call_id: Identifier linking this result to the original call.
        """
        self._append(
            Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tool_call_id,
            )
        )

    def clear(self) -> None:
        """Remove all messages (system prompt is preserved)."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_messages(self) -> list[Message]:
        """Return the full message list ready for provider consumption.

        Returns:
            List beginning with the system prompt (if set) followed by the
            sliding-window conversation history.
        """
        result: list[Message] = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    @property
    def message_count(self) -> int:
        """Number of non-system messages in the window."""
        return len(self._messages)

    @property
    def token_estimate(self) -> int:
        """Rough total token count across all messages in the window."""
        counter = self.token_counter or (lambda t: len(t) // 4)
        total = 0
        if self._system_message:
            total += counter(self._system_message.content)
        for msg in self._messages:
            total += counter(msg.content)
        return total

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the full history (including system prompt) to JSON."""
        data: list[dict] = []
        if self._system_message:
            data.append(self._system_message.to_dict())
        data.extend(m.to_dict() for m in self._messages)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str, max_messages: int = 50) -> ConversationMemory:
        """Restore a :class:`ConversationMemory` from a JSON string.

        Args:
            raw: JSON string previously produced by :meth:`to_json`.
            max_messages: Sliding-window size for the restored object.

        Returns:
            A new :class:`ConversationMemory` instance.
        """
        mem = cls(max_messages=max_messages)
        for item in json.loads(raw):
            role = MessageRole(item["role"])
            msg = Message(
                role=role,
                content=item["content"],
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
            )
            if role == MessageRole.SYSTEM:
                mem._system_message = msg
            else:
                mem._messages.append(msg)
        return mem

    def save(self, path: Path) -> None:
        """Persist the conversation to a JSON file.

        Args:
            path: Destination file path (parent directories are created).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, max_messages: int = 50) -> ConversationMemory:
        """Load a conversation from a JSON file.

        Args:
            path: Source file path.
            max_messages: Sliding-window size for the restored object.

        Returns:
            A new :class:`ConversationMemory` instance.
        """
        return cls.from_json(path.read_text(encoding="utf-8"), max_messages)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            # Evict the oldest pair (user + assistant) to keep context coherent
            self._messages.pop(0)
            if self._messages:
                self._messages.pop(0)
        logger.debug(
            "Memory: added %s message (%d in window)",
            message.role.value,
            len(self._messages),
        )
