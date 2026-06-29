"""In-memory conversation history storage."""

from __future__ import annotations

from threading import RLock

from sain_glm_agent.domain.entities import ConversationHistory, Message
from sain_glm_agent.infrastructure.memory.base import BaseMemoryStore


class InMemoryStore(BaseMemoryStore):
    """Thread-safe in-memory store for short-lived sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationHistory] = {}
        self._lock = RLock()

    def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session history."""
        with self._lock:
            history = self._sessions.setdefault(
                session_id,
                ConversationHistory(session_id=session_id, messages=[]),
            )
            history.messages.append(message)

    def get_history(self, session_id: str) -> ConversationHistory:
        """Return the full history for a session."""
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                return ConversationHistory(session_id=session_id, messages=[])
            return ConversationHistory.model_validate(history.model_dump(mode="json"))

    def clear(self, session_id: str) -> None:
        """Remove a session if it exists."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all known session identifiers."""
        with self._lock:
            return sorted(self._sessions.keys())
