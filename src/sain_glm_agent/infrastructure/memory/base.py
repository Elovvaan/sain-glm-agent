"""Abstract base class for memory stores."""

from __future__ import annotations

from abc import abstractmethod

from sain_glm_agent.domain.interfaces import BaseMemoryStore as DomainBaseMemoryStore


class BaseMemoryStore(DomainBaseMemoryStore):
    """Concrete extension point for memory stores."""

    @abstractmethod
    def add_message(self, session_id: str, message) -> None:
        """Add a message to a session."""

    @abstractmethod
    def get_history(self, session_id: str):
        """Return conversation history for a session."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Remove all data for a session."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List stored session identifiers."""
