"""Abstract interfaces shared by the framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sain_glm_agent.domain.entities import (
    ConversationHistory,
    FileInfo,
    Message,
    ProviderRequest,
    ProviderResponse,
    RepositoryInfo,
)


class BaseModelProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a completion from a normalized request."""

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Describe provider capabilities."""


class BaseMemoryStore(ABC):
    """Interface for storing conversation history."""

    @abstractmethod
    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session."""

    @abstractmethod
    def get_history(self, session_id: str) -> ConversationHistory:
        """Fetch history for a session."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Remove a session from storage."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all known session identifiers."""


class BaseRepositoryService(ABC):
    """Interface for repository metadata and file access."""

    @abstractmethod
    def get_repo_info(self, owner: str, name: str) -> RepositoryInfo:
        """Return repository metadata."""

    @abstractmethod
    def list_files(
        self,
        owner: str,
        name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[FileInfo]:
        """List files in a repository directory."""

    @abstractmethod
    def get_file_content(
        self,
        owner: str,
        name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Fetch the content of a file."""

    @abstractmethod
    def get_directory_tree(
        self,
        owner: str,
        name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[FileInfo]:
        """Recursively list a repository directory tree."""


class BaseTool(ABC):
    """Interface for executable tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with normalized parameters."""
