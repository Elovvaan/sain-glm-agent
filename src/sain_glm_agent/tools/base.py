"""Base definitions for safe tool execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for tools exposed to the agent."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return structured data."""
