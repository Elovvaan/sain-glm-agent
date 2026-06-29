"""Base classes for model providers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from sain_glm_agent.domain.entities import ProviderRequest, ProviderResponse
from sain_glm_agent.domain.interfaces import BaseModelProvider


class BaseProvider(BaseModelProvider):
    """Reusable base implementation for model providers."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a completion response."""

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return a capabilities map for the provider."""
