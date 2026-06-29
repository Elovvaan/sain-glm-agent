"""Base provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Interface implemented by all model providers."""

    name: str

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for the given model request."""
