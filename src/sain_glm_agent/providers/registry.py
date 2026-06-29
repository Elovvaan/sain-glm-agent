"""Provider registry for pluggable model backends."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import ModelProvider
from .glm import GLMProvider

ProviderFactory = Callable[[Settings], ModelProvider]


class ProviderRegistry:
    """Registry mapping provider names to provider factories."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        """Register a provider factory."""

        self._factories[name.lower()] = factory

    def create(self, settings: Settings) -> ModelProvider:
        """Instantiate a provider for the current settings."""

        name = settings.provider.lower()
        factory = self._factories.get(name)
        if factory is None:
            supported = ", ".join(sorted(self._factories)) or "none"
            raise ConfigurationError(
                f"Unsupported provider '{settings.provider}'. Registered providers: {supported}."
            )
        return factory(settings)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Return the registered provider names."""

        return tuple(sorted(self._factories))


def build_default_registry() -> ProviderRegistry:
    """Build the default provider registry with extension points for new backends."""

    registry = ProviderRegistry()
    registry.register("glm", GLMProvider)
    return registry
