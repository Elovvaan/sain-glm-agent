"""Provider registry and factory helpers."""

from __future__ import annotations

from sain_glm_agent.domain.interfaces import BaseModelProvider
from sain_glm_agent.infrastructure.providers.glm import GLMProvider


class ProviderRegistry:
    """Registry for model provider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseModelProvider] = {}

    def register(self, name: str, provider: BaseModelProvider) -> None:
        """Register a provider instance under a unique name."""
        key = name.strip().lower()
        if not key:
            raise ValueError("Provider name cannot be empty.")
        self._providers[key] = provider

    def get_provider(self, name: str) -> BaseModelProvider:
        """Return a provider by name or raise a helpful error."""
        key = name.strip().lower()
        try:
            return self._providers[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "none"
            raise ValueError(
                f"Unknown provider '{name}'. Available providers: {available}."
            ) from exc


def create_provider_from_settings(settings) -> BaseModelProvider:
    """Create the active provider instance from application settings."""
    registry = ProviderRegistry()
    registry.register(
        "glm",
        GLMProvider(
            api_key=getattr(settings, "glm_api_key", None),
            base_url=getattr(settings, "glm_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            default_model=getattr(settings, "glm_model", "glm-4-flash"),
        ),
    )
    return registry.get_provider(getattr(settings, "active_provider", "glm"))
