"""Provider factory — creates the correct :class:`BaseProvider` from settings.

The factory reads the :class:`~sain_glm_agent.config.settings.Settings` object
and instantiates the matching provider, forwarding the appropriate API key,
model name, and generation parameters.

Usage::

    from sain_glm_agent.providers import ProviderFactory

    provider = ProviderFactory.create()          # uses env / .env settings
    response = provider.chat([...])
"""

from __future__ import annotations

from sain_glm_agent.config.settings import ProviderName, Settings, get_settings
from sain_glm_agent.providers.base import BaseProvider


class ProviderFactory:
    """Static factory for model providers.

    All provider-specific import and instantiation logic is centralised here so
    the rest of the framework remains provider-agnostic.
    """

    @staticmethod
    def create(settings: Settings | None = None) -> BaseProvider:
        """Instantiate the provider configured in *settings*.

        Args:
            settings: Application settings. Defaults to :func:`get_settings`.

        Returns:
            A fully initialised :class:`BaseProvider` subclass.

        Raises:
            ValueError: If the provider requires an API key that is not set.
            ValueError: If the provider name is unknown.
        """
        cfg = settings or get_settings()
        name = cfg.provider

        common = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }

        if name == ProviderName.GLM:
            return ProviderFactory._create_glm(cfg, common)
        if name == ProviderName.OPENAI:
            return ProviderFactory._create_openai(cfg, common)
        if name == ProviderName.CLAUDE:
            return ProviderFactory._create_claude(cfg, common)
        if name == ProviderName.GEMINI:
            return ProviderFactory._create_gemini(cfg, common)
        if name == ProviderName.DEEPSEEK:
            return ProviderFactory._create_deepseek(cfg, common)
        if name == ProviderName.QWEN:
            return ProviderFactory._create_qwen(cfg, common)
        if name == ProviderName.LOCAL:
            return ProviderFactory._create_local(cfg, common)

        raise ValueError(f"Unknown provider: {name!r}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_key(key: str | None, provider: str) -> str:
        if not key:
            raise ValueError(
                f"API key for provider '{provider}' is not set. "
                f"Check your .env file or environment variables."
            )
        return key

    @staticmethod
    def _create_glm(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.glm import GLMProvider

        api_key = ProviderFactory._require_key(cfg.zhipuai_api_key, "glm")
        return GLMProvider(
            api_key=api_key,
            model=cfg.glm_model,
            base_url=cfg.glm_base_url,
            **common,
        )

    @staticmethod
    def _create_openai(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import OpenAIProvider

        api_key = ProviderFactory._require_key(cfg.openai_api_key, "openai")
        return OpenAIProvider(api_key=api_key, model=cfg.openai_model, **common)

    @staticmethod
    def _create_claude(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import ClaudeProvider

        api_key = ProviderFactory._require_key(cfg.anthropic_api_key, "claude")
        return ClaudeProvider(api_key=api_key, model=cfg.claude_model, **common)

    @staticmethod
    def _create_gemini(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import GeminiProvider

        api_key = ProviderFactory._require_key(cfg.google_api_key, "gemini")
        return GeminiProvider(api_key=api_key, model=cfg.gemini_model, **common)

    @staticmethod
    def _create_deepseek(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import DeepSeekProvider

        api_key = ProviderFactory._require_key(cfg.deepseek_api_key, "deepseek")
        return DeepSeekProvider(api_key=api_key, model=cfg.deepseek_model, **common)

    @staticmethod
    def _create_qwen(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import QwenProvider

        api_key = ProviderFactory._require_key(cfg.dashscope_api_key, "qwen")
        return QwenProvider(api_key=api_key, model=cfg.qwen_model, **common)

    @staticmethod
    def _create_local(cfg: Settings, common: dict) -> BaseProvider:
        from sain_glm_agent.providers.stubs import LocalProvider

        return LocalProvider(**common)
