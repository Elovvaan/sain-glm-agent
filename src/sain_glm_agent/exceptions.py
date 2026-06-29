"""Custom exceptions for SAIN GLM Agent."""


class SainAgentError(Exception):
    """Base exception for the project."""


class ConfigurationError(SainAgentError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(SainAgentError):
    """Raised when a model provider fails."""


class RepositoryError(SainAgentError):
    """Raised when repository inspection fails."""


class ToolExecutionError(SainAgentError):
    """Raised when a configured tool cannot be executed safely."""
