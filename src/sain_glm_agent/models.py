"""Core data models used across the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AgentAction(str, Enum):
    """High-level agent actions supported by the orchestration layer."""

    ANALYZE = "analyze"
    PLAN = "plan"
    GENERATE = "generate"
    PREPARE_PR = "prepare-pr"


@dataclass(slots=True)
class Message:
    """A chat message sent to or received from a model provider."""

    role: str
    content: str


@dataclass(slots=True)
class GenerationConfig:
    """Sampling and response limits for a model request."""

    temperature: float = 0.2
    max_tokens: int = 3000


@dataclass(slots=True)
class ModelRequest:
    """Input payload sent to a provider implementation."""

    model: str
    system_prompt: str
    messages: list[Message]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    """Normalized provider response returned to the orchestration layer."""

    provider: str
    model: str
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class RepositorySnapshot:
    """Summary of repository state used when building prompts."""

    root: Path
    files: list[str]
    directories: list[str]
    key_file_excerpts: dict[str, str]
    detected_languages: list[str]
    git_status: str
    diff_summary: str

    def to_prompt(self) -> str:
        """Serialize the snapshot into a compact prompt-friendly string."""

        file_list = "\n".join(f"- {path}" for path in self.files[:25]) or "- No files discovered"
        directory_list = (
            "\n".join(f"- {path}" for path in self.directories[:20]) or "- No directories discovered"
        )
        excerpts = []
        for path, content in self.key_file_excerpts.items():
            excerpts.append(f"## {path}\n{content}")
        excerpt_text = "\n\n".join(excerpts) if excerpts else "No key file excerpts available."
        languages = ", ".join(self.detected_languages) if self.detected_languages else "Unknown"
        git_status = self.git_status or "Clean working tree"
        diff_summary = self.diff_summary or "No current diff"
        return (
            f"Repository root: {self.root}\n"
            f"Detected languages: {languages}\n"
            f"Git status:\n{git_status}\n\n"
            f"Diff summary:\n{diff_summary}\n\n"
            f"Top directories:\n{directory_list}\n\n"
            f"Top files:\n{file_list}\n\n"
            f"Key file excerpts:\n{excerpt_text}"
        )


@dataclass(slots=True)
class AgentResult:
    """Structured output returned by the repository assistant."""

    action: AgentAction
    provider: str
    model: str
    repository_summary: str
    content: str
    usage: dict[str, int] = field(default_factory=dict)
