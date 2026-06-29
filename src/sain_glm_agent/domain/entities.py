"""Core domain entities used across the framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single conversational message."""

    role: str
    content: str
    timestamp: datetime | None = None


class ConversationHistory(BaseModel):
    """Represents all messages for a session."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)


class RepositoryInfo(BaseModel):
    """Metadata describing a GitHub repository."""

    owner: str
    name: str
    description: str | None = None
    language: str | None = None
    default_branch: str = "main"
    stars: int = 0
    topics: list[str] = Field(default_factory=list)


class FileInfo(BaseModel):
    """Information about a repository file or directory."""

    path: str
    name: str
    type: str
    size: int | None = None
    content: str | None = None


class AnalysisResult(BaseModel):
    """Structured analysis for a repository."""

    repo: RepositoryInfo
    file_tree: list[FileInfo] = Field(default_factory=list)
    language_breakdown: dict[str, int] = Field(default_factory=dict)
    summary: str


class ChangeProposal(BaseModel):
    """A proposed set of repository changes."""

    title: str
    description: str
    files_to_modify: list[str] = Field(default_factory=list)
    reasoning: str
    checklist: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Top-level response returned by the coding agent."""

    answer: str
    proposals: list[ChangeProposal] = Field(default_factory=list)
    code_snippets: dict[str, str] = Field(default_factory=dict)
    session_id: str


class PRDraft(BaseModel):
    """A pull request draft generated from a change plan."""

    title: str
    body: str
    base_branch: str
    head_branch: str
    changed_files: list[str] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)


class ProviderRequest(BaseModel):
    """Normalized model-provider request payload."""

    messages: list[Message]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[Any] | None = None


class ProviderResponse(BaseModel):
    """Normalized model-provider response payload."""

    content: str
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str
