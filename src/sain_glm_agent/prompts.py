"""Prompt management for repository tasks."""

from __future__ import annotations

from string import Template

from .models import AgentAction


SYSTEM_TEMPLATE = Template(
    """
You are SAIN GLM Agent, a production-focused AI coding assistant for GitHub repositories.
Operate with strong separation of concerns, prioritize safe changes, and return concise, actionable output.
When code changes are requested, reason about architecture, affected files, validation steps, and pull-request preparation.
""".strip()
)

ACTION_TEMPLATES: dict[AgentAction, Template] = {
    AgentAction.ANALYZE: Template(
        """
Repository summary:
$repository_summary

Relevant file context:
$file_context

Recent conversation:
$recent_history

User request:
$task

Provide repository analysis, important findings, risks, and recommended next steps.
""".strip()
    ),
    AgentAction.PLAN: Template(
        """
Repository summary:
$repository_summary

Relevant file context:
$file_context

Recent conversation:
$recent_history

User request:
$task

Produce a step-by-step implementation plan with architecture considerations, validation strategy, and likely files to change.
""".strip()
    ),
    AgentAction.GENERATE: Template(
        """
Repository summary:
$repository_summary

Relevant file context:
$file_context

Recent conversation:
$recent_history

User request:
$task

Generate production-quality code guidance, including concrete file-level change recommendations and validation steps.
""".strip()
    ),
    AgentAction.PREPARE_PR: Template(
        """
Repository summary:
$repository_summary

Relevant file context:
$file_context

Current diff summary:
$diff_summary

Recent conversation:
$recent_history

User request:
$task

Draft a pull request summary with user-visible changes, testing notes, and rollout risks.
""".strip()
    ),
}


class PromptManager:
    """Render prompts for the configured agent actions."""

    def system_prompt(self) -> str:
        """Return the shared system prompt."""

        return SYSTEM_TEMPLATE.substitute()

    def render(
        self,
        action: AgentAction,
        *,
        repository_summary: str,
        file_context: str,
        recent_history: str,
        task: str,
        diff_summary: str = "No current diff.",
    ) -> str:
        """Render a prompt for a specific action."""

        return ACTION_TEMPLATES[action].substitute(
            repository_summary=repository_summary,
            file_context=file_context,
            recent_history=recent_history,
            task=task,
            diff_summary=diff_summary,
        )
