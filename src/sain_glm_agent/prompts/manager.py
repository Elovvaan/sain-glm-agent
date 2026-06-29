"""Prompt template management."""

from __future__ import annotations

import json
from typing import Any

from sain_glm_agent.domain.entities import Message


class PromptManager:
    """Builds system prompts and normalized message lists."""

    ASK_PROMPT = (
        "You are SAIN GLM Agent, a repository-aware coding assistant. Answer clearly, "
        "cite repository context when available, and prefer actionable guidance."
    )
    ANALYZE_PROMPT = (
        "You are SAIN GLM Agent, performing software architecture analysis. Identify "
        "structure, dominant technologies, important files, and likely maintenance risks."
    )
    PLAN_PROMPT = (
        "You are SAIN GLM Agent, planning code changes. Produce implementation guidance, "
        "highlight impacted files, and describe verification steps."
    )
    GENERATE_PROMPT = (
        "You are SAIN GLM Agent, generating high-quality production code. Favor correctness, "
        "clarity, and testability."
    )

    _PROMPTS = {
        "ask": ASK_PROMPT,
        "analyze": ANALYZE_PROMPT,
        "plan": PLAN_PROMPT,
        "generate": GENERATE_PROMPT,
    }

    def get_system_prompt(self, task_type: str) -> str:
        """Return the system prompt template for a task type."""
        key = task_type.strip().lower()
        try:
            return self._PROMPTS[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._PROMPTS))
            raise ValueError(
                f"Unsupported task type '{task_type}'. Supported types: {supported}."
            ) from exc

    def build_messages(
        self,
        system_prompt: str,
        user_query: str,
        context: Any | None = None,
    ) -> list[Message]:
        """Build a provider-ready message list from prompts and optional context."""
        content = user_query.strip()
        if context is not None:
            if isinstance(context, str):
                context_text = context
            else:
                context_text = json.dumps(context, indent=2, default=str, ensure_ascii=False)
            content = f"{content}\n\nRepository context:\n{context_text}"
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=content),
        ]
