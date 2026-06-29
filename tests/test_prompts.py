from __future__ import annotations

import pytest

from sain_glm_agent.prompts.manager import PromptManager


def test_get_system_prompt() -> None:
    manager = PromptManager()
    assert "repository-aware" in manager.get_system_prompt("ask")
    assert "planning code changes" in manager.get_system_prompt("plan")


def test_get_system_prompt_invalid() -> None:
    manager = PromptManager()
    with pytest.raises(ValueError, match="Unsupported task type"):
        manager.get_system_prompt("unknown")


def test_build_messages_includes_context() -> None:
    manager = PromptManager()
    messages = manager.build_messages("sys", "question", {"repo": "demo"})
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "Repository context" in messages[1].content
    assert '"demo"' in messages[1].content
