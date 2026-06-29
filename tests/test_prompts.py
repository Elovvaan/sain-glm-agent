"""Tests for prompt rendering."""

from __future__ import annotations

import unittest

from sain_glm_agent.models import AgentAction
from sain_glm_agent.prompts import PromptManager


class PromptManagerTests(unittest.TestCase):
    def test_render_includes_task_and_context(self) -> None:
        manager = PromptManager()
        rendered = manager.render(
            AgentAction.PLAN,
            repository_summary="repo summary",
            file_context="file context",
            recent_history="history",
            task="add provider",
            diff_summary="diff summary",
        )
        self.assertIn("repo summary", rendered)
        self.assertIn("file context", rendered)
        self.assertIn("add provider", rendered)
