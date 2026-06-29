"""Tests for assistant orchestration."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sain_glm_agent.agent import RepositoryAssistant
from sain_glm_agent.config import Settings
from sain_glm_agent.memory import ConversationMemory
from sain_glm_agent.models import ModelRequest, ModelResponse
from sain_glm_agent.providers.base import ModelProvider


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(provider=self.name, model=request.model, content="done")


class RepositoryAssistantTests(unittest.TestCase):
    def test_plan_changes_updates_memory_and_calls_provider(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            settings = Settings(api_key="unused", data_dir=root / ".sain")
            settings.ensure_directories()
            provider = FakeProvider()
            memory = ConversationMemory(settings.memory_file)
            assistant = RepositoryAssistant(root, settings, provider=provider, memory=memory)
            result = assistant.plan_changes("add CLI")
        self.assertEqual(result.content, "done")
        self.assertEqual(provider.requests[0].metadata["action"], "plan")
        self.assertEqual(memory.turns[-1].content, "done")
