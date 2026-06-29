"""Tests for sain_glm_agent.agent.state and orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

from sain_glm_agent.agent.orchestrator import AgentOrchestrator
from sain_glm_agent.agent.state import AgentState, AgentStatus
from sain_glm_agent.providers.base import ModelResponse

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_initial_status(self):
        state = AgentState(task="test task")
        assert state.status == AgentStatus.IDLE
        assert state.iteration_count == 0
        assert state.final_answer is None

    def test_add_step(self):
        state = AgentState(task="t")
        step = state.add_step(thought="thinking", action="tool", observation="result")
        assert step.step_index == 0
        assert state.iteration_count == 1

    def test_mark_complete(self):
        state = AgentState(task="t")
        state.mark_complete("The answer is 42.")
        assert state.status == AgentStatus.COMPLETE
        assert state.final_answer == "The answer is 42."
        assert state.finished_at is not None

    def test_mark_failed(self):
        state = AgentState(task="t")
        state.mark_failed("Something went wrong")
        assert state.status == AgentStatus.FAILED
        assert "Something went wrong" in state.error

    def test_elapsed_seconds(self):
        import time

        state = AgentState(task="t")
        time.sleep(0.01)
        assert state.elapsed_seconds > 0

    def test_str_representation(self):
        state = AgentState(task="Do something useful")
        s = str(state)
        assert "AgentState" in s
        assert "idle" in s

    def test_metadata_store(self):
        state = AgentState(task="t")
        state.metadata["repo"] = "owner/repo"
        assert state.metadata["repo"] == "owner/repo"


# ---------------------------------------------------------------------------
# AgentOrchestrator — uses a mock provider
# ---------------------------------------------------------------------------


def _make_mock_provider(responses: list[str]) -> MagicMock:
    """Create a mock provider that returns successive responses."""
    provider = MagicMock()
    provider.model = "mock-model"

    side_effects = [
        ModelResponse(content=r, model="mock-model", provider="mock")
        for r in responses
    ]
    provider.chat.side_effect = side_effects
    return provider


class TestAgentOrchestrator:
    def test_direct_final_answer(self):
        """Provider immediately returns a Final Answer."""
        provider = _make_mock_provider(
            ["Thought: I know the answer\nFinal Answer: The answer is 42."]
        )
        agent = AgentOrchestrator(provider=provider)
        state = agent.run("What is 6 × 7?", reset_memory=True)

        assert state.status == AgentStatus.COMPLETE
        assert "42" in state.final_answer

    def test_tool_call_then_final_answer(self):
        """Provider calls a tool once, then returns a Final Answer."""
        provider = _make_mock_provider(
            [
                'Thought: I need to calculate\nAction: add\nAction Input: {"a": 3, "b": 4}',
                "Thought: I have the result\nFinal Answer: The sum is 7.",
            ]
        )
        agent = AgentOrchestrator(provider=provider)

        from sain_glm_agent.tools.registry import Tool

        agent._registry.register(
            Tool("add", "Add two numbers", {"a": {}, "b": {}}, fn=lambda a, b: a + b)
        )

        state = agent.run("What is 3 + 4?", reset_memory=True)

        assert state.status == AgentStatus.COMPLETE
        assert "7" in state.final_answer

    def test_iteration_limit(self):
        """Agent should stop and use last output when limit is reached."""
        # Provider never returns a Final Answer
        never_ends = ["Thought: still thinking"] * 5
        provider = _make_mock_provider(never_ends)
        agent = AgentOrchestrator(provider=provider, max_iterations=3)
        state = agent.run("Loop forever", reset_memory=True)

        # Should be COMPLETE (limit hit) with the last response as answer
        assert state.status == AgentStatus.COMPLETE
        assert state.iteration_count <= 3

    def test_provider_error_returns_failed(self):
        """A provider exception should set status to FAILED."""
        provider = MagicMock()
        provider.model = "mock-model"
        provider.chat.side_effect = RuntimeError("API down")

        agent = AgentOrchestrator(provider=provider)
        state = agent.run("Do something", reset_memory=True)

        assert state.status == AgentStatus.FAILED
        assert state.error is not None

    def test_reset_memory(self):
        """reset_memory=True should clear history before the run."""
        provider = _make_mock_provider(
            ["Final Answer: Done."] * 2
        )
        agent = AgentOrchestrator(provider=provider)
        agent.run("First task", reset_memory=True)
        agent.run("Second task", reset_memory=True)

        # Both runs should have been called
        assert provider.chat.call_count == 2

    def test_add_tool_refreshes_system_prompt(self):
        """add_tool should register the tool without crashing."""
        provider = _make_mock_provider(["Final Answer: ok"])
        agent = AgentOrchestrator(provider=provider)
        agent.add_tool("ping", "Ping the system.", lambda: "pong")
        assert agent._registry.get("ping") is not None
