"""Agent state machine — tracks the agent's lifecycle and current context.

The :class:`AgentState` is a mutable snapshot of the agent's progress through
a task.  It is passed between the orchestrator and individual processing steps
to provide full observability without hidden global state.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    """Lifecycle phases of a single agent run."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentStep:
    """Record of a single reasoning / action step.

    Attributes:
        step_index: Zero-based position in the current run.
        thought: Model's internal reasoning text.
        action: Tool name or action identifier.
        action_input: Arguments passed to the action.
        observation: Result / output of the action.
        timestamp: Unix timestamp when the step was recorded.
    """

    step_index: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        parts = [f"Step {self.step_index}"]
        if self.thought:
            parts.append(f"  Thought: {self.thought[:120]}")
        if self.action:
            parts.append(f"  Action: {self.action}")
        if self.observation:
            parts.append(f"  Observation: {self.observation[:120]}")
        return "\n".join(parts)


@dataclass
class AgentState:
    """Complete snapshot of the agent's current run.

    Attributes:
        run_id: Unique identifier for this run (UUID4).
        task: Natural-language description of the objective.
        status: Current lifecycle phase.
        steps: Ordered list of reasoning steps taken so far.
        final_answer: The agent's concluded response (set on completion).
        metadata: Arbitrary key-value store for caller-defined context.
        created_at: Unix timestamp of run creation.
        finished_at: Unix timestamp of run completion (``None`` if running).
        error: Error message if status is ``FAILED``.
    """

    task: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: AgentStatus = AgentStatus.IDLE
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_step(
        self,
        thought: str = "",
        action: str = "",
        action_input: dict[str, Any] | None = None,
        observation: str = "",
    ) -> AgentStep:
        """Append a new step and return it.

        Args:
            thought: Model's reasoning text.
            action: Tool / action name.
            action_input: Arguments for the action.
            observation: Result of the action.

        Returns:
            The newly appended :class:`AgentStep`.
        """
        step = AgentStep(
            step_index=len(self.steps),
            thought=thought,
            action=action,
            action_input=action_input or {},
            observation=observation,
        )
        self.steps.append(step)
        return step

    def mark_complete(self, answer: str) -> None:
        """Transition to :attr:`AgentStatus.COMPLETE`.

        Args:
            answer: The final answer / result text.
        """
        self.final_answer = answer
        self.status = AgentStatus.COMPLETE
        self.finished_at = time.time()

    def mark_failed(self, error: str) -> None:
        """Transition to :attr:`AgentStatus.FAILED`.

        Args:
            error: Description of the failure.
        """
        self.error = error
        self.status = AgentStatus.FAILED
        self.finished_at = time.time()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since run creation."""
        end = self.finished_at or time.time()
        return end - self.created_at

    @property
    def iteration_count(self) -> int:
        """Number of steps taken so far."""
        return len(self.steps)

    def __str__(self) -> str:
        return (
            f"AgentState(run_id={self.run_id[:8]}, status={self.status.value}, "
            f"task={self.task[:50]!r}, steps={self.iteration_count})"
        )
