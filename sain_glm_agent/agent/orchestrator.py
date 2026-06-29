"""Agent orchestrator — the core reasoning loop of SAIN GLM Agent.

The orchestrator implements a **ReAct** (Reason + Act) loop:

1. Build a context message from the conversation history, available tools, and
   the current task.
2. Send the context to the model provider.
3. Parse the response for a *Final Answer* or a *Tool Call*.
4. If a tool call is found, execute it and add the result as an observation.
5. Repeat until a final answer is produced or the iteration limit is reached.

Usage::

    from sain_glm_agent.agent import AgentOrchestrator
    from sain_glm_agent.providers.glm import GLMProvider

    provider = GLMProvider(api_key="...", model="glm-4-flash")
    agent = AgentOrchestrator(provider=provider)
    state = agent.run("Explain the architecture of numpy.")
    print(state.final_answer)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

from sain_glm_agent.agent.state import AgentState, AgentStatus
from sain_glm_agent.memory.conversation import ConversationMemory
from sain_glm_agent.prompts.manager import PromptManager
from sain_glm_agent.providers.base import BaseProvider
from sain_glm_agent.tools.executor import ToolExecutor
from sain_glm_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Regex patterns for parsing model output
_THOUGHT_PATTERN = re.compile(
    r"(?:Thought|思考)[:\s]+(.*?)(?=\n(?:Action|Final Answer|最终答案)|$)",
    re.S,
)
_ACTION_PATTERN = re.compile(r"Action[:\s]+(\w+)", re.I)
_INPUT_PATTERN = re.compile(r"Action Input[:\s]+({.*?}|.*?)(?=\nObservation|\Z)", re.S)
_FINAL_PATTERN = re.compile(
    r"(?:Final Answer|最终答案)[:\s]+(.*)", re.S | re.I
)

# ReAct system prompt injected into the conversation
_REACT_SYSTEM_SUFFIX = """
## Reasoning format

You MUST follow this format for every response:

Thought: <your reasoning about the current situation>
Action: <tool_name>  (or "none" if no tool is needed)
Action Input: {{"param": "value"}}

When you have enough information to answer completely, respond with:
Final Answer: <your complete answer>

Available tools:
{tool_schemas}
"""


class AgentOrchestrator:
    """Core ReAct loop that drives the SAIN GLM Agent.

    Args:
        provider: The model provider to use for reasoning.
        tool_registry: Optional tool registry; an empty one is created if
            not supplied.
        memory: Optional conversation memory; a new one is created if not
            supplied.
        prompt_manager: Optional prompt manager; a new one with built-in
            templates is created if not supplied.
        max_iterations: Maximum reasoning steps before giving up.
        max_tokens: Token limit forwarded to the provider.
        temperature: Sampling temperature forwarded to the provider.
    """

    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        prompt_manager: PromptManager | None = None,
        max_iterations: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> None:
        self._provider = provider
        self._registry = tool_registry or ToolRegistry()
        self._memory = memory or ConversationMemory()
        self._prompts = prompt_manager or PromptManager()
        self._executor = ToolExecutor(self._registry)
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._setup_system_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str, *, reset_memory: bool = False) -> AgentState:
        """Execute the ReAct loop for the given task.

        Args:
            task: Natural-language description of what the agent should do.
            reset_memory: If ``True``, clear conversation history before
                starting (useful for fresh tasks).

        Returns:
            :class:`AgentState` with status ``COMPLETE`` or ``FAILED``.
        """
        if reset_memory:
            self._memory.clear()

        state = AgentState(task=task)
        state.status = AgentStatus.PLANNING
        logger.info("Agent run started | task=%s", task[:80])

        # Add the task as the first user message
        self._memory.add_user(task)

        for iteration in range(self._max_iterations):
            state.status = AgentStatus.EXECUTING
            logger.debug("Iteration %d/%d", iteration + 1, self._max_iterations)

            # ── Model call ───────────────────────────────────────────
            try:
                response = self._provider.chat(
                    self._memory.get_messages(),
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
            except Exception as exc:
                err = f"Provider error on iteration {iteration + 1}: {exc}"
                logger.error(err)
                state.mark_failed(err)
                return state

            raw_text = response.content.strip()
            self._memory.add_assistant(raw_text)

            logger.debug("Model response (%d tokens):\n%s", response.total_tokens, raw_text[:300])

            # ── Parse response ───────────────────────────────────────
            thought = self._extract_thought(raw_text)
            final_answer = self._extract_final_answer(raw_text)

            if final_answer:
                state.add_step(thought=thought, observation="[final answer]")
                state.mark_complete(final_answer)
                logger.info(
                    "Agent run complete after %d steps (%.2fs)",
                    state.iteration_count,
                    state.elapsed_seconds,
                )
                return state

            # ── Tool execution ───────────────────────────────────────
            action, action_input = self._extract_action(raw_text)

            if action and action != "none":
                tool_result = self._executor.run(action, action_input)
                observation = str(tool_result)
                logger.debug("Tool %s → %s", action, observation[:120])

                # Feed observation back into conversation
                obs_message = f"Observation: {observation}"
                self._memory.add_user(obs_message)

                state.add_step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            else:
                # Model produced no tool call and no final answer — nudge it
                state.add_step(thought=thought)
                nudge = (
                    "Please continue. Either call a tool or provide your "
                    "Final Answer."
                )
                self._memory.add_user(nudge)

        # Iteration limit reached
        last_answer = raw_text if raw_text else "No answer produced."
        state.mark_complete(last_answer)
        logger.warning(
            "Iteration limit (%d) reached. Using last model output.",
            self._max_iterations,
        )
        return state

    def add_tool(
        self,
        name: str,
        description: str,
        fn,
        parameters: dict | None = None,
    ) -> None:
        """Convenience method to register a tool and refresh the system prompt.

        Args:
            name: Tool name.
            description: Short description.
            fn: Callable implementing the tool.
            parameters: Optional parameter schema dict.
        """
        self._registry.register_fn(
            name=name,
            description=description,
            parameters=parameters or {},
        )(fn)
        self._setup_system_prompt()  # Refresh tool list in system prompt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_system_prompt(self) -> None:
        """Rebuild and set the system prompt with current tool schemas."""
        tool_schemas_str = (
            json.dumps(self._registry.schemas(), indent=2)
            if self._registry.all_tools()
            else "No tools available."
        )
        base = self._prompts.render(
            "system_coding_agent",
            current_date=date.today().isoformat(),
        )
        full_system = base + _REACT_SYSTEM_SUFFIX.format(tool_schemas=tool_schemas_str)
        self._memory.set_system(full_system)

    @staticmethod
    def _extract_thought(text: str) -> str:
        m = _THOUGHT_PATTERN.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        m = _FINAL_PATTERN.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_action(text: str) -> tuple[str, dict]:
        action_m = _ACTION_PATTERN.search(text)
        if not action_m:
            return "", {}
        action = action_m.group(1).strip()

        input_m = _INPUT_PATTERN.search(text)
        if input_m:
            raw = input_m.group(1).strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return action, parsed
            except json.JSONDecodeError:
                # Try to extract as plain string argument
                return action, {"input": raw}
        return action, {}
