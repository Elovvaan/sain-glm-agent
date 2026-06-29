"""Agent orchestration for repository-focused reasoning tasks."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .memory import ConversationMemory
from .models import AgentAction, AgentResult, GenerationConfig, Message, ModelRequest
from .prompts import PromptManager
from .providers import ModelProvider, build_default_registry
from .repository import RepositoryAnalyzer
from .tools import ToolExecutor


class RepositoryAssistant:
    """High-level repository assistant powered by a pluggable model provider."""

    def __init__(
        self,
        repository_root: Path,
        settings: Settings,
        provider: ModelProvider | None = None,
        memory: ConversationMemory | None = None,
        prompts: PromptManager | None = None,
        tools: ToolExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.repository_root = repository_root.resolve()
        self.provider = provider or build_default_registry().create(settings)
        self.memory = memory or ConversationMemory(self.settings.memory_file)
        self.prompts = prompts or PromptManager()
        self.tools = tools or ToolExecutor(self.settings.allowed_commands)
        self.repository = RepositoryAnalyzer(self.repository_root, self.settings)

    def analyze_repository(self, task: str) -> AgentResult:
        """Analyze the repository in the context of the task."""

        return self._run_action(AgentAction.ANALYZE, task)

    def plan_changes(self, task: str) -> AgentResult:
        """Produce an implementation plan for the requested change."""

        return self._run_action(AgentAction.PLAN, task)

    def generate_code(self, task: str) -> AgentResult:
        """Generate code-oriented guidance for the requested change."""

        return self._run_action(AgentAction.GENERATE, task)

    def prepare_pull_request(self, task: str) -> AgentResult:
        """Draft a pull-request summary from the current repository state."""

        return self._run_action(AgentAction.PREPARE_PR, task)

    def _run_action(self, action: AgentAction, task: str) -> AgentResult:
        snapshot = self.repository.scan()
        repository_summary = snapshot.to_prompt()
        file_context = self.repository.build_file_context(task, snapshot=snapshot)
        diff_summary = self.repository.prepare_pull_request_context()
        recent_history = self.memory.render_recent_history()
        prompt = self.prompts.render(
            action,
            repository_summary=repository_summary,
            file_context=file_context,
            recent_history=recent_history,
            task=task,
            diff_summary=diff_summary,
        )
        request = ModelRequest(
            model=self.settings.model,
            system_prompt=self.prompts.system_prompt(),
            messages=[Message(role="user", content=prompt)],
            generation=GenerationConfig(
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            ),
            metadata={"action": action.value, "repository_root": str(self.repository_root)},
        )
        response = self.provider.generate(request)
        self.memory.add_turn("user", f"{action.value}: {task}")
        self.memory.add_turn("assistant", response.content)
        return AgentResult(
            action=action,
            provider=response.provider,
            model=response.model,
            repository_summary=repository_summary,
            content=response.content,
            usage=response.usage,
        )
