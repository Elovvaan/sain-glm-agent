"""Application orchestrator for repository-aware coding workflows."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from sain_glm_agent.domain.entities import (
    AgentResponse,
    AnalysisResult,
    ChangeProposal,
    Message,
    ProviderRequest,
    RepositoryInfo,
)
from sain_glm_agent.domain.interfaces import (
    BaseMemoryStore,
    BaseModelProvider,
    BaseRepositoryService,
)
from sain_glm_agent.prompts.manager import PromptManager
from sain_glm_agent.tools.repository import ToolExecutionEngine


class CodingAgent:
    """Coordinates repository context, prompts, memory, tools, and model providers."""

    def __init__(
        self,
        settings: Any,
        provider: BaseModelProvider,
        repo_service: BaseRepositoryService,
        memory_store: BaseMemoryStore,
        prompt_manager: PromptManager,
        tool_engine: ToolExecutionEngine,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.repo_service = repo_service
        self.memory_store = memory_store
        self.prompt_manager = prompt_manager
        self.tool_engine = tool_engine
        self.logger = logger

    def ask(self, query: str, repo: str, session_id: str | None = None) -> AgentResponse:
        """Answer a repository-aware question and persist the conversation."""
        active_session = session_id or str(uuid4())
        owner, name = self._parse_repo(repo)
        context = self._load_repo_context(owner, name)
        history = self.memory_store.get_history(active_session)
        system_prompt = self.prompt_manager.get_system_prompt("ask")
        messages = self.prompt_manager.build_messages(system_prompt, query, context)
        if history.messages:
            messages = [messages[0], *history.messages[-10:], messages[-1]]

        response = self.provider.complete(
            ProviderRequest(
                messages=messages,
                model=getattr(self.settings, "glm_model", "glm-4-flash"),
            )
        )
        self.memory_store.add_message(
            active_session,
            Message(role="user", content=query),
        )
        self.memory_store.add_message(
            active_session,
            Message(role="assistant", content=response.content),
        )
        self.logger.info(
            "Completed ask workflow",
            extra={"session_id": active_session, "repo": repo},
        )
        return AgentResponse(
            answer=response.content,
            proposals=[],
            code_snippets={},
            session_id=active_session,
        )

    def analyze(self, repo: str, session_id: str | None = None) -> AnalysisResult:
        """Perform a repository analysis and generate a concise summary."""
        _ = session_id or str(uuid4())
        owner, name = self._parse_repo(repo)
        repo_info = self.repo_service.get_repo_info(owner, name)
        file_tree = self.repo_service.get_directory_tree(owner, name)
        language_breakdown = self._build_language_breakdown(file_tree)
        summary = self._build_analysis_summary(repo_info, file_tree, language_breakdown)
        self.logger.info("Completed analyze workflow", extra={"repo": repo})
        return AnalysisResult(
            repo=repo_info,
            file_tree=file_tree,
            language_breakdown=language_breakdown,
            summary=summary,
        )

    def plan(self, task: str, repo: str, session_id: str | None = None) -> AgentResponse:
        """Plan a repository change and return a structured proposal."""
        active_session = session_id or str(uuid4())
        analysis = self.analyze(repo, active_session)
        system_prompt = self.prompt_manager.get_system_prompt("plan")
        context = {
            "repository": analysis.repo.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in analysis.file_tree[:50]],
            "language_breakdown": analysis.language_breakdown,
        }
        messages = self.prompt_manager.build_messages(system_prompt, task, context)
        response = self.provider.complete(
            ProviderRequest(
                messages=messages,
                model=getattr(self.settings, "glm_model", "glm-4-flash"),
            )
        )
        files_to_modify = self._recommend_files(task, analysis.file_tree)
        checklist = self._build_checklist(task, files_to_modify)
        proposal = ChangeProposal(
            title=f"Plan for: {task}",
            description=response.content,
            files_to_modify=files_to_modify,
            reasoning="Generated from repository structure and model-assisted planning.",
            checklist=checklist,
        )
        self.memory_store.add_message(
            active_session,
            Message(role="user", content=task),
        )
        self.memory_store.add_message(
            active_session,
            Message(role="assistant", content=response.content),
        )
        self.logger.info(
            "Completed plan workflow",
            extra={"session_id": active_session, "repo": repo},
        )
        return AgentResponse(
            answer=response.content,
            proposals=[proposal],
            code_snippets={},
            session_id=active_session,
        )

    def _parse_repo(self, repo: str) -> tuple[str, str]:
        if "/" not in repo:
            raise ValueError("Repository must be in the format 'owner/name'.")
        owner, name = repo.split("/", 1)
        if not owner or not name:
            raise ValueError("Repository must include both owner and name.")
        return owner, name

    def _load_repo_context(self, owner: str, name: str) -> dict[str, Any]:
        repo_info = self.tool_engine.execute("get_repo_info", {"owner": owner, "name": name})
        file_listing = self.tool_engine.execute(
            "list_files", {"owner": owner, "name": name, "path": "", "ref": None}
        )
        return {
            "repo": repo_info.get("result", {}),
            "top_level_files": file_listing.get("result", {}),
        }

    def _build_language_breakdown(self, file_tree: list[Any]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for item in file_tree:
            if item.type != "file":
                continue
            suffix = Path(item.path).suffix.lower().lstrip(".") or "unknown"
            counter[suffix] += 1
        return dict(sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])))

    def _build_analysis_summary(
        self,
        repo_info: RepositoryInfo,
        file_tree: list[Any],
        language_breakdown: dict[str, int],
    ) -> str:
        default_summary = (
            f"Repository {repo_info.owner}/{repo_info.name} contains "
            f"{len(file_tree)} tracked paths. "
            f"Primary file types: "
            f"{', '.join(list(language_breakdown)[:5]) or 'unknown'}."
        )
        try:
            if not getattr(self.settings, "glm_api_key", None):
                return default_summary
            system_prompt = self.prompt_manager.get_system_prompt("analyze")
            context = {
                "repository": repo_info.model_dump(mode="json"),
                "file_count": len(file_tree),
                "sample_files": [item.model_dump(mode="json") for item in file_tree[:50]],
                "language_breakdown": language_breakdown,
            }
            messages = self.prompt_manager.build_messages(
                system_prompt,
                "Provide a concise architectural analysis of this repository.",
                context,
            )
            response = self.provider.complete(
                ProviderRequest(
                    messages=messages,
                    model=getattr(self.settings, "glm_model", "glm-4-flash"),
                    temperature=0.3,
                    max_tokens=512,
                )
            )
            return response.content
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Falling back to static analysis summary: %s", exc)
            return default_summary

    def _recommend_files(self, task: str, file_tree: list[Any]) -> list[str]:
        keywords = {
            chunk.lower()
            for chunk in task.replace("/", " ").replace("-", " ").split()
            if chunk
        }
        matched = [
            item.path
            for item in file_tree
            if item.type == "file" and any(keyword in item.path.lower() for keyword in keywords)
        ]
        if matched:
            return matched[:10]
        fallback = [item.path for item in file_tree if item.type == "file"]
        return fallback[:10]

    def _build_checklist(self, task: str, files_to_modify: list[str]) -> list[str]:
        file_focus = files_to_modify[:3]
        checklist = [
            f"Confirm requirements for: {task}",
            "Review repository constraints and dependencies",
            "Implement and validate the necessary changes",
            "Add or update automated tests for the change",
        ]
        if file_focus:
            checklist.insert(2, f"Update key files: {', '.join(file_focus)}")
        return checklist
