from __future__ import annotations

from types import SimpleNamespace

from sain_glm_agent.application.agent import CodingAgent
from sain_glm_agent.domain.entities import FileInfo, ProviderResponse, RepositoryInfo
from sain_glm_agent.infrastructure.logging.setup import setup_logging
from sain_glm_agent.infrastructure.memory.inmemory import InMemoryStore
from sain_glm_agent.prompts.manager import PromptManager
from sain_glm_agent.tools.repository import GetRepoInfoTool, ListFilesTool, ToolExecutionEngine


class MockProvider:
    def complete(self, request):
        return ProviderResponse(
            content=f"Processed: {request.messages[-1].content[:30]}",
            model=request.model,
            usage={"prompt_tokens": 1},
            finish_reason="stop",
        )

    def get_capabilities(self) -> dict:
        return {"provider": "mock"}


class MockRepoService:
    def get_repo_info(self, owner: str, name: str) -> RepositoryInfo:
        return RepositoryInfo(
            owner=owner,
            name=name,
            description="Demo repo",
            language="Python",
            default_branch="main",
            stars=10,
            topics=["demo"],
        )

    def list_files(self, owner: str, name: str, path: str = "", ref: str | None = None):
        return [
            FileInfo(path="README.md", name="README.md", type="file", size=100),
            FileInfo(path="src", name="src", type="dir"),
            FileInfo(path="src/app.py", name="app.py", type="file", size=200),
        ]

    def get_file_content(self, owner: str, name: str, path: str, ref: str | None = None) -> str:
        return "print('hello')"

    def get_directory_tree(self, owner: str, name: str, path: str = "", ref: str | None = None):
        return self.list_files(owner, name, path=path, ref=ref)


def build_agent() -> CodingAgent:
    repo_service = MockRepoService()
    tool_engine = ToolExecutionEngine([GetRepoInfoTool(repo_service), ListFilesTool(repo_service)])
    return CodingAgent(
        settings=SimpleNamespace(glm_model="glm-4-flash", glm_api_key="secret"),
        provider=MockProvider(),
        repo_service=repo_service,
        memory_store=InMemoryStore(),
        prompt_manager=PromptManager(),
        tool_engine=tool_engine,
        logger=setup_logging("DEBUG"),
    )


def test_ask_updates_memory() -> None:
    agent = build_agent()
    response = agent.ask("What files exist?", "owner/repo", session_id="s1")
    history = agent.memory_store.get_history("s1")
    assert response.session_id == "s1"
    assert len(history.messages) == 2
    assert history.messages[0].content == "What files exist?"


def test_analyze_returns_summary_and_breakdown() -> None:
    agent = build_agent()
    result = agent.analyze("owner/repo", session_id="s2")
    assert result.repo.name == "repo"
    assert result.language_breakdown["py"] == 1
    assert result.summary


def test_plan_returns_proposal() -> None:
    agent = build_agent()
    response = agent.plan("Update the Python app", "owner/repo", session_id="s3")
    assert response.proposals
    assert response.proposals[0].files_to_modify
    assert response.session_id == "s3"
