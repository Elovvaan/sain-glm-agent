from __future__ import annotations

from typer.testing import CliRunner

from sain_glm_agent.domain.entities import (
    AgentResponse,
    AnalysisResult,
    ChangeProposal,
    FileInfo,
    RepositoryInfo,
)
from sain_glm_agent.interfaces.cli.main import app

runner = CliRunner()


class StubAgent:
    def ask(self, query: str, repo: str, session_id: str | None = None) -> AgentResponse:
        return AgentResponse(
            answer=f"Asked about {repo}: {query}",
            proposals=[],
            code_snippets={},
            session_id=session_id or "s1",
        )

    def analyze(self, repo: str) -> AnalysisResult:
        owner, name = repo.split("/", 1)
        return AnalysisResult(
            repo=RepositoryInfo(owner=owner, name=name, default_branch="main", stars=0, topics=[]),
            file_tree=[FileInfo(path="README.md", name="README.md", type="file")],
            language_breakdown={"md": 1},
            summary="Repository summary",
        )

    def plan(self, task: str, repo: str, session_id: str | None = None) -> AgentResponse:
        proposal = ChangeProposal(
            title="Plan",
            description=f"Plan for {task}",
            files_to_modify=["README.md"],
            reasoning="Because it is relevant.",
            checklist=["Do the work"],
        )
        return AgentResponse(
            answer="Plan created",
            proposals=[proposal],
            code_snippets={},
            session_id=session_id or "s2",
        )


def test_cli_ask(monkeypatch) -> None:
    monkeypatch.setattr(
        "sain_glm_agent.interfaces.cli.main.build_agent",
        lambda require_provider=True: StubAgent(),
    )
    result = runner.invoke(app, ["ask", "owner/repo", "What is this?"])
    assert result.exit_code == 0
    assert "Asked about owner/repo" in result.stdout


def test_cli_analyze(monkeypatch) -> None:
    monkeypatch.setattr(
        "sain_glm_agent.interfaces.cli.main.build_agent",
        lambda require_provider=False: StubAgent(),
    )
    result = runner.invoke(app, ["analyze", "owner/repo"])
    assert result.exit_code == 0
    assert "Repository summary" in result.stdout
    assert "md" in result.stdout


def test_cli_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        "sain_glm_agent.interfaces.cli.main.build_agent",
        lambda require_provider=True: StubAgent(),
    )
    result = runner.invoke(app, ["plan", "owner/repo", "Add docs"])
    assert result.exit_code == 0
    assert "Plan created" in result.stdout
    assert "README.md" in result.stdout


def test_cli_handles_errors(monkeypatch) -> None:
    def broken_build_agent(require_provider=True):
        raise ValueError("Missing GLM_API_KEY")

    monkeypatch.setattr("sain_glm_agent.interfaces.cli.main.build_agent", broken_build_agent)
    result = runner.invoke(app, ["ask", "owner/repo", "Hi"])
    assert result.exit_code == 1
    assert "Missing GLM_API_KEY" in result.stdout
