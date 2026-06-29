"""Typer CLI for SAIN GLM Agent."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sain_glm_agent.application.agent import CodingAgent
from sain_glm_agent.infrastructure.config.settings import get_settings, validate_settings
from sain_glm_agent.infrastructure.github.repository import GitHubRepositoryService
from sain_glm_agent.infrastructure.logging.setup import setup_logging
from sain_glm_agent.infrastructure.memory.file_store import FileMemoryStore
from sain_glm_agent.infrastructure.providers.registry import create_provider_from_settings
from sain_glm_agent.prompts.manager import PromptManager
from sain_glm_agent.tools.repository import (
    GetFileContentTool,
    GetRepoInfoTool,
    ListFilesTool,
    ToolExecutionEngine,
)

load_dotenv()

app = typer.Typer(name="sain-agent", help="SAIN GLM Agent CLI")
console = Console()


def build_agent(require_provider: bool = True) -> CodingAgent:
    """Construct a fully configured agent instance for the CLI."""
    settings = get_settings()
    if require_provider:
        settings = validate_settings(settings)
    logger = setup_logging(settings.log_level)
    repo_service = GitHubRepositoryService(
        token=settings.github_token,
        base_url=settings.github_base_url,
    )
    provider = create_provider_from_settings(settings)
    memory_store = FileMemoryStore(Path(".sain_glm_agent_memory"))
    prompt_manager = PromptManager()
    tool_engine = ToolExecutionEngine(
        [
            GetRepoInfoTool(repo_service),
            ListFilesTool(repo_service),
            GetFileContentTool(repo_service),
        ]
    )
    return CodingAgent(
        settings=settings,
        provider=provider,
        repo_service=repo_service,
        memory_store=memory_store,
        prompt_manager=prompt_manager,
        tool_engine=tool_engine,
        logger=logger,
    )


@app.command()
def ask(repo: str, query: str, session_id: str | None = None) -> None:
    """Ask a repository-aware question."""
    try:
        agent = build_agent(require_provider=True)
        response = agent.ask(query=query, repo=repo, session_id=session_id or str(uuid4()))
        console.print(Panel(response.answer, title=f"Answer · session {response.session_id}"))
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def analyze(repo: str) -> None:
    """Analyze a repository structure and languages."""
    try:
        agent = build_agent(require_provider=False)
        result = agent.analyze(repo=repo)
        summary = Panel(result.summary, title=f"Analysis · {result.repo.owner}/{result.repo.name}")
        table = Table(title="Language Breakdown")
        table.add_column("Language / Extension")
        table.add_column("Count", justify="right")
        for language, count in result.language_breakdown.items():
            table.add_row(language, str(count))
        console.print(summary)
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def plan(repo: str, task: str, session_id: str | None = None) -> None:
    """Create a change plan for a repository task."""
    try:
        agent = build_agent(require_provider=True)
        response = agent.plan(task=task, repo=repo, session_id=session_id or str(uuid4()))
        console.print(Panel(response.answer, title=f"Plan · session {response.session_id}"))
        for proposal in response.proposals:
            checklist = "\n".join(f"- {item}" for item in proposal.checklist)
            files = "\n".join(f"- {item}" for item in proposal.files_to_modify) or "- None"
            console.print(
                Panel(
                    f"{proposal.description}\n\nFiles:\n{files}\n\nChecklist:\n{checklist}",
                    title=proposal.title,
                )
            )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover
    app()
