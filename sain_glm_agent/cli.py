"""Command-line interface for SAIN GLM Agent.

Entry point: ``sain-agent`` (configured in ``pyproject.toml``).

Sub-commands
------------
``chat``
    Interactive chat session with the configured model provider.
``run``
    Execute a single task and print the result.
``repo``
    GitHub repository assistant sub-commands (``analyse``, ``pr``).
``info``
    Print current configuration (redacting secrets).

Usage examples::

    sain-agent chat
    sain-agent run "Summarise the purpose of numpy"
    sain-agent repo analyse owner/repo --objective "Add pagination"
    sain-agent info
"""

from __future__ import annotations

import sys

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_agent():
    """Build and return a configured :class:`AgentOrchestrator`."""
    from sain_glm_agent.agent.orchestrator import AgentOrchestrator
    from sain_glm_agent.config import get_settings
    from sain_glm_agent.logging_ import setup_logging
    from sain_glm_agent.providers import ProviderFactory

    cfg = get_settings()
    setup_logging(
        level=cfg.log_level,
        log_format=cfg.log_format.value,
        log_file=cfg.log_file,
    )

    try:
        provider = ProviderFactory.create(cfg)
    except (ValueError, ImportError) as exc:
        click.echo(click.style(f"Provider error: {exc}", fg="red"), err=True)
        sys.exit(1)

    return AgentOrchestrator(
        provider=provider,
        max_iterations=cfg.max_iterations,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


def _print_state(state) -> None:
    """Pretty-print an :class:`AgentState`."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        answer = state.final_answer or state.error or "No output."
        panel = Panel(
            answer,
            title=f"[bold green]Result[/bold green] — {state.status.value}",
            border_style="green" if state.final_answer else "red",
            subtitle=f"{state.iteration_count} step(s) · {state.elapsed_seconds:.1f}s",
        )
        console.print(panel)
    except ImportError:
        click.echo(state.final_answer or state.error or "No output.")


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="sain-glm-agent", prog_name="sain-agent")
def main() -> None:
    """SAIN GLM Agent — AI-powered software engineering assistant."""


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@main.command()
@click.option("--stream/--no-stream", default=False, help="Stream tokens to stdout.")
def chat(stream: bool) -> None:
    """Start an interactive chat session."""
    from sain_glm_agent.agent.orchestrator import AgentOrchestrator
    from sain_glm_agent.config import get_settings
    from sain_glm_agent.logging_ import setup_logging
    from sain_glm_agent.providers import ProviderFactory

    cfg = get_settings()
    setup_logging(level=cfg.log_level, log_format=cfg.log_format.value, log_file=cfg.log_file)

    try:
        provider = ProviderFactory.create(cfg)
    except (ValueError, ImportError) as exc:
        click.echo(click.style(f"Provider error: {exc}", fg="red"), err=True)
        sys.exit(1)

    agent = AgentOrchestrator(
        provider=provider,
        max_iterations=cfg.max_iterations,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    click.echo(click.style("SAIN GLM Agent — interactive chat", bold=True))
    click.echo(f"Provider: {provider}  |  Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            user_input = click.prompt(click.style("You", fg="cyan"))
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            click.echo("Goodbye!")
            break

        if stream:
            from sain_glm_agent.providers.base import Message, MessageRole

            click.echo(click.style("Agent: ", fg="green"), nl=False)
            try:
                for chunk in provider.stream_chat(
                    [Message(MessageRole.USER, user_input)]
                ):
                    click.echo(chunk, nl=False)
                click.echo()
            except Exception as exc:  # noqa: BLE001
                click.echo(click.style(f"\nError: {exc}", fg="red"), err=True)
        else:
            state = agent.run(user_input)
            answer = state.final_answer or state.error or "No response."
            click.echo(click.style("Agent: ", fg="green") + answer + "\n")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@main.command()
@click.argument("task")
@click.option("--verbose", "-v", is_flag=True, help="Show reasoning steps.")
def run(task: str, verbose: bool) -> None:
    """Execute a single TASK and print the result."""
    agent = _get_agent()
    state = agent.run(task, reset_memory=True)

    if verbose:
        for step in state.steps:
            click.echo(click.style(str(step), dim=True))

    _print_state(state)
    sys.exit(0 if state.final_answer else 1)


# ---------------------------------------------------------------------------
# repo
# ---------------------------------------------------------------------------


@main.group()
def repo() -> None:
    """GitHub repository assistant commands."""


@repo.command("analyse")
@click.argument("repo_name")
@click.option("--objective", "-o", default="", help="Task objective for the analysis.")
@click.option("--max-files", default=200, show_default=True, help="Max files to scan.")
def repo_analyse(repo_name: str, objective: str, max_files: int) -> None:
    """Analyse REPO_NAME (owner/repo) and print a structured report."""
    from sain_glm_agent.config import get_settings
    from sain_glm_agent.logging_ import setup_logging
    from sain_glm_agent.repository import GitHubClient, RepositoryAnalyzer

    cfg = get_settings()
    setup_logging(level=cfg.log_level, log_format=cfg.log_format.value)

    if not cfg.github_token:
        click.echo(click.style("GITHUB_TOKEN is not set.", fg="red"), err=True)
        sys.exit(1)

    client = GitHubClient(token=cfg.github_token)
    analyser = RepositoryAnalyzer(client)

    click.echo(f"Analysing {repo_name} …")
    try:
        analysis = analyser.analyse(repo_name, objective=objective, max_files=max_files)
    except Exception as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    click.echo("\n" + analysis.to_context_str())


@repo.command("pr")
@click.argument("repo_name")
@click.option("--task", "-t", required=True, help="Describe the change to implement.")
@click.option("--branch", "-b", default="", help="Branch name (auto-generated if omitted).")
@click.option("--base", default="main", show_default=True, help="Base branch.")
@click.option("--dry-run", is_flag=True, help="Plan only — do not push or open PR.")
def repo_pr(
    repo_name: str,
    task: str,
    branch: str,
    base: str,
    dry_run: bool,
) -> None:
    """Open a pull request on REPO_NAME to accomplish TASK."""
    import time as _time

    from sain_glm_agent.config import get_settings
    from sain_glm_agent.logging_ import setup_logging
    from sain_glm_agent.repository import GitHubClient, RepositoryAnalyzer

    cfg = get_settings()
    setup_logging(level=cfg.log_level, log_format=cfg.log_format.value)

    if not cfg.github_token:
        click.echo(click.style("GITHUB_TOKEN is not set.", fg="red"), err=True)
        sys.exit(1)

    client = GitHubClient(token=cfg.github_token)
    analyser = RepositoryAnalyzer(client)

    click.echo(f"Analysing {repo_name} …")
    try:
        analysis = analyser.analyse(repo_name, objective=task)
    except Exception as exc:
        click.echo(click.style(f"Error analysing repo: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Build context for the agent
    context = analysis.to_context_str()
    full_task = f"{task}\n\nRepository context:\n{context}"

    click.echo("Running agent …")
    agent = _get_agent()
    state = agent.run(full_task, reset_memory=True)

    if dry_run:
        click.echo(click.style("\n[Dry run — no changes pushed]\n", fg="yellow"))
        _print_state(state)
        return

    if not state.final_answer:
        click.echo(click.style("Agent did not produce an answer.", fg="red"), err=True)
        sys.exit(1)

    # Create branch + PR
    branch_name = branch or f"sain-agent/{_time.strftime('%Y%m%d-%H%M%S')}"
    try:
        gh_repo = client.get_repo(repo_name)
        client.create_branch(gh_repo, branch_name, base_branch=base)
        pr = client.create_pull_request(
            gh_repo,
            title=f"[SAIN Agent] {task[:72]}",
            body=state.final_answer,
            head_branch=branch_name,
            base_branch=base,
        )
        click.echo(click.style(f"\n✓ PR #{pr.number} opened: {pr.html_url}", fg="green"))
    except Exception as exc:
        click.echo(click.style(f"GitHub error: {exc}", fg="red"), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command()
def info() -> None:
    """Print current configuration (secrets are redacted)."""
    from sain_glm_agent.config import get_settings

    cfg = get_settings()

    def _redact(v: str | None) -> str:
        if not v:
            return "(not set)"
        return v[:4] + "****" + v[-2:] if len(v) > 8 else "****"

    rows = [
        ("Provider", cfg.provider.value),
        ("GLM model", cfg.glm_model),
        ("ZhipuAI key", _redact(cfg.zhipuai_api_key)),
        ("GitHub token", _redact(cfg.github_token)),
        ("Max tokens", str(cfg.max_tokens)),
        ("Temperature", str(cfg.temperature)),
        ("Max iterations", str(cfg.max_iterations)),
        ("Log level", cfg.log_level),
        ("Log format", cfg.log_format.value),
    ]

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="SAIN GLM Agent — Configuration", show_header=True)
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value")
        for k, v in rows:
            table.add_row(k, v)
        Console().print(table)
    except ImportError:
        for k, v in rows:
            click.echo(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
