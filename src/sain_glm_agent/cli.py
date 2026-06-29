"""Command-line interface for SAIN GLM Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import RepositoryAssistant
from .config import Settings
from .exceptions import ConfigurationError, ProviderError, RepositoryError, ToolExecutionError
from .logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="SAIN GLM Agent repository assistant")
    parser.add_argument("command", choices=["analyze", "plan", "generate", "prepare-pr", "config"])
    parser.add_argument("task", nargs="?", help="Task or request for the agent")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path to inspect (defaults to current working directory)",
    )
    parser.add_argument("--provider", help="Override SAIN_PROVIDER for this invocation")
    parser.add_argument("--model", help="Override SAIN_MODEL for this invocation")
    parser.add_argument("--json", action="store_true", help="Emit responses as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.provider:
            settings.provider = args.provider
        if args.model:
            settings.model = args.model
        setup_logging(settings.log_level)
        if args.command == "config":
            print(json.dumps(settings.to_redacted_dict(), indent=2))
            return 0
        if not args.task:
            parser.error("the following arguments are required: task")
        assistant = RepositoryAssistant(Path(args.repo), settings)
        result = _dispatch_command(assistant, args.command, args.task)
        if args.json:
            print(
                json.dumps(
                    {
                        "action": result.action.value,
                        "provider": result.provider,
                        "model": result.model,
                        "usage": result.usage,
                        "content": result.content,
                    },
                    indent=2,
                )
            )
        else:
            print(result.content)
        return 0
    except (ConfigurationError, ProviderError, RepositoryError, ToolExecutionError) as error:
        parser.exit(status=2, message=f"error: {error}\n")


def _dispatch_command(assistant: RepositoryAssistant, command: str, task: str):
    if command == "analyze":
        return assistant.analyze_repository(task)
    if command == "plan":
        return assistant.plan_changes(task)
    if command == "generate":
        return assistant.generate_code(task)
    return assistant.prepare_pull_request(task)
