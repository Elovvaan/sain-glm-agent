"""Local tool execution helpers with command allow-listing."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .exceptions import ToolExecutionError


class ToolExecutor:
    """Run local commands with a strict allow-list and bounded execution."""

    def __init__(self, allowed_commands: Sequence[str]) -> None:
        self.allowed_commands = set(allowed_commands)

    def run(self, command: Sequence[str] | str, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        """Execute an allowed command and return the completed process."""

        parts = shlex.split(command) if isinstance(command, str) else list(command)
        if not parts:
            raise ToolExecutionError("Cannot execute an empty command.")
        binary = Path(parts[0]).name
        if binary not in self.allowed_commands:
            raise ToolExecutionError(
                f"Command '{binary}' is not allowed. Configure SAIN_ALLOWED_COMMANDS to permit it."
            )
        try:
            return subprocess.run(
                parts,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ToolExecutionError(f"Failed to execute command '{binary}': {error}") from error
