"""Tool executor — safely invokes registered tools and captures their output.

The executor looks up a tool from the :class:`~sain_glm_agent.tools.registry.ToolRegistry`,
calls it with the provided arguments, captures any exceptions, and returns a
structured :class:`~sain_glm_agent.tools.registry.ToolResult`.

Usage::

    from sain_glm_agent.tools import ToolRegistry, ToolExecutor

    registry = ToolRegistry()

    @registry.register_fn("greet", "Say hello.", {"name": {"type": "string"}})
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    executor = ToolExecutor(registry)
    result = executor.run("greet", {"name": "World"})
    print(result.output)   # Hello, World!
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sain_glm_agent.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools from a :class:`ToolRegistry`, capturing errors gracefully.

    Args:
        registry: The tool catalogue to draw from.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Look up and invoke a synchronous tool.

        Args:
            tool_name: Name of the registered tool to call.
            arguments: Keyword arguments to pass to the tool function.

        Returns:
            :class:`ToolResult` with either the output or a captured error.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: '{tool_name}'. "
                f"Available: {self._registry.names}",
            )

        logger.debug("Executing tool: %s(%s)", tool_name, _fmt_args(arguments))
        try:
            if tool.is_async:
                output = asyncio.run(tool.fn(**arguments))
            else:
                output = tool.fn(**arguments)
            result_str = _to_str(output)
            logger.debug("Tool %s succeeded (%d chars)", tool_name, len(result_str))
            return ToolResult(tool_name=tool_name, success=True, output=result_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %s raised %s: %s", tool_name, type(exc).__name__, exc)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def run_async(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Look up and invoke a tool, with native async support.

        Args:
            tool_name: Name of the registered tool to call.
            arguments: Keyword arguments to pass to the tool function.

        Returns:
            :class:`ToolResult` with either the output or a captured error.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: '{tool_name}'",
            )

        logger.debug("Async-executing tool: %s(%s)", tool_name, _fmt_args(arguments))
        try:
            if tool.is_async:
                output = await tool.fn(**arguments)
            else:
                output = await asyncio.to_thread(tool.fn, **arguments)
            result_str = _to_str(output)
            return ToolResult(tool_name=tool_name, success=True, output=result_str)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_from_json(self, tool_name: str, arguments_json: str) -> ToolResult:
        """Convenience wrapper that parses a JSON argument string first.

        Args:
            tool_name: Registered tool name.
            arguments_json: JSON-encoded argument dict (as produced by LLM
                tool-call responses).

        Returns:
            :class:`ToolResult`.
        """
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Invalid JSON arguments: {exc}",
            )
        return self.run(tool_name, arguments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_str(value: Any) -> str:
    """Convert a tool return value to a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _fmt_args(args: dict[str, Any]) -> str:
    """Format arguments dict for debug logging (truncated)."""
    items = ", ".join(f"{k}={v!r}"[:40] for k, v in args.items())
    return items[:120]
