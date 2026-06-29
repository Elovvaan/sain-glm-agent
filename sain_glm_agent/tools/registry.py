"""Tool registry — catalogue of callable tools available to the agent.

Tools are the agent's actuators: they let it read files, run code, search the
web, call APIs, etc.  Each tool is described by a :class:`Tool` dataclass so
the agent can communicate the available tools to the LLM in a structured way.

Adding a new tool::

    from sain_glm_agent.tools.registry import ToolRegistry, Tool

    registry = ToolRegistry()

    @registry.register_fn(
        name="read_file",
        description="Read a file from the local filesystem.",
        parameters={
            "path": {"type": "string", "description": "Absolute path to the file."}
        },
    )
    def read_file(path: str) -> str:
        return open(path).read()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """The outcome of a tool execution.

    Attributes:
        tool_name: Name of the tool that was called.
        success: Whether the tool ran without error.
        output: String output produced by the tool.
        error: Error message if the tool raised an exception.
    """

    tool_name: str
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"[Tool error — {self.tool_name}]: {self.error}"


@dataclass
class Tool:
    """Descriptor for a single callable tool.

    Attributes:
        name: Unique tool identifier (snake_case recommended).
        description: Short description shown to the model.
        parameters: JSON-Schema-like parameter definitions.
        fn: The Python callable implementing the tool.
        is_async: Whether *fn* is a coroutine.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    is_async: bool = False

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function schema for this tool.

        Returns:
            A dict suitable for the ``tools`` parameter of the chat API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


class ToolRegistry:
    """Mutable catalogue of tools available to the agent.

    Tools can be added via :meth:`register` (direct) or the
    :meth:`register_fn` decorator shorthand.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a :class:`Tool` instance.

        Args:
            tool: The tool to add.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_fn(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator that wraps a function as a :class:`Tool` and registers it.

        Args:
            name: Tool name.
            description: Short description.
            parameters: Parameter schema dict.

        Returns:
            Decorator that registers the wrapped function.
        """
        params = parameters or {}

        def decorator(fn: Callable) -> Callable:
            import asyncio

            is_async = asyncio.iscoroutinefunction(fn)
            tool = Tool(
                name=name,
                description=description,
                parameters=params,
                fn=fn,
                is_async=is_async,
            )
            self.register(tool)
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return a tool by name, or ``None`` if not found."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Return the OpenAI-style schema list for all registered tools."""
        return [t.to_schema() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)
