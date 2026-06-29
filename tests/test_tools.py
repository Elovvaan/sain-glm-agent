"""Tests for sain_glm_agent.tools.registry and executor."""

from __future__ import annotations

import pytest

from sain_glm_agent.tools.executor import ToolExecutor
from sain_glm_agent.tools.registry import Tool, ToolRegistry, ToolResult

# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_str_success(self):
        r = ToolResult(tool_name="t", success=True, output="hello")
        assert str(r) == "hello"

    def test_str_failure(self):
        r = ToolResult(tool_name="t", success=False, error="oops")
        assert "oops" in str(r)
        assert "t" in str(r)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(name="add", description="Add two numbers", parameters={}, fn=lambda a, b: a + b)
        reg.register(tool)
        assert reg.get("add") is tool

    def test_duplicate_raises(self):
        reg = ToolRegistry()
        tool = Tool(name="x", description="", parameters={}, fn=lambda: None)
        reg.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(tool)

    def test_get_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_all_tools(self):
        reg = ToolRegistry()
        reg.register(Tool("a", "", {}, fn=lambda: None))
        reg.register(Tool("b", "", {}, fn=lambda: None))
        assert len(reg.all_tools()) == 2

    def test_names_sorted(self):
        reg = ToolRegistry()
        reg.register(Tool("z", "", {}, fn=lambda: None))
        reg.register(Tool("a", "", {}, fn=lambda: None))
        assert reg.names == ["a", "z"]

    def test_schemas(self):
        reg = ToolRegistry()
        reg.register(
            Tool(
                name="greet",
                description="Say hello.",
                parameters={"name": {"type": "string"}},
                fn=lambda name: f"Hello, {name}!",
            )
        )
        schemas = reg.schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"

    def test_register_fn_decorator(self):
        reg = ToolRegistry()

        @reg.register_fn(
            name="double",
            description="Double a number.",
            parameters={"n": {"type": "integer"}},
        )
        def double(n: int) -> int:
            return n * 2

        assert reg.get("double") is not None
        # Original function still works
        assert double(5) == 10


# ---------------------------------------------------------------------------
# Tool.to_schema
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_schema_structure(self):
        tool = Tool(
            name="search",
            description="Search the web.",
            parameters={"query": {"type": "string", "description": "Search query"}},
            fn=lambda query: "results",
        )
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def _make_executor(self):
        reg = ToolRegistry()
        reg.register(Tool("add", "Add", {"a": {}, "b": {}}, fn=lambda a, b: a + b))
        reg.register(Tool("fail", "Always fails", {}, fn=lambda: 1 / 0))
        reg.register(Tool("stringify", "Return dict", {}, fn=lambda: {"key": "val"}))
        return ToolExecutor(reg)

    def test_successful_run(self):
        ex = self._make_executor()
        result = ex.run("add", {"a": 3, "b": 4})
        assert result.success is True
        assert result.output == "7"

    def test_unknown_tool(self):
        ex = self._make_executor()
        result = ex.run("nonexistent", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_exception_captured(self):
        ex = self._make_executor()
        result = ex.run("fail", {})
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_dict_output_serialised(self):
        ex = self._make_executor()
        result = ex.run("stringify", {})
        assert result.success is True
        assert "key" in result.output

    def test_run_from_json_valid(self):
        ex = self._make_executor()
        result = ex.run_from_json("add", '{"a": 10, "b": 5}')
        assert result.success is True
        assert result.output == "15"

    def test_run_from_json_invalid(self):
        ex = self._make_executor()
        result = ex.run_from_json("add", "not json")
        assert result.success is False
        assert "Invalid JSON" in result.error

    @pytest.mark.asyncio
    async def test_run_async(self):
        reg = ToolRegistry()
        reg.register(Tool("echo", "Echo", {"msg": {}}, fn=lambda msg: msg))
        ex = ToolExecutor(reg)
        result = await ex.run_async("echo", {"msg": "hello"})
        assert result.success is True
        assert result.output == "hello"
