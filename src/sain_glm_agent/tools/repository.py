"""Repository read tools and execution engine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sain_glm_agent.domain.interfaces import BaseRepositoryService
from sain_glm_agent.tools.base import BaseTool


class GetRepoInfoTool(BaseTool):
    """Return repository metadata."""

    name = "get_repo_info"
    description = "Get metadata for a GitHub repository"
    parameters_schema = {
        "type": "object",
        "required": ["owner", "name"],
        "properties": {
            "owner": {"type": "string"},
            "name": {"type": "string"},
        },
    }

    def __init__(self, repo_service: BaseRepositoryService) -> None:
        self.repo_service = repo_service

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch repository information."""
        repo = self.repo_service.get_repo_info(params["owner"], params["name"])
        return repo.model_dump(mode="json")


class ListFilesTool(BaseTool):
    """List files in a repository directory."""

    name = "list_files"
    description = "List files in a GitHub repository directory"
    parameters_schema = {
        "type": "object",
        "required": ["owner", "name"],
        "properties": {
            "owner": {"type": "string"},
            "name": {"type": "string"},
            "path": {"type": "string"},
            "ref": {"type": ["string", "null"]},
        },
    }

    def __init__(self, repo_service: BaseRepositoryService) -> None:
        self.repo_service = repo_service

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a serialized directory listing."""
        items = self.repo_service.list_files(
            params["owner"],
            params["name"],
            path=params.get("path", ""),
            ref=params.get("ref"),
        )
        return {"files": [item.model_dump(mode="json") for item in items]}


class GetFileContentTool(BaseTool):
    """Read a repository file content."""

    name = "get_file_content"
    description = "Get file content from a GitHub repository"
    parameters_schema = {
        "type": "object",
        "required": ["owner", "name", "path"],
        "properties": {
            "owner": {"type": "string"},
            "name": {"type": "string"},
            "path": {"type": "string"},
            "ref": {"type": ["string", "null"]},
        },
    }

    def __init__(self, repo_service: BaseRepositoryService) -> None:
        self.repo_service = repo_service

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return repository file content."""
        content = self.repo_service.get_file_content(
            params["owner"],
            params["name"],
            params["path"],
            ref=params.get("ref"),
        )
        return {"path": params["path"], "content": content}


class ToolExecutionEngine:
    """Safely execute an allowlisted set of tools."""

    def __init__(self, tools: Iterable[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an allowlisted tool with basic error normalization."""
        if tool_name not in self._tools:
            return {"ok": False, "error": f"Tool '{tool_name}' is not registered."}
        try:
            result = self._tools[tool_name].execute(params)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata for registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]
