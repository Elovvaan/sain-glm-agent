"""GitHub REST repository adapter."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from sain_glm_agent.domain.entities import FileInfo, PRDraft, RepositoryInfo
from sain_glm_agent.domain.interfaces import BaseRepositoryService


class GitHubRepositoryService(BaseRepositoryService):
    """Read-only adapter for GitHub repository metadata and contents."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=self.timeout, headers=self._build_headers())

    def get_repo_info(self, owner: str, name: str) -> RepositoryInfo:
        """Return GitHub repository metadata."""
        data = self._request("GET", f"/repos/{owner}/{name}")
        return RepositoryInfo(
            owner=owner,
            name=data["name"],
            description=data.get("description"),
            language=data.get("language"),
            default_branch=data.get("default_branch", "main"),
            stars=data.get("stargazers_count", 0),
            topics=data.get("topics") or [],
        )

    def list_files(
        self,
        owner: str,
        name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[FileInfo]:
        """List files in a repository path using the contents API."""
        normalized_path = path.strip("/")
        endpoint = f"/repos/{owner}/{name}/contents"
        if normalized_path:
            endpoint = f"{endpoint}/{normalized_path}"
        data = self._request("GET", endpoint, params={"ref": ref} if ref else None)
        items = data if isinstance(data, list) else [data]
        results = [
            FileInfo(
                path=item.get("path", normalized_path),
                name=item.get("name", normalized_path or name),
                type=item.get("type", "file"),
                size=item.get("size"),
            )
            for item in items
        ]
        return sorted(results, key=lambda item: (item.type != "dir", item.path))

    def get_file_content(
        self,
        owner: str,
        name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Fetch and decode a repository file content."""
        normalized_path = path.strip("/")
        if not normalized_path:
            raise ValueError("A file path is required.")
        endpoint = f"/repos/{owner}/{name}/contents/{normalized_path}"
        data = self._request("GET", endpoint, params={"ref": ref} if ref else None)
        if data.get("type") != "file":
            raise ValueError(f"Path '{path}' is not a file.")
        content = data.get("content")
        if content and data.get("encoding") == "base64":
            return base64.b64decode(content).decode("utf-8")
        download_url = data.get("download_url")
        if download_url:
            response = self._client.get(download_url)
            response.raise_for_status()
            return response.text
        raise RuntimeError(f"Unable to retrieve content for '{path}'.")

    def get_directory_tree(
        self,
        owner: str,
        name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[FileInfo]:
        """Recursively traverse a repository directory tree."""
        tree: list[FileInfo] = []
        for item in self.list_files(owner, name, path=path, ref=ref):
            tree.append(item)
            if item.type == "dir":
                tree.extend(self.get_directory_tree(owner, name, path=item.path, ref=ref))
        return tree

    def prepare_pr_draft(
        self,
        owner: str,
        name: str,
        task: str,
        changed_files: list[str],
        base_branch: str | None = None,
        head_branch: str = "feature/sain-glm-agent",
        reasoning: str | None = None,
    ) -> PRDraft:
        """Create a pull request draft from a task description and file list."""
        repo = self.get_repo_info(owner, name)
        resolved_base = base_branch or repo.default_branch
        checklist = [
            "Review implementation against repository requirements",
            "Run automated tests and checks",
            "Verify documentation and configuration changes",
        ]
        bullet_files = (
            "\n".join(f"- {file_path}" for file_path in changed_files)
            or "- No files listed"
        )
        body = (
            f"## Summary\n{task}\n\n"
            f"## Reasoning\n{reasoning or 'Prepared from repository analysis.'}\n\n"
            f"## Changed Files\n{bullet_files}\n\n"
            f"## Checklist\n" + "\n".join(f"- [ ] {item}" for item in checklist)
        )
        return PRDraft(
            title=f"{task.strip().rstrip('.')}" or "Repository update",
            body=body,
            base_branch=resolved_base,
            head_branch=head_branch,
            changed_files=changed_files,
            checklist=checklist,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "sain-glm-agent",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError("GitHub request timed out. Please retry.") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 404:
                raise RuntimeError("GitHub repository or path was not found.") from exc
            if status_code in {401, 403}:
                raise RuntimeError(
                    "GitHub access denied. Check GITHUB_TOKEN permissions or rate limits."
                ) from exc
            raise RuntimeError(f"GitHub API request failed with status {status_code}.") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Unable to reach GitHub API: {exc}") from exc
