"""GitHub API client for SAIN GLM Agent.

Wraps `PyGithub <https://pygithub.readthedocs.io>`_ to expose the subset of
GitHub operations needed by the agent:

* Fetching repository metadata and file trees
* Reading file contents
* Creating branches and committing files
* Opening pull requests

Usage::

    from sain_glm_agent.repository import GitHubClient

    client = GitHubClient(token="ghp_...")
    repo = client.get_repo("org/project")
    tree = client.get_file_tree(repo)
    content = client.read_file(repo, "src/main.py")
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RepoFile:
    """Metadata and optional content for a single repository file.

    Attributes:
        path: Repository-relative path (e.g. ``src/main.py``).
        name: Filename component.
        size: File size in bytes.
        content: Decoded UTF-8 content (``None`` if not yet fetched).
        sha: Git blob SHA.
        download_url: Raw download URL.
    """

    path: str
    name: str
    size: int
    content: str | None = None
    sha: str = ""
    download_url: str = ""


class GitHubClient:
    """Thin wrapper around the PyGithub client.

    Args:
        token: GitHub personal-access token (requires ``repo`` scope for
            private repos; ``public_repo`` for public repos).
        base_url: Override for GitHub Enterprise.  Defaults to
            ``https://api.github.com``.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
    ) -> None:
        self._token = token
        self._base_url = base_url
        self._gh = self._build_client()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        try:
            from github import Github  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyGithub is required. Install it: pip install PyGithub"
            ) from exc
        return Github(login_or_token=self._token, base_url=self._base_url)

    # ------------------------------------------------------------------
    # Repository
    # ------------------------------------------------------------------

    def get_repo(self, repo_full_name: str) -> Any:
        """Return a PyGithub Repository object.

        Args:
            repo_full_name: ``owner/repo`` string.

        Returns:
            A ``github.Repository.Repository`` instance.
        """
        logger.debug("Fetching repo: %s", repo_full_name)
        return self._gh.get_repo(repo_full_name)

    def get_file_tree(
        self,
        repo: Any,
        path: str = "",
        max_files: int = 500,
    ) -> list[RepoFile]:
        """Recursively list repository files.

        Args:
            repo: PyGithub Repository.
            path: Sub-directory to start from (empty = root).
            max_files: Maximum number of files to return.

        Returns:
            List of :class:`RepoFile` objects (content not pre-loaded).
        """
        result: list[RepoFile] = []
        self._collect_files(repo, path, result, max_files)
        logger.debug("File tree: %d files in %s", len(result), repo.full_name)
        return result

    def _collect_files(
        self,
        repo: Any,
        path: str,
        result: list[RepoFile],
        limit: int,
    ) -> None:
        if len(result) >= limit:
            return
        try:
            contents = repo.get_contents(path)
        except Exception as exc:
            logger.warning("Could not list %s: %s", path, exc)
            return

        if not isinstance(contents, list):
            contents = [contents]

        for item in contents:
            if len(result) >= limit:
                break
            if item.type == "dir":
                self._collect_files(repo, item.path, result, limit)
            else:
                result.append(
                    RepoFile(
                        path=item.path,
                        name=item.name,
                        size=item.size,
                        sha=item.sha,
                        download_url=item.download_url or "",
                    )
                )

    def read_file(self, repo: Any, path: str) -> str:
        """Fetch and decode a single file's content.

        Args:
            repo: PyGithub Repository.
            path: Repository-relative file path.

        Returns:
            Decoded UTF-8 string content.

        Raises:
            RuntimeError: If the file cannot be read.
        """
        try:
            file_obj = repo.get_contents(path)
            if isinstance(file_obj, list):
                file_obj = file_obj[0]
            if file_obj.encoding == "base64":
                return base64.b64decode(file_obj.content).decode("utf-8", errors="replace")
            return file_obj.decoded_content.decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Failed to read '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    def create_branch(
        self,
        repo: Any,
        branch_name: str,
        base_branch: str = "main",
    ) -> Any:
        """Create a new branch from *base_branch*.

        Args:
            repo: PyGithub Repository.
            branch_name: Name for the new branch.
            base_branch: Branch to branch off from.

        Returns:
            A PyGithub ``GitRef`` object.
        """
        base_ref = repo.get_git_ref(f"heads/{base_branch}")
        base_sha = base_ref.object.sha
        ref = repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=base_sha,
        )
        logger.info("Created branch %s from %s (%s)", branch_name, base_branch, base_sha[:7])
        return ref

    def commit_file(
        self,
        repo: Any,
        path: str,
        content: str,
        message: str,
        branch: str,
    ) -> None:
        """Create or update a single file via the GitHub API.

        Args:
            repo: PyGithub Repository.
            path: Repository-relative path.
            content: New file content (UTF-8 string).
            message: Commit message.
            branch: Target branch name.
        """
        try:
            existing = repo.get_contents(path, ref=branch)
            if isinstance(existing, list):
                existing = existing[0]
            repo.update_file(
                path=path,
                message=message,
                content=content,
                sha=existing.sha,
                branch=branch,
            )
            logger.info("Updated %s on branch %s", path, branch)
        except Exception:
            repo.create_file(
                path=path,
                message=message,
                content=content,
                branch=branch,
            )
            logger.info("Created %s on branch %s", path, branch)

    def create_pull_request(
        self,
        repo: Any,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> Any:
        """Open a pull request.

        Args:
            repo: PyGithub Repository.
            title: PR title.
            body: PR body (Markdown).
            head_branch: Branch containing the changes.
            base_branch: Target branch.
            draft: Whether to create as a draft PR.

        Returns:
            A PyGithub ``PullRequest`` object.
        """
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft,
        )
        logger.info("Opened PR #%d: %s", pr.number, title)
        return pr
