"""Repository analysis utilities."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import Settings
from .exceptions import RepositoryError
from .models import RepositorySnapshot

LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".rb": "Ruby",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
}
SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
KEY_FILE_NAMES = {"README.md", "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"}


class RepositoryAnalyzer:
    """Inspect repository structure and gather prompt context."""

    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root.resolve()
        self.settings = settings
        if not self.root.exists():
            raise RepositoryError(f"Repository path does not exist: {self.root}")

    def scan(self) -> RepositorySnapshot:
        """Walk the repository and build a snapshot."""

        files: list[str] = []
        directories: list[str] = []
        languages: set[str] = set()
        key_excerpts: dict[str, str] = {}

        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if any(part in SKIP_DIRECTORIES for part in path.parts):
                continue
            if path.is_dir():
                directories.append(relative)
                continue
            files.append(relative)
            if suffix_language := LANGUAGE_MAP.get(path.suffix.lower()):
                languages.add(suffix_language)
            if path.name in KEY_FILE_NAMES and relative not in key_excerpts:
                key_excerpts[relative] = self._safe_read_excerpt(path)

        return RepositorySnapshot(
            root=self.root,
            files=files,
            directories=directories,
            key_file_excerpts=key_excerpts,
            detected_languages=sorted(languages),
            git_status=self._run_git(["status", "--short"]),
            diff_summary=self._run_git(["diff", "--stat"]),
        )

    def read_file(self, relative_path: str) -> str:
        """Read a repository file safely."""

        path = (self.root / relative_path).resolve()
        if self.root not in path.parents and path != self.root:
            raise RepositoryError(f"Attempted to read outside repository root: {relative_path}")
        if not path.exists() or not path.is_file():
            raise RepositoryError(f"File not found: {relative_path}")
        return self._safe_read_excerpt(path, limit=self.settings.max_file_bytes)

    def build_file_context(self, task: str, snapshot: RepositorySnapshot | None = None) -> str:
        """Select repository files likely relevant to the task and return excerpted context."""

        snapshot = snapshot or self.scan()
        tokens = [token for token in re.split(r"\W+", task.lower()) if len(token) > 2]
        scored: list[tuple[int, str]] = []
        for relative in snapshot.files:
            score = 0
            lowered = relative.lower()
            for token in tokens:
                if token in lowered:
                    score += 2
            if relative in snapshot.key_file_excerpts:
                score += 1
            scored.append((score, relative))
        selected = [relative for _, relative in sorted(scored, reverse=True)[: self.settings.max_context_files]]
        seen: set[str] = set()
        blocks: list[str] = []
        for relative in selected:
            if relative in seen:
                continue
            seen.add(relative)
            blocks.append(f"## {relative}\n{self.read_file(relative)}")
        if not blocks:
            return "No file context available."
        return "\n\n".join(blocks)

    def prepare_pull_request_context(self) -> str:
        """Collect Git diff information useful for pull request drafting."""

        changed_files = self._run_git(["diff", "--name-only"])
        diff = self._run_git(["diff", "--stat"])
        return (
            f"Changed files:\n{changed_files or 'No changed files'}\n\n"
            f"Diff summary:\n{diff or 'No diff summary'}"
        )

    def _safe_read_excerpt(self, path: Path, limit: int | None = None) -> str:
        """Read and truncate a text file while tolerating binary or invalid input."""

        byte_limit = limit or self.settings.max_file_bytes
        raw = path.read_bytes()[:byte_limit]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "<binary or non-UTF-8 content omitted>"
        return text.strip() or "<empty file>"

    def _run_git(self, args: list[str]) -> str:
        """Run a Git command in the repository and return trimmed stdout."""

        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepositoryError(f"Failed to run git {' '.join(args)}: {error}") from error
        output = completed.stdout.strip() or completed.stderr.strip()
        return output.strip()
