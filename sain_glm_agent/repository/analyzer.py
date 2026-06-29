"""Repository analyser — provides structured insights about a GitHub repo.

The analyser uses the :class:`~sain_glm_agent.repository.github_client.GitHubClient`
to fetch repository data and builds a structured :class:`RepoAnalysis` that the
agent can use for planning and code generation.

Usage::

    from sain_glm_agent.repository import GitHubClient, RepositoryAnalyzer

    client = GitHubClient(token="ghp_...")
    analyser = RepositoryAnalyzer(client)
    analysis = analyser.analyse("owner/repo", objective="Add unit tests")
    print(analysis.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sain_glm_agent.repository.github_client import GitHubClient, RepoFile

logger = logging.getLogger(__name__)

# File extensions considered source code (for language detection)
_CODE_EXTENSIONS: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript (JSX)",
    ".tsx": "TypeScript (TSX)",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
    ".swift": "Swift",
    ".sh": "Shell",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
}

# Files / paths indicating a particular framework / tooling
_FRAMEWORK_INDICATORS: dict[str, str] = {
    "pyproject.toml": "Python (pyproject.toml)",
    "setup.py": "Python (setup.py)",
    "package.json": "Node.js",
    "go.mod": "Go module",
    "Cargo.toml": "Rust (Cargo)",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java (Gradle)",
    "Gemfile": "Ruby (Bundler)",
    "composer.json": "PHP (Composer)",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    ".github/workflows": "GitHub Actions",
}


@dataclass
class RepoAnalysis:
    """Structured analysis of a GitHub repository.

    Attributes:
        repo_full_name: ``owner/repo`` identifier.
        default_branch: Repository default branch.
        description: Repository description from GitHub.
        primary_language: Most-used programming language.
        detected_languages: All languages found, with file counts.
        detected_frameworks: Framework / tooling indicators found.
        total_files: Total number of files analysed.
        file_tree: All files found (without content).
        key_files: Files considered most important for the given objective.
        readme_content: Content of ``README.md`` (if present).
        objective: The task the agent is trying to accomplish.
        summary: Human-readable summary paragraph.
    """

    repo_full_name: str
    default_branch: str = "main"
    description: str = ""
    primary_language: str = ""
    detected_languages: dict[str, int] = field(default_factory=dict)
    detected_frameworks: list[str] = field(default_factory=list)
    total_files: int = 0
    file_tree: list[RepoFile] = field(default_factory=list)
    key_files: list[RepoFile] = field(default_factory=list)
    readme_content: str = ""
    objective: str = ""
    summary: str = ""

    def to_context_str(self, max_file_tree_lines: int = 60) -> str:
        """Render the analysis as a compact context string for the LLM prompt.

        Args:
            max_file_tree_lines: Truncate file tree after this many lines.

        Returns:
            Multi-line string ready to inject into a prompt.
        """
        lines: list[str] = [
            f"# Repository: {self.repo_full_name}",
            f"Description: {self.description or 'N/A'}",
            f"Default branch: {self.default_branch}",
            f"Primary language: {self.primary_language or 'Unknown'}",
            f"Frameworks / tooling: {', '.join(self.detected_frameworks) or 'None detected'}",
            f"Total files: {self.total_files}",
            "",
            "## File tree (partial)",
        ]
        for i, f in enumerate(self.file_tree):
            if i >= max_file_tree_lines:
                lines.append(f"  … ({self.total_files - max_file_tree_lines} more files)")
                break
            lines.append(f"  {f.path}")

        if self.readme_content:
            snippet = self.readme_content[:1000]
            lines += ["", "## README (first 1000 chars)", snippet]

        if self.objective:
            lines += ["", "## Objective", self.objective]

        return "\n".join(lines)


class RepositoryAnalyzer:
    """Builds a :class:`RepoAnalysis` by inspecting a GitHub repository.

    Args:
        client: An authenticated :class:`GitHubClient`.
    """

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    def analyse(
        self,
        repo_full_name: str,
        objective: str = "",
        max_files: int = 300,
        fetch_key_files: bool = True,
    ) -> RepoAnalysis:
        """Fetch repository data and build a :class:`RepoAnalysis`.

        Args:
            repo_full_name: ``owner/repo`` string.
            objective: Task description to guide which files are marked as key.
            max_files: Maximum number of files to scan.
            fetch_key_files: Whether to download content of key files.

        Returns:
            A populated :class:`RepoAnalysis`.
        """
        logger.info("Analysing repository: %s", repo_full_name)
        repo = self._client.get_repo(repo_full_name)

        analysis = RepoAnalysis(
            repo_full_name=repo_full_name,
            default_branch=repo.default_branch or "main",
            description=repo.description or "",
            objective=objective,
        )

        # File tree
        analysis.file_tree = self._client.get_file_tree(repo, max_files=max_files)
        analysis.total_files = len(analysis.file_tree)

        # Language detection
        analysis.detected_languages = self._detect_languages(analysis.file_tree)
        if analysis.detected_languages:
            analysis.primary_language = max(
                analysis.detected_languages, key=lambda k: analysis.detected_languages[k]
            )

        # Framework detection
        analysis.detected_frameworks = self._detect_frameworks(analysis.file_tree)

        # README
        analysis.readme_content = self._fetch_readme(repo)

        # Key files
        if fetch_key_files:
            analysis.key_files = self._identify_key_files(
                repo, analysis.file_tree, objective
            )

        # Summary
        analysis.summary = self._build_summary(analysis)

        logger.info(
            "Analysis complete: %d files, primary language=%s",
            analysis.total_files,
            analysis.primary_language,
        )
        return analysis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_languages(files: list[RepoFile]) -> dict[str, int]:
        from pathlib import Path

        counts: dict[str, int] = {}
        for f in files:
            ext = Path(f.name).suffix.lower()
            lang = _CODE_EXTENSIONS.get(ext)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def _detect_frameworks(files: list[RepoFile]) -> list[str]:
        file_paths = {f.path for f in files}
        found: list[str] = []
        for indicator, label in _FRAMEWORK_INDICATORS.items():
            if any(p == indicator or p.startswith(indicator) for p in file_paths):
                found.append(label)
        return found

    def _fetch_readme(self, repo: Any) -> str:
        for name in ("README.md", "README.rst", "README.txt", "README"):
            try:
                return self._client.read_file(repo, name)
            except Exception:
                continue
        return ""

    def _identify_key_files(
        self,
        repo: Any,
        files: list[RepoFile],
        objective: str,
    ) -> list[RepoFile]:
        """Heuristically select the most relevant files for the objective."""
        objective_lower = objective.lower()
        scored: list[tuple[int, RepoFile]] = []

        for f in files:
            score = 0
            path_lower = f.path.lower()

            # Always include top-level config and entry points
            if "/" not in f.path:
                score += 2
            if any(
                kw in path_lower
                for kw in ("main", "app", "index", "cli", "entry", "init")
            ):
                score += 3
            if any(
                kw in path_lower
                for kw in ("test", "spec", "readme", "config", "settings")
            ):
                score += 2
            # Keyword match against objective
            if objective_lower:
                for word in objective_lower.split():
                    if len(word) > 3 and word in path_lower:
                        score += 5
            # Prefer source files over binaries/assets
            from pathlib import Path as _Path

            ext = _Path(f.name).suffix.lower()
            if ext in _CODE_EXTENSIONS:
                score += 1

            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        key_files: list[RepoFile] = []
        for _, f in scored[:10]:
            try:
                content = self._client.read_file(repo, f.path)
                key_files.append(
                    RepoFile(
                        path=f.path,
                        name=f.name,
                        size=f.size,
                        content=content,
                        sha=f.sha,
                        download_url=f.download_url,
                    )
                )
            except Exception:
                key_files.append(f)
        return key_files

    @staticmethod
    def _build_summary(analysis: RepoAnalysis) -> str:
        lang = analysis.primary_language or "unknown language"
        fw = (
            f" using {', '.join(analysis.detected_frameworks[:3])}"
            if analysis.detected_frameworks
            else ""
        )
        return (
            f"Repository '{analysis.repo_full_name}' is primarily written in {lang}{fw}. "
            f"It contains {analysis.total_files} files. "
            f"Default branch: {analysis.default_branch}."
        )
