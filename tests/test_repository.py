"""Tests for repository analysis."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sain_glm_agent.config import Settings
from sain_glm_agent.repository import RepositoryAnalyzer


class RepositoryAnalyzerTests(unittest.TestCase):
    def test_scan_and_build_context(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "agent.py").write_text("print('hello')\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            settings = Settings(api_key="unused")
            analyzer = RepositoryAnalyzer(root, settings)
            snapshot = analyzer.scan()
            context = analyzer.build_file_context("update agent", snapshot=snapshot)
        self.assertIn("README.md", snapshot.files)
        self.assertIn("Python", snapshot.detected_languages)
        self.assertIn("agent.py", context)
