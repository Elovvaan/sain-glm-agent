"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sain_glm_agent.config import Settings
from sain_glm_agent.exceptions import ConfigurationError


class SettingsTests(unittest.TestCase):
    def test_from_env_uses_secure_defaults(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "SAIN_API_KEY": "secret-key",
                "SAIN_DATA_DIR": tmpdir,
            },
            clear=False,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.provider, "glm")
        self.assertEqual(settings.model, "glm-5.2")
        self.assertEqual(settings.api_key, "secret-key")
        self.assertEqual(settings.data_dir, Path(tmpdir))

    def test_to_redacted_dict_hides_api_key(self) -> None:
        settings = Settings(api_key="secret")
        redacted = settings.to_redacted_dict()
        self.assertEqual(redacted["api_key"], "***redacted***")

    def test_invalid_temperature_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings(temperature=9).validate()
