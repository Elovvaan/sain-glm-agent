"""Tests for the GLM provider adapter."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from sain_glm_agent.config import Settings
from sain_glm_agent.models import Message, ModelRequest
from sain_glm_agent.providers.glm import GLMProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class GLMProviderTests(unittest.TestCase):
    def test_generate_normalizes_content(self) -> None:
        provider = GLMProvider(Settings(api_key="secret"))
        request = ModelRequest(
            model="glm-5.2",
            system_prompt="system",
            messages=[Message(role="user", content="hello")],
        )
        payload = {
            "choices": [{"message": {"content": "world"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
            response = provider.generate(request)
        self.assertEqual(response.content, "world")
        self.assertEqual(response.usage["prompt_tokens"], 12)
