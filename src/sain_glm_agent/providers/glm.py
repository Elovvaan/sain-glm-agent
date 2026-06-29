"""GLM provider implementation using an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
from urllib import error, request as urllib_request

from ..config import Settings
from ..exceptions import ProviderError
from ..models import Message, ModelRequest, ModelResponse
from .base import ModelProvider


class GLMProvider(ModelProvider):
    """Primary provider implementation targeting GLM-5.2."""

    name = "glm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.require_api_key()

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call the configured GLM endpoint and normalize the response."""

        payload = {
            "model": request.model,
            "messages": self._build_messages(request.system_prompt, request.messages),
            "temperature": request.generation.temperature,
            "max_tokens": request.generation.max_tokens,
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib_request.Request(
            url=self._endpoint(),
            data=encoded,
            headers={
                "Authorization": f"******",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=self.settings.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as http_error:
            details = http_error.read().decode("utf-8", errors="ignore")
            raise ProviderError(f"GLM request failed with HTTP {http_error.code}: {details}") from http_error
        except error.URLError as url_error:
            raise ProviderError(f"GLM request failed: {url_error.reason}") from url_error

        content = self._extract_content(response_payload)
        usage = response_payload.get("usage", {})
        return ModelResponse(
            provider=self.name,
            model=request.model,
            content=content,
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int | float)},
            raw=response_payload,
        )

    def _endpoint(self) -> str:
        base_url = self.settings.base_url
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _build_messages(system_prompt: str, messages: list[Message]) -> list[dict[str, str]]:
        rendered = [{"role": "system", "content": system_prompt}]
        rendered.extend({"role": message.role, "content": message.content} for message in messages)
        return rendered

    @staticmethod
    def _extract_content(payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("GLM response did not include choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderError("GLM response choice was malformed.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("GLM response did not include a message.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("GLM response message content was not text.")
        return content.strip()
