"""GLM provider implementation backed by the BigModel chat completions API."""

from __future__ import annotations

from typing import Any

import httpx

from sain_glm_agent.domain.entities import ProviderRequest, ProviderResponse
from sain_glm_agent.infrastructure.providers.base import BaseProvider


class GLMProvider(BaseProvider):
    """Provider for GLM-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        default_model: str = "glm-4-flash",
        timeout: float = 30.0,
        client: httpx.Client | Any | None = None,
    ) -> None:
        super().__init__(name="glm")
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._client = client or httpx.Client(
            timeout=self.timeout,
            headers=self._build_headers(),
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Call the GLM chat completions endpoint and normalize the result."""
        if not self.api_key:
            raise ValueError(
                "GLM API key is missing. Set GLM_API_KEY in your environment or .env file."
            )

        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools

        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "GLM request timed out. Please retry with a longer timeout."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            details = exc.response.text
            if status_code in {401, 403}:
                raise RuntimeError(
                    "GLM authentication failed. Verify GLM_API_KEY and provider access."
                ) from exc
            if status_code == 429:
                raise RuntimeError(
                    "GLM rate limit exceeded. Retry later or reduce request volume."
                ) from exc
            if status_code >= 500:
                raise RuntimeError(
                    "GLM service is unavailable right now. Please retry shortly."
                ) from exc
            raise RuntimeError(f"GLM request failed ({status_code}): {details}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Unable to reach GLM service: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("GLM response did not contain any completion choices.")
        message_content = choices[0].get("message", {}).get("content", "")
        if isinstance(message_content, list):
            message_content = "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in message_content
            )
        usage = data.get("usage") or {}
        finish_reason = choices[0].get("finish_reason", "stop")
        model = data.get("model") or request.model or self.default_model
        return ProviderResponse(
            content=str(message_content),
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def get_capabilities(self) -> dict[str, Any]:
        """Return a static description of provider capabilities."""
        return {
            "provider": self.name,
            "supports_chat": True,
            "supports_tools": True,
            "default_model": self.default_model,
            "base_url": self.base_url,
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers
