from __future__ import annotations

import httpx
import pytest

from sain_glm_agent.domain.entities import Message, ProviderRequest
from sain_glm_agent.infrastructure.providers.glm import GLMProvider
from sain_glm_agent.infrastructure.providers.registry import (
    ProviderRegistry,
    create_provider_from_settings,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.last_json = None

    def post(self, url: str, json: dict) -> FakeResponse:
        self.last_json = json
        return self._response


def test_glm_provider_complete_success() -> None:
    fake_client = FakeClient(
        FakeResponse(
            {
                "model": "glm-4-flash",
                "choices": [{"message": {"content": "Hello from GLM"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
        )
    )
    provider = GLMProvider(api_key="secret", client=fake_client)
    response = provider.complete(
        ProviderRequest(messages=[Message(role="user", content="Hi")], model="glm-4-flash")
    )
    assert response.content == "Hello from GLM"
    assert response.usage["prompt_tokens"] == 10
    assert fake_client.last_json["messages"][0]["content"] == "Hi"


def test_glm_provider_requires_key() -> None:
    provider = GLMProvider(api_key=None, client=FakeClient(FakeResponse({})))
    with pytest.raises(ValueError, match="GLM API key"):
        provider.complete(
            ProviderRequest(
                messages=[Message(role="user", content="Hi")],
                model="glm",
            )
        )


def test_provider_registry_and_factory() -> None:
    registry = ProviderRegistry()
    provider = GLMProvider(
        api_key="secret",
        client=FakeClient(
            FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {},
                }
            )
        ),
    )
    registry.register("glm", provider)
    assert registry.get_provider("glm") is provider

    class SettingsObj:
        glm_api_key = "secret"
        glm_base_url = "https://open.bigmodel.cn/api/paas/v4"
        glm_model = "glm-4-flash"
        active_provider = "glm"

    created = create_provider_from_settings(SettingsObj())
    assert created.get_capabilities()["provider"] == "glm"
