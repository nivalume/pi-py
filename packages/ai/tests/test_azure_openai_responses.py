"""Offline tests for Azure OpenAI Responses (ports pi's ``azure-openai-*.test.ts`` Responses cases)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pi_ai import Context, Model, Tool, Type, UserMessage, normalize_context
from pi_ai.api import azure_openai_responses
from pi_ai.api.azure_openai_config import resolve_azure_base_url, resolve_azure_config, resolve_deployment_name
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.utils.misc import get_pi_user_agent

AZURE_ENV = (
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_RESOURCE_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in AZURE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def azure_model() -> Model:
    model = get_builtin_model("azure", "gpt-4o-mini")
    assert model is not None
    return model


def context(**kwargs: Any):
    return normalize_context(Context(messages=[UserMessage("hello", timestamp=0)], **kwargs))


def completed(response_id: str = "resp_azure") -> list[Any]:
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
    ]


async def run(mock_server: Any, model: Model | None = None, ctx: Any = None, **options: Any) -> Any:
    mock_server.add_sse(completed())
    return await azure_openai_responses.stream(
        model or azure_model(),
        ctx or context(),
        {"api_key": "test-api-key", "http_client": mock_server.client(), **options},
    ).result()


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://marc-quicktests-resource.cognitiveservices.azure.com",
            "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1",
        ),
        ("https://marc-quicktests-resource.ai.azure.com", "https://marc-quicktests-resource.ai.azure.com/openai/v1"),
        ("https://my-resource.openai.azure.com", "https://my-resource.openai.azure.com/openai/v1"),
        ("https://my-resource.openai.azure.com/", "https://my-resource.openai.azure.com/openai/v1"),
        (
            "https://my-resource.cognitiveservices.azure.com/openai",
            "https://my-resource.cognitiveservices.azure.com/openai/v1",
        ),
        (
            "https://my-resource.cognitiveservices.azure.com/openai/v1",
            "https://my-resource.cognitiveservices.azure.com/openai/v1",
        ),
        (
            "https://my-resource.services.ai.azure.com/openai/v1/responses",
            "https://my-resource.services.ai.azure.com/openai/v1",
        ),
        ("https://my-proxy.example.com/v1", "https://my-proxy.example.com/v1"),
        (
            "https://my-resource.openai.azure.com/openai?api-version=2024-12-01",
            "https://my-resource.openai.azure.com/openai/v1",
        ),
        ("https://my-proxy.example.com/v1?custom=true", "https://my-proxy.example.com/v1?custom=true"),
    ],
)
def test_base_url_normalization(clean_env: pytest.MonkeyPatch, base_url: str, expected: str) -> None:
    clean_env.setenv("AZURE_OPENAI_BASE_URL", base_url)
    assert resolve_azure_base_url(azure_model()) == expected


def test_base_url_sources(clean_env: pytest.MonkeyPatch) -> None:
    model = azure_model()
    with pytest.raises(ValueError, match="Azure OpenAI base URL is required"):
        resolve_azure_base_url(dataclasses.replace(model, base_url=""))
    assert resolve_azure_base_url(model, {"azure_resource_name": "res"}) == "https://res.openai.azure.com/openai/v1"
    clean_env.setenv("AZURE_OPENAI_RESOURCE_NAME", "my-resource")
    assert resolve_azure_base_url(model) == "https://my-resource.openai.azure.com/openai/v1"
    assert (
        resolve_azure_base_url(model, {"azure_base_url": " https://x.openai.azure.com "})
        == "https://x.openai.azure.com/openai/v1"
    )
    assert resolve_azure_base_url(model, {"env": {"AZURE_OPENAI_BASE_URL": "https://p.example.com/v1"}}) == (
        "https://p.example.com/v1"
    )


def test_api_version_and_deployment_names(clean_env: pytest.MonkeyPatch) -> None:
    model = azure_model()
    options = {"azure_base_url": "https://r.openai.azure.com"}
    assert resolve_azure_config(model, options).api_version == "v1"
    assert resolve_azure_config(model, {**options, "azure_api_version": "2025-01-01"}).api_version == "2025-01-01"
    assert resolve_deployment_name(model) == model.id
    clean_env.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", f" other=x, {model.id}=my-deploy=ignored ,bad")
    assert resolve_deployment_name(model) == "my-deploy"
    assert resolve_deployment_name(model, {"azure_deployment_name": "explicit"}) == "explicit"


async def test_invalid_base_url_is_a_stream_error(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AZURE_OPENAI_BASE_URL", "not-a-url")
    result = await azure_openai_responses.stream(azure_model(), context(), {"api_key": "k"}).result()
    assert result.stop_reason == "error"
    assert "Invalid Azure OpenAI base URL" in (result.error_message or "")


async def test_request_wire_format(mock_server: Any) -> None:
    result = await run(mock_server, azure_base_url="https://my-resource.openai.azure.com", session_id="x" * 67)
    assert result.stop_reason == "stop"
    assert result.response_id == "resp_azure"
    request = mock_server.last
    assert request.url == "https://my-resource.openai.azure.com/openai/v1/responses?api-version=v1"
    assert request.headers["api-key"] == "test-api-key"
    assert "authorization" not in request.headers
    assert request.headers["user-agent"] == get_pi_user_agent()
    body = request.json()
    assert body["model"] == "gpt-4o-mini"
    assert body["store"] is False
    assert body["prompt_cache_key"] == "x" * 64


async def test_deployment_name_is_the_request_model(mock_server: Any) -> None:
    await run(mock_server, azure_base_url="https://r.openai.azure.com", azure_deployment_name="prod-mini")
    assert mock_server.last.json()["model"] == "prod-mini"


async def test_api_version_replaces_base_url_query(mock_server: Any) -> None:
    await run(mock_server, azure_base_url="https://my-proxy.example.com/v1?custom=true", azure_api_version="2025-04-01")
    assert mock_server.last.url == "https://my-proxy.example.com/v1/responses?api-version=2025-04-01"


async def test_honors_supports_strict_mode_false(mock_server: Any) -> None:
    base = azure_model()
    model = dataclasses.replace(base, compat={**(base.compat or {}), "supportsStrictMode": False})
    ctx = context(
        tools=[
            Tool(
                "preferred",
                "Preferred constrained tool",
                Type.Object({"value": Type.String()}),
                constrained_sampling={"type": "json_schema", "strict": "prefer"},
            )
        ]
    )
    await run(mock_server, model, ctx, azure_base_url="https://r.openai.azure.com")
    assert "strict" not in mock_server.last.json()["tools"][0]


async def test_user_agent_override(mock_server: Any) -> None:
    await run(mock_server, azure_base_url="https://r.openai.azure.com", headers={"User-Agent": "custom-agent"})
    assert mock_server.last.headers["user-agent"] == "custom-agent"


async def test_forwards_provider_events_in_order(mock_server: Any) -> None:
    model = azure_model()
    received: list[Any] = []
    models: list[Any] = []

    async def on_event(event: Any, event_model: Any) -> None:
        received.append(event)
        models.append(event_model)

    result = await run(
        mock_server, model, azure_base_url="https://r.openai.azure.com", on_provider_stream_event=on_event
    )
    assert received == completed()
    assert models == [model, model]
    assert result.stop_reason == "stop"
    assert result.response_id == "resp_azure"


TOOL_MODEL = Model(
    id="test-deployment",
    name="Test Deployment",
    api="azure-openai-responses",
    provider="azure",
    base_url="http://127.0.0.1:9/openai/v1",
    reasoning=False,
    context_window=10_000,
    max_tokens=1_000,
)


def payload_capture() -> tuple[dict[str, Any], Any]:
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _model: Any) -> None:
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    return captured, on_payload


async def test_forwards_provider_specific_tool_choice() -> None:
    captured, on_payload = payload_capture()
    ctx = context(tools=[Tool("read", "Read a file", Type.Object({"path": Type.String()}))])
    result = await azure_openai_responses.stream(
        TOOL_MODEL, ctx, {"api_key": "test-key", "tool_choice": "required", "on_payload": on_payload}
    ).result()
    assert result.error_message == "payload captured"
    assert captured["payload"]["tool_choice"] == "required"
    assert len(captured["payload"]["tools"]) == 1


async def test_forwards_simple_tool_choice() -> None:
    captured, on_payload = payload_capture()
    ctx = context(tools=[Tool("read", "Read a file", Type.Object({"path": Type.String()}))])
    await azure_openai_responses.stream_simple(
        TOOL_MODEL, ctx, {"api_key": "test-key", "tool_choice": "none", "on_payload": on_payload}
    ).result()
    assert captured["payload"]["tool_choice"] == "none"
    assert len(captured["payload"]["tools"]) == 1


def test_stream_simple_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="No API key for provider: azure"):
        azure_openai_responses.stream_simple(TOOL_MODEL, context(), {})


async def test_http_error_prefix(mock_server: Any) -> None:
    mock_server.add_json({"error": {"message": "Deployment not found"}}, status=404)
    result = await azure_openai_responses.stream(
        azure_model(),
        context(),
        {"api_key": "k", "http_client": mock_server.client(), "azure_base_url": "https://r.openai.azure.com"},
    ).result()
    assert result.stop_reason == "error"
    assert (result.error_message or "").startswith("Azure OpenAI API error (404)")
