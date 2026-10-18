"""End-to-end streaming tests for the Google adapters against the in-process mock server.

Ports of upstream test/google-raw-stop-reason.test.ts, google-thinking-level-map.test.ts and
google-vertex-api-key-resolution.test.ts (which mock ``@google/genai``), plus wire-level checks
of the official ``google-genai`` SDK's URLs, headers, errors and ADC credentials.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import google.auth
import google.oauth2.credentials
import httpx
import pytest
from google.auth.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.genai import types as genai_types
from google.genai import version as genai_version

from pi_ai import AbortController, Context, Model, Tool, UserMessage, normalize_context
from pi_ai.api import google_generative_ai, google_shared, google_vertex
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.utils.misc import get_pi_user_agent

CONTEXT = normalize_context(Context(messages=[UserMessage("hello")]))
#: The label the SDK puts in front of the User-Agent and sends as x-goog-api-client.
SDK_LABEL = f"google-genai-sdk/{genai_version.__version__} gl-python/{sys.version.split()[0]}"
PI_USER_AGENT = f"{SDK_LABEL} {get_pi_user_agent()}"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK reads these from os.environ; keep the developer's environment out of the tests.
    for name in (
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def adc_file(tmp_path: Path) -> Path:
    path = tmp_path / "adc.json"
    path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token",
                "quota_project_id": "quota-project",
            }
        )
    )
    return path


class FakeCredentials(Credentials):
    """ADC credentials whose refresh hands out a fixed token without network access."""

    def __init__(self, token: str = "adc-token") -> None:
        super().__init__()
        self.next_token = token
        self.refreshes = 0

    def refresh(self, request: Any) -> None:
        self.refreshes += 1
        self.token = self.next_token


def google_model(id: str = "gemini-2.5-flash", **overrides: Any) -> Model:
    model = get_builtin_model("google", id)
    assert model is not None
    # Catalog entries are shared; never mutate them.
    return dataclasses.replace(model, **overrides)


def vertex_model(id: str = "gemini-3-flash-preview", **overrides: Any) -> Model:
    model = get_builtin_model("google-vertex", id)
    assert model is not None
    # Catalog entries are shared; never mutate them.
    return dataclasses.replace(model, **overrides)


def stop_chunk(finish_reason: str = "STOP", function_call: bool = False) -> dict[str, Any]:
    candidate: dict[str, Any] = {"finishReason": finish_reason}
    if function_call:
        candidate["content"] = {
            "parts": [{"functionCall": {"id": "call-1", "name": "echo", "args": {"value": "truncated"}}}]
        }
    return {
        "responseId": "google-response-id",
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0, "totalTokenCount": 1},
    }


def token_response() -> httpx.Response:
    return httpx.Response(200, json={"access_token": "ya29.token", "expires_in": 3599, "token_type": "Bearer"})


async def collect(stream: Any) -> tuple[list[Any], Any]:
    events = [event async for event in stream]
    return events, await stream.result()


# ---------------------------------------------------------------------------
# Gemini API: text, thinking, tools, usage
# ---------------------------------------------------------------------------


async def test_text_and_thinking_stream(mock_server: Any) -> None:
    mock_server.add_sse(
        [
            {
                "responseId": "resp_1",
                "candidates": [
                    {"content": {"parts": [{"text": "Let me ", "thought": True, "thoughtSignature": "c2ln"}]}}
                ],
            },
            {"candidates": [{"content": {"parts": [{"text": "think.", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {
                "candidates": [
                    {"content": {"parts": [{"text": " world", "thoughtSignature": "dGV4dA=="}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "cachedContentTokenCount": 4,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 3,
                    "totalTokenCount": 18,
                },
            },
        ]
    )
    model = google_model()
    events, message = await collect(
        google_generative_ai.stream(model, CONTEXT, {"api_key": "key", "http_client": mock_server.client()})
    )
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "STOP"
    assert message.response_id == "resp_1"
    assert message.content[0].thinking == "Let me think."
    assert message.content[0].thinking_signature == "c2ln"
    assert message.content[1].text == "Hello world"
    assert message.content[1].text_signature == "dGV4dA=="
    usage = message.usage
    assert (usage.input, usage.output, usage.cache_read, usage.reasoning, usage.total_tokens) == (6, 8, 4, 3, 18)
    assert usage.cost.total > 0

    request = mock_server.last
    assert request.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert request.headers["x-goog-api-key"] == "key"
    # The SDK prefixes pi's User-Agent with its own label and sends that label as x-goog-api-client.
    assert request.headers["user-agent"] == PI_USER_AGENT
    assert request.headers["x-goog-api-client"] == SDK_LABEL
    assert request.headers["content-type"] == "application/json"
    assert request.json() == {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {}}


async def test_tool_calls_and_generated_ids(mock_server: Any) -> None:
    mock_server.add_sse(
        [
            {"candidates": [{"content": {"parts": [{"text": "Calling"}]}}]},
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"name": "bash", "args": {"command": "ls"}},
                                    "thoughtSignature": "c2ln",
                                },
                                {"functionCall": {"id": "dup", "name": "bash", "args": {"command": "pwd"}}},
                                {"functionCall": {"id": "dup", "name": "bash"}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        ]
    )
    tool = Tool("bash", "Run a command", {"type": "object", "properties": {"command": {"type": "string"}}})
    context = normalize_context(Context(system_prompt="sys", messages=[UserMessage("go")], tools=[tool]))
    events, message = await collect(
        google_generative_ai.stream(
            google_model(), context, {"api_key": "key", "http_client": mock_server.client(), "tool_choice": "auto"}
        )
    )
    assert message.stop_reason == "toolUse"
    calls = [block for block in message.content if block.type == "toolCall"]
    assert len(calls) == 3
    assert calls[0].id.startswith("bash_") and calls[0].thought_signature == "c2ln"
    assert calls[0].arguments == {"command": "ls"}
    assert calls[1].id == "dup"
    assert calls[2].id != "dup" and calls[2].arguments == {}
    deltas = [event.delta for event in events if event.type == "toolcall_delta"]
    assert deltas == ['{"command":"ls"}', '{"command":"pwd"}', "{}"]
    assert [event.type for event in events][1:4] == ["text_start", "text_delta", "text_end"]
    body = mock_server.last.json()
    assert body["systemInstruction"] == {"role": "user", "parts": [{"text": "sys"}]}
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "bash"
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}


# ---------------------------------------------------------------------------
# Raw stop reasons and provider stream events (google-raw-stop-reason.test.ts)
# ---------------------------------------------------------------------------


def vertex_options(mock_server: Any, **extra: Any) -> dict[str, Any]:
    return {"project": "test-project", "location": "us-central1", "http_client": mock_server.client(), **extra}


@pytest.fixture
def fake_adc(monkeypatch: pytest.MonkeyPatch) -> list[FakeCredentials]:
    """Replace ``google.auth.default()`` (what the SDK calls for ADC); records the credentials it hands out."""
    issued: list[FakeCredentials] = []

    def fake_default(scopes: Any = None, **_kwargs: Any) -> tuple[Credentials, str]:
        assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = FakeCredentials()
        issued.append(credentials)
        return credentials, "adc-project"

    monkeypatch.setattr(google.auth, "default", fake_default)
    return issued


async def test_raw_finish_reason_google(mock_server: Any) -> None:
    mock_server.add_sse([stop_chunk("MALFORMED_FUNCTION_CALL")])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "test-api-key", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "MALFORMED_FUNCTION_CALL"
    assert message.error_message == "Provider stopped with: MALFORMED_FUNCTION_CALL"


async def test_raw_finish_reason_vertex(mock_server: Any, fake_adc: Any) -> None:
    mock_server.add_sse([stop_chunk("SAFETY")])
    message = await google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server)).result()
    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "SAFETY"
    assert message.error_message == "Provider stopped with: SAFETY"


@pytest.mark.parametrize("adapter", ["google", "vertex"])
async def test_max_tokens_with_tool_call_is_length(mock_server: Any, fake_adc: Any, adapter: str) -> None:
    mock_server.add_sse([stop_chunk("MAX_TOKENS", function_call=True)])
    if adapter == "google":
        s = google_generative_ai.stream(google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()})
    else:
        s = google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server))
    message = await s.result()
    assert message.stop_reason == "length"
    assert message.raw_stop_reason == "MAX_TOKENS"
    assert any(block.type == "toolCall" for block in message.content)


@pytest.mark.parametrize("adapter", ["google", "vertex"])
async def test_stop_with_tool_call_is_tool_use(mock_server: Any, fake_adc: Any, adapter: str) -> None:
    mock_server.add_sse([stop_chunk("STOP", function_call=True)])
    if adapter == "google":
        s = google_generative_ai.stream(google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()})
    else:
        s = google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server))
    message = await s.result()
    assert message.stop_reason == "toolUse"
    assert message.raw_stop_reason == "STOP"


@pytest.mark.parametrize("adapter", ["google", "vertex"])
async def test_forwards_provider_stream_events(mock_server: Any, fake_adc: Any, adapter: str) -> None:
    chunks = [
        {"responseId": "resp_google", "candidates": [{"content": {"parts": [{"text": "hello"}]}}]},
        {
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
        },
    ]
    mock_server.add_sse(chunks)
    received: list[Any] = []
    models: list[Any] = []

    async def on_event(chunk: Any, model: Any) -> None:
        await asyncio.sleep(0)
        received.append(chunk)
        models.append(model)

    if adapter == "google":
        model = google_model()
        s = google_generative_ai.stream(
            model, CONTEXT, {"api_key": "k", "http_client": mock_server.client(), "on_provider_stream_event": on_event}
        )
    else:
        model = vertex_model()
        s = google_vertex.stream(model, CONTEXT, vertex_options(mock_server, on_provider_stream_event=on_event))
    result = await s.result()
    # Like upstream, the hook receives the SDK's GenerateContentResponse objects.
    assert all(isinstance(chunk, genai_types.GenerateContentResponse) for chunk in received)
    dumped = [
        chunk.model_dump(mode="json", by_alias=True, exclude_none=True, exclude={"sdk_http_response"})
        for chunk in received
    ]
    assert dumped == chunks
    assert models == [model, model]
    assert result.stop_reason == "stop"
    assert result.response_id == "resp_google"
    assert [block.text for block in result.content] == ["hello"]


async def test_missing_finish_reason(mock_server: Any, fake_adc: Any) -> None:
    mock_server.add_sse([{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.error_message == "Google stream ended without a finish reason"
    mock_server.add_sse([{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}])
    message = await google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server)).result()
    assert message.error_message == "Google Vertex stream ended without a finish reason"
    assert message.content[0].text == "hi"


# ---------------------------------------------------------------------------
# User agent and headers
# ---------------------------------------------------------------------------


async def test_user_agent_defaults_and_overrides(mock_server: Any) -> None:
    mock_server.add_sse([stop_chunk()])
    await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert mock_server.last.headers["user-agent"] == PI_USER_AGENT
    mock_server.add_sse([stop_chunk()])
    await google_generative_ai.stream(
        google_model(),
        CONTEXT,
        {
            "api_key": "k",
            "http_client": mock_server.client(),
            "headers": {"User-Agent": "custom-agent", "x-goog-api-key": "mine"},
        },
    ).result()
    assert mock_server.last.headers["user-agent"] == f"{SDK_LABEL} custom-agent"
    # Like the SDK, an explicit API key header is not overwritten.
    assert mock_server.last.headers["x-goog-api-key"] == "mine"


async def test_custom_base_url_and_env_base_url(mock_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_server.add_sse([stop_chunk()])
    model = google_model(base_url="https://proxy.example.com/gemini/")
    await google_generative_ai.stream(model, CONTEXT, {"api_key": "k", "http_client": mock_server.client()}).result()
    assert (
        mock_server.last.url == "https://proxy.example.com/gemini/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )

    mock_server.add_sse([stop_chunk()])
    await google_generative_ai.stream(
        google_model(base_url=""),
        CONTEXT,
        {"api_key": "k", "http_client": mock_server.client(), "env": {"GOOGLE_GEMINI_BASE_URL": "https://env.test"}},
    ).result()
    assert mock_server.last.url == "https://env.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"


# ---------------------------------------------------------------------------
# Errors, retries, abort
# ---------------------------------------------------------------------------


async def test_http_error_message_is_the_json_body(mock_server: Any) -> None:
    body = {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}
    mock_server.add_json(body, status=400)
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == json.dumps(body, separators=(",", ":"))


async def test_retries_retryable_status(mock_server: Any) -> None:
    mock_server.add_json({"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}, status=503)
    mock_server.add_sse([stop_chunk()])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client(), "max_retries": 1}
    ).result()
    assert message.stop_reason == "stop"
    assert len(mock_server.requests) == 2


async def test_in_stream_error_chunk(mock_server: Any) -> None:
    error_chunk = {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}
    mock_server.add_sse([{"candidates": [{"content": {"parts": [{"text": "x"}]}}]}, error_chunk])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == f"got status: RESOURCE_EXHAUSTED. {json.dumps(error_chunk, separators=(',', ':'))}"


async def test_missing_api_key() -> None:
    message = await google_generative_ai.stream(google_model(), CONTEXT, {}).result()
    assert message.error_message == "No API key for provider: google"
    with pytest.raises(ValueError, match="No API key for provider: google"):
        google_generative_ai.stream_simple(google_model(), CONTEXT, {})


async def test_already_aborted_signal() -> None:
    controller = AbortController()
    controller.abort()
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "signal": controller.signal}
    ).result()
    assert message.stop_reason == "aborted"
    assert message.error_message == "Request aborted"


async def test_abort_mid_stream() -> None:
    controller = AbortController()

    async def body() -> Any:
        yield b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n\n'
        await asyncio.sleep(10)
        yield b""

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})
        )
    )
    s = google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": client, "signal": controller.signal}
    )
    async for event in s:
        if event.type == "text_delta":
            controller.abort()
    message = await s.result()
    assert message.stop_reason == "aborted"
    assert message.content[0].text == "partial"


# ---------------------------------------------------------------------------
# Thinking configuration through stream_simple (google-thinking-level-map.test.ts)
# ---------------------------------------------------------------------------


def custom_model(api: str, id: str, thinking_level_map: dict[str, str | None]) -> Model:
    return Model(
        id=id,
        api=api,
        provider="test-google" if api == "google-generative-ai" else "test-vertex",
        base_url="https://example.invalid/v1beta" if api == "google-generative-ai" else "https://example.invalid/v1",
        reasoning=True,
        thinking_level_map=thinking_level_map,
        context_window=128000,
        max_tokens=4096,
    )


async def capture_payload(api: str, model: Model, reasoning: str | None = None, budgets: Any = None) -> dict[str, Any]:
    captured: list[Any] = []

    def on_payload(payload: Any, _model: Any) -> None:
        captured.append(payload)
        raise RuntimeError("payload captured")

    options: dict[str, Any] = {"api_key": "test", "on_payload": on_payload}
    if reasoning:
        options["reasoning"] = reasoning
    if budgets:
        options["thinking_budgets"] = budgets
    module = google_generative_ai if api == "google-generative-ai" else google_vertex
    result = await module.stream_simple(model, CONTEXT, options).result()
    assert "payload captured" in (result.error_message or "")
    return captured[0]


APIS = ["google-generative-ai", "google-vertex"]
NO_MINIMAL = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": None,
    "max": None,
}


@pytest.mark.parametrize("api", APIS)
async def test_lowest_supported_level_when_reasoning_omitted(api: str) -> None:
    payload = await capture_payload(api, custom_model(api, "gemini-3.8-flash", NO_MINIMAL))
    assert payload["config"]["thinkingConfig"] == {"thinkingLevel": "LOW"}


@pytest.mark.parametrize("api", APIS)
async def test_native_medium_for_gemini_31_pro(api: str) -> None:
    payload = await capture_payload(api, custom_model(api, "gemini-3.1-pro-preview", NO_MINIMAL), "medium")
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "MEDIUM"}


@pytest.mark.parametrize("api", APIS)
async def test_disables_gemini_25_thinking_when_reasoning_omitted(api: str) -> None:
    payload = await capture_payload(api, custom_model(api, "gemini-2.5-flash", {}))
    assert payload["config"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.parametrize("api", APIS)
async def test_thinking_levels_for_gemini_4(api: str) -> None:
    payload = await capture_payload(api, custom_model(api, "gemini-4-argon", {}), "medium")
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "MEDIUM"}


@pytest.mark.parametrize("reasoning", ["xhigh", "max"])
async def test_google_extended_levels(reasoning: str) -> None:
    model = custom_model("google-generative-ai", "gemini-3.7-flash", {"xhigh": "high", "max": "high"})
    payload = await capture_payload("google-generative-ai", model, reasoning)
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


async def test_uppercase_mapping_and_budgets() -> None:
    payload = await capture_payload(
        "google-generative-ai", custom_model("google-generative-ai", "gemini-3.7-flash", {"high": "LOW"}), "high"
    )
    assert payload["config"]["thinkingConfig"]["thinkingLevel"] == "LOW"
    payload = await capture_payload(
        "google-generative-ai",
        custom_model("google-generative-ai", "gemini-2.5-flash", {"xhigh": "high"}),
        "xhigh",
        {"high": 1234},
    )
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 1234}
    payload = await capture_payload(
        "google-vertex", custom_model("google-vertex", "gemini-3.7-flash", {"xhigh": "high"}), "xhigh"
    )
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}
    payload = await capture_payload(
        "google-vertex", custom_model("google-vertex", "gemini-2.5-flash", {"max": "high"}), "max", {"high": 4321}
    )
    assert payload["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 4321}


async def test_default_budgets() -> None:
    assert google_generative_ai.get_google_budget(google_model("gemini-2.5-pro"), "high") == 32768
    assert google_generative_ai.get_google_budget(google_model("gemini-2.5-flash-lite"), "minimal") == 512
    assert google_vertex.get_google_budget(vertex_model("gemini-2.5-flash"), "minimal") == 128
    assert google_generative_ai.get_google_budget(google_model("gemini-3-flash-preview"), "low") == -1


async def test_on_payload_replacement_is_sent(mock_server: Any) -> None:
    mock_server.add_sse([stop_chunk()])

    def on_payload(payload: Any, _model: Any) -> Any:
        return {**payload, "config": {**payload["config"], "topK": 3}}

    await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client(), "on_payload": on_payload}
    ).result()
    assert mock_server.last.json()["generationConfig"] == {"topK": 3}


# ---------------------------------------------------------------------------
# Vertex endpoints and auth (google-vertex-api-key-resolution.test.ts)
# ---------------------------------------------------------------------------

VERTEX_PATH = "publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"


@pytest.mark.parametrize("api_key", ["<authenticated>", "gcp-vertex-credentials", None])
async def test_vertex_falls_back_to_adc(mock_server: Any, fake_adc: Any, api_key: str | None) -> None:
    mock_server.add_sse([stop_chunk()])
    options = vertex_options(mock_server)
    if api_key:
        options["api_key"] = api_key
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "stop", message.error_message
    request = mock_server.last
    assert request.url == (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/{VERTEX_PATH}"
    )
    assert request.headers["authorization"] == "Bearer adc-token"
    assert "x-goog-api-key" not in request.headers
    assert request.headers["user-agent"] == PI_USER_AGENT
    assert len(fake_adc) == 1 and fake_adc[0].refreshes == 1


async def test_vertex_placeholder_env_key_uses_adc(
    mock_server: Any, fake_adc: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "<authenticated>")
    mock_server.add_sse([stop_chunk()])
    await google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server)).result()
    assert mock_server.last.headers["authorization"] == "Bearer adc-token"


async def test_vertex_api_key_express_mode(mock_server: Any, fake_adc: Any) -> None:
    mock_server.add_sse([stop_chunk()])
    message = await google_vertex.stream(
        vertex_model(),
        CONTEXT,
        {
            "api_key": "AIzaSyExampleRealisticLookingApiKey123456",
            "http_client": mock_server.client(),
            "env": {"GOOGLE_CLOUD_PROJECT": "ignored"},
        },
    ).result()
    assert message.stop_reason == "stop"
    request = mock_server.last
    assert request.url == f"https://aiplatform.googleapis.com/v1/{VERTEX_PATH}"
    assert request.headers["x-goog-api-key"] == "AIzaSyExampleRealisticLookingApiKey123456"
    assert "authorization" not in request.headers
    assert fake_adc == []


@pytest.mark.parametrize(
    ("location", "host"),
    [
        ("global", "https://aiplatform.googleapis.com"),
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("europe-west4", "https://europe-west4-aiplatform.googleapis.com"),
    ],
)
async def test_vertex_endpoints_by_location(mock_server: Any, fake_adc: Any, location: str, host: str) -> None:
    mock_server.add_sse([stop_chunk()])
    await google_vertex.stream(vertex_model(), CONTEXT, vertex_options(mock_server, location=location)).result()
    assert mock_server.last.url == f"{host}/v1/projects/test-project/locations/{location}/{VERTEX_PATH}"


@pytest.mark.parametrize("use_api_key", [False, True])
async def test_vertex_custom_base_url_is_a_collection(mock_server: Any, fake_adc: Any, use_api_key: bool) -> None:
    mock_server.add_sse([stop_chunk()])
    options = (
        {"api_key": "AIzaSyKey", "http_client": mock_server.client()} if use_api_key else vertex_options(mock_server)
    )
    await google_vertex.stream(vertex_model(base_url="https://proxy.example.com"), CONTEXT, options).result()
    assert mock_server.last.url == f"https://proxy.example.com/v1/{VERTEX_PATH}"


async def test_vertex_base_url_with_api_version(mock_server: Any, fake_adc: Any) -> None:
    mock_server.add_sse([stop_chunk()])
    base = "https://proxy.example.com/v1/projects/test-project/locations/global"
    await google_vertex.stream(vertex_model(base_url=base), CONTEXT, vertex_options(mock_server)).result()
    assert mock_server.last.url == f"{base}/{VERTEX_PATH}"


async def test_vertex_project_and_location_from_env(mock_server: Any, fake_adc: Any) -> None:
    mock_server.add_sse([stop_chunk()])
    env = {"GCLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "asia-east1"}
    await google_vertex.stream(vertex_model(), CONTEXT, {"http_client": mock_server.client(), "env": env}).result()
    assert mock_server.last.url.startswith(
        "https://asia-east1-aiplatform.googleapis.com/v1/projects/env-project/locations/asia-east1/"
    )
    assert mock_server.last.headers["authorization"] == "Bearer adc-token"


async def test_vertex_requires_project_and_location(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(name, raising=False)
    message = await google_vertex.stream(vertex_model(), CONTEXT, {}).result()
    assert message.error_message == (
        "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
    )
    message = await google_vertex.stream(vertex_model(), CONTEXT, {"project": "p"}).result()
    assert message.error_message == (
        "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
    )


# ---------------------------------------------------------------------------
# ADC credentials (google-auth through the SDK)
# ---------------------------------------------------------------------------

# google.auth.load_credentials_from_file (what keyFilename maps to) warns that it accepts any type.
_load_file_warning = pytest.mark.filterwarnings("ignore:The load_credentials_from_file method is deprecated")


@pytest.fixture
def fake_refresh(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refresh authorized_user credentials without calling oauth2.googleapis.com."""
    refreshed: list[str] = []

    def refresh(self: google.oauth2.credentials.Credentials, request: Any) -> None:
        refreshed.append(self.refresh_token or "")
        self.token = "ya29.token"
        self.expiry = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=1)

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", refresh)
    return refreshed


@_load_file_warning
async def test_vertex_adc_credentials_file(mock_server: Any, adc_file: Path, fake_refresh: list[str]) -> None:
    mock_server.add_sse([stop_chunk()])
    mock_server.add_sse([stop_chunk()])
    client = mock_server.client()
    options = {
        **vertex_options(mock_server, env={"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)}),
        "http_client": client,
    }
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "stop", message.error_message
    request = mock_server.last
    assert request.headers["authorization"] == "Bearer ya29.token"
    assert request.headers["x-goog-user-project"] == "quota-project"
    assert fake_refresh == ["refresh-token"]

    # The SDK client (and with it the credentials and their token) is reused.
    await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert len(mock_server.requests) == 2
    assert mock_server.last.headers["authorization"] == "Bearer ya29.token"
    assert fake_refresh == ["refresh-token"]


async def test_vertex_default_adc_is_loaded_once_per_client(mock_server: Any, fake_adc: list[FakeCredentials]) -> None:
    client = mock_server.client()
    for _ in range(3):
        mock_server.add_sse([stop_chunk()])
        message = await google_vertex.stream(
            vertex_model(), CONTEXT, {**vertex_options(mock_server), "http_client": client}
        ).result()
        assert message.stop_reason == "stop", message.error_message
    assert len(fake_adc) == 1 and fake_adc[0].refreshes == 1


@_load_file_warning
async def test_vertex_missing_credentials_file(tmp_path: Path) -> None:
    options = {
        "project": "p",
        "location": "us-central1",
        "env": {"GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "x")},
    }
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "error"
    assert "was not found" in (message.error_message or "")


@_load_file_warning
async def test_vertex_adc_refresh_failure_becomes_stream_error(
    mock_server: Any, adc_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refresh(self: Any, request: Any) -> None:
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", refresh)
    options = vertex_options(mock_server, env={"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)})
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "error"
    assert "invalid_grant" in (message.error_message or "")
    assert mock_server.requests == []


# ---------------------------------------------------------------------------
# SDK transport details
# ---------------------------------------------------------------------------


async def test_sdk_client_is_cached_per_http_client(mock_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Any] = []
    real_client = google_generative_ai.Client

    def counting_client(*args: Any, **kwargs: Any) -> Any:
        built.append(kwargs)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(google_generative_ai, "Client", counting_client)
    client = mock_server.client()
    for headers in ({"x-session": "a"}, {"x-session": "b"}):
        mock_server.add_sse([stop_chunk()])
        await google_generative_ai.stream(
            google_model(), CONTEXT, {"api_key": "k", "http_client": client, "headers": headers}
        ).result()
        # Request headers are sent per request, so they do not split the cache.
        assert mock_server.last.headers["x-session"] == headers["x-session"]
    assert len(built) == 1
    assert built[0]["http_options"].httpx_async_client is client

    mock_server.add_sse([stop_chunk()])
    await google_generative_ai.stream(google_model(), CONTEXT, {"api_key": "other", "http_client": client}).result()
    assert len(built) == 2
    assert mock_server.last.headers["x-goog-api-key"] == "other"


async def test_on_response_and_timeout(mock_server: Any) -> None:
    mock_server.add_sse([stop_chunk()], headers={"x-request-id": "req-1"})
    responses: list[Any] = []
    message = await google_generative_ai.stream(
        google_model(),
        CONTEXT,
        {
            "api_key": "k",
            "http_client": mock_server.client(),
            "timeout_ms": 30_500,
            "on_response": lambda response, _model: responses.append(response),
        },
    ).result()
    assert message.stop_reason == "stop"
    assert [(response.status, response.headers.get("x-request-id")) for response in responses] == [(200, "req-1")]
    # HttpOptions.timeout is also announced to the server.
    assert mock_server.last.headers["x-server-timeout"] == "31"


async def test_non_json_error_body(mock_server: Any) -> None:
    mock_server.add(httpx.Response(502, text="bad gateway", headers={"content-type": "text/html"}))
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert json.loads(message.error_message or "") == {
        "error": {"message": "bad gateway", "code": 502, "status": "Bad Gateway"}
    }


async def test_in_stream_error_is_not_retried(mock_server: Any) -> None:
    error_chunk = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
    mock_server.add_sse([error_chunk])
    mock_server.add_sse([stop_chunk()])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client(), "max_retries": 2}
    ).result()
    assert message.error_message == f"got status: UNAVAILABLE. {json.dumps(error_chunk, separators=(',', ':'))}"
    assert len(mock_server.requests) == 1


async def test_empty_stream_and_thought_signature_round_trip(mock_server: Any) -> None:
    mock_server.add_sse([])
    message = await google_generative_ai.stream(
        google_model(), CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert message.error_message == "Google stream ended without a finish reason"

    # The SDK decodes thoughtSignature to bytes; pi keeps the base64 text it received.
    mock_server.add_sse(
        [
            {
                "candidates": [
                    {"content": {"parts": [{"text": "a", "thoughtSignature": "AAAAAAAA/+8="}]}, "finishReason": "STOP"}
                ]
            }
        ]
    )
    model = google_model("gemini-3-flash-preview")
    first = await google_generative_ai.stream(
        model, CONTEXT, {"api_key": "k", "http_client": mock_server.client()}
    ).result()
    assert first.content[0].text_signature == "AAAAAAAA/+8="
    mock_server.add_sse([stop_chunk()])
    context = normalize_context(Context(messages=[UserMessage("hello"), first, UserMessage("again")]))
    await google_generative_ai.stream(model, context, {"api_key": "k", "http_client": mock_server.client()}).result()
    replayed = mock_server.last.json()["contents"][1]
    assert replayed["role"] == "model"
    [part] = replayed["parts"]
    assert part["text"] == "a"
    # The SDK re-encodes the bytes as URL-safe base64, which the API accepts for bytes fields.
    assert part["thoughtSignature"] == "AAAAAAAA_-8="
    assert base64.urlsafe_b64decode(part["thoughtSignature"]) == base64.b64decode("AAAAAAAA/+8=")


async def test_abort_closes_the_response() -> None:
    controller = AbortController()
    closed = asyncio.Event()

    async def body() -> Any:
        try:
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n\n'
            await asyncio.sleep(10)
            yield b""
        finally:
            closed.set()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})
        )
    )
    s = google_vertex.stream(
        vertex_model(), CONTEXT, {"api_key": "AIzaKey", "http_client": client, "signal": controller.signal}
    )
    async for event in s:
        if event.type == "text_delta":
            controller.abort()
    message = await s.result()
    assert message.stop_reason == "aborted"
    await asyncio.wait_for(closed.wait(), 1)


def test_google_api_error_wording() -> None:
    from google.genai import errors as genai_errors

    request = httpx.Request("POST", "https://example.test")
    body = {"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}}
    response = httpx.Response(403, json=body, request=request)
    converted = google_shared.google_api_error(genai_errors.ClientError(403, body, response))
    assert str(converted) == json.dumps(body, separators=(",", ":"))
    assert converted.status == 403
    assert converted.headers is None


# ---------------------------------------------------------------------------
# Live API (skipped without credentials)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")
async def test_live_gemini_text() -> None:
    message = await google_generative_ai.stream_simple(
        google_model(), CONTEXT, {"api_key": os.environ["GEMINI_API_KEY"], "max_tokens": 64}
    ).result()
    assert message.stop_reason == "stop", message.error_message
    assert message.response_id
