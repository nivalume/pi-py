"""End-to-end streaming tests for the Google adapters against the in-process mock server.

Ports of upstream test/google-raw-stop-reason.test.ts, google-thinking-level-map.test.ts and
google-vertex-api-key-resolution.test.ts (which mock ``@google/genai``), plus REST-level checks
of URLs, headers, errors and ADC token loading.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

import pi_ai.auth.google_adc as google_adc
from pi_ai import AbortController, Context, Model, Tool, UserMessage, normalize_context
from pi_ai.api import google_generative_ai, google_vertex
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.utils.misc import get_pi_user_agent

CONTEXT = normalize_context(Context(messages=[UserMessage("hello")]))


@pytest.fixture(autouse=True)
def _isolate_adc(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the built-in credentials file support even when google-auth happens to be installed.
    monkeypatch.setattr(google_adc, "_import_google_auth", lambda: None)
    for name in ("GOOGLE_CLOUD_API_KEY", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    google_adc.clear_google_auth_cache()


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
    assert request.headers["user-agent"] == get_pi_user_agent()
    assert request.headers["x-goog-api-client"].startswith("google-genai-sdk/")
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
def fake_adc(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_headers(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return {"authorization": "Bearer adc-token"}

    monkeypatch.setattr(google_vertex, "get_google_auth_headers", fake_headers)
    return calls


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
    assert received == chunks
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
    assert mock_server.last.headers["user-agent"] == get_pi_user_agent()
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
    assert mock_server.last.headers["user-agent"] == "custom-agent"
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
    assert request.headers["user-agent"] == get_pi_user_agent()
    assert len(fake_adc) == 1


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
    assert fake_adc[0]["env"] == env


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
# ADC token loading
# ---------------------------------------------------------------------------


async def test_vertex_adc_authorized_user_file(mock_server: Any, adc_file: Path) -> None:
    mock_server.add(token_response())
    mock_server.add_sse([stop_chunk()])
    mock_server.add_sse([stop_chunk()])
    options = vertex_options(mock_server, env={"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)})
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "stop", message.error_message

    token_request, model_request = mock_server.requests
    assert token_request.url == "https://oauth2.googleapis.com/token"
    assert token_request.headers["content-type"] == "application/x-www-form-urlencoded"
    form = dict(pair.split("=", 1) for pair in token_request.body.decode().split("&"))
    assert form == {
        "grant_type": "refresh_token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "refresh_token": "refresh-token",
    }
    assert model_request.headers["authorization"] == "Bearer ya29.token"
    assert model_request.headers["x-goog-user-project"] == "quota-project"

    # The token is cached until it nears expiry.
    await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert len(mock_server.requests) == 3
    assert mock_server.last.headers["authorization"] == "Bearer ya29.token"


async def test_adc_token_refreshes_when_expiring(mock_server: Any, adc_file: Path) -> None:
    mock_server.add(httpx.Response(200, json={"access_token": "short", "expires_in": 60}))
    mock_server.add(httpx.Response(200, json={"access_token": "second", "expires_in": 3600}))
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)}
    first = await google_adc.get_google_auth_headers(env=env, http_client=mock_server.client())
    second = await google_adc.get_google_auth_headers(env=env, http_client=mock_server.client())
    assert first["authorization"] == "Bearer short"
    assert second["authorization"] == "Bearer second"
    assert await google_adc.get_google_access_token(env=env, http_client=mock_server.client()) == "second"
    assert len(mock_server.requests) == 2


async def test_adc_well_known_file(mock_server: Any, adc_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    config_dir = adc_file.parent / "gcloud"
    config_dir.mkdir()
    os.replace(adc_file, config_dir / "application_default_credentials.json")
    mock_server.add(token_response())
    headers = await google_adc.get_google_auth_headers(
        env={"CLOUDSDK_CONFIG": str(config_dir)}, http_client=mock_server.client()
    )
    assert headers["authorization"] == "Bearer ya29.token"


async def test_adc_errors(mock_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(google_adc.GoogleAuthError, match="Could not load the default credentials"):
        await google_adc.get_google_auth_headers(env={"CLOUDSDK_CONFIG": str(tmp_path / "none")})
    with pytest.raises(google_adc.GoogleAuthError, match="does not exist"):
        await google_adc.get_google_auth_headers(env={"GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "missing.json")})

    external = tmp_path / "external.json"
    external.write_text(json.dumps({"type": "external_account"}))
    with pytest.raises(google_adc.GoogleAuthError, match="pip install google-auth"):
        await google_adc.get_google_auth_headers(env={"GOOGLE_APPLICATION_CREDENTIALS": str(external)})

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"type": "authorized_user", "client_id": "x"}))
    with pytest.raises(google_adc.GoogleAuthError, match="does not contain a client_secret field"):
        await google_adc.get_google_auth_headers(env={"GOOGLE_APPLICATION_CREDENTIALS": str(incomplete)})


async def test_adc_token_endpoint_failures(mock_server: Any, adc_file: Path) -> None:
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)}
    mock_server.add_json(
        {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}, status=400
    )
    with pytest.raises(Exception, match="invalid_grant: Token has been expired or revoked."):
        await google_adc.get_google_auth_headers(env=env, http_client=mock_server.client())

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect ECONNREFUSED", request=request)

    mock_server.add(refuse)
    with pytest.raises(
        google_adc.GoogleAuthError, match=r"request to https://oauth2.googleapis.com/token failed, reason:"
    ):
        await google_adc.get_google_auth_headers(env=env, http_client=mock_server.client())


async def test_vertex_adc_failure_becomes_stream_error(mock_server: Any, adc_file: Path) -> None:
    mock_server.add_json({"error": "invalid_grant"}, status=400)
    options = vertex_options(mock_server, env={"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)})
    message = await google_vertex.stream(vertex_model(), CONTEXT, options).result()
    assert message.stop_reason == "error"
    assert "invalid_grant" in (message.error_message or "")


def _have_rsa_signer() -> bool:
    try:
        import cryptography  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        try:
            import google.auth.crypt  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            return False
    return True


@pytest.mark.skipif(_have_rsa_signer(), reason="an RSA signer is installed")
async def test_service_account_without_signer(tmp_path: Path) -> None:
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps({"type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com", "private_key": "x"})
    )
    with pytest.raises(google_adc.GoogleAuthError, match=r"pip install google-auth"):
        await google_adc.get_google_auth_headers(env={"GOOGLE_APPLICATION_CREDENTIALS": str(path)})


@pytest.mark.skipif(not _have_rsa_signer(), reason="needs cryptography or google-auth")
async def test_service_account_jwt_grant(mock_server: Any, tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization  # type: ignore[import-not-found]
    from cryptography.hazmat.primitives.asymmetric import rsa  # type: ignore[import-not-found]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps({"type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com", "private_key": pem})
    )
    mock_server.add(token_response())
    headers = await google_adc.get_google_auth_headers(
        env={"GOOGLE_APPLICATION_CREDENTIALS": str(path)}, http_client=mock_server.client()
    )
    assert headers == {"authorization": "Bearer ya29.token"}
    assert b"grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer" in mock_server.last.body


async def test_google_auth_library_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import types

    class FakeCredentials:
        def __init__(self) -> None:
            self.valid = False
            self.refreshed = 0

        def refresh(self, request: Any) -> None:
            self.refreshed += 1
            self.valid = True

        def apply(self, headers: dict[str, str]) -> None:
            headers["authorization"] = "Bearer from-google-auth"

    credentials = FakeCredentials()
    transport = types.SimpleNamespace(Request=object, Response=object)
    fake = types.SimpleNamespace(
        default=lambda scopes: (credentials, "project"),
        load_credentials_from_file=lambda path, scopes: (credentials, None),
        transport=transport,
        exceptions=types.SimpleNamespace(TransportError=OSError),
    )
    monkeypatch.setattr(google_adc, "_import_google_auth", lambda: fake)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert await google_adc.get_google_auth_headers() == {"authorization": "Bearer from-google-auth"}
    assert await google_adc.get_google_auth_headers() == {"authorization": "Bearer from-google-auth"}
    assert credentials.refreshed == 1


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
