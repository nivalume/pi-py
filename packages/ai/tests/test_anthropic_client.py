"""Anthropic Messages through the official SDK: auth, headers, transport, retries, abort and errors.

Ports the offline parts of test/anthropic-auth-token.test.ts, anthropic-federation-sdk.test.ts and
anthropic-strict-tool-fallback.test.ts, plus checks of the ``httpx`` -> SDK (``httpx2``) bridge.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
from anthropic import AsyncAnthropic

from pi_ai import AbortController, Context, Model, Tool, Type, UserMessage, normalize_context
from pi_ai.api import anthropic_messages
from pi_ai.api.anthropic_messages import CLAUDE_CODE_VERSION, stream, stream_simple
from pi_ai.utils.misc import get_pi_user_agent

TEXT_EVENTS: list[Any] = [
    ("message_start", {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}}),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ),
    ("message_stop", {"type": "message_stop"}),
]


def sse_bytes(events: list[Any]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def sse_response(events: list[Any] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        200, content=sse_bytes(events or TEXT_EVENTS), headers={"content-type": "text/event-stream", **(headers or {})}
    )


def error_response(status: int, message: str, kind: str = "invalid_request_error", **headers: str) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"type": kind, "message": message}}, headers=headers)


def make_model(**overrides: Any) -> Model:
    fields: dict[str, Any] = {
        "id": "claude-test",
        "name": "Claude Test",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "reasoning": False,
        "input": ["text"],
        "context_window": 100_000,
        "max_tokens": 4096,
    }
    fields.update(overrides)
    return Model(**fields)


def context() -> Any:
    return normalize_context(Context(system_prompt="System prompt.", messages=[UserMessage("Hello", timestamp=1)]))


async def run(mock_server: Any, model: Model | None = None, **options: Any) -> Any:
    s = stream(model or make_model(), context(), {"http_client": mock_server.client(), **options})
    return await s.result()


@pytest.fixture(autouse=True)
def clean_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_WORKSPACE_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
        "PI_CACHE_RETENTION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(anthropic_messages, "_federation_cache", None)


# ---------------------------------------------------------------------------
# Auth and client headers
# ---------------------------------------------------------------------------


async def test_api_key_auth_sends_sdk_and_pi_headers(mock_server: Any) -> None:
    mock_server.add(sse_response())
    result = await run(mock_server, api_key="sk-ant-api-key")

    assert result.stop_reason == "stop"
    request = mock_server.last
    assert request.method == "POST"
    assert request.url == "https://api.anthropic.com/v1/messages?beta=true"
    headers = request.headers
    assert headers["x-api-key"] == "sk-ant-api-key"
    assert "authorization" not in headers
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["user-agent"] == get_pi_user_agent()
    assert headers["accept"] == "application/json"
    assert headers["content-type"] == "application/json"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    # SDK telemetry headers; pi retries itself, so every attempt is the SDK's first.
    assert headers["x-stainless-lang"] == "python"
    assert headers["x-stainless-retry-count"] == "0"


async def test_oauth_tokens_use_bearer_auth_and_claude_code_identity(mock_server: Any) -> None:
    mock_server.add(sse_response())
    result = await run(mock_server, api_key="sk-ant-oat01-token")

    assert result.stop_reason == "stop"
    headers = mock_server.last.headers
    assert headers["authorization"] == "Bearer sk-ant-oat01-token"
    assert "x-api-key" not in headers
    assert headers["user-agent"] == f"claude-cli/{CLAUDE_CODE_VERSION}"
    assert headers["x-app"] == "cli"
    assert headers["anthropic-beta"].split(",")[:2] == ["claude-code-20250219", "oauth-2025-04-20"]


async def test_copilot_uses_bearer_auth_and_dynamic_headers(mock_server: Any) -> None:
    mock_server.add(sse_response())
    model = make_model(provider="github-copilot", base_url="https://api.githubcopilot.com")
    result = await run(mock_server, model, api_key="copilot-token", headers={"Editor-Version": "pi/1"})

    assert result.stop_reason == "stop"
    headers = mock_server.last.headers
    assert mock_server.last.url == "https://api.githubcopilot.com/v1/messages?beta=true"
    assert headers["authorization"] == "Bearer copilot-token"
    assert "x-api-key" not in headers
    assert headers["x-initiator"] == "user"
    assert headers["openai-intent"] == "conversation-edits"
    assert headers["editor-version"] == "pi/1"


async def test_header_owned_auth_skips_sdk_credentials_and_oauth_shaping(
    mock_server: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Neither the SDK's env keys nor its federation/profile chain may add credentials.
    token_file = tmp_path / "identity.jwt"
    token_file.write_text("header.payload.signature")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-test")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))
    mock_server.add(sse_response())

    result = await run(mock_server, headers={"Authorization": "Bearer gateway-token"})

    assert result.stop_reason == "stop"
    assert len(mock_server.requests) == 1
    headers = mock_server.last.headers
    assert headers["authorization"] == "Bearer gateway-token"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
    assert mock_server.last.json()["system"][0]["text"] == "System prompt."


async def test_null_headers_remove_sdk_auth_headers(mock_server: Any) -> None:
    # Cloudflare AI Gateway shape: gateway auth only, provider auth suppressed.
    mock_server.add(sse_response())
    model = make_model(provider="cloudflare-ai-gateway", base_url="https://gateway.ai.cloudflare.com/v1/a/g/anthropic")
    result = await run(
        mock_server,
        model,
        api_key="ignored-key",
        headers={"cf-aig-authorization": "Bearer cf", "Authorization": None, "x-api-key": None},
    )

    assert result.stop_reason == "stop"
    headers = mock_server.last.headers
    assert headers["cf-aig-authorization"] == "Bearer cf"
    assert "authorization" not in headers
    assert "x-api-key" not in headers
    assert mock_server.last.url == "https://gateway.ai.cloudflare.com/v1/a/g/anthropic/v1/messages?beta=true"


async def test_explicit_headers_override_defaults_and_betas(mock_server: Any) -> None:
    mock_server.add(sse_response())
    mock_server.add(sse_response())
    model = make_model(headers={"X-Model": "m"})
    await run(
        mock_server,
        model,
        api_key="k",
        headers={
            "User-Agent": "custom-client",
            "anthropic-beta": "custom-beta",
            "anthropic-dangerous-direct-browser-access": None,
        },
    )
    headers = mock_server.last.headers
    assert headers["user-agent"] == "custom-client"
    assert headers["anthropic-beta"] == "custom-beta"
    assert headers["x-model"] == "m"
    assert "anthropic-dangerous-direct-browser-access" not in headers

    await run(
        mock_server,
        make_model(compat={"supportsEagerToolInputStreaming": False}),
        api_key="k",
        headers={"anthropic-beta": None},
    )
    assert "anthropic-beta" not in mock_server.last.headers


async def test_session_affinity_headers(mock_server: Any) -> None:
    mock_server.add(sse_response())
    mock_server.add(sse_response())
    model = make_model(provider="openrouter", base_url="https://openrouter.ai/api")
    await run(mock_server, model, api_key="k", session_id="session-1")
    assert mock_server.last.headers["x-session-id"] == "session-1"
    assert mock_server.last.url == "https://openrouter.ai/api/v1/messages?beta=true"

    await run(mock_server, model, api_key="k", session_id="session-1", cache_retention="none")
    assert "x-session-id" not in mock_server.last.headers


async def test_base_url_falls_back_to_anthropic_base_url_env(mock_server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.test/anthropic/")
    mock_server.add(sse_response())
    await run(mock_server, make_model(base_url=""), api_key="k")
    assert mock_server.last.url == "https://proxy.test/anthropic/v1/messages?beta=true"


async def test_request_fields_outside_the_sdk_signature_stay_in_the_body(mock_server: Any) -> None:
    mock_server.add(sse_response())

    def on_payload(payload: dict[str, Any], _model: Any) -> dict[str, Any]:
        return {**payload, "temperature": 0.3, "inference_geo": "us", "top_k": 5, "user_profile_id": "prof_1"}

    result = await run(mock_server, api_key="k", on_payload=on_payload, metadata={"user_id": "u"})

    assert result.stop_reason == "stop"
    body = mock_server.last.json()
    assert body["temperature"] == 0.3
    assert body["inference_geo"] == "us"
    assert body["top_k"] == 5
    assert body["metadata"] == {"user_id": "u"}
    assert body["stream"] is True
    assert "betas" not in body and "user_profile_id" not in body
    assert mock_server.last.headers["anthropic-user-profile-id"] == "prof_1"


async def test_custom_sdk_client_option(mock_server: Any) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=sse_bytes(TEXT_EVENTS), headers={"content-type": "text/event-stream"})

    client = AsyncAnthropic(
        api_key="custom-key",
        base_url="https://custom.test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    result = await stream(make_model(), context(), {"client": client}).result()

    assert result.stop_reason == "stop"
    assert str(seen[0].url) == "https://custom.test/v1/messages?beta=true"
    assert seen[0].headers["x-api-key"] == "custom-key"


async def test_custom_sdk_client_does_not_retry_on_its_own() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})

    client = AsyncAnthropic(
        api_key="custom-key",
        base_url="https://custom.test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    assert client.max_retries == 2
    result = await stream(make_model(), context(), {"client": client}).result()

    assert result.stop_reason == "error"
    assert result.error_message == '529 {"type":"error","error":{"type":"overloaded_error","message":"busy"}}'
    assert len(seen) == 1


async def test_accepts_an_httpx2_client_as_http_client() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=sse_bytes(TEXT_EVENTS), headers={"content-type": "text/event-stream"})

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    result = await stream(make_model(), context(), {"api_key": "k", "http_client": http_client}).result()
    assert result.stop_reason == "stop"
    assert seen[0].headers["x-api-key"] == "k"


async def test_decodes_compressed_responses(mock_server: Any) -> None:
    mock_server.add(
        httpx.Response(
            200,
            content=gzip.compress(sse_bytes(TEXT_EVENTS)),
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )
    )
    result = await run(mock_server, api_key="k")
    assert result.stop_reason == "stop"
    assert result.content[0].text == "Hi"


async def test_streams_chunked_bodies(mock_server: Any) -> None:
    body = sse_bytes(TEXT_EVENTS)

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for start in range(0, len(body), 7):
                yield body[start : start + 7]

    mock_server.add(httpx.Response(200, stream=Chunks(), headers={"content-type": "text/event-stream"}))
    result = await run(mock_server, api_key="k")
    assert result.stop_reason == "stop"
    assert result.content[0].text == "Hi"


# ---------------------------------------------------------------------------
# Raw response, retries, timeouts and errors
# ---------------------------------------------------------------------------


async def test_on_response_sees_the_raw_response_before_events(mock_server: Any) -> None:
    mock_server.add(sse_response(headers={"request-id": "req_1", "anthropic-ratelimit-requests-remaining": "9"}))
    order: list[str] = []

    def on_response(response: Any, _model: Any) -> None:
        order.append(f"response {response.status} {response.headers['request-id']}")

    s = stream(
        make_model(), context(), {"api_key": "k", "http_client": mock_server.client(), "on_response": on_response}
    )
    async for event in s:
        order.append(event.type)
    assert order[:2] == ["response 200 req_1", "start"]


async def test_retries_retryable_statuses_with_pi_backoff(mock_server: Any) -> None:
    mock_server.add(error_response(529, "Overloaded", "overloaded_error", **{"retry-after-ms": "1"}))
    mock_server.add(error_response(500, "boom", "api_error", **{"retry-after-ms": "1"}))
    mock_server.add(sse_response())
    result = await run(mock_server, api_key="k", max_retries=2)

    assert result.stop_reason == "stop"
    assert len(mock_server.requests) == 3
    # Each retry is a fresh SDK request.
    assert [request.headers["x-stainless-retry-count"] for request in mock_server.requests] == ["0", "0", "0"]


async def test_does_not_retry_without_max_retries_or_for_client_errors(mock_server: Any) -> None:
    mock_server.add(error_response(529, "Overloaded", "overloaded_error"))
    result = await run(mock_server, api_key="k")
    assert result.stop_reason == "error"
    assert result.error_message == '529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
    assert len(mock_server.requests) == 1

    mock_server.add(error_response(400, "bad request"))
    result = await run(mock_server, api_key="k", max_retries=3)
    assert (
        result.error_message == '400 {"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}'
    )
    assert len(mock_server.requests) == 2


async def test_server_retry_delays_above_the_limit_fail_fast(mock_server: Any) -> None:
    mock_server.add(error_response(429, "slow down", "rate_limit_error", **{"retry-after": "120"}))
    result = await run(mock_server, api_key="k", max_retries=2)
    assert result.stop_reason == "error"
    assert result.error_message.startswith("Server requested 120s retry delay (max: 60s). 429 ")
    assert len(mock_server.requests) == 1


async def test_non_json_error_bodies(mock_server: Any) -> None:
    mock_server.add(httpx.Response(502, text="Bad gateway"))
    result = await run(mock_server, api_key="k")
    assert result.error_message == "502 Bad gateway"


async def test_connection_and_timeout_errors_use_sdk_wording_and_retry(mock_server: Any) -> None:
    def connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    mock_server.add(connect_error)
    result = await run(mock_server, api_key="k")
    assert result.stop_reason == "error"
    assert result.error_message == "Connection error."

    mock_server.add(read_timeout)
    result = await run(mock_server, api_key="k")
    assert result.error_message == "Request timed out."

    mock_server.add(connect_error)
    mock_server.add(sse_response())
    result = await run(mock_server, api_key="k", max_retries=1)
    assert result.stop_reason == "stop"


async def test_timeout_ms_is_the_sdk_request_timeout(mock_server: Any) -> None:
    timeouts: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions.get("timeout"))
        return sse_response()

    mock_server.add(handler)
    await run(mock_server, api_key="k", timeout_ms=1500)
    assert mock_server.last.headers["x-stainless-timeout"] == "1.5"
    assert timeouts[0]["read"] == 1.5


async def test_strict_tool_limit_rejection_retries_with_non_strict_preferred_tools(mock_server: Any) -> None:
    grammar_error = "The compiled grammar is too large, which would cause performance issues."
    model = make_model(id="claude-haiku-4-5", compat={"supportsStrictTools": True})
    preferred = Tool(
        "preferred",
        "Look up a value",
        Type.Object({"value": Type.String()}),
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
    )
    transcript = normalize_context(Context(tools=[preferred], messages=[UserMessage("Hello", timestamp=1)]))
    mock_server.add(error_response(400, grammar_error))
    mock_server.add(sse_response())
    responses: list[Any] = []

    s = stream(
        model,
        transcript,
        {"api_key": "k", "http_client": mock_server.client(), "on_response": lambda r, _m: responses.append(r)},
    )
    result = await s.result()

    assert result.stop_reason == "stop"
    first, second = (request.json()["tools"][0] for request in mock_server.requests)
    assert first["strict"] is True
    assert "strict" not in second
    assert len(responses) == 1
    assert result.diagnostics[0].type == "anthropic_strict_tools_fallback"
    assert grammar_error in result.diagnostics[0].error.message


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------


async def test_abort_before_the_request(mock_server: Any) -> None:
    controller = AbortController()
    controller.abort()
    result = await run(mock_server, api_key="k", signal=controller.signal)
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert mock_server.requests == []


async def test_abort_mid_stream_stops_promptly_and_closes_the_response(mock_server: Any) -> None:
    controller = AbortController()
    closed = asyncio.Event()
    head = sse_bytes(TEXT_EVENTS[:3])

    class Hanging(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield head
            await asyncio.sleep(3600)
            yield b""

        async def aclose(self) -> None:
            closed.set()

    mock_server.add(httpx.Response(200, stream=Hanging(), headers={"content-type": "text/event-stream"}))

    def on_event(event: Any, _model: Any) -> None:
        if event.get("type") == "content_block_delta":
            controller.abort()

    s = stream(
        make_model(),
        context(),
        {
            "api_key": "k",
            "http_client": mock_server.client(),
            "signal": controller.signal,
            "on_provider_stream_event": on_event,
        },
    )
    result = await asyncio.wait_for(s.result(), 5)

    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert result.content[0].text == "Hi"
    assert closed.is_set()


async def test_abort_during_retry_backoff(mock_server: Any) -> None:
    controller = AbortController()
    mock_server.add(error_response(529, "Overloaded", "overloaded_error", **{"retry-after": "30"}))
    s = stream(
        make_model(),
        context(),
        {"api_key": "k", "http_client": mock_server.client(), "signal": controller.signal, "max_retries": 3},
    )
    asyncio.get_running_loop().call_later(0.05, controller.abort)
    result = await asyncio.wait_for(s.result(), 5)
    assert result.stop_reason == "aborted"
    assert len(mock_server.requests) == 1


# ---------------------------------------------------------------------------
# Workload identity federation (the SDK's token exchange and cache)
# ---------------------------------------------------------------------------


@pytest.fixture
def federation_env(tmp_path: Path) -> dict[str, str]:
    token_file = tmp_path / "identity.jwt"
    token_file.write_text("header.payload.signature")
    return {
        "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
        "ANTHROPIC_ORGANIZATION_ID": "org-test",
        "ANTHROPIC_IDENTITY_TOKEN_FILE": str(token_file),
    }


def federation_handler(exchanges: list[dict[str, Any]], messages: list[httpx.Request], statuses: list[int]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/token":
            exchanges.append(json.loads(request.content))
            token = f"federated-token-{len(exchanges)}"
            return httpx.Response(200, json={"access_token": token, "expires_in": 3600, "token_type": "Bearer"})
        messages.append(request)
        status = statuses.pop(0) if statuses else 200
        if status != 200:
            return error_response(status, "invalid token", "authentication_error")
        return sse_response()

    return handler


async def test_federation_exchanges_the_identity_token_once_across_requests(federation_env: dict[str, str]) -> None:
    exchanges: list[dict[str, Any]] = []
    messages: list[httpx.Request] = []
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(federation_handler(exchanges, messages, [])))

    for _ in range(3):
        result = await stream(make_model(), context(), {"env": federation_env, "http_client": http_client}).result()
        assert result.stop_reason == "stop", result.error_message

    assert len(exchanges) == 1
    assert exchanges[0]["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
    assert exchanges[0]["assertion"] == "header.payload.signature"
    assert exchanges[0]["federation_rule_id"] == "fdrl_test"
    assert exchanges[0]["organization_id"] == "org-test"
    assert len(messages) == 3
    for request in messages:
        assert request.headers["authorization"] == "Bearer federated-token-1"
        assert "x-api-key" not in request.headers
        assert "oauth-2025-04-20" in request.headers["anthropic-beta"]


async def test_federation_token_is_exchanged_again_after_a_401(federation_env: dict[str, str]) -> None:
    exchanges: list[dict[str, Any]] = []
    messages: list[httpx.Request] = []
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(federation_handler(exchanges, messages, [401])))
    options = {"env": federation_env, "http_client": http_client}

    first = await stream(make_model(), context(), options).result()
    assert first.stop_reason == "error"
    assert first.error_message.startswith("401 ")
    second = await stream(make_model(), context(), options).result()

    assert second.stop_reason == "stop"
    assert len(exchanges) == 2
    assert messages[-1].headers["authorization"] == "Bearer federated-token-2"


def test_stream_simple_accepts_federation_without_a_key(federation_env: dict[str, str]) -> None:
    assert stream_simple(make_model(), context(), {"env": federation_env}) is not None
    with pytest.raises(ValueError, match="No API key for provider: anthropic"):
        stream_simple(make_model(), context(), {"env": {}})
