"""Wire-level tests for the Bedrock adapter's AWS SDK (boto3) transport.

A real boto3 ``bedrock-runtime`` client signs and sends each request; a botocore
``before-send`` hook captures it and returns a canned response, with event-stream
frames encoded by :func:`encode_event_stream_message`. Payload and event conversion
are covered with a fake client in ``test_bedrock_converse_stream.py``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import struct
import sys
import threading
import zlib
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import EndpointConnectionError

from pi_ai import AbortController, AbortSignal, Context, Model, normalize_context
from pi_ai.api import bedrock_converse_stream as bcs
from pi_ai.api.bedrock_converse_stream import BOTO3_INSTALL_HINT, BedrockCredentials, stream, stream_simple
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.types import AssistantMessage, ImageContent, TextContent, ToolResultMessage, UserMessage
from pi_ai.utils.overflow import is_context_overflow

_AWS_ENV = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEDROCK_SKIP_AUTH",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_USE_FIPS_ENDPOINT",
    "AWS_RETRY_MODE",
    "AWS_MAX_ATTEMPTS",
    "AWS_AUTH_SCHEME_PREFERENCE",
    "HTTPS_PROXY",
    "https_proxy",
    "PI_CACHE_RETENTION",
)


@pytest.fixture(autouse=True)
def aws_env(monkeypatch, tmp_path):
    for name in _AWS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDTEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")


# -- event-stream encoding + fake HTTP ---------------------------------------------------------------


def encode_event_stream_message(headers: dict[str, str], payload: bytes = b"") -> bytes:
    """An ``application/vnd.amazon.eventstream`` frame with string headers."""
    encoded = b"".join(
        bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(value.encode())) + value.encode()
        for name, value in headers.items()
    )
    prelude = struct.pack(">II", 12 + len(encoded) + len(payload) + 4, len(encoded))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + encoded + payload
    return message + struct.pack(">I", zlib.crc32(message))


def event(event_type: str, payload: dict[str, Any]) -> bytes:
    return encode_event_stream_message(
        {":message-type": "event", ":event-type": event_type, ":content-type": "application/json"},
        json.dumps(payload).encode(),
    )


def exception(exception_type: str, payload: dict[str, Any]) -> bytes:
    return encode_event_stream_message(
        {":message-type": "exception", ":exception-type": exception_type, ":content-type": "application/json"},
        json.dumps(payload).encode(),
    )


def items_to_frames(items: list[dict[str, Any]]) -> list[bytes]:
    frames = []
    for item in items:
        ((name, payload),) = item.items()
        if name.endswith("Exception"):
            frames.append(exception(name, payload))
        else:
            frames.append(event(name, payload))
    return frames


class FakeRaw:
    """Stands in for the urllib3 response botocore reads the body from."""

    def __init__(self, chunks: list[bytes], gate: threading.Event | None = None) -> None:
        self._chunks = chunks
        self._gate = gate
        self.closed = False
        self.shut_down = False

    def stream(self, *_args: Any, **_kwargs: Any) -> Iterator[bytes]:
        yield from self._chunks
        if self._gate is not None:
            # Block like a socket read until shutdown() interrupts it.
            self._gate.wait(10)
            if self.shut_down:
                raise ConnectionResetError("socket shut down")

    def read(self, *_args: Any, **_kwargs: Any) -> bytes:
        return b"".join(self._chunks)

    def shutdown(self) -> None:
        self.shut_down = True
        if self._gate is not None:
            self._gate.set()

    def close(self) -> None:
        self.closed = True


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], chunks: list[bytes], gate: threading.Event | None = None):
        self.status = status
        self.headers = headers
        self.raw = FakeRaw(chunks, gate)


class WireRequest:
    def __init__(self, request: Any) -> None:
        self.url: str = request.url
        self.headers = {
            name.lower(): value.decode() if isinstance(value, bytes) else value
            for name, value in request.headers.items()
        }
        body = request.body
        self.body: bytes = body if isinstance(body, bytes) else (body or b"")

    def json(self) -> Any:
        return json.loads(self.body)


class Wire:
    """Queue of canned responses for the boto3 client; records every request."""

    def __init__(self) -> None:
        self.responses: list[FakeResponse | Callable[[WireRequest], FakeResponse]] = []
        self.requests: list[WireRequest] = []
        self.keys: list[Any] = []

    def add(self, response: FakeResponse | Callable[[WireRequest], FakeResponse]) -> None:
        self.responses.append(response)

    @property
    def last(self) -> WireRequest:
        return self.requests[-1]

    def handler(self, request: Any, **_kwargs: Any) -> AWSResponse:
        recorded = WireRequest(request)
        self.requests.append(recorded)
        item = self.responses.pop(0) if self.responses else FakeResponse(500, {}, [b'{"message":"no response"}'])
        response = item(recorded) if callable(item) else item
        return AWSResponse(request.url, response.status, response.headers, response.raw)


@pytest.fixture
def wire(monkeypatch):
    fake = Wire()
    original = bcs._new_client

    def new_client(key: Any) -> Any:
        client = original(key)
        fake.keys.append(key)
        client.meta.events.register("before-send.bedrock-runtime.ConverseStream", fake.handler)
        return client

    monkeypatch.setattr(bcs, "_new_client", new_client)
    # No real backoff between SDK retries.
    monkeypatch.setattr("botocore.retries.standard.ExponentialBackoff.delay_amount", lambda *_a, **_k: 0)
    bcs._client_cache.clear()
    yield fake
    bcs._client_cache.clear()


def eventstream_response(
    items: list[dict[str, Any]], headers: dict[str, str] | None = None, chunked: bool = False
) -> FakeResponse:
    data = b"".join(items_to_frames(items))
    chunks = [data[index : index + 7] for index in range(0, len(data), 7)] if chunked else [data]
    return FakeResponse(
        200,
        {"content-type": "application/vnd.amazon.eventstream", "x-amzn-requestid": "request-id", **(headers or {})},
        chunks,
    )


def simple_stop(stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    return [{"messageStart": {"role": "assistant"}}, {"messageStop": {"stopReason": stop_reason}}]


def _model(model_id: str = "us.anthropic.claude-opus-4-8") -> Model:
    model = get_builtin_model("amazon-bedrock", model_id)
    assert model is not None
    return model


def _hello():
    return normalize_context(Context(messages=[UserMessage("hello")]))


NOVA_MODEL = Model(
    id="amazon.nova-lite-v1:0",
    name="Nova Lite",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    reasoning=False,
    input=["text", "image"],
    context_window=300000,
    max_tokens=5000,
)


async def _stream(model: Model | None = None, **options: Any) -> AssistantMessage:
    return await stream(model or _model(), _hello(), {"cache_retention": "none", **options}).result()


# -- requests ------------------------------------------------------------------------------------------


async def test_sends_signed_converse_stream_request(wire, keys):
    wire.add(
        eventstream_response(
            [
                *simple_stop()[:1],
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
                simple_stop()[1],
            ]
        )
    )
    model = dataclasses.replace(_model(), id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    message = await _stream(model)

    assert message.stop_reason == "stop", message.error_message
    assert message.content == [TextContent("hi")]
    request = wire.last
    assert request.url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream"
    )
    assert request.headers["content-type"] == "application/json"
    authorization = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
    assert "/us-east-1/bedrock/aws4_request" in authorization
    assert request.headers["user-agent"].startswith("Boto3/")
    body = request.json()
    assert "modelId" not in body
    assert body["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert body["inferenceConfig"] == {"maxTokens": model.max_tokens}


async def test_image_bytes_are_base64_on_the_wire(wire, keys):
    messages = [
        ToolResultMessage("call-1", "codemode", [TextContent("Script failed"), ImageContent("AAAA", "image/png")], True)
    ]
    wire.add(eventstream_response(simple_stop()))
    await stream(NOVA_MODEL, normalize_context(Context(messages=messages)), {"cache_retention": "none"}).result()
    content = wire.last.json()["messages"][0]["content"][0]["toolResult"]["content"][1]
    assert content == {"image": {"source": {"bytes": "AAAA"}, "format": "png"}}


async def test_bearer_token_auth(wire):
    wire.add(eventstream_response(simple_stop()))
    message = await _stream(api_key="bedrock-api-key")
    assert message.stop_reason == "stop", message.error_message
    assert wire.last.headers["authorization"] == "Bearer bedrock-api-key"


async def test_bearer_token_from_environment(wire, keys, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-token")
    wire.add(eventstream_response(simple_stop()))
    await _stream()
    assert wire.last.headers["authorization"] == "Bearer env-token"
    wire.add(eventstream_response(simple_stop()))
    await _stream(env={"AWS_BEARER_TOKEN_BEDROCK": "scoped-token"})
    assert wire.last.headers["authorization"] == "Bearer scoped-token"


async def test_skip_auth_signs_with_dummy_credentials(wire, monkeypatch):
    # botocore would otherwise prefer the ambient bearer token.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-token")
    wire.add(eventstream_response(simple_stop()))
    await _stream(api_key="ignored", env={"AWS_BEDROCK_SKIP_AUTH": "1"})
    assert wire.last.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=dummy-access-key/")


async def test_explicit_credentials_option(wire, keys):
    async def provider() -> BedrockCredentials:
        return BedrockCredentials("AKIDEXPLICIT", "s", "token")

    wire.add(eventstream_response(simple_stop()))
    await _stream(credentials=provider)
    headers = wire.last.headers
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXPLICIT/")
    assert headers["x-amz-security-token"] == "token"


async def test_custom_headers_are_signed_and_reserved_headers_skipped(wire, keys):
    wire.add(eventstream_response(simple_stop()))
    await _stream(
        headers={
            "x-custom": "v",
            "User-Agent": "custom-agent",
            "authorization": "evil",
            "x-amz-date": "evil",
            "Authorization": "evil2",
            "X-Amz-Date": "evil2",
            "HOST": "evil3",
            "x-removed": None,
        },
    )
    headers = wire.last.headers
    assert headers["x-custom"] == "v"
    assert headers["user-agent"] == "custom-agent"
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert headers["x-amz-date"] not in ("evil", "evil2")
    assert headers.get("host") in (None, "bedrock-runtime.us-east-1.amazonaws.com")
    assert "x-removed" not in headers
    signed = headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
    assert "x-custom" in signed


async def test_custom_headers_apply_to_every_attempt_and_do_not_leak(wire, keys):
    wire.add(FakeResponse(503, {"x-amzn-errortype": "ServiceUnavailableException"}, [b'{"message":"busy"}']))
    wire.add(eventstream_response(simple_stop()))
    message = await _stream(headers={"x-custom": "v"})
    assert message.stop_reason == "stop", message.error_message
    assert [request.headers.get("x-custom") for request in wire.requests] == ["v", "v"]
    # The cached client is reused without the previous call's headers.
    wire.add(eventstream_response(simple_stop()))
    await _stream()
    assert "x-custom" not in wire.last.headers
    assert len(wire.keys) == 1


async def test_stream_simple_forwards_headers(wire, keys):
    wire.add(eventstream_response(simple_stop()))
    await stream_simple(_model(), _hello(), {"headers": {"x-custom": "v"}}).result()
    assert wire.last.headers["x-custom"] == "v"


# -- region, endpoint and credentials resolution ---------------------------------------------------------


async def test_request_url_follows_resolved_region_and_endpoint(wire, keys, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    wire.add(eventstream_response(simple_stop()))
    await _stream()
    assert wire.last.url.startswith("https://bedrock-runtime.us-east-2.amazonaws.com/model/")
    assert "/us-east-2/bedrock/aws4_request" in wire.last.headers["authorization"]

    model = dataclasses.replace(_model(), base_url="https://bedrock-vpc.example.com/prefix/")
    wire.add(eventstream_response(simple_stop()))
    await _stream(model)
    assert wire.last.url == "https://bedrock-vpc.example.com/prefix/model/us.anthropic.claude-opus-4-8/converse-stream"

    monkeypatch.setenv("AWS_REGION", "cn-north-1")
    wire.add(eventstream_response(simple_stop()))
    await _stream()
    assert wire.last.url.startswith("https://bedrock-runtime.cn-north-1.amazonaws.com.cn/")


async def test_region_and_credentials_from_shared_profile(wire, monkeypatch, tmp_path):
    (tmp_path / "config").write_text("[profile work]\nregion = eu-west-3\n")
    (tmp_path / "credentials").write_text("[work]\naws_access_key_id = AKIDPROFILE\naws_secret_access_key = s\n")
    monkeypatch.setenv("AWS_PROFILE", "work")
    wire.add(eventstream_response(simple_stop()))
    message = await _stream()
    assert message.stop_reason == "stop", message.error_message
    assert wire.last.url.startswith("https://bedrock-runtime.eu-west-3.amazonaws.com/")
    assert wire.last.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDPROFILE/20")


async def test_scoped_profile_wins_over_ambient_keys(wire, keys, tmp_path):
    (tmp_path / "credentials").write_text("[scoped]\naws_access_key_id = AKIDSCOPED\naws_secret_access_key = s\n")
    wire.add(eventstream_response(simple_stop()))
    await _stream(env={"AWS_PROFILE": "scoped"})
    assert wire.last.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDSCOPED/")


async def test_region_missing_error(wire, tmp_path, monkeypatch):
    (tmp_path / "config").write_text("[profile noregion]\n")
    monkeypatch.setenv("AWS_PROFILE", "noregion")
    message = await _stream()
    assert message.stop_reason == "error"
    assert message.error_message == "Region is missing"


async def test_missing_credentials_error(wire):
    message = await _stream()
    assert message.error_message == "Could not load credentials from any providers"
    assert not wire.requests


# -- client configuration and cache -------------------------------------------------------------------


async def test_clients_are_cached_by_configuration(wire, keys):
    for _ in range(3):
        wire.add(eventstream_response(simple_stop()))
    await _stream()
    await _stream()
    await _stream(region="eu-west-1")
    assert len(wire.requests) == 3
    assert len(wire.keys) == 2
    assert wire.keys[0].region == "us-east-1"
    assert wire.keys[1].region == "eu-west-1"


async def test_sdk_retry_and_timeout_configuration(wire, keys):
    wire.add(eventstream_response(simple_stop()))
    await _stream(max_retries=5, timeout_ms=1500)
    client = next(iter(bcs._client_cache.values()))
    config = client.meta.config
    assert config.retries == {"total_max_attempts": 6, "mode": "standard"}
    assert (config.connect_timeout, config.read_timeout) == (1.5, 1.5)


async def test_scoped_proxy_configuration(wire, keys):
    wire.add(eventstream_response(simple_stop()))
    await _stream(env={"HTTPS_PROXY": "http://proxy.local:8080"})
    client = next(iter(bcs._client_cache.values()))
    assert client.meta.config.proxies == {"http": "http://proxy.local:8080", "https": "http://proxy.local:8080"}


async def test_missing_boto3_ends_the_stream_with_install_hint(monkeypatch, keys):
    bcs._client_cache.clear()
    monkeypatch.setitem(sys.modules, "boto3", None)
    message = await _stream()
    assert message.stop_reason == "error"
    assert message.error_message == BOTO3_INSTALL_HINT
    assert 'pip install "pi-ai-python[bedrock]"' in BOTO3_INSTALL_HINT


# -- response stream -------------------------------------------------------------------------------------


async def test_decodes_chunked_event_stream(wire, keys):
    arguments = '{"path":"/workspace/file.js","edits":[{"oldText":"a","newText":"b","":""}]}'
    wire.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant", "p": "abcdefgh"}},
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "t", "name": "edit"}},
                    }
                },
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments[:20]}}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments[20:]}}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"unknownFutureEvent": {"x": 1}},
                {"messageStop": {"stopReason": "tool_use"}},
                {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}}},
            ],
            chunked=True,
        )
    )
    received: list[Any] = []
    message = await _stream(on_provider_stream_event=lambda item, _m: received.append(item))
    assert message.stop_reason == "toolUse", message.error_message
    assert message.content[0].arguments == json.loads(arguments)
    assert received[0] == {"messageStart": {"role": "assistant"}}  # padding is not a modeled member
    assert len(received) == 7
    assert (message.usage.input, message.usage.output, message.usage.total_tokens) == (3, 4, 7)


async def test_redacted_reasoning_bytes(wire):
    data = b"\x00\x01opaque\xff"
    wire.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"reasoningContent": {"redactedContent": base64.b64encode(data).decode()}},
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    message = await _stream(api_key="k")
    assert message.content[0].redacted is True
    assert message.content[0].thinking_signature == base64.b64encode(data).decode()


async def test_forwards_raw_response_headers_to_on_response(wire, keys):
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    wire.add(
        FakeResponse(
            200,
            {
                "content-type": "application/vnd.amazon.eventstream",
                "x-bifrost-provider": "bedrock",
                "x-bifrost-resolved-model": model_id,
                "x-amzn-requestid": "req-123",
            },
            [],
        )
    )
    responses: list[Any] = []
    result = await _stream(_model(model_id), on_response=lambda r, _m: responses.append(r))
    # The empty event stream fails, but the header callback fired before stream consumption.
    assert result.stop_reason == "error"
    assert result.error_message == "Bedrock stream ended without a stop reason"
    assert len(responses) == 1
    assert responses[0].status == 200
    assert responses[0].headers["x-amzn-requestid"] == "req-123"
    assert responses[0].headers["x-bifrost-provider"] == "bedrock"
    assert responses[0].headers["x-bifrost-resolved-model"] == model_id


async def test_corrupted_event_stream_is_an_error(wire, keys):
    frame = bytearray(event("messageStart", {"role": "assistant"}))
    frame[-1] ^= 0xFF
    wire.add(FakeResponse(200, {"content-type": "application/vnd.amazon.eventstream"}, [bytes(frame)]))
    result = await _stream()
    assert result.stop_reason == "error"
    assert "checksum" in (result.error_message or "").lower()


async def test_abort_mid_stream_interrupts_the_worker(wire, keys):
    gate = threading.Event()
    response = FakeResponse(
        200,
        {"content-type": "application/vnd.amazon.eventstream"},
        [
            event("messageStart", {"role": "assistant"}),
            event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "partial"}}),
        ],
        gate,
    )
    wire.add(response)
    controller = AbortController()
    s = stream(_model(), _hello(), {"cache_retention": "none", "signal": controller.signal})
    async for item in s:
        if item.type == "text_delta":
            controller.abort()
    message = await s.result()
    assert message.stop_reason == "aborted"
    assert message.content == [TextContent("partial")]
    for _ in range(200):
        if response.raw.closed and not any(thread.name == "pi-ai-bedrock" for thread in threading.enumerate()):
            break
        await asyncio.sleep(0.01)
    assert response.raw.shut_down
    assert response.raw.closed
    assert not any(thread.name == "pi-ai-bedrock" for thread in threading.enumerate())


async def test_already_aborted_signal_sends_nothing(wire, keys):
    message = await _stream(signal=AbortSignal.abort())
    assert message.stop_reason == "aborted"
    assert not wire.requests


# -- errors ----------------------------------------------------------------------------------------------

DIAGNOSTIC_TYPE = "bedrock_response_failure"
VALIDATION_MESSAGE = "The provided model identifier is invalid."
REQUEST_ID = "11111111-2222-3333-4444-555555555555"


def _diagnostic(message: AssistantMessage) -> Any:
    return next((d for d in message.diagnostics or [] if d.type == DIAGNOSTIC_TYPE), None)


def _error_response(status: int, error_type: str | None, body: Any, request_id: str = REQUEST_ID) -> FakeResponse:
    headers = {"x-amzn-requestid": request_id}
    if error_type is not None:
        headers["x-amzn-errortype"] = error_type
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return FakeResponse(status, headers, [content])


async def _run_bedrock(**options: Any) -> AssistantMessage:
    return await _stream(max_retries=0, **options)


async def test_records_status_code_and_request_id_for_http_errors(wire, keys):
    wire.add(
        _error_response(
            400,
            "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/",
            {"message": VALIDATION_MESSAGE},
        )
    )
    message = await _run_bedrock()
    diagnostic = _diagnostic(message)
    assert message.stop_reason == "error"
    assert message.error_message == f"Validation error: {VALIDATION_MESSAGE}"
    assert diagnostic.details == {"status": 400, "errorCode": "ValidationException", "requestId": REQUEST_ID}
    assert diagnostic.error is None


async def test_context_overflow_validation_error(wire, keys):
    wire.add(_error_response(400, "ValidationException", {"message": "Input is too long for requested model."}))
    message = await _run_bedrock()
    assert message.error_message == "Validation error: Input is too long for requested model."
    assert is_context_overflow(message, 200_000)


async def test_mid_stream_modeled_exception(wire, keys):
    wire.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"throttlingException": {"message": "Too many tokens, please wait before trying again."}},
            ],
            headers={"x-amzn-requestid": REQUEST_ID},
        )
    )
    received: list[Any] = []
    message = await _run_bedrock(on_provider_stream_event=lambda item, _m: received.append(item))
    assert message.stop_reason == "error"
    assert message.error_message == "Throttling error: Too many tokens, please wait before trying again."
    # Throttling must not be mistaken for a context overflow ("too many tokens").
    assert not is_context_overflow(message, 200_000)
    assert _diagnostic(message).details == {"errorCode": "ThrottlingException", "requestId": REQUEST_ID}
    assert isinstance(received[-1]["throttlingException"], bcs.BedrockServiceError)


async def test_unmodeled_mid_stream_error(wire, keys):
    frames = event("messageStart", {"role": "assistant"}) + encode_event_stream_message(
        {
            ":message-type": "error",
            ":error-code": "ModelStreamErrorException",
            ":error-message": "Model stream terminated unexpectedly.",
        }
    )
    wire.add(FakeResponse(200, {"x-amzn-requestid": REQUEST_ID}, [frames]))
    message = await _run_bedrock()
    assert message.error_message == "Model stream terminated unexpectedly."
    assert _diagnostic(message).details == {"errorCode": "ModelStreamErrorException", "requestId": REQUEST_ID}


async def test_unknown_exception_frame(wire, keys):
    # botocore drops the code and body of exception types missing from its service model.
    frames = event("messageStart", {"role": "assistant"}) + exception("fooException", {"message": "foo"})
    wire.add(FakeResponse(200, {"x-amzn-requestid": REQUEST_ID}, [frames]))
    message = await _run_bedrock()
    assert message.stop_reason == "error"
    assert message.error_message == "UnknownError"
    assert _diagnostic(message).details == {"requestId": REQUEST_ID}


async def test_transport_failure_has_no_provider_code(wire, keys):
    def fail(request: WireRequest) -> FakeResponse:
        raise EndpointConnectionError(endpoint_url=request.url)

    wire.add(fail)
    message = await _run_bedrock()
    assert message.stop_reason == "error"
    assert message.error_message.startswith("Could not connect to the endpoint URL")
    assert _diagnostic(message) is None


async def test_no_diagnostic_for_aborted_turn(wire, keys):
    wire.add(_error_response(400, "ValidationException", {"message": VALIDATION_MESSAGE}))
    message = await _run_bedrock(signal=AbortSignal.abort())
    assert message.stop_reason == "aborted"
    assert _diagnostic(message) is None


async def test_drops_over_long_header_values(wire, keys):
    wire.add(_error_response(400, f"{'E' * 5000}Exception", {"message": VALIDATION_MESSAGE}, "R" * 5000))
    message = await _run_bedrock()
    assert _diagnostic(message).details == {"status": 400}


async def test_unknown_error_code_placeholder_and_raw_bodies(wire, keys):
    wire.add(_error_response(403, None, {"message": "blocked by gw"}))
    message = await _run_bedrock()
    assert message.error_message == "Unknown: blocked by gw"
    assert _diagnostic(message).details == {"status": 403, "requestId": REQUEST_ID}

    wire.add(_error_response(403, None, b"<html>blocked</html>"))
    message = await _run_bedrock()
    assert message.error_message == "Unknown: 403: <html>blocked</html>"


async def test_data_retention_hint(wire, keys):
    wire.add(
        _error_response(
            400, "ValidationException", {"message": "data retention mode 'default' is not available for this model"}
        )
    )
    message = await _run_bedrock()
    assert message.error_message.endswith(
        "See https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html for supported data retention modes."
    )


async def test_sdk_retries_throttling_by_default(wire, keys):
    wire.add(_error_response(429, "ThrottlingException", {"message": "slow down"}))
    wire.add(eventstream_response(simple_stop()))
    message = await _stream()
    assert message.stop_reason == "stop", message.error_message
    assert len(wire.requests) == 2


async def test_sdk_gives_up_after_three_attempts(wire, keys):
    for _ in range(3):
        wire.add(_error_response(429, "ThrottlingException", {"message": "slow down"}))
    message = await _stream()
    assert message.error_message == "Throttling error: slow down"
    assert len(wire.requests) == 3


async def test_max_retries_zero_disables_sdk_retries(wire, keys):
    wire.add(_error_response(429, "ThrottlingException", {"message": "slow down"}))
    message = await _run_bedrock()
    assert message.error_message == "Throttling error: slow down"
    assert len(wire.requests) == 1
