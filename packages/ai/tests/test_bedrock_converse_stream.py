"""Offline tests for the Bedrock adapter (ports of upstream ``test/bedrock-*.test.ts``).

Upstream mocks the AWS SDK; here the ConverseStream HTTP call is served by ``mock_server``
with binary event-stream fixtures built by :func:`encode_event_stream_message`.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from pi_ai import AbortSignal, Context, Model, Tool, Type, normalize_context
from pi_ai.api.bedrock_converse_stream import (
    BedrockServiceError,
    resolve_bedrock_client_config,
    stream,
    stream_simple,
)
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils import aws
from pi_ai.utils.aws import AwsCredentials, encode_event_stream_message
from pi_ai.utils.overflow import is_context_overflow
from pi_ai.utils.retry import is_retryable_assistant_error

_AWS_ENV = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEDROCK_SKIP_AUTH",
    "AWS_BEDROCK_FORCE_CACHE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_USE_FIPS_ENDPOINT",
    "PI_CACHE_RETENTION",
)


@pytest.fixture(autouse=True)
def aws_env(monkeypatch, tmp_path):
    for name in _AWS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(aws, "_botocore_available", lambda: False)
    aws.clear_credentials_cache()
    yield
    aws.clear_credentials_cache()


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDTEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")


# -- fixtures ---------------------------------------------------------------------------------------


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


def eventstream_response(
    items: list[dict[str, Any]], headers: dict[str, str] | None = None, chunked: bool = False
) -> httpx.Response:
    data = b"".join(items_to_frames(items))
    content: Any = data
    if chunked:

        async def chunks():
            for index in range(0, len(data), 7):
                yield data[index : index + 7]

        content = chunks()
    return httpx.Response(
        200,
        content=content,
        headers={
            "content-type": "application/vnd.amazon.eventstream",
            "x-amzn-requestid": "request-id",
            **(headers or {}),
        },
    )


def simple_stop(stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    return [{"messageStart": {"role": "assistant"}}, {"messageStop": {"stopReason": stop_reason}}]


def b64(data: dict[str, Any]) -> dict[str, Any]:
    return data


def _model(model_id: str = "us.anthropic.claude-opus-4-8") -> Model:
    model = get_builtin_model("amazon-bedrock", model_id)
    assert model is not None
    return model


def _hello():
    return normalize_context(Context(messages=[UserMessage("hello")]))


BASE_MODEL = Model(
    id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    name="Claude Sonnet 4.5 (US)",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    reasoning=True,
    input=["text", "image"],
    context_window=200000,
    max_tokens=64000,
    compat={"supportsStrictMode": True},
)
NOVA_MODEL = dataclasses.replace(BASE_MODEL, id="amazon.nova-lite-v1:0", name="Nova Lite", reasoning=False, compat=None)
GPT_MODEL = Model(
    id="global.openai.gpt-5.6-terra",
    name="GPT-5.6 Terra (Global)",
    api="bedrock-converse-stream",
    provider="amazon-bedrock",
    base_url="https://bedrock-runtime.ap-northeast-1.amazonaws.com",
    reasoning=True,
    context_window=400000,
    max_tokens=128000,
)


def _assistant(content: list[Any], model: Model = BASE_MODEL, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content, api="bedrock-converse-stream", provider="amazon-bedrock", model=model.id, stop_reason=stop_reason
    )


class PayloadCaptured(Exception):
    pass


async def capture_payload(
    context: Context | list[Any], model: Model = BASE_MODEL, raise_after: bool = False, **options: Any
) -> dict[str, Any]:
    """Capture the command input; the already-aborted signal stops the request (like upstream)."""
    if isinstance(context, list):
        context = Context(messages=context)
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _model: Any) -> Any:
        captured.update(payload)
        if raise_after:
            raise PayloadCaptured()
        return payload

    opts: dict[str, Any] = {"cache_retention": "none", "signal": AbortSignal.abort(), "on_payload": on_payload}
    opts.update(options)
    if raise_after:
        opts.pop("signal")
    message = await stream(model, normalize_context(context), opts).result()
    assert message.stop_reason in ("aborted", "error")
    assert captured, message.error_message
    return captured


# -- event stream + SigV4 basics are covered in test_bedrock_aws.py; adapter tests follow ----------


async def test_sends_signed_converse_stream_request(mock_server, keys):
    mock_server.add(
        eventstream_response(
            [
                *simple_stop()[:1],
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
                simple_stop()[1],
            ]
        )
    )
    model = dataclasses.replace(_model(), id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    message = await stream(model, _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()

    assert message.stop_reason == "stop"
    assert message.content == [TextContent("hi")]
    request = mock_server.last
    assert request.url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream"
    )
    assert request.headers["content-type"] == "application/json"
    authorization = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
    assert "/us-east-1/bedrock/aws4_request" in authorization
    assert "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date," in authorization
    body = request.json()
    assert "modelId" not in body
    assert body["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert body["inferenceConfig"] == {"maxTokens": model.max_tokens}


async def test_bearer_token_auth(mock_server):
    mock_server.add(eventstream_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {"cache_retention": "none", "api_key": "bedrock-api-key", "http_client": mock_server.client()},
    ).result()
    headers = mock_server.last.headers
    assert headers["authorization"] == "Bearer bedrock-api-key"
    assert "x-amz-date" not in headers


async def test_skip_auth_signs_with_dummy_credentials(mock_server):
    mock_server.add(eventstream_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {
            "cache_retention": "none",
            "api_key": "ignored",
            "env": {"AWS_BEDROCK_SKIP_AUTH": "1"},
            "http_client": mock_server.client(),
        },
    ).result()
    assert mock_server.last.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=dummy-access-key/")


async def test_explicit_credentials_option(mock_server, keys):
    mock_server.add(eventstream_response(simple_stop()))

    async def provider() -> AwsCredentials:
        return AwsCredentials("AKIDEXPLICIT", "s", "token")

    await stream(
        _model(), _hello(), {"cache_retention": "none", "credentials": provider, "http_client": mock_server.client()}
    ).result()
    headers = mock_server.last.headers
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXPLICIT/")
    assert headers["x-amz-security-token"] == "token"


# -- custom headers ---------------------------------------------------------------------------------


async def test_custom_headers_are_signed_and_reserved_headers_skipped(mock_server, keys):
    mock_server.add(eventstream_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {
            "cache_retention": "none",
            "http_client": mock_server.client(),
            "headers": {
                "x-custom": "v",
                "authorization": "evil",
                "x-amz-date": "evil",
                "Authorization": "evil2",
                "X-Amz-Date": "evil2",
                "HOST": "evil3",
                "x-removed": None,
            },
        },
    ).result()
    headers = mock_server.last.headers
    assert headers["x-custom"] == "v"
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert headers["host"] == "bedrock-runtime.us-east-1.amazonaws.com"
    assert headers["x-amz-date"] != "evil2"
    assert "x-removed" not in headers
    signed = headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
    assert "x-custom" in signed


async def test_custom_header_signature_verifies(mock_server, keys):
    # Recompute the signature server-side from the received request.
    mock_server.add(eventstream_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {"cache_retention": "none", "http_client": mock_server.client(), "headers": {"x-custom": "v"}},
    ).result()
    request = mock_server.last
    signed_names = request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
    amz_date = request.headers["x-amz-date"]
    now = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    resigned = aws.sign_request(
        "POST",
        request.url,
        {name: request.headers[name] for name in signed_names if not name.startswith("x-amz-")},
        request.body,
        AwsCredentials("AKIDTEST", "secret"),
        region="us-east-1",
        service="bedrock",
        now=now,
    )
    assert resigned["authorization"] == request.headers["authorization"]


async def test_stream_simple_forwards_headers(mock_server, keys):
    mock_server.add(eventstream_response(simple_stop()))
    await stream_simple(
        _model(), _hello(), {"headers": {"x-custom": "v"}, "http_client": mock_server.client()}
    ).result()
    assert mock_server.last.headers["x-custom"] == "v"


# -- credentials priority + endpoint resolution -----------------------------------------------------


def test_prefers_explicit_and_scoped_profiles_over_ambient_keys(keys):
    model = _model()
    config = resolve_bedrock_client_config(model, {"profile": "explicit-profile"})
    assert config.profile == "explicit-profile"
    assert config.credentials is None
    config = resolve_bedrock_client_config(model, {"env": {"AWS_PROFILE": "scoped-profile"}})
    assert config.profile == "scoped-profile"
    assert config.credentials is None


def test_uses_ambient_keys_without_profile(keys):
    config = resolve_bedrock_client_config(_model(), {})
    assert config.profile is None
    assert config.credentials == AwsCredentials("AKIDTEST", "secret")


def test_uses_ambient_keys_when_only_ambient_profile_is_set(keys, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "ambient-profile")
    config = resolve_bedrock_client_config(_model(), {})
    assert config.profile == "ambient-profile"
    assert config.credentials == AwsCredentials("AKIDTEST", "secret")


def test_eu_inference_profiles_use_eu_central_endpoint():
    assert (
        _model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0").base_url
        == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    )


def test_does_not_pin_standard_endpoints_when_region_is_configured(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    config = resolve_bedrock_client_config(_model(), {})
    assert config.region == "us-east-2"
    assert config.endpoint is None


def test_derives_region_from_builtin_eu_endpoint():
    config = resolve_bedrock_client_config(_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0"), {})
    assert config.endpoint == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    assert config.region == "eu-central-1"


def test_missing_regions_for_explicit_scoped_and_ambient_profiles(monkeypatch):
    model = _model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
    for options, profile in (
        ({"profile": "bedrock-profile"}, "bedrock-profile"),
        ({"env": {"AWS_PROFILE": "scoped-bedrock-profile"}}, "scoped-bedrock-profile"),
    ):
        config = resolve_bedrock_client_config(model, options)
        assert config.profile == profile
        assert config.endpoint == "https://bedrock-runtime.eu-central-1.amazonaws.com"
        assert config.region == "eu-central-1"
    monkeypatch.setenv("AWS_PROFILE", "ambient-bedrock-profile")
    config = resolve_bedrock_client_config(model, {})
    assert config.profile == "ambient-bedrock-profile"
    assert config.endpoint is None
    assert config.region is None


def test_custom_endpoints_pass_through(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    model = dataclasses.replace(_model(), base_url="https://bedrock-vpc.example.com")
    config = resolve_bedrock_client_config(model, {})
    assert config.endpoint == "https://bedrock-vpc.example.com"
    assert config.region == "us-west-2"


@pytest.mark.parametrize(
    ("model_id", "region"),
    [
        ("arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123", "us-west-2"),
        ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123", "us-gov-west-1"),
    ],
)
def test_extracts_region_from_inference_profile_arn(monkeypatch, model_id, region):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    config = resolve_bedrock_client_config(dataclasses.replace(_model(), id=model_id), {})
    assert config.region == region


def test_preserves_ambient_profile_auth_for_custom_model_ids(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "bedrock-profile")
    model = dataclasses.replace(
        _model(), id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example"
    )
    config = resolve_bedrock_client_config(model, {})
    assert config.profile == "bedrock-profile"
    assert config.token is None


def test_generic_api_key_is_a_bearer_token():
    assert resolve_bedrock_client_config(_model(), {"api_key": "bedrock-api-key"}).token == "bedrock-api-key"
    assert resolve_bedrock_client_config(_model(), {"api_key": "<authenticated>"}).token is None


async def test_request_url_follows_resolved_region_and_endpoint(mock_server, keys, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    mock_server.add(eventstream_response(simple_stop()))
    await stream(_model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()
    assert mock_server.last.url.startswith("https://bedrock-runtime.us-east-2.amazonaws.com/model/")
    assert "/us-east-2/bedrock/aws4_request" in mock_server.last.headers["authorization"]

    model = dataclasses.replace(_model(), base_url="https://bedrock-vpc.example.com/prefix/")
    mock_server.add(eventstream_response(simple_stop()))
    await stream(model, _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()
    assert (
        mock_server.last.url
        == "https://bedrock-vpc.example.com/prefix/model/us.anthropic.claude-opus-4-8/converse-stream"
    )

    monkeypatch.setenv("AWS_REGION", "cn-north-1")
    mock_server.add(eventstream_response(simple_stop()))
    await stream(_model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()
    assert mock_server.last.url.startswith("https://bedrock-runtime.cn-north-1.amazonaws.com.cn/")


async def test_region_and_credentials_from_shared_profile(mock_server, monkeypatch, tmp_path):
    (tmp_path / "config").write_text("[profile work]\nregion = eu-west-3\n")
    (tmp_path / "credentials").write_text("[work]\naws_access_key_id = AKIDPROFILE\naws_secret_access_key = s\n")
    monkeypatch.setenv("AWS_PROFILE", "work")
    mock_server.add(eventstream_response(simple_stop()))
    message = await stream(
        _model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "stop", message.error_message
    assert mock_server.last.url.startswith("https://bedrock-runtime.eu-west-3.amazonaws.com/")
    assert mock_server.last.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDPROFILE/20")


async def test_region_missing_error(mock_server, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "noregion")
    message = await stream(
        _model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == "Region is missing"


async def test_missing_credentials_error(mock_server):
    message = await stream(
        _model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.error_message == "Could not load credentials from any providers"
    assert not mock_server.requests


# -- message conversion -----------------------------------------------------------------------------


async def test_gates_native_strict_tool_use_by_model_capability():
    tool = Tool(
        "lookup",
        "Look up a value",
        Type.Object({"value": Type.String()}),
        constrained_sampling={"type": "json_schema", "strict": "require"},
    )
    context = Context(messages=[UserMessage("Use the tool")], tools=[tool])
    payload = await capture_payload(context)
    spec = payload["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["strict"] is True
    assert spec["inputSchema"]["json"]["additionalProperties"] is False

    tool.constrained_sampling = {"type": "json_schema", "strict": "prefer"}
    payload = await capture_payload(context, NOVA_MODEL)
    assert "strict" not in payload["toolConfig"]["tools"][0]["toolSpec"]


async def test_tool_choice_mapping():
    tool = Tool("lookup", "d", Type.Object({}))
    context = Context(messages=[UserMessage("x")], tools=[tool])
    assert (await capture_payload(context, tool_choice="auto"))["toolConfig"]["toolChoice"] == {"auto": {}}
    assert (await capture_payload(context, tool_choice="any"))["toolConfig"]["toolChoice"] == {"any": {}}
    assert (await capture_payload(context, tool_choice={"type": "tool", "name": "lookup"}))["toolConfig"][
        "toolChoice"
    ] == {"tool": {"name": "lookup"}}
    assert "toolConfig" not in await capture_payload(context, tool_choice="none")


async def test_preserves_empty_property_names_in_streamed_tool_arguments(mock_server, keys):
    arguments = '{"path":"/workspace/foobar/file.js","edits":[{"oldText":"first","newText":"updated first"},{"oldText":"second","newText":"updated second","":""}]}'
    mock_server.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "tool-1", "name": "edit"}},
                    }
                },
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments[:20]}}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments[20:]}}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
            ],
            chunked=True,
        )
    )
    s = stream(BASE_MODEL, _hello(), {"cache_retention": "none", "http_client": mock_server.client()})
    types = [e.type async for e in s]
    message = await s.result()
    assert types == ["start", "toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end", "done"]
    assert message.stop_reason == "toolUse"
    assert message.content[0] == ToolCall("tool-1", "edit", json.loads(arguments))


async def test_skips_unknown_user_and_assistant_content_blocks():
    @dataclasses.dataclass
    class Unknown:
        type: str = "unknown"
        data: str = "foo"

    payload = await capture_payload([UserMessage([TextContent("hello"), Unknown()])])  # type: ignore[list-item]
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]

    payload = await capture_payload([_assistant([TextContent("hello"), Unknown()])])  # type: ignore[list-item]
    assert payload["messages"] == [{"role": "assistant", "content": [{"text": "hello"}]}]

    payload = await capture_payload([UserMessage([Unknown()])])  # type: ignore[list-item]
    assert payload["messages"] == [{"role": "user", "content": [{"text": "<empty>"}]}]

    payload = await capture_payload([_assistant([Unknown()])])  # type: ignore[list-item]
    assert payload["messages"] == []


async def test_blank_and_surrogate_only_text_handling():
    assert (await capture_payload([UserMessage("   ")]))["messages"][0]["content"] == [{"text": "<empty>"}]
    payload = await capture_payload([UserMessage([TextContent(""), TextContent("hello")])])
    assert payload["messages"][0]["content"] == [{"text": "hello"}]
    assert (await capture_payload([UserMessage("\ud83d")]))["messages"][0]["content"] == [{"text": "<empty>"}]
    assert (await capture_payload([_assistant([TextContent("\ud83d")])]))["messages"] == []
    payload = await capture_payload([ToolResultMessage("tool-1", "tool", [TextContent("")], False)])
    assert payload["messages"][0]["content"][0]["toolResult"]["content"] == [{"text": "<empty>"}]


async def test_moves_images_of_error_tool_results_after_the_tool_results():
    # Regression for #10794: Claude rejects error tool results that contain images.
    messages = [
        _assistant([ToolCall("call-1", "codemode", {}), ToolCall("call-2", "read", {})], stop_reason="toolUse"),
        ToolResultMessage(
            "call-1",
            "codemode",
            [TextContent("Script failed"), ImageContent("AAAA", "image/png"), TextContent("Script error")],
            True,
        ),
        ToolResultMessage("call-2", "read", [ImageContent("BBBB", "image/png")], False),
    ]
    payload = await capture_payload(messages)
    content = payload["messages"][1]["content"]
    assert len(content) == 4
    assert content[0]["toolResult"] == {
        "toolUseId": "call-1",
        "content": [
            {"text": "Script failed"},
            {"text": "(image attached after the tool results)"},
            {"text": "Script error"},
        ],
        "status": "error",
    }
    assert content[1]["toolResult"]["status"] == "success"
    assert content[1]["toolResult"]["content"] == [
        {"image": {"source": {"bytes": base64.b64decode("BBBB")}, "format": "png"}}
    ]
    assert content[2] == {"text": "Images from the error result of tool call call-1 (codemode):"}
    assert content[3]["image"]["format"] == "png"


async def test_keeps_images_in_error_tool_results_for_non_claude_models(mock_server, keys):
    messages = [
        ToolResultMessage("call-1", "codemode", [TextContent("Script failed"), ImageContent("AAAA", "image/png")], True)
    ]
    payload = await capture_payload(messages, NOVA_MODEL)
    assert payload["messages"][0]["content"] == [
        {
            "toolResult": {
                "toolUseId": "call-1",
                "content": [
                    {"text": "Script failed"},
                    {"image": {"source": {"bytes": b"\x00\x00\x00"}, "format": "png"}},
                ],
                "status": "error",
            }
        }
    ]
    # On the wire, image bytes are base64.
    mock_server.add(eventstream_response(simple_stop()))
    await stream(
        NOVA_MODEL,
        normalize_context(Context(messages=messages)),
        {"cache_retention": "none", "http_client": mock_server.client()},
    ).result()
    wire = mock_server.last.json()["messages"][0]["content"][0]["toolResult"]["content"][1]
    assert wire == {"image": {"source": {"bytes": "AAAA"}, "format": "png"}}


async def test_removes_empty_property_names_only_from_replayed_input():
    arguments = {
        "path": "/workspace/foobar/file.js",
        "edits": [
            {"oldText": "first", "newText": "updated first"},
            {"oldText": "second", "newText": "updated second", "": ""},
        ],
    }
    messages = [
        _assistant([ToolCall("tool-1", "edit", arguments)], stop_reason="toolUse"),
        ToolResultMessage("tool-1", "edit", [TextContent("done")], False),
        UserMessage("Continue"),
    ]
    payload = await capture_payload(messages)
    assert payload["messages"][0]["content"][0]["toolUse"]["input"] == {
        "path": "/workspace/foobar/file.js",
        "edits": [{"oldText": "first", "newText": "updated first"}, {"oldText": "second", "newText": "updated second"}],
    }
    assert arguments["edits"][1] == {"oldText": "second", "newText": "updated second", "": ""}


async def test_thinking_replay_signatures():
    messages = [
        UserMessage("hi"),
        _assistant(
            [ThinkingContent("signed", thinking_signature="sig"), ThinkingContent("unsigned"), TextContent("ok")]
        ),
        UserMessage("next"),
    ]
    payload = await capture_payload(messages)
    assert payload["messages"][1]["content"] == [
        {"reasoningContent": {"reasoningText": {"text": "signed", "signature": "sig"}}},
        {"text": "unsigned"},
        {"text": "ok"},
    ]
    gpt_messages = [
        UserMessage("hi"),
        _assistant([ThinkingContent("t", thinking_signature="sig")], GPT_MODEL),
        UserMessage("n"),
    ]
    payload = await capture_payload(gpt_messages, GPT_MODEL)
    assert payload["messages"][1]["content"] == [{"reasoningContent": {"reasoningText": {"text": "t"}}}]


async def test_foreign_tool_call_ids_are_normalized():
    messages = [
        AssistantMessage(
            [ToolCall("call|with spaces", "t", {})],
            api="openai-responses",
            provider="openai",
            model="gpt",
            stop_reason="toolUse",
        ),
        ToolResultMessage("call|with spaces", "t", [TextContent("r")], False),
    ]
    payload = await capture_payload(messages)
    assert payload["messages"][0]["content"][0]["toolUse"]["toolUseId"] == "call_with_spaces"
    assert payload["messages"][1]["content"][0]["toolResult"]["toolUseId"] == "call_with_spaces"


async def test_system_prompt_and_cache_points():
    context = Context(system_prompt="You are helpful.", messages=[UserMessage("Hello")])
    payload = await capture_payload(context, cache_retention="short")
    assert payload["system"] == [{"text": "You are helpful."}, {"cachePoint": {"type": "default"}}]
    assert payload["messages"][-1]["content"][-1] == {"cachePoint": {"type": "default"}}
    payload = await capture_payload(context, cache_retention="long")
    assert payload["system"][1] == {"cachePoint": {"type": "default", "ttl": "1h"}}
    payload = await capture_payload(context, cache_retention="none")
    assert payload["system"] == [{"text": "You are helpful."}]
    payload = await capture_payload(context, NOVA_MODEL, cache_retention="short")
    assert payload["system"] == [{"text": "You are helpful."}]


# -- redacted reasoning -----------------------------------------------------------------------------

REDACTED_B64 = "cnNuXzVaVnJpZjRKMGJYSXFtV2RsZWRqN1FJRmVOaWtSUWJF"
REDACTED_BYTES = base64.b64decode(REDACTED_B64)


def _redacted_delta(index: int, data: bytes) -> dict[str, Any]:
    return {
        "contentBlockDelta": {
            "contentBlockIndex": index,
            "delta": {"reasoningContent": {"redactedContent": base64.b64encode(data).decode()}},
        }
    }


async def _run_gpt(mock_server, items: list[dict[str, Any]]) -> AssistantMessage:
    mock_server.add(eventstream_response(items))
    return await stream(GPT_MODEL, _hello(), {"http_client": mock_server.client(), "api_key": "k"}).result()


async def test_redacted_reasoning_stream(mock_server):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "done"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(mock_server, items)
    assert response.stop_reason == "stop", response.error_message
    assert [c.type for c in response.content] == ["thinking", "text"]
    assert response.content[1] == TextContent("done")
    thinking = response.content[0]
    assert thinking.redacted is True
    assert thinking.thinking_signature == REDACTED_B64


async def test_redacted_reasoning_without_content_block_stop(mock_server):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES),
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(mock_server, items)
    assert response.content[0].thinking_signature == REDACTED_B64


async def test_redacted_reasoning_split_across_deltas(mock_server):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES[:7]),
        _redacted_delta(0, REDACTED_BYTES[7:]),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(mock_server, items)
    assert response.content[0].thinking_signature == REDACTED_B64
    assert response.content[0].thinking == "[Reasoning redacted]"


async def test_replays_redacted_reasoning():
    messages = [
        UserMessage("hello"),
        _assistant(
            [ThinkingContent("", thinking_signature=REDACTED_B64, redacted=True), TextContent("done")], GPT_MODEL
        ),
        UserMessage("continue"),
    ]
    payload = await capture_payload(messages, GPT_MODEL)
    assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant["content"] == [{"reasoningContent": {"redactedContent": REDACTED_BYTES}}, {"text": "done"}]

    messages = [
        UserMessage("read the file"),
        _assistant(
            [
                ThinkingContent("", thinking_signature=REDACTED_B64, redacted=True),
                ToolCall("tool-1", "read", {"path": "/tmp/a.txt"}),
            ],
            GPT_MODEL,
            "toolUse",
        ),
        ToolResultMessage("tool-1", "read", [TextContent("file body")], False),
    ]
    payload = await capture_payload(messages, GPT_MODEL)
    assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant["content"] == [
        {"reasoningContent": {"redactedContent": REDACTED_BYTES}},
        {"toolUse": {"toolUseId": "tool-1", "name": "read", "input": {"path": "/tmp/a.txt"}}},
    ]


async def test_signed_thinking_stream(mock_server, keys):
    mock_server.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think"}}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    message = await stream(
        BASE_MODEL, _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.content == [ThinkingContent("think", thinking_signature="sig")]


# -- thinking payload -------------------------------------------------------------------------------

THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
ADAPTIVE_WITH_BINDING = {
    "type": "adaptive",
    "display": "summarized",
    "block_binding": {"prefix_mismatch_behavior": "drop_block"},
}


async def thinking_payload(model: Model, **options: Any) -> dict[str, Any]:
    options.setdefault("reasoning", "high")
    return await capture_payload([UserMessage("Hello")], model, raise_after=True, **options)


def _opus_4_8() -> Model:
    return dataclasses.replace(
        _model("global.anthropic.claude-opus-4-6-v1"),
        id="global.anthropic.claude-opus-4-8-v1",
        name="Claude Opus 4.8 (Global)",
    )


@pytest.mark.parametrize(
    ("model_id", "reasoning", "effort"),
    [
        (None, "high", "high"),
        (None, "xhigh", "xhigh"),
        ("global.anthropic.claude-fable-5", "high", "high"),
        ("global.anthropic.claude-sonnet-5", "high", "high"),
        ("global.anthropic.claude-opus-5", "high", "high"),
        ("global.anthropic.claude-opus-5", "xhigh", "xhigh"),
        ("global.anthropic.claude-fable-5", "xhigh", "xhigh"),
        ("global.anthropic.claude-opus-5-5", "high", "high"),
    ],
)
async def test_adaptive_thinking_with_block_binding(model_id, reasoning, effort):
    model = _opus_4_8() if model_id is None else _model(model_id)
    payload = await thinking_payload(model, reasoning=reasoning)
    fields = payload["additionalModelRequestFields"]
    assert fields["thinking"] == ADAPTIVE_WITH_BINDING
    assert fields["output_config"] == {"effort": effort}
    assert fields["anthropic_beta"] == [THINKING_BINDING_CONTROLS_BETA]


@pytest.mark.parametrize("model_id", ["global.anthropic.claude-opus-4-6-v1", "global.anthropic.claude-sonnet-4-6"])
async def test_omits_block_binding_for_4_6_models(model_id):
    # Bedrock rejects block_binding on 4.6 models: "Extra inputs are not permitted" (#10324)
    fields = (await thinking_payload(_model(model_id)))["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "anthropic_beta" not in fields


async def test_omits_display_for_govcloud_model_ids():
    model = dataclasses.replace(
        _model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        id="us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        name="Claude Sonnet 4.5 (GovCloud)",
    )
    fields = (await thinking_payload(model))["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert fields["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]


async def test_omits_display_for_govcloud_regions():
    fields = (await thinking_payload(_opus_4_8(), region="us-gov-west-1"))["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive"}
    assert fields["output_config"] == {"effort": "high"}
    assert "anthropic_beta" not in fields


async def test_budget_thinking_options():
    model = _model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    fields = (
        await thinking_payload(
            model,
            reasoning="low",
            thinking_budgets={"low": 3000},
            thinking_display="omitted",
            interleaved_thinking=False,
        )
    )["additionalModelRequestFields"]
    assert fields == {"thinking": {"type": "enabled", "budget_tokens": 3000, "display": "omitted"}}
    fields = (await thinking_payload(model, reasoning="max"))["additionalModelRequestFields"]
    assert fields["thinking"]["budget_tokens"] == 16384


@pytest.mark.parametrize(
    ("reasoning", "effort"),
    [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "max")],
)
async def test_openai_reasoning_effort(reasoning, effort):
    # Regression for #9331.
    for model_id in ("global.openai.gpt-6-sol", "us.openai.gpt-6-luna", "global.openai.gpt-5.6-sol"):
        payload = await thinking_payload(_model(model_id), reasoning=reasoning)
        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": effort}}, model_id


async def test_openai_effort_when_only_name_identifies_gpt():
    model = dataclasses.replace(
        _model("global.openai.gpt-6-sol"),
        id="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
        name="GPT-6 Sol",
    )
    assert (await thinking_payload(model, reasoning="medium"))["additionalModelRequestFields"] == {
        "reasoning": {"effort": "medium"}
    }


async def test_gpt_oss_flat_reasoning_effort():
    model = _model("openai.gpt-oss-120b-1:0")
    for reasoning, effort in (("minimal", "low"), ("medium", "medium"), ("xhigh", "high")):
        assert (await thinking_payload(model, reasoning=reasoning))["additionalModelRequestFields"] == {
            "reasoning_effort": effort
        }


async def test_no_reasoning_fields_when_off():
    payload = await capture_payload([UserMessage("Hello")], _model("global.openai.gpt-6-sol"), raise_after=True)
    assert "additionalModelRequestFields" not in payload


async def test_application_inference_profile_support():
    profile_id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile"
    model = dataclasses.replace(_model("global.anthropic.claude-opus-4-6-v1"), id=profile_id, name="Claude Opus 4.6")
    fields = (await thinking_payload(model))["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"] == {"effort": "high"}

    model = dataclasses.replace(_model("global.anthropic.claude-opus-4-6-v1"), id=profile_id, name="Claude Sonnet 4.6")
    payload = await capture_payload(
        Context(system_prompt="You are helpful.", messages=[UserMessage("Hello")]),
        model,
        raise_after=True,
        cache_retention="short",
    )
    assert len(payload["system"]) == 2
    assert "cachePoint" in payload["system"][1]
    assert "cachePoint" in payload["messages"][-1]["content"][-1]

    model = dataclasses.replace(
        _model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), id=profile_id, name="Claude Sonnet 4.5"
    )
    fields = (await thinking_payload(model))["additionalModelRequestFields"]
    assert fields["thinking"]["type"] == "enabled"
    assert isinstance(fields["thinking"]["budget_tokens"], int)
    assert fields["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]


async def test_stream_simple_budget_thinking_adjusts_max_tokens():
    model = _model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _m: Any) -> Any:
        captured.update(payload)
        raise PayloadCaptured()

    await stream_simple(model, _hello(), {"reasoning": "medium", "max_tokens": 1000, "on_payload": on_payload}).result()
    assert captured["inferenceConfig"]["maxTokens"] == 9192
    assert (
        captured["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 8192 - 24
    )  # leaves 1024 answer tokens

    captured.clear()
    await stream_simple(_opus_4_8(), _hello(), {"reasoning": "low", "on_payload": on_payload}).result()
    assert captured["additionalModelRequestFields"]["output_config"] == {"effort": "low"}


# -- stream events, stop reasons, usage, headers ----------------------------------------------------


async def test_forwards_stream_items_in_order(mock_server, keys):
    items = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}},
    ]
    mock_server.add(eventstream_response([{**item} for item in items]))
    received: list[Any] = []
    models: list[Any] = []
    model = _model()

    async def on_event(item: Any, event_model: Any) -> None:
        received.append(item)
        models.append(event_model)

    result = await stream(
        model,
        _hello(),
        {"cache_retention": "none", "http_client": mock_server.client(), "on_provider_stream_event": on_event},
    ).result()
    assert received == items
    assert models == [model] * 4
    assert result.stop_reason == "stop"
    assert result.content == [TextContent("hello")]
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (1, 1, 2)


async def test_strips_event_padding_and_forwards_exception_items(mock_server, keys):
    mock_server.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant", "p": "abcdefgh"}},
                {"internalServerException": {"message": "bedrock stream failed"}},
            ]
        )
    )
    received: list[Any] = []
    result = await stream(
        _model(),
        _hello(),
        {
            "cache_retention": "none",
            "http_client": mock_server.client(),
            "on_provider_stream_event": lambda i, _m: received.append(i),
        },
    ).result()
    assert received[0] == {"messageStart": {"role": "assistant"}}
    assert isinstance(received[1]["internalServerException"], BedrockServiceError)
    assert result.stop_reason == "error"
    assert result.error_message == "Internal server error: bedrock stream failed"
    assert is_retryable_assistant_error(result)


@pytest.mark.parametrize(
    ("raw", "stop_reason", "error_message"),
    [
        ("end_turn", "stop", None),
        ("stop_sequence", "stop", None),
        ("max_tokens", "length", None),
        ("model_context_window_exceeded", "length", None),
        ("guardrail_intervened", "error", "Provider stopped with: guardrail_intervened"),
    ],
)
async def test_raw_stop_reasons(mock_server, keys, raw, stop_reason, error_message):
    mock_server.add(eventstream_response(simple_stop(raw)))
    message = await stream(
        _model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == stop_reason
    assert message.raw_stop_reason == raw
    assert message.error_message == error_message


async def test_prices_1h_cache_writes_at_twice_input(mock_server, keys):
    # Regression test for https://github.com/earendil-works/pi/issues/9457
    model = _model()
    mock_server.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 100,
                            "outputTokens": 5,
                            "totalTokens": 1_000_105,
                            "cacheWriteInputTokens": 1_000_000,
                            "cacheDetails": [
                                {"ttl": "1h", "inputTokens": 150_000},
                                {"ttl": "5m", "inputTokens": 600_000},
                                {"ttl": "1h", "inputTokens": 250_000},
                            ],
                        }
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    result = await stream(model, _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()
    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    expected = (600_000 * model.cost.cache_write + 400_000 * model.cost.input * 2) / 1_000_000
    assert result.usage.cost.cache_write == pytest.approx(expected, abs=1e-10)


async def test_forwards_raw_response_headers_to_on_response(mock_server, keys):
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    mock_server.add(
        httpx.Response(
            200,
            content=b"",
            headers={
                "content-type": "application/vnd.amazon.eventstream",
                "x-bifrost-provider": "bedrock",
                "x-bifrost-resolved-model": model_id,
                "x-amzn-requestid": "req-123",
            },
        )
    )
    responses: list[Any] = []
    result = await stream(
        _model(model_id),
        _hello(),
        {
            "cache_retention": "none",
            "http_client": mock_server.client(),
            "on_response": lambda r, _m: responses.append(r),
        },
    ).result()
    # The empty event stream fails, but the header callback fired before stream consumption.
    assert result.stop_reason == "error"
    assert result.error_message == "Bedrock stream ended without a stop reason"
    assert len(responses) == 1
    assert responses[0].status == 200
    assert responses[0].headers["x-amzn-requestid"] == "req-123"
    assert responses[0].headers["x-bifrost-provider"] == "bedrock"
    assert responses[0].headers["x-bifrost-resolved-model"] == model_id


async def test_corrupted_event_stream_is_an_error(mock_server, keys):
    frame = bytearray(event("messageStart", {"role": "assistant"}))
    frame[-1] ^= 0xFF
    mock_server.add(
        httpx.Response(200, content=bytes(frame), headers={"content-type": "application/vnd.amazon.eventstream"})
    )
    result = await stream(_model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}).result()
    assert result.stop_reason == "error"
    assert result.error_message == "Event stream message checksum mismatch"


# -- error metadata ---------------------------------------------------------------------------------

DIAGNOSTIC_TYPE = "bedrock_response_failure"
VALIDATION_MESSAGE = "The provided model identifier is invalid."
REQUEST_ID = "11111111-2222-3333-4444-555555555555"


def _diagnostic(message: AssistantMessage) -> Any:
    return next((d for d in message.diagnostics or [] if d.type == DIAGNOSTIC_TYPE), None)


def _error_response(status: int, error_type: str | None, body: Any, request_id: str = REQUEST_ID) -> httpx.Response:
    headers = {"x-amzn-requestid": request_id}
    if error_type is not None:
        headers["x-amzn-errortype"] = error_type
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Response(status, content=content, headers=headers)


async def _run_bedrock(mock_server, **options: Any) -> AssistantMessage:
    return await stream(
        _model(),
        _hello(),
        {"cache_retention": "none", "http_client": mock_server.client(), "max_retries": 0, **options},
    ).result()


async def test_records_status_code_and_request_id_for_http_errors(mock_server, keys):
    mock_server.add(
        _error_response(
            400,
            "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/",
            {"message": VALIDATION_MESSAGE},
        )
    )
    message = await _run_bedrock(mock_server)
    diagnostic = _diagnostic(message)
    assert message.stop_reason == "error"
    assert message.error_message == f"Validation error: {VALIDATION_MESSAGE}"
    assert diagnostic.details == {"status": 400, "errorCode": "ValidationException", "requestId": REQUEST_ID}
    assert diagnostic.error is None


async def test_mid_stream_modeled_exception(mock_server, keys):
    mock_server.add(
        eventstream_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"throttlingException": {"message": "Too many tokens, please wait before trying again."}},
            ],
            headers={"x-amzn-requestid": REQUEST_ID},
        )
    )
    message = await _run_bedrock(mock_server)
    assert message.stop_reason == "error"
    assert message.error_message == "Throttling error: Too many tokens, please wait before trying again."
    # Throttling must not be mistaken for a context overflow ("too many tokens").
    assert not is_context_overflow(message, 200_000)
    assert _diagnostic(message).details == {"errorCode": "ThrottlingException", "requestId": REQUEST_ID}


async def test_unmodeled_mid_stream_error(mock_server, keys):
    frames = event("messageStart", {"role": "assistant"}) + encode_event_stream_message(
        {
            ":message-type": "error",
            ":error-code": "ModelStreamErrorException",
            ":error-message": "Model stream terminated unexpectedly.",
        }
    )
    mock_server.add(httpx.Response(200, content=frames, headers={"x-amzn-requestid": REQUEST_ID}))
    message = await _run_bedrock(mock_server)
    assert message.error_message == "Model stream terminated unexpectedly."
    assert _diagnostic(message).details == {"errorCode": "ModelStreamErrorException", "requestId": REQUEST_ID}


async def test_unknown_exception_frame_uses_body(mock_server, keys):
    frames = event("messageStart", {"role": "assistant"}) + exception("fooException", {"message": "foo"})
    mock_server.add(httpx.Response(200, content=frames, headers={"x-amzn-requestid": REQUEST_ID}))
    message = await _run_bedrock(mock_server)
    assert message.error_message == '{"message": "foo"}'
    assert _diagnostic(message).details == {"errorCode": "fooException", "requestId": REQUEST_ID}


async def test_transport_failure_has_no_provider_code(mock_server, keys):
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("socket hang up", request=request)

    mock_server.add(fail)
    message = await _run_bedrock(mock_server)
    assert message.stop_reason == "error"
    assert message.error_message == "Connection error: socket hang up"
    assert _diagnostic(message) is None


async def test_no_diagnostic_for_aborted_turn(mock_server, keys):
    mock_server.add(_error_response(400, "ValidationException", {"message": VALIDATION_MESSAGE}))
    message = await _run_bedrock(mock_server, signal=AbortSignal.abort())
    assert message.stop_reason == "aborted"
    assert _diagnostic(message) is None


async def test_drops_over_long_header_values(mock_server, keys):
    mock_server.add(_error_response(400, f"{'E' * 5000}Exception", {"message": VALIDATION_MESSAGE}, "R" * 5000))
    message = await _run_bedrock(mock_server)
    assert _diagnostic(message).details == {"status": 400}


async def test_unknown_error_code_placeholder_and_raw_bodies(mock_server, keys):
    mock_server.add(_error_response(403, None, {"message": "blocked by gw"}))
    message = await _run_bedrock(mock_server)
    assert message.error_message == "Unknown: blocked by gw"
    assert _diagnostic(message).details == {"status": 403, "requestId": REQUEST_ID}

    mock_server.add(_error_response(403, None, b"<html>blocked</html>"))
    message = await _run_bedrock(mock_server)
    assert message.error_message == "Unknown: 403: <html>blocked</html>"


async def test_data_retention_hint(mock_server, keys):
    mock_server.add(
        _error_response(
            400, "ValidationException", {"message": "data retention mode 'default' is not available for this model"}
        )
    )
    message = await _run_bedrock(mock_server)
    assert message.error_message.endswith(
        "See https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html for supported data retention modes."
    )


async def test_retries_throttling_by_default(mock_server, keys, monkeypatch):
    async def no_sleep(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("pi_ai.utils.http.sleep", no_sleep)
    mock_server.add(_error_response(429, "ThrottlingException", {"message": "slow down"}))
    mock_server.add(eventstream_response(simple_stop()))
    message = await stream(
        _model(), _hello(), {"cache_retention": "none", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "stop"
    assert len(mock_server.requests) == 2
