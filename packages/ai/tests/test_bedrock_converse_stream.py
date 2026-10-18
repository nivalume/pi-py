"""Offline tests for the Bedrock adapter (ports of upstream ``test/bedrock-*.test.ts``).

Upstream mocks the AWS SDK client's ``send``; here a fake boto3 client replaces the cached
``bedrock-runtime`` client: ``converse_stream(**params)`` records the command input and
returns an SDK-shaped response whose ``stream`` yields the given items. Wire-level behavior
(signing, endpoints, error bodies, retries, event-stream decoding) is covered with a real
boto3 client in ``test_bedrock_aws.py``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError, EndpointConnectionError, EventStreamError

from pi_ai import AbortController, AbortSignal, Context, Model, Tool, Type, normalize_context
from pi_ai.api import bedrock_converse_stream as bcs
from pi_ai.api.bedrock_converse_stream import (
    BedrockCredentials,
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


# -- fake boto3 client ----------------------------------------------------------------------------------


class FakeEventStream:
    """Iterates SDK stream items; exception items are raised the way botocore raises them."""

    def __init__(self, items: list[Any], gate: threading.Event | None = None) -> None:
        self._items = items
        self._gate = gate
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        for item in self._items:
            if isinstance(item, BaseException):
                raise item
            yield item
        if self._gate is not None:
            self._gate.wait(10)
            if self.closed:
                raise ConnectionResetError("stream closed")

    def close(self) -> None:
        self.closed = True
        if self._gate is not None:
            self._gate.set()


def stream_exception(name: str, message: str) -> EventStreamError:
    return EventStreamError({"Error": {"Code": name, "Message": message}}, "ConverseStream")


def to_sdk_item(item: Any) -> Any:
    if isinstance(item, BaseException):
        return item
    ((name, payload),) = item.items()
    if name.endswith("Exception"):
        return stream_exception(name, payload.get("message", ""))
    return item


def sdk_response(
    items: list[dict[str, Any]], headers: dict[str, str] | None = None, gate: threading.Event | None = None
) -> dict[str, Any]:
    all_headers = {"content-type": "application/vnd.amazon.eventstream", "x-amzn-requestid": "request-id"}
    all_headers.update(headers or {})
    return {
        "ResponseMetadata": {
            "RequestId": all_headers["x-amzn-requestid"],
            "HTTPStatusCode": 200,
            "HTTPHeaders": all_headers,
            "RetryAttempts": 0,
        },
        "stream": FakeEventStream([to_sdk_item(item) for item in items], gate),
    }


class FakeBedrockClient:
    def __init__(self) -> None:
        self.responses: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.keys: list[Any] = []
        #: Custom headers visible to botocore's request hooks during each call.
        self.headers: list[Any] = []

    def add(self, response: Any) -> None:
        self.responses.append(response)

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]

    def converse_stream(self, **params: Any) -> Any:
        self.calls.append(params)
        self.headers.append(getattr(bcs._call_state, "headers", None))
        response = self.responses.pop(0) if self.responses else sdk_response([])
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def fake_client(monkeypatch):
    fake = FakeBedrockClient()

    def get_client(key: Any) -> FakeBedrockClient:
        fake.keys.append(key)
        return fake

    monkeypatch.setattr(bcs, "_get_client", get_client)
    return fake


def simple_stop(stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    return [{"messageStart": {"role": "assistant"}}, {"messageStop": {"stopReason": stop_reason}}]


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


# -- the SDK call --------------------------------------------------------------------------------------


async def test_sdk_receives_the_command_input(fake_client, keys):
    fake_client.add(
        sdk_response(
            [
                *simple_stop()[:1],
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
                simple_stop()[1],
            ]
        )
    )
    model = dataclasses.replace(_model(), id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    captured: list[Any] = []
    message = await stream(
        model, _hello(), {"cache_retention": "none", "on_payload": lambda p, _m: captured.append(p)}
    ).result()
    assert message.stop_reason == "stop", message.error_message
    assert message.content == [TextContent("hi")]
    assert fake_client.last == {
        "modelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "inferenceConfig": {"maxTokens": model.max_tokens},
    }
    assert fake_client.last == captured[0]
    key = fake_client.keys[-1]
    assert (key.region, key.endpoint, key.profile) == ("us-east-1", model.base_url, None)
    assert key.credentials == ("AKIDTEST", "secret", None)
    assert key.token is None


async def test_on_payload_replacement_is_sent_and_binary_members_stay_bytes(fake_client, keys):
    messages = [
        ToolResultMessage("call-1", "codemode", [TextContent("Script failed"), ImageContent("AAAA", "image/png")], True)
    ]

    def on_payload(payload: Any, _model: Any) -> Any:
        return {**payload, "requestMetadata": {"team": "a"}, "toolConfig": None}

    fake_client.add(sdk_response(simple_stop()))
    await stream(
        NOVA_MODEL, normalize_context(Context(messages=messages)), {"cache_retention": "none", "on_payload": on_payload}
    ).result()
    params = fake_client.last
    assert params["requestMetadata"] == {"team": "a"}
    assert "toolConfig" not in params  # None members are dropped
    image = params["messages"][0]["content"][0]["toolResult"]["content"][1]
    assert image == {"image": {"source": {"bytes": b"\x00\x00\x00"}, "format": "png"}}


async def test_bearer_token_config(fake_client):
    fake_client.add(sdk_response(simple_stop()))
    await stream(_model(), _hello(), {"cache_retention": "none", "api_key": "bedrock-api-key"}).result()
    key = fake_client.keys[-1]
    assert key.token == "bedrock-api-key"
    assert not key.force_sigv4


async def test_skip_auth_uses_dummy_credentials(fake_client):
    fake_client.add(sdk_response(simple_stop()))
    await stream(
        _model(), _hello(), {"cache_retention": "none", "api_key": "ignored", "env": {"AWS_BEDROCK_SKIP_AUTH": "1"}}
    ).result()
    key = fake_client.keys[-1]
    assert key.credentials == ("dummy-access-key", "dummy-secret-key", None)
    assert key.token is None
    assert key.force_sigv4


@pytest.mark.parametrize(
    "credentials",
    [
        BedrockCredentials("AKIDEXPLICIT", "s", "token"),
        {"access_key_id": "AKIDEXPLICIT", "secret_access_key": "s", "session_token": "token"},
        {"accessKeyId": "AKIDEXPLICIT", "secretAccessKey": "s", "sessionToken": "token"},
        Credentials("AKIDEXPLICIT", "s", "token"),
        lambda: BedrockCredentials("AKIDEXPLICIT", "s", "token"),
    ],
)
async def test_explicit_credentials_option(fake_client, keys, credentials):
    async def provider() -> Any:
        return credentials() if callable(credentials) else credentials

    for value in (credentials, provider):
        fake_client.add(sdk_response(simple_stop()))
        message = await stream(_model(), _hello(), {"cache_retention": "none", "credentials": value}).result()
        assert message.stop_reason == "stop", message.error_message
        assert fake_client.keys[-1].credentials == ("AKIDEXPLICIT", "s", "token")


async def test_invalid_credentials_option(fake_client):
    message = await stream(_model(), _hello(), {"cache_retention": "none", "credentials": {"x": 1}}).result()
    assert message.error_message == "Bedrock credentials must provide access_key_id and secret_access_key"
    assert not fake_client.calls


async def test_custom_headers_reach_the_request_hooks_without_reserved_ones(fake_client, keys):
    fake_client.add(sdk_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {
            "cache_retention": "none",
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
    assert fake_client.headers == [{"x-custom": "v"}]
    assert getattr(bcs._call_state, "headers", None) is None


async def test_stream_simple_forwards_headers(fake_client, keys):
    fake_client.add(sdk_response(simple_stop()))
    await stream_simple(_model(), _hello(), {"headers": {"x-custom": "v"}}).result()
    assert fake_client.headers == [{"x-custom": "v"}]


async def test_client_options_from_stream_options(fake_client, keys):
    fake_client.add(sdk_response(simple_stop()))
    await stream(
        _model(),
        _hello(),
        {
            "cache_retention": "none",
            "max_retries": 0,
            "timeout_ms": 2500,
            "env": {"HTTPS_PROXY": "http://proxy.local:3128"},
        },
    ).result()
    key = fake_client.keys[-1]
    assert key.max_attempts == 1
    assert key.timeout == 2.5
    assert key.proxy == "http://proxy.local:3128"
    fake_client.add(sdk_response(simple_stop()))
    await stream(_model(), _hello(), {"cache_retention": "none"}).result()
    key = fake_client.keys[-1]
    assert (key.max_attempts, key.timeout, key.proxy) == (None, 600.0, None)


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
    assert config.credentials == BedrockCredentials("AKIDTEST", "secret")


def test_uses_ambient_keys_when_only_ambient_profile_is_set(keys, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "ambient-profile")
    config = resolve_bedrock_client_config(_model(), {})
    assert config.profile == "ambient-profile"
    assert config.credentials == BedrockCredentials("AKIDTEST", "secret")


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


async def test_preserves_empty_property_names_in_streamed_tool_arguments(fake_client, keys):
    arguments = '{"path":"/workspace/foobar/file.js","edits":[{"oldText":"first","newText":"updated first"},{"oldText":"second","newText":"updated second","":""}]}'
    fake_client.add(
        sdk_response(
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
        )
    )
    s = stream(BASE_MODEL, _hello(), {"cache_retention": "none"})
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


async def test_keeps_images_in_error_tool_results_for_non_claude_models():
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
            "delta": {"reasoningContent": {"redactedContent": data}},
        }
    }


async def _run_gpt(fake_client, items: list[dict[str, Any]]) -> AssistantMessage:
    fake_client.add(sdk_response(items))
    return await stream(GPT_MODEL, _hello(), {"api_key": "k"}).result()


async def test_redacted_reasoning_stream(fake_client):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "done"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(fake_client, items)
    assert response.stop_reason == "stop", response.error_message
    assert [c.type for c in response.content] == ["thinking", "text"]
    assert response.content[1] == TextContent("done")
    thinking = response.content[0]
    assert thinking.redacted is True
    assert thinking.thinking_signature == REDACTED_B64


async def test_redacted_reasoning_without_content_block_stop(fake_client):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES),
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(fake_client, items)
    assert response.content[0].thinking_signature == REDACTED_B64


async def test_redacted_reasoning_split_across_deltas(fake_client):
    items = [
        {"messageStart": {"role": "assistant"}},
        _redacted_delta(0, REDACTED_BYTES[:7]),
        _redacted_delta(0, REDACTED_BYTES[7:]),
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response = await _run_gpt(fake_client, items)
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


async def test_signed_thinking_stream(fake_client, keys):
    fake_client.add(
        sdk_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think"}}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    message = await stream(BASE_MODEL, _hello(), {"cache_retention": "none"}).result()
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


async def test_forwards_stream_items_in_order(fake_client, keys):
    items = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}},
    ]
    fake_client.add(sdk_response([{**item} for item in items]))
    received: list[Any] = []
    models: list[Any] = []
    model = _model()

    async def on_event(item: Any, event_model: Any) -> None:
        received.append(item)
        models.append(event_model)

    result = await stream(
        model,
        _hello(),
        {"cache_retention": "none", "on_provider_stream_event": on_event},
    ).result()
    assert received == items
    assert models == [model] * 4
    assert result.stop_reason == "stop"
    assert result.content == [TextContent("hello")]
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (1, 1, 2)


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
async def test_raw_stop_reasons(fake_client, keys, raw, stop_reason, error_message):
    fake_client.add(sdk_response(simple_stop(raw)))
    message = await stream(_model(), _hello(), {"cache_retention": "none"}).result()
    assert message.stop_reason == stop_reason
    assert message.raw_stop_reason == raw
    assert message.error_message == error_message


async def test_prices_1h_cache_writes_at_twice_input(fake_client, keys):
    # Regression test for https://github.com/earendil-works/pi/issues/9457
    model = _model()
    fake_client.add(
        sdk_response(
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
    result = await stream(model, _hello(), {"cache_retention": "none"}).result()
    assert result.usage.cache_write == 1_000_000
    assert result.usage.cache_write_1h == 400_000
    expected = (600_000 * model.cost.cache_write + 400_000 * model.cost.input * 2) / 1_000_000
    assert result.usage.cost.cache_write == pytest.approx(expected, abs=1e-10)


async def test_forwards_exception_items_before_raising(fake_client, keys):
    fake_client.add(
        sdk_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"internalServerException": {"message": "bedrock stream failed"}},
            ]
        )
    )
    received: list[Any] = []
    result = await stream(
        _model(),
        _hello(),
        {"cache_retention": "none", "on_provider_stream_event": lambda i, _m: received.append(i)},
    ).result()
    assert received[0] == {"messageStart": {"role": "assistant"}}
    error = received[1]["internalServerException"]
    assert isinstance(error, BedrockServiceError)
    assert error.name == "InternalServerException"
    assert result.stop_reason == "error"
    assert result.error_message == "Internal server error: bedrock stream failed"
    assert is_retryable_assistant_error(result)


async def test_forwards_response_metadata_headers_to_on_response(fake_client, keys):
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    fake_client.add(sdk_response([], headers={"x-bifrost-provider": "bedrock", "x-amzn-requestid": "req-123"}))
    responses: list[Any] = []
    result = await stream(
        _model(model_id), _hello(), {"cache_retention": "none", "on_response": lambda r, _m: responses.append(r)}
    ).result()
    # The empty event stream fails, but the header callback fired before stream consumption.
    assert result.stop_reason == "error"
    assert result.error_message == "Bedrock stream ended without a stop reason"
    assert [(r.status, r.headers["x-amzn-requestid"], r.headers["x-bifrost-provider"]) for r in responses] == [
        (200, "req-123", "bedrock")
    ]


async def test_rejects_user_message_start(fake_client, keys):
    fake_client.add(sdk_response([{"messageStart": {"role": "user"}}]))
    result = await stream(_model(), _hello(), {"cache_retention": "none"}).result()
    assert result.error_message == "Unexpected assistant message start but got user message start instead"


# -- error metadata ---------------------------------------------------------------------------------

DIAGNOSTIC_TYPE = "bedrock_response_failure"
VALIDATION_MESSAGE = "The provided model identifier is invalid."
REQUEST_ID = "11111111-2222-3333-4444-555555555555"


def _diagnostic(message: AssistantMessage) -> Any:
    return next((d for d in message.diagnostics or [] if d.type == DIAGNOSTIC_TYPE), None)


async def _run_bedrock(**options: Any) -> AssistantMessage:
    return await stream(_model(), _hello(), {"cache_retention": "none", **options}).result()


async def test_client_error_without_raw_response_uses_the_parsed_error(fake_client, keys):
    fake_client.add(
        ClientError(
            {
                "Error": {"Code": "ValidationException", "Message": VALIDATION_MESSAGE},
                "ResponseMetadata": {"HTTPStatusCode": 400, "RequestId": REQUEST_ID},
            },
            "ConverseStream",
        )
    )
    message = await _run_bedrock()
    assert message.error_message == f"Validation error: {VALIDATION_MESSAGE}"
    assert _diagnostic(message).details == {"status": 400, "errorCode": "ValidationException", "requestId": REQUEST_ID}


async def test_mid_stream_modeled_exception(fake_client, keys):
    fake_client.add(
        sdk_response(
            [
                {"messageStart": {"role": "assistant"}},
                {"throttlingException": {"message": "Too many tokens, please wait before trying again."}},
            ],
            headers={"x-amzn-requestid": REQUEST_ID},
        )
    )
    message = await _run_bedrock()
    assert message.stop_reason == "error"
    assert message.error_message == "Throttling error: Too many tokens, please wait before trying again."
    # Throttling must not be mistaken for a context overflow ("too many tokens").
    assert not is_context_overflow(message, 200_000)
    assert _diagnostic(message).details == {"errorCode": "ThrottlingException", "requestId": REQUEST_ID}


async def test_unmodeled_mid_stream_error(fake_client, keys):
    fake_client.add(
        sdk_response(
            [
                {"messageStart": {"role": "assistant"}},
                stream_exception("ModelStreamErrorException", "Model stream terminated unexpectedly."),
            ],
            headers={"x-amzn-requestid": REQUEST_ID},
        )
    )
    message = await _run_bedrock()
    assert message.error_message == "Model stream terminated unexpectedly."
    assert _diagnostic(message).details == {"errorCode": "ModelStreamErrorException", "requestId": REQUEST_ID}


async def test_transport_failure_has_no_provider_code(fake_client, keys):
    fake_client.add(EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"))
    message = await _run_bedrock()
    assert message.stop_reason == "error"
    assert message.error_message == (
        'Could not connect to the endpoint URL: "https://bedrock-runtime.us-east-1.amazonaws.com"'
    )
    assert _diagnostic(message) is None


async def test_no_diagnostic_for_aborted_turn(fake_client, keys):
    message = await _run_bedrock(signal=AbortSignal.abort())
    assert message.stop_reason == "aborted"
    assert _diagnostic(message) is None
    assert not fake_client.calls


# -- abort ------------------------------------------------------------------------------------------


async def test_abort_closes_the_event_stream(fake_client, keys):
    gate = threading.Event()
    response = sdk_response(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "partial"}}},
        ],
        gate=gate,
    )
    fake_client.add(response)
    controller = AbortController()
    s = stream(_model(), _hello(), {"cache_retention": "none", "signal": controller.signal})
    types = []
    async for item in s:
        types.append(item.type)
        if item.type == "text_delta":
            controller.abort()
    message = await s.result()
    assert types == ["start", "text_start", "text_delta", "error"]
    assert message.stop_reason == "aborted"
    assert message.content == [TextContent("partial")]
    assert response["stream"].closed


async def test_abort_before_the_response_arrives(fake_client, keys):
    release = threading.Event()
    entered = threading.Event()
    response = sdk_response(simple_stop())

    def converse_stream(**params: Any) -> Any:
        entered.set()
        release.wait(10)
        return response

    fake_client.converse_stream = converse_stream  # type: ignore[method-assign]
    controller = AbortController()
    s = stream(_model(), _hello(), {"cache_retention": "none", "signal": controller.signal})
    task = asyncio.ensure_future(s.result())
    while not entered.is_set():
        await asyncio.sleep(0.005)
    controller.abort()
    message = await task
    assert message.stop_reason == "aborted"
    release.set()
    for _ in range(200):
        if response["stream"].closed:
            break
        await asyncio.sleep(0.005)
    # The worker closes the late response instead of reading it.
    assert response["stream"].closed


async def test_many_events_are_delivered_in_order(fake_client, keys):
    deltas = [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": str(i % 10)}}} for i in range(5000)]
    fake_client.add(sdk_response([simple_stop()[0], *deltas, simple_stop()[1]]))
    message = await _run_bedrock()
    assert message.stop_reason == "stop"
    assert message.content == [TextContent("".join(str(i % 10) for i in range(5000)))]
