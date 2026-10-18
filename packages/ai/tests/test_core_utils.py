from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any

import pytest

from pi_ai import (
    AbortController,
    AbortError,
    AbortSignal,
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    RetryCallbacks,
    RetryPolicy,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Type,
    Usage,
    UserMessage,
    abort_scope,
    is_context_overflow,
    is_recoverable_length,
    is_retryable_assistant_error,
    normalize_context,
    retry_assistant_call,
    uuidv7,
)
from pi_ai.api.simple_options import build_base_options
from pi_ai.api.transform_messages import split_error_tool_result_images, transform_messages
from pi_ai.serialization import dumps, from_dict, loads, message_from_dict, model_from_dict, to_dict
from pi_ai.utils.estimate import estimate_context_tokens
from pi_ai.utils.misc import provider_headers_to_record, sanitize_surrogates, short_hash

# --- hashing / ids / unicode -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "k4n83c7h0j2b"),
        ("a", "m8735310ae7sx"),
        ("call_abc|fc_123", "jp01dsxufnom"),
        ("toolu_01ABCDEFGHIJKLMNOPQRSTUVWX", "gdi2ks8d0w8m"),
        ("emoji 🙈 ok", "yd82u41hv4qe6"),
        ("x" * 500, "130aesn1aldl4j"),
    ],
)
def test_short_hash_matches_typescript(text: str, expected: str) -> None:
    # Expected values computed with pi's shortHash in Node.js.
    assert short_hash(text) == expected


_UUID_V7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _timestamp(uuid: str) -> int:
    return int(uuid.replace("-", "")[:12], 16)


def test_uuidv7_ordering_and_followers() -> None:
    ordinary = [uuidv7() for _ in range(5)]
    assert all(_UUID_V7.match(uuid) for uuid in ordinary)
    assert ordinary == sorted(ordinary)
    assert len(set(ordinary)) == len(ordinary)
    followers = [uuidv7(1_000), uuidv7(1_000)]
    assert [_timestamp(uuid) for uuid in followers] == [1_000, 1_000]
    assert len(set(followers)) == 2


@pytest.mark.parametrize("timestamp", [0, 2**48 - 1])
def test_uuidv7_timestamp_boundaries(timestamp: int) -> None:
    assert _timestamp(uuidv7(timestamp)) == timestamp


@pytest.mark.parametrize("timestamp", [-1, 2**48, 1.5])
def test_uuidv7_rejects_invalid_timestamps(timestamp: Any) -> None:
    with pytest.raises(ValueError):
        uuidv7(timestamp)


def test_sanitize_surrogates() -> None:
    assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"
    assert sanitize_surrogates("Text \ud83d here") == "Text  here"
    assert sanitize_surrogates("pair 🙈 joined") == "pair 🙈 joined"
    assert sanitize_surrogates("lone low \ude48") == "lone low "


def test_provider_headers_to_record_is_case_insensitive_and_supports_deletion() -> None:
    merged = provider_headers_to_record({"A": "1", "B": "2"}, {"a": "3", "b": None})
    assert merged == {"a": "3"}
    assert provider_headers_to_record(None, {}) is None


# --- abort -------------------------------------------------------------------


async def test_abort_scope_interrupts_pending_awaits() -> None:
    controller = AbortController()
    asyncio.get_running_loop().call_later(0.01, controller.abort)
    with pytest.raises(AbortError):
        async with abort_scope(controller.signal):
            await asyncio.sleep(10)


async def test_abort_scope_raises_immediately_for_aborted_signal() -> None:
    with pytest.raises(AbortError):
        async with abort_scope(AbortSignal.abort("stop")):
            pass


async def test_abort_scope_lets_external_cancellation_propagate() -> None:
    controller = AbortController()

    async def body() -> None:
        async with abort_scope(controller.signal):
            await asyncio.sleep(10)

    task = asyncio.ensure_future(body())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_abort_from_another_thread() -> None:
    controller = AbortController()
    threading.Timer(0.01, controller.abort).start()
    with pytest.raises(AbortError):
        async with abort_scope(controller.signal):
            await asyncio.sleep(10)


async def test_signal_any_and_timeout() -> None:
    first = AbortController()
    combined = AbortSignal.any([first.signal, None])
    assert not combined.aborted
    first.abort("why")
    assert combined.aborted and str(combined.reason) == "why"
    timed = AbortSignal.timeout(0.01)
    await timed.wait()
    assert timed.aborted and isinstance(timed.reason, TimeoutError)


def test_signal_listeners_and_removal() -> None:
    controller = AbortController()
    calls: list[str] = []
    remove = controller.signal.add_listener(lambda: calls.append("a"))
    controller.signal.add_listener(lambda: calls.append("b"))
    remove()
    controller.abort()
    controller.abort()
    assert calls == ["b"]
    late: list[str] = []
    controller.signal.add_listener(lambda: late.append("late"))
    assert late == ["late"]


# --- estimate ------------------------------------------------------------------

_MODEL = Model(
    id="test-model",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    context_window=10_000,
    max_tokens=8_000,
)


def _assistant(timestamp: int, total: int) -> AssistantMessage:
    return AssistantMessage(
        [TextContent("kept")],
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=Usage(total, 0, 0, 0, total),
        timestamp=timestamp,
    )


def test_estimate_reserves_3_5_chars_per_token() -> None:
    context = normalize_context(Context(messages=[_assistant(100, 2_000), UserMessage("x" * 3_500, timestamp=200)]))
    estimate = estimate_context_tokens(context)
    assert (estimate.tokens, estimate.usage_tokens, estimate.trailing_tokens, estimate.last_usage_index) == (
        3_000,
        2_000,
        1_000,
        0,
    )
    assert build_base_options(_MODEL, context)["max_tokens"] == 2_904


def test_estimate_ignores_stale_usage_after_inserted_message() -> None:
    context = normalize_context(
        Context(
            system_prompt="system",
            messages=[
                UserMessage("summary", timestamp=200),
                _assistant(100, 9_500),
                UserMessage("x" * 4_000, timestamp=300),
            ],
        )
    )
    estimate = estimate_context_tokens(context)
    assert (estimate.tokens, estimate.usage_tokens, estimate.trailing_tokens, estimate.last_usage_index) == (
        1_149,
        0,
        1_149,
        None,
    )
    assert build_base_options(_MODEL, context)["max_tokens"] == 4_755


def test_estimate_uses_usage_again_after_response() -> None:
    context = normalize_context(
        Context(
            messages=[
                UserMessage("summary", timestamp=200),
                _assistant(100, 9_500),
                UserMessage("new prompt", timestamp=300),
                _assistant(400, 2_000),
                UserMessage("tail", timestamp=500),
            ]
        )
    )
    estimate = estimate_context_tokens(context)
    assert (estimate.tokens, estimate.usage_tokens, estimate.trailing_tokens, estimate.last_usage_index) == (
        2_002,
        2_000,
        2,
        3,
    )


# --- transform_messages -------------------------------------------------------

_KIMI = Model(
    id="moonshotai/Kimi-K3",
    api="openai-completions",
    provider="sference",
    base_url="https://api.sference.com/v1",
    reasoning=True,
    input=["text", "image"],
)


def _from(model: str, content: list[Any], stop_reason: str = "toolUse") -> AssistantMessage:
    return AssistantMessage(
        content, api="openai-completions", provider="sference", model=model, stop_reason=stop_reason
    )  # type: ignore[arg-type]


# https://github.com/earendil-works/pi/issues/10995
def test_does_not_send_other_models_thinking_as_text() -> None:
    messages = [
        UserMessage("Why does curl work but node fails?"),
        _from(
            "zai-org/GLM-5.3",
            [
                ThinkingContent("Candidates: proxy, IPv6, TLS.", thinking_signature="reasoning_content"),
                TextContent("Let me check the TLS versions."),
                ToolCall("call_1", "bash", {"command": "curl -v"}),
            ],
        ),
        ToolResultMessage("call_1", "bash", [TextContent("TLSv1.2")], False),
    ]
    result = transform_messages(messages, _KIMI)  # type: ignore[arg-type]
    assistant = next(message for message in result if message.role == "assistant")
    assert assistant.content == [
        TextContent("Let me check the TLS versions."),
        ToolCall("call_1", "bash", {"command": "curl -v"}),
    ]


def test_keeps_thinking_only_from_target_model() -> None:
    def thinking(text: str) -> ThinkingContent:
        return ThinkingContent(text, thinking_signature="reasoning_content")

    messages = [
        UserMessage("1"),
        _from("moonshotai/Kimi-K3", [thinking("kimi 1"), TextContent("a")], "stop"),
        UserMessage("2"),
        _from("zai-org/GLM-5.3", [thinking("glm 1"), TextContent("b")], "stop"),
        UserMessage("3"),
        _from("moonshotai/Kimi-K3", [thinking("kimi 2"), TextContent("c")], "stop"),
        UserMessage("hi"),
    ]
    contents = [message.content for message in transform_messages(messages, _KIMI) if message.role == "assistant"]  # type: ignore[arg-type]
    assert contents == [
        [thinking("kimi 1"), TextContent("a")],
        [TextContent("b")],
        [thinking("kimi 2"), TextContent("c")],
    ]


def test_lax_content_is_normalized() -> None:
    text_only = Model(id="test-model", api="openai-completions", provider="openai", input=["text"])
    messages = [
        UserMessage(None),  # type: ignore[arg-type]
        AssistantMessage(None, api="openai-completions", provider="openai", model="test-model"),  # type: ignore[arg-type]
        ToolResultMessage("call_1", "web_search", None),  # type: ignore[arg-type]
    ]
    result = transform_messages(messages, text_only)  # type: ignore[arg-type]
    assert len(result) == 3
    assert all(message.content == [] for message in result)


def test_synthesizes_results_for_orphaned_tool_calls_and_holds_system_messages() -> None:
    model = Model(id="m", api="openai-completions", provider="p")
    messages = [
        UserMessage("go"),
        AssistantMessage(
            [ToolCall("a", "read", {}), ToolCall("b", "read", {})],
            api="openai-completions",
            provider="p",
            model="m",
            stop_reason="toolUse",
        ),
        SystemMessage("late instruction"),
        ToolResultMessage("a", "read", [TextContent("ok")]),
        UserMessage("next"),
    ]
    result = transform_messages(messages, model)  # type: ignore[arg-type]
    assert [message.role for message in result] == ["user", "assistant", "toolResult", "toolResult", "system", "user"]
    synthetic = result[3]
    assert (
        synthetic.tool_call_id == "b"
        and synthetic.is_error
        and synthetic.content == [TextContent("No result provided")]
    )


def test_skips_errored_assistant_turns_and_downgrades_images() -> None:
    model = Model(id="m", api="openai-completions", provider="p", input=["text"])
    messages = [
        UserMessage([TextContent("look"), ImageContent("abc", "image/png"), ImageContent("def", "image/png")]),
        AssistantMessage(
            [TextContent("partial")], api="openai-completions", provider="p", model="m", stop_reason="error"
        ),
    ]
    result = transform_messages(messages, model)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0].content == [TextContent("look"), TextContent("(image omitted: model does not support images)")]


def test_normalizes_tool_call_ids_for_other_models() -> None:
    model = Model(id="claude", api="anthropic-messages", provider="anthropic")
    messages = [
        AssistantMessage(
            [ToolCall("call|weird", "read", {})],
            api="openai-responses",
            provider="openai",
            model="gpt",
            stop_reason="toolUse",
        ),
        ToolResultMessage("call|weird", "read", [TextContent("ok")]),
    ]
    result = transform_messages(messages, model, lambda tool_id, _model, _source: tool_id.replace("|", "_"))  # type: ignore[arg-type]
    assert result[0].content[0].id == "call_weird"
    assert result[1].tool_call_id == "call_weird"


def test_split_error_tool_result_images() -> None:
    message = ToolResultMessage(
        "c1", "shot", [TextContent("failed"), ImageContent("a", "image/png"), ImageContent("b", "image/png")], True
    )
    split = split_error_tool_result_images(message)
    assert split is not None
    assert split.content == [TextContent("failed"), TextContent("(image attached after the tool results)")]
    assert len(split.images) == 2
    assert "c1 (shot)" in split.caption
    assert (
        split_error_tool_result_images(ToolResultMessage("c1", "shot", [ImageContent("a", "image/png")], False)) is None
    )


# --- overflow / retry ----------------------------------------------------------------


def _error(text: str, provider: str = "openai", stop_reason: str = "error") -> AssistantMessage:
    return AssistantMessage([], api="x", provider=provider, model="m", stop_reason=stop_reason, error_message=text)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "prompt is too long: 213462 tokens > 200000 maximum",
        "Your input exceeds the context window of this model",
        "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
        "Please reduce the length of the messages or completion",
        "the request exceeds the available context size, try increasing it",
        "Range of input length should be [1, 30720]",
    ],
)
def test_detects_context_overflow(text: str) -> None:
    assert is_context_overflow(_error(text))


def test_overflow_excludes_rate_limits_and_detects_silent_overflow() -> None:
    assert not is_context_overflow(_error("Throttling error: Too many tokens"))
    assert is_context_overflow(_error("400 status code (no body)", provider="cerebras"))
    silent = AssistantMessage(
        [], api="x", provider="zai", model="m", usage=Usage(120, 10, 0, 0, 130), stop_reason="stop"
    )
    assert is_context_overflow(silent, 100)
    truncated = AssistantMessage(
        [], api="x", provider="xiaomi", model="m", usage=Usage(100, 0, 0, 0, 100), stop_reason="length"
    )
    assert is_context_overflow(truncated, 100)
    assert is_recoverable_length(
        AssistantMessage([], api="x", provider="p", model="m", usage=Usage(0, 5), stop_reason="length"), 10
    )


@pytest.mark.parametrize(
    ("text", "retryable"),
    [
        ("429 Too Many Requests", True),
        ("Overloaded", True),
        ("Connection error.", True),
        ("Request timed out.", True),
        ("Anthropic stream ended before message_stop", True),
        ("insufficient_quota: billing", False),
        ("400 invalid request", False),
    ],
)
def test_retry_classification(text: str, retryable: bool) -> None:
    assert is_retryable_assistant_error(_error(text)) is retryable


async def test_retry_assistant_call_retries_transient_errors() -> None:
    responses = [_error("503 Service Unavailable"), _error("overloaded"), _error("ok", stop_reason="stop")]
    events: list[Any] = []

    async def produce() -> AssistantMessage:
        return responses.pop(0)

    callbacks = RetryCallbacks(
        on_retry_scheduled=lambda attempt, max_attempts, delay, error: events.append(("scheduled", attempt, delay)),
        on_retry_attempt_start=lambda: events.append(("start",)),
        on_retry_finished=lambda success, attempt, error: events.append(("finished", success, attempt)),
    )
    result = await retry_assistant_call(produce, RetryPolicy(max_retries=3, base_delay_ms=1), None, callbacks)
    assert result.stop_reason == "stop"
    assert events == [("scheduled", 1, 1), ("start",), ("scheduled", 2, 2), ("start",), ("finished", True, 2)]


async def test_retry_assistant_call_normalizes_abort_during_backoff() -> None:
    controller = AbortController()

    async def produce() -> AssistantMessage:
        asyncio.get_running_loop().call_later(0.01, controller.abort)
        return _error("503 Service Unavailable")

    result = await retry_assistant_call(produce, RetryPolicy(max_retries=3, base_delay_ms=10_000), controller.signal)
    assert result.stop_reason == "aborted"
    assert result.error_message is None


async def test_retry_assistant_call_fails_fast_for_non_retryable() -> None:
    calls = {"count": 0}

    async def produce() -> AssistantMessage:
        calls["count"] += 1
        return _error("insufficient_quota")

    result = await retry_assistant_call(produce, RetryPolicy(max_retries=3, base_delay_ms=1))
    assert result.stop_reason == "error" and calls["count"] == 1


# --- serialization -------------------------------------------------------------------


def test_round_trips_context_in_pi_json_format() -> None:
    tool = Tool(
        "echo",
        "Echo",
        Type.Object({"text": Type.String()}),
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
    )
    context = Context(
        system_prompt="sys",
        tools=[tool],
        messages=[
            UserMessage([TextContent("hi"), ImageContent("abc", "image/png")], timestamp=1),
            AssistantMessage(
                [
                    ThinkingContent("t", thinking_signature="sig"),
                    ToolCall("c1", "echo", {"text": "x"}, thought_signature="ts"),
                ],
                api="anthropic-messages",
                provider="anthropic",
                model="claude",
                usage=Usage(1, 2, 3, 4, 10, cache_write_1h=2),
                stop_reason="toolUse",
                response_id="msg_1",
                timestamp=2,
            ),
            ToolResultMessage(
                "c1", "echo", [TextContent("x")], False, details={"keep": {"snake_case": 1}}, timestamp=3
            ),
        ],
    )
    data = json.loads(dumps(context))
    assert data["systemPrompt"] == "sys"
    assistant = data["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["stopReason"] == "toolUse"
    assert assistant["responseId"] == "msg_1"
    assert assistant["usage"]["cacheWrite1h"] == 2
    assert assistant["usage"]["cost"] == {
        "input": 0.0,
        "output": 0.0,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
        "total": 0.0,
    }
    assert assistant["content"][1] == {
        "type": "toolCall",
        "id": "c1",
        "name": "echo",
        "arguments": {"text": "x"},
        "thoughtSignature": "ts",
    }
    tool_result = data["messages"][2]
    assert tool_result["toolCallId"] == "c1" and tool_result["isError"] is False
    assert tool_result["details"] == {"keep": {"snake_case": 1}}
    assert data["messages"][0]["content"][1] == {"type": "image", "data": "abc", "mimeType": "image/png"}
    assert data["tools"][0]["constrainedSampling"] == {"type": "json_schema", "strict": "prefer"}
    assert loads(dumps(context)) == context


def test_decodes_typescript_messages_and_models() -> None:
    message = message_from_dict(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi", "textSignature": "s"}],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt",
            "usage": {
                "input": 1,
                "output": 2,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 3,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 5,
            "durationMs": 7,
            "futureField": "ignored",
        }
    )
    assert isinstance(message, AssistantMessage)
    assert message.content == [TextContent("hi", text_signature="s")]
    assert message.duration_ms == 7
    assert message_from_dict({"role": "notification", "text": "custom"}) == {"role": "notification", "text": "custom"}

    model = model_from_dict(
        {
            "id": "m",
            "name": "M",
            "api": "openai-completions",
            "provider": "p",
            "baseUrl": "https://x",
            "reasoning": True,
            "input": ["text"],
            "cost": {
                "input": 1,
                "output": 2,
                "cacheRead": 0,
                "cacheWrite": 0,
                "tiers": [{"inputTokensAbove": 10, "input": 2, "output": 4, "cacheRead": 0, "cacheWrite": 0}],
            },
            "contextWindow": 100,
            "maxTokens": 10,
            "compat": {"supportsDeveloperRole": False},
        }
    )
    assert isinstance(model, Model)
    assert model.cost.tiers[0].input_tokens_above == 10  # type: ignore[index]
    assert model.compat == {"supportsDeveloperRole": False}
    encoded = to_dict(model)
    assert (
        encoded["type"] == "chat"
        and encoded["baseUrl"] == "https://x"
        and encoded["compat"] == {"supportsDeveloperRole": False}
    )
    assert from_dict(Model, encoded) == model
