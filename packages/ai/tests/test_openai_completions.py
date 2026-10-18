"""Streaming, wire format and error handling of the openai-completions API."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import httpx
import pytest

from pi_ai import (
    AbortController,
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Type,
    UserMessage,
    normalize_context,
)
from pi_ai.api.openai_completions import stream, stream_simple
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.types import Model, ModelCost

STOP_CHUNK: dict[str, Any] = {
    "choices": [{"delta": {}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}


def ctx(*messages: Any, system_prompt: str | None = None, tools: list[Tool] | None = None):
    return normalize_context(Context(messages=list(messages), system_prompt=system_prompt, tools=tools))


def openai_model(**overrides: Any) -> Model:
    base = get_builtin_model("openai", "gpt-4o-mini")
    assert base is not None
    return dataclasses.replace(base, **{"api": "openai-completions", "compat": None, **overrides})


def catalog(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"{provider}/{model_id} missing from catalog"
    return model


def assistant(content: list[Any], *, model: Model | None = None, stop_reason: str = "stop", **kwargs: Any):
    return AssistantMessage(
        content,
        api=kwargs.pop("api", model.api if model else "openai-completions"),
        provider=kwargs.pop("provider", model.provider if model else "openai"),
        model=kwargs.pop("model_id", model.id if model else "gpt-4o-mini"),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        **kwargs,
    )


async def run(
    mock_server: Any,
    model: Model,
    context: Any,
    options: dict[str, Any] | None = None,
    *,
    chunks: list[Any] | None = None,
    simple: bool = True,
) -> tuple[AssistantMessage, list[Any]]:
    if chunks is not None or not mock_server.responses:
        mock_server.add_sse([*(chunks if chunks is not None else [STOP_CHUNK]), "[DONE]"])
    opts = {"api_key": "test", "http_client": mock_server.client(), **(options or {})}
    s = (stream_simple if simple else stream)(model, context, opts)
    events = [event async for event in s]
    return await s.result(), events


HI = UserMessage("Hi")
READ_TOOL = Tool("read", "Read a file", Type.Object({"path": Type.String()}))


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


async def test_wire_format(mock_server):
    model = openai_model(headers={"X-Model": "model-header"})
    message, events = await run(
        mock_server,
        model,
        ctx(HI, system_prompt="sys"),
        {"api_key": "sk-test", "headers": {"X-Extra": "1"}, "max_tokens": 100},
        simple=False,
    )
    assert message.stop_reason == "stop"
    assert [event.type for event in events] == ["start", "done"]
    request = mock_server.last
    assert request.method == "POST"
    assert request.url == "https://api.openai.com/v1/chat/completions"
    headers = request.headers
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["accept"] == "application/json"
    assert headers["content-type"] == "application/json"
    assert headers["user-agent"].startswith("pi (")
    assert headers["x-model"] == "model-header"
    assert headers["x-extra"] == "1"
    assert request.json() == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": 100,
    }


async def test_options_headers_override_and_suppress_defaults(mock_server):
    model = openai_model(base_url="https://proxy.test/v1/", headers={"X-Model": "a"})
    await run(
        mock_server,
        model,
        ctx(HI),
        {"headers": {"User-Agent": "custom", "x-model": None, "Authorization": None}},
        simple=False,
    )
    request = mock_server.last
    assert request.url == "https://proxy.test/v1/chat/completions"
    assert request.headers["user-agent"] == "custom"
    assert "x-model" not in request.headers
    assert "authorization" not in request.headers


async def test_empty_base_url_uses_openai_default(mock_server):
    await run(mock_server, openai_model(base_url=""), ctx(HI), simple=False)
    assert mock_server.last.url == "https://api.openai.com/v1/chat/completions"


def test_stream_simple_throws_synchronously_without_api_key():
    model = Model(id="test-model", api="openai-completions", provider="test-provider", base_url="https://x.invalid")
    with pytest.raises(RuntimeError, match="No API key for provider: test-provider"):
        stream_simple(model, ctx(UserMessage("hello")), {})


async def test_stream_reports_missing_api_key_as_error_event():
    model = Model(id="test-model", api="openai-completions", provider="test-provider", base_url="https://x.invalid")
    message = await stream(model, ctx(UserMessage("hello")), {}).result()
    assert message.stop_reason == "error"
    assert message.error_message == "No API key for provider: test-provider"


async def test_authorization_header_stands_in_for_api_key(mock_server):
    model = catalog("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6")
    model = dataclasses.replace(model, base_url="https://gateway.ai.cloudflare.com/v1/acc/gw/compat")
    message, _ = await run(
        mock_server,
        model,
        ctx(HI, system_prompt="You are helpful."),
        {
            "api_key": None,
            "headers": {"Authorization": None, "cf-aig-authorization": "Bearer cf-token"},
            "max_tokens": 1234,
            "reasoning": "high",
            "session_id": "session-1",
        },
    )
    assert message.stop_reason == "stop"
    request = mock_server.last
    assert "authorization" not in request.headers
    assert request.headers["cf-aig-authorization"] == "Bearer cf-token"
    assert request.headers["session_id"] == "session-1"
    assert request.headers["x-client-request-id"] == "session-1"
    assert request.headers["x-session-affinity"] == "session-1"
    payload = request.json()
    assert payload["messages"][0]["role"] == "system"
    assert payload["max_tokens"] == 1234
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "store" not in payload


async def test_inline_authorization_is_preserved(mock_server):
    await run(
        mock_server,
        openai_model(),
        ctx(HI),
        {"api_key": None, "headers": {"Authorization": "Bearer upstream-token"}},
    )
    assert mock_server.last.headers["authorization"] == "Bearer upstream-token"


async def test_github_copilot_dynamic_headers(mock_server):
    model = Model(
        id="gpt-4o",
        api="openai-completions",
        provider="github-copilot",
        base_url="https://api.individual.githubcopilot.com",
        input=["text", "image"],
    )
    image_message = UserMessage([TextContent("look"), ImageContent("ZmFrZQ==", "image/png")])
    await run(mock_server, model, ctx(image_message))
    headers = mock_server.last.headers
    assert headers["x-initiator"] == "user"
    assert headers["openai-intent"] == "conversation-edits"
    assert headers["copilot-vision-request"] == "true"

    tool_call = ToolCall("call_1", "read", {"path": "a"})
    history = ctx(
        UserMessage("read"),
        assistant([tool_call], model=model, stop_reason="toolUse"),
        ToolResultMessage("call_1", "read", [TextContent("ok")]),
    )
    await run(mock_server, model, history)
    headers = mock_server.last.headers
    assert headers["x-initiator"] == "agent"
    assert "copilot-vision-request" not in headers


async def test_on_payload_replaces_body_and_on_response_sees_status(mock_server):
    seen: dict[str, Any] = {}

    def on_payload(payload: Any, model: Model) -> Any:
        seen["model"] = payload["model"]
        return {**payload, "temperature": 0.1}

    async def on_response(response: Any, model: Model) -> None:
        seen["status"] = response.status
        seen["request_id"] = response.headers.get("x-request-id")

    mock_server.add_sse([STOP_CHUNK, "[DONE]"], headers={"x-request-id": "req-1"})
    await run(mock_server, openai_model(), ctx(HI), {"on_payload": on_payload, "on_response": on_response})
    assert seen == {"model": "gpt-4o-mini", "status": 200, "request_id": "req-1"}
    assert mock_server.last.json()["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Stream handling
# ---------------------------------------------------------------------------


async def test_maps_non_standard_finish_reason_to_error(mock_server):
    chunks = [
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
        {**STOP_CHUNK, "choices": [{"delta": {}, "finish_reason": "network_error"}]},
    ]
    message, _ = await run(mock_server, catalog("zai", "glm-5.2"), ctx(HI), chunks=chunks)
    assert message.stop_reason == "error"
    assert message.error_message == "Provider finish_reason: network_error"


async def test_ignores_null_stream_chunks(mock_server):
    chunks = [
        None,
        {"id": "chatcmpl-test", "choices": [{"delta": {"content": "OK"}, "finish_reason": None}]},
        {
            "id": "chatcmpl-test",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    ]
    message, _ = await run(mock_server, openai_model(), ctx(UserMessage("Reply with exactly OK")), chunks=chunks)
    assert message.stop_reason == "stop"
    assert message.error_message is None
    assert message.response_id == "chatcmpl-test"
    assert message.usage.total_tokens == 4
    assert message.content == [TextContent("OK")]


async def test_errors_when_stream_ends_without_finish_reason(mock_server):
    chunks = [
        {"id": "chatcmpl-truncated", "choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]},
        {"id": "chatcmpl-truncated", "choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]},
    ]
    message, events = await run(mock_server, openai_model(), ctx(HI), chunks=chunks)
    assert message.stop_reason == "error"
    assert message.error_message == "Stream ended without finish_reason"
    assert events[-1].type == "error"


async def test_accepts_streams_without_finish_reason_when_compat_disables_it(mock_server):
    chunks = [{"id": "x", "choices": [{"delta": {"content": "complete answer"}, "finish_reason": None}]}]
    model = openai_model(compat={"supportsFinishReason": False})
    message, _ = await run(mock_server, model, ctx(HI), chunks=chunks)
    assert message.stop_reason == "stop"
    assert message.error_message is None
    assert message.content == [TextContent("complete answer")]

    tool_chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read"}}]}}]},
    ]
    message, _ = await run(mock_server, model, ctx(HI, tools=[READ_TOOL]), chunks=tool_chunks)
    assert message.stop_reason == "toolUse"


async def test_ignores_empty_custom_objects_on_function_tool_call_deltas(mock_server):
    chunks = [
        {
            "id": "chatcmpl-empty-custom",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                                "custom": {},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    ]
    message, _ = await run(mock_server, openai_model(), ctx(HI, tools=[READ_TOOL]), chunks=chunks)
    assert message.content == [ToolCall("call_1", "read", {"path": "README.md"})]


async def test_coalesces_tool_call_deltas_by_stable_index(mock_server):
    def tool_delta(tool_id: str, name: str | None, arguments: str, finish: str | None = None) -> dict[str, Any]:
        return {
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ]
                    },
                    "finish_reason": finish,
                }
            ],
        }

    chunks = [
        tool_delta("functions.read:0", "read", ""),
        tool_delta("chatcmpl-tool-a", None, '{"path":"README'),
        {**tool_delta("chatcmpl-tool-b", None, '.md"}', "tool_calls"), "usage": STOP_CHUNK["usage"]},
    ]
    message, events = await run(mock_server, openai_model(), ctx(HI, tools=[READ_TOOL]), chunks=chunks)
    assert message.stop_reason == "toolUse"
    assert [event.content_index for event in events if event.type.startswith("toolcall_")] == [0, 0, 0, 0, 0]
    assert message.content == [ToolCall("functions.read:0", "read", {"path": "README.md"})]


async def test_accumulates_mixed_content_reasoning_and_parallel_tool_calls(mock_server):
    def call(index: int | None, tool_id: str | None, arguments: str, name: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "function", "function": {"arguments": arguments}}
        if index is not None:
            item["index"] = index
        if tool_id is not None:
            item["id"] = tool_id
        if name is not None:
            item["function"]["name"] = name
        return item

    chunks = [
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": "answer 1",
                        "reasoning_content": "think 1",
                        "tool_calls": [
                            call(0, "tc_read_initial", '{"path":"README', "read"),
                            call(1, "tc_grep_initial", '{"pattern":"TODO', "grep"),
                            call(None, "tc_list_no_index", '{"path":"packages', "list"),
                            call(None, "tc_write_no_index", '{"path":"out', "write"),
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": " answer 2",
                        "tool_calls": [
                            call(1, "tc_grep_changed", '","path":"src'),
                            call(None, "tc_write_no_index", '.txt","content":"ok"}'),
                            call(None, "tc_list_no_index", '/ai"}'),
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mixed-deltas",
            "choices": [
                {
                    "delta": {
                        "content": "\n",
                        "reasoning_content": " think 2",
                        "tool_calls": [call(0, "tc_read_changed", '.md"}'), call(1, None, '"}')],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ]
    tools = [
        READ_TOOL,
        Tool("grep", "Search", Type.Object({"pattern": Type.String(), "path": Type.String()})),
        Tool("list", "List", Type.Object({"path": Type.String()})),
        Tool("write", "Write", Type.Object({"path": Type.String(), "content": Type.String()})),
    ]
    message, events = await run(mock_server, openai_model(), ctx(HI, tools=tools), chunks=chunks)
    assert message.stop_reason == "toolUse"
    types = [event.type for event in events]
    for kind, count in {
        "text_start": 1,
        "text_delta": 3,
        "text_end": 1,
        "thinking_start": 1,
        "thinking_delta": 2,
        "thinking_end": 1,
        "toolcall_start": 4,
        "toolcall_delta": 9,
        "toolcall_end": 4,
    }.items():
        assert types.count(kind) == count, kind
    by_index: dict[int, list[str]] = {}
    for event in events:
        if event.type.startswith("toolcall_"):
            by_index.setdefault(event.content_index, []).append(event.type)
    assert by_index[2] == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    assert by_index[3] == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    assert by_index[4] == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    assert by_index[5] == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    assert message.content == [
        ThinkingContent("think 1 think 2", thinking_signature="reasoning_content"),
        TextContent("answer 1 answer 2\n"),
        ToolCall("tc_read_initial", "read", {"path": "README.md"}),
        ToolCall("tc_grep_initial", "grep", {"pattern": "TODO", "path": "src"}),
        ToolCall("tc_list_no_index", "list", {"path": "packages/ai"}),
        ToolCall("tc_write_no_index", "write", {"path": "out.txt", "content": "ok"}),
    ]
    assert message.usage.reasoning == 2


async def test_streams_partial_tool_arguments_and_reports_invalid_json(mock_server):
    deltas = ['{"pa', 'th":"a"', " oops"]
    gates = [asyncio.Event() for _ in deltas]

    async def body():
        for index, arguments in enumerate(deltas):
            call: dict[str, Any] = {"index": 0, "function": {"arguments": arguments}}
            if index == 0:
                call.update(id="c1", function={"name": "read", "arguments": arguments})
            finish = "tool_calls" if index == len(deltas) - 1 else None
            data = {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": finish}]}
            yield f"data: {json.dumps(data)}\n\n".encode()
            await gates[index].wait()
        yield b"data: [DONE]\n\n"

    mock_server.add(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    s = stream(openai_model(), ctx(HI, tools=[READ_TOOL]), {"api_key": "k", "http_client": mock_server.client()})
    seen: list[Any] = []
    async for event in s:
        if event.type == "toolcall_delta":
            seen.append(dict(event.partial.content[0].arguments))
            gates[len(seen) - 1].set()
    message = await s.result()
    assert seen == [{}, {"path": "a"}, {"path": "a"}]
    tool_call = message.content[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.arguments == {}
    assert tool_call.arguments_error is not None
    assert tool_call.arguments_error.endswith('Received arguments:\n{"path":"a" oops')


async def test_opencode_go_reasoning_deltas_use_reasoning_content(mock_server):
    chunks = [{"id": "x", "choices": [{"delta": {"reasoning": "think"}, "finish_reason": "stop"}]}]
    model = dataclasses.replace(catalog("opencode-go", "kimi-k3"), compat=None)
    message, _ = await run(mock_server, model, ctx(UserMessage("Use reasoning.")), chunks=chunks)
    assert message.content == [ThinkingContent("think", thinking_signature="reasoning_content")]

    message, _ = await run(mock_server, openai_model(), ctx(UserMessage("Use reasoning.")), chunks=chunks)
    assert message.content == [ThinkingContent("think", thinking_signature="reasoning")]


async def test_uses_first_non_empty_reasoning_field(mock_server):
    chunks = [{"choices": [{"delta": {"reasoning_content": "", "reasoning": "a", "reasoning_text": "a"}}]}, STOP_CHUNK]
    message, _ = await run(mock_server, openai_model(), ctx(HI), chunks=chunks)
    assert message.content == [ThinkingContent("a", thinking_signature="reasoning")]


async def test_grammar_tool_calls_stream_as_json_arguments(mock_server):
    model = openai_model(compat={"supportsOpenAIGrammarTools": True})
    tool = Tool(
        "apply_patch",
        "Apply a patch",
        Type.Object({"patch": Type.String()}),
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/"}},
    )
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "custom",
                                "custom": {"name": "apply_patch", "input": 'a"b'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "type": "custom", "custom": {"input": "c"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    message, events = await run(mock_server, model, ctx(HI, tools=[tool]), chunks=chunks)
    payload = mock_server.last.json()
    assert payload["tools"] == [
        {
            "type": "custom",
            "custom": {
                "name": "apply_patch",
                "description": "Apply a patch",
                "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: /.+/"}},
            },
        }
    ]
    deltas = [event.delta for event in events if event.type == "toolcall_delta"]
    assert "".join(deltas) == '{"patch":"a\\"bc"}'
    assert json.loads("".join(deltas)) == {"patch": 'a"bc'}
    assert message.content == [ToolCall("call_1", "apply_patch", {"patch": 'a"bc'})]
    assert message.stop_reason == "toolUse"


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def usage_chunk(usage: dict[str, Any], *, in_choice: bool = False) -> dict[str, Any]:
    choice: dict[str, Any] = {"delta": {}, "finish_reason": "stop"}
    if in_choice:
        return {"id": "u", "choices": [{**choice, "usage": usage}]}
    return {"id": "u", "choices": [choice], "usage": usage}


async def test_does_not_double_count_reasoning_tokens(mock_server):
    usage = {"prompt_tokens": 10, "completion_tokens": 33, "completion_tokens_details": {"reasoning_tokens": 21}}
    message, _ = await run(mock_server, openai_model(), ctx(HI), chunks=[usage_chunk(usage)])
    assert (message.usage.input, message.usage.output, message.usage.total_tokens) == (10, 33, 43)
    assert message.usage.reasoning == 21


@pytest.mark.parametrize("in_choice", [False, True])
async def test_preserves_cache_read_and_write_tokens(mock_server, in_choice):
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    chunks = [
        {"id": "c", "choices": [{"delta": {"content": "OK"}, "finish_reason": None}]},
        usage_chunk(usage, in_choice=in_choice),
    ]
    message, _ = await run(mock_server, openai_model(), ctx(HI), chunks=chunks)
    # cached_tokens is documented as cache reads; cache_write_tokens is separate.
    assert message.usage.input == 20
    assert message.usage.cache_read == 50
    assert message.usage.cache_write == 30
    assert message.usage.total_tokens == 105


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 100, "completion_tokens": 5, "prompt_cache_hit_tokens": 40},  # DeepSeek
        {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40},  # Kimi
    ],
)
async def test_alternative_cached_token_fields(mock_server, usage):
    message, _ = await run(mock_server, openai_model(), ctx(HI), chunks=[usage_chunk(usage)])
    assert (message.usage.input, message.usage.cache_read, message.usage.total_tokens) == (60, 40, 105)


async def test_usage_cost_uses_model_pricing(mock_server):
    model = openai_model(cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=0))
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "prompt_tokens_details": {"cached_tokens": 500_000},
    }
    message, _ = await run(mock_server, model, ctx(HI), chunks=[usage_chunk(usage)])
    assert message.usage.cost.input == pytest.approx(0.5)
    assert message.usage.cost.output == pytest.approx(2.0)
    assert message.usage.cost.cache_read == pytest.approx(0.25)
    assert message.usage.cost.total == pytest.approx(2.75)


# ---------------------------------------------------------------------------
# Stop reasons, response metadata, provider events
# ---------------------------------------------------------------------------

RAW_MODEL = Model(
    id="test-model",
    api="openai-completions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    context_window=128_000,
    max_tokens=4096,
)


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason", "error_message"),
    [
        ("stop", "stop", None),
        ("end", "stop", None),
        ("length", "length", None),
        ("tool_calls", "toolUse", None),
        ("function_call", "toolUse", None),
        ("content_filter", "error", "Provider finish_reason: content_filter"),
    ],
)
async def test_raw_stop_reasons(mock_server, finish_reason, stop_reason, error_message):
    chunks = [{"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}]
    message, _ = await run(mock_server, RAW_MODEL, ctx(UserMessage("hello")), chunks=chunks, simple=False)
    assert message.stop_reason == stop_reason
    assert message.raw_stop_reason == finish_reason
    assert message.error_message == error_message


OPENROUTER_AUTO = Model(
    id="openrouter/auto",
    api="openai-completions",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    context_window=200_000,
    max_tokens=8192,
)


async def test_response_model_surfaces_routed_model(mock_server):
    chunks = [
        {
            "id": "chatcmpl-1",
            "model": "anthropic/claude-opus-4.8",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        },
        {
            "id": "chatcmpl-1",
            "model": "anthropic/claude-opus-4.8",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    message, _ = await run(mock_server, OPENROUTER_AUTO, ctx(UserMessage("hi")), chunks=chunks, simple=False)
    assert message.model == "openrouter/auto"
    assert message.response_model == "anthropic/claude-opus-4.8"
    assert message.provider == "openrouter"
    assert message.stop_reason == "stop"


@pytest.mark.parametrize("chunk_models", [["openrouter/auto", "openrouter/auto"], [None, ""]])
async def test_response_model_stays_unset_for_echo_or_missing_model(mock_server, chunk_models):
    chunks = []
    for chunk_model in chunk_models:
        chunk: dict[str, Any] = {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
        if chunk_model is not None:
            chunk["model"] = chunk_model
        chunks.append(chunk)
    chunks.append({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    message, _ = await run(mock_server, OPENROUTER_AUTO, ctx(UserMessage("hi")), chunks=chunks, simple=False)
    assert message.model == "openrouter/auto"
    assert message.response_model is None


async def test_exposes_provider_chunks_including_openrouter_metadata(mock_server):
    first = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    final = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0012, "is_byok": False},
        "openrouter_metadata": {"strategy": "direct", "region": "iad"},
    }
    events: list[Any] = []
    model = dataclasses.replace(OPENROUTER_AUTO, reasoning=False)
    message, _ = await run(
        mock_server,
        model,
        ctx(UserMessage("hi")),
        {"on_provider_stream_event": lambda data, _model: events.append(data)},
        chunks=[first, final],
    )
    assert message.content == [TextContent("hello")]
    assert events == [first, final]


# ---------------------------------------------------------------------------
# reasoning_details streaming and replay
# ---------------------------------------------------------------------------

REASONING_DETAIL = {"type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature"}
SIGNED_TEXT_DETAIL = {
    "type": "reasoning.text",
    "text": "I should call the read tool.",
    "signature": "sha256:signed-text",
    "id": "reasoning-text-1",
    "format": "anthropic-claude-v1",
    "index": 0,
}
SUMMARY_DETAIL = {
    "type": "reasoning.summary",
    "summary": "Decided to inspect the requested file.",
    "id": "reasoning-summary-1",
    "format": "anthropic-claude-v1",
    "index": 1,
}
GEMINI = Model(
    id="google/gemini-test",
    api="openai-completions",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    reasoning=True,
    context_window=100_000,
    max_tokens=4096,
)


def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "google/gemini-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


TOOL_CALL_CHUNK = chunk(
    {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
            }
        ]
    }
)
FOLLOW_UP = [chunk({"content": "ok"}), chunk({}, "stop")]


async def run_gemini(mock_server, chunks: list[Any], messages: list[Any] | None = None) -> AssistantMessage:
    message, _ = await run(mock_server, GEMINI, ctx(*(messages or []), tools=[READ_TOOL]), chunks=chunks, simple=False)
    return message


def assistant_payload(mock_server) -> dict[str, Any]:
    return next(m for m in mock_server.last.json()["messages"] if m["role"] == "assistant")


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


async def test_preserves_reasoning_details_in_thinking_signature(mock_server):
    message = await run_gemini(
        mock_server, [chunk({"reasoning_details": [REASONING_DETAIL]}), TOOL_CALL_CHUNK, chunk({}, "tool_calls")]
    )
    thinking = next(block for block in message.content if block.type == "thinking")
    assert thinking == ThinkingContent("", thinking_signature=compact([REASONING_DETAIL]))
    tool_call = next(block for block in message.content if block.type == "toolCall")
    assert tool_call == ToolCall("call_1", "read", {"path": "README.md"})

    await run_gemini(mock_server, FOLLOW_UP, [message])
    assert assistant_payload(mock_server)["reasoning_details"] == [REASONING_DETAIL]


async def test_falls_back_to_encrypted_tool_call_signatures(mock_server):
    message = await run_gemini(
        mock_server, [chunk({"reasoning_details": [REASONING_DETAIL]}), TOOL_CALL_CHUNK, chunk({}, "tool_calls")]
    )
    message.content = [block for block in message.content if block.type != "thinking"]
    message.content[0].thought_signature = json.dumps(REASONING_DETAIL)

    await run_gemini(mock_server, FOLLOW_UP, [message])
    assert assistant_payload(mock_server)["reasoning_details"] == [REASONING_DETAIL]


async def test_preserves_signed_text_and_summary_details_in_order(mock_server):
    message = await run_gemini(
        mock_server,
        [
            chunk({"reasoning": SIGNED_TEXT_DETAIL["text"], "reasoning_details": [SIGNED_TEXT_DETAIL]}),
            chunk({"reasoning_details": [REASONING_DETAIL, SUMMARY_DETAIL]}),
            TOOL_CALL_CHUNK,
            chunk({}, "tool_calls"),
        ],
    )
    expected = [SIGNED_TEXT_DETAIL, REASONING_DETAIL, SUMMARY_DETAIL]
    thinking = next(block for block in message.content if block.type == "thinking")
    assert thinking == ThinkingContent(SIGNED_TEXT_DETAIL["text"], thinking_signature=compact(expected))

    await run_gemini(mock_server, FOLLOW_UP, [message])
    payload = assistant_payload(mock_server)
    assert payload["reasoning_details"] == expected
    assert "reasoning" not in payload


async def test_merges_consecutive_text_and_summary_deltas(mock_server):
    text_delta = {"type": "reasoning.text", "text": "The", "index": 0}
    text_delta_signed = {
        "type": "reasoning.text",
        "text": " user wants the time.",
        "signature": "sha256:text-signature",
        "format": "openai-responses-v1",
        "index": 0,
    }
    summary_delta = {"type": "reasoning.summary", "summary": "Looked", "index": 0}
    summary_delta_format = {
        "type": "reasoning.summary",
        "summary": " up time.",
        "format": "openai-responses-v1",
        "index": 0,
    }
    later_summary = {
        "type": "reasoning.summary",
        "summary": "After encrypted block.",
        "format": "openai-responses-v1",
        "index": 0,
    }
    expected = [
        {
            "type": "reasoning.text",
            "text": "The user wants the time.",
            "index": 0,
            "signature": "sha256:text-signature",
            "format": "openai-responses-v1",
        },
        {"type": "reasoning.summary", "summary": "Looked up time.", "index": 0, "format": "openai-responses-v1"},
        REASONING_DETAIL,
        later_summary,
    ]
    message = await run_gemini(
        mock_server,
        [
            chunk({"reasoning_details": [text_delta]}),
            chunk({"reasoning_details": [text_delta_signed]}),
            chunk({"reasoning_details": [summary_delta]}),
            chunk({"reasoning_details": [summary_delta_format]}),
            chunk({"reasoning_details": [REASONING_DETAIL]}),
            chunk({"reasoning_details": [later_summary]}),
            TOOL_CALL_CHUNK,
            chunk({}, "tool_calls"),
        ],
    )
    thinking = next(block for block in message.content if block.type == "thinking")
    assert thinking == ThinkingContent("", thinking_signature=compact(expected))
    # The streamed deltas themselves are not mutated.
    assert text_delta == {"type": "reasoning.text", "text": "The", "index": 0}

    await run_gemini(mock_server, FOLLOW_UP, [message])
    assert assistant_payload(mock_server)["reasoning_details"] == expected


async def test_merged_details_drop_fields_cleared_by_later_deltas(mock_server):
    # ``target.id ??= source.id`` assigns undefined when the later delta has no id.
    message = await run_gemini(
        mock_server,
        [
            chunk({"reasoning_details": [{"type": "reasoning.text", "text": "a", "id": None, "format": ""}]}),
            chunk({"reasoning_details": [{"type": "reasoning.text", "text": "b"}]}),
            chunk({}, "stop"),
        ],
    )
    assert message.content[0].thinking_signature == '[{"type":"reasoning.text","text":"ab"}]'


async def test_ignores_invalid_reasoning_details(mock_server):
    message = await run_gemini(
        mock_server,
        [
            chunk({"reasoning_details": [{"type": "reasoning.text", "text": 1}, {"type": "other"}, "x"]}),
            chunk({}, "stop"),
        ],
    )
    assert message.content == []


async def test_places_thinking_before_text_in_combined_deltas(mock_server):
    message = await run_gemini(
        mock_server, [chunk({"role": "assistant", "reasoning_content": "thinking", "content": "answer"}, "stop")]
    )
    assert message.content == [
        ThinkingContent("thinking", thinking_signature="reasoning_content"),
        TextContent("answer"),
    ]
    message = await run_gemini(
        mock_server, [chunk({"reasoning_details": [REASONING_DETAIL], "content": "answer"}, "stop")]
    )
    assert [block.type for block in message.content] == ["thinking", "text"]


# ---------------------------------------------------------------------------
# Errors and retries
# ---------------------------------------------------------------------------

RETRY_MODEL = Model(
    id="test-model",
    api="openai-completions",
    provider="opencode-go",
    base_url="https://opencode.ai/zen/go/v1",
    context_window=1000,
    max_tokens=100,
)
OK_CHUNKS = [
    {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {"content": "ok"}}]},
    {"id": "chatcmpl-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    "[DONE]",
]


async def test_honors_provider_retries(mock_server):
    mock_server.add_json({"error": {"message": "rate limited"}}, status=429, headers={"retry-after-ms": "100"})
    mock_server.add_json({"error": {"message": "server error"}}, status=500, headers={"retry-after-ms": "100"})
    mock_server.add_sse(OK_CHUNKS)
    message, _ = await run(
        mock_server, RETRY_MODEL, ctx(UserMessage("hi")), {"max_retries": 2, "max_retry_delay_ms": 100}, simple=False
    )
    assert message.stop_reason == "stop"
    assert message.content == [TextContent("ok")]
    assert len(mock_server.requests) == 3


async def test_does_not_retry_by_default(mock_server):
    mock_server.add_json({"error": {"message": "rate limited"}}, status=429)
    message, _ = await run(mock_server, RETRY_MODEL, ctx(UserMessage("hi")), simple=False)
    assert message.stop_reason == "error"
    assert len(mock_server.requests) == 1


async def test_fails_immediately_when_retry_delay_exceeds_limit(mock_server):
    mock_server.add_json({"error": {"message": "rate limited"}}, status=429, headers={"retry-after": "277403"})
    message, _ = await run(
        mock_server, RETRY_MODEL, ctx(UserMessage("hi")), {"max_retries": 2, "max_retry_delay_ms": 1000}, simple=False
    )
    assert message.stop_reason == "error"
    assert "Server requested 277403s retry delay (max: 1s)" in (message.error_message or "")
    assert "rate limited" in (message.error_message or "")
    assert len(mock_server.requests) == 1


ERROR_MODEL = dataclasses.replace(OPENROUTER_AUTO, id="test-model", reasoning=False)


async def test_http_error_surfaces_status_and_body(mock_server):
    mock_server.add_json({"error": "blocked by gateway WAF"}, status=403)
    message, events = await run(mock_server, ERROR_MODEL, ctx(UserMessage("hi")), simple=False)
    assert message.stop_reason == "error"
    assert "403" in (message.error_message or "")
    assert "blocked by gateway WAF" in (message.error_message or "")
    assert message.error_message != "403 status code (no body)"
    assert [event.type for event in events] == ["error"]


async def test_does_not_double_print_openrouter_metadata_raw(mock_server):
    raw = "upstream WAF blocked policy XYZ"
    mock_server.add_json(
        {"error": {"message": "Provider returned error", "code": 403, "metadata": {"raw": raw}}}, status=403
    )
    message, _ = await run(mock_server, ERROR_MODEL, ctx(UserMessage("hi")), simple=False)
    assert (message.error_message or "").count(raw) == 1


async def test_appends_openrouter_metadata_raw_when_body_escapes_it(mock_server):
    raw = 'upstream said "no"'
    mock_server.add_json({"error": {"message": "Provider returned error", "metadata": {"raw": raw}}}, status=400)
    message, _ = await run(mock_server, ERROR_MODEL, ctx(UserMessage("hi")), simple=False)
    assert message.error_message is not None
    assert message.error_message.endswith(f"\n{raw}")


async def test_in_stream_error_chunks_raise_like_the_sdk(mock_server):
    chunks = [
        {"id": "c", "choices": [{"delta": {"content": "partial"}}]},
        {"error": {"message": "Overloaded", "code": 529, "metadata": {"raw": "upstream overloaded"}}},
    ]
    seen: list[Any] = []
    message, events = await run(
        mock_server,
        ERROR_MODEL,
        ctx(UserMessage("hi")),
        {"on_provider_stream_event": lambda data, _model: seen.append(data)},
        chunks=chunks,
        simple=False,
    )
    assert message.stop_reason == "error"
    assert message.error_message == "Overloaded\nupstream overloaded"
    assert message.content == [TextContent("partial")]
    assert seen == [chunks[0]]
    assert events[-1].type == "error"


async def test_aborted_before_request(mock_server):
    controller = AbortController()
    controller.abort()
    message, events = await run(mock_server, openai_model(), ctx(HI), {"signal": controller.signal}, simple=False)
    assert message.stop_reason == "aborted"
    assert message.error_message == "Request aborted"
    assert [event.type for event in events] == ["error"]
    assert mock_server.requests == []


async def test_aborted_mid_stream_keeps_partial_content(mock_server):
    controller = AbortController()
    release = asyncio.Event()

    async def body():
        yield b'data: {"id":"c","choices":[{"delta":{"content":"hello"}}]}\n\n'
        await release.wait()
        yield b"data: [DONE]\n\n"

    mock_server.add(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    s = stream(
        openai_model(),
        ctx(HI),
        {"api_key": "k", "http_client": mock_server.client(), "signal": controller.signal},
    )
    types: list[str] = []
    async for event in s:
        types.append(event.type)
        if event.type == "text_delta":
            controller.abort()
    message = await s.result()
    assert message.stop_reason == "aborted"
    assert message.error_message == "Request was aborted"
    assert message.content == [TextContent("hello")]
    assert types == ["start", "text_start", "text_delta", "text_end", "error"]


# ---------------------------------------------------------------------------
# OpenAI SDK transport
# ---------------------------------------------------------------------------


async def test_status_error_keeps_the_parsed_error_body(mock_server):
    mock_server.add_json(
        {"error": {"message": "Invalid value for 'model'", "type": "invalid_request_error"}}, status=400
    )
    message, _ = await run(mock_server, openai_model(), ctx(HI), simple=False)
    assert message.stop_reason == "error"
    # JS SDK APIError: status 400, error = body.error, formatted by normalizeProviderError.
    assert message.error_message == '400: {"message":"Invalid value for \'model\'","type":"invalid_request_error"}'


async def test_status_error_without_json_body(mock_server):
    mock_server.add(httpx.Response(502, text="Bad Gateway"))
    message, _ = await run(mock_server, openai_model(), ctx(HI), simple=False)
    assert message.error_message == "502 Bad Gateway"


@pytest.mark.parametrize(
    ("error", "expected"),
    [(httpx.ConnectError("refused"), "Connection error."), (httpx.ReadTimeout("slow"), "Request timed out.")],
)
async def test_transport_errors_use_sdk_messages(mock_server, error, expected):
    def fail(_request: httpx.Request) -> httpx.Response:
        raise error

    mock_server.add(fail)
    message, _ = await run(mock_server, openai_model(), ctx(HI), simple=False)
    assert message.stop_reason == "error"
    assert message.error_message == expected


async def test_retries_are_fresh_sdk_requests(mock_server):
    mock_server.add_json({"error": {"message": "rate limited"}}, status=429, headers={"retry-after-ms": "1"})
    mock_server.add_sse(OK_CHUNKS)
    message, _ = await run(
        mock_server, RETRY_MODEL, ctx(UserMessage("hi")), {"max_retries": 1, "max_retry_delay_ms": 100}, simple=False
    )
    assert message.stop_reason == "stop"
    # pi retries around the SDK (max_retries=0), so every attempt is a first SDK attempt.
    assert [request.headers["x-stainless-retry-count"] for request in mock_server.requests] == ["0", "0"]


async def test_on_payload_body_is_sent_unchanged(mock_server):
    replacement = {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hi", "custom_part": True}],
        "stream": True,
        "provider_specific": {"nested": [1, None]},
    }
    message, _ = await run(
        mock_server, openai_model(), ctx(HI), {"on_payload": lambda _params, _model: replacement}, simple=False
    )
    assert message.stop_reason == "stop"
    assert mock_server.last.json() == replacement


async def test_uses_pooled_http_client_without_http_client_option(mock_server, monkeypatch):
    from pi_ai.api import openai_sdk

    seen: list[str] = []
    client = mock_server.client()

    def pooled(url: str = "", options: Any = None) -> httpx.AsyncClient:
        seen.append(url)
        return client

    monkeypatch.setattr(openai_sdk, "get_http_client", pooled)
    mock_server.add_sse([STOP_CHUNK, "[DONE]"])
    s = stream(openai_model(), ctx(HI), {"api_key": "k"})
    message = await s.result()
    assert message.stop_reason == "stop"
    assert seen == ["https://api.openai.com/v1"]
    assert not client.is_closed
