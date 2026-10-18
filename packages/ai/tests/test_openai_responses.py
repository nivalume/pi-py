"""Offline tests for the OpenAI Responses API family (shared conversion/stream processing and openai-responses).

Ports pi's ``openai-responses-*.test.ts``, ``xai-responses.test.ts``, ``cache-retention.test.ts``
(Responses parts), ``responseid.test.ts`` (offline) and ``openai-responses-chatgpt-sign-in.test.ts``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest

from pi_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    Context,
    ImageContent,
    Model,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Type,
    Usage,
    UserMessage,
    normalize_context,
)
from pi_ai.api import openai_responses
from pi_ai.api.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.utils.misc import get_pi_user_agent, short_hash

OPENAI_PROVIDERS = {"openai", "openai-codex", "opencode"}


def make_model(**overrides: Any) -> Model:
    fields: dict[str, Any] = {
        "id": "gpt-5-mini",
        "name": "GPT-5 Mini",
        "api": "openai-responses",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "reasoning": True,
        "input": ["text"],
        "context_window": 400000,
        "max_tokens": 128000,
    }
    fields.update(overrides)
    return Model(**fields)


def builtin(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"{provider}/{model_id} missing from catalog"
    return model


def make_output(model: Model) -> AssistantMessage:
    return AssistantMessage([], api=model.api, provider=model.provider, model=model.id, stop_reason="pending")


async def agen(events: list[Any]):
    for event in events:
        yield event


class RecordingStream(AssistantMessageEventStream):
    def __init__(self) -> None:
        super().__init__()
        self.pushed: list[Any] = []
        self.stop_reasons: list[str] = []

    def push(self, event: Any) -> None:
        self.pushed.append(event)
        partial = getattr(event, "partial", None)
        if partial is not None:
            self.stop_reasons.append(partial.stop_reason)
        super().push(event)


def simple_context(system_prompt: str | None = "sys", text: str = "hi", **kwargs: Any):
    return normalize_context(Context(system_prompt=system_prompt, messages=[UserMessage(text, timestamp=0)], **kwargs))


def completed_event(**response: Any) -> dict[str, Any]:
    return {"type": "response.completed", "response": {"id": "resp_test", "status": "completed", **response}}


async def run_stream(
    mock_server: Any,
    model: Model,
    context: Any,
    options: dict[str, Any] | None = None,
    events: list[Any] | None = None,
) -> tuple[AssistantMessage, dict[str, Any] | None]:
    captured: dict[str, Any] = {}

    def on_payload(payload: Any, _model: Any) -> None:
        captured["payload"] = payload

    mock_server.add_sse(events if events is not None else [completed_event()])
    opts = {"api_key": "sk-test-key", "http_client": mock_server.client(), "on_payload": on_payload, **(options or {})}
    message = await openai_responses.stream(model, context, opts).result()
    return message, captured.get("payload")


# ---------------------------------------------------------------------------
# convert_responses_messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "\r\n", " \n\t"])
def test_empty_tool_result_uses_placeholder(text: str) -> None:
    model = builtin("openai", "gpt-4o-mini")
    assistant = AssistantMessage(
        [ToolCall("tool-1", "bash", {"command": "true"})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="toolUse",
    )
    context = normalize_context(
        Context(
            messages=[
                UserMessage("Run the command"),
                assistant,
                ToolResultMessage("tool-1", "bash", [TextContent(text)], False),
            ]
        )
    )
    items = convert_responses_messages(model, context, OPENAI_PROVIDERS)
    output = next(item for item in items if item.get("type") == "function_call_output")
    assert output["output"] == "(no tool output)"


def test_tool_result_images_stay_in_function_call_output() -> None:
    model = make_model(input=["text", "image"])
    assistant = AssistantMessage(
        [ToolCall("call_1|fc_1", "get_image", {})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="toolUse",
    )
    context = normalize_context(
        Context(
            messages=[
                UserMessage("go"),
                assistant,
                ToolResultMessage(
                    "call_1|fc_1",
                    "get_image",
                    [TextContent("A red circle"), ImageContent("AAAA", "image/png")],
                    False,
                ),
            ]
        )
    )
    items = convert_responses_messages(model, context, OPENAI_PROVIDERS)
    index = next(i for i, item in enumerate(items) if item.get("type") == "function_call_output")
    output = items[index]["output"]
    assert output == [
        {"type": "input_text", "text": "A red circle"},
        {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,AAAA"},
    ]
    assert not [item for item in items[index + 1 :] if item.get("role") == "user"]

    text_only = convert_responses_messages(make_model(), context, OPENAI_PROVIDERS)
    text_output = next(item for item in text_only if item.get("type") == "function_call_output")["output"]
    assert text_output == "A red circle\n(tool image omitted: model does not support images)"


COPILOT_RAW_TOOL_CALL_ID = (
    "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6S"
    "LUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6"
    "TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTy"
    "Eei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA=="
)


def test_foreign_tool_call_ids_are_hashed() -> None:
    model = builtin("openai-codex", "gpt-5.5")
    assistant = AssistantMessage(
        [ToolCall(COPILOT_RAW_TOOL_CALL_ID, "edit", {"path": "src/styles/app.css"})],
        api="openai-responses",
        provider="github-copilot",
        model="gpt-5.5",
        stop_reason="toolUse",
    )
    context = normalize_context(
        Context(
            system_prompt="You are concise.",
            messages=[
                UserMessage("Use the tool."),
                assistant,
                ToolResultMessage(COPILOT_RAW_TOOL_CALL_ID, "edit", [TextContent("ok")], False),
            ],
        )
    )
    items = convert_responses_messages(model, context, OPENAI_PROVIDERS)
    function_call = next(item for item in items if item.get("type") == "function_call")
    expected = f"fc_{short_hash(COPILOT_RAW_TOOL_CALL_ID.split('|')[1])}"
    assert function_call["id"] == expected
    assert len(function_call["id"]) <= 64
    assert function_call["id"].replace("fc_", "", 1).isalnum()
    output = next(item for item in items if item.get("type") == "function_call_output")
    assert output["call_id"] == function_call["call_id"]


def test_fallback_message_ids_are_unique_per_text_block() -> None:
    model = builtin("openai-codex", "gpt-5.5")
    assistant = AssistantMessage(
        [TextContent("first answer"), TextContent("visible answer")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-8",
    )
    context = normalize_context(Context(system_prompt="You are concise.", messages=[UserMessage("hello"), assistant]))
    items = convert_responses_messages(model, context, OPENAI_PROVIDERS)
    ids = [item["id"] for item in items if item.get("type") == "message"]
    assert ids == ["msg_pi_1", "msg_pi_1_1"]


def test_text_signatures_replay_ids_and_phases() -> None:
    model = make_model()
    assistant = AssistantMessage(
        [
            TextContent("a", text_signature=json.dumps({"v": 1, "id": "msg_a", "phase": "commentary"})),
            TextContent("b", text_signature="msg_legacy"),
            TextContent("c", text_signature="x" * 80),
        ],
        api=model.api,
        provider=model.provider,
        model=model.id,
    )
    items = convert_responses_messages(model, normalize_context(Context(messages=[assistant])), OPENAI_PROVIDERS)
    assert [(item["id"], item.get("phase")) for item in items] == [
        ("msg_a", "commentary"),
        ("msg_legacy", None),
        (f"msg_{short_hash('x' * 80)}", None),
    ]
    assert "phase" not in items[1]


def test_system_prompt_role_and_omission() -> None:
    context = simple_context("Be careful.")
    reasoning = convert_responses_messages(make_model(), context, OPENAI_PROVIDERS)
    assert reasoning[0] == {"role": "developer", "content": "Be careful."}
    plain = convert_responses_messages(make_model(reasoning=False), context, OPENAI_PROVIDERS)
    assert plain[0]["role"] == "system"
    no_dev = convert_responses_messages(make_model(compat={"supportsDeveloperRole": False}), context, OPENAI_PROVIDERS)
    assert no_dev[0]["role"] == "system"
    without = convert_responses_messages(make_model(), context, OPENAI_PROVIDERS, {"include_system_prompt": False})
    assert without == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


def test_tool_additions_as_additional_tools_and_tool_search() -> None:
    tool_a = Tool("a", "A", Type.Object({"x": Type.String()}))
    tool_b = Tool("b", "B", Type.Object({"y": Type.String()}))
    context = normalize_context(
        Context(
            system_prompt="sys",
            tools=[tool_a],
            messages=[UserMessage("one"), SystemMessage("", tools_added=[tool_b]), UserMessage("two")],
        )
    )
    model = make_model()
    additional = convert_responses_messages(
        model,
        context,
        OPENAI_PROVIDERS,
        {"supports_mid_convo_system_messages": True, "supports_additional_tools": True},
    )
    item = next(item for item in additional if item.get("type") == "additional_tools")
    assert item["role"] == "developer"
    assert [tool["name"] for tool in item["tools"]] == ["b"]

    search = convert_responses_messages(
        model, context, OPENAI_PROVIDERS, {"supports_mid_convo_system_messages": True, "supports_tool_search": True}
    )
    call = next(item for item in search if item.get("type") == "tool_search_call")
    output = next(item for item in search if item.get("type") == "tool_search_output")
    assert call["call_id"] == output["call_id"] == f"pi_tool_load_{short_hash('system:1:b')}"
    assert call["arguments"] == {"query": "b", "limit": 1}
    assert output["tools"][0]["defer_loading"] is True


def test_convert_responses_tools_strict_and_grammar() -> None:
    tools = [
        Tool("ordinary", "O", Type.Object({"path": Type.String(), "offset": Type.Optional(Type.Number())})),
        Tool(
            "constrained",
            "C",
            Type.Object({"value": Type.String()}),
            constrained_sampling={"type": "json_schema", "strict": "prefer"},
        ),
        Tool(
            "grammar",
            "G",
            Type.Object({"payload": Type.String()}),
            constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /[a-z]+/"}},
        ),
    ]
    converted = convert_responses_tools(tools, {"supports_strict_mode": True, "supports_openai_grammar_tools": True})
    assert [tool.get("strict") for tool in converted[:2]] == [False, True]
    assert converted[1]["parameters"]["additionalProperties"] is False
    assert converted[2] == {
        "type": "custom",
        "name": "grammar",
        "description": "G",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[a-z]+/"},
    }
    nulls = convert_responses_tools(tools[:1], {"strict": None})
    assert "strict" in nulls[0] and nulls[0]["strict"] is None
    no_strict = convert_responses_tools(tools[:2], {"supports_strict_mode": False})
    assert all("strict" not in tool for tool in no_strict)


# ---------------------------------------------------------------------------
# process_responses_stream
# ---------------------------------------------------------------------------


def function_call_events(arguments_json: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "delta": '{"path":"README.md"'},
        {"type": "response.function_call_arguments.delta", "delta": ',"content":"updated"}'},
        {"type": "response.function_call_arguments.done", "arguments": arguments_json},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "edit",
                "arguments": arguments_json,
            },
        },
        completed_event(),
    ]


async def test_tool_call_streaming_finalizes_arguments() -> None:
    model = make_model()
    output = make_output(model)
    stream = RecordingStream()
    args = '{"path":"README.md","content":"updated"}'
    await process_responses_stream(agen(function_call_events(args)), output, stream, model)
    assert len(output.content) == 1
    tool_call = output.content[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.arguments == {"path": "README.md", "content": "updated"}
    assert tool_call.id == "call_test|fc_test"
    end = next(event for event in stream.pushed if event.type == "toolcall_end")
    assert end.tool_call is tool_call
    deltas = [event.delta for event in stream.pushed if event.type == "toolcall_delta"]
    assert "".join(deltas) == args
    assert output.stop_reason == "toolUse"
    assert output.error_message is None


async def test_malformed_final_arguments_are_marked() -> None:
    model = make_model()
    output = make_output(model)
    args = '{"path":"README.md","content":"updated","mode":.5}'
    await process_responses_stream(agen(function_call_events(args)), output, RecordingStream(), model)
    tool_call = output.content[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.arguments == {}
    assert tool_call.arguments_error is not None
    assert tool_call.arguments_error.endswith(f"Received arguments:\n{args}")


async def test_namespace_round_trips_from_output_item_done() -> None:
    model = make_model(id="gpt-5.4", name="GPT-5.4")
    output = make_output(model)
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": "",
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": '{"value":"hello"}',
                "namespace": "dynamic_tools",
            },
        },
        completed_event(),
    ]
    await process_responses_stream(agen(events), output, RecordingStream(), model)
    tool_call = output.content[0]
    assert isinstance(tool_call, ToolCall)
    assert (tool_call.id, tool_call.arguments, tool_call.namespace) == (
        "call_test|fc_test",
        {"value": "hello"},
        "dynamic_tools",
    )
    assert output.error_message is None
    replayed = next(
        item
        for item in convert_responses_messages(model, normalize_context(Context(messages=[output])), {"openai"})
        if item.get("type") == "function_call"
    )
    assert replayed == {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": '{"value":"hello"}',
        "namespace": "dynamic_tools",
    }


async def test_custom_tool_namespace_round_trips() -> None:
    model = make_model(id="gpt-5.4")
    output = make_output(model)
    grammar = {"query": "input"}
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_test",
                "call_id": "call_test",
                "name": "query",
                "input": "",
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_test",
                "call_id": "call_test",
                "name": "query",
                "input": "hello",
                "namespace": "dynamic_tools",
            },
        },
        completed_event(),
    ]
    stream = RecordingStream()
    await process_responses_stream(agen(events), output, stream, model, {"grammar_tool_input_properties": grammar})
    tool_call = output.content[0]
    assert isinstance(tool_call, ToolCall)
    assert (tool_call.arguments, tool_call.namespace) == ({"input": "hello"}, "dynamic_tools")
    assert [event.delta for event in stream.pushed if event.type == "toolcall_delta"] == ['{"input":"hello"}']
    replayed = next(
        item
        for item in convert_responses_messages(
            model,
            normalize_context(Context(messages=[output])),
            {"openai"},
            {"grammar_tool_input_properties": grammar},
        )
        if item.get("type") == "custom_tool_call"
    )
    assert replayed == {
        "type": "custom_tool_call",
        "id": "ctc_test",
        "call_id": "call_test",
        "name": "query",
        "input": "hello",
        "namespace": "dynamic_tools",
    }


def test_namespaces_dropped_for_targets_that_cannot_replay_them() -> None:
    model = make_model(id="gpt-5.4")
    output = make_output(model)
    output.content.extend(
        [
            ToolCall("call_function|fc_test", "lookup", {"value": "hello"}, namespace="dynamic_tools"),
            ToolCall("call_custom|ctc_test", "query", {"input": "hello"}, namespace="dynamic_tools"),
        ]
    )
    targets = [
        dataclasses.replace(model, id="gpt-5.2"),
        dataclasses.replace(model, provider="azure"),
        dataclasses.replace(model, api="openai-codex-responses", provider="openai-codex", id="gpt-5.3-codex-spark"),
    ]
    for target in targets:
        items = convert_responses_messages(
            target,
            normalize_context(Context(messages=[output])),
            {"openai"},
            {"grammar_tool_input_properties": {"query": "input"}},
        )
        function_call = next(item for item in items if item.get("type") == "function_call")
        custom_call = next(item for item in items if item.get("type") == "custom_tool_call")
        assert "namespace" not in function_call
        assert "namespace" not in custom_call

    plain = make_output(model)
    plain.content.append(ToolCall("call_test|fc_test", "lookup", {"value": "hello"}))
    item = next(
        item
        for item in convert_responses_messages(model, normalize_context(Context(messages=[plain])), {"openai"})
        if item.get("type") == "function_call"
    )
    assert "namespace" not in item


async def test_rejects_streams_without_terminal_event() -> None:
    model = make_model()
    events = [
        {"type": "response.created", "response": {"id": "resp_early_eof"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs", "summary": []},
        },
        {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "partial"},
    ]
    with pytest.raises(RuntimeError, match="OpenAI Responses stream ended before a terminal response event"):
        await process_responses_stream(agen(events), make_output(model), RecordingStream(), model)


async def test_rejects_unfinished_tool_calls() -> None:
    model = make_model()
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"command":"rm -rf /tmp/build'},
        completed_event(),
    ]
    with pytest.raises(RuntimeError, match=r"unfinished tool call: bash \(call_1\|fc_1\)"):
        await process_responses_stream(agen(events), make_output(model), RecordingStream(), model)


async def test_rejects_parallel_tool_calls_without_output_index() -> None:
    # llama.cpp omits output_index from every event and sends both done events after all deltas.
    model = make_model()

    def call(n: str, arguments: str) -> dict[str, Any]:
        return {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": f"call_{n}",
            "name": "bash",
            "arguments": arguments,
        }

    events = [
        {"type": "response.output_item.added", "item": call("a", "")},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": '{"command":"echo a"}'},
        {"type": "response.output_item.added", "item": call("b", "")},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_b", "delta": '{"command":"echo b"}'},
        {"type": "response.output_item.done", "item": call("a", '{"command":"echo a"}')},
        {"type": "response.output_item.done", "item": call("b", '{"command":"echo b"}')},
        {"type": "response.completed", "response": {"id": "resp", "status": "completed"}},
    ]
    with pytest.raises(RuntimeError, match=r"unfinished tool call: bash \(call_a\|fc_a\)"):
        await process_responses_stream(agen(events), make_output(model), RecordingStream(), model)


def phased_events(phases: tuple[str, str], terminal: str = "completed") -> list[dict[str, Any]]:
    message = {"type": "message", "id": "msg_phase", "role": "assistant"}
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": [], "phase": phases[0]},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                **message,
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                "phase": phases[1],
            },
        },
    ]
    if terminal == "incomplete":
        events.append(
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_phase",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        )
    else:
        events.append({"type": "response.completed", "response": {"id": "resp_phase", "status": "completed"}})
    return events


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (("commentary", "commentary"), ["pending", "pending"]),
        (("final_answer", "final_answer"), ["stop", "stop"]),
        (("commentary", "final_answer"), ["pending", "stop"]),
    ],
)
async def test_tracks_message_phases(phases: tuple[str, str], expected: list[str]) -> None:
    model = make_model()
    output = make_output(model)
    stream = RecordingStream()
    await process_responses_stream(agen(phased_events(phases)), output, stream, model)
    assert stream.stop_reasons == expected
    assert output.stop_reason == "stop"
    text = output.content[0]
    assert isinstance(text, TextContent)
    assert text.text == "answer"
    assert text.text_signature == json.dumps({"v": 1, "id": "msg_phase", "phase": phases[1]}, separators=(",", ":"))


async def test_incomplete_terminal_replaces_provisional_stop() -> None:
    model = make_model()
    output = make_output(model)
    stream = RecordingStream()
    await process_responses_stream(
        agen(phased_events(("final_answer", "final_answer"), "incomplete")), output, stream, model
    )
    assert stream.stop_reasons == ["stop", "stop"]
    assert output.stop_reason == "length"


async def test_completed_terminal_usage() -> None:
    model = make_model(cost={"input": 0, "output": 0})  # type: ignore[arg-type]
    model = make_model()
    output = make_output(model)
    events = [
        completed_event(
            id="resp_completed",
            usage={
                "input_tokens": 20,
                "output_tokens": 7,
                "total_tokens": 27,
                "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 4},
            },
        )
    ]
    await process_responses_stream(agen(events), output, RecordingStream(), model)
    assert output.response_id == "resp_completed"
    assert output.stop_reason == "stop"
    assert output.raw_stop_reason == "completed"
    usage = output.usage
    assert (usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens, usage.reasoning) == (
        15,
        7,
        2,
        3,
        27,
        4,
    )


@pytest.mark.parametrize(
    ("reason", "stop_reason", "error_message"),
    [
        ("max_output_tokens", "length", None),
        ("content_filter", "error", "Response incomplete: content_filter"),
        ("max_time_limit", "error", "Response incomplete: max_time_limit"),
    ],
)
async def test_incomplete_terminal_events(reason: str, stop_reason: str, error_message: str | None) -> None:
    model = make_model()
    output = make_output(model)
    events = [
        {
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": reason},
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 12,
                    "total_tokens": 42,
                    "input_tokens_details": {"cached_tokens": 5},
                },
            },
        }
    ]
    await process_responses_stream(agen(events), output, RecordingStream(), model)
    assert output.stop_reason == stop_reason
    assert output.raw_stop_reason == f"incomplete.{reason}"
    assert output.error_message == error_message
    assert (output.usage.input, output.usage.cache_read) == (25, 5)


async def test_failed_terminal_event_raises_provider_error() -> None:
    model = make_model()
    output = make_output(model)
    events = [
        {
            "type": "response.failed",
            "response": {"id": "resp_failed", "status": "failed", "error": {"code": "server_error", "message": "boom"}},
        }
    ]
    with pytest.raises(RuntimeError, match="server_error: boom"):
        await process_responses_stream(agen(events), output, RecordingStream(), model)
    assert output.raw_stop_reason == "failed"


async def test_error_event_raises() -> None:
    model = make_model()
    with pytest.raises(RuntimeError, match="Error Code bad: nope"):
        await process_responses_stream(
            agen([{"type": "error", "code": "bad", "message": "nope"}]), make_output(model), RecordingStream(), model
        )


async def test_reasoning_summary_and_signature() -> None:
    model = make_model()
    output = make_output(model)
    item = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Thought"}]}
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        },
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Thou"},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "ght"},
        {"type": "response.reasoning_summary_part.done", "output_index": 0},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        completed_event(),
    ]
    stream = RecordingStream()
    await process_responses_stream(agen(events), output, stream, model)
    block = output.content[0]
    assert block.type == "thinking"
    assert block.thinking == "Thought"
    assert json.loads(block.thinking_signature or "") == item
    assert [e.type for e in stream.pushed] == [
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
    ]


def azure_model() -> Model:
    return make_model(api="azure-openai-responses", provider="azure", base_url="https://example.invalid")


async def run_reasoning_replay(done_item: dict[str, Any], completed_item: dict[str, Any]) -> dict[str, Any]:
    model = azure_model()
    output = make_output(model)
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": done_item["id"], "summary": []},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": done_item},
        {
            "type": "response.completed",
            "response": {"id": "resp_test", "status": "completed", "output": [completed_item]},
        },
    ]
    await process_responses_stream(agen(events), output, RecordingStream(), model)
    context = normalize_context(Context(messages=[UserMessage("first"), output, UserMessage("follow-up")]))
    items = convert_responses_messages(model, context, {"azure"})
    return next(item for item in items if item.get("type") == "reasoning")


async def test_reasoning_replay_preserves_existing_encrypted_content() -> None:
    done = {"type": "reasoning", "id": "rs_done", "summary": [], "encrypted_content": "from-output-item-done"}
    replayed = await run_reasoning_replay(done, {**done, "encrypted_content": "from-response-completed"})
    assert replayed["encrypted_content"] == "from-output-item-done"


async def test_reasoning_replay_backfills_encrypted_content() -> None:
    done = {"type": "reasoning", "id": "rs_missing", "summary": []}
    replayed = await run_reasoning_replay(done, {**done, "encrypted_content": "from-response-completed"})
    assert replayed == {
        "type": "reasoning",
        "id": "rs_missing",
        "summary": [],
        "encrypted_content": "from-response-completed",
    }


# ---------------------------------------------------------------------------
# openai-responses stream: payload and headers
# ---------------------------------------------------------------------------


async def test_omits_reasoning_for_copilot_without_reasoning(mock_server: Any) -> None:
    _, payload = await run_stream(mock_server, builtin("github-copilot", "gpt-5-mini"), simple_context())
    assert payload is not None
    assert "reasoning" not in payload


async def test_forwards_required_tool_choice(mock_server: Any) -> None:
    context = simple_context(tools=[Tool("ping", "Ping", Type.Object({"value": Type.String()}))])
    _, payload = await run_stream(mock_server, builtin("openai", "gpt-5.4"), context, {"tool_choice": "required"})
    assert payload is not None
    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"]] == ["ping"]


async def test_cloudflare_strict_mode_is_explicit(mock_server: Any) -> None:
    model = builtin("cloudflare-ai-gateway", "gpt-5.6-sol")
    context = simple_context(
        tools=[
            Tool(
                "ordinary",
                "An ordinary tool",
                Type.Object({"path": Type.String(), "offset": Type.Optional(Type.Number())}),
            ),
            Tool(
                "constrained",
                "A constrained tool",
                Type.Object({"value": Type.String()}),
                constrained_sampling={"type": "json_schema", "strict": "prefer"},
            ),
        ]
    )
    _, payload = await run_stream(mock_server, model, context)
    assert (model.compat or {}).get("supportsStrictMode") is True
    assert payload is not None
    assert [(tool["name"], tool["strict"]) for tool in payload["tools"]] == [("ordinary", False), ("constrained", True)]


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.1", "gpt-5.2", "gpt-5.3-codex", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-6-luna"],
)
async def test_sends_none_reasoning_effort_when_off(mock_server: Any, model_id: str) -> None:
    _, payload = await run_stream(mock_server, builtin("openai", model_id), simple_context())
    assert payload is not None
    assert payload["reasoning"] == {"effort": "none"}


@pytest.mark.parametrize("model_id", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"])
async def test_omits_reasoning_when_off_unsupported(mock_server: Any, model_id: str) -> None:
    _, payload = await run_stream(mock_server, builtin("openai", model_id), simple_context())
    assert payload is not None
    assert "reasoning" not in payload


async def test_request_wire_format(mock_server: Any) -> None:
    model = builtin("openai", "gpt-5.4")
    message, payload = await run_stream(
        mock_server,
        model,
        simple_context(),
        {"reasoning_effort": "high", "max_tokens": 4, "temperature": 0.2, "session_id": "s1"},
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            completed_event(id="resp_1"),
            "[DONE]",
        ],
    )
    request = mock_server.last
    assert request.url == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer sk-test-key"
    assert request.headers["user-agent"] == get_pi_user_agent()
    assert request.headers["accept"] == "application/json"
    assert request.headers["content-type"] == "application/json"
    body = request.json()
    assert body == payload
    assert body["model"] == "gpt-5.4"
    assert body["stream"] is True and body["store"] is False
    assert body["max_output_tokens"] == 16
    assert body["temperature"] == 0.2
    assert body["prompt_cache_key"] == "s1"
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert message.stop_reason == "stop"
    assert message.response_id == "resp_1"


async def capture_headers(
    mock_server: Any, options: dict[str, Any], model: Model | None = None
) -> tuple[httpx.Headers, Any]:
    _, payload = await run_stream(mock_server, model or builtin("openai", "gpt-5.4"), simple_context(), options)
    return mock_server.last.headers, payload


async def test_cache_affinity_headers_for_openai(mock_server: Any) -> None:
    headers, _ = await capture_headers(mock_server, {"session_id": "session-123"})
    assert headers.get("session_id") == "session-123"
    assert headers.get("x-client-request-id") == "session-123"


async def test_prompt_cache_key_is_clamped(mock_server: Any) -> None:
    _, payload = await capture_headers(mock_server, {"session_id": "x" * 67})
    assert payload["prompt_cache_key"] == "x" * 64


async def test_cache_affinity_headers_for_proxy(mock_server: Any) -> None:
    model = dataclasses.replace(
        builtin("openai", "gpt-5.4"), provider="opencode", base_url="https://proxy.example.com/v1"
    )
    headers, _ = await capture_headers(mock_server, {"session_id": "session-123"}, model)
    assert headers.get("session_id") == "session-123"
    assert headers.get("x-client-request-id") == "session-123"


@pytest.mark.parametrize(
    ("provider", "base_url", "compat"),
    [
        ("proxy", "https://proxy.example.com/v1", {"sessionAffinityFormat": "openrouter"}),
        ("openrouter", "https://openrouter.ai/api/v1", None),
    ],
)
async def test_openrouter_session_affinity(mock_server: Any, provider: str, base_url: str, compat: Any) -> None:
    model = dataclasses.replace(builtin("openai", "gpt-5.4"), provider=provider, base_url=base_url, compat=compat)
    headers, payload = await capture_headers(mock_server, {"session_id": "session-x"}, model)
    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") is None
    assert headers.get("x-session-id") == "session-x"
    assert "session_id" not in payload
    assert payload["prompt_cache_key"] == "session-x"


async def test_openai_nosession_affinity(mock_server: Any) -> None:
    model = dataclasses.replace(
        builtin("openai", "gpt-5.4"),
        provider="proxy",
        base_url="https://proxy.example.com/v1",
        compat={"sessionAffinityFormat": "openai-nosession"},
    )
    headers, payload = await capture_headers(mock_server, {"session_id": "session-proxy"}, model)
    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") == "session-proxy"
    assert headers.get("x-session-id") is None
    assert payload["prompt_cache_key"] == "session-proxy"


async def test_opencode_uses_nosession_format(mock_server: Any) -> None:
    model = builtin("opencode", "gpt-5.4")
    assert (model.compat or {}).get("sessionAffinityFormat") == "openai-nosession"
    headers, payload = await capture_headers(mock_server, {"session_id": "session-opencode"}, model)
    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") == "session-opencode"
    assert payload["prompt_cache_key"] == "session-opencode"


async def test_explicit_headers_override_affinity_headers(mock_server: Any) -> None:
    headers, _ = await capture_headers(
        mock_server,
        {
            "session_id": "session-123",
            "headers": {"session_id": "override-session", "x-client-request-id": "override-request"},
        },
    )
    assert headers.get("session_id") == "override-session"
    assert headers.get("x-client-request-id") == "override-request"


async def test_no_affinity_headers_with_cache_retention_none(mock_server: Any) -> None:
    headers, _ = await capture_headers(mock_server, {"cache_retention": "none", "session_id": "session-123"})
    assert headers.get("session_id") is None
    assert headers.get("x-client-request-id") is None


async def test_none_header_suppresses_default(mock_server: Any) -> None:
    headers, _ = await capture_headers(mock_server, {"headers": {"User-Agent": None, "X-Extra": "1"}})
    assert "user-agent" not in headers or headers["user-agent"] != get_pi_user_agent()
    assert headers["x-extra"] == "1"


async def test_copilot_dynamic_headers(mock_server: Any) -> None:
    headers, _ = await capture_headers(mock_server, {}, builtin("github-copilot", "gpt-5-mini"))
    assert headers["x-initiator"] == "user"
    assert headers["openai-intent"] == "conversation-edits"


@pytest.mark.parametrize(
    ("model_id", "service_tier", "response_tier", "multiplier"),
    [
        ("gpt-5.4", "priority", "priority", 2),
        ("gpt-5.5", "priority", "priority", 2.5),
        ("gpt-5.5", "flex", "flex", 0.5),
        # GPT-6 models report Fast mode as "fast" even when "priority" is requested (#10034)
        ("gpt-6-luna", "priority", "fast", 2),
        ("gpt-6-luna", "fast", "fast", 2),
    ],
)
async def test_service_tier_cost_multiplier(
    mock_server: Any, model_id: str, service_tier: str, response_tier: str, multiplier: float
) -> None:
    model = builtin("openai", model_id)
    tokens = 100_000
    scale = tokens / 1_000_000
    message, _ = await run_stream(
        mock_server,
        model,
        simple_context(),
        {"service_tier": service_tier},
        [
            completed_event(
                service_tier=response_tier,
                usage={
                    "input_tokens": tokens,
                    "output_tokens": tokens,
                    "total_tokens": tokens * 2,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            )
        ],
    )
    assert message.usage.cost.input == pytest.approx(model.cost.input * multiplier * scale)
    assert message.usage.cost.output == pytest.approx(model.cost.output * multiplier * scale)
    assert message.usage.cost.total == pytest.approx((model.cost.input + model.cost.output) * multiplier * scale)


async def test_max_output_tokens_compat(mock_server: Any) -> None:
    _, payload = await run_stream(mock_server, builtin("openai", "gpt-5.4"), simple_context(), {"max_tokens": 1024})
    assert payload is not None and payload["max_output_tokens"] == 1024
    base = builtin("openai", "gpt-5.4")
    model = dataclasses.replace(base, compat={**(base.compat or {}), "supportsMaxOutputTokens": False})
    _, payload = await run_stream(mock_server, model, simple_context(), {"max_tokens": 1024})
    assert payload is not None and "max_output_tokens" not in payload


async def test_on_provider_stream_event_receives_raw_events_in_order(mock_server: Any) -> None:
    model = make_model()
    received: list[Any] = []
    models: list[Any] = []

    async def on_event(event: Any, event_model: Any) -> None:
        received.append(event)
        models.append(event_model)

    events = [
        {"type": "response.created", "response": {"id": "resp_wrapper_early_eof"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs", "summary": []},
        },
        {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "partial reasoning"},
    ]
    mock_server.add_sse(events)
    stream = openai_responses.stream(
        model,
        simple_context(""),
        {"api_key": "test", "http_client": mock_server.client(), "on_provider_stream_event": on_event},
    )
    seen = [event async for event in stream]
    result = await stream.result()
    assert [event["type"] for event in received] == [e["type"] for e in events]
    assert models == [model, model, model]
    assert seen[0].type == "start"
    assert seen[-1].type == "error"
    assert result.stop_reason == "error"
    assert result.error_message == "OpenAI Responses stream ended before a terminal response event"


async def test_http_error_is_formatted_with_prefix(mock_server: Any) -> None:
    mock_server.add_json({"error": {"message": "Invalid model", "type": "invalid_request_error"}}, status=400)
    message = await openai_responses.stream(
        make_model(), simple_context(), {"api_key": "sk-x", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message is not None
    assert message.error_message.startswith("OpenAI API error (400)")
    assert "Invalid model" in message.error_message


async def test_missing_api_key() -> None:
    model = make_model()
    message = await openai_responses.stream(model, simple_context(), {}).result()
    assert message.stop_reason == "error"
    assert message.error_message == "No API key for provider: openai"
    with pytest.raises(RuntimeError, match="No API key for provider: openai"):
        openai_responses.stream_simple(model, simple_context(), {})


async def test_authorization_header_allows_missing_api_key(mock_server: Any) -> None:
    mock_server.add_sse([completed_event()])
    message = await openai_responses.stream_simple(
        make_model(),
        simple_context(),
        {"headers": {"Authorization": "Bearer custom"}, "http_client": mock_server.client()},
    ).result()
    assert message.stop_reason == "stop"
    assert mock_server.last.headers["authorization"] == "Bearer custom"


async def test_stream_simple_maps_reasoning(mock_server: Any) -> None:
    mock_server.add_sse([completed_event()])
    model = builtin("openai", "gpt-5.4")
    await openai_responses.stream_simple(
        model, simple_context(), {"api_key": "sk-x", "http_client": mock_server.client(), "reasoning": "minimal"}
    ).result()
    body = mock_server.last.json()
    # gpt-5.4 maps minimal to unsupported, so it clamps to low.
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["max_output_tokens"] > 0


# ---------------------------------------------------------------------------
# Usage limit and Sign in with ChatGPT
# ---------------------------------------------------------------------------

USAGE_LIMIT_ERROR = {"code": "subscription_sharing_usage_limit_exceeded", "message": "Usage limit reached."}


async def test_usage_limit_rejected_request_links_to_chatgpt(mock_server: Any) -> None:
    mock_server.add_json({"error": {**USAGE_LIMIT_ERROR, "type": "rate_limit_error"}}, status=429)
    result = await openai_responses.stream(
        make_model(), simple_context(""), {"api_key": "test", "http_client": mock_server.client()}
    ).result()
    assert result.stop_reason == "error"
    assert "subscription_sharing_usage_limit_exceeded" in (result.error_message or "")
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in (result.error_message or "")


async def test_usage_limit_failed_stream_links_to_chatgpt(mock_server: Any) -> None:
    event = {
        "type": "response.failed",
        "response": {"id": "resp_failed", "status": "failed", "error": USAGE_LIMIT_ERROR},
    }
    mock_server.add_sse([("response.failed", event)])
    result = await openai_responses.stream(
        make_model(), simple_context(""), {"api_key": "test", "http_client": mock_server.client()}
    ).result()
    assert result.stop_reason == "error"
    assert "subscription_sharing_usage_limit_exceeded: Usage limit reached." in (result.error_message or "")
    assert "Check your ChatGPT usage" in (result.error_message or "")


async def capture_sign_in_payload(mock_server: Any, api_key: str, model: Model | None = None) -> dict[str, Any]:
    mock_server.add(httpx.Response(500))
    captured: dict[str, Any] = {}
    await openai_responses.stream(
        model or make_model(),
        simple_context(""),
        {
            "api_key": api_key,
            "max_tokens": 1000,
            "temperature": 0.5,
            "cache_retention": "long",
            "on_payload": lambda payload, _m: captured.setdefault("payload", payload),
            "http_client": mock_server.client(),
        },
    ).result()
    return captured["payload"]


async def test_chatgpt_sign_in_omits_rejected_fields(mock_server: Any) -> None:
    payload = await capture_sign_in_payload(mock_server, "chatgpt-access-token")
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert "prompt_cache_retention" not in payload


async def test_chatgpt_sign_in_omits_prompt_cache_options(mock_server: Any) -> None:
    model = make_model(compat={"supportsExplicitPromptCacheMode": True})
    sign_in = await capture_sign_in_payload(mock_server, "chatgpt-access-token", model)
    api_key = await capture_sign_in_payload(mock_server, "sk-proj-test", model)
    assert "prompt_cache_options" not in sign_in
    assert api_key["prompt_cache_options"] == {"ttl": "30m"}


@pytest.mark.parametrize(
    ("api_key", "base_url"),
    [("sk-proj-test", "https://api.openai.com/v1"), ("gateway-key", "https://gateway.example.com/v1")],
)
async def test_api_keys_keep_sign_in_fields(mock_server: Any, api_key: str, base_url: str) -> None:
    payload = await capture_sign_in_payload(mock_server, api_key, make_model(base_url=base_url))
    assert payload["max_output_tokens"] == 1000
    assert payload["temperature"] == 0.5
    assert payload["prompt_cache_retention"] == "24h"


# ---------------------------------------------------------------------------
# Mid-conversation effort (configuration_update)
# ---------------------------------------------------------------------------

LEVEL_MAP = {
    "off": "none",
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def luna(compat: dict[str, Any] | None = None) -> Model:
    return make_model(
        id="gpt-6-luna",
        name="GPT-6 Luna",
        thinking_level_map=dict(LEVEL_MAP),
        context_window=272000,
        compat={"supportsMidConvoEffort": True} if compat is None else compat,
    )


def update(effort: str) -> dict[str, Any]:
    return {"type": "configuration_update", "reasoning": {"effort": effort}}


def response_events(response_id: str, with_tool_call: bool) -> list[dict[str, Any]]:
    item = (
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed",
        }
        if with_tool_call
        else {
            "type": "message",
            "id": f"msg_{response_id}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
        }
    )
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
    ]


async def run_luna(
    mock_server: Any,
    messages: list[Any],
    effort: str | None,
    *,
    api_key: str = "sk-test",
    model: Model | None = None,
    with_tool_call: bool = False,
) -> tuple[dict[str, Any], AssistantMessage]:
    mock_server.add_sse(response_events("resp_1", with_tool_call))
    options: dict[str, Any] = {"api_key": api_key, "http_client": mock_server.client()}
    if effort is not None:
        options["reasoning_effort"] = effort
    message = await openai_responses.stream(
        model or luna(), normalize_context(Context(messages=list(messages))), options
    ).result()
    assert message.stop_reason != "error", message.error_message
    return mock_server.last.json(), message


def user(text: str) -> UserMessage:
    return UserMessage(text, timestamp=0)


async def test_configuration_update_keeps_effort_constant(mock_server: Any) -> None:
    history: list[Any] = [user("one")]
    first_payload, first = await run_luna(mock_server, history, "high")
    history += [first, user("two")]
    second_payload, second = await run_luna(mock_server, history, "high")
    history += [second, user("three")]
    third_payload, third = await run_luna(mock_server, history, "low")

    assert first_payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "one"}]},
        update("high"),
    ]
    for payload in (first_payload, second_payload, third_payload):
        assert payload["reasoning"]["effort"] == "medium"
    assert second_payload["input"][: len(first_payload["input"])] == first_payload["input"]
    assert third_payload["input"][: len(second_payload["input"])] == second_payload["input"]
    assert [item for item in third_payload["input"] if item.get("type") == "configuration_update"] == [
        update("high"),
        update("high"),
        update("low"),
    ]
    assert third_payload["input"][-1] == update("low")
    assert [m.provider_thinking_level for m in (first, second, third)] == ["high", "high", "low"]


async def test_configuration_update_after_tool_loop(mock_server: Any) -> None:
    history: list[Any] = [user("one")]
    tool_payload, tool_message = await run_luna(mock_server, history, "high", with_tool_call=True)
    history += [tool_message, ToolResultMessage("call_1|fc_1", "lookup", [TextContent("result")], False, timestamp=0)]
    after_payload, after = await run_luna(mock_server, history, "low")
    assert after_payload["input"][: len(tool_payload["input"])] == tool_payload["input"]
    assert after_payload["input"][-2]["type"] == "function_call_output"
    assert after_payload["input"][-2]["call_id"] == "call_1"
    assert after_payload["input"][-1] == update("low")
    assert after.provider_thinking_level == "low"


async def test_configuration_update_only_for_replies_of_this_model(mock_server: Any) -> None:
    _, unstamped = await run_luna(mock_server, [user("one")], "high", model=luna({}))
    other_provider = unstamped.shallow_copy()
    other_provider.provider = "opencode"
    other_provider.provider_thinking_level = "high"
    other_model = unstamped.shallow_copy()
    other_model.model = "gpt-6-sol"
    other_model.provider_thinking_level = "high"
    recorded = unstamped.shallow_copy()
    recorded.provider_thinking_level = "high"
    payload, _ = await run_luna(
        mock_server,
        [
            user("one"),
            unstamped,
            user("two"),
            other_provider,
            user("three"),
            other_model,
            user("four"),
            recorded,
            user("five"),
        ],
        "low",
    )
    assert [item.get("type") or item.get("role") for item in payload["input"]] == [
        "user",
        "message",
        "user",
        "message",
        "user",
        "message",
        "user",
        "configuration_update",
        "message",
        "user",
        "configuration_update",
    ]


@pytest.mark.parametrize(
    ("api_key", "compat", "level", "effort"),
    [
        ("chatgpt-access-token", None, "low", "low"),
        ("sk-test", {"supportsMidConvoEffort": False}, "low", "low"),
        ("sk-test", None, None, "none"),
    ],
)
async def test_reasoning_effort_without_configuration_update(
    mock_server: Any, api_key: str, compat: Any, level: str | None, effort: str
) -> None:
    model = luna(compat)
    _, first = await run_luna(mock_server, [user("one")], "high", api_key=api_key, model=model)
    payload, message = await run_luna(
        mock_server, [user("one"), first, user("two")], level, api_key=api_key, model=model
    )
    assert payload["reasoning"]["effort"] == effort
    assert not any(item.get("type") == "configuration_update" for item in payload["input"])
    assert message.provider_thinking_level is None


# ---------------------------------------------------------------------------
# Cache retention
# ---------------------------------------------------------------------------


@pytest.fixture
def no_cache_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
    return monkeypatch


async def test_pi_cache_retention_long_for_proxy(mock_server: Any, no_cache_env: pytest.MonkeyPatch) -> None:
    no_cache_env.setenv("PI_CACHE_RETENTION", "long")
    model = dataclasses.replace(builtin("openai", "gpt-4o-mini"), base_url="https://my-proxy.example.com/v1")
    _, payload = await run_stream(
        mock_server, model, simple_context("You are a helpful assistant."), {"api_key": "sk-fake"}
    )
    assert payload is not None and payload["prompt_cache_retention"] == "24h"


async def test_cache_retention_unset_has_no_retention(mock_server: Any, no_cache_env: pytest.MonkeyPatch) -> None:
    _, payload = await run_stream(mock_server, builtin("openai", "gpt-4o-mini"), simple_context())
    assert payload is not None and "prompt_cache_retention" not in payload


async def test_long_retention_respects_compat(mock_server: Any, no_cache_env: pytest.MonkeyPatch) -> None:
    model = dataclasses.replace(builtin("openai", "gpt-4o-mini"), compat={"supportsLongCacheRetention": False})
    _, payload = await run_stream(
        mock_server, model, simple_context(), {"cache_retention": "long", "session_id": "session-compat-false"}
    )
    assert payload is not None and "prompt_cache_retention" not in payload


async def test_cache_retention_none_disables_implicit_writes(
    mock_server: Any, no_cache_env: pytest.MonkeyPatch
) -> None:
    _, payload = await run_stream(
        mock_server, builtin("openai", "gpt-5.6-sol"), simple_context(), {"cache_retention": "none", "session_id": "s"}
    )
    assert payload is not None
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
    assert payload["prompt_cache_options"] == {"mode": "explicit"}

    _, payload = await run_stream(
        mock_server, builtin("openai", "gpt-4o-mini"), simple_context(), {"cache_retention": "none", "session_id": "s"}
    )
    assert payload is not None
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_options" not in payload


@pytest.mark.parametrize(
    ("model_id", "retention", "cache_options"),
    [
        ("gpt-4o-mini", "24h", None),
        ("gpt-6-astra", None, {"ttl": "30m"}),
        ("gpt-6-sol", None, {"ttl": "30m"}),
        ("gpt-6-luna", None, {"ttl": "30m"}),
    ],
)
async def test_long_cache_field_per_model(
    mock_server: Any, no_cache_env: pytest.MonkeyPatch, model_id: str, retention: str | None, cache_options: Any
) -> None:
    _, payload = await run_stream(
        mock_server,
        builtin("openai", model_id),
        simple_context(),
        {"cache_retention": "long", "session_id": "session-2"},
    )
    assert payload is not None
    assert payload["prompt_cache_key"] == "session-2"
    assert payload.get("prompt_cache_retention") == retention
    assert payload.get("prompt_cache_options") == cache_options


@pytest.mark.parametrize("model_id", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra", "gpt-6-luna"])
def test_documented_ttl_does_not_enable_cache_warming(model_id: str) -> None:
    assert builtin("openai", model_id).prompt_cache is None


# ---------------------------------------------------------------------------
# xAI over Responses
# ---------------------------------------------------------------------------


def xai_completed() -> list[Any]:
    return [
        completed_event(
            id="resp_xai_test",
            output=[],
            usage={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
            },
        ),
        "[DONE]",
    ]


async def test_xai_uses_responses_with_bearer_auth(mock_server: Any) -> None:
    model = builtin("xai", "grok-4.5")
    assert model.api == "openai-responses"
    message, _ = await run_stream(
        mock_server,
        model,
        simple_context("You are a careful coding assistant.", "hello"),
        {
            "api_key": "xai-test-token",
            "session_id": "pi-session-123",
            "cache_retention": "long",
            "reasoning_effort": "medium",
        },
        xai_completed(),
    )
    assert message.stop_reason == "stop", message.error_message
    request = mock_server.last
    assert request.url == "https://api.x.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer xai-test-token"
    assert request.headers["user-agent"] == get_pi_user_agent()
    assert request.headers["session_id"] == "pi-session-123"
    body = request.json()
    assert body["model"] == "grok-4.5"
    assert body["store"] is False and body["stream"] is True
    assert body["prompt_cache_key"] == "pi-session-123"
    assert body["reasoning"]["effort"] == "medium"
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "prompt_cache_retention" not in body
    assert {"role": "developer", "content": "You are a careful coding assistant."} in body["input"]


async def test_xai_requests_encrypted_reasoning_without_effort(mock_server: Any) -> None:
    await run_stream(
        mock_server, builtin("xai", "grok-4.5"), simple_context(None, "hello"), {"api_key": "x"}, xai_completed()
    )
    body = mock_server.last.json()
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "reasoning" not in body


@pytest.mark.parametrize(("model_id", "effort"), [("grok-4.7", "xhigh"), ("grok-4.3", "low")])
async def test_xai_reasoning_efforts(mock_server: Any, model_id: str, effort: str) -> None:
    await run_stream(
        mock_server,
        builtin("xai", model_id),
        simple_context(None, "hello"),
        {"api_key": "x", "reasoning_effort": effort},
        xai_completed(),
    )
    body = mock_server.last.json()
    assert mock_server.last.url == "https://api.x.ai/v1/responses"
    assert body["reasoning"]["effort"] == effort
    assert body["include"] == ["reasoning.encrypted_content"]


async def test_xai_explicit_user_agent_override(mock_server: Any) -> None:
    await run_stream(
        mock_server,
        builtin("xai", "grok-4.5"),
        simple_context(None, "hello"),
        {"api_key": "x", "headers": {"User-Agent": "custom-agent"}},
        xai_completed(),
    )
    assert mock_server.last.headers["user-agent"] == "custom-agent"


# ---------------------------------------------------------------------------
# Offline responseId / usage parsing
# ---------------------------------------------------------------------------


async def test_exposes_response_id(mock_server: Any) -> None:
    message, _ = await run_stream(
        mock_server,
        make_model(),
        simple_context(),
        {},
        [
            {"type": "response.created", "response": {"id": "resp_abc"}},
            *response_events("resp_abc", False)[1:],
        ],
    )
    assert message.response_id == "resp_abc"
    assert message.content and isinstance(message.content[0], TextContent)
    assert message.content[0].text == "answer"
    assert isinstance(message.usage, Usage)


async def test_aborted_mid_stream(mock_server: Any) -> None:
    import asyncio

    from pi_ai import AbortController

    controller = AbortController()
    release = asyncio.Event()

    async def body():
        yield b'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
        yield (
            b'data: {"type":"response.output_item.added","output_index":0,'
            b'"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}\n\n'
        )
        await release.wait()

    mock_server.add(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    stream = openai_responses.stream(
        make_model(),
        simple_context(),
        {"api_key": "k", "http_client": mock_server.client(), "signal": controller.signal},
    )
    types: list[str] = []
    async for event in stream:
        types.append(event.type)
        if event.type == "text_start":
            controller.abort()
    message = await stream.result()
    assert message.stop_reason == "aborted"
    assert message.error_message == "Request was aborted"
    assert types[-1] == "error"


async def test_in_stream_error_payload_raises_like_the_sdk(mock_server: Any) -> None:
    mock_server.add_sse([{"error": {"message": "upstream exploded", "code": "server_error"}}])
    message = await openai_responses.stream(
        make_model(), simple_context(), {"api_key": "sk-x", "http_client": mock_server.client()}
    ).result()
    assert message.stop_reason == "error"
    assert message.error_message == "upstream exploded"


async def test_status_error_retries_through_sdk(mock_server: Any) -> None:
    mock_server.add_json({"error": {"message": "busy"}}, status=503, headers={"retry-after-ms": "1"})
    mock_server.add_sse([completed_event()])
    message = await openai_responses.stream(
        make_model(), simple_context(), {"api_key": "sk-x", "http_client": mock_server.client(), "max_retries": 1}
    ).result()
    assert message.stop_reason == "stop"
    assert len(mock_server.requests) == 2
