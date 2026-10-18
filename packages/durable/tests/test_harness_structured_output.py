"""Port of ``test/harness-structured-output.test.ts``.

Dropped: "gives callers of bash its retained output and exit code, also for a nonzero exit" needs upstream's
``createBashTool`` (``src/tools/``), which phase 1 does not port.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from durable_harness_support import ChatSetup, add_hooks, add_tool, all_entries, assert_match, chat_setup, open_chat
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.serialization import to_json_value
from pi_ai.types import AssistantMessage, ImageContent, TextContent, ToolResultMessage
from pi_durable import MemoryStorage, NestedCallDoc, ToolResultEntry, ToolTask, define_tool, watch_events

Count = Type.Object({"count": Type.Number()})


def returning(name: str, result: Any, **extra: Any) -> Any:
    """A tool that returns ``result``, with ``extra`` such as a schema."""

    async def execute(*_args: Any) -> Any:
        return result

    return define_tool(name=name, description=name, parameters=Type.Object({}), execute=execute, **extra)


def call(name: str) -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, {}, id="c1")], stop_reason="toolUse")


DONE = faux_assistant_message([faux_text("done")])


async def nested(setup: ChatSetup, name: str) -> Any:
    """Run ``name`` as a nested call of a ``probe`` tool and return what the probe received."""
    received: list[Any] = []

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        received.append(await api.execute_tool(name, {}, call_context))
        return {}

    add_tool(
        setup.registry, define_tool(name="probe", description="probe", parameters=Type.Object({}), execute=execute)
    )
    setup.faux.set_responses([call("probe"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await opened.harness.close(context)
    return received[0]


async def direct(setup: ChatSetup, name: str) -> ToolResultMessage:
    """Run ``name`` as a model-issued call and return its transcript message."""
    setup.faux.set_responses([call(name), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    await opened.harness.close(context)
    return entry["model"][0]  # type: ignore[return-value]


def codes(result: Any) -> list[Any]:
    return [diagnostic.get("code") for diagnostic in result.get("diagnostics", [])]


async def test_gives_callers_a_schema_less_tools_output_one_text_as_a_string_one_image_as_itself_else_the_list() -> (
    None
):
    text = TextContent("one")
    image = ImageContent("AAAA", "image/png")
    text_json = {"type": "text", "text": "one"}
    image_json = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
    cases: list[tuple[list[Any], Any]] = [
        ([text], "one"),
        ([image], image_json),
        ([], ""),
        ([text, image], [text_json, image_json]),
        ([text, text], [text_json, text_json]),
    ]
    for output, value in cases:
        setup = chat_setup()
        add_tool(setup.registry, returning("plain", {"output": output}))
        received = await nested(setup, "plain")
        assert received == {
            "taskId": received["taskId"],
            "structuredOutput": value,
            "isError": False,
            "diagnostics": [],
            "durationMs": received["durationMs"],
        }


async def test_gives_callers_the_bounded_output_as_the_model_sees_it_without_the_rendered_diagnostics() -> None:
    setup = chat_setup()
    add_tool(
        setup.registry,
        returning(
            "long",
            {"output": [TextContent("a\nb\nc")], "diagnostics": [{"severity": "info", "message": "note"}]},
            output_limits={"maxLines": 1},
        ),
    )
    received = await nested(setup, "long")
    assert received["structuredOutput"] == "a\n"
    assert codes(received) == [None, "truncated"]


async def test_stores_exactly_what_callers_and_events_get_without_the_models_output_and_keeps_a_stored_null() -> None:
    setup = chat_setup()
    add_tool(setup.registry, returning("plain", {"output": [TextContent("a")]}))
    add_tool(setup.registry, returning("nothing", {"structuredOutput": None}, structured_output_schema=Type.Null()))
    received: list[Any] = []

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        received.append(await api.execute_tool("plain", {}, call_context))
        received.append(await api.execute_tool("nothing", {}, call_context))
        return {}

    add_tool(
        setup.registry, define_tool(name="probe", description="probe", parameters=Type.Object({}), execute=execute)
    )
    setup.faux.set_responses([call("probe"), DONE])
    storage = MemoryStorage()
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    # The result each nested call's document got, as committed.
    stored: list[Any] = []

    def listener(publication: Any, _context: Any) -> None:
        for change in publication["changes"]:
            if change["type"] != "document" or change["record"]["kind"] != NestedCallDoc.definition.kind:
                continue
            value = change["value"]
            if value is not None and "result" in value and len(change["ops"]) > 0:
                stored.append(value["result"])

    harness.subscribe_commits(listener)
    stream = await watch_events(harness, root.id, context)
    ends: list[Any] = []

    async def on_events(batch: Any, _context: Any) -> None:
        for event in batch:
            if event["type"] == "tool_execution_end" and "parentToolCallId" in event:
                ends.append(event["result"])

    stream.start(on_events)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await harness.wait_for_idle(context)
    await stream.stop()
    await harness.close(context)

    assert_match(received[0], {"structuredOutput": "a", "isError": False})
    assert_match(received[1], {"structuredOutput": None, "isError": False})
    assert "structuredOutput" in received[1]
    assert "output" not in received[0]
    assert stored == received
    assert ends == received


async def test_gives_callers_a_schema_tools_validated_structured_output_which_the_transcript_does_not_store() -> None:
    setup = chat_setup()
    result = {"output": [TextContent("3 things")], "structuredOutput": {"count": 3}}
    add_tool(setup.registry, returning("counter", result, structured_output_schema=Count))
    assert_match(await nested(setup, "counter"), {"structuredOutput": {"count": 3}, "isError": False})

    model_setup = chat_setup()
    add_tool(model_setup.registry, returning("counter", result, structured_output_schema=Count))
    stored = await direct(model_setup, "counter")
    assert stored.is_error is False
    assert stored.content == [TextContent("3 things")]
    assert "structuredOutput" not in json.dumps(to_json_value(stored))


async def test_turns_structured_output_that_breaks_the_schema_into_an_error_result_for_callers_and_a_report_for_the_models_calls() -> (
    None
):
    cases: list[tuple[str, Any, bool, str]] = [
        ("wrong", {"structuredOutput": {"count": "three"}}, True, "does not match its schema: count:"),
        ("missing", {"output": []}, True, "returned no structuredOutput"),
        ("undeclared", {"structuredOutput": {"count": 3}}, False, "declares no structuredOutputSchema"),
    ]
    for name, result, schema, message in cases:
        extra: dict[str, Any] = {"structured_output_schema": Count} if schema else {}
        setup = chat_setup()
        add_tool(setup.registry, returning(name, result, **extra))
        received = await nested(setup, name)
        assert received["isError"] is True
        # The broken value is dropped; a schema-less tool's error result still carries its output, here none.
        if schema:
            assert "structuredOutput" not in received
        else:
            assert received["structuredOutput"] == ""
        assert codes(received) == ["invalid_structured_output"]
        assert message in received["diagnostics"][0]["message"]

        model_setup = chat_setup()
        add_tool(model_setup.registry, returning(name, result, **extra))
        # The model never sees structured output: its call stands, and the host gets a report.
        stored = await direct(model_setup, name)
        assert stored.is_error is False
        assert message not in json.dumps(to_json_value(stored.content))
        assert message in "\n".join(str(error) for error in model_setup.reports)


async def test_gives_callers_the_output_of_an_error_result_of_a_tool_without_a_schema() -> None:
    setup = chat_setup()
    add_tool(setup.registry, returning("bad", {"output": [TextContent("bad")], "isError": True}))
    assert_match(await nested(setup, "bad"), {"isError": True, "structuredOutput": "bad"})


async def test_gives_callers_no_structured_output_for_an_error_result_the_harness_wrote_only_diagnostics() -> None:
    setup = chat_setup()
    received = await nested(setup, "missing")
    assert received["isError"] is True
    assert len(received["diagnostics"]) == 1 and received["diagnostics"][0]["code"] == "tool_unavailable"
    assert "structuredOutput" not in received


async def test_gives_callers_a_schema_tools_error_text_as_a_diagnostic_never_as_structured_output() -> None:
    setup = chat_setup()
    add_tool(
        setup.registry,
        returning("fails", {"output": [TextContent("nope")], "isError": True}, structured_output_schema=Count),
    )
    # Its text reaches the caller as an error diagnostic: structuredOutput is only ever the tool's own value.
    received = await nested(setup, "fails")
    assert_match(
        received, {"isError": True, "diagnostics": [{"severity": "error", "code": "tool_error", "message": "nope"}]}
    )
    assert "structuredOutput" not in received

    # The model's call keeps the text in its output, without the extra diagnostic.
    model_setup = chat_setup()
    add_tool(
        model_setup.registry,
        returning("fails", {"output": [TextContent("nope")], "isError": True}, structured_output_schema=Count),
    )
    stored = await direct(model_setup, "fails")
    assert stored.content == [TextContent("nope")]

    # A thrown error's message is in its tool_error diagnostic, not in the output.
    thrown = chat_setup()

    async def throws(*_args: Any) -> Any:
        raise RuntimeError("permission denied")

    add_tool(
        thrown.registry,
        define_tool(
            name="throws",
            description="throws",
            parameters=Type.Object({}),
            structured_output_schema=Count,
            execute=throws,
        ),
    )
    thrown_result = await nested(thrown, "throws")
    # One diagnostic: the thrown message, not a second one from its (empty) output.
    assert_match(
        thrown_result, {"isError": True, "diagnostics": [{"code": "tool_error", "message": "permission denied"}]}
    )
    assert len(thrown_result["diagnostics"]) == 1
    assert "structuredOutput" not in thrown_result

    # An error result whose own diagnostics say what went wrong gets no second one from its output.
    explained = chat_setup()
    add_tool(
        explained.registry,
        returning(
            "fails",
            {
                "output": [TextContent("partial log")],
                "isError": True,
                "diagnostics": [{"severity": "error", "code": "quota", "message": "quota exceeded"}],
            },
            structured_output_schema=Count,
        ),
    )
    assert (await nested(explained, "fails"))["diagnostics"] == [
        {"severity": "error", "code": "quota", "message": "quota exceeded"}
    ]

    # An error result that has structured output keeps it.
    kept = chat_setup()
    with_value = {"output": [TextContent("nope")], "structuredOutput": {"count": 0}, "isError": True}
    add_tool(kept.registry, returning("fails", with_value, structured_output_schema=Count))
    assert_match(await nested(kept, "fails"), {"isError": True, "structuredOutput": {"count": 0}})


async def test_keeps_structured_output_when_after_tool_replaces_the_output_and_validates_one_after_tool_replaces() -> (
    None
):
    setup = chat_setup()
    add_tool(setup.registry, returning("plain", {"output": [TextContent("secret")]}))
    add_tool(setup.registry, returning("counter", {"structuredOutput": {"count": 1}}, structured_output_schema=Count))
    add_tool(setup.registry, returning("other", {"structuredOutput": {"count": 1}}, structured_output_schema=Count))

    def after_tool(hook_call: Any, result: Any, *_rest: Any) -> Any:
        if hook_call.name == "other":
            return {**result, "structuredOutput": {"count": "broken"}}
        if hook_call.name == "probe":
            return result
        return {**result, "output": [TextContent("rewritten")]}

    add_hooks(setup.registry, ToolTask, {"after_tool": after_tool})
    results: dict[str, Any] = {}

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        results["plain"] = await api.execute_tool("plain", {}, call_context)
        results["counter"] = await api.execute_tool("counter", {}, call_context)
        results["other"] = await api.execute_tool("other", {}, call_context)
        return {}

    add_tool(
        setup.registry, define_tool(name="probe", description="probe", parameters=Type.Object({}), execute=execute)
    )
    setup.faux.set_responses([call("probe"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    await opened.harness.close(context)
    # A schema-less tool's value is its output, so redacting the output redacts what programs get.
    assert results["plain"]["structuredOutput"] == "rewritten"
    assert_match(results["counter"], {"structuredOutput": {"count": 1}})
    assert results["other"]["isError"] is True
    assert codes(results["other"]) == ["invalid_structured_output"]


@pytest.mark.skip(reason="needs upstream's createBashTool (src/tools/), not ported in phase 1")
async def test_gives_callers_of_bash_its_retained_output_and_exit_code_also_for_a_nonzero_exit() -> None:
    pass
