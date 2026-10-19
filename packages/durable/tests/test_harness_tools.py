"""Port of ``test/harness-tools.test.ts``."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from durable_harness_support import (
    ChatSetup,
    Installed,
    aborted,
    add_hooks,
    add_task,
    add_tool,
    all_entries,
    chat_setup,
    open_chat,
    text_of,
)
from durable_session_support import context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_durable import (
    AgentDoc,
    GenerationTask,
    Harness,
    LiveDoc,
    MemoryStorage,
    ToolResultEntry,
    ToolTask,
    configure,
    define_extension,
    define_task,
    define_tool,
    hook,
    section,
)
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools import create_bash_tool, create_edit_tool, create_read_tool

EchoParameters = Type.Object({"text": Type.Optional(Type.String())})


def tool(name: str, execute: Any, **extra: Any) -> Any:
    return define_tool(name=name, description=f"The {name} tool", parameters=EchoParameters, execute=execute, **extra)


def calls(*items: tuple[str, dict[str, Any], str]) -> AssistantMessage:
    """A tool-calling answer with one call per ``(name, args, id)``."""
    return faux_assistant_message(
        [faux_tool_call(name, args, id=id) for name, args, id in items], stop_reason="toolUse"
    )


DONE = faux_assistant_message([faux_text("done")])


@dataclasses.dataclass
class Ran:
    harness: Harness
    root: Any
    entries: list[Any]
    status: str


async def run(setup: ChatSetup, responses: list[Any], prepare: Any = None, *, env: Any = None) -> Ran:
    setup.faux.set_responses(responses)
    opened = await open_chat(MemoryStorage(), setup, env=env)
    if prepare is not None:
        await prepare(opened.harness, opened.root)
    settled = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    return Ran(opened.harness, opened.root, await all_entries(opened.root), settled["status"])


def results(entries: list[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if ToolResultEntry.is_(entry)]


def result_text(message: ToolResultMessage) -> str:
    return "|".join(item.text if isinstance(item, TextContent) else f"[{item.type}]" for item in message.content)


def output(text: str) -> Any:
    return {"output": [TextContent(text)]}


async def nothing(*_args: Any) -> Any:
    return {"output": []}


async def _never(*_args: Any) -> None:
    return None


# ─── tool round ─────────────────────────────────────────────────────────────


async def test_runs_input_tool_call_tool_result_and_answer_and_settles_the_input() -> None:
    setup = chat_setup()

    async def echo(args: Any, *_rest: Any) -> Any:
        return output(f"echo {args.get('text')}")

    add_tool(setup.registry, tool("echo", echo))
    ran = await run(setup, [calls(("echo", {"text": "hi"}, "c1")), DONE])
    assert ran.status == "done"
    assert [entry["kind"] for entry in ran.entries] == [
        "pi.user",
        "pi.system",
        "pi.assistant",
        "pi.tool-result",
        "pi.assistant",
    ]
    system = ran.entries[1]["model"][0]
    assert isinstance(system, SystemMessage) and system.tools_added is not None
    assert [(declared.name, declared.description) for declared in system.tools_added] == [("echo", "The echo tool")]
    assert isinstance(system.tools_added[0].parameters, dict)
    [result] = results(ran.entries)
    assert (result.tool_call_id, result.tool_name, result.is_error) == ("c1", "echo", False)
    assert result_text(result) == "echo hi"
    assert ran.entries[3]["data"] == {"diagnostics": []}
    assert ran.entries[3].get("byTaskId") is not None
    # The second request sees the tool result right after its call.
    assert setup.faux.state.call_count == 2
    assert await ran.harness.snapshot(LiveDoc, ran.root.id, context) == {}
    await ran.harness.close(context)


async def test_gives_tools_and_hooks_the_harness_models() -> None:
    """#10395"""
    setup = chat_setup()
    seen: list[Any] = []

    async def echo(_args: Any, api: Any, _context: Any) -> Any:
        seen.append(api.models)
        return {"output": []}

    def before_tool(_call: Any, api: Any, _context: Any) -> None:
        seen.append(api.models)

    add_tool(setup.registry, tool("echo", echo))
    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    ran = await run(setup, [calls(("echo", {}, "c1")), DONE])
    assert ran.status == "done"
    assert len(seen) == 2 and all(models is setup.models for models in seen)
    await ran.harness.close(context)


async def test_answers_calls_to_tools_the_request_did_not_offer_without_a_task() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", nothing))
    ran = await run(setup, [calls(("ghost", {}, "c1"), ("echo", {}, "c2")), DONE])
    assert ran.status == "done"
    ghost, echo = results(ran.entries)
    assert ghost.tool_call_id == "c1" and ghost.is_error is True
    assert result_text(ghost) == "<harness>\n[error] Tool ghost is not available\n</harness>"
    assert echo.tool_call_id == "c2" and echo.is_error is False
    ghost_entry = next(entry for entry in ran.entries if ToolResultEntry.is_(entry))
    assert ghost_entry["data"] == {
        "diagnostics": [{"severity": "error", "code": "tool_unavailable", "message": "Tool ghost is not available"}]
    }
    tasks = await ran.harness.commit(lambda tx: tx.scan_tasks({"conversationId": ran.root.id}, 20), context)
    assert len([task for task in tasks["items"] if task["kind"] == "pi.tool"]) == 1
    await ran.harness.close(context)


async def test_answers_a_call_to_a_tool_deactivated_after_preparation_with_tool_unavailable() -> None:
    setup = chat_setup()
    seen: list[str] = []

    async def echo(*_args: Any) -> Any:
        seen.append("ran")
        return {"output": []}

    add_tool(setup.registry, tool("echo", echo))
    holder: list[Any] = []

    async def deactivate(*_args: Any) -> Any:
        await holder[0].configure({"tools": []}, context)
        return calls(("echo", {}, "c1"))

    async def prepare(_harness: Any, conversation: Any) -> None:
        holder.append(conversation)

    ran = await run(setup, [deactivate, DONE], prepare)
    assert seen == []
    assert result_text(results(ran.entries)[0]) == "<harness>\n[error] Tool echo is not available\n</harness>"
    # The next preparation removes it.
    systems = [entry for entry in ran.entries if entry["kind"] == "pi.system"]
    assert [reference.name for reference in systems[-1]["model"][0].tools_removed] == ["echo"]
    await ran.harness.close(context)


async def test_removes_unregistered_active_tools_from_the_offer_and_adds_them_back_after_re_registration() -> None:
    setup = chat_setup()
    echo = tool("echo", nothing)
    registration = add_tool(setup.registry, echo)
    first = await run(setup, [DONE])
    registration.dispose()
    setup.faux.set_responses([calls(("echo", {}, "c1")), DONE])
    second = await (await first.root.submit({"type": "input", "content": "again"}, context)).wait(context)
    assert second["status"] == "done"
    entries = await all_entries(first.root)
    removal = [entry for entry in entries if entry["kind"] == "pi.system"][-1]["model"][0]
    assert [reference.name for reference in removal.tools_removed] == ["echo"]
    assert results(entries)[0].is_error is True
    # The stored agent is not rewritten; the tool is only not resolved.
    assert list((await first.root.agent(context))["tools"]) == []

    add_tool(setup.registry, echo)
    setup.faux.set_responses([DONE])
    await (await first.root.submit({"type": "input", "content": "back"}, context)).wait(context)
    entries = await all_entries(first.root)
    addition = [entry for entry in entries if entry["kind"] == "pi.system"][-1]["model"][0]
    assert [declared.name for declared in addition.tools_added] == ["echo"]
    await first.harness.close(context)


async def test_produces_tool_unavailable_when_the_implementation_is_unregistered_before_its_task_runs() -> None:
    setup = chat_setup()
    second: list[Installed] = []

    async def first_tool(*_args: Any) -> Any:
        second[0].dispose()
        return {"output": []}

    async def second_tool(*_args: Any) -> Any:
        return output("ran")

    add_tool(setup.registry, tool("first", first_tool, execution_mode="sequential"))
    second.append(add_tool(setup.registry, tool("second", second_tool)))
    ran = await run(setup, [calls(("first", {}, "c1"), ("second", {}, "c2")), DONE])
    assert ran.status == "done"
    _first, late = results(ran.entries)
    assert result_text(late) == "<harness>\n[error] Tool second is not available\n</harness>"
    await ran.harness.close(context)


async def test_reads_the_execution_mode_when_a_round_starts_and_keeps_it_for_the_round() -> None:
    setup = chat_setup()
    events: list[str] = []

    def slow(name: str) -> Any:
        async def execute(*_args: Any) -> Any:
            events.append(f"start {name}")
            # Changed after the round started: not seen by this round.
            setup.settings["toolExecution"] = "parallel"
            await asyncio.sleep(0.02)
            events.append(f"end {name}")
            return {"output": []}

        return execute

    add_tool(setup.registry, tool("a", slow("a")))
    add_tool(setup.registry, tool("b", slow("b")))

    # Changed while the model request runs: the round that follows uses it.
    def request(*_args: Any) -> Any:
        setup.settings["toolExecution"] = "sequential"
        return calls(("a", {}, "c1"), ("b", {}, "c2"))

    ran = await run(setup, [request, DONE])
    assert events == ["start a", "end a", "start b", "end b"]
    await ran.harness.close(context)


async def _trace(setup: ChatSetup) -> list[str]:
    events: list[str] = []

    def slow(name: str) -> Any:
        async def execute(*_args: Any) -> Any:
            events.append(f"start {name}")
            await asyncio.sleep(0.02)
            events.append(f"end {name}")
            return {"output": []}

        return execute

    add_tool(setup.registry, tool("a", slow("a")))
    add_tool(setup.registry, tool("b", slow("b")))
    ran = await run(setup, [calls(("a", {}, "c1"), ("b", {}, "c2")), DONE])
    await ran.harness.close(context)
    return events


async def test_runs_a_round_in_parallel_by_default_and_sequentially_when_configured_or_required_by_a_tool() -> None:
    assert (await _trace(chat_setup()))[:2] == ["start a", "start b"]
    configured = chat_setup()
    configured.settings["toolExecution"] = "sequential"
    assert await _trace(configured) == ["start a", "end a", "start b", "end b"]
    per_tool = chat_setup()
    events: list[str] = []

    async def a(*_args: Any) -> Any:
        await asyncio.sleep(0.02)
        events.append("a")
        return {"output": []}

    async def b(*_args: Any) -> Any:
        events.append("b")
        return {"output": []}

    add_tool(per_tool.registry, tool("a", a, execution_mode="sequential"))
    add_tool(per_tool.registry, tool("b", b))
    ran = await run(per_tool, [calls(("a", {}, "c1"), ("b", {}, "c2")), DONE])
    tasks = (await ran.harness.commit(lambda tx: tx.scan_tasks({"conversationId": ran.root.id}, 20), context))["items"]
    tools = sorted((task for task in tasks if task["kind"] == "pi.tool"), key=lambda task: task["id"])
    # The generation owns both tools and creates the second only after the first ended.
    generation = next(task for task in tasks if task["kind"] == "pi.generation")
    assert [task["owner"] for task in tools] == [generation["id"], generation["id"]]
    assert events == ["a", "b"]
    await ran.harness.close(context)


# ─── tool results ───────────────────────────────────────────────────────────


async def test_uses_retained_output_and_the_last_details_when_the_result_omits_them() -> None:
    setup = chat_setup()

    async def log(_args: Any, api: Any, _context: Any) -> Any:
        api.output("line 1\n")
        api.output(b"line 2\nline 3\n")
        api.diagnostic({"severity": "info", "message": "from api"})
        await api.details({"step": 1}, context)
        await api.details({"step": 2}, context)
        return {"diagnostics": [{"severity": "warn", "message": "from result"}]}

    add_tool(setup.registry, tool("log", log, output_limits={"maxLines": 2}))
    ran = await run(setup, [calls(("log", {}, "c1")), DONE])
    [result] = results(ran.entries)
    assert result.details == {"step": 2}
    assert result_text(result) == (
        "line 1\nline 2\n|<harness>\n[info] from api\n[warn] from result\n"
        "[warn] Output truncated to its beginning: 1 lines, 7 bytes dropped\n</harness>"
    )
    entry = next(candidate for candidate in ran.entries if ToolResultEntry.is_(candidate))
    assert [diagnostic.get("code") for diagnostic in entry["data"]["diagnostics"]] == [None, None, "truncated"]
    await ran.harness.close(context)


async def test_offers_the_tail_window_with_the_configured_pace_and_accepts_skipped_output() -> None:
    setup = chat_setup()
    setup.settings["progress"] = {"outputIntervalMs": 250}
    windows: list[Any] = []

    async def tailed(_args: Any, api: Any, _context: Any) -> Any:
        windows.append(api.output_window)
        api.output("dropped\n")
        api.output("x\ny\n", {"bytes": 8, "newlines": 1, "endsWithNewline": True})
        return {}

    async def headed(_args: Any, api: Any, _context: Any) -> Any:
        windows.append(api.output_window)
        return {}

    add_tool(setup.registry, tool("tailed", tailed, output_limits={"maxLines": 1, "retain": "tail"}))
    add_tool(setup.registry, tool("headed", headed, output_limits={"maxLines": 1}))
    ran = await run(setup, [calls(("tailed", {}, "c1"), ("headed", {}, "c2")), DONE])
    assert {"maxBytes": 50 * 1024, "maxLines": 1, "minIntervalMs": 250, "bytesPerSecond": 100 * 1024} in windows
    assert None in windows
    # 8 bytes written, 8 skipped, then "x\n" dropped by the window: 3 lines, 18 bytes in all.
    c1 = next(result for result in results(ran.entries) if result.tool_call_id == "c1")
    assert result_text(c1) == "y\n|<harness>\n[warn] Output truncated to its end: 3 lines, 18 bytes dropped\n</harness>"
    await ran.harness.close(context)


async def test_bounds_explicit_text_content_and_keeps_other_content() -> None:
    setup = chat_setup()
    image = ImageContent("AAAA", "image/png")

    async def big(*_args: Any) -> Any:
        return {"output": [TextContent("a\nb\n"), image, TextContent("c\nd\n")]}

    add_tool(setup.registry, tool("big", big, output_limits={"maxLines": 2, "retain": "tail"}))
    ran = await run(setup, [calls(("big", {}, "c1")), DONE])
    assert result_text(results(ran.entries)[0]) == (
        "[image]|c\nd\n|<harness>\n[warn] Output truncated to its end: 2 lines, 4 bytes dropped\n</harness>"
    )
    await ran.harness.close(context)


async def test_turns_a_raise_into_a_tool_error_result_with_the_partial_output() -> None:
    setup = chat_setup()

    async def fail(_args: Any, api: Any, _context: Any) -> Any:
        api.output("partial\n")
        raise RuntimeError("boom")

    add_tool(setup.registry, tool("fail", fail))
    ran = await run(setup, [calls(("fail", {}, "c1")), DONE])
    assert ran.status == "done"
    [result] = results(ran.entries)
    assert result.is_error is True
    assert result_text(result) == "partial\n|<harness>\n[error] boom\n</harness>"
    await ran.harness.close(context)


async def test_records_how_long_execute_took_excluding_hooks_and_nothing_for_calls_that_did_not_run() -> None:
    """#10549"""
    setup = chat_setup()

    async def slow(*_args: Any) -> Any:
        await asyncio.sleep(0.03)
        return {"output": []}

    async def thrower(*_args: Any) -> Any:
        await asyncio.sleep(0.03)
        raise RuntimeError("boom")

    async def before_tool(call: Any, *_rest: Any) -> Any:
        await asyncio.sleep(0.1)
        return {"block": "no"} if call.id == "blocked" else None

    add_tool(setup.registry, tool("slow", slow))
    add_tool(setup.registry, tool("thrower", thrower))
    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    ran = await run(setup, [calls(("slow", {}, "slow"), ("thrower", {}, "thrower"), ("slow", {}, "blocked")), DONE])
    by_id = {result.tool_call_id: result for result in results(ran.entries)}
    for id in ("slow", "thrower"):
        duration = by_id[id].duration_ms
        assert duration is not None and 25 <= duration < 100
    assert by_id["thrower"].is_error is True
    assert by_id["blocked"].duration_ms is None
    await ran.harness.close(context)


async def test_validates_arguments_before_and_after_before_tool_and_applies_blocks_and_replacements() -> None:
    setup = chat_setup()
    seen: list[Any] = []

    async def echo(args: Any, *_rest: Any) -> Any:
        seen.append(args)
        return {"output": []}

    def before_tool(call: Any, *_rest: Any) -> Any:
        if call.id == "block":
            return {"block": "not today"}
        if call.id == "throw":
            raise RuntimeError("hook failed")
        if call.id == "bad":
            return {"arguments": {"text": {"not": "a string"}}}
        return {"arguments": {"text": f"{call.arguments.get('text')}!"}}

    add_tool(setup.registry, tool("echo", echo))
    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    ran = await run(
        setup,
        [
            calls(
                ("echo", {"text": 1}, "coerced"),
                ("echo", {"text": {"not": "a string"}}, "invalid"),
                ("echo", {}, "block"),
                ("echo", {}, "throw"),
                ("echo", {"text": "x"}, "bad"),
                ("echo", {"text": "x"}, "ok"),
            ),
            DONE,
        ],
    )
    # Parallel tools append their results in completion order.
    by_id = {result.tool_call_id: (result.is_error, result_text(result)) for result in results(ran.entries)}
    assert by_id["block"] == (True, "<harness>\n[error] Tool call blocked: not today\n</harness>")
    assert by_id["throw"] == (True, "<harness>\n[error] Tool call blocked: hook failed\n</harness>")
    assert by_id["bad"][0] is True and "Validation failed" in by_id["bad"][1]
    assert by_id["ok"] == (False, "")
    assert by_id["invalid"][0] is True and "Validation failed" in by_id["invalid"][1]
    # pi-ai coerces a number to a string before the first validation.
    assert by_id["coerced"] == (False, "")
    assert {"text": "1!"} in seen and {"text": "x!"} in seen
    assert len(seen) == 2
    await ran.harness.close(context)


async def test_repairs_arguments_with_prepare_arguments_before_validation_and_a_raising_repair_is_invalid() -> None:
    setup = chat_setup()
    seen: list[Any] = []

    async def echo(args: Any, *_rest: Any) -> Any:
        seen.append(args)
        return {"output": []}

    def prepare(args: Any) -> Any:
        text = args.get("text")
        if text == "throw":
            raise RuntimeError("cannot repair")
        return {"text": f"#{text}"} if isinstance(text, int) else args

    add_tool(setup.registry, tool("echo", echo, prepare_arguments=prepare))
    ran = await run(setup, [calls(("echo", {"text": 7}, "fixed"), ("echo", {"text": "throw"}, "broken")), DONE])
    by_id = {result.tool_call_id: result_text(result) for result in results(ran.entries)}
    assert by_id["fixed"] == ""
    assert by_id["broken"] == "<harness>\n[error] cannot repair\n</harness>"
    assert seen == [{"text": "#7"}]
    # The stored call keeps what the model sent.
    call = next(item for item in ran.entries[2]["model"][0].content if isinstance(item, ToolCall))
    assert call.arguments == {"text": 7}
    await ran.harness.close(context)


async def test_lets_the_first_before_tool_block_win_and_skips_later_handlers() -> None:
    setup = chat_setup()
    add_tool(setup.registry, tool("echo", nothing))
    asked: list[str] = []

    def first(*_args: Any) -> Any:
        asked.append("first")
        return {"block": "first says no"}

    def second(*_args: Any) -> Any:
        asked.append("second")
        return {"block": "second says no"}

    add_hooks(setup.registry, ToolTask, {"before_tool": first})
    add_hooks(setup.registry, ToolTask, {"before_tool": second})
    ran = await run(setup, [calls(("echo", {}, "c1")), DONE])
    assert asked == ["first"]
    assert result_text(results(ran.entries)[0]) == "<harness>\n[error] Tool call blocked: first says no\n</harness>"
    await ran.harness.close(context)


async def test_chains_after_tool_replacements_and_observes_the_round_with_after_tools() -> None:
    setup = chat_setup()

    async def echo(*_args: Any) -> Any:
        return output("raw")

    add_tool(setup.registry, tool("echo", echo))
    add_hooks(
        setup.registry, ToolTask, {"after_tool": lambda _call, result, *_: {**result, "output": [TextContent("first")]}}
    )

    def replaced(_call: Any, result: Any, *_rest: Any) -> Any:
        texts = "|".join(item.text if isinstance(item, TextContent) else "" for item in result.get("output") or [])
        return {**result, "details": {"replaced": texts}}

    add_hooks(setup.registry, ToolTask, {"after_tool": replaced})
    observed: list[Any] = []
    add_hooks(
        setup.registry,
        GenerationTask,
        {"after_tools": lambda assistant, entries, *_: observed.extend([assistant, entries])},
    )
    ran = await run(setup, [calls(("echo", {}, "c1")), DONE])
    [result] = results(ran.entries)
    assert result_text(result) == "first"
    assert result.details == {"replaced": "first"}
    result_entry = next(entry for entry in ran.entries if ToolResultEntry.is_(entry))
    assert observed == [ran.entries[2]["id"], [result_entry["id"]]]
    await ran.harness.close(context)


async def test_runs_the_hooks_of_the_selected_extensions_a_task_owned_child_copies_its_owners_selection() -> None:
    setup = chat_setup()
    called_in: list[Any] = []
    Echo = define_extension("echo", tools=[tool("echo", nothing)])
    Audit = define_extension(
        "audit", hooks=[hook(ToolTask, {"before_tool": lambda _call, api, _ctx: called_in.append(api.conversation_id)})]
    )
    setup.registry.install(Echo)
    setup.registry.install(Audit)
    # Audit is installed but not in the default selection.
    setup.settings["extensions"] = [Echo]
    first = await run(setup, [DONE])
    await first.root.configure({"extensions": {"add": [Audit]}}, context)
    # An owner task no registered definition takes stays live and pending.
    owner = define_task(
        name="test.owner", version=1, initial=lambda _input: {"phase": "never"}, phases={"never": _never}, abort=_never
    )

    async def create(tx: Any) -> Any:
        task_id = await tx.create_task(
            owner, {}, {"ownership": {"kind": "conversation"}, "conversationId": first.root.id}
        )
        return (await tx.create_conversation({"ownership": {"kind": "task", "taskId": task_id}}))["id"]

    child_id = await first.harness.commit(create, context)
    child = await first.harness.conversation(child_id, context)
    assert child is not None
    other = await first.harness.create_conversation(
        {"ownership": {"kind": "ownerless"}, "agent": {"model": {"provider": "faux", "modelId": "faux-1"}}}, context
    )
    for conversation in (first.root, child, other):
        setup.faux.set_responses([calls(("echo", {}, "c1")), DONE])
        await (await conversation.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert called_in == [first.root.id, child.id]
    await first.harness.close(context)


async def test_applies_add_tools_and_terminates_only_when_every_result_of_the_round_asks_to() -> None:
    setup = chat_setup()

    async def stop_execute(*_args: Any) -> Any:
        return {"output": [], "control": {"terminate": True}}

    async def grow_execute(*_args: Any) -> Any:
        return {"output": [], "control": {"addTools": ["extra", "stop"]}}

    stop = tool("stop", stop_execute)
    grow = tool("grow", grow_execute)
    add_tool(setup.registry, stop)
    add_tool(setup.registry, grow)
    add_tool(setup.registry, tool("extra", nothing))

    async def prepare(_harness: Any, root: Any) -> None:
        await root.configure({"tools": [stop, grow]}, context)

    first = await run(setup, [calls(("stop", {}, "c1"))], prepare)
    assert first.status == "done"
    assert first.entries[-1]["kind"] == "pi.tool-result"
    tasks = await first.harness.commit(lambda tx: tx.scan_tasks({"conversationId": first.root.id}, 20), context)
    assert all(task["state"]["status"] == "terminal" for task in tasks["items"])

    setup.faux.set_responses([calls(("stop", {}, "c1"), ("grow", {}, "c2")), DONE])
    second = await (await first.root.submit({"type": "input", "content": "again"}, context)).wait(context)
    assert second["status"] == "done"
    assert (await all_entries(first.root))[-1]["kind"] == "pi.assistant"
    # addTools appends to the stored tool list, skipping names it already holds.
    assert (await first.harness.snapshot(AgentDoc, first.root.id, context))["tools"] == ["stop", "grow", "extra"]  # type: ignore[index]
    await first.harness.close(context)


async def test_does_not_terminate_on_terminate_false() -> None:
    """#10858"""
    setup = chat_setup()

    async def keep_execute(*_args: Any) -> Any:
        return {"output": [], "control": {"terminate": False}}

    keep = tool("keep", keep_execute)
    add_tool(setup.registry, keep)

    async def prepare(_harness: Any, root: Any) -> None:
        await root.configure({"tools": [keep]}, context)

    ran = await run(setup, [calls(("keep", {}, "c1")), DONE], prepare)
    assert ran.status == "done"
    assert ran.entries[-1]["kind"] == "pi.assistant"
    await ran.harness.close(context)


# ─── generation hooks ───────────────────────────────────────────────────────


async def test_replaces_request_messages_observes_responses_and_continues_on_yield() -> None:
    setup = chat_setup()
    requests: list[list[str]] = []

    def record(request: Any, *_rest: Any) -> Any:
        requests.append([f"{message.role}:{text_of(message) or ''}" for message in request.messages])
        return faux_assistant_message([faux_text(f"answer {len(requests)}")])

    add_hooks(
        setup.registry,
        GenerationTask,
        {
            "before_request": lambda request, *_: {
                "messages": [*request["messages"], UserMessage("injected", timestamp=0)]
            }
        },
    )
    responses: list[str] = []
    add_hooks(
        setup.registry, GenerationTask, {"after_response": lambda message, *_: responses.append(text_of(message) or "")}
    )
    yields = [0]

    def on_yield(*_args: Any) -> Any:
        yields[0] += 1
        return {"continue": [TextContent("keep going")]} if yields[0] == 1 else None

    add_hooks(setup.registry, GenerationTask, {"on_yield": on_yield})
    ran = await run(setup, [record, record])
    assert ran.status == "done"
    assert requests[0][-2:] == ["user:go", "user:injected"]
    assert responses == ["answer 1", "answer 2"]
    assert [entry["kind"] for entry in ran.entries] == ["pi.user", "pi.assistant", "pi.user", "pi.assistant"]
    # The injected message was used for the request only.
    assert not any(text_of((entry.get("model") or [None])[0]) == "injected" for entry in ran.entries)
    await ran.harness.close(context)


async def test_keeps_the_runs_input_open_across_an_on_yield_continuation_and_answers_it_with_the_final_answer() -> None:
    setup = chat_setup()
    yields = [0]
    holder: dict[str, Any] = {}

    def on_yield(*_args: Any) -> Any:
        yields[0] += 1
        return {"continue": "again"} if yields[0] == 1 else None

    add_hooks(setup.registry, GenerationTask, {"on_yield": on_yield})

    async def second(*_args: Any) -> Any:
        harness: Harness = holder["harness"]
        live = await harness.snapshot(LiveDoc, 1, context)  # type: ignore[arg-type]
        holder["input"] = live["run"]["inputs"][0]  # type: ignore[index]
        submission = await harness.submission(holder["input"], context)
        holder["status"] = (await submission.status(context))["status"]  # type: ignore[union-attr]
        return faux_assistant_message([faux_text("second")])

    async def prepare(opened: Any, _root: Any) -> None:
        holder["harness"] = opened

    ran = await run(setup, [faux_assistant_message([faux_text("first")]), second], prepare)
    assert holder["status"] == "placed"
    answers = [entry for entry in ran.entries if entry["kind"] == "pi.assistant"]
    submission = await ran.harness.submission(holder["input"], context)
    status = await submission.status(context)  # type: ignore[union-attr]
    assert status["status"] == "done" and status["answer"] == answers[1]["id"]  # type: ignore[typeddict-item]
    await ran.harness.close(context)


async def test_observes_responses_that_arrive_by_polling_a_deferred_request() -> None:
    setup = chat_setup(deferred={"pending_fetches": 1, "poll_after_ms": 1})
    observed: list[str] = []
    add_hooks(
        setup.registry,
        GenerationTask,
        {"after_response": lambda message, *_: observed.append(f"{message.stop_reason}:{text_of(message) or ''}")},
    )
    setup.settings["stream"] = {"deferred": True}
    ran = await run(setup, [faux_assistant_message([faux_text("late")])])
    assert ran.status == "done"
    # The still-deferred results are not terminal.
    assert observed == ["stop:late"]
    await ran.harness.close(context)


async def test_lets_the_first_on_yield_continuation_win_and_reports_raises_without_stopping_later_handlers() -> None:
    setup = chat_setup()
    called: list[str] = []

    def throwing(*_args: Any) -> None:
        called.append("throwing observer")
        raise RuntimeError("observer failed")

    add_hooks(setup.registry, GenerationTask, {"after_response": throwing})
    add_hooks(setup.registry, GenerationTask, {"after_response": lambda *_: called.append("next observer")})
    yields = [0]

    def first_yield(*_args: Any) -> Any:
        yields[0] += 1
        return {"continue": "first"} if yields[0] == 1 else None

    def second_yield(*_args: Any) -> Any:
        called.append("second onYield")
        return {"continue": "second"} if called.count("second onYield") == 1 else None

    add_hooks(setup.registry, GenerationTask, {"on_yield": first_yield})
    add_hooks(setup.registry, GenerationTask, {"on_yield": second_yield})
    ran = await run(
        setup,
        [
            faux_assistant_message([faux_text("a")]),
            faux_assistant_message([faux_text("b")]),
            faux_assistant_message([faux_text("c")]),
        ],
    )
    # The first continuation skips the second handler; on the next answer the second handler's continuation wins.
    users = [text_of(entry["model"][0]) for entry in ran.entries if entry["kind"] == "pi.user"]
    assert users == ["go", "first", "second"]
    assert called.count("second onYield") == 2
    assert called.count("next observer") == 3
    assert "observer failed" in [str(error) for error in setup.reports]
    await ran.harness.close(context)


async def test_keeps_durable_hook_decisions_in_task_memos() -> None:
    setup = chat_setup()
    asked = [0]
    add_tool(setup.registry, tool("echo", nothing))

    async def before_tool(_call: Any, api: Any, _context: Any) -> None:
        asked[0] += 1
        decision = await api.memo("approval:decision", "approved", context)
        assert await api.memo("approval:decision", "denied", context) == decision

    add_hooks(setup.registry, ToolTask, {"before_tool": before_tool})
    ran = await run(setup, [calls(("echo", {}, "c1")), DONE])
    assert asked[0] == 1
    await ran.harness.close(context)


# ─── tool execution api ─────────────────────────────────────────────────────


async def test_builds_the_environment_per_call_from_the_conversations_cwd_and_runs_commits_memos_and_child_tasks() -> (
    None
):
    setup = chat_setup()

    async def child_run(task: Any, runtime: Any, call_context: Any) -> None:
        result = task["input"]["n"] * 2
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "completed", "result": result}},
            call_context,
        )

    child = define_task(
        name="test.child", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": child_run}, abort=_never
    )
    add_task(setup.registry, child)
    seen: list[Any] = []

    async def probe(_args: Any, api: Any, _context: Any) -> Any:
        seen.append(None if api.env is None else api.env.cwd)
        seen.append([each.name for each in (await api.agent(context))["tools"]])
        seen.append(api.registry.extension("tool:probe") is not None)

        async def change(tx: Any) -> Any:
            # The next call runs in the new directory: its environment is built when it executes.
            await configure(tx, api.conversation_id, {"cwd": "/"})
            return await tx.append_entry(api.conversation_id, {"kind": "test.note", "data": api.call_id})

        entry = await api.commit(change, context)
        seen.append(entry.get("byTaskId") == api.task_id)
        seen.extend([await api.memo("m", 1, context), await api.memo("m", 2, context)])
        id = await api.create_task(child, {"n": 21}, {"ownership": {"kind": "conversation"}}, context)
        done = await api.wait_for_task(id, context)
        seen.append(done["state"]["outcome"])
        return {"output": []}

    add_tool(setup.registry, tool("probe", probe))
    setup.faux.set_responses([calls(("probe", {}, "c1"), ("probe", {}, "c2")), DONE])
    setup.settings["toolExecution"] = "sequential"
    targets: list[Any] = []

    def env(target: Any, _context: Any) -> Any:
        targets.append((target["conversationId"], target.get("cwd")))
        return LocalExecutionEnv(cwd=target.get("cwd") or "/tmp")

    opened = await open_chat(MemoryStorage(), setup, env=env)
    await opened.root.configure({"cwd": "/tmp"}, context)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    outcome = {"status": "completed", "result": 42}

    def call(cwd: str) -> list[Any]:
        return [cwd, ["probe"], True, True, 1, 1, outcome]

    assert seen == [*call("/tmp"), *call("/")]
    assert (opened.root.id, "/tmp") in targets
    assert (opened.root.id, "/") in targets
    await opened.harness.close(context)


async def test_answers_a_raising_environment_with_a_tool_error_result_and_reports_it_once_while_preparing() -> None:
    setup = chat_setup()

    async def probe(*_args: Any) -> Any:
        return output("ran")

    add_tool(setup.registry, tool("probe", probe))

    def env(*_args: Any) -> Any:
        raise RuntimeError("no sandbox")

    ran = await run(setup, [calls(("probe", {}, "c1")), DONE], env=env)
    [result] = results(ran.entries)
    assert result.is_error is True
    assert result_text(result) == "<harness>\n[error] no sandbox\n</harness>"
    # Each preparation reports the failure and renders without an environment.
    assert len([error for error in setup.reports if str(error) == "no sandbox"]) == 2
    await ran.harness.close(context)


# ─── coding tools ───────────────────────────────────────────────────────────


async def test_answers_a_failing_command_with_its_retained_tail_and_diagnostics_in_order(tmp_path: Any) -> None:
    setup = chat_setup()
    add_tool(setup.registry, create_bash_tool())
    command = "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done; exit 7"
    setup.faux.set_responses([calls(("bash", {"command": command}, "b")), DONE])
    env = LocalExecutionEnv(cwd=str(tmp_path))
    opened = await open_chat(MemoryStorage(), setup, env=env)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    [result] = results([entry])
    assert result.is_error is True
    text = result_text(result)
    assert text.startswith("line-1001\n")
    assert [diagnostic["code"] for diagnostic in entry["data"]["diagnostics"]] == [
        "full_output",
        "exit_code",
        "truncated",
    ]
    assert "line-3000\n|<harness>\n[info] Full output: " in text
    assert "\n[error] Command exited with code 7\n[warn] Output truncated to its end: 1000 lines, " in text
    await opened.harness.close(context)


async def test_reads_edits_and_runs_a_command_in_one_run_then_answers(tmp_path: Any) -> None:
    (tmp_path / "notes.txt").write_text("hello world\n")
    setup = chat_setup()
    setup.registry.install(
        define_extension("coding", tools=[create_read_tool(), create_edit_tool(), create_bash_tool()])
    )
    setup.faux.set_responses(
        [
            calls(("read", {"path": "notes.txt"}, "r")),
            calls(("edit", {"path": "notes.txt", "edits": [{"oldText": "world", "newText": "durable"}]}, "e")),
            calls(("bash", {"command": "cat notes.txt"}, "b")),
            DONE,
        ]
    )
    env = LocalExecutionEnv(cwd=str(tmp_path))
    opened = await open_chat(MemoryStorage(), setup, env=env)
    settled = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    entries = await all_entries(opened.root)
    assert [(result.tool_name, result.is_error, result_text(result)) for result in results(entries)] == [
        ("read", False, "hello world\n"),
        ("edit", False, "Successfully replaced 1 block(s) in notes.txt."),
        ("bash", False, "hello durable\n"),
    ]
    assert entries[-1]["kind"] == "pi.assistant"
    assert (tmp_path / "notes.txt").read_text() == "hello durable\n"
    await opened.harness.close(context)


# ─── tool progress and lifetime ─────────────────────────────────────────────


async def test_applies_the_default_output_limits() -> None:
    setup = chat_setup()

    async def lines(_args: Any, api: Any, _context: Any) -> Any:
        for index in range(1, 2501):
            api.output(f"{index}\n")
        return {}

    add_tool(setup.registry, tool("lines", lines))
    ran = await run(setup, [calls(("lines", {}, "c1")), DONE])
    text = result_text(results(ran.entries)[0])
    assert text.startswith("1\n2\n")
    assert text.endswith(
        "\n2000\n|<harness>\n[warn] Output truncated to its beginning: 500 lines, 2500 bytes dropped\n</harness>"
    )
    await ran.harness.close(context)


async def test_sanitizes_running_output_but_keeps_explicit_result_content_as_the_tool_returned_it() -> None:
    setup = chat_setup()
    slot_output: list[Any] = []

    async def noisy(_args: Any, api: Any, _context: Any) -> Any:
        api.output("a\u0007b\r\n")
        await api.details({"ready": True}, context)
        live = await api.snapshot(LiveDoc, api.conversation_id, context)
        slot_output.append(live["tools"][0].get("output"))
        return {}

    async def explicit(*_args: Any) -> Any:
        return output("c\u001bd")

    add_tool(setup.registry, tool("noisy", noisy))
    add_tool(setup.registry, tool("explicit", explicit))
    ran = await run(setup, [calls(("noisy", {}, "c1"), ("explicit", {}, "c2")), DONE])
    assert slot_output == ["ab\n"]
    by_id = {result.tool_call_id: result_text(result) for result in results(ran.entries)}
    assert by_id["c1"] == "ab\n"
    assert by_id["c2"] == "c\u001bd"
    await ran.harness.close(context)


async def test_drops_control_keys_set_to_undefined_instead_of_faulting() -> None:
    from pi_chord.json import UNDEFINED

    setup = chat_setup()

    async def grow(*_args: Any) -> Any:
        return {"output": [], "control": {"addTools": ["extra"], "terminate": UNDEFINED}}

    add_tool(setup.registry, tool("grow", grow))
    extra = tool("extra", nothing)
    add_tool(setup.registry, extra)

    async def prepare(_harness: Any, conversation: Any) -> None:
        await conversation.configure({"tools": {"remove": [extra]}}, context)

    ran = await run(setup, [calls(("grow", {}, "c1")), DONE], prepare)
    assert ran.status == "done"
    # addTools deletes the name from a stored ``{"remove"}`` filter.
    assert (await ran.harness.snapshot(AgentDoc, ran.root.id, context))["tools"] == {"remove": []}  # type: ignore[index]
    await ran.harness.close(context)


async def test_uses_explicit_none_details_instead_of_the_last_reported_value() -> None:
    setup = chat_setup()

    async def null(_args: Any, api: Any, _context: Any) -> Any:
        await api.details({"old": 1}, context)
        return {"output": [], "details": None}

    add_tool(setup.registry, tool("null", null))
    ran = await run(setup, [calls(("null", {}, "c1")), DONE])
    assert results(ran.entries)[0].details is None
    await ran.harness.close(context)


async def test_settles_details_futures_with_coalesced_progress_commits_and_the_terminal_commit() -> None:
    setup = chat_setup()
    settled: list[str] = []

    async def details(_args: Any, api: Any, _context: Any) -> Any:
        # Three updates in one throttle window coalesce; the last is still pending when execute() returns.
        async def tracked(value: Any, name: str) -> None:
            await api.details(value, context)
            settled.append(name)

        first = asyncio.ensure_future(tracked({"n": 1}, "first"))
        second = asyncio.ensure_future(tracked({"n": 2}, "second"))
        await first
        third = asyncio.ensure_future(tracked({"n": 3}, "third"))
        await asyncio.sleep(0)
        holder.extend([second, third])
        return {"output": []}

    holder: list[Any] = []
    add_tool(setup.registry, tool("details", details))
    ran = await run(setup, [calls(("details", {}, "c1")), DONE])
    await asyncio.gather(*holder)
    assert settled == ["first", "second", "third"]
    assert results(ran.entries)[0].details == {"n": 3}
    await ran.harness.close(context)


async def test_finishes_a_call_under_the_implementation_it_resolved_when_the_tool_is_replaced_mid_call() -> None:
    setup = chat_setup()
    started = deferred()
    finished = deferred()

    async def v1(*_args: Any) -> Any:
        resolve(started)
        await finished
        return output("v1")

    async def v2(*_args: Any) -> Any:
        return output("v2")

    add_tool(setup.registry, tool("work", v1))
    setup.faux.set_responses([calls(("work", {}, "c1")), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await started
    # The same extension name replaces the old one in place.
    add_tool(setup.registry, tool("work", v2))
    resolve(finished)
    await submission.wait(context)
    assert result_text(results(await all_entries(opened.root))[0]) == "v1"
    await opened.harness.close(context)


async def test_uses_a_section_and_hook_extension_reloaded_mid_run_from_the_runs_next_request() -> None:
    setup = chat_setup()
    requests: list[str] = []

    def prompt(version: str) -> Any:
        return define_extension(
            "prompt",
            sections=[section("mode", lambda *_: version)],
            hooks=[hook(GenerationTask, {"before_request": lambda *_: requests.append(version)})],
        )

    setup.registry.install(prompt("v1"))
    running = deferred()
    reloaded = deferred()

    async def work(*_args: Any) -> Any:
        resolve(running)
        await reloaded
        return {"output": []}

    add_tool(setup.registry, tool("work", work))
    setup.faux.set_responses([calls(("work", {}, "c1")), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await running
    setup.registry.install(prompt("v2"))
    resolve(reloaded)
    await submission.wait(context)
    sections = [
        entry["model"][0].sections
        for entry in await all_entries(opened.root)
        if entry.get("model")
        and isinstance(entry["model"][0], SystemMessage)
        and entry["model"][0].sections is not None
    ]
    assert sections == [{"mode": "<mode>\nv1\n</mode>"}, {"mode": "<mode>\nv2\n</mode>"}]
    assert requests == ["v1", "v2"]
    await opened.harness.close(context)


async def test_rejects_invocation_bound_waits_and_stops_watches_when_the_tools_invocation_ends() -> None:
    setup = chat_setup()
    never = define_task(
        name="test.never", version=1, initial=lambda _input: {"phase": "never"}, phases={"never": _never}, abort=_never
    )
    holder: dict[str, Any] = {}

    async def detach(_args: Any, api: Any, _context: Any) -> Any:
        # The child's definition is not registered, so it stays pending.
        child = await api.create_task(never, {}, {"ownership": {"kind": "conversation"}}, context)
        wait = asyncio.ensure_future(api.wait_for_task(child, context))
        wait.add_done_callback(lambda future: future.exception())
        holder["wait"] = wait
        watch = await api.watch_doc(LiveDoc, api.conversation_id, context)
        holder["closed"] = watch.closed
        return {"output": []}

    add_tool(setup.registry, tool("detach", detach))
    ran = await run(setup, [calls(("detach", {}, "c1")), DONE])
    with pytest.raises(Exception, match="invocation has ended"):
        await holder["wait"]
    await holder["closed"]
    await ran.harness.close(context)


async def test_rejects_details_still_waiting_when_the_call_is_aborted_during_after_tool() -> None:
    setup = chat_setup()
    holder: dict[str, Any] = {}
    in_after_tool = deferred()

    async def slow(_args: Any, api: Any, _context: Any) -> Any:
        api.output("first\n")
        # The output commit is in flight, so these details wait for the next throttle window.
        pending = api.details({"step": 1}, context)
        pending.add_done_callback(lambda future: future.cancelled() or future.exception())
        holder["details"] = pending
        return {}

    async def after_tool(_call: Any, _result: Any, _api: Any, call_context: Any) -> Any:
        resolve(in_after_tool)
        await aborted(call_context.abort_signal)
        return None

    add_tool(setup.registry, tool("slow", slow))
    add_hooks(setup.registry, ToolTask, {"after_tool": after_tool})
    setup.faux.set_responses([calls(("slow", {}, "c1")), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await in_after_tool
    task_id = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))["tools"][0]["taskId"]  # type: ignore[index]
    await opened.harness.abort_task(task_id, context)
    with pytest.raises(BaseException) as caught:
        await holder["details"]
    assert "has settled" not in str(caught.value)
    await submission.wait(context)
    await opened.harness.close(context)


async def test_answers_an_aborted_tool_with_only_its_durable_output_discarding_buffered_output() -> None:
    setup = chat_setup()
    buffered = deferred()

    async def slow(_args: Any, api: Any, call_context: Any) -> Any:
        api.output("durable\n")
        # The first output commits at once; this one waits for the next throttle window.
        await asyncio.sleep(0.02)
        api.output("buffered\n")
        resolve(buffered)
        await aborted(call_context.abort_signal)
        return {}

    add_tool(setup.registry, tool("slow", slow))
    setup.faux.set_responses([calls(("slow", {}, "c1")), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "go"}, context)
    await buffered
    task_id = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))["tools"][0]["taskId"]  # type: ignore[index]
    await opened.harness.abort_task(task_id, context)
    await submission.wait(context)
    assert result_text(results(await all_entries(opened.root))[0]) == (
        "durable\n|<harness>\n[error] Tool slow was aborted\n</harness>"
    )
    await opened.harness.close(context)


def test_output_limits_bound_text_given_as_pi_json() -> None:
    from pi_chord.output import OutputLimits
    from pi_durable.harness.tool import _bound_content

    content = [{"type": "text", "text": "x" * 100}, {"type": "image", "data": "AA==", "mimeType": "image/png"}]
    bounded, dropped_bytes, _ = _bound_content(content, OutputLimits(max_bytes=10, max_lines=100, retain="head"))
    assert dropped_bytes == 90
    assert isinstance(bounded[0], TextContent) and len(bounded[0].text.encode()) <= 10
    assert bounded[1].type == "image" and bounded[1].mime_type == "image/png"
