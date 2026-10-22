"""Hook seams beyond upstream's (from delta): ``on_input``, counted and capped ``on_yield`` continuations,
``before_tool`` context, and what a hook reaches through its api; ``exec`` with standard input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from durable_harness_support import all_entries, chat_setup, open_chat
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import SystemMessage, TextContent, ToolResultMessage
from pi_durable import (
    AgentDoc,
    ContextEntry,
    GenerationTask,
    LiveDoc,
    MemoryStorage,
    ToolResultEntry,
    ToolTask,
    define_extension,
    define_tool,
    hook,
)
from pi_durable.env.local import LocalExecutionEnv


def answer(text: str) -> Any:
    return faux_assistant_message([faux_text(text)])


def recording(text: str, seen: list[Any]) -> Any:
    """A faux step answering ``text`` that records the request's messages."""

    def step(request_context: Any, _options: Any, _state: Any, _model: Any) -> Any:
        seen.append(list(request_context.messages))
        return answer(text)

    return step


def user_texts(run_input: Any) -> list[str]:
    texts: list[str] = []
    for entry in run_input["entries"]:
        for message in entry.get("model") or []:
            if message.role == "user":
                content = message.content
                texts.append(content if isinstance(content, str) else "".join(part.text for part in content))
    return texts


class Gate:
    """``on_input``: blocks input containing "forbidden", adds context otherwise."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def on_input(self, run_input: Any, _api: Any, _context: Any) -> Any:
        texts = user_texts(run_input)
        self.calls.append(texts)
        if any("forbidden" in text for text in texts):
            return {"block": "no forbidden words"}
        return {"context": f"checked {','.join(texts)}"}


async def test_on_input_blocks_a_run() -> None:
    setup = chat_setup()
    gate = Gate()
    setup.registry.install(define_extension("gate", hooks=[hook(GenerationTask, {"on_input": gate.on_input})]))
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "say something forbidden"}, context)
    settled = await submission.wait(context)
    assert (settled["status"], settled["reason"], settled["detail"]) == ("unanswered", "blocked", "no forbidden words")
    assert setup.faux.state.call_count == 0
    assert [entry["kind"] for entry in await all_entries(opened.root)] == ["pi.user"]
    live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
    assert live is not None
    assert "run" not in live
    await opened.harness.close(context)


async def test_on_input_context_is_added_once_per_run() -> None:
    setup = chat_setup()
    gate = Gate()

    async def read(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("file")]}

    setup.registry.install(
        define_extension(
            "gate",
            tools=[define_tool(name="read", description="Read", parameters=Type.Object({}), execute=read)],
            hooks=[hook(GenerationTask, {"on_input": gate.on_input})],
        )
    )
    seen: list[Any] = []
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("read", {}, id="c1")], stop_reason="toolUse"),
            recording("answer", seen),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    settled = await (await opened.root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    assert settled["status"] == "done"
    # The successor generation after the tool round does not run the hooks again.
    assert gate.calls == [["hi"]]
    entries = await all_entries(opened.root)
    assert [entry["kind"] for entry in entries[:2]] == ["pi.user", ContextEntry.kind]
    [message] = entries[1]["model"]
    assert isinstance(message, SystemMessage)
    assert message.content == "checked hi"
    assert any(isinstance(item, SystemMessage) and item.content == "checked hi" for item in seen[0])
    live = await opened.harness.snapshot(LiveDoc, opened.root.id, context)
    assert live is not None
    assert "run" not in live
    await opened.harness.close(context)


async def test_on_input_without_hooks_commits_nothing() -> None:
    setup = chat_setup()
    setup.faux.set_responses([answer("hello")])
    opened = await open_chat(MemoryStorage(), setup)
    submission = await opened.root.submit({"type": "input", "content": "hi"}, context)
    await submission.wait(context)
    assert ContextEntry.kind not in [entry["kind"] for entry in await all_entries(opened.root)]
    await opened.harness.close(context)


async def test_continuations_are_counted_and_capped() -> None:
    setup = chat_setup()
    setup.settings["maxContinuations"] = 2
    seen: list[int] = []

    async def persistent(_answer: Any, api: Any, call_context: Any) -> Any:
        live = await api.snapshot(LiveDoc, api.conversation_id, call_context)
        seen.append(live["run"].get("continuations", 0))
        return {"continue": "again"}

    setup.registry.install(define_extension("persistent", hooks=[hook(GenerationTask, {"on_yield": persistent})]))
    setup.faux.set_responses([answer("one"), answer("two"), answer("three")])
    opened = await open_chat(MemoryStorage(), setup)
    settled = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    assert seen == [0, 1, 2]
    assert setup.faux.state.call_count == 3
    assert "The run took 2 continuations; the next one is ignored" in [str(error) for error in setup.reports]
    await opened.harness.close(context)


def note(text: str) -> Any:
    return define_extension(f"note-{text}", hooks=[hook(ToolTask, {"before_tool": lambda *_args: {"context": text}})])


async def test_before_tool_context_follows_the_result() -> None:
    setup = chat_setup()

    async def echo(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("hi")]}

    def redact(_call: Any, result: Any, _api: Any, _context: Any) -> Any:
        return {**result, "output": [TextContent("[redacted]")]}

    setup.registry.install(
        define_extension(
            "tools", tools=[define_tool(name="echo", description="Echo", parameters=Type.Object({}), execute=echo)]
        )
    )
    setup.registry.install(note("note one"))
    setup.registry.install(note("note two"))
    setup.registry.install(define_extension("redact", hooks=[hook(ToolTask, {"after_tool": redact})]))
    setup.faux.set_responses(
        [faux_assistant_message([faux_tool_call("echo", {}, id="c1")], stop_reason="toolUse"), answer("done")]
    )
    opened = await open_chat(MemoryStorage(), setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    [result] = [entry["model"][0] for entry in await all_entries(opened.root) if ToolResultEntry.is_(entry)]
    assert isinstance(result, ToolResultMessage)
    # After ``after_tool``, which cannot remove it.
    assert [item.text for item in result.content if isinstance(item, TextContent)] == [
        "[redacted]",
        "note one\n\nnote two",
    ]
    await opened.harness.close(context)


async def test_hooks_reach_the_environment_and_documents(tmp_path: Path) -> None:
    setup = chat_setup()
    seen: dict[str, Any] = {}

    async def echo(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("hi")]}

    async def probe(_call: Any, api: Any, call_context: Any) -> None:
        seen["cwd"] = await api.cwd(call_context)
        seen["env"] = await api.env(call_context) is not None
        seen["agent"] = await api.snapshot(AgentDoc, api.conversation_id, call_context) is not None
        seen["record"] = (await api.conversation_record(call_context))["id"]
        seen["model"] = (await api.agent(call_context))["model"]

    setup.registry.install(
        define_extension(
            "tools",
            tools=[define_tool(name="echo", description="Echo", parameters=Type.Object({}), execute=echo)],
            hooks=[hook(ToolTask, {"before_tool": probe})],
        )
    )
    setup.faux.set_responses(
        [faux_assistant_message([faux_tool_call("echo", {}, id="c1")], stop_reason="toolUse"), answer("done")]
    )
    opened = await open_chat(MemoryStorage(), setup, env=LocalExecutionEnv(cwd=str(tmp_path)))
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert seen == {
        "cwd": str(tmp_path),
        "env": True,
        "agent": True,
        "record": opened.root.id,
        "model": {"provider": "faux", "modelId": "faux-1"},
    }
    await opened.harness.close(context)


async def test_exec_writes_standard_input(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    chunks: list[tuple[str, str]] = []
    options: Any = {"stdin": '{"a": 1}', "onOutput": lambda text, _context, info: chunks.append((info["stream"], text))}
    result = await env.exec("cat; echo done >&2", options, context)
    assert result["ok"] is True
    assert result["value"]["exitCode"] == 0
    assert "".join(text for stream, text in chunks if stream == "stdout") == '{"a": 1}'
    assert "".join(text for stream, text in chunks if stream == "stderr") == "done\n"
    empty = await env.exec("cat", {"onOutput": lambda text, _context, _info: chunks.append(("late", text))}, context)
    assert empty["ok"] is True
    assert not any(stream == "late" for stream, _text in chunks)
