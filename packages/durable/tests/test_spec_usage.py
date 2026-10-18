"""Port of ``test/spec-usage.test.ts`` (the usage examples of ``docs/spec.md``).

Upstream only compile-checks the spec's examples; here each one runs: the Session sequences against a
Session, the Harness sequences against a Harness with the faux provider.

The coding tools of ``src/tools/`` (``CodingTools``, ``createReadTool``, ``createEditTool``,
``createBashTool``) are not ported in phase 1; the examples that use them (section 7.1 ``Timing`` and
``Venv``, the ``host`` and ``childInToolCommit`` sequences) run with plain stand-in tools of the same
names. ``liveSettings`` is a settings function (``HarnessOptions.settings`` may be one), Python's
equivalent of upstream's class with getters; ``containerEnv`` builds a ``LocalExecutionEnv`` for both
branches instead of a container.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import aborted, all_entries, chat_setup, open_chat, text_of, tool, wait_for
from durable_session_support import StorageBackend, context, create_conversation, open_test_session

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import SystemMessage, TextContent, ToolResultMessage
from pi_durable import (
    AgentDoc,
    AssistantEntry,
    GenerationTask,
    Harness,
    LiveDoc,
    MemoryStorage,
    ToolTask,
    configure,
    create_registry,
    define_doc,
    define_extension,
    define_task,
    define_tool,
    hook,
    section,
    wrap_tool,
)
from pi_durable.env.local import LocalExecutionEnv


async def _noop(*_args: Any) -> None:
    return None


Follow = define_task(
    name="app.follow", version=1, initial=lambda _input: {"phase": "follow"}, phases={"follow": _noop}, abort=_noop
)


def test_defines_the_section_5_1_payment_task() -> None:
    """Section 5.1: a task with intent, effect, outcome."""

    async def prepare(_task: Any, runtime: Any, context: Any) -> None:
        await runtime.commit(
            lambda tx, current: {"status": "running", "checkpoint": {"phase": "charge", "key": "k"}}, context
        )

    async def charge(task: Any, runtime: Any, context: Any) -> None:
        key = task["state"]["checkpoint"]["key"]  # idempotent by key

        async def outcome(tx: Any, current: Any) -> Any:
            entry = await tx.append_entry(current["conversationId"], {"kind": "app.receipt", "data": {"id": key}})
            return {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": entry["id"]}}}

        await runtime.commit(outcome, context)

    async def abort(_task: Any, runtime: Any, context: Any) -> None:
        # The abort handler decides the outcome; returning without one faults the task.
        await runtime.commit(
            lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted", "reason": "user"}}, context
        )

    Payment = define_task(
        name="app.payment",
        version=1,
        initial=lambda _input: {"phase": "prepare"},
        phases={"prepare": prepare, "charge": charge},
        abort=abort,
    )
    assert Payment.definition.initial(None) == {"phase": "prepare"}
    assert set(Payment.definition.phases) == {"prepare", "charge"}


async def test_section_4_table_reads_before_the_first_table_write_documents_stay_usable(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    async def seed(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["generation"] = {"step": 1}

    await session.commit(seed, context)

    async def change(tx: Any) -> None:
        conversation = await tx.conversation(conversation_id)  # table read
        live = await tx.doc(LiveDoc, conversation_id)

        await tx.append_entry(conversation_id, {"kind": "app.message"})  # first table write
        del live["generation"]  # document mutation remains valid
        await tx.create_task(
            Follow, {}, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
        )  # further table writes are fine
        assert conversation == {"id": conversation_id}

    await session.commit(change, context)
    assert await session.snapshot(LiveDoc, conversation_id, context) == {}
    kinds = sorted(write["type"] for write in storage.commits[-1])
    assert kinds == ["document.change", "entry", "task"]


async def test_section_3_4_drafts_are_revoked_after_their_commit(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    escaped: list[Any] = []

    async def change(tx: Any) -> None:
        escaped.append(await tx.doc(LiveDoc, conversation_id))

    await session.commit(change, context)
    with pytest.raises(TypeError):  # raises: the draft was revoked
        escaped[0]["generation"] = None


# ─── Harness sequences ──────────────────────────────────────────────────────

SONNET = {"provider": "faux", "modelId": "faux-1"}
#: The faux provider has one model; the cheaper model of the spec is the same one under another name here.
HAIKU = {"provider": "faux", "modelId": "faux-1"}


@dataclasses.dataclass
class Offered:
    tools: list[str]
    sections: dict[str, str | None]


def offered(request: Any) -> Offered:
    """The tools and sections in effect for a request, replaying its system messages."""
    tools: list[str] = []
    sections: dict[str, str | None] = {}
    for message in request.messages:
        if not isinstance(message, SystemMessage):
            continue
        for name in message.tools_removed or ():
            if name in tools:
                tools.remove(name)
        tools.extend(added.name for added in message.tools_added or ())
        for key, value in (message.sections or {}).items():
            if value is None:
                sections.pop(key, None)
            else:
                sections[key] = value
    return Offered(tools, sections)


def results_of(entries: list[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if entry["kind"] == "pi.tool-result"]


async def test_sections_2_2_and_7_host_setup_tool_filters_extension_selection_plan_mode_and_reload(
    tmp_path: Path,
) -> None:
    setup = chat_setup()
    registry = setup.registry
    # Stand-ins for createReadTool(), createEditTool() and createBashTool().
    read_tool, edit_tool, bash_tool = tool("read"), tool("edit"), tool("bash")
    CodingTools = define_extension("coding-tools", tools=[read_tool, edit_tool, bash_tool])

    # Section 7.1: extensions and host setup.
    ContextFiles = define_extension("context-files", sections=[section("agents-md", lambda _input, _c: "Use tabs.")])
    Skills = define_extension("skills", sections=[section("skills", lambda _input, _c: "skills v1")])
    SkillsV2 = define_extension("skills", sections=[section("skills", lambda _input, _c: "skills v2")])
    Coding = define_extension(
        "coding",
        sections=[
            section("preamble", lambda _input, _c: "You are an expert coding assistant.", tag=False),
            # The environment the host built for this conversation, in the conversation's directory.
            section("cwd", lambda input, _c: input["env"] and f"Working directory: {input['env'].cwd}"),
        ],
    )
    dangerous: list[str] = []

    def is_dangerous(call: Any) -> bool:
        return call.name == "danger"

    async def permissions(call: Any, _api: Any, _context: Any) -> Any:
        dangerous.append(call.name)
        return {"block": "Needs approval"} if is_dangerous(call) else None

    Permissions = define_extension("permissions", hooks=[hook(ToolTask, {"before_tool": permissions})])
    # A role and a review loop, for conversations that select it.
    Reviewer = define_extension(
        "reviewer",
        sections=[section("role", lambda _input, _c: "You review diffs. Report problems as a list. Never edit files.")],
        hooks=[hook(GenerationTask, {"on_yield": lambda *_: {"continue": "second pass"}})],
    )
    metrics: list[tuple[str, float]] = []

    def timing(wrapped: Any) -> Any:
        async def execute(args: Any, api: Any, call_context: Any) -> Any:
            start = time.monotonic()
            try:
                return await wrapped.execute(args, api, call_context)
            finally:
                metrics.append(("bash", time.monotonic() - start))

        return dataclasses.replace(wrapped, execute=execute)

    Timing = define_extension("timing", wraps=[wrap_tool(bash_tool, timing)])

    async def venv_bash(_args: Any, _api: Any, _context: Any) -> Any:
        return {"output": [TextContent("in .venv")]}

    # A bash inside a Python virtualenv for one conversation: it replaces CodingTools' bash in place,
    # and Timing, if selected, wraps it.
    Venv = define_extension("venv", tools=[define_tool(name="bash", description="bash in .venv", execute=venv_bash)])

    # Section 7.2: hooks reading extension state.
    PlanModeDoc = define_doc(
        kind="app.plan-mode",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda: {"enabled": False},
    )

    def writes(call: Any) -> bool:
        return call.name == "edit"

    async def plan_mode(call: Any, api: Any, call_context: Any) -> Any:
        # An absent document means plan mode is off.
        state = await api.snapshot(PlanModeDoc, api.conversation_id, call_context)
        return {"block": "Plan mode: read-only"} if state is not None and state["enabled"] and writes(call) else None

    PlanMode = define_extension("plan-mode", hooks=[hook(ToolTask, {"before_tool": plan_mode})])

    for extension in (CodingTools, Coding, ContextFiles, Skills, Permissions, Reviewer):
        registry.install(extension)
    harness = await Harness.open(
        MemoryStorage(),
        {
            "models": setup.models,
            "registry": registry,
            # Reviewer is installed but not selected by default: only conversations that select it get its role
            # and hooks.
            "settings": {"extensions": [CodingTools, Coding, ContextFiles, Skills, Permissions, PlanMode]},
            "env": lambda target, _c: LocalExecutionEnv(cwd=target.get("cwd") or os.getcwd()),
        },
        context,
    )
    # The conversation remembers its model and directory; a restart elsewhere keeps both.
    root = await harness.root(context, agent={"model": SONNET, "cwd": str(tmp_path)})

    async def stored_tools() -> Any:
        return (await harness.snapshot(AgentDoc, root.id, context)).get("tools")

    await root.configure({"tools": {"remove": [edit_tool]}}, context)
    assert await stored_tools() == {"remove": ["edit"]}
    await root.configure({"tools": {"remove": [bash_tool]}}, context)  # edit is offered again
    assert await stored_tools() == {"remove": ["bash"]}
    await root.configure({"tools": None}, context)  # every tool of the selected extensions again
    assert await stored_tools() is None

    registry.install(Timing)
    registry.install(Venv)  # installed, but not in the default selection
    conversation = root
    await conversation.configure({"extensions": {"add": [Venv]}}, context)

    registry.install(PlanMode)
    # PlanMode is in the default selection; /plan toggles this conversation's state.

    async def enable(tx: Any) -> None:
        (await tx.doc(PlanModeDoc, root.id))["enabled"] = True

    await root.commit(enable, context)

    requests: list[Offered] = []

    def record(answer: Any) -> Any:
        def step(request: Any, *_rest: Any) -> Any:
            requests.append(offered(request))
            return answer

        return step

    setup.faux.set_responses(
        [
            record(
                faux_assistant_message(
                    [faux_tool_call("edit", {}, id="e1"), faux_tool_call("bash", {}, id="b1")], stop_reason="toolUse"
                )
            ),
            record(faux_assistant_message([faux_text("done")])),
            record(faux_assistant_message([faux_tool_call("bash", {}, id="b2")], stop_reason="toolUse")),
            record(faux_assistant_message([faux_text("again")])),
            record(faux_assistant_message([faux_text("bye")])),
        ]
    )
    settled = await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    assert requests[0].tools == ["read", "edit", "bash"]
    assert requests[0].sections == {
        "preamble": "You are an expert coding assistant.",
        "cwd": f"<cwd>\nWorking directory: {tmp_path}\n</cwd>",
        "agents-md": "<agents-md>\nUse tabs.\n</agents-md>",
        "skills": "<skills>\nskills v1\n</skills>",
    }
    edit_result, bash_result = results_of(await all_entries(root))
    assert edit_result.is_error and "Tool call blocked: Plan mode: read-only" in (text_of(edit_result) or "")
    # Venv's bash; Timing is installed but not selected.
    assert not bash_result.is_error and text_of(bash_result) == "in .venv"
    assert metrics == []
    assert dangerous == ["edit", "bash"]
    await conversation.configure({"extensions": {"add": [Venv, Timing]}}, context)

    registry.install(Chat)
    registry.install(SkillsV2)  # same name: conversations selecting skills render v2 at their next request
    await (await root.submit({"type": "input", "content": "more"}, context)).wait(context)
    assert requests[-1].sections["skills"] == "<skills>\nskills v2\n</skills>"
    # Selected now, Timing wraps Venv's bash.
    assert text_of(results_of(await all_entries(root))[-1]) == "in .venv"
    assert [name for name, _ms in metrics] == ["bash"]
    registry.uninstall(Skills)  # selecting conversations get a system delta removing its section; nothing is rewritten
    await (await root.submit({"type": "input", "content": "last"}, context)).wait(context)
    assert "skills" not in requests[-1].sections
    await harness.close(context)


# Section 7.4.
Chat = define_extension(
    "chat", sections=[section("preamble", lambda _input, _c: "You are a helpful assistant.", tag=False)]
)


async def test_section_7_3_a_subagent_tool_delegates_to_a_child_conversation() -> None:
    setup = chat_setup()

    async def answer_text(api: Any, entry: Any, call_context: Any) -> str:
        found = await api.commit(lambda tx: tx.entry(AssistantEntry, entry), call_context)
        return text_of(found["model"][0]) or ""

    async def execute(args: Any, api: Any, call_context: Any) -> Any:
        task = args["task"]

        async def spawn(tx: Any) -> Any:
            existing = (await tx.scan_conversations({"ownerTaskId": api.task_id}, 1))["items"]
            if existing:
                return existing[0]["id"]
            # Starts as a copy of this conversation's agent: model, thinking level, cwd, extensions, tools.
            created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            # Without this extension, the child is not offered this tool.
            await configure(tx, created["id"], {"extensions": {"remove": [Subagent]}})
            return created["id"]

        child = await api.commit(spawn, call_context)
        await api.details({"conversationId": child}, call_context)
        handle = await api.conversation(child, call_context)
        request = {"type": "input", "content": task, "requestId": f"subagent:{api.task_id}"}
        settled = await (await handle.submit(request, call_context)).wait(call_context)
        if settled["status"] != "done" or settled["type"] != "input":
            raise RuntimeError(f"Subagent failed: {settled['status']}")
        return {"output": [TextContent(await answer_text(api, settled["answer"], call_context))]}

    Subagent = define_extension(
        "subagent",
        tools=[
            define_tool(
                name="subagent",
                description="Delegate a self-contained task to a subagent and get its answer back.",
                parameters=Type.Object({"task": Type.String()}),
                replay="safe",
                execute=execute,
            )
        ],
    )
    setup.registry.install(Subagent)
    child_tools: list[list[str]] = []

    def child_answer(request: Any, *_rest: Any) -> Any:
        child_tools.append(offered(request).tools)
        return faux_assistant_message([faux_text("child says hi")])

    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("subagent", {"task": "greet"}, id="s1")], stop_reason="toolUse"),
            child_answer,
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    settled = await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert settled["status"] == "done"
    (result,) = results_of(await all_entries(root))
    assert text_of(result) == "child says hi"
    assert child_tools == [[]]
    children = await harness.commit(lambda tx: tx.scan_conversations({}, 10), context)
    assert len(children["items"]) == 2
    await harness.close(context)


async def test_section_7_3_a_named_subagents_spawn_commit() -> None:
    async def hold(*_args: Any) -> None:
        return None

    Anchor = define_task(
        name="app.anchor", version=1, initial=lambda _input: {"phase": "hold"}, phases={"hold": hold}, abort=hold
    )
    Reporter = define_task(
        name="app.reporter", version=1, initial=lambda _input: {"phase": "report"}, phases={"report": hold}, abort=hold
    )
    SubagentTools = define_extension("subagent-tools", tasks=[Anchor, Reporter], tools=[tool("subagent")])
    setup = chat_setup()
    setup.registry.install(SubagentTools)
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"extensions": [SubagentTools]}, context)
    name = "scout"

    async def spawn(tx: Any) -> Any:
        # In subagentTool's spawn commit, after the name checks:
        anchor = await tx.create_task(Anchor, None, {"ownership": {"kind": "conversation"}, "background": True})
        # Owned by a task of the parent: starts as a copy of the parent's agent.
        child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": anchor}})
        await configure(
            tx,
            child["id"],
            {
                "extensions": {"remove": [SubagentTools]},
                "instructions": f'You are the subagent "{name}". Answer the main agent\'s requests.',
            },
        )
        return child["id"]

    child = await root.commit(spawn, context)
    agent = await harness.snapshot(AgentDoc, child, context)
    assert agent["extensions"] == {"remove": ["subagent-tools"]}
    assert agent["instructions"] == 'You are the subagent "scout". Answer the main agent\'s requests.'
    assert agent["model"] == {"provider": "faux", "modelId": "faux-1"}
    await harness.close(context)


async def test_section_2_2_a_tools_commit_creates_a_configured_child() -> None:
    setup = chat_setup()
    read_tool = tool("read")  # stand-in for createReadTool()
    worktree = "/tmp/worktree"
    created: list[Any] = []

    async def spawn(_args: Any, api: Any, call_context: Any) -> Any:
        async def child_in_tool_commit(tx: Any) -> None:
            # In a tool's commit. The child starts as a copy of this conversation's agent: model, extensions, tools, cwd.
            child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            # A cheaper model, only the read tool, and its own worktree; everything else stays as copied.
            await configure(tx, child["id"], {"model": HAIKU, "tools": [read_tool], "cwd": worktree})
            created.append(child["id"])

        await api.commit(child_in_tool_commit, call_context)
        return {"output": []}

    setup.registry.install(
        define_extension("tools", tools=[read_tool, define_tool(name="spawn", description="Spawns", execute=spawn)])
    )
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("spawn", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"instructions": "Be brief.", "thinkingLevel": "low"}, context)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    agent = await harness.snapshot(AgentDoc, created[0], context)
    assert agent["model"] == HAIKU and agent["tools"] == ["read"] and agent["cwd"] == worktree
    # Copied from the parent.
    assert agent["instructions"] == "Be brief." and agent["thinkingLevel"] == "low"
    await harness.close(context)


async def test_section_2_2_settings_read_live() -> None:
    class SettingsManager:
        def __init__(self) -> None:
            self.values: dict[str, Any] = {"timeoutMs": 30_000, "autoCompact": True}

        def get(self, key: str) -> Any:
            return self.values[key]

        def set(self, key: str, value: Any) -> None:
            self.values[key] = value

    manager = SettingsManager()

    # The user's settings, read live: a function the Harness calls at each use.
    def user_settings() -> Any:
        return {
            "stream": {"timeoutMs": manager.get("timeoutMs")},
            "compaction": {"enabled": manager.get("autoCompact")},
        }

    setup = chat_setup()
    timeouts: list[Any] = []

    def step(_request: Any, options: Any, *_rest: Any) -> Any:
        timeouts.append(options.get("timeout_ms"))
        return faux_assistant_message([faux_text("ok")])

    setup.faux.set_responses([step, step])
    harness = await Harness.open(
        MemoryStorage(), {"models": setup.models, "registry": setup.registry, "settings": user_settings}, context
    )
    root = await harness.root(context, agent={"model": SONNET})
    await (await root.submit({"type": "input", "content": "a"}, context)).wait(context)
    manager.set("timeoutMs", 5_000)  # no Session write; the next request reads it
    manager.set("autoCompact", False)  # every conversation follows at its next threshold check
    await (await root.submit({"type": "input", "content": "b"}, context)).wait(context)
    assert timeouts == [30_000, 5_000]
    await harness.close(context)


async def test_section_2_2_an_environment_per_conversation(tmp_path: Path) -> None:
    setup = chat_setup()
    registry = create_registry()
    # Absent: the conversation runs locally. Only conversations with this document run in a container.
    # Subagents do not copy it: their creator writes it too when they should run in the container.
    ContainerDoc = define_doc(
        kind="app.container",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda: {"image": "node:22"},
    )
    built: list[tuple[Any, str | None]] = []

    async def env(target: Any, call_context: Any) -> Any:
        container = await target["read"].snapshot(ContainerDoc, target["conversationId"], call_context)
        image = None if container is None else container["image"]
        built.append((target["conversationId"], image))
        # A container's environment in the spec; a local one for both branches here.
        return LocalExecutionEnv(cwd=target.get("cwd") or str(tmp_path))

    seen: list[str] = []

    async def where(_args: Any, api: Any, _context: Any) -> Any:
        seen.append(api.env.cwd)
        return {"output": []}

    registry.install(define_extension("where", tools=[define_tool(name="where", description="cwd", execute=where)]))
    setup.models.set_provider(setup.faux.provider)
    harness = await Harness.open(MemoryStorage(), {"models": setup.models, "registry": registry, "env": env}, context)
    root = await harness.root(context, agent={"model": SONNET})

    async def contain(tx: Any) -> None:
        await tx.doc(ContainerDoc, root.id)

    await root.commit(contain, context)
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("where", {}, id="w1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert seen == [str(tmp_path)]
    assert built and all(entry == (root.id, "node:22") for entry in built)
    await harness.close(context)


async def test_section_5_1_runs_the_payment_task_to_its_receipt_and_aborts_one_through_its_handler() -> None:
    charged: list[str] = []
    cancelled: list[Any] = []
    keys = iter(["k1", "k2"])

    async def prepare(_task: Any, runtime: Any, call_context: Any) -> None:
        key = next(keys)
        await runtime.commit(
            lambda tx, current: {"status": "running", "checkpoint": {"phase": "charge", "key": key}}, call_context
        )

    async def charge(task: Any, runtime: Any, call_context: Any) -> None:
        key = task["state"]["checkpoint"]["key"]
        if key == "k2":
            await aborted(call_context.abort_signal)  # held until aborted
        charged.append(key)  # idempotent by key
        receipt = {"id": f"receipt-{key}"}

        async def outcome(tx: Any, current: Any) -> Any:
            entry = await tx.append_entry(current["conversationId"], {"kind": "app.receipt", "data": receipt})
            return {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": entry["id"]}}}

        await runtime.commit(outcome, call_context)

    async def abort(task: Any, runtime: Any, call_context: Any) -> None:
        # The abort handler decides the outcome; returning without one faults the task.
        cancelled.append(task["state"]["checkpoint"])
        await runtime.commit(
            lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted", "reason": "user"}}, call_context
        )

    Payment = define_task(
        name="app.payment",
        version=1,
        initial=lambda _input: {"phase": "prepare"},
        phases={"prepare": prepare, "charge": charge},
        abort=abort,
    )
    setup = chat_setup()
    setup.registry.install(define_extension("payments", tasks=[Payment]))
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    paid = await root.commit(lambda tx: tx.create_task(Payment, None, {"ownership": {"kind": "conversation"}}), context)
    settled = await harness.wait_for_task(paid, context)
    entry_id = settled["state"]["outcome"]["result"]["entryId"]
    assert [entry["data"] for entry in await all_entries(root) if entry["id"] == entry_id] == [{"id": "receipt-k1"}]

    held = await root.commit(lambda tx: tx.create_task(Payment, None, {"ownership": {"kind": "conversation"}}), context)

    async def charging() -> bool:
        task = await harness.get_task(held, context)
        return task is not None and task["state"].get("checkpoint", {}).get("phase") == "charge"

    await wait_for(charging)
    await harness.abort_task(held, context)
    stopped = await harness.wait_for_task(held, context)
    assert stopped["state"]["outcome"] == {"status": "aborted", "reason": "user"}
    assert cancelled == [{"phase": "charge", "key": "k2"}]
    assert charged == ["k1"]
    await harness.close(context)
