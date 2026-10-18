"""Port of ``test/harness-tasks.test.ts``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from durable_harness_support import (
    aborted,
    aborted_with,
    add_hooks,
    add_task,
    add_tool,
    completed,
    eventually,
    open_tasks,
    settled,
    user,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, flush, resolve

from pi_ai import AbortController, Type, create_models
from pi_chord.context import Context, with_abort_signal, with_cancel
from pi_durable import (
    Harness,
    MemoryStorage,
    SessionFailed,
    create_registry,
    define_doc,
    define_entry,
    define_task,
    define_tool,
)


def _default_abort(runtime: Any, ctx: Context) -> Any:
    return runtime.commit(lambda _tx, _current: aborted_with("test"), ctx)


def one_step(name: str, run: Any, abort: Any = None) -> Any:
    """A one-phase task; the default abort handler settles ``aborted`` with reason "test"."""
    abort_handler = abort if abort is not None else _default_abort

    async def phase(task: Any, runtime: Any, ctx: Context) -> None:
        await run(task, runtime, ctx)

    async def on_abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await abort_handler(runtime, ctx)

    return define_task(
        name=name, version=1, initial=lambda _input: {"phase": "run"}, phases={"run": phase}, abort=on_abort
    )


def gated(name: str, gate: asyncio.Future[Any]) -> Any:
    """A one-phase task that waits for ``gate`` and completes with ``None``."""

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    return one_step(name, run)


async def start(conversation: Any, task: Any, *, background: bool | None = None) -> Any:
    options: Any = {"ownership": {"kind": "conversation"}}
    if background is not None:
        options["background"] = background
    return await conversation.commit(lambda tx: tx.create_task(task, None, options), context)


@dataclass(slots=True)
class OpenedRoot:
    harness: Harness
    registry: Any
    reports: list[BaseException]
    infos: list[Any]
    root: Any


async def open_root(tasks: list[Any], *, storage: Any = None, **options: Any) -> OpenedRoot:
    opened = await open_tasks(storage if storage is not None else MemoryStorage(), tasks, **options)
    root = await opened.harness.root(context)
    return OpenedRoot(opened.harness, opened.registry, opened.reports, opened.infos, root)


def controlled() -> ControlledStorage:
    return ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])


async def mark_durably(harness: Harness, id: Any) -> asyncio.Future[Any]:
    """Start ``abort_task()`` and return once its mark is durable, before it has joined the run."""
    aborting = harness.abort_task(id, context)
    while not ((await harness.get_task(id, context)) or {}).get("abortRequested"):
        await flush()
    return aborting


def with_signal(signal: Any) -> Context:
    return with_abort_signal(signal, context)


async def _nothing(*_args: Any) -> None:
    return None


def _running(checkpoint: Any) -> Any:
    return {"status": "running", "checkpoint": checkpoint}


# ─── task phases ────────────────────────────────────────────────────────────


async def test_continues_one_invocation_through_checkpoint_progress_and_completes_with_a_typed_result() -> None:
    seen: list[int] = []
    runtimes: set[int] = set()

    async def count(task: Any, runtime: Any, ctx: Context) -> None:
        seen.append(task["state"]["checkpoint"]["n"])
        runtimes.add(id(runtime))

        def change(_tx: Any, current: Any) -> Any:
            n = current["state"]["checkpoint"]["n"] + 1
            return completed(n) if n == task["input"]["to"] else _running({"phase": "count", "n": n})

        await runtime.commit(change, ctx)

    Counter = define_task(
        name="test.counter",
        version=1,
        initial=lambda _input: {"phase": "count", "n": 0},
        phases={"count": count},
        abort=_nothing,
    )
    opened = await open_root([Counter])
    harness, root = opened.harness, opened.root
    task_id = await root.commit(
        lambda tx: tx.create_task(Counter, {"to": 3}, {"ownership": {"kind": "conversation"}}), context
    )
    harness.resume()
    receipt = await harness.wait_for_task(task_id, context)
    outcome = receipt["state"]["outcome"]
    assert outcome["status"] == "completed" and outcome["result"] == 3
    assert seen == [0, 1, 2]
    assert len(runtimes) == 1
    await harness.close(context)


async def test_faults_a_phase_without_durable_progress_and_a_raising_phase() -> None:
    async def idle(*_args: Any) -> None:
        return None

    async def document_only(_task: Any, runtime: Any, ctx: Context) -> None:
        # A commit that returns no state is not progress.
        await runtime.commit(lambda _tx, _current: None, ctx)

    async def throws(*_args: Any) -> None:
        raise RuntimeError("boom")

    Idle = one_step("test.idle", idle)
    DocumentOnly = one_step("test.document-only", document_only)
    Throws = one_step("test.throws", throws)
    opened = await open_root([Idle, DocumentOnly, Throws])
    harness, root = opened.harness, opened.root
    ids = [await start(root, Idle), await start(root, DocumentOnly), await start(root, Throws)]
    harness.resume()
    outcomes = [(await harness.wait_for_task(id, context))["state"]["outcome"] for id in ids]
    assert outcomes == [
        {"status": "faulted", "error": {"message": "Task test.idle phase run returned without durable progress"}},
        {
            "status": "faulted",
            "error": {"message": "Task test.document-only phase run returned without durable progress"},
        },
        {"status": "faulted", "error": {"message": "boom"}},
    ]
    await harness.close(context)


async def test_keeps_a_committed_terminal_outcome_when_the_handler_raises_afterwards() -> None:
    late: list[Any] = []

    async def done(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed("ok"), ctx)
        try:
            await runtime.commit(lambda _tx, _current: None, ctx)
        except Exception as error:
            late.append(error)
        raise RuntimeError("after terminal")

    Done = one_step("test.done", done)
    opened = await open_root([Done])
    harness, root = opened.harness, opened.root
    id = await start(root, Done)
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "completed", "result": "ok"}
    await eventually(lambda: len(late) > 0)
    assert f"Task {id} is terminal" in str(late[0])
    await harness.close(context)


async def test_compares_checkpoints_by_value_including_lists() -> None:
    async def collect(task: Any, runtime: Any, ctx: Context) -> None:
        items = task["state"]["checkpoint"]["items"]
        # Two rounds of progress, then an equal copy of the checkpoint, which is no progress.
        following = [*items, f"item{len(items)}"] if len(items) < 2 else list(items)
        await runtime.commit(lambda _tx, _current: _running({"phase": "collect", "items": following}), ctx)

    Collect = define_task(
        name="test.collect",
        version=1,
        initial=lambda _input: {"phase": "collect", "items": []},
        phases={"collect": collect},
        abort=_nothing,
    )
    opened = await open_root([Collect])
    harness, root = opened.harness, opened.root
    id = await root.commit(lambda tx: tx.create_task(Collect, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "faulted",
        "error": {"message": "Task test.collect phase collect returned without durable progress"},
    }
    await harness.close(context)


async def test_commits_results_with_entries_atomically_keeps_memos_until_terminal_and_retires_task_documents() -> None:
    Progress = define_doc(kind="test.task-progress", version=1, scope="task", initial=lambda: {"lines": []})
    Answer = define_entry("answer")

    async def child_run(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Child = one_step("test.child", child_run)
    child_ids: list[Any] = []
    progress_seen: list[Any] = []

    async def write(task: Any, runtime: Any, ctx: Context) -> None:
        assert await runtime.memo("choice", ctx) is None
        winners = await asyncio.gather(runtime.memo("choice", "a", ctx), runtime.memo("choice", "b", ctx))
        assert list(winners) == ["a", "a"]
        assert await runtime.memo("choice", ctx) == "a"
        # Memo names never resolve to inherited object properties.
        assert await runtime.memo("toString", ctx) is None
        assert await runtime.memo("toString", "own", ctx) == "own"

        async def change(tx: Any, _current: Any) -> Any:
            (await tx.doc(Progress, task["id"]))["lines"].append("wrote")
            # Task creation defaults to the task's own conversation.
            child_ids.append(await tx.create_task(Child, None, {"ownership": {"kind": "conversation"}}))
            return _running({"phase": "answer"})

        await runtime.commit(change, ctx)

    async def answer(task: Any, runtime: Any, ctx: Context) -> None:
        assert task["memos"] == {"choice": "a", "toString": "own"}

        async def read(tx: Any, _current: Any) -> None:
            progress_seen.extend((await tx.doc(Progress, task["id"]))["lines"])

        await runtime.commit(read, ctx)

        async def complete(tx: Any, current: Any) -> Any:
            entry = await tx.append_entry(current["conversationId"], {"kind": "answer", "model": [user("done")]})
            return completed({"entryId": entry["id"]})

        await runtime.commit(complete, ctx)

    Writer = define_task(
        name="test.writer",
        version=1,
        initial=lambda _input: {"phase": "write"},
        phases={"write": write, "answer": answer},
        abort=_nothing,
    )
    opened = await open_root([Writer, Child])
    harness, root = opened.harness, opened.root
    conversation = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    id = await conversation.commit(
        lambda tx: tx.create_task(Writer, None, {"ownership": {"kind": "conversation"}}), context
    )
    harness.resume()
    receipt = await harness.wait_for_task(id, context)
    assert "checkpoint" not in receipt["state"]
    assert "memos" not in receipt
    outcome = receipt["state"]["outcome"]
    assert outcome["status"] == "completed"
    entry = await harness.commit(lambda tx: tx.entry(outcome["result"]["entryId"]), context)
    assert Answer.is_(entry) is True
    assert entry["conversationId"] == conversation.id
    assert progress_seen == ["wrote"]
    assert await harness.snapshot(Progress, id, context) is None
    assert await harness.get_task(id, context) == receipt
    assert (await harness.wait_for_task(child_ids[0], context))["conversationId"] == conversation.id
    assert root.id != conversation.id
    await harness.close(context)


def _waiter_task(name: str, on: list[Any], order: list[str] | None = None, outcomes: list[str] | None = None) -> Any:
    async def wait(_task: Any, runtime: Any, ctx: Context) -> None:
        if order is not None:
            order.append("wait")
        await runtime.commit(
            lambda _tx, _current: {
                "status": "waiting",
                "checkpoint": {"phase": "resume"},
                "on": list(on),
                "policy": "allSettled",
            },
            ctx,
        )

    async def resume(_task: Any, runtime: Any, ctx: Context) -> None:
        if order is not None:
            order.append("resume")
        if outcomes is not None:
            outcomes.extend(outcome["status"] for outcome in await runtime.outcomes(on, ctx))
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: aborted_with("test"), ctx)

    return define_task(
        name=name,
        version=1,
        initial=lambda _input: {"phase": "wait"},
        phases={"wait": wait, "resume": resume},
        abort=abort,
    )


async def test_resumes_a_waiting_task_once_every_task_in_on_is_terminal_whatever_the_outcome() -> None:
    gate = deferred()
    order: list[str] = []
    on: list[Any] = []
    outcomes: list[str] = []
    Waiter = _waiter_task("test.waiter", on, order, outcomes)

    async def first_run(_task: Any, runtime: Any, ctx: Context) -> None:
        order.append("first")
        await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    async def faulting_run(*_args: Any) -> None:
        order.append("faulting")
        raise RuntimeError("fails")

    First = one_step("test.first", first_run)
    Faulting = one_step("test.faulting", faulting_run)
    opened = await open_root([First, Faulting, Waiter])
    harness, root = opened.harness, opened.root
    first = await start(root, First)
    faulting = await start(root, Faulting)
    on.extend([first, faulting])
    waiter = await root.commit(
        lambda tx: tx.create_task(Waiter, None, {"ownership": {"kind": "conversation"}}), context
    )
    harness.resume()
    await eventually(lambda: len(order) == 3)
    await flush()
    assert sorted(order) == ["faulting", "first", "wait"]
    state = (await harness.get_task(waiter, context))["state"]
    assert state["status"] == "waiting" and state["on"] == on
    resolve(gate)
    await harness.wait_for_task(waiter, context)
    assert order[-1] == "resume"
    assert outcomes == ["completed", "faulted"]
    await harness.close(context)


async def test_stamps_started_at_at_the_first_run_and_ended_at_when_terminal_keeping_started_at_through_a_wait() -> (
    None
):
    """#10549"""
    clock = [1_000]
    gate = deferred()
    on: list[Any] = []
    Waiter = _waiter_task("test.timed-waiter", on)
    Held = gated("test.timed-held", gate)
    opened = await open_root([Waiter, Held], now=lambda: clock[0])
    harness, root = opened.harness, opened.root
    held = await start(root, Held)
    on.append(held)
    waiter = await root.commit(
        lambda tx: tx.create_task(Waiter, None, {"ownership": {"kind": "conversation"}}), context
    )

    async def times() -> list[Any]:
        tasks = (await harness.inspect(context))["tasks"]
        return [
            [task["record"]["id"], task["record"].get("startedAt"), task["record"].get("endedAt")] for task in tasks
        ]

    assert await times() == [[held, None, None], [waiter, None, None]]

    clock[0] = 2_000
    harness.resume()

    async def waiting() -> bool:
        return (await harness.get_task(waiter, context))["state"]["status"] == "waiting"

    await eventually(waiting)
    assert await times() == [[held, 2_000, None], [waiter, 2_000, None]]

    # The waiter runs again at 5_000; its start stays the first run.
    clock[0] = 5_000
    resolve(gate)
    settled_waiter = await harness.wait_for_task(waiter, context)
    settled_held = await harness.get_task(held, context)
    assert [settled_held.get("startedAt"), settled_held.get("endedAt")] == [2_000, 5_000]
    assert [settled_waiter.get("startedAt"), settled_waiter.get("endedAt")] == [2_000, 5_000]
    await harness.close(context)


async def test_keeps_one_registry_snapshot_per_phase_and_refreshes_it_at_the_phase_boundary() -> None:
    seen: list[Any] = []
    phase_gate = deferred()
    entered = deferred()

    def tools(snapshot: Any) -> list[str]:
        return [entry["tool"].name for entry in snapshot.tools()]

    async def a(_task: Any, runtime: Any, ctx: Context) -> None:
        before = runtime.registry
        resolve(entered)
        await phase_gate
        seen.append({"phase": "a", "tools": tools(runtime.registry), "same": runtime.registry is before})
        await runtime.commit(lambda _tx, _current: _running({"phase": "b"}), ctx)

    async def b(_task: Any, runtime: Any, ctx: Context) -> None:
        seen.append({"phase": "b", "tools": tools(runtime.registry), "same": True})
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Snapshots = define_task(
        name="test.snapshots", version=1, initial=lambda _input: {"phase": "a"}, phases={"a": a, "b": b}, abort=_nothing
    )
    opened = await open_root([Snapshots])
    harness, registry, root = opened.harness, opened.registry, opened.root
    id = await root.commit(lambda tx: tx.create_task(Snapshots, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await entered

    async def execute(*_args: Any) -> Any:
        return {}

    add_tool(registry, define_tool(name="late", description="late", parameters=Type.Object({}), execute=execute))
    resolve(phase_gate)
    await harness.wait_for_task(id, context)
    assert seen == [{"phase": "a", "tools": [], "same": True}, {"phase": "b", "tools": ["late"], "same": True}]
    await harness.close(context)


async def test_resolves_the_agent_at_first_use_in_a_phase_from_its_snapshot_and_keeps_it_for_the_phase() -> None:
    pings: list[str] = []
    seen: list[Any] = []
    before_use = deferred()
    after_use = deferred()
    waiting_before_use = deferred()
    waiting_after_use = deferred()

    async def ping(runtime: Any) -> list[str]:
        pings.clear()
        await runtime.hooks.each("ping", lambda handler: handler())
        return list(pings)

    async def a(_task: Any, runtime: Any, ctx: Context) -> None:
        resolve(waiting_before_use)
        await before_use
        first = await runtime.agent(ctx)
        seen.append({"phase": "a", "pings": await ping(runtime), "thinking": first["thinkingLevel"]})
        resolve(waiting_after_use)
        await after_use
        second = await runtime.agent(ctx)
        seen.append(
            {"phase": "a", "pings": await ping(runtime), "thinking": second["thinkingLevel"], "same": second is first}
        )
        await runtime.commit(lambda _tx, _current: _running({"phase": "b"}), ctx)

    async def b(_task: Any, runtime: Any, ctx: Context) -> None:
        agent = await runtime.agent(ctx)
        seen.append({"phase": "b", "pings": await ping(runtime), "thinking": agent["thinkingLevel"]})
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Phases = define_task(
        name="test.agent-phases",
        version=1,
        initial=lambda _input: {"phase": "a"},
        phases={"a": a, "b": b},
        abort=_nothing,
    )
    opened = await open_root([Phases])
    harness, registry, root = opened.harness, opened.registry, opened.root
    add_hooks(registry, Phases, {"ping": lambda: pings.append("before")})
    id = await root.commit(lambda tx: tx.create_task(Phases, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()

    # Configured during the phase but before its first use: the lazy resolution reads it.
    await waiting_before_use
    await root.configure({"thinkingLevel": "low"}, context)
    resolve(before_use)
    # Configured and installed after the first use: not seen until the next phase.
    await waiting_after_use
    add_hooks(registry, Phases, {"ping": lambda: pings.append("late")})
    await root.configure({"thinkingLevel": "high"}, context)
    resolve(after_use)

    await harness.wait_for_task(id, context)
    assert seen == [
        {"phase": "a", "pings": ["before"], "thinking": "low"},
        {"phase": "a", "pings": ["before"], "thinking": "low", "same": True},
        {"phase": "b", "pings": ["before", "late"], "thinking": "high"},
    ]
    await harness.close(context)


async def test_rejects_an_agent_wait_whose_caller_context_is_cancelled_without_affecting_the_phases_resolution() -> (
    None
):
    results: dict[str, Any] = {}

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        caller = with_cancel(ctx)
        caller.cancel(RuntimeError("caller gone"))
        try:
            await runtime.agent(caller.context)
        except Exception as error:
            results["cancelled"] = error
        results["resolved"] = (await runtime.agent(ctx))["thinkingLevel"]
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Agent = one_step("test.agent-cancel", run)
    opened = await open_root([Agent])
    harness, root = opened.harness, opened.root
    await harness.wait_for_task(await start(root, Agent), context)
    assert str(results["cancelled"]) == "caller gone"
    assert results["resolved"] == "off"
    await harness.close(context)


async def test_observes_a_failed_agent_resolution_whose_only_caller_stopped_waiting() -> None:
    fail = [False]

    def settings() -> Any:
        if fail[0]:
            raise RuntimeError("settings broke")
        return None

    resolved: list[Any] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        caller = with_cancel(ctx)
        caller.cancel(RuntimeError("caller gone"))
        fail[0] = True
        try:
            await runtime.agent(caller.context)
        except Exception:
            pass
        # Let an unobserved failure surface before the task ends.
        await asyncio.sleep(0)
        try:
            resolved.append(await runtime.agent(ctx))
        except Exception as error:
            resolved.append(error)
        fail[0] = False
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Agent = one_step("test.agent-failure", run)
    opened = await open_root([Agent], settings=settings)
    harness, root = opened.harness, opened.root
    await harness.wait_for_task(await start(root, Agent), context)
    assert str(resolved[0]) == "settings broke"
    await harness.close(context)


# ─── task runtime ───────────────────────────────────────────────────────────


async def test_rejects_runtime_operations_after_the_invocation_ends_and_stops_its_watches() -> None:
    Notes = define_doc(kind="test.task-notes", version=1, scope="session", initial=lambda: {"text": ""})
    Absent = define_doc(kind="test.task-absent", version=1, scope="session", initial=lambda: {"text": ""})
    captured: dict[str, Any] = {"absent": "unset"}
    delivered: list[str] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        captured["runtime"] = runtime
        captured["context"] = ctx
        captured["absent"] = await runtime.watch_doc(Absent, ctx)
        watch = await runtime.watch_doc(Notes, ctx)

        async def listener(value: Any, *_rest: Any) -> None:
            delivered.append("retired" if value is None else value["text"])

        watch.start(listener)
        captured["closed"] = watch.closed
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Watcher = one_step("test.watcher", run)
    opened = await open_root([Watcher])
    harness, root = opened.harness, opened.root

    async def hello(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "hello"

    await harness.commit(hello, context)
    id = await start(root, Watcher)
    harness.resume()
    await harness.wait_for_task(id, context)
    assert captured["absent"] is None
    # The watch stops when the step after the phase ends the invocation; later commits deliver nothing.
    assert await captured["closed"] == {"reason": "stopped"}

    async def after(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "after"

    await harness.commit(after, context)
    await flush()
    assert delivered == []
    runtime = captured["runtime"]
    for operation in (
        lambda: runtime.commit(lambda _tx, _current: None, context),
        lambda: runtime.memo("x", context),
        lambda: runtime.memo("x", 1, context),
        lambda: runtime.sleep_until(0, {"phase": "run"}, context),
        lambda: runtime.watch_doc(Notes, context),
        # The handler's own context is cancelled by now; the ended invocation still wins.
        lambda: runtime.agent(captured["context"]),
    ):
        with pytest.raises(Exception, match="invocation has ended"):
            await operation()
    with pytest.raises(Exception, match="invocation has ended"):
        runtime.now()
    with pytest.raises(Exception, match="invocation has ended"):
        runtime.report(RuntimeError("late"))
    await harness.close(context)


async def test_reads_committed_documents_and_context_through_the_runtime_and_forwards_the_clock_and_reports() -> None:
    Notes = define_doc(
        kind="test.runtime-notes",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"text": ""},
    )
    captured: dict[str, Any] = {}
    seen: list[Any] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        captured["runtime"] = runtime
        first, second = [entry["id"] for entry in (await runtime.context(runtime.conversation_id, ctx))["entries"]]
        seen.append((await runtime.snapshot(Notes, runtime.conversation_id, ctx))["text"])
        seen.append((await runtime.snapshot_as_of(Notes, runtime.conversation_id, first, ctx))["text"])
        seen.append(len((await runtime.context(runtime.conversation_id, ctx, at=first))["entries"]))
        seen.append(len((await runtime.context(runtime.conversation_id, ctx, at=second))["messages"]))
        seen.append(runtime.now())
        runtime.report(RuntimeError("reported"))
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Reader = one_step("test.reader", run)
    opened = await open_root([Reader], now=lambda: 1234)
    harness, root, reports = opened.harness, opened.root, opened.reports
    for text in ("one", "two"):

        async def write(tx: Any, text: str = text) -> None:
            (await tx.doc(Notes, root.id))["text"] = text
            await tx.append_entry(root.id, {"kind": "message", "model": [user(text)]})

        await root.commit(write, context)
    id = await start(root, Reader)
    harness.resume()
    await harness.wait_for_task(id, context)
    assert seen == ["two", "one", 1, 2, 1234]
    assert [str(error) for error in reports] == ["reported"]
    # The step after the phase ends the invocation.
    await flush()
    with pytest.raises(Exception, match="invocation has ended"):
        await captured["runtime"].snapshot(Notes, root.id, context)
    with pytest.raises(Exception, match="invocation has ended"):
        await captured["runtime"].context(root.id, context)
    await harness.close(context)


async def test_orders_runtime_commits_against_the_step_one_queued_before_it_lands_one_after_it_rejects() -> None:
    storage = controlled()
    holder: dict[str, Any] = {}

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        # Hold the line, queue a commit without awaiting it, and return; the step queues behind that commit.
        holder["held"] = storage.hold_commits()

        async def block(tx: Any) -> None:
            await tx.append_entry(runtime.conversation_id, {"kind": "blocker"})

        holder["harness"].commit(block, context)
        holder["before"] = runtime.commit(lambda _tx, _current: completed("before"), context)

        # Queued after the handler returned, so after the step.
        def later() -> None:
            holder["after"] = runtime.commit(lambda _tx, _current: completed("after"), context)

        asyncio.get_running_loop().call_soon(later)

    Detached = one_step("test.detached", run)
    opened = await open_root([Detached], storage=storage)
    harness, root = opened.harness, opened.root
    holder["harness"] = harness
    id = await start(root, Detached)
    harness.resume()
    await eventually(lambda: "held" in holder)
    await holder["held"].entered
    await eventually(lambda: "after" in holder)
    holder["held"].release()
    await holder["before"]
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "completed", "result": "before"}
    with pytest.raises(Exception, match="invocation has ended|is terminal"):
        await holder["after"]
    await harness.close(context)


async def test_stops_a_watch_whose_acquisition_finishes_after_the_invocation_ended() -> None:
    Notes = define_doc(kind="test.late-watch", version=1, scope="session", initial=lambda: {"text": ""})
    storage = controlled()
    holder: dict[str, Any] = {}

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        holder["held"] = storage.hold_commits()

        async def block(tx: Any) -> None:
            await tx.append_entry(runtime.conversation_id, {"kind": "blocker"})

        holder["harness"].commit(block, context)

        # Starts after the handler returned, so its line job queues behind the step that ends the invocation.
        async def watch() -> Any:
            try:
                return await runtime.watch_doc(Notes, context)
            except Exception as error:
                return error

        asyncio.get_running_loop().call_soon(lambda: holder.__setitem__("watching", asyncio.ensure_future(watch())))

    Late = one_step("test.late-watch", run)
    opened = await open_root([Late], storage=storage)
    harness, root = opened.harness, opened.root
    holder["harness"] = harness

    async def seed(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "x"

    await harness.commit(seed, context)
    id = await start(root, Late)
    harness.resume()
    await eventually(lambda: "held" in holder)
    await holder["held"].entered
    await eventually(lambda: "watching" in holder)
    holder["held"].release()
    assert "invocation has ended" in str(await holder["watching"])
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "faulted"
    await harness.close(context)


# ─── task scheduling ────────────────────────────────────────────────────────


async def test_fails_the_harness_on_a_failed_reservation_commit_reopening_runs_the_task() -> None:
    runs = [0]

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        runs[0] += 1
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Once = one_step("test.once", run)
    storage = controlled()
    opened = await open_root([Once], storage=storage)
    harness, root, reports = opened.harness, opened.root, opened.reports
    id = await start(root, Once)
    waiting = harness.wait_for_task(id, context)
    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    harness.resume()
    end = await harness.closed
    assert end == {"reason": "failed", "error": failure}
    assert reports == [failure]
    # The pending wait and every later call get SessionFailed with the storage error.
    with pytest.raises(SessionFailed) as caught:
        await waiting
    assert caught.value.cause is failure
    with pytest.raises(SessionFailed) as caught:
        await harness.get_task(id, context)
    assert caught.value.cause is failure
    with pytest.raises(SessionFailed):
        await root.submit({"type": "input", "content": "x"}, context)
    assert runs[0] == 0
    await harness.close(context)

    reopened = await open_root([Once], storage=await storage.reopen())
    reopened.harness.resume()
    assert (await reopened.harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    assert runs[0] == 1
    await reopened.harness.close(context)


async def test_fails_the_harness_instead_of_running_a_task_again_when_its_fault_write_fails() -> None:
    runs = [0]
    storage = controlled()
    failure = RuntimeError("disk gone")

    async def run(*_args: Any) -> None:
        runs[0] += 1
        # The next commit is the step's fault write.
        if runs[0] == 1:
            storage.fail_next_commit(failure)
        raise RuntimeError("boom")

    Throws = one_step("test.failed-fault", run)
    opened = await open_root([Throws], storage=storage)
    harness, root, reports = opened.harness, opened.root, opened.reports
    id = await start(root, Throws)
    harness.resume()
    assert await harness.closed == {"reason": "failed", "error": failure}
    assert runs[0] == 1
    assert reports == [failure]
    await harness.close(context)

    # Reopening recovers the task as after a crash: still running, so its phase runs again.
    reopened = await open_root([Throws], storage=await storage.reopen())
    reopened.harness.resume()
    assert (await reopened.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "faulted",
        "error": {"message": "boom"},
    }
    assert runs[0] == 2
    await reopened.harness.close(context)


async def test_waits_for_harness_and_conversation_idleness_counting_blocked_work_and_ignoring_background_tasks() -> (
    None
):
    gates: dict[Any, asyncio.Future[Any]] = {}

    async def run(task: Any, runtime: Any, ctx: Context) -> None:
        gate = deferred()
        gates[task["id"]] = gate
        await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Gated = one_step("test.gated", run)
    opened = await open_root([Gated])
    harness, root = opened.harness, opened.root
    await harness.wait_for_idle(context)
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    foreground = await start(root, Gated)
    background = await start(root, Gated, background=True)
    elsewhere = await start(other, Gated)
    # Work that has not started yet is live; cancelling a wait only rejects that wait.
    cancelled = AbortController()
    cancelled_wait = harness.wait_for_idle(with_signal(cancelled.signal))
    cancelled.abort(RuntimeError("stop waiting"))
    with pytest.raises(Exception, match="stop waiting"):
        await cancelled_wait
    already = AbortController()
    already.abort(RuntimeError("already cancelled"))
    with pytest.raises(Exception, match="already cancelled"):
        await root.wait_for_idle(with_signal(already.signal))
    harness.resume()
    await eventually(lambda: len(gates) == 3)
    root_idle = root.wait_for_idle(context)
    harness_idle = harness.wait_for_idle(context)
    resolve(gates[foreground])
    await root_idle
    assert await settled(asyncio.shield(harness_idle)) is False
    resolve(gates[elsewhere])
    await harness_idle
    assert (await harness.get_task(background, context))["state"]["status"] == "running"
    resolve(gates[background])
    await harness.wait_for_task(background, context)
    await harness.close(context)
    with pytest.raises(Exception, match="closed"):
        await harness.wait_for_idle(context)
    with pytest.raises(Exception, match="closed"):
        await root.wait_for_idle(context)


async def test_pages_the_newest_tasks_first_without_scanning_older_ones() -> None:
    """#10546"""
    Idle = one_step("test.idle", _nothing)
    opened = await open_root([Idle])
    harness, root = opened.harness, opened.root
    ids = [await start(root, Idle) for _ in range(5)]
    first = await harness.commit(lambda tx: tx.scan_tasks({"order": "descending"}, 2), context)
    assert [task["id"] for task in first["items"]] == [ids[4], ids[3]]
    second = await harness.commit(lambda tx: tx.scan_tasks({}, 2, first["next"]), context)
    assert [task["id"] for task in second["items"]] == [ids[2], ids[1]]
    await harness.close(context)


async def test_rejects_unknown_tasks_and_reports_terminal_tasks() -> None:
    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Done = one_step("test.quick", run)
    opened = await open_root([Done])
    harness, root = opened.harness, opened.root
    unknown: Any = 999_999
    assert await harness.get_task(unknown, context) is None
    with pytest.raises(Exception, match="does not exist"):
        await harness.wait_for_task(unknown, context)
    with pytest.raises(Exception, match="does not exist"):
        await harness.abort_task(unknown, context)
    id = await start(root, Done)
    harness.resume()
    await harness.wait_for_task(id, context)
    assert (await harness.wait_for_task(id, context))["state"]["status"] == "terminal"
    assert await harness.abort_task(id, context) == "terminal"
    await harness.close(context)
    with pytest.raises(Exception, match="closed"):
        harness.resume()


async def test_rejects_task_waits_cancelled_or_closed_while_queued_on_the_line_and_pending_waits_on_close() -> None:
    gate = deferred()
    Blocking = gated("test.wait-close", gate)
    storage = controlled()
    opened = await open_root([Blocking], storage=storage)
    harness, root = opened.harness, opened.root
    id = await start(root, Blocking)

    async def block(tx: Any) -> None:
        await tx.append_entry(root.id, {"kind": "blocker"})

    # Hold the line with a commit, then queue waits behind it.
    held = storage.hold_commits()
    blocker = harness.commit(block, context)
    await held.entered
    controller = AbortController()
    cancelled_while_queued = harness.wait_for_task(id, with_signal(controller.signal))
    controller.abort(RuntimeError("wait cancelled"))
    held.release()
    await blocker
    with pytest.raises(Exception, match="wait cancelled"):
        await cancelled_while_queued

    pending = harness.wait_for_task(id, context)
    idle = harness.wait_for_idle(context)
    conversation_idle = root.wait_for_idle(context)
    await flush()
    held_again = storage.hold_commits()
    blocker_again = harness.commit(block, context)
    await held_again.entered
    closed_while_queued = harness.wait_for_task(id, context)
    closing = harness.close(context)
    held_again.release()
    await blocker_again
    for waiting in (closed_while_queued, pending, idle, conversation_idle):
        with pytest.raises(Exception, match="closed"):
            await waiting
    resolve(gate)
    await closing


# ─── task abort ─────────────────────────────────────────────────────────────


async def test_rejects_the_runs_commits_and_memo_writes_after_the_mark_and_settles_through_a_fresh_abort_invocation() -> (
    None
):
    reached = deferred()
    errors: list[str] = []
    memo_read: list[Any] = []
    abort_runtimes: list[Any] = []
    run_runtime: list[Any] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        run_runtime.append(runtime)
        await runtime.memo("kept", 1, ctx)
        resolve(reached)
        # Keep working after the signal: every later write of this run must reject.
        try:
            await aborted(runtime.signal)
        except Exception:
            pass
        memo_read.append(await runtime.memo("kept", context))
        try:
            await runtime.memo("late", 2, context)
        except Exception as error:
            errors.append(str(error))
        try:
            await runtime.commit(lambda _tx, _current: completed(None), context)
        except Exception as error:
            errors.append(str(error))

    async def abort(runtime: Any, ctx: Context) -> None:
        abort_runtimes.append(runtime)

        def change(_tx: Any, current: Any) -> Any:
            assert current["abortRequested"] is True
            assert current["memos"] == {"kept": 1}
            return aborted_with("mark")

        await runtime.commit(change, ctx)

    Marked = one_step("test.marked", run, abort)
    opened = await open_root([Marked])
    harness, root = opened.harness, opened.root
    id = await start(root, Marked)
    harness.resume()
    await reached
    assert await harness.abort_task(id, context) == "marked"
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "mark"}
    assert memo_read == [1]
    assert errors == [f"Task {id} has a durable abort mark", f"Task {id} has a durable abort mark"]
    assert len(abort_runtimes) == 1
    assert abort_runtimes[0] is not run_runtime[0]
    await harness.close(context)


async def test_starts_no_further_phase_after_a_mark_that_lands_during_a_phase_with_progress() -> None:
    reached = deferred()
    proceed = deferred()
    phases: list[str] = []

    async def one(_task: Any, runtime: Any, ctx: Context) -> None:
        phases.append("one")
        await runtime.commit(lambda _tx, _current: _running({"phase": "two"}), ctx)
        resolve(reached)
        # Ignores the signal and returns normally after the mark.
        await proceed

    async def two(*_args: Any) -> None:
        phases.append("two")

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        phases.append("abort")
        await runtime.commit(lambda _tx, _current: aborted_with("boundary"), ctx)

    Two = define_task(
        name="test.mark-boundary",
        version=1,
        initial=lambda _input: {"phase": "one"},
        phases={"one": one, "two": two},
        abort=abort,
    )
    opened = await open_root([Two])
    harness, root = opened.harness, opened.root
    id = await root.commit(lambda tx: tx.create_task(Two, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await reached
    aborting = await mark_durably(harness, id)
    resolve(proceed)
    assert await aborting == "marked"
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "boundary"}
    assert phases == ["one", "abort"]
    await harness.close(context)


async def test_signals_and_joins_the_run_before_returning_then_runs_the_abort_handler() -> None:
    reached = deferred()
    release = deferred()
    run_ended = [False]

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        resolve(reached)
        try:
            await aborted(runtime.signal)
        finally:
            run_ended[0] = True

    # Upstream's abort handler commits at once and still loses the race with the second abort_task(); Python's
    # commits settle in fewer loop turns, so the handler waits for the second call here.
    async def abort(runtime: Any, ctx: Context) -> None:
        await release
        await runtime.commit(lambda _tx, _current: aborted_with("test"), ctx)

    Signalled = one_step("test.signalled", run, abort)
    opened = await open_root([Signalled])
    harness, root = opened.harness, opened.root
    id = await start(root, Signalled)
    harness.resume()
    await reached
    assert await harness.abort_task(id, context) == "marked"
    assert run_ended[0] is True
    assert await harness.abort_task(id, context) == "marked"
    resolve(release)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "test"}
    await harness.close(context)


async def test_aborts_waiting_work_before_its_wait_ends_and_faults_abort_handlers_that_raise_or_settle_nothing() -> (
    None
):
    gate = deferred()
    First = gated("test.dependency", gate)
    first: list[Any] = []

    async def wait(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(
            lambda _tx, _current: {
                "status": "waiting",
                "checkpoint": {"phase": "run"},
                "on": [first[0]],
                "policy": "allSettled",
            },
            ctx,
        )

    async def lazy_abort(*_args: Any) -> None:
        return None

    async def throwing_abort(*_args: Any) -> None:
        raise RuntimeError("abort failed")

    Lazy = one_step("test.lazy-abort", wait, lazy_abort)
    Throwing = one_step("test.throwing-abort", wait, throwing_abort)
    opened = await open_root([First, Lazy, Throwing])
    harness, root = opened.harness, opened.root
    first.append(await start(root, First))
    lazy = await start(root, Lazy)
    throwing = await start(root, Throwing)
    harness.resume()

    def waiting(id: Any) -> Any:
        async def check() -> bool:
            return (await harness.get_task(id, context))["state"]["status"] == "waiting"

        return check

    await eventually(waiting(throwing))
    await eventually(waiting(lazy))
    await harness.abort_task(lazy, context)
    await harness.abort_task(throwing, context)
    assert (await harness.wait_for_task(lazy, context))["state"]["outcome"] == {
        "status": "faulted",
        "error": {"message": f"Abort handler of task {lazy} returned without a terminal outcome"},
    }
    assert (await harness.wait_for_task(throwing, context))["state"]["outcome"] == {
        "status": "faulted",
        "error": {"message": "abort failed"},
    }
    assert (await harness.get_task(first[0], context))["state"]["status"] == "running"
    resolve(gate)
    await harness.wait_for_task(first[0], context)
    await harness.close(context)


async def test_does_not_signal_a_running_abort_handler_when_aborted_again_and_a_cancelled_caller_leaves_the_mark() -> (
    None
):
    reached = deferred()
    proceed = deferred()
    run_release = deferred()
    state: dict[str, Any] = {"aborts": 0}

    async def run(*_args: Any) -> None:
        # Ignores the signal, so the first caller is still joining when it gives up.
        await run_release

    async def abort(runtime: Any, ctx: Context) -> None:
        state["aborts"] += 1
        resolve(reached)
        await proceed
        state["signalled"] = runtime.signal.aborted
        await runtime.commit(lambda _tx, _current: aborted_with("once"), ctx)

    Run = one_step("test.abort-again", run, abort)
    opened = await open_root([Run])
    harness, root = opened.harness, opened.root
    id = await start(root, Run)
    harness.resume()
    await flush()
    # This caller gives up while joining; the mark and the abort invocation are unaffected.
    controller = AbortController()
    cancelled = harness.abort_task(id, with_signal(controller.signal))
    while not ((await harness.get_task(id, context)) or {}).get("abortRequested"):
        await flush()
    controller.abort(RuntimeError("caller gave up"))
    with pytest.raises(Exception, match="caller gave up"):
        await cancelled
    resolve(run_release)
    await reached
    assert await harness.abort_task(id, context) == "marked"
    resolve(proceed)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "once"}
    assert state["signalled"] is False
    assert state["aborts"] == 1
    await harness.close(context)


async def test_keeps_a_terminal_outcome_committed_by_an_abort_handler_that_raises_afterwards() -> None:
    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        await aborted(runtime.signal)

    async def abort(runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: aborted_with("done"), ctx)
        raise RuntimeError("after terminal")

    Run = one_step("test.abort-then-throw", run, abort)
    opened = await open_root([Run])
    harness, root = opened.harness, opened.root
    id = await start(root, Run)
    harness.resume()
    await flush()
    await harness.abort_task(id, context)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "done"}
    await harness.close(context)


async def test_never_starts_phase_one_when_the_abort_lands_while_the_reservation_settles() -> None:
    ran = [False]

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        ran[0] = True
        await aborted(runtime.signal)

    Reserved = one_step("test.reserved", run)
    storage = controlled()
    opened = await open_root([Reserved], storage=storage)
    harness, root = opened.harness, opened.root
    id = await start(root, Reserved)
    held = storage.hold_commits()
    harness.resume()
    await held.entered
    # The reservation commit is in storage; the abort mark commit queues behind it.
    aborting = harness.abort_task(id, context)
    held.release()
    assert await aborting == "marked"
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "test"}
    assert ran[0] is False
    await harness.close(context)


async def test_lets_an_abort_mark_win_over_a_fault_that_races_it() -> None:
    storage = controlled()
    holder: dict[str, Any] = {}

    async def run(task: Any, *_rest: Any) -> None:
        holder["held"] = storage.hold_commits()
        holder["marking"] = holder["harness"].abort_task(task["id"], context)
        await holder["held"].entered
        # The mark is in storage but not committed when the handler raises, so the fault commit queues behind it.
        raise RuntimeError("would fault")

    Racing = one_step("test.racing-fault", run)
    opened = await open_root([Racing], storage=storage)
    harness, root = opened.harness, opened.root
    holder["harness"] = harness
    id = await start(root, Racing)
    harness.resume()
    await eventually(lambda: "held" in holder)
    await holder["held"].entered
    await flush()
    holder["held"].release()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "test"}
    assert await holder["marking"] == "marked"
    await harness.close(context)


# ─── task close ─────────────────────────────────────────────────────────────


def _task_states(storage: ControlledStorage, id: Any) -> list[str]:
    return [
        write["value"]["state"]["status"]
        for writes in storage.commits
        for write in writes
        if write["type"] == "task" and write["value"]["id"] == id
    ]


async def test_stops_without_outcomes_and_starts_no_fresh_phase_or_abort_invocation_while_closing() -> None:
    reached = deferred()
    proceed = deferred()
    phases: list[str] = []
    abort_ran = [False]

    async def one(_task: Any, runtime: Any, ctx: Context) -> None:
        phases.append("one")
        await runtime.commit(lambda _tx, _current: _running({"phase": "two"}), ctx)
        resolve(reached)
        # Ignores signals and returns normally once released; the closing rule wins over the next phase.
        await proceed

    async def two(*_args: Any) -> None:
        phases.append("two")

    async def abort(*_args: Any) -> None:
        abort_ran[0] = True

    Two = define_task(
        name="test.two",
        version=1,
        initial=lambda _input: {"phase": "one"},
        phases={"one": one, "two": two},
        abort=abort,
    )
    storage = controlled()
    opened = await open_root([Two], storage=storage)
    harness, root = opened.harness, opened.root
    id = await root.commit(lambda tx: tx.create_task(Two, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await reached
    aborting = await mark_durably(harness, id)
    record = await harness.get_task(id, context)
    commits = len(storage.commits)
    closing = harness.close(context)
    resolve(proceed)
    await closing
    assert await aborting == "marked"
    # No outcome: close writes only the stopped task back as pending, so a reopen counts no restart.
    assert len(storage.commits) == commits + 1
    [last] = storage.commits[-1]
    assert last["type"] == "task"
    assert last["value"]["id"] == id and last["value"]["abortRequested"] is True
    assert last["value"]["state"] == {"status": "pending", "checkpoint": {"phase": "two"}}
    assert phases == ["one"]
    assert abort_ran[0] is False
    assert record["abortRequested"] is True
    assert record["state"] == {"status": "running", "checkpoint": {"phase": "two"}}
    with pytest.raises(Exception, match="closed"):
        await harness.commit(lambda _tx: None, context)


async def test_seals_admission_before_signalling_handlers_and_stops_watches_before_joining_them() -> None:
    Notes = define_doc(kind="test.close-notes", version=1, scope="session", initial=lambda: {"text": ""})
    reached = deferred()
    holder: dict[str, Any] = {}

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        # Uses a context close does not cancel, then waits for the watch to close.
        watch = await runtime.watch_doc(Notes, context)

        def on_abort() -> None:
            holder["from_listener"] = holder["harness"].commit(lambda _tx: None, context)

        runtime.signal.add_listener(on_abort)
        resolve(reached)
        holder["watch_end"] = await watch.closed

    Stubborn = one_step("test.stubborn", run)
    opened = await open_root([Stubborn])
    harness, root = opened.harness, opened.root
    holder["harness"] = harness

    async def seed(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "x"

    await harness.commit(seed, context)
    await start(root, Stubborn)
    harness.resume()
    await reached
    await harness.close(context)
    with pytest.raises(Exception, match="closed"):
        await holder["from_listener"]
    assert holder["watch_end"] == {"reason": "session_closed"}


async def test_writes_no_fault_when_close_seals_while_the_step_after_a_failed_phase_is_queued() -> None:
    storage = controlled()
    holder: dict[str, Any] = {}

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        holder["held"] = storage.hold_commits()

        async def block(tx: Any) -> None:
            await tx.append_entry(runtime.conversation_id, {"kind": "blocker"})

        holder["harness"].commit(block, context)
        raise RuntimeError("would fault")

    Throws = one_step("test.close-fault", run)
    opened = await open_root([Throws], storage=storage)
    harness, root = opened.harness, opened.root
    holder["harness"] = harness
    id = await start(root, Throws)
    harness.resume()
    await eventually(lambda: "held" in holder)
    await holder["held"].entered
    await flush()
    closing = harness.close(context)
    holder["held"].release()
    await closing
    # Created, reserved, and written back as pending by close: no fault.
    assert _task_states(storage, id) == ["pending", "running", "pending"]


async def test_starts_no_abort_handler_whose_reservation_settles_while_closing() -> None:
    ran = [False]

    async def abort(*_args: Any) -> None:
        ran[0] = True

    Marked = one_step("test.close-abort-reservation", _nothing, abort)
    storage = controlled()
    opened = await open_root([Marked], storage=storage)
    harness, root = opened.harness, opened.root
    id = await start(root, Marked)
    assert await harness.abort_task(id, context) == "marked"
    held = storage.hold_commits()
    harness.resume()
    await held.entered
    closing = harness.close(context)
    held.release()
    await closing
    assert ran[0] is False
    # The reservation landed; close wrote the task back as pending.
    reserved = [write for write in storage.commits[-2] if write["type"] == "task"]
    assert len(reserved) == 1
    assert reserved[0]["value"]["id"] == id and reserved[0]["value"]["abortRequested"] is True
    assert reserved[0]["value"]["state"]["status"] == "running"
    [last] = storage.commits[-1]
    assert last["type"] == "task" and last["value"]["id"] == id and last["value"]["abortRequested"] is True
    assert last["value"]["state"]["status"] == "pending"


async def test_rejects_a_runtime_commit_that_was_queued_on_the_line_when_close_sealed_it() -> None:
    storage = controlled()
    holder: dict[str, Any] = {}
    proceed = deferred()

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        holder["held"] = storage.hold_commits()

        async def block(tx: Any) -> None:
            await tx.append_entry(runtime.conversation_id, {"kind": "blocker"})

        holder["harness"].commit(block, context)

        async def queued() -> Any:
            try:
                return await runtime.commit(lambda _tx, _current: completed(None), context)
            except Exception as error:
                return error

        holder["queued"] = asyncio.ensure_future(queued())
        # Ignores the close signal, so the invocation is still alive when the queued commit reaches the line.
        await proceed

    Queued = one_step("test.close-queued-commit", run)
    opened = await open_root([Queued], storage=storage)
    harness, root = opened.harness, opened.root
    holder["harness"] = harness
    id = await start(root, Queued)
    harness.resume()
    await eventually(lambda: "held" in holder)
    await holder["held"].entered
    await flush()
    closing = harness.close(context)
    holder["held"].release()
    assert "Harness is closed" in str(await holder["queued"])
    resolve(proceed)
    await closing
    # Created, reserved, and written back as pending by close; the queued outcome never landed.
    assert _task_states(storage, id) == ["pending", "running", "pending"]


async def test_starts_no_next_phase_when_close_seals_during_a_step_that_decided_to_continue() -> None:
    phases: list[str] = []
    registry = create_registry()
    holder: dict[str, Any] = {"close_on_snapshot": False}

    # The step refreshes the snapshot after progress, inside its line callback and after its closing check.
    class Reader:
        def snapshot(self) -> Any:
            if holder["close_on_snapshot"]:
                holder["close_on_snapshot"] = False
                holder["closing"] = holder["harness"].close(context)
            return registry.snapshot()

        def subscribe(self, listener: Any) -> Any:
            return registry.subscribe(listener)

    async def one(_task: Any, runtime: Any, ctx: Context) -> None:
        phases.append("one")
        await runtime.commit(lambda _tx, _current: _running({"phase": "two"}), ctx)
        holder["close_on_snapshot"] = True

    async def two(*_args: Any) -> None:
        phases.append("two")

    Two = define_task(
        name="test.close-in-step",
        version=1,
        initial=lambda _input: {"phase": "one"},
        phases={"one": one, "two": two},
        abort=_nothing,
    )
    add_task(registry, Two)
    harness = await Harness.open(MemoryStorage(), {"models": create_models(), "registry": Reader()}, context)
    holder["harness"] = harness
    root = await harness.root(context)
    await root.commit(lambda tx: tx.create_task(Two, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await eventually(lambda: "closing" in holder)
    await holder["closing"]
    assert phases == ["one"]


async def test_joins_a_reservation_that_settles_while_closing_without_starting_its_handler() -> None:
    ran = [False]

    async def run(*_args: Any) -> None:
        ran[0] = True

    Never = one_step("test.close-reservation", run)
    storage = controlled()
    opened = await open_root([Never], storage=storage)
    harness, root = opened.harness, opened.root
    id = await start(root, Never)
    held = storage.hold_commits()
    harness.resume()
    await held.entered
    closing = harness.close(context)
    held.release()
    await closing
    assert ran[0] is False
    # The reservation landed; close wrote the task back as pending.
    reserved = [write for write in storage.commits[-2] if write["type"] == "task"]
    assert len(reserved) == 1 and reserved[0]["value"]["id"] == id
    assert reserved[0]["value"]["state"]["status"] == "running"
    [last] = storage.commits[-1]
    assert last["type"] == "task" and last["value"]["id"] == id and last["value"]["state"]["status"] == "pending"
