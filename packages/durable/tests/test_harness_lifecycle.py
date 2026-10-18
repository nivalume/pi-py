"""Port of ``test/harness-lifecycle.test.ts``."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import (
    add_hooks,
    add_task,
    add_tool,
    chat_setup,
    completed,
    counting_reader,
    open_chat,
    open_tasks,
    settled,
    wait_for,
)
from durable_session_support import (
    STORAGE_BACKENDS,
    ControlledStorage,
    context,
    controlled_storage,
    deferred,
    eventually,
    flush,
    resolve,
)

from pi_ai import AbortController, Type, create_models, faux_assistant_message, faux_text, faux_tool_call
from pi_chord.context import Context, with_abort_signal
from pi_durable import (
    GenerationTask,
    Harness,
    MemoryStorage,
    create_registry,
    define_doc,
    define_task,
    define_tool,
)
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Storage


def one_step(name: str, run: Any = None) -> Any:
    """A one-phase task running ``run`` and completing with ``None``."""

    async def phase(_task: Any, runtime: Any, ctx: Context) -> None:
        if run is not None:
            await run()
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    async def abort(*_args: Any) -> None:
        return None

    return define_task(
        name=name, version=1, initial=lambda _input: {"phase": "run"}, phases={"run": phase}, abort=abort
    )


async def start(conversation: Any, task: Any, background: bool = False) -> Any:
    return await conversation.commit(
        lambda tx: tx.create_task(task, None, {"ownership": {"kind": "conversation"}, "background": background}),
        context,
    )


def with_signal(signal: Any) -> Context:
    return with_abort_signal(signal, context)


async def storage_open(storage: Storage) -> bool:
    """Whether a raw Storage read still succeeds, as code of a joined invocation relies on."""
    try:
        await storage.task(1, context)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


async def memory_storage() -> ControlledStorage:
    return await controlled_storage(STORAGE_BACKENDS["memory"])


async def seed_running_task(storage: Storage) -> None:
    """Commit a conversation with a ``running`` task, as a crash leaves it, so open has a reconciliation commit."""
    conversation_id = await storage.mint_id()
    id = await storage.mint_id()
    state = {"status": "running", "checkpoint": {"phase": "run"}}
    task = {
        "id": id,
        "conversationId": conversation_id,
        "kind": "test.seeded",
        "version": 1,
        "input": None,
        "background": False,
        "state": state,
        "abortRequested": False,
    }
    await storage.commit(
        [{"type": "conversation", "value": {"id": conversation_id}}, {"type": "task", "value": task}],  # type: ignore[list-item]
        context,
    )


async def test_closes_without_a_cancelled_caller_context_and_raises_the_original_error_when_open_fails() -> None:
    storage = await memory_storage()
    await seed_running_task(storage)
    reader = counting_reader(create_registry())
    held = storage.hold_commits()
    storage.fail_next_commit(RuntimeError("disk full"))
    controller = AbortController()
    opening = asyncio.ensure_future(
        Harness.open(storage, {"models": create_models(), "registry": reader}, with_signal(controller.signal))
    )
    await held.entered
    controller.abort(RuntimeError("caller gave up"))
    held.release()
    with pytest.raises(RuntimeError, match="disk full"):
        await opening
    assert reader.subscriptions() == 0
    assert await storage_open(storage) is False


async def test_raises_the_open_error_and_reports_it_once_though_closing_then_fails_too() -> None:
    class FailingClose(ControlledStorage):
        async def close(self, context: Context) -> None:
            await super().close(context)
            raise RuntimeError("close failed")

    storage = FailingClose(MemoryStorage(), STORAGE_BACKENDS["memory"])
    await seed_running_task(storage)
    storage.fail_next_commit(RuntimeError("disk full"))
    reports: list[BaseException] = []
    with pytest.raises(RuntimeError, match="disk full"):
        await Harness.open(
            storage,
            {
                "models": create_models(),
                "registry": create_registry(),
                "onReport": lambda error, _info: reports.append(error),
            },
            context,
        )
    # The storage failure that failed open is the cause; the close that fails after it adds no report.
    assert [str(error) for error in reports] == ["disk full"]


async def test_joins_a_task_handler_that_ignores_its_signal_before_closing_storage() -> None:
    storage = MemoryStorage()
    reached = deferred()
    gate = deferred()
    read_after_release: list[bool] = []

    async def run() -> None:
        resolve(reached)
        await gate
        read_after_release.append(await storage_open(storage))

    stubborn = one_step("test.close-stubborn", run)
    opened = await open_tasks(storage, [stubborn])
    harness = opened.harness
    await start(await harness.root(context), stubborn)
    harness.resume()
    await reached
    closing = harness.close(context)
    assert await settled(asyncio.shield(closing)) is False
    assert await storage_open(storage) is True
    resolve(gate)
    await closing
    assert read_after_release == [True]
    assert await storage_open(storage) is False


@pytest.mark.parametrize("where", ["tool", "hook"])
async def test_joins_a_tool_execute_and_a_hook_that_ignore_their_signal_before_closing_storage(where: str) -> None:
    setup = chat_setup()
    storage = MemoryStorage()
    reached = deferred()
    gate = deferred()
    read_after_release: list[bool] = []

    async def stubborn() -> None:
        resolve(reached)
        await gate
        read_after_release.append(await storage_open(storage))

    async def execute(_args: Any, _api: Any, _context: Any) -> Any:
        if where == "tool":
            await stubborn()
        return {"output": []}

    async def before_request(*_args: Any) -> None:
        if where == "hook":
            await stubborn()
        return None

    add_tool(setup.registry, define_tool(name="wait", description="wait", parameters=Type.Object({}), execute=execute))
    add_hooks(setup.registry, GenerationTask, {"before_request": before_request})
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("wait", {}, id="c1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(storage, setup)
    await opened.root.submit({"type": "input", "content": "go"}, context)
    await reached
    closing = opened.harness.close(context)
    assert await settled(asyncio.shield(closing)) is False
    assert await storage_open(storage) is True
    resolve(gate)
    await closing
    assert read_after_release == [True]
    assert await storage_open(storage) is False


async def test_lets_a_new_harness_open_the_same_storage_once_close_resolved(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    log: list[str] = []
    gate = deferred()
    generation = [1]

    async def run() -> None:
        mine = generation[0]
        log.append(f"start {mine}")
        if mine == 1:
            await gate
        log.append(f"end {mine}")

    stubborn = one_step("test.generations", run)
    first = await open_tasks(await open_sqlite_storage(path), [stubborn])
    id = await start(await first.harness.root(context), stubborn)
    first.harness.resume()
    # SQLite runs on executor threads: wait in real time, not a fixed number of turns.
    await wait_for(lambda: len(log) == 1)

    async def close_and_log() -> None:
        await first.harness.close(context)
        log.append("closed")

    closing = asyncio.ensure_future(close_and_log())
    assert await settled(asyncio.shield(closing)) is False
    resolve(gate)
    await closing
    assert log == ["start 1", "end 1", "closed"]
    generation[0] = 2
    second = await open_tasks(await open_sqlite_storage(path), [stubborn])
    second.harness.resume()
    await second.harness.wait_for_task(id, context)
    assert log == ["start 1", "end 1", "closed", "start 2", "end 2"]
    await second.harness.close(context)


async def test_keeps_shutting_down_after_a_cancelled_close_and_a_second_close_awaits_the_same_shutdown() -> None:
    storage = MemoryStorage()
    reached = deferred()
    gate = deferred()

    async def run() -> None:
        resolve(reached)
        await gate

    stubborn = one_step("test.close-cancelled", run)
    opened = await open_tasks(storage, [stubborn])
    harness = opened.harness
    await start(await harness.root(context), stubborn)
    harness.resume()
    await reached
    controller = AbortController()
    cancelled = harness.close(with_signal(controller.signal))
    controller.abort(RuntimeError("stop waiting"))
    with pytest.raises(Exception, match="stop waiting"):
        await cancelled
    # Admission stays sealed and the invocation still holds Storage open.
    with pytest.raises(Exception, match="closed"):
        await harness.commit(lambda _tx: None, context)
    assert await storage_open(storage) is True
    second = harness.close(context)
    assert await settled(asyncio.shield(second)) is False
    resolve(gate)
    await second
    assert await storage_open(storage) is False


async def test_settles_durably_a_commit_whose_committer_was_cancelled_while_it_was_in_storage() -> None:
    storage = await memory_storage()
    opened = await open_tasks(storage, [])
    harness = opened.harness
    root = await harness.root(context)
    held = storage.hold_commits()
    controller = AbortController()

    async def change(tx: Any) -> Any:
        return (await tx.append_entry(root.id, {"kind": "note"}))["id"]

    committing = root.commit(change, with_signal(controller.signal))
    await held.entered
    controller.abort(RuntimeError("committer gave up"))
    held.release()
    id = await committing
    page = await root.entries({}, 10, None, context)
    assert [entry["id"] for entry in page["items"]] == [id]
    await harness.close(context)


async def test_publishes_no_frame_to_states_and_watches_from_a_commit_that_settles_during_close() -> None:
    Notes = define_doc(kind="test.close-frames", version=1, scope="session", initial=lambda: {"text": ""})
    storage = await memory_storage()
    opened = await open_tasks(storage, [])
    harness = opened.harness
    root = await harness.root(context)

    async def before(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "before"

    await harness.commit(before, context)
    doc_state = await harness.document_state(Notes, context)
    doc_watch = await harness.watch_doc(Notes, context)
    view_state = await root.view_state(context)
    view_watch = await root.watch(context)
    graph_state = await harness.task_graph(context)
    graph_watch = await harness.watch_task_graph(context)
    frames: list[str] = []

    def on(name: str) -> Any:
        def listener(_value: Any, _context: Any, delivery: Any) -> None:
            if delivery.kind == "update":
                frames.append(name)

        return listener

    async def push(name: str) -> None:
        frames.append(name)

    graph_state.subscribe(on("graphState"))
    graph_watch.start(lambda *_: push("graphWatch"))
    doc_state.subscribe(on("docState"))
    view_state.subscribe(on("viewState"))
    doc_watch.start(lambda *_: push("docWatch"))
    view_watch.start(lambda *_: push("viewWatch"))
    doc_value = doc_state.value
    view_value = view_state.value

    held = storage.hold_commits()

    async def during(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "during close"
        await tx.append_entry(root.id, {"kind": "note"})
        await tx.create_task(
            one_step("test.close-graph"), None, {"ownership": {"kind": "conversation"}, "conversationId": root.id}
        )

    committing = harness.commit(during, context)
    await held.entered
    closing = harness.close(context)
    held.release()
    await committing
    await closing
    await flush()
    assert frames == []
    assert doc_state.value is doc_value
    assert view_state.value is view_value
    assert graph_state.value == {"tasks": {}}
    assert await graph_watch.closed == {"reason": "session_closed"}
    assert await doc_watch.closed == {"reason": "session_closed"}
    assert await view_watch.closed == {"reason": "session_closed"}
    # The commit itself settled.
    assert any(write["type"] == "entry" for write in storage.commits[-1])


async def test_leaves_no_subscription_behind_a_watch_acquisition_cancelled_on_the_line() -> None:
    Notes = define_doc(kind="test.cancelled-watch", version=1, scope="session", initial=lambda: {"text": ""})
    storage = await memory_storage()
    opened = await open_tasks(storage, [])
    harness = opened.harness
    root = await harness.root(context)

    async def seed(tx: Any) -> None:
        (await tx.doc(Notes))["text"] = "x"

    await harness.commit(seed, context)
    # Count commit subscriptions of document observers.
    subscriptions = [0]
    subscribe = harness.observe_commits

    def counting(listener: Any) -> Any:
        subscriptions[0] += 1
        unsubscribe = subscribe(listener)
        active = [True]

        def stop() -> None:
            if active[0]:
                subscriptions[0] -= 1
            active[0] = False
            unsubscribe()

        return stop

    harness.observe_commits = counting  # type: ignore[method-assign]

    # Cancel a document watch while its acquisition loads the document on the line.
    await harness.unload_documents()
    find = storage.hold_find_document()
    controller = AbortController()
    doc_watch = harness.watch_doc(Notes, with_signal(controller.signal))
    await find.entered
    controller.abort(RuntimeError("cancelled"))
    find.release()
    with pytest.raises(Exception, match="cancelled"):
        await doc_watch
    assert subscriptions[0] == 0

    # Cancel a view watch while it builds its mount on the line.
    await harness.unload_documents()
    find = storage.hold_find_document()
    controller = AbortController()
    view_watch = root.watch(with_signal(controller.signal))
    await find.entered
    controller.abort(RuntimeError("cancelled"))
    find.release()
    with pytest.raises(Exception, match="cancelled"):
        await view_watch
    # No observer kept the mount: the next observer builds a new one, and the one after that another.
    first = await root.view_state(context)
    first_value = first.value
    first.dispose()
    second = await root.view_state(context)
    assert second.value is not first_value
    second.dispose()
    await harness.close(context)


async def _outcome(awaitable: Any) -> str:
    try:
        value = await awaitable
    except Exception as error:
        return "closed" if re.search("closed", str(error)) else str(error)
    return "undefined" if value is None else "resolved"


async def test_rejects_conversation_and_harness_operations_once_close_begins() -> None:
    storage = await memory_storage()
    opened = await open_tasks(storage, [])
    harness = opened.harness
    root = await harness.root(context)
    entry = await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    submission = await root.submit({"type": "write", "entry": {"kind": "note"}}, context)

    held = storage.hold_commits()

    async def block(tx: Any) -> None:
        await tx.append_entry(root.id, {"kind": "blocker"})

    blocking = harness.commit(block, context)
    await held.entered
    queued_inspect = harness.inspect(context)
    closing = harness.close(context)
    operations: dict[str, Any] = {
        "submit": lambda: root.submit({"type": "input", "content": "x"}, context),
        "agent": lambda: root.agent(context),
        "configure": lambda: root.configure({"thinkingLevel": "low"}, context),
        "commit": lambda: root.commit(lambda _tx: None, context),
        "context": lambda: root.context(context),
        "entries": lambda: root.entries({}, 10, None, context),
        "fork": lambda: root.fork(entry["id"], {"ownership": {"kind": "ownerless"}}, context),
        "compact": lambda: root.compact(None, context),
        "reset": lambda: root.reset(None, context),
        "abort": lambda: root.abort(context),
        "conversationIdle": lambda: root.wait_for_idle(context),
        "viewState": lambda: root.view_state(context),
        "watch": lambda: root.watch(context),
        "taskGraph": lambda: harness.task_graph(context),
        "watchTaskGraph": lambda: harness.watch_task_graph(context),
        "status": lambda: submission.status(context),
        "wait": lambda: submission.wait(context),
        "abortSubmission": lambda: submission.abort(context),
        "root": lambda: harness.root(context),
        "conversation": lambda: harness.conversation(root.id, context),
        "createConversation": lambda: harness.create_conversation({"ownership": {"kind": "ownerless"}}, context),
        "getTask": lambda: harness.get_task(1, context),  # type: ignore[arg-type]
        "inspect": lambda: harness.inspect(context),
        "submission": lambda: harness.submission(submission.id, context),
        "abortTask": lambda: harness.abort_task(1, context),  # type: ignore[arg-type]
        "waitForTask": lambda: harness.wait_for_task(1, context),  # type: ignore[arg-type]
        "harnessIdle": lambda: harness.wait_for_idle(context),
        "usage": lambda: harness.usage(context),
    }
    outcomes: dict[str, str] = {}
    for name, operation in operations.items():
        try:
            awaitable = operation()
        except Exception as error:
            outcomes[name] = "closed" if "closed" in str(error) else str(error)
            continue
        outcome = await _outcome(awaitable)
        outcomes[name] = "closed" if outcome == "closed" else outcome
    assert outcomes == dict.fromkeys(operations, "closed")
    with pytest.raises(Exception, match="closed"):
        harness.resume()
    held.release()
    await blocking
    assert (await queued_inspect)["scheduling"] == "closing"
    await closing


async def test_completes_reads_queued_at_the_seal_and_rejects_queued_waits_and_acquisitions() -> None:
    Notes = define_doc(kind="test.queued-at-seal", version=1, scope="session", initial=lambda: {"text": ""})
    Absent = define_doc(kind="test.queued-absent", version=1, scope="session", initial=lambda: {"text": ""})
    pending = one_step("test.queued-pending")
    storage = await memory_storage()
    opened = await open_tasks(storage, [])
    harness = opened.harness
    root = await harness.root(context)

    async def create(tx: Any) -> None:
        await tx.doc(Notes)

    await harness.commit(create, context)
    task_id = await start(root, pending)
    write = await root.submit({"type": "write", "entry": {"kind": "note"}}, context)
    await write.wait(context)

    held = storage.hold_commits()

    async def block(tx: Any) -> None:
        await tx.append_entry(root.id, {"kind": "blocker"})

    blocking = harness.commit(block, context)
    await held.entered
    queued = {
        "context": asyncio.ensure_future(_outcome(root.context(context))),
        "snapshot": asyncio.ensure_future(_outcome(harness.snapshot(Notes, context))),
        "settledWait": asyncio.ensure_future(_outcome(write.wait(context))),
        "absentWatch": asyncio.ensure_future(_outcome(harness.watch_doc(Absent, context))),
        "absentState": asyncio.ensure_future(_outcome(harness.document_state(Absent, context))),
        "waitForTask": asyncio.ensure_future(_outcome(harness.wait_for_task(task_id, context))),
        "documentState": asyncio.ensure_future(_outcome(harness.document_state(Notes, context))),
        "watchDoc": asyncio.ensure_future(_outcome(harness.watch_doc(Notes, context))),
        "viewState": asyncio.ensure_future(_outcome(root.view_state(context))),
        "watch": asyncio.ensure_future(_outcome(root.watch(context))),
        "taskGraph": asyncio.ensure_future(_outcome(harness.task_graph(context))),
    }
    closing = harness.close(context)
    held.release()
    await blocking
    outcomes = {name: await outcome for name, outcome in queued.items()}
    assert outcomes == {
        "context": "resolved",
        "snapshot": "resolved",
        "settledWait": "resolved",
        "absentWatch": "undefined",
        "absentState": "undefined",
        "waitForTask": "closed",
        "documentState": "closed",
        "watchDoc": "closed",
        "viewState": "closed",
        "watch": "closed",
        "taskGraph": "closed",
    }
    await closing


async def _paused() -> dict[str, Any]:
    """Open a paused Harness with one background task pending and a queued write submission."""
    ran = [False]

    async def run() -> None:
        ran[0] = True

    marker = one_step("test.marker", run)
    opened = await open_tasks(MemoryStorage(), [marker])
    harness = opened.harness
    root = await harness.root(context)
    # Background, so idle waits and conversation abort leave it alone.
    id = await start(root, marker, True)

    async def queue(tx: Any) -> Any:
        return (await tx.create_submission({"conversationId": root.id, "type": "write", "status": "queued"}))["id"]

    submission_id = await harness.commit(queue, context)
    return {"harness": harness, "root": root, "id": id, "submissionId": submission_id, "ran": lambda: ran[0]}


async def test_never_schedules_from_a_read_only_viewer() -> None:
    opened = await _paused()
    harness: Harness = opened["harness"]
    root = opened["root"]
    Notes = define_doc(kind="test.viewer-notes", version=1, scope="session", initial=lambda: {"text": ""})

    async def create(tx: Any) -> None:
        await tx.doc(Notes)

    await harness.commit(create, context)
    await harness.inspect(context)
    await harness.get_task(opened["id"], context)
    await harness.usage(context)
    await harness.conversation(root.id, context)
    await harness.snapshot(Notes, context)
    (await harness.document_state(Notes, context)).dispose()
    await (await harness.watch_doc(Notes, context)).stop()
    submission = await harness.submission(opened["submissionId"], context)
    assert submission is not None
    await submission.status(context)
    await root.agent(context)
    await root.context(context)
    await root.entries({}, 10, None, context)
    (await root.view_state(context)).dispose()
    await (await root.watch(context)).stop()
    (await harness.task_graph(context)).dispose()
    await (await harness.watch_task_graph(context)).stop()
    await flush()
    await flush()
    assert opened["ran"]() is False
    assert (await harness.inspect(context))["scheduling"] == "paused"
    await harness.close(context)


async def _submission_wait(opened: dict[str, Any]) -> Any:
    submission = await opened["harness"].submission(opened["submissionId"], context)
    return await submission.wait(context)


@pytest.mark.parametrize(
    "call",
    [
        lambda opened: opened["root"].submit({"type": "write", "entry": {"kind": "note"}}, context),
        lambda opened: opened["root"].compact(None, context),
        lambda opened: opened["root"].abort(context),
        lambda opened: opened["root"].wait_for_idle(context),
        _submission_wait,
        lambda opened: opened["harness"].wait_for_task(opened["id"], context),
        lambda opened: opened["harness"].wait_for_idle(context),
    ],
    ids=["submit", "compact", "abort", "conversationIdle", "submissionWait", "waitForTask", "harnessIdle"],
)
async def test_schedules_from_every_progress_call(call: Any) -> None:
    opened = await _paused()

    async def pending_call() -> None:
        # A queued write's wait settles only on placement; close rejects it.
        try:
            await call(opened)
        except Exception:
            pass

    pending = asyncio.ensure_future(pending_call())
    await eventually(lambda: opened["ran"]())
    await opened["harness"].close(context)
    await pending


async def test_runs_what_the_registry_holds_at_resume_a_definition_installed_or_replaced_after_open() -> None:
    log: list[str] = []

    def define(label: str) -> Any:
        async def run() -> None:
            log.append(label)

        return one_step("test.late-definition", run)

    storage = MemoryStorage()
    first = await open_tasks(storage, [])
    root = await first.harness.root(context)
    missing = await start(root, define("unused"))
    tasks = (await first.harness.inspect(context))["tasks"]
    assert [task["state"]["kind"] for task in tasks] == ["blocked"]
    assert tasks[0]["state"]["reason"] == "missing_task"  # type: ignore[typeddict-item]
    installed = add_task(first.registry, define("v1"))
    assert [task["state"]["kind"] for task in (await first.harness.inspect(context))["tasks"]] == ["ready"]
    # The same extension name replaces the definition in place, still before resume.
    installed.dispose()
    add_task(first.registry, define("v2"))
    first.harness.resume()
    await first.harness.wait_for_task(missing, context)
    assert log == ["v2"]
    await first.harness.close(context)
