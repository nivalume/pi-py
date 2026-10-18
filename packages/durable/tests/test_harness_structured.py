"""Port of ``test/harness-structured.test.ts``: task ownership, waits, holds, abort order, and recovery."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import (
    ChatSetup,
    aborted,
    add_hooks,
    add_task,
    add_tool,
    all_entries,
    assert_match,
    chat_setup,
    open_chat,
    open_tasks,
    settled,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import TextContent, Usage
from pi_chord.context import Context
from pi_durable import (
    GenerationTask,
    Harness,
    LiveDoc,
    MemoryStorage,
    SessionFailed,
    ToolResultEntry,
    create_session,
    define_doc,
    define_task,
    define_tool,
    watch_events,
)
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

# ─── A scriptable task ──────────────────────────────────────────────────────

#: Per-name behavior of ``Node``: ``run(runtime, ctx)``, ``resume(runtime, ctx, round)``, ``abort(runtime, ctx)``.
behaviors: dict[str, dict[str, Callable[..., Awaitable[Any]]]] = {}
#: Gates resolved with an ending: "completed", "failed", or "throw".
gates: dict[str, asyncio.Future[Any]] = {}
#: Handler starts and ends in order, such as ``run:a``, ``abort:a``.
log: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    yield
    behaviors.clear()
    gates.clear()
    log.clear()


def gate(name: str) -> asyncio.Future[Any]:
    found = gates.get(name)
    if found is None:
        found = deferred()
        gates[name] = found
    return found


def open_gate(name: str, ending: str = "completed") -> None:
    resolve(gate(name), ending)


def script(name: str, **behavior: Callable[..., Awaitable[Any]]) -> None:
    behaviors[name] = behavior


async def race(future: asyncio.Future[Any], signal: Any) -> Any:
    """``Promise.race([future, aborted(signal)])``."""
    cancelled = asyncio.ensure_future(aborted(signal))
    try:
        await asyncio.wait([future, cancelled], return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not cancelled.done():
            cancelled.cancel()
    if future.done():
        return future.result()
    return cancelled.result()


async def default_run(runtime: Any, ctx: Context, name: str) -> None:
    """Wait for the gate or the abort signal, then end as the gate says."""
    ending = await race(gate(name), runtime.signal)
    if ending == "throw":
        raise RuntimeError(f"{name} threw")
    await runtime.commit(lambda _tx, _current: end(ending, name), ctx)


def end(ending: str, name: str) -> Any:
    if ending == "completed":
        return {"status": "terminal", "outcome": {"status": "completed", "result": name}}
    return {"status": "terminal", "outcome": {"status": "failed", "error": {"message": f"{name} failed"}}}


def terminal(status: str, **extra: Any) -> Any:
    return {"status": "terminal", "outcome": {"status": status, **extra}}


def commit_aborted(runtime: Any, ctx: Context, **extra: Any) -> Awaitable[Any]:
    return runtime.commit(lambda _tx, _current: terminal("aborted", **extra), ctx)


async def _node_run(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    log.append(f"run:{name}")
    behavior = behaviors.get(name, {}).get("run")
    await (default_run(runtime, ctx, name) if behavior is None else behavior(runtime, ctx))


async def _node_resume(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    log.append(f"resume:{name}")
    behavior = behaviors.get(name, {}).get("resume")
    if behavior is not None:
        await behavior(runtime, ctx, task["state"]["checkpoint"]["round"])
        return
    await runtime.commit(lambda _tx, _current: end("completed", name), ctx)


async def _node_abort(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    log.append(f"abort:{name}")
    behavior = behaviors.get(name, {}).get("abort")
    if behavior is not None:
        await behavior(runtime, ctx)
        return
    await commit_aborted(runtime, ctx, result=name)


#: A task that runs its name's behavior.
Node: Any = define_task(
    name="test.node",
    version=1,
    initial=lambda _input: {"phase": "run"},
    phases={"run": _node_run, "resume": _node_resume},
    abort=_node_abort,
)


async def _nothing(*_args: Any) -> None:
    return None


#: Never registered: aborting it can only orphan it.
Unregistered: Any = define_task(
    name="test.unregistered",
    version=1,
    initial=lambda _input: {"phase": "run"},
    phases={"run": _nothing},
    abort=_nothing,
)

TaskNotes: Any = define_doc(kind="test.task-notes", version=1, scope="task", initial=lambda: {"text": ""})

OWN_CONVERSATION: Any = {"ownership": {"kind": "conversation"}}


def owned(owner: Any) -> Any:
    return {"ownership": {"kind": "task", "taskId": owner}}


def spawn(tx: Any, owner: Any, name: str) -> Awaitable[Any]:
    """A child task named ``name``, owned by ``owner``."""
    return tx.create_task(Node, {"name": name}, owned(owner))


def wait_on(on: Sequence[Any], policy: str, round: int = 1) -> Any:
    """Wait on ``on``, resuming in round ``round``."""
    return {"status": "waiting", "checkpoint": {"phase": "resume", "round": round}, "on": list(on), "policy": policy}


def running_resume(round: int = 1) -> Any:
    return {"status": "running", "checkpoint": {"phase": "resume", "round": round}}


def committing(next_state: Callable[[], Any]) -> Callable[[Any, Context], Awaitable[Any]]:
    """A behavior that commits ``next_state()``."""

    def behavior(runtime: Any, ctx: Context, *_rest: Any) -> Awaitable[Any]:
        return runtime.commit(lambda _tx, _current: next_state(), ctx)

    return behavior


def gated_abort(name: str) -> Callable[[Any, Context], Awaitable[Any]]:
    """An abort behavior that waits for the gate ``name`` and then aborts."""

    async def behavior(runtime: Any, ctx: Context) -> None:
        await gate(name)
        await commit_aborted(runtime, ctx)

    return behavior


async def start(conversation: Any, name: str, options: Any = OWN_CONVERSATION) -> Any:
    return await conversation.commit(lambda tx: tx.create_task(Node, {"name": name}, options), context)


@dataclasses.dataclass(slots=True)
class Opened:
    harness: Harness
    root: Any
    registry: Any
    reports: list[BaseException]


async def open_nodes(storage: Any = None) -> Opened:
    opened = await open_tasks(storage if storage is not None else MemoryStorage(), [Node])
    root = await opened.harness.root(context)
    opened.harness.resume()
    return Opened(opened.harness, root, opened.registry, opened.reports)


async def state(harness: Harness, id: Any) -> Any:
    record = await harness.get_task(id, context)
    assert record is not None
    return record["state"]


async def outcome_of(harness: Harness, id: Any) -> str:
    return (await harness.wait_for_task(id, context))["state"]["outcome"]["status"]


async def until(check: Callable[[], bool | Awaitable[bool]]) -> None:
    await wait_for(check)


def status_is(harness: Harness, id: Callable[[], Any] | Any, status: str) -> Callable[[], Awaitable[bool]]:
    async def check() -> bool:
        return (await state(harness, id() if callable(id) else id))["status"] == status

    return check


async def faulted_with(harness: Harness, id: Any, message: str) -> None:
    outcome = (await harness.wait_for_task(id, context))["state"]["outcome"]
    assert outcome["status"] == "faulted", outcome
    assert message in outcome["error"]["message"], outcome


async def abort_requested(harness: Harness, id: Any) -> bool:
    record = await harness.get_task(id, context)
    assert record is not None
    return record["abortRequested"]


async def inspected_state(harness: Harness, id: Any) -> Any:
    found = next((task for task in (await harness.inspect(context))["tasks"] if task["record"]["id"] == id), None)
    return None if found is None else found["state"]


# ─── Ownership ──────────────────────────────────────────────────────────────


async def test_creates_a_child_in_its_owners_conversation_with_its_owner_recorded() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            return wait_on(child, "allSettled")

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(lambda: len(child) > 0)
    record = await harness.get_task(child[0], context)
    assert_match(record, {"owner": parent, "conversationId": root.id, "background": False})
    assert "owner" not in (await harness.get_task(parent, context))  # type: ignore[operator]
    open_gate("child")
    assert await outcome_of(harness, parent) == "completed"
    await harness.close(context)


async def test_rejects_no_ownership_a_missing_owner_a_child_in_another_conversation_and_a_background_child() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    parent = await start(root, "parent")
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)

    def create(options: Any) -> Awaitable[Any]:
        return root.commit(lambda tx: tx.create_task(Node, {"name": "x"}, options), context)

    with pytest.raises(Exception):  # noqa: B017
        await create({})
    with pytest.raises(Exception, match="does not exist"):
        await create(owned(999_999))
    with pytest.raises(Exception, match="owner's conversation"):
        await create({**owned(parent), "conversationId": other.id})
    with pytest.raises(Exception, match="cannot be background"):
        await create({**owned(parent), "background": True})
    # The owner's conversation is the default, even from a commit bound to another conversation.
    child = await other.commit(lambda tx: spawn(tx, parent, "child"), context)
    assert (await harness.get_task(child, context))["conversationId"] == root.id  # type: ignore[index]
    open_gate("parent")
    open_gate("child")
    await harness.wait_for_task(parent, context)
    await harness.close(context)


async def test_rejects_new_owned_work_below_an_owner_that_is_completing_terminal_or_abort_marked() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            await spawn(tx, runtime.task_id, "child")
            return running_resume()

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "completing"))

    def create_child() -> Awaitable[Any]:
        return root.commit(lambda tx: spawn(tx, parent, "late"), context)

    def create_conversation() -> Awaitable[Any]:
        return root.commit(
            lambda tx: tx.create_conversation({"ownership": {"kind": "task", "taskId": parent}}), context
        )

    with pytest.raises(Exception, match="is completing"):
        await create_child()
    with pytest.raises(Exception, match="is completing"):
        await create_conversation()
    open_gate("child")
    await harness.wait_for_task(parent, context)
    with pytest.raises(Exception, match="is terminal"):
        await create_child()
    with pytest.raises(Exception, match="is terminal"):
        await create_conversation()

    script("slow", abort=gated_abort("abort.slow"))
    slow = await start(root, "slow")
    await until(lambda: "run:slow" in log)
    await harness.abort_task(slow, context)
    with pytest.raises(Exception, match="is abort-marked"):
        await root.commit(lambda tx: spawn(tx, slow, "late"), context)
    open_gate("abort.slow")
    assert await outcome_of(harness, slow) == "aborted"
    await harness.close(context)


async def test_cannot_create_a_child_in_its_finishing_commit_but_work_it_starts_in_an_owned_conversation_holds_it() -> (
    None
):
    opened = await open_nodes()
    harness, root = opened.harness, opened.root

    async def eager_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            await spawn(tx, runtime.task_id, "never")
            return end("completed", "eager")

        await runtime.commit(change, ctx)

    script("eager", run=eager_run)
    eager = await start(root, "eager")
    await faulted_with(harness, eager, "is completing")
    # The rejected commit wrote nothing.
    nodes = await harness.commit(lambda tx: tx.scan_tasks({"kind": "test.node"}, 20), context)
    assert [task["input"]["name"] for task in nodes["items"]] == ["eager"]

    child: list[Any] = []

    async def host_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(
                (await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}}))["id"]
            )
            return running_resume()

        await runtime.commit(change, ctx)

    async def host_resume(runtime: Any, ctx: Context, _round: int) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            await tx.create_task(Node, {"name": "inner"}, {**OWN_CONVERSATION, "conversationId": child[0]})
            return end("completed", "host")

        await runtime.commit(change, ctx)

    script("host", run=host_run, resume=host_resume)
    host = await start(root, "host")
    await until(status_is(harness, host, "completing"))
    assert await settled(harness.wait_for_task(host, context)) is False
    open_gate("inner")
    assert await outcome_of(harness, host) == "completed"
    await harness.close(context)


# ─── Waiting ────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Found:
    ids: list[Any] = dataclasses.field(default_factory=list)
    outcomes: list[str] = dataclasses.field(default_factory=list)


def parent_of(parent: str, children: Sequence[str], policy: str) -> Found:
    """Script ``parent`` to spawn ``children`` in one commit and wait on them with ``policy``; it records their
    outcomes."""
    found = Found()

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            for name in children:
                found.ids.append(await spawn(tx, runtime.task_id, name))
            return wait_on(found.ids, policy)

        await runtime.commit(change, ctx)

    async def resume(runtime: Any, ctx: Context, _round: int) -> None:
        found.outcomes = [outcome["status"] for outcome in await runtime.outcomes(found.ids, ctx)]
        await runtime.commit(lambda _tx, _current: end("completed", parent), ctx)

    script(parent, run=run, resume=resume)
    return found


async def test_resumes_once_every_awaited_task_is_terminal_and_reads_their_outcomes_in_order_all_settled() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    found = Found()

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            for name in ("ok", "fails", "throws", "aborted"):
                found.ids.append(await spawn(tx, runtime.task_id, name))
            found.ids.append(await tx.create_task(Unregistered, {"name": "orphan"}, owned(runtime.task_id)))
            return wait_on(found.ids, "allSettled")

        await runtime.commit(change, ctx)

    async def resume(runtime: Any, ctx: Context, _round: int) -> None:
        found.outcomes = [outcome["status"] for outcome in await runtime.outcomes(found.ids, ctx)]
        await runtime.commit(lambda _tx, _current: end("completed", "parent"), ctx)

    script("parent", run=run, resume=resume)
    parent = await start(root, "parent")
    await until(lambda: len(found.ids) == 5)
    await until(status_is(harness, parent, "waiting"))
    open_gate("ok")
    open_gate("fails", "failed")
    open_gate("throws", "throw")
    await harness.abort_task(found.ids[3], context)
    assert await harness.abort_task(found.ids[4], context) == "marked"
    assert await outcome_of(harness, parent) == "completed"
    assert found.outcomes == ["completed", "failed", "faulted", "aborted", "orphaned"]
    # allSettled never marks siblings.
    assert [line for line in log if line.startswith("abort:")] == ["abort:aborted"]
    await harness.close(context)


@pytest.mark.parametrize("ending", ["failed", "throw"])
async def test_fails_fast_when_a_child_ends_failed_or_faulted_its_live_siblings_are_aborted_the_parent_is_not(
    ending: str,
) -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    found = parent_of("checkout", ["p1", "p2", "p3", "p4"], "failFast")
    parent = await start(root, "checkout")
    await until(status_is(harness, parent, "waiting"))
    open_gate("p2", ending)
    assert await outcome_of(harness, parent) == "completed"
    assert found.outcomes == ["aborted", "faulted" if ending == "throw" else "failed", "aborted", "aborted"]
    assert (await harness.get_task(parent, context))["abortRequested"] is False  # type: ignore[index]
    assert sorted(line for line in log if line.startswith("abort:")) == ["abort:p1", "abort:p3", "abort:p4"]
    await harness.close(context)


async def test_fails_fast_on_a_held_failure_before_the_failing_child_drains() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    found = Found()

    async def p1_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            await spawn(tx, runtime.task_id, "grandchild")
            return running_resume()

        await runtime.commit(change, ctx)

    script("p1", run=p1_run, resume=committing(lambda: end("failed", "p1")))
    script("grandchild", abort=gated_abort("abort.grandchild"))
    outside: list[Any] = []

    async def parent_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            for name in ("p1", "p2"):
                found.ids.append(await spawn(tx, runtime.task_id, name))
            outside.append(await spawn(tx, runtime.task_id, "outside"))
            return wait_on(found.ids, "failFast")

        await runtime.commit(change, ctx)

    script("parent", run=parent_run)
    parent = await start(root, "parent")
    await until(lambda: len(found.ids) == 2)
    # p1 holds `failed` while its grandchild's abort handler runs; p2 is already aborted.
    assert await outcome_of(harness, found.ids[1]) == "aborted"
    assert_match(await state(harness, found.ids[0]), {"status": "completing", "outcome": {"status": "failed"}})
    assert (await state(harness, parent))["status"] == "waiting"
    open_gate("abort.grandchild")
    await until(lambda: "resume:parent" in log)
    # Only the other tasks in `on` are marked: not the failed one, not the parent, not a child outside `on`.
    assert await outcome_of(harness, found.ids[0]) == "failed"
    assert [
        await abort_requested(harness, found.ids[0]),
        await abort_requested(harness, found.ids[1]),
        await abort_requested(harness, outside[0]),
    ] == [False, True, False]
    assert await abort_requested(harness, parent) is False
    assert (await state(harness, outside[0]))["status"] == "running"
    # Finished, the parent holds for the child outside `on`.
    await until(status_is(harness, parent, "completing"))
    open_gate("outside")
    assert await outcome_of(harness, parent) == "completed"
    await harness.close(context)


async def test_waits_with_all_settled_on_tasks_it_does_not_own_including_already_terminal_ones() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    done = await start(root, "done")
    open_gate("done")
    await harness.wait_for_task(done, context)
    live = await start(root, "live")
    outcomes: list[str] = []

    async def resume(runtime: Any, ctx: Context, _round: int) -> None:
        outcomes.extend(outcome["status"] for outcome in await runtime.outcomes([done, live], ctx))
        await runtime.commit(lambda _tx, _current: end("completed", "parent"), ctx)

    script("parent", run=committing(lambda: wait_on([done, live], "allSettled")), resume=resume)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "waiting"))
    # A task it does not own is not its work: the parent may finish while it lives, but here it waits for it.
    open_gate("live", "failed")
    assert await outcome_of(harness, parent) == "completed"
    assert outcomes == ["completed", "failed"]
    await harness.close(context)


async def test_rejects_waits_on_itself_its_owner_a_missing_task_and_fail_fast_on_a_task_it_does_not_own() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    other = await start(root, "other")
    cases: list[tuple[str, Callable[[Any], list[Any]], str, str]] = [
        ("self", lambda runtime: [runtime.task_id], "allSettled", "is a wait cycle"),
        ("missing", lambda _runtime: [999_999], "allSettled", "does not exist"),
        ("foreign", lambda _runtime: [other], "failFast", "only on tasks it owns"),
    ]
    for name, on, policy, message in cases:

        def make(on: Callable[[Any], list[Any]], policy: str) -> Callable[[Any, Context], Awaitable[Any]]:
            def run(runtime: Any, ctx: Context) -> Awaitable[Any]:
                return runtime.commit(lambda _tx, _current: wait_on(on(runtime), policy), ctx)

            return run

        script(name, run=make(on, policy))
        id = await start(root, name)
        await faulted_with(harness, id, message)
    # A child waiting on its owner could never resume.
    child: list[Any] = []
    parent: list[Any] = []
    script("child", run=committing(lambda: wait_on([parent[0]], "allSettled")))

    async def parent_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            return wait_on(child, "allSettled")

        await runtime.commit(change, ctx)

    script("parent", run=parent_run)
    parent.append(await start(root, "parent"))
    assert (await harness.wait_for_task(parent[0], context))["state"]["outcome"]["status"] == "completed"
    await faulted_with(harness, child[0], "is a wait cycle")
    open_gate("other")
    await harness.close(context)


async def create_nodes(root: Any, ids: dict[str, Any], names: Sequence[str]) -> None:
    async def change(tx: Any) -> None:
        for name in names:
            ids[name] = await tx.create_task(Node, {"name": name}, OWN_CONVERSATION)

    await root.commit(change, context)


def waits_for(ids: dict[str, Any], name: str) -> Callable[[Any, Context], Awaitable[Any]]:
    """A run behavior that waits on ``ids[name]`` with allSettled."""
    return committing(lambda: wait_on([ids[name]], "allSettled"))


# #10411; this test and the next two are from #10533 by @odysseus0.
async def test_rejects_the_wait_that_closes_a_cycle() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    script("a", run=waits_for(ids, "b"))
    script("b", run=waits_for(ids, "a"))
    await create_nodes(root, ids, ["a", "b"])
    # a -> b and b -> a are waits, and whichever comes second closes the cycle.
    outcomes = await asyncio.gather(outcome_of(harness, ids["a"]), outcome_of(harness, ids["b"]))
    assert sorted(outcomes) == ["completed", "faulted"]
    await harness.close(context)


# #10411
async def test_rejects_the_wait_that_closes_a_cycle_through_a_held_owner() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    child = spawn_and_finish("held", "child")
    script("child", run=waits_for(ids, "other"))
    script("other", run=waits_for(ids, "held"))
    await create_nodes(root, ids, ["held", "other"])
    # held -> child is a hold; child -> other -> held are waits, and whichever wait comes second closes the cycle.
    assert await outcome_of(harness, ids["held"]) == "completed"
    outcomes = await asyncio.gather(outcome_of(harness, child.ids[0]), outcome_of(harness, ids["other"]))
    assert sorted(outcomes) == ["completed", "faulted"]
    await harness.close(context)


# #10411
async def test_rejects_a_cycle_through_holds_two_levels_deep() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    child = spawn_and_finish("held", "child")
    grandchild = spawn_and_finish("child", "grandchild")
    script("grandchild", run=waits_for(ids, "other"))
    script("other", run=waits_for(ids, "held"))
    await create_nodes(root, ids, ["held", "other"])
    # held -> child -> grandchild are holds; grandchild -> other -> held are waits.
    await until(lambda: len(grandchild.ids) > 0)
    outcomes = await asyncio.gather(outcome_of(harness, grandchild.ids[0]), outcome_of(harness, ids["other"]))
    assert sorted(outcomes) == ["completed", "faulted"]
    assert await outcome_of(harness, child.ids[0]) == "completed"
    await harness.close(context)


# #10411
async def test_lets_a_background_task_wait_on_the_owner_of_its_conversation() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    script("background", run=waits_for(ids, "owner"))

    async def change(tx: Any) -> Any:
        ids["owner"] = await tx.create_task(Node, {"name": "owner"}, OWN_CONVERSATION)
        conversation = await tx.create_conversation({"ownership": {"kind": "task", "taskId": ids["owner"]}})
        options = {**OWN_CONVERSATION, "conversationId": conversation["id"], "background": True}
        return await tx.create_task(Node, {"name": "background"}, options)

    background = await root.commit(change, context)
    # background -> owner is a wait but owner -> background is no hold, so there is no cycle.
    await until(status_is(harness, background, "waiting"))
    open_gate("owner")
    assert await outcome_of(harness, background) == "completed"
    await harness.close(context)


# #10411: the issue's orchestrator, worker, and reviewer.
async def test_rejects_a_cycle_of_a_hold_and_two_waits_that_each_look_reasonable() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    reviewer_waits = deferred()

    # The orchestrator spawns a worker and finishes; the running worker holds it as completing.
    async def orchestrator_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            ids["worker"] = await spawn(tx, runtime.task_id, "worker")
            return running_resume()

        await runtime.commit(change, ctx)

    # The reviewer waits for the orchestrator's result.
    async def reviewer_run(runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: wait_on([ids["orchestrator"]], "allSettled"), ctx)
        resolve(reviewer_waits)

    # The worker waits for the reviewer, closing the cycle.
    async def worker_run(runtime: Any, ctx: Context) -> None:
        await reviewer_waits
        await runtime.commit(lambda _tx, _current: wait_on([ids["reviewer"]], "allSettled"), ctx)

    script("orchestrator", run=orchestrator_run)
    script("reviewer", run=reviewer_run)
    script("worker", run=worker_run)
    ids["orchestrator"] = await start(root, "orchestrator")
    await until(status_is(harness, ids["orchestrator"], "completing"))
    ids["reviewer"] = await start(root, "reviewer")
    outcomes = await asyncio.gather(
        *(outcome_of(harness, ids[name]) for name in ("orchestrator", "worker", "reviewer"))
    )
    assert list(outcomes) == ["completed", "faulted", "completed"]
    worker = (await harness.wait_for_task(ids["worker"], context))["state"]["outcome"]
    assert (
        f"{ids['worker']} -> {ids['reviewer']} -> {ids['orchestrator']} -> {ids['worker']} is a wait cycle"
        in json.dumps(worker)
    )
    await harness.close(context)


# #10411
async def test_allows_a_diamond_of_waits() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    # A diamond: top waits on left and right, both wait on bottom. No cycle.
    script("top", run=committing(lambda: wait_on([ids["left"], ids["right"]], "allSettled")))
    script("left", run=waits_for(ids, "bottom"))
    script("right", run=waits_for(ids, "bottom"))
    await create_nodes(root, ids, ["bottom", "left", "right", "top"])
    open_gate("bottom")
    assert await outcome_of(harness, ids["top"]) == "completed"
    await harness.close(context)


# #10411
async def test_follows_no_wait_of_an_abort_marked_task_which_waits_only_for_its_owned_work() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    cleanup = deferred()

    # b owns c and waits on a; aborting b leaves c's cleanup pending, so b waits only for c.
    async def b_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            ids["c"] = await spawn(tx, runtime.task_id, "c")
            return wait_on([ids["a"]], "allSettled")

        await runtime.commit(change, ctx)

    async def c_abort(runtime: Any, ctx: Context) -> None:
        await cleanup
        await commit_aborted(runtime, ctx, result="c")

    a_waits = deferred()

    async def a_run(runtime: Any, ctx: Context) -> None:
        await a_waits
        await runtime.commit(lambda _tx, _current: wait_on([ids["b"]], "allSettled"), ctx)

    script("b", run=b_run)
    script("c", abort=c_abort)
    script("a", run=a_run, resume=committing(lambda: end("completed", "a")))
    await create_nodes(root, ids, ["a", "b"])
    await until(status_is(harness, ids["b"], "waiting"))
    assert await harness.abort_task(ids["b"], context) == "marked"
    await until(lambda: "abort:c" in log)
    # a -> b would close a -> b -> a through b's stale wait; b now waits only for c.
    resolve(a_waits)
    await until(status_is(harness, ids["a"], "waiting"))
    resolve(cleanup)
    assert await outcome_of(harness, ids["b"]) == "aborted"
    assert await outcome_of(harness, ids["a"]) == "completed"
    await harness.close(context)


# #10411
async def test_rejects_a_longer_cycle_of_waits() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: dict[str, Any] = {}
    names = ["x", "y", "z"]
    for index, name in enumerate(names):
        script(name, run=waits_for(ids, names[(index + 1) % len(names)]))
    await create_nodes(root, ids, names)
    outcomes = await asyncio.gather(*(outcome_of(harness, ids[name]) for name in names))
    assert len([outcome for outcome in outcomes if outcome == "faulted"]) == 1
    assert len([outcome for outcome in outcomes if outcome == "completed"]) == 2
    await harness.close(context)


async def test_resumes_at_the_next_pass_when_it_waits_on_nothing() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    script("parent", run=committing(lambda: wait_on([], "failFast")))
    assert await outcome_of(harness, await start(root, "parent")) == "completed"
    assert log == ["run:parent", "resume:parent"]
    await harness.close(context)


async def test_keeps_running_phases_after_spawning_and_waits_on_subsets_in_sequence() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    ids: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            for name in ("a", "b", "c"):
                ids.append(await spawn(tx, runtime.task_id, name))
            return running_resume(0)

        await runtime.commit(change, ctx)

    async def resume(runtime: Any, ctx: Context, round: int) -> None:
        log.append(f"round:{round}")
        if round == 0:
            await runtime.commit(lambda _tx, _current: wait_on([ids[0]], "allSettled", 1), ctx)
            return
        if round == 1:
            await runtime.commit(lambda _tx, _current: wait_on([ids[1], ids[2]], "failFast", 2), ctx)
            return
        await runtime.commit(lambda _tx, _current: end("completed", "parent"), ctx)

    script("parent", run=run, resume=resume)
    parent = await start(root, "parent")
    await until(lambda: "round:0" in log)
    await until(status_is(harness, parent, "waiting"))
    open_gate("b")
    assert await settled(harness.wait_for_task(ids[1], context)) is True
    assert (await state(harness, parent))["status"] == "waiting"
    open_gate("a")
    await until(lambda: "round:1" in log)
    open_gate("c")
    assert await outcome_of(harness, parent) == "completed"
    assert [line for line in log if line.startswith("round:")] == ["round:0", "round:1", "round:2"]
    await harness.close(context)


# ─── Completing ─────────────────────────────────────────────────────────────


async def test_holds_a_finished_task_until_its_owned_work_drains_waiters_and_task_documents_wait_for_the_final_commit() -> (
    None
):
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            (await tx.doc(TaskNotes, runtime.task_id))["text"] = "notes"
            child.append(await spawn(tx, runtime.task_id, "child"))
            return running_resume()

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "completing"))
    assert await state(harness, parent) == {
        "status": "completing",
        "outcome": {"status": "completed", "result": "parent"},
    }
    waiter = asyncio.ensure_future(harness.wait_for_task(parent, context))
    assert await settled(waiter) is False
    assert await harness.snapshot(TaskNotes, parent, context) == {"text": "notes"}
    assert await inspected_state(harness, parent) == {"kind": "completing"}
    assert await settled(root.wait_for_idle(context)) is False
    open_gate("child")
    assert (await waiter)["state"]["outcome"] == {"status": "completed", "result": "parent"}
    assert await harness.snapshot(TaskNotes, parent, context) is None
    await root.wait_for_idle(context)
    assert len(child) == 1
    await harness.close(context)


async def test_keeps_holding_for_ordinary_work_created_during_the_hold() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    conversation: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            conversation.append(
                (await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}}))["id"]
            )
            await tx.create_task(Node, {"name": "first"}, {**OWN_CONVERSATION, "conversationId": conversation[0]})
            return running_resume()

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "completing"))
    child = await harness.conversation(conversation[0], context)
    await start(child, "second")
    open_gate("first")
    await until(lambda: "run:second" in log)
    assert (await state(harness, parent))["status"] == "completing"
    open_gate("second")
    assert await outcome_of(harness, parent) == "completed"
    await harness.close(context)


@pytest.mark.parametrize("ending", ["failed", "throw"], ids=["a held failure", "a held scheduler fault"])
async def test_aborts_the_work_below_a_held_failure_then_finishes_with_the_held_outcome(ending: str) -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            return running_resume()

        await runtime.commit(change, ctx)

    async def resume(runtime: Any, ctx: Context, _round: int) -> None:
        if ending == "throw":
            raise RuntimeError("parent threw")
        await runtime.commit(lambda _tx, _current: end("failed", "parent"), ctx)

    script("parent", run=run, resume=resume)
    parent = await start(root, "parent")
    await until(lambda: len(child) > 0)
    assert await outcome_of(harness, child[0]) == "aborted"
    assert await outcome_of(harness, parent) == ("faulted" if ending == "throw" else "failed")
    await harness.close(context)


async def test_only_marks_a_completing_task_when_aborted_the_work_below_is_aborted_the_held_outcome_stays() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            return running_resume()

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "completing"))
    assert await harness.abort_task(parent, context) == "marked"
    assert await outcome_of(harness, child[0]) == "aborted"
    settled_parent = await harness.wait_for_task(parent, context)
    assert settled_parent["state"]["outcome"] == {"status": "completed", "result": "parent"}
    assert settled_parent["abortRequested"] is True
    assert "abort:parent" not in log
    assert await harness.abort_task(parent, context) == "terminal"
    await harness.close(context)


# ─── Abort order ────────────────────────────────────────────────────────────


def chain(harness: Callable[[], Harness], name: str, child: str) -> Found:
    """Script ``name`` to spawn ``child`` and wait on it; its abort handler records the child's status first."""
    found = Found()

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            found.ids.append(await spawn(tx, runtime.task_id, child))
            return wait_on(found.ids, "allSettled")

        await runtime.commit(change, ctx)

    async def abort(runtime: Any, ctx: Context) -> None:
        log.append(f"{name} saw {(await state(harness(), found.ids[0]))['status']}")
        await commit_aborted(runtime, ctx)

    script(name, run=run, abort=abort)
    return found


async def test_runs_abort_handlers_bottom_up_across_three_levels_each_after_the_level_below_is_terminal() -> None:
    holder: list[Opened] = []
    b = chain(lambda: holder[0].harness, "a", "b")
    c = chain(lambda: holder[0].harness, "b", "c")
    holder.append(await open_nodes())
    harness, root = holder[0].harness, holder[0].root
    a = await start(root, "a")
    await until(lambda: "run:c" in log)
    await until(status_is(harness, lambda: c.ids[0], "running"))
    assert await harness.abort_task(a, context) == "marked"
    assert await outcome_of(harness, a) == "aborted"
    assert [line for line in log if line.startswith("abort:") or " saw " in line] == [
        "abort:c",
        "abort:b",
        "b saw terminal",
        "abort:a",
        "a saw terminal",
    ]
    assert len(b.ids) == 1
    await harness.close(context)


async def test_does_not_wait_for_a_task_it_waits_on_but_does_not_own() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    other = await start(root, "other")
    script("parent", run=committing(lambda: wait_on([other], "allSettled")))
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "waiting"))
    await harness.abort_task(parent, context)
    assert await outcome_of(harness, parent) == "aborted"
    assert (await state(harness, other))["status"] == "running"
    open_gate("other")
    await harness.close(context)


async def test_reports_an_abort_marked_task_as_waiting_for_its_live_owned_work() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child: list[Any] = []
    script("child", abort=gated_abort("abort.child"))

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            return wait_on(child, "allSettled")

        await runtime.commit(change, ctx)

    script("parent", run=run)
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "waiting"))
    await harness.abort_task(parent, context)
    await until(lambda: "abort:child" in log)
    assert await inspected_state(harness, parent) == {"kind": "waiting", "on": [child[0]]}
    open_gate("abort.child")
    assert await outcome_of(harness, parent) == "aborted"
    await harness.close(context)


async def test_faults_an_abort_handler_that_tries_to_wait() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    script("parent", abort=committing(lambda: wait_on([], "allSettled")))
    parent = await start(root, "parent")
    await until(lambda: "run:parent" in log)
    await harness.abort_task(parent, context)
    await faulted_with(harness, parent, "cannot wait")
    await harness.close(context)


async def test_orphans_a_blocked_task_only_after_its_owned_work_drained() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root

    async def change(tx: Any) -> Any:
        owner = await tx.create_task(Unregistered, {"name": "owner"}, OWN_CONVERSATION)
        return owner, await spawn(tx, owner, "child")

    owner, child = await root.commit(change, context)
    await until(lambda: "run:child" in log)
    assert await harness.abort_task(owner, context) == "marked"
    assert await outcome_of(harness, child) == "aborted"
    assert (await harness.wait_for_task(owner, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "missing_task",
    }
    await harness.close(context)


async def test_cascades_through_task_and_conversation_edges_bottom_up() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    b: list[Any] = []
    x: list[Any] = []

    async def a_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            b.append(await spawn(tx, runtime.task_id, "b"))
            return wait_on(b, "allSettled")

        await runtime.commit(change, ctx)

    async def b_run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            conversation = await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}})
            x.append(
                await tx.create_task(Node, {"name": "x"}, {**OWN_CONVERSATION, "conversationId": conversation["id"]})
            )
            return running_resume()

        await runtime.commit(change, ctx)

    async def b_resume(runtime: Any, _ctx: Context, _round: int) -> None:
        await aborted(runtime.signal)

    script("a", run=a_run)
    script("b", run=b_run, resume=b_resume)
    a = await start(root, "a")
    await until(lambda: "run:x" in log)
    await harness.abort_task(a, context)
    assert await outcome_of(harness, a) == "aborted"
    assert await outcome_of(harness, x[0]) == "aborted"
    assert [line for line in log if line.startswith("abort:")] == ["abort:x", "abort:b", "abort:a"]
    await harness.close(context)


# ─── Background and terminal owners ─────────────────────────────────────────


async def test_keeps_background_work_through_conversation_abort_background_true_aborts_it_and_waits() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    foreground = await start(root, "foreground")
    child: list[Any] = []
    below: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child.append(await spawn(tx, runtime.task_id, "child"))
            conversation = await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}})
            below.append(
                await tx.create_task(
                    Node, {"name": "below"}, {**OWN_CONVERSATION, "conversationId": conversation["id"]}
                )
            )
            return wait_on(child, "allSettled")

        await runtime.commit(change, ctx)

    script("background", run=run)
    background = await start(root, "background", {**OWN_CONVERSATION, "background": True})
    await until(lambda: "run:below" in log and "run:child" in log)
    await root.abort(context)
    assert await outcome_of(harness, foreground) == "aborted"
    for id in (background, child[0], below[0]):
        assert (await state(harness, id))["status"] != "terminal"
    await root.abort(context, background=True)
    for id in (background, child[0], below[0]):
        assert (await state(harness, id))["status"] == "terminal"
    await harness.close(context)


async def test_never_cascades_from_a_terminal_owner_an_aborted_subagents_conversation_runs_new_work_normally() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    conversation: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            conversation.append(
                (await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}}))["id"]
            )
            return running_resume()

        await runtime.commit(change, ctx)

    async def resume(runtime: Any, _ctx: Context, _round: int) -> None:
        await aborted(runtime.signal)

    script("agent", run=run, resume=resume)
    agent = await start(root, "agent")
    await until(lambda: "resume:agent" in log)
    await harness.abort_task(agent, context)
    assert await outcome_of(harness, agent) == "aborted"
    child = await harness.conversation(conversation[0], context)
    question = await start(child, "question")
    open_gate("question")
    assert await outcome_of(harness, question) == "completed"
    await harness.close(context)


# ─── Recovery ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class Seeded:
    parent: Any
    children: list[Any]


SetTask = Callable[[Any, dict[str, Any]], Awaitable[None]]


async def seed(path: Path, children: Sequence[str], edit: Callable[[SetTask, Seeded], Awaitable[None]]) -> Seeded:
    """Write records a crash could leave, without a Harness: ``parent`` with children named ``children``, then
    ``edit`` replaces states through the internal task write."""
    session = create_session(await open_sqlite_storage(str(path)))

    async def create(tx: Any) -> Seeded:
        root = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        parent = await tx.create_task(Node, {"name": "parent"}, {**OWN_CONVERSATION, "conversationId": root["id"]})
        ids = [await spawn(tx, parent, name) for name in children]
        return Seeded(parent, ids)

    seeded = await session.commit(create, context)

    async def set_task(id: Any, patch: dict[str, Any]) -> None:
        async def change(tx: Any) -> None:
            record = await tx.task(id)
            tx.set_task({**record, **patch})

        await session.commit(change, context)

    await edit(set_task, seeded)
    await session.close(context)
    return seeded


COMPLETED = {"status": "terminal", "outcome": {"status": "completed", "result": "done"}}
FAILED = {"status": "terminal", "outcome": {"status": "failed", "error": {"message": "declined"}}}


def waiting_state(on: Sequence[Any], policy: str) -> Any:
    return {"status": "waiting", "checkpoint": {"phase": "resume", "round": 1}, "on": list(on), "policy": policy}


async def test_resumes_a_parent_whose_awaited_children_finished_before_the_crash(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"

    async def edit(set_task: SetTask, seeded: Seeded) -> None:
        await set_task(seeded.children[0], {"state": COMPLETED})
        await set_task(seeded.parent, {"state": waiting_state(seeded.children, "allSettled")})

    seeded = await seed(path, ["c1"], edit)
    opened = await open_nodes(await open_sqlite_storage(str(path)))
    assert await outcome_of(opened.harness, seeded.parent) == "completed"
    assert log == ["resume:parent"]
    assert len(seeded.children) == 1
    await opened.harness.close(context)


async def test_marks_fail_fast_siblings_a_crash_left_unmarked(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"

    async def edit(set_task: SetTask, seeded: Seeded) -> None:
        await set_task(seeded.children[0], {"state": FAILED})
        await set_task(seeded.parent, {"state": waiting_state(seeded.children, "failFast")})

    seeded = await seed(path, ["c1", "c2"], edit)
    opened = await open_nodes(await open_sqlite_storage(str(path)))
    assert await outcome_of(opened.harness, seeded.children[1]) == "aborted"
    assert await outcome_of(opened.harness, seeded.parent) == "completed"
    assert "run:c2" not in log
    await opened.harness.close(context)


async def test_finalizes_a_held_outcome_whose_work_drained_before_the_crash_and_keeps_one_whose_work_lives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    held = {"status": "completing", "outcome": {"status": "completed", "result": "parent"}}

    async def edit(set_task: SetTask, seeded: Seeded) -> None:
        await set_task(seeded.parent, {"state": held})

    seeded = await seed(path, ["c1"], edit)
    opened = await open_nodes(await open_sqlite_storage(str(path)))
    await until(lambda: "run:c1" in log)
    assert await state(opened.harness, seeded.parent) == held
    await opened.harness.close(context)

    opened = await open_nodes(await open_sqlite_storage(str(path)))
    open_gate("c1")
    assert await outcome_of(opened.harness, seeded.children[0]) == "completed"
    assert await outcome_of(opened.harness, seeded.parent) == "completed"
    await opened.harness.close(context)

    drained = tmp_path / "drained.sqlite"

    async def edit_drained(set_task: SetTask, seeded: Seeded) -> None:
        await set_task(seeded.children[0], {"state": COMPLETED})
        await set_task(seeded.parent, {"state": held})

    second = await seed(drained, ["c1"], edit_drained)
    opened = await open_nodes(await open_sqlite_storage(str(drained)))
    assert await outcome_of(opened.harness, second.parent) == "completed"
    await opened.harness.close(context)


async def test_resumes_a_bottom_up_abort_a_crash_interrupted_before_the_cascade(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"

    async def edit(set_task: SetTask, seeded: Seeded) -> None:
        await set_task(seeded.parent, {"abortRequested": True, "state": waiting_state(seeded.children, "allSettled")})

    seeded = await seed(path, ["c1"], edit)
    opened = await open_nodes(await open_sqlite_storage(str(path)))
    assert await outcome_of(opened.harness, seeded.parent) == "aborted"
    assert await outcome_of(opened.harness, seeded.children[0]) == "aborted"
    assert [line for line in log if line.startswith("abort:")] == ["abort:c1", "abort:parent"]
    await opened.harness.close(context)


async def test_reopens_a_checkout_waiting_on_live_payments_and_finishes_it(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    found = parent_of("checkout", ["p1", "p2"], "failFast")
    opened = await open_nodes(await open_sqlite_storage(path))
    parent = await start(opened.root, "checkout")
    await until(status_is(opened.harness, parent, "waiting"))
    await until(lambda: "run:p1" in log and "run:p2" in log)
    await opened.harness.close(context)

    opened = await open_nodes(await open_sqlite_storage(path))
    found.ids[:] = (await state(opened.harness, parent))["on"]
    open_gate("p1")
    open_gate("p2")
    assert await outcome_of(opened.harness, parent) == "completed"
    assert found.outcomes == ["completed", "completed"]
    await opened.harness.close(context)


# ─── Built-in tool rounds ───────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class Blocking:
    started: asyncio.Future[None]
    registration: Any


def blocking_tool(name: str) -> Blocking:
    started = deferred()

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        api.output("partial")
        resolve(started)
        return await aborted(call_context.abort_signal)

    return Blocking(
        started,
        define_tool(
            name=name,
            description=f"The {name} tool",
            parameters=Type.Object({}),
            execution_mode="sequential",
            execute=execute,
        ),
    )


def tool_calls(*ids: tuple[str, str]) -> Any:
    return faux_assistant_message([faux_tool_call(name, {}, id=id) for name, id in ids], stop_reason="toolUse")


def noop() -> Any:
    async def execute(*_args: Any) -> Any:
        return {"output": []}

    return define_tool(name="noop", description="Does nothing", parameters=Type.Object({}), execute=execute)


async def run_task_id(harness: Harness, root: Any) -> Any:
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    return live["run"]["taskId"]  # type: ignore[typeddict-item]


async def test_owns_its_tool_tasks_waits_for_them_and_hands_the_run_to_a_conversation_owned_generation() -> None:
    setup = chat_setup()
    add_tool(setup.registry, noop())
    setup.faux.set_responses([tool_calls(("noop", "c1"), ("noop", "c2")), faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    assert (await (await root.submit({"type": "input", "content": "go"}, context)).wait(context))["status"] == "done"
    tasks = (await harness.commit(lambda tx: tx.scan_tasks({"conversationId": root.id}, 20), context))["items"]
    first, second = [task for task in tasks if task["kind"] == "pi.generation"]
    assert [task.get("owner") for task in tasks if task["kind"] == "pi.tool"] == [first["id"], first["id"]]
    assert "owner" not in first
    assert "owner" not in second
    assistant = next(entry for entry in await all_entries(root) if entry["kind"] == "pi.assistant")
    assert first["state"] == {
        "status": "terminal",
        "outcome": {"status": "completed", "result": {"entryId": assistant["id"]}},
    }
    await harness.close(context)


async def test_aborts_a_parallel_round_with_its_generation_tools_first_then_the_generation_with_every_result_written() -> (
    None
):
    setup = chat_setup()
    one = blocking_tool("one")
    two = blocking_tool("two")
    add_tool(setup.registry, dataclasses.replace(one.registration, execution_mode="parallel"))
    add_tool(setup.registry, dataclasses.replace(two.registration, execution_mode="parallel"))
    setup.faux.set_responses([tool_calls(("one", "c1"), ("two", "c2"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await asyncio.gather(one.started, two.started)
    generation = await run_task_id(harness, root)
    await harness.abort_task(generation, context)
    assert_match(await submission.wait(context), {"status": "unanswered", "reason": "aborted"})
    results = [entry for entry in await all_entries(root) if ToolResultEntry.is_(entry)]
    assert [entry["model"][0].tool_call_id for entry in results] == ["c1", "c2"]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    assert (await harness.wait_for_task(generation, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)


async def test_answers_the_unstarted_calls_of_an_aborted_sequential_round_with_aborted_results_in_call_order() -> None:
    setup = chat_setup()
    one = blocking_tool("one")
    add_tool(setup.registry, one.registration)
    add_tool(setup.registry, blocking_tool("two").registration)
    setup.faux.set_responses([tool_calls(("one", "c1"), ("two", "c2"), ("two", "c3"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await one.started
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    assert ["taskId" in slot for slot in live["tools"]] == [True, False, False]  # type: ignore[typeddict-item]
    generation = await run_task_id(harness, root)
    await harness.abort_task(generation, context)
    assert_match(await submission.wait(context), {"status": "unanswered", "reason": "aborted"})
    results = [entry["model"][0] for entry in await all_entries(root) if ToolResultEntry.is_(entry)]
    assert [result.tool_call_id for result in results] == ["c1", "c2", "c3"]
    assert [result.is_error for result in results] == [True, True, True]
    assert results[1].content == [TextContent("<harness>\n[error] Tool two was aborted\n</harness>")]
    tools = (
        await harness.commit(lambda tx: tx.scan_tasks({"conversationId": root.id, "kind": "pi.tool"}, 20), context)
    )["items"]
    assert len(tools) == 1
    await harness.close(context)


def hook_owned_work(setup: ChatSetup, harness: list[Harness], hook: str = "after_tools") -> None:
    """An extension's hook starts work owned by the generation, which holds it while the next turn runs."""

    async def create(api: Any, call_context: Any) -> None:
        await harness[0].commit(lambda tx: tx.create_task(Node, {"name": "hooked"}, owned(api.task_id)), call_context)

    if hook == "after_tools":

        async def after_tools(_assistant: Any, _results: Any, api: Any, call_context: Any) -> None:
            await create(api, call_context)

        add_hooks(setup.registry, GenerationTask, {"after_tools": after_tools})
    else:

        async def before_request(_request: Any, api: Any, call_context: Any) -> None:
            await create(api, call_context)

        add_hooks(setup.registry, GenerationTask, {"before_request": before_request})


async def first_generation(harness: Harness) -> Any:
    return (await harness.commit(lambda tx: tx.scan_tasks({"kind": "pi.generation"}, 1), context))["items"][0]


async def test_ends_a_turn_at_the_generations_hold_before_its_successors_turn_starts() -> None:
    setup = chat_setup()
    add_task(setup.registry, Node)
    add_tool(setup.registry, noop())
    holder: list[Harness] = []
    hook_owned_work(setup, holder)
    setup.faux.set_responses([tool_calls(("noop", "c1")), faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    holder.append(harness)
    stream = await watch_events(harness, root.id, context)
    events: list[Any] = []

    async def listener(batch: Any, _context: Any) -> None:
        events.extend(batch)

    stream.start(listener)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    first = await first_generation(harness)
    assert (await state(harness, first["id"]))["status"] == "completing"

    def turns() -> list[str]:
        return [event["type"] for event in events if event["type"].startswith("turn_")]

    await wait_for(lambda: len(turns()) == 4)
    assert turns() == ["turn_start", "turn_end", "turn_start", "turn_end"]
    open_gate("hooked")
    assert await outcome_of(harness, first["id"]) == "completed"
    await wait_for(lambda: len(turns()) == 4)
    await stream.stop()
    await harness.close(context)


class _WithStream:
    """``models`` with ``stream_simple`` replaced (upstream's Proxy)."""

    def __init__(self, models: Any, stream_simple: Any) -> None:
        self._models = models
        self.stream_simple = stream_simple

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


@dataclasses.dataclass
class _Start:
    partial: Any
    type: str = "start"


class _InvalidFinalStream:
    """A stream whose final message is not strict JSON, so the classification commit throws and the task faults."""

    def __init__(self) -> None:
        self._final = faux_assistant_message("final")
        self._final.usage = Usage(input=lambda: None)  # type: ignore[arg-type]

    async def __aiter__(self) -> AsyncIterator[Any]:
        yield _Start(faux_assistant_message("partial", stop_reason="pending"))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)

    async def result(self) -> Any:
        return self._final


def invalid_final_stream(*_args: Any, **_kwargs: Any) -> Any:
    return _InvalidFinalStream()


class FailingFinalization(ControlledStorage):
    """Fails the commit that makes ``reject`` terminal, once."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__(MemoryStorage(), STORAGE_BACKENDS["memory"])
        self.reject: Any = None
        self.failure = failure

    async def commit(self, writes: Any, context: Context) -> Any:
        finalizes = any(
            write["type"] == "task"
            and write["value"]["id"] == self.reject
            and write["value"]["state"]["status"] == "terminal"
            for write in writes
        )
        if finalizes:
            self.reject = None
            raise self.failure
        return await super().commit(writes, context)


async def test_keeps_run_control_with_a_faulted_generation_until_its_owned_work_drains_and_fails_the_harness_on_a_failed_final_commit() -> (
    None
):
    base = chat_setup()
    setup = ChatSetup(
        base.faux, _WithStream(base.models, invalid_final_stream), base.registry, base.reports, base.settings
    )
    add_task(setup.registry, Node)
    holder: list[Harness] = []
    hook_owned_work(setup, holder, "before_request")
    script("hooked", abort=gated_abort("abort.hooked"))
    failure = RuntimeError("disk gone")
    storage = FailingFinalization(failure)
    opened = await open_chat(storage, setup)
    harness, root = opened.harness, opened.root
    holder.append(harness)
    submission = await root.submit({"type": "input", "content": "hi"}, context)

    async def completing() -> bool:
        live = await harness.snapshot(LiveDoc, root.id, context)
        run = None if live is None else live.get("run")
        return run is not None and (await state(harness, run["taskId"]))["status"] == "completing"

    await wait_for(completing)
    generation = await run_task_id(harness, root)
    assert_match(await state(harness, generation), {"status": "completing", "outcome": {"status": "faulted"}})
    waiting = asyncio.ensure_future(submission.wait(context))
    assert await settled(waiting) is False
    assert await run_task_id(harness, root) == generation
    # The faulted outcome is cancellation intent: the hooked work is aborted, then the run settles.
    await wait_for(lambda: "abort:hooked" in log)
    storage.reject = generation
    open_gate("abort.hooked")
    # The final commit, with the run's cleanup, fails: nothing of the cleanup lands, and the Harness fails.
    assert await harness.closed == {"reason": "failed", "error": failure}
    with pytest.raises(SessionFailed) as raised:
        await waiting
    assert raised.value.cause is failure
    await harness.close(context)
    # Reopening finalizes the run; the partial becomes one aborted entry.
    reopened = await open_chat(await storage.reopen(), setup)
    holder[0] = reopened.harness
    again = await reopened.harness.submission(submission.id, context)
    assert again is not None
    assert_match(await again.wait(context), {"status": "unanswered", "reason": "faulted"})
    assert await reopened.harness.snapshot(LiveDoc, reopened.root.id, context) == {}
    assert [entry["kind"] for entry in await all_entries(reopened.root)] == ["pi.user", "pi.assistant"]
    await reopened.harness.close(context)


# ─── Review follow-ups ──────────────────────────────────────────────────────


def versioned(version: int) -> Any:
    """A waiter of its own kind, so its definition can be removed or replaced."""

    def wait(task: Any, runtime: Any, ctx: Context) -> Awaitable[Any]:
        next_state = {
            "status": "waiting",
            "checkpoint": {"phase": "resume"},
            "on": task["input"]["on"],
            "policy": "allSettled",
        }
        return runtime.commit(lambda _tx, _current: next_state, ctx)

    def resume(_task: Any, runtime: Any, ctx: Context) -> Awaitable[Any]:
        return runtime.commit(lambda _tx, _current: terminal("completed", result="v1"), ctx)

    def migrated(_task: Any, runtime: Any, ctx: Context) -> Awaitable[Any]:
        return runtime.commit(lambda _tx, _current: terminal("completed", result="v2"), ctx)

    def abort(_task: Any, runtime: Any, ctx: Context) -> Awaitable[Any]:
        return commit_aborted(runtime, ctx)

    migrate = (lambda input, *_rest: {"input": input, "checkpoint": {"phase": "migrated"}}) if version == 2 else None
    return define_task(
        name="test.versioned",
        version=version,
        initial=lambda _input: {"phase": "wait"},
        phases={"wait": wait, "resume": resume, "migrated": migrated},
        abort=abort,
        migrate=migrate,
    )


def spawn_and_finish(name: str, child: str) -> Found:
    """Script ``name`` to spawn ``child`` and then complete, holding while the child lives."""
    found = Found()

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            found.ids.append(await spawn(tx, runtime.task_id, child))
            return running_resume()

        await runtime.commit(change, ctx)

    script(name, run=run)
    return found


async def test_keeps_a_waiting_task_blocked_without_its_definition_and_resumes_it_migrated_under_a_newer_one() -> None:
    opened = await open_nodes()
    harness, root, registry = opened.harness, opened.root, opened.registry
    registration = add_task(registry, versioned(1))
    other = await start(root, "other")
    waiter = await root.commit(lambda tx: tx.create_task(versioned(1), {"on": [other]}, OWN_CONVERSATION), context)
    await until(status_is(harness, waiter, "waiting"))
    registration.dispose()
    open_gate("other")
    await harness.wait_for_task(other, context)
    assert await inspected_state(harness, waiter) == {"kind": "blocked", "reason": "missing_task"}
    assert (await state(harness, waiter))["status"] == "waiting"
    add_task(registry, versioned(2))
    assert (await harness.wait_for_task(waiter, context))["state"]["outcome"] == {"status": "completed", "result": "v2"}
    await harness.close(context)


async def test_orphans_an_aborted_waiting_task_without_its_definition_at_once_leaving_the_task_it_waits_on_running() -> (
    None
):
    opened = await open_nodes()
    harness, root, registry = opened.harness, opened.root, opened.registry
    registration = add_task(registry, versioned(1))
    other = await start(root, "other")
    waiter = await root.commit(lambda tx: tx.create_task(versioned(1), {"on": [other]}, OWN_CONVERSATION), context)
    await until(status_is(harness, waiter, "waiting"))
    registration.dispose()
    assert await harness.abort_task(waiter, context) == "marked"
    assert (await state(harness, waiter))["outcome"] == {"status": "orphaned", "reason": "missing_task"}
    assert (await state(harness, other))["status"] == "running"
    open_gate("other")
    await harness.close(context)


async def test_never_migrates_a_held_outcome_a_newer_definition_leaves_it_and_its_version_alone() -> None:
    opened = await open_nodes()
    harness, root, registry, reports = opened.harness, opened.root, opened.registry, opened.reports

    def holder(version: int) -> Any:
        async def run(task: Any, runtime: Any, ctx: Context) -> None:
            async def change(tx: Any, _current: Any) -> Any:
                await spawn(tx, task["id"], "child")
                return None

            await runtime.commit(change, ctx)
            await runtime.commit(lambda _tx, _current: terminal("completed", result="held"), ctx)

        def abort(_task: Any, runtime: Any, ctx: Context) -> Awaitable[Any]:
            return commit_aborted(runtime, ctx)

        def migrate(*_args: Any) -> Any:
            raise RuntimeError("never migrates")

        return define_task(
            name="test.holder",
            version=version,
            initial=lambda _input: {"phase": "run"},
            phases={"run": run},
            abort=abort,
            migrate=migrate,
        )

    add_task(registry, holder(1))
    parent = await root.commit(lambda tx: tx.create_task(holder(1), None, OWN_CONVERSATION), context)
    await until(status_is(harness, parent, "completing"))
    # The same extension name replaces the old one in place.
    add_task(registry, holder(2))
    open_gate("child")
    settled_parent = await harness.wait_for_task(parent, context)
    assert settled_parent["state"]["outcome"] == {"status": "completed", "result": "held"}
    assert settled_parent["version"] == 1
    assert reports == []
    await harness.close(context)


async def test_treats_a_held_task_as_live_outcomes_rejects_and_a_task_waiting_on_it_resumes_at_its_final_commit() -> (
    None
):
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    spawn_and_finish("held", "child")
    held = await start(root, "held")
    await until(status_is(harness, held, "completing"))
    rejection = [""]

    async def run(runtime: Any, ctx: Context) -> None:
        try:
            await runtime.outcomes([held], ctx)
            rejection[0] = "resolved"
        except Exception as error:
            rejection[0] = str(error)
        await runtime.commit(lambda _tx, _current: wait_on([held], "allSettled"), ctx)

    script("reader", run=run)
    reader = await start(root, "reader")
    await until(status_is(harness, reader, "waiting"))
    assert rejection[0] == f"Task {held} is not terminal"
    open_gate("child")
    assert await outcome_of(harness, reader) == "completed"
    assert "resume:reader" in log
    await harness.close(context)


# ─── Conversation abort and boundaries ──────────────────────────────────────


async def test_marks_a_held_completed_task_with_conversation_abort_the_work_below_is_aborted_the_outcome_stays() -> (
    None
):
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    child = spawn_and_finish("held", "child")
    held = await start(root, "held")
    await until(status_is(harness, held, "completing"))
    await root.abort(context)
    assert await outcome_of(harness, child.ids[0]) == "aborted"
    record = await harness.wait_for_task(held, context)
    assert record["state"]["outcome"]["status"] == "completed"
    assert record["abortRequested"] is True
    await harness.close(context)


async def test_marks_and_awaits_only_the_background_work_reached_when_background_true_is_admitted() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    script("first", abort=gated_abort("abort.first"))
    background = {**OWN_CONVERSATION, "background": True}
    first = await start(root, "first", background)
    await until(lambda: "run:first" in log)
    aborting = asyncio.ensure_future(root.abort(context, background=True))
    await until(lambda: abort_requested(harness, first))
    later = await start(root, "later", background)
    open_gate("abort.first")
    await aborting
    assert (await state(harness, later))["status"] != "terminal"
    assert await abort_requested(harness, later) is False
    await harness.abort_task(later, context)
    await harness.close(context)


async def test_stops_a_cascade_at_an_unmarked_background_task_but_not_at_a_marked_one() -> None:
    opened = await open_nodes()
    harness, root = opened.harness, opened.root
    script("owner", abort=gated_abort("abort.owner"))

    async def change(tx: Any) -> Any:
        owner = await tx.create_task(Node, {"name": "owner"}, OWN_CONVERSATION)
        outer = await tx.create_conversation({"ownership": {"kind": "task", "taskId": owner}})
        background = await tx.create_task(
            Node, {"name": "background"}, {**OWN_CONVERSATION, "conversationId": outer["id"], "background": True}
        )
        inner = await tx.create_conversation({"ownership": {"kind": "task", "taskId": background}})
        return {"owner": owner, "outer": outer["id"], "background": background, "inner": inner["id"]}

    tree = await root.commit(change, context)
    await until(lambda: "run:owner" in log and "run:background" in log)
    # The owner's abort handler starts at once: background work is not its ordinary owned work.
    await harness.abort_task(tree["owner"], context)
    await until(lambda: "abort:owner" in log)
    inner = await harness.conversation(tree["inner"], context)
    outer = await harness.conversation(tree["outer"], context)
    shielded = await start(inner, "shielded")
    exposed = await start(outer, "exposed")
    assert await outcome_of(harness, exposed) == "aborted"
    assert await abort_requested(harness, shielded) is False
    # Marked directly, the background task cascades into its own subtree.
    await harness.abort_task(tree["background"], context)
    assert await outcome_of(harness, shielded) == "aborted"
    assert await outcome_of(harness, tree["background"]) == "aborted"
    open_gate("abort.owner")
    assert await outcome_of(harness, tree["owner"]) == "aborted"
    await harness.close(context)


async def test_fails_the_harness_on_a_failed_finalization_which_reopening_applies() -> None:
    failure = RuntimeError("disk gone")
    storage = FailingFinalization(failure)
    opened = await open_nodes(storage)
    harness, root, reports = opened.harness, opened.root, opened.reports
    spawn_and_finish("parent", "child")
    parent = await start(root, "parent")
    await until(status_is(harness, parent, "completing"))
    storage.reject = parent
    open_gate("child")
    assert await harness.closed == {"reason": "failed", "error": failure}
    assert reports == [failure]
    await harness.close(context)
    reopened = await open_nodes(await storage.reopen())
    assert await outcome_of(reopened.harness, parent) == "completed"
    await reopened.harness.close(context)


# ─── Recovery through ownership edges ───────────────────────────────────────


async def test_shows_an_abort_marked_owner_waiting_for_work_in_its_owned_conversation_before_resume_after_reopen(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "session.sqlite")
    session = create_session(await open_sqlite_storage(path))

    async def create(tx: Any) -> Any:
        conversation = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        owner = await tx.create_task(
            Node, {"name": "owner"}, {**OWN_CONVERSATION, "conversationId": conversation["id"]}
        )
        child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": owner}})
        inner = await tx.create_task(Node, {"name": "inner"}, {**OWN_CONVERSATION, "conversationId": child["id"]})
        return {"owner": owner, "inner": inner}

    tree = await session.commit(create, context)

    async def mark(tx: Any) -> None:
        record = await tx.task(tree["owner"])
        tx.set_task({**record, "abortRequested": True})

    await session.commit(mark, context)
    await session.close(context)

    harness = (await open_tasks(await open_sqlite_storage(path), [Node])).harness
    assert await inspected_state(harness, tree["owner"]) == {"kind": "waiting", "on": [tree["inner"]]}
    harness.resume()
    assert await outcome_of(harness, tree["owner"]) == "aborted"
    assert [line for line in log if line.startswith("abort:")] == ["abort:inner", "abort:owner"]
    await harness.close(context)


async def test_keeps_the_root_busy_after_reopen_for_work_below_a_child_tasks_owned_conversation(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    inner: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            conversation = await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}})
            inner.append(
                await tx.create_task(
                    Node, {"name": "inner"}, {**OWN_CONVERSATION, "conversationId": conversation["id"]}
                )
            )
            return running_resume()

        await runtime.commit(change, ctx)

    script("child", run=run)
    parent_of("parent", ["child"], "allSettled")
    opened = await open_nodes(await open_sqlite_storage(path))
    parent = await start(opened.root, "parent")
    await until(lambda: "run:inner" in log)
    await opened.harness.close(context)

    opened = await open_nodes(await open_sqlite_storage(path))
    assert await settled(opened.root.wait_for_idle(context)) is False
    open_gate("inner")
    await opened.root.wait_for_idle(context)
    assert await outcome_of(opened.harness, parent) == "completed"
    assert len(inner) == 1
    await opened.harness.close(context)


async def test_keeps_a_finished_background_ancestor_a_boundary_after_reopen_which_background_true_crosses(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "session.sqlite")
    conversation: list[Any] = []

    async def run(runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            conversation.append(
                (await tx.create_conversation({"ownership": {"kind": "task", "taskId": runtime.task_id}}))["id"]
            )
            return running_resume()

        await runtime.commit(change, ctx)

    script("child", run=run)
    parent_of("background", ["child"], "allSettled")
    opened = await open_nodes(await open_sqlite_storage(path))
    background = await start(opened.root, "background", {**OWN_CONVERSATION, "background": True})
    await opened.harness.wait_for_task(background, context)
    await opened.harness.close(context)

    opened = await open_nodes(await open_sqlite_storage(path))
    below = await start(await opened.harness.conversation(conversation[0], context), "below")
    await until(lambda: "run:below" in log)
    await opened.root.wait_for_idle(context)
    await opened.root.abort(context)
    assert (await state(opened.harness, below))["status"] == "running"
    await opened.root.abort(context, background=True)
    assert await outcome_of(opened.harness, below) == "aborted"
    await opened.harness.close(context)


# ─── Tool rounds and events ─────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class Listened:
    stream: Any
    events: list[Any]


async def listen(harness: Harness, conversation: Any) -> Listened:
    stream = await watch_events(harness, conversation.id, context)
    events: list[Any] = []

    async def listener(batch: Any, _context: Any) -> None:
        events.extend(batch)

    stream.start(listener)
    return Listened(stream, events)


def labels(events: Sequence[Any]) -> list[str]:
    """Event types, with tool ends and message ends labelled by call ID and whether an entry came along."""
    shown: list[str] = []
    for event in events:
        if event["type"] == "tool_execution_end":
            shown.append(f"end:{event['toolCallId']}:{'true' if 'entry' in event else 'false'}")
        elif event["type"] == "message_end":
            model = event["entry"].get("model")
            message = model[0] if model else None
            role = None if message is None else message.role
            shown.append(f"result:{message.tool_call_id}" if role == "toolResult" else f"message:{role}")  # type: ignore[union-attr]
    return shown


async def test_ends_an_aborted_sequential_rounds_unstarted_calls_with_their_result_entries_right_before_them() -> None:
    setup = chat_setup()
    one = blocking_tool("one")
    add_tool(setup.registry, one.registration)
    add_tool(setup.registry, blocking_tool("two").registration)
    setup.faux.set_responses([tool_calls(("one", "c1"), ("two", "c2"), ("two", "c3"))])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listened = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await one.started
    await harness.abort_task(await run_task_id(harness, root), context)
    await submission.wait(context)
    await wait_for(lambda: "result:c3" in labels(listened.events))
    assert [label for label in labels(listened.events) if not label.startswith("message:")] == [
        "end:c1:true",
        "result:c1",
        "end:c2:true",
        "result:c2",
        "end:c3:true",
        "result:c3",
    ]
    await listened.stream.stop()
    await harness.close(context)


async def test_emits_one_turn_end_per_generation_also_for_a_stream_attached_while_it_holds() -> None:
    setup = chat_setup()
    add_task(setup.registry, Node)
    add_tool(setup.registry, noop())
    holder: list[Harness] = []
    hook_owned_work(setup, holder)
    setup.faux.set_responses([tool_calls(("noop", "c1")), faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    holder.append(harness)
    early = await listen(harness, root)
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    late = await listen(harness, root)
    open_gate("hooked")
    first = await first_generation(harness)
    await harness.wait_for_task(first["id"], context)
    await root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)
    await wait_for(lambda: any(event["type"] == "entry_appended" for event in late.events))

    def turns(events: Sequence[Any]) -> int:
        return len([event for event in events if event["type"] == "turn_end"])

    assert turns(early.events) == 2
    assert turns(late.events) == 0
    await early.stream.stop()
    await late.stream.stop()
    await harness.close(context)


async def test_lets_a_tool_that_owns_live_work_finish_its_call_at_the_hold_while_the_generation_waits_for_its_final_commit() -> (
    None
):
    setup = chat_setup()
    add_task(setup.registry, Node)

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        async def change(tx: Any) -> None:
            child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            await tx.create_task(Node, {"name": "sub"}, {**OWN_CONVERSATION, "conversationId": child["id"]})

        await api.commit(change, call_context)
        return {"output": [TextContent("started")]}

    add_tool(
        setup.registry,
        define_tool(
            name="delegate",
            description="Starts work in a conversation it owns",
            parameters=Type.Object({}),
            execute=execute,
        ),
    )
    setup.faux.set_responses([tool_calls(("delegate", "c1")), faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listened = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)

    async def tool_done() -> bool:
        live = await harness.snapshot(LiveDoc, root.id, context)
        tools = None if live is None else live.get("tools")
        return bool(tools) and tools[0]["status"] == "done"  # type: ignore[index]

    await wait_for(tool_done)
    live: Any = await harness.snapshot(LiveDoc, root.id, context)
    tool = live["tools"][0]["taskId"]
    assert "entry" in live["tools"][0]
    assert (await state(harness, tool))["status"] == "completing"
    assert (await state(harness, live["run"]["taskId"]))["status"] == "waiting"
    await wait_for(lambda: "end:c1:true" in labels(listened.events))
    waiting = asyncio.ensure_future(submission.wait(context))
    assert await settled(waiting) is False
    open_gate("sub")
    assert (await waiting)["status"] == "done"
    assert await outcome_of(harness, tool) == "completed"
    await listened.stream.stop()
    await harness.close(context)


async def test_holds_a_faulted_tools_slot_and_task_failed_until_the_work_it_owns_drained() -> None:
    setup = chat_setup()
    add_task(setup.registry, Node)
    script("held", abort=gated_abort("abort.held"))

    async def execute(_args: Any, api: Any, call_context: Any) -> Any:
        await api.commit(lambda tx: tx.create_task(Node, {"name": "held"}, owned(api.task_id)), call_context)
        return {"output": [], "details": {"fn": lambda: 1}}

    add_tool(
        setup.registry,
        define_tool(
            name="broken",
            description="Starts owned work, then returns a result that is not strict JSON",
            parameters=Type.Object({}),
            execute=execute,
        ),
    )
    setup.faux.set_responses([tool_calls(("broken", "c1")), faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    listened = await listen(harness, root)
    submission = await root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: "abort:held" in log)
    live: Any = await harness.snapshot(LiveDoc, root.id, context)
    tool = live["tools"][0]["taskId"]
    assert_match(await state(harness, tool), {"status": "completing", "outcome": {"status": "faulted"}})
    assert live["tools"][0]["status"] != "done"
    assert not any(event["type"] == "task_failed" for event in listened.events)
    open_gate("abort.held")
    assert (await submission.wait(context))["status"] == "done"
    assert await outcome_of(harness, tool) == "faulted"
    await wait_for(lambda: any(event["type"] == "task_failed" for event in listened.events))
    assert "end:c1:false" in labels(listened.events)
    await listened.stream.stop()
    await harness.close(context)
