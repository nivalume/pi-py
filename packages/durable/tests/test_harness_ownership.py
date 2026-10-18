"""Port of ``test/harness-ownership.test.ts``.

Adaptations:

- ``waitUntil`` (real-time polling) is ``wait_for``; SQLite runs on executor threads here.
- Upstream rebinds ``storage.commit`` on the SQLite storage instance; a :class:`StorageSpy` subclass intercepts it.
- "tracks ownership chains deeper than the call stack" builds a chain twice Python's recursion limit deep
  (upstream: 20,000 tasks, against JavaScript's stack).
- Two cases close a Harness with an owner cancelled before its slow abort handler starts; they hold the Session line
  behind the abort mark until close has sealed (see :func:`cancelled_owner_closed`).

Dropped assertions: "keeps ownership queries off finished subtrees" and "finalizes many held siblings that end
together in one pass" count ``Set`` iterations by patching ``Set.prototype[Symbol.iterator]``, a measure of the
scheduler's walks that Python's built-in ``set`` cannot be instrumented for; both cases keep their behavior.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from durable_harness_support import (
    StorageSpy,
    aborted,
    add_task,
    add_tool,
    chat_setup,
    gated,
    match_object,
    open_chat,
    open_tasks,
    race_aborted,
    settled,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import AbortController, Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import TextContent
from pi_chord.context import Context, with_abort_signal
from pi_durable import (
    LiveDoc,
    MemoryStorage,
    configure,
    create_session,
    define_doc,
    define_task,
    define_tool,
)
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Seq, StorageWrite

#: Outcome ("completed" or "failed") a held task commits once its gate opens.
gates: dict[str, asyncio.Future[Any]] = {}
#: How often each named task started its run phase.
runs: dict[str, int] = {}


def gate(name: str) -> asyncio.Future[Any]:
    found = gates.get(name)
    if found is None:
        found = deferred()
        gates[name] = found
    return found


@pytest.fixture(autouse=True)
def _clear() -> Iterator[None]:
    yield
    gates.clear()
    runs.clear()


def _terminal(outcome: Any) -> Any:
    return lambda _tx, _current: {"status": "terminal", "outcome": outcome}


async def _abort(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(_terminal({"status": "aborted"}), ctx)


async def _hold(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    runs[name] = runs.get(name, 0) + 1
    ending = await race_aborted(gate(name), runtime.signal)
    await runtime.commit(
        _terminal(
            {"status": "completed", "result": None}
            if ending == "completed"
            else {"status": "failed", "error": {"message": "gate failed"}}
        ),
        ctx,
    )


async def _hold_abort(task: Any, runtime: Any, ctx: Context) -> None:
    if task["input"].get("slowAbort") is True:
        await gate(f"abort.{task['input']['name']}")
    await runtime.commit(_terminal({"status": "aborted"}), ctx)


#: A task that holds until its named gate opens or it is aborted. With ``slowAbort``, its abort handler first waits
#: for the gate ``abort.<name>``.
Hold = define_task(
    name="test.hold", version=1, initial=lambda _input: {"phase": "hold"}, phases={"hold": _hold}, abort=_hold_abort
)


async def _wait(task: Any, runtime: Any, ctx: Context) -> None:
    on = task["input"]["on"]
    await runtime.commit(
        lambda _tx, _current: {"status": "waiting", "checkpoint": {"phase": "done"}, "on": on, "policy": "allSettled"},
        ctx,
    )


async def _done(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(_terminal({"status": "completed", "result": None}), ctx)


#: Waits on the tasks in its input, then completes; its abort handler ends it ``aborted``.
Waiter = define_task(
    name="test.waiter",
    version=1,
    initial=lambda _input: {"phase": "wait"},
    phases={"wait": _wait, "done": _done},
    abort=_abort,
)


async def _spawn(task: Any, runtime: Any, ctx: Context) -> None:
    async def change(tx: Any, _current: Any) -> Any:
        child = await tx.create_conversation({"ownership": {"kind": "task", "taskId": task["id"]}})
        await tx.create_task(
            Hold,
            {"name": task["input"]["name"]},
            {"ownership": {"kind": "conversation"}, "conversationId": child["id"]},
        )
        return {"status": "terminal", "outcome": {"status": "completed", "result": None}}

    await runtime.commit(change, ctx)


#: Tries to create, in the commit that ends it, a conversation it owns with a held task in it, named by its input.
Spawner = define_task(
    name="test.spawner", version=1, initial=lambda _input: {"phase": "spawn"}, phases={"spawn": _spawn}, abort=_abort
)


async def _nothing(*_args: Any) -> None:
    return None


#: Never registered: aborting it can only orphan it.
Unregistered = define_task(
    name="test.unregistered",
    version=1,
    initial=lambda _input: {"phase": "hold"},
    phases={"hold": _nothing},
    abort=_nothing,
)


def open_gate(name: str, ending: str) -> None:
    resolve(gate(name), ending)


CONVERSATION: Any = {"ownership": {"kind": "conversation"}}


def owned_by(task_id: Any) -> Any:
    return {"ownership": {"kind": "task", "taskId": task_id}}


def in_conversation(conversation_id: Any, **extra: Any) -> Any:
    return {"ownership": {"kind": "conversation"}, "conversationId": conversation_id, **extra}


class Tree(NamedTuple):
    owner: Any
    child: Any
    inner: Any


async def owned_child(
    parent: Any, name: str, *, background: bool = False, slow_inner: bool = False, slow_owner: bool = False
) -> Tree:
    """In ``parent``, stage task ``owner`` and a conversation it owns holding task ``inner``, in one commit."""

    async def change(tx: Any) -> Tree:
        owner = await tx.create_task(
            Hold,
            {"name": name, **({"slowAbort": True} if slow_owner else {})},
            {"ownership": {"kind": "conversation"}, "background": background},
        )
        child = await tx.create_conversation(owned_by(owner))
        inner = await tx.create_task(
            Hold, {"name": f"{name}.inner", **({"slowAbort": True} if slow_inner else {})}, in_conversation(child["id"])
        )
        return Tree(owner, child["id"], inner)

    return await parent.commit(change, context)


async def busy(conversation: Any, task: Any) -> None:
    """Mark a conversation busy with ``task`` standing in for its run, so submissions queue."""

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation.id))["run"] = {"taskId": task, "inputs": []}

    await conversation.commit(change, context)


async def status(harness: Any, id: Any) -> Any:
    record = await harness.get_task(id, context)
    assert record is not None
    return record["state"]


async def is_status(harness: Any, id: Any, expected: str) -> bool:
    return (await status(harness, id))["status"] == expected


async def abort_requested(harness: Any, id: Any) -> bool:
    return bool(((await harness.get_task(id, context)) or {}).get("abortRequested"))


async def outcome(harness: Any, id: Any) -> str:
    return (await harness.wait_for_task(id, context))["state"]["outcome"]["status"]


def queued_note(text: str) -> Any:
    """A write that affects model context, so it queues while busy; a data-only write would be appended at once."""
    return {"kind": "note", "model": [{"role": "user", "content": text, "timestamp": 0}]}


class OpenedRoot(NamedTuple):
    harness: Any
    registry: Any
    reports: list[BaseException]
    root: Any


async def open_harness(storage: Any = None) -> OpenedRoot:
    opened = await open_tasks(storage if storage is not None else MemoryStorage(), [Hold, Waiter, Spawner])
    root = await opened.harness.root(context)
    opened.harness.resume()
    return OpenedRoot(opened.harness, opened.registry, opened.reports, root)


async def create_in(conversation: Any, name: str, **input: Any) -> Any:
    return await conversation.commit(lambda tx: tx.create_task(Hold, {"name": name, **input}, CONVERSATION), context)


# ─── ownership ──────────────────────────────────────────────────────────────


async def test_keeps_a_conversation_busy_while_its_owned_foreground_subtree_has_live_work_holding_the_completed_owner() -> (
    None
):
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner")
    open_gate("owner", "completed")
    await wait_for(lambda: is_status(harness, tree.owner, "completing"))
    idle = root.wait_for_idle(context)
    assert await settled(idle) is False
    assert await settled(harness.wait_for_idle(context)) is False
    open_gate("owner.inner", "completed")
    await idle
    await harness.wait_for_idle(context)
    assert (await status(harness, tree.owner))["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_holds_an_owner_for_work_in_a_conversation_its_ended_child_task_owns() -> None:
    harness, _registry, _reports, root = await open_harness()

    async def create(tx: Any) -> tuple[Any, Any, Any]:
        owner = await tx.create_task(Hold, {"name": "owner"}, CONVERSATION)
        child = await tx.create_task(Hold, {"name": "child"}, owned_by(owner))
        conversation = await tx.create_conversation(owned_by(child))
        return owner, child, conversation["id"]

    owner, child, conversation = await root.commit(create, context)
    open_gate("child", "completed")
    assert await outcome(harness, child) == "completed"
    # The ended child's conversation still belongs to the owner above it.
    handle = await harness.conversation(conversation, context)
    await create_in(handle, "late")
    open_gate("owner", "completed")
    await wait_for(lambda: is_status(harness, owner, "completing"))
    open_gate("late", "completed")
    assert await outcome(harness, owner) == "completed"
    await harness.close(context)


async def test_cascades_through_ended_tasks_into_an_owned_background_subtree_when_aborting_across_background() -> None:
    harness, _registry, _reports, root = await open_harness()

    # root conversation -> A -> B -> conversation -> background C -> D
    async def create(tx: Any) -> dict[str, Any]:
        a = await tx.create_task(Hold, {"name": "a"}, CONVERSATION)
        b = await tx.create_task(Hold, {"name": "b"}, owned_by(a))
        conversation = await tx.create_conversation(owned_by(b))
        c = await tx.create_task(Hold, {"name": "c"}, in_conversation(conversation["id"], background=True))
        d = await tx.create_task(Hold, {"name": "d"}, owned_by(c))
        return {"a": a, "b": b, "c": c, "d": d}

    tree = await root.commit(create, context)
    open_gate("b", "completed")
    open_gate("a", "completed")
    assert await outcome(harness, tree["a"]) == "completed"
    await root.abort(context, background=True)
    for id in [tree["c"], tree["d"]]:
        assert await outcome(harness, id) == "aborted"
    await harness.close(context)


async def test_keeps_ownership_queries_off_finished_subtrees() -> None:
    harness, _registry, _reports, root = await open_harness()
    # Ended chains under a live owner, each ending in an empty owned conversation.
    owner = await create_in(root, "owner")
    chains = 50
    tasks: list[Any] = []
    for n in range(chains):

        async def create(tx: Any, n: int = n) -> list[Any]:
            first = await tx.create_task(Hold, {"name": f"c{n}.0"}, owned_by(owner))
            second = await tx.create_task(Hold, {"name": f"c{n}.1"}, owned_by(first))
            await tx.create_conversation(owned_by(second))
            return [first, second]

        tasks.extend(await root.commit(create, context))
    for n in range(chains):
        open_gate(f"c{n}.1", "completed")
        open_gate(f"c{n}.0", "completed")
    for id in tasks:
        await harness.wait_for_task(id, context)
    # Upstream also counts Set iterations while the owner settles (see the module docstring).
    open_gate("owner", "completed")
    assert await outcome(harness, owner) == "completed"
    await harness.close(context)


async def test_tracks_ownership_chains_deeper_than_the_call_stack() -> None:
    harness, _registry, _reports, root = await open_harness()
    # Tasks, each owning the next, created in one commit above a live leaf; they end top-down and hold. Upstream
    # takes 20,000 against JavaScript's stack; twice Python's recursion limit is as deep for Python, and the
    # scheduler's wake scan is linear in live tasks per commit, so upstream's count would take minutes here.
    depth = 2 * sys.getrecursionlimit()

    async def create(tx: Any) -> tuple[Any, Any]:
        top = await tx.create_task(Hold, {"name": "top"}, CONVERSATION)
        owner = top
        for n in range(depth):
            owner = await tx.create_task(Hold, {"name": f"link{n}"}, owned_by(owner))
        leaf = await tx.create_task(Hold, {"name": "leaf"}, owned_by(owner))
        return top, leaf

    top, leaf = await root.commit(create, context)
    for n in range(depth):
        open_gate(f"link{n}", "completed")
    open_gate("top", "completed")
    await wait_for(lambda: is_status(harness, top, "completing"), 60_000)
    open_gate("leaf", "completed")
    assert await outcome(harness, leaf) == "completed"
    assert await outcome(harness, top) == "completed"
    await harness.close(context)


async def _p_b_a_c(tx: Any) -> dict[str, Any]:
    """Live P owns B, which owns A, which owns the empty conversation C."""
    p = await tx.create_task(Hold, {"name": "p"}, CONVERSATION)
    b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
    a = await tx.create_task(Hold, {"name": "a"}, owned_by(b))
    c = await tx.create_conversation(owned_by(a))
    return {"p": p, "a": a, "b": b, "c": c["id"]}


async def test_holds_an_owner_for_work_a_commit_creates_below_ended_tasks_loaded_from_storage(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_harness(await open_sqlite_storage(path))
    # Live P owns B, which owns A, which owns the empty conversation C; B and A end.
    tree = await opened.root.commit(_p_b_a_c, context)
    open_gate("a", "completed")
    open_gate("b", "completed")
    await opened.harness.wait_for_task(tree["a"], context)
    await opened.harness.wait_for_task(tree["b"], context)
    await opened.harness.close(context)
    # After reopen, B and A are known only once a chain load reads them.
    opened = await open_harness(await open_sqlite_storage(path))
    harness = opened.harness
    c = await harness.conversation(tree["c"], context)
    d = await create_in(c, "d")
    open_gate("p", "completed")
    await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    open_gate("d", "completed")
    assert await outcome(harness, d) == "completed"
    assert await outcome(harness, tree["p"]) == "completed"
    await harness.close(context)


async def test_holds_an_owner_for_work_created_later_below_ended_tasks_without_a_reopen() -> None:
    harness, _registry, _reports, root = await open_harness()
    # Live P owns B, which owns A, which owns the empty conversation C; B and A end, C stays.
    tree = await root.commit(_p_b_a_c, context)
    open_gate("a", "completed")
    open_gate("b", "completed")
    await harness.wait_for_task(tree["b"], context)
    c = await harness.conversation(tree["c"], context)
    d = await create_in(c, "d")
    open_gate("p", "completed")
    await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    open_gate("d", "completed")
    assert await outcome(harness, d) == "completed"
    assert await outcome(harness, tree["p"]) == "completed"
    await harness.close(context)


class Bridge(NamedTuple):
    harness: Any
    root: Any
    p: Any
    c: Any


async def ended_bridge(depth: int) -> Bridge:
    """Live P owns ``depth`` ended tasks in a chain, the last owning conversation C, empty when they ended.

    Work started in C later is still P's: every case below must reach it through the ended tasks.
    """
    harness, _registry, _reports, root = await open_harness()

    async def create(tx: Any) -> dict[str, Any]:
        p = await tx.create_task(Hold, {"name": "p", "slowAbort": True}, CONVERSATION)
        links: list[Any] = []
        owner = p
        for n in range(depth):
            owner = await tx.create_task(Hold, {"name": f"link{n}"}, owned_by(owner))
            links.append(owner)
        c = await tx.create_conversation(owned_by(owner))
        return {"p": p, "links": links, "c": c["id"]}

    tree = await root.commit(create, context)
    for n in range(depth - 1, -1, -1):
        open_gate(f"link{n}", "completed")
    for id in tree["links"]:
        await harness.wait_for_task(id, context)
    c = await harness.conversation(tree["c"], context)
    return Bridge(harness, root, tree["p"], c)


@pytest.mark.parametrize("depth", [2, 5])
async def test_aborts_work_below_ended_tasks_with_their_live_owner_which_waits_for_it(depth: int) -> None:
    harness, _root, p, c = await ended_bridge(depth)
    d = await create_in(c, "d")
    await harness.abort_task(p, context)
    assert await outcome(harness, d) == "aborted"
    # P's abort handler ran only after D ended.
    open_gate("abort.p", "completed")
    assert await outcome(harness, p) == "aborted"
    await harness.close(context)


@pytest.mark.parametrize("depth", [2, 5])
async def test_keeps_the_root_busy_and_aborts_with_it_work_below_ended_tasks(depth: int) -> None:
    harness, root, p, c = await ended_bridge(depth)
    open_gate("p", "completed")
    await harness.wait_for_task(p, context)
    # P has ended: D and D2 hang below ended tasks only, and are still the root conversation's work.
    d = await create_in(c, "d")
    d2 = await create_in(c, "d2")
    assert await settled(root.wait_for_idle(context)) is False
    assert await settled(harness.wait_for_idle(context)) is False
    await root.abort(context)
    for id in [d, d2]:
        assert await outcome(harness, id) == "aborted"
    await harness.close(context)


async def test_reaches_work_below_ended_tasks_across_two_owned_conversations() -> None:
    harness, _registry, _reports, root = await open_harness()

    # Live P -> ended B -> conversation C -> ended X -> ended A -> conversation C2; work starts in C2 later. X owns no
    # conversation, only the task A, so only owning a known task keeps it.
    async def create(tx: Any) -> dict[str, Any]:
        p = await tx.create_task(Hold, {"name": "p"}, CONVERSATION)
        b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
        c = await tx.create_conversation(owned_by(b))
        x = await tx.create_task(Hold, {"name": "x"}, in_conversation(c["id"]))
        a = await tx.create_task(Hold, {"name": "a"}, owned_by(x))
        c2 = await tx.create_conversation(owned_by(a))
        return {"p": p, "b": b, "x": x, "c2": c2["id"]}

    tree = await root.commit(create, context)
    open_gate("a", "completed")
    open_gate("x", "completed")
    open_gate("b", "completed")
    await harness.wait_for_task(tree["b"], context)
    c2 = await harness.conversation(tree["c2"], context)
    d = await create_in(c2, "d")
    open_gate("p", "completed")
    await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    assert await settled(root.wait_for_idle(context)) is False
    await root.abort(context)
    assert await outcome(harness, d) == "aborted"
    assert await outcome(harness, tree["p"]) == "completed"
    await harness.close(context)


@pytest.mark.parametrize("ending", ["completes", "is aborted"])
async def test_frees_an_owner_once_the_last_work_below_ended_tasks_ends_after_a_background_task_there(
    ending: str,
) -> None:
    harness, _registry, _reports, root = await open_harness()

    # Live P owns B, which owns A, which owns conversation C with background Z and, later, foreground D.
    async def create(tx: Any) -> dict[str, Any]:
        tree = await _p_b_a_c(tx)
        tree["z"] = await tx.create_task(Hold, {"name": "z"}, in_conversation(tree["c"], background=True))
        return tree

    tree = await root.commit(create, context)
    open_gate("a", "completed")
    open_gate("b", "completed")
    await harness.wait_for_task(tree["b"], context)
    c = await harness.conversation(tree["c"], context)
    d = await create_in(c, "d", slowAbort=True)
    if ending == "completes":
        open_gate("p", "completed")
        await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    else:
        # P waits for D's abort before its own abort handler runs.
        await harness.abort_task(tree["p"], context)
        await wait_for(lambda: abort_requested(harness, d))
    # Z ends first, D last.
    open_gate("z", "completed")
    await harness.wait_for_task(tree["z"], context)
    if ending == "completes":
        open_gate("d", "completed")
    else:
        open_gate("abort.d", "completed")
    await harness.wait_for_task(d, context)
    expected = "completed" if ending == "completes" else "aborted"
    assert await outcome(harness, tree["p"]) == expected
    await harness.close(context)


async def test_finalizes_many_held_siblings_that_end_together_in_one_pass() -> None:
    harness, _registry, _reports, root = await open_harness()
    # Top holds 1000 held children, each holding one live leaf; the leaves end in one commit.
    count = 1000

    async def create(tx: Any) -> tuple[Any, list[Any]]:
        top = await tx.create_task(Hold, {"name": "top"}, CONVERSATION)
        siblings = [await tx.create_task(Hold, {"name": f"s{n}"}, owned_by(top)) for n in range(count)]
        # Leaves in reverse sibling order: the last sibling is freed first, so a holder checked after each freed
        # sibling would scan the siblings already finalized each time.
        leaves = [
            await tx.create_task(Hold, {"name": f"l{n}"}, owned_by(siblings[n])) for n in range(count - 1, -1, -1)
        ]
        return top, leaves

    top, leaves = await root.commit(create, context)
    for n in range(count):
        open_gate(f"s{n}", "completed")
    open_gate("top", "completed")
    await wait_for(lambda: is_status(harness, top, "completing"), 30_000)

    async def end_leaves(tx: Any) -> None:
        # Reads before the first write. Ended last to first, so a holder checked after each sibling would scan the
        # ones already finalized every time.
        records = [await tx.task(id) for id in leaves]
        for record in records:
            tx.set_task({**record, "state": {"status": "terminal", "outcome": {"status": "completed", "result": None}}})

    await root.commit(end_leaves, context)
    # Upstream also counts Set iterations while it all finalizes (see the module docstring).
    assert await outcome(harness, top) == "completed"
    await harness.close(context)


async def test_rejects_a_conversation_created_in_the_commit_that_ends_its_owner_task() -> None:
    """Pins the Transaction rule that a conversation cannot be created in the commit that ends its owner task."""
    harness, _registry, _reports, root = await open_harness()
    spawner = await root.commit(lambda tx: tx.create_task(Spawner, {"name": "spawned"}, CONVERSATION), context)
    result = (await harness.wait_for_task(spawner, context))["state"]["outcome"]
    assert result["status"] == "faulted"
    assert list(result) == ["status", "error"] and list(result["error"]) == ["message"]
    assert result["error"]["message"] in (
        f"Conversation owner task {spawner} is completing",
        f"Conversation owner task {spawner} is terminal",
    )
    await harness.close(context)


async def test_stops_idle_traversal_at_a_background_owner() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "background", background=True)
    await root.wait_for_idle(context)
    await harness.wait_for_idle(context)
    # The background child is its own scope.
    child = await harness.conversation(tree.child, context)
    assert await settled(child.wait_for_idle(context)) is False
    await child.abort(context)
    await harness.close(context)


async def test_cascades_an_abort_mark_to_the_owned_foreground_subtree_and_withdraws_its_queued_inputs() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner")
    nested = await harness.conversation(tree.child, context)
    deeper = await owned_child(nested, "deeper")
    shielded = await owned_child(nested, "shielded", background=True)
    # Busy conversations queue submissions; the inner task stands in for the child's run.
    await busy(nested, tree.inner)
    queued = (await nested.submit({"type": "input", "content": "later"}, context)).id
    assert await harness.abort_task(tree.owner, context) == "marked"
    for id in [tree.owner, tree.inner, deeper.owner, deeper.inner]:
        assert await outcome(harness, id) == "aborted"
    # A nested background owner is a boundary.
    assert (await status(harness, shielded.owner))["status"] != "terminal"
    assert (await status(harness, shielded.inner))["status"] != "terminal"
    submission = await harness.submission(queued, context)
    assert match_object(await submission.status(context), {"status": "unanswered", "reason": "aborted"})
    await harness.abort_task(shielded.owner, context)
    await harness.close(context)


async def test_cascades_a_failed_owner_but_not_a_completed_one() -> None:
    harness, _registry, _reports, root = await open_harness()
    failed = await owned_child(root, "failed")
    completed_tree = await owned_child(root, "done")
    open_gate("failed", "failed")
    open_gate("done", "completed")
    assert await outcome(harness, failed.inner) == "aborted"
    assert await outcome(harness, failed.owner) == "failed"
    await wait_for(lambda: is_status(harness, completed_tree.owner, "completing"))
    assert (await status(harness, completed_tree.inner))["status"] != "terminal"
    open_gate("done.inner", "completed")
    assert await outcome(harness, completed_tree.owner) == "completed"
    await harness.close(context)


async def test_aborts_a_background_task_directly_with_its_ordinary_subtree() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "background", background=True)
    await harness.abort_task(tree.owner, context)
    assert await outcome(harness, tree.inner) == "aborted"
    await harness.close(context)


async def test_aborts_a_conversation_queued_inputs_withdrawn_writes_kept_foreground_work_aborted_background_kept() -> (
    None
):
    harness, _registry, _reports, root = await open_harness()
    foreground = await owned_child(root, "foreground")
    background = await owned_child(root, "background", background=True)
    await busy(root, foreground.owner)
    input = (await root.submit({"type": "input", "content": "later"}, context)).id
    write = (await root.submit({"type": "write", "entry": queued_note("w")}, context)).id
    await root.abort(context)
    for id in [foreground.owner, foreground.inner]:
        assert (await status(harness, id))["status"] == "terminal"
    assert (await status(harness, background.owner))["status"] != "terminal"
    assert (await status(harness, background.inner))["status"] != "terminal"
    assert match_object(await (await harness.submission(input, context)).status(context), {"reason": "aborted"})
    assert (await (await harness.submission(write, context)).status(context))["status"] == "queued"
    await harness.abort_task(background.owner, context)
    await harness.close(context)


async def test_aborts_work_created_below_a_held_failed_owner_but_not_below_a_terminal_one() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner", slow_inner=True)
    open_gate("owner", "failed")
    await wait_for(lambda: abort_requested(harness, tree.inner))
    assert (await status(harness, tree.owner))["status"] == "completing"
    child = await harness.conversation(tree.child, context)
    during = await create_in(child, "during")
    assert await outcome(harness, during) == "aborted"
    open_gate("abort.owner.inner", "completed")
    assert await outcome(harness, tree.owner) == "failed"
    # A terminal owner never cascades: interrogating its conversation runs normally.
    after = await create_in(child, "after")
    open_gate("after", "completed")
    assert await outcome(harness, after) == "completed"
    await harness.close(context)


async def test_withdraws_queued_inputs_below_the_aborted_task_but_keeps_its_own_conversations_queue_and_queued_writes() -> (
    None
):
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner")
    child = await harness.conversation(tree.child, context)
    await busy(root, tree.owner)
    await busy(child, tree.inner)
    own = (await root.submit({"type": "input", "content": "own"}, context)).id
    below = (await child.submit({"type": "input", "content": "below"}, context)).id
    write = (await child.submit({"type": "write", "entry": queued_note("w")}, context)).id
    await harness.abort_task(tree.owner, context)
    await harness.wait_for_task(tree.inner, context)

    async def status_of(id: Any) -> str:
        return (await (await harness.submission(id, context)).status(context))["status"]

    assert await status_of(own) == "queued"
    assert await status_of(below) == "unanswered"
    assert await status_of(write) == "queued"
    await harness.close(context)


async def test_keeps_a_nested_background_owners_subtree_when_its_cancelled_background_owner_is_aborted() -> None:
    harness, _registry, _reports, root = await open_harness()
    outer = await owned_child(root, "outer", background=True)
    inner = await owned_child(await harness.conversation(outer.child, context), "inner", background=True)
    await harness.abort_task(outer.owner, context)
    assert await outcome(harness, outer.inner) == "aborted"
    assert (await status(harness, inner.owner))["status"] != "terminal"
    assert (await status(harness, inner.inner))["status"] != "terminal"
    await harness.abort_task(inner.owner, context)
    await harness.close(context)


async def test_decides_idle_after_reopen_from_owner_edges_it_has_to_load_first(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_harness(await open_sqlite_storage(path))
    foreground = await owned_child(opened.root, "fg")
    deeper = await owned_child(await opened.harness.conversation(foreground.child, context), "fg2")
    background = await owned_child(opened.root, "bg", background=True)
    open_gate("fg", "completed")
    open_gate("fg2", "completed")
    # Both owners hold their outcomes while the work below them runs.
    harness = opened.harness
    await wait_for(lambda: is_status(harness, foreground.owner, "completing"))
    await wait_for(lambda: is_status(harness, deeper.owner, "completing"))
    await harness.close(context)

    opened = await open_harness(await open_sqlite_storage(path))
    # Two levels below completed foreground owners, the inner tasks keep the root busy.
    assert await settled(opened.root.wait_for_idle(context)) is False
    assert await settled(opened.harness.wait_for_idle(context)) is False
    open_gate("fg.inner", "completed")
    open_gate("fg2.inner", "completed")
    # The background subtree does not count.
    await opened.root.wait_for_idle(context)
    await opened.harness.wait_for_idle(context)
    assert (await status(opened.harness, foreground.owner))["status"] == "terminal"
    assert (await status(opened.harness, background.inner))["status"] != "terminal"
    await opened.harness.abort_task(background.owner, context)
    await opened.harness.close(context)


@pytest.mark.parametrize(
    ("abort_requested_flag", "state"),
    [
        (False, {"status": "completing", "outcome": {"status": "failed", "error": {"message": "crash"}}}),
        (True, {"status": "pending", "checkpoint": {"phase": "hold"}}),
    ],
    ids=["a held failed owner", "an abort-marked owner"],
)
async def test_derives_marks_a_crash_left_unapplied_below_an_owner_at_open(
    tmp_path: Path, abort_requested_flag: bool, state: Any
) -> None:
    path = tmp_path / "session.sqlite"
    # Without a Harness, nothing derives marks: a cancelled owner with a live task below it.
    session = create_session(await open_sqlite_storage(path))

    async def create(tx: Any) -> tuple[Any, Any]:
        root = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        owner = await tx.create_task(Hold, {"name": "gone"}, in_conversation(root["id"]))
        child = await tx.create_conversation(owned_by(owner))
        return owner, await tx.create_task(Hold, {"name": "orphan"}, in_conversation(child["id"]))

    owner, inner = await session.commit(create, context)

    async def cancel(tx: Any) -> None:
        record = await tx.task(owner)
        tx.set_task({**record, "abortRequested": abort_requested_flag, "state": state})

    await session.commit(cancel, context)
    await session.close(context)

    harness = (await open_harness(await open_sqlite_storage(path))).harness
    assert await outcome(harness, inner) == "aborted"
    # The owner finishes only after the work below it.
    assert await outcome(harness, owner) == ("aborted" if abort_requested_flag else "failed")
    await harness.close(context)


async def test_withdraws_an_input_queued_below_a_held_failed_owner_after_its_cascade() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner", slow_inner=True)
    child = await harness.conversation(tree.child, context)
    # The inner task stands in for the child's run, which stays busy after the cascade.
    await busy(child, tree.inner)
    open_gate("owner", "failed")
    await wait_for(lambda: abort_requested(harness, tree.inner))
    late = await child.submit({"type": "input", "content": "late"}, context)
    assert match_object(await late.wait(context), {"status": "unanswered", "reason": "aborted"})
    open_gate("abort.owner.inner", "completed")
    await harness.wait_for_task(tree.owner, context)
    await harness.close(context)


async def cancelled_owner_closed(path: Path) -> Tree:
    """An owner left live and cancelled, before its slow abort handler starts, by a close; its inner task ended.

    Python: the Session line is held behind the abort mark until close has sealed, so the abort handler, which
    ignores its signal, never starts in this Harness. Upstream awaits ``abortTask()`` and then closes: JavaScript
    reaches the close before the abort invocation starts by microtask order alone, while Python's loop turns let the
    scheduler start it first.
    """
    opened = await open_harness(await open_sqlite_storage(path))
    tree = await owned_child(opened.root, "owner", slow_owner=True)
    open_gate("owner.inner", "completed")
    await opened.harness.wait_for_task(tree.inner, context)
    # The owner stays live and cancelled in its slow abort handler.
    aborting = opened.harness.abort_task(tree.owner, context)
    entered = deferred()
    release = deferred()

    async def hold(_tx: Any) -> None:
        resolve(entered)
        await release

    holding = opened.root.commit(hold, context)
    await entered
    closing = opened.harness.close(context)
    resolve(release)
    await holding
    assert await aborting == "marked"
    await closing
    return tree


async def test_marks_work_admitted_after_reopen_below_a_cancelled_owner_whose_edge_was_not_loaded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    tree = await cancelled_owner_closed(path)

    # The child is empty at open, so nothing loads its edge until new work arrives.
    opened = await open_harness(await open_sqlite_storage(path))
    child = await opened.harness.conversation(tree.child, context)
    late = await create_in(child, "late")
    assert await outcome(opened.harness, late) == "aborted"
    open_gate("abort.owner", "completed")
    assert await outcome(opened.harness, tree.owner) == "aborted"
    await opened.harness.close(context)


async def test_cascades_from_an_owner_the_scheduler_orphans() -> None:
    harness, _registry, _reports, root = await open_harness()

    async def create(tx: Any) -> tuple[Any, Any]:
        owner = await tx.create_task(Unregistered, {"name": "unregistered"}, CONVERSATION)
        child = await tx.create_conversation(owned_by(owner))
        return owner, await tx.create_task(Hold, {"name": "below"}, in_conversation(child["id"]))

    owner, inner = await root.commit(create, context)
    assert await harness.abort_task(owner, context) == "marked"
    assert (await harness.wait_for_task(owner, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "missing_task",
    }
    assert await outcome(harness, inner) == "aborted"
    await harness.close(context)


async def test_cancels_only_the_callers_wait_never_the_shared_work() -> None:
    harness, _registry, _reports, root = await open_harness()
    tree = await owned_child(root, "owner")
    waiting = AbortController()
    idle = root.wait_for_idle(with_abort_signal(waiting.signal, context))
    waiting.abort(RuntimeError("stop waiting"))
    with pytest.raises(Exception, match="stop waiting"):
        await idle
    assert (await status(harness, tree.inner))["status"] != "terminal"

    # Cancelling an abort after its commit leaves the marks in place.
    slow = await owned_child(root, "slow")

    async def make_slow(tx: Any) -> None:
        record = await tx.task(slow.owner)
        tx.set_task({**record, "input": {"name": "slow", "slowAbort": True}})

    await root.commit(make_slow, context)
    aborting = AbortController()
    abort = root.abort(with_abort_signal(aborting.signal, context))
    await wait_for(lambda: abort_requested(harness, slow.owner))
    aborting.abort(RuntimeError("stop aborting"))
    with pytest.raises(Exception, match="stop aborting"):
        await abort
    open_gate("abort.slow", "completed")
    for id in [tree.owner, tree.inner, slow.owner, slow.inner]:
        assert await outcome(harness, id) == "aborted"
    await harness.close(context)


async def test_aborts_a_waiting_child_whose_awaited_task_completes_in_the_commit_that_marks_its_owner() -> None:
    harness, _registry, _reports, root = await open_harness()

    async def create(tx: Any) -> tuple[Any, Any, Any]:
        owner = await tx.create_task(Hold, {"name": "owner"}, CONVERSATION)
        child = await tx.create_conversation(owned_by(owner))
        dependency = await tx.create_task(Hold, {"name": "dependency"}, CONVERSATION)
        blocked = await tx.create_task(Waiter, {"on": [dependency]}, in_conversation(child["id"]))
        return owner, dependency, blocked

    owner, dependency, blocked = await root.commit(create, context)
    await wait_for(lambda: runs.get("dependency") == 1 and runs.get("owner") == 1)
    await wait_for(lambda: is_status(harness, blocked, "waiting"))

    # One commit completes the dependency and marks the owner.
    async def change(tx: Any) -> None:
        completed_dependency = await tx.task(dependency)
        marked = await tx.task(owner)
        tx.set_task(
            {
                **completed_dependency,
                "state": {"status": "terminal", "outcome": {"status": "completed", "result": None}},
            }
        )
        tx.set_task({**marked, "abortRequested": True})

    await root.commit(change, context)
    assert await outcome(harness, blocked) == "aborted"
    await harness.close(context)


class RejectMark(StorageSpy):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.reject_mark: Any = None

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        if any(_marks(write, self.reject_mark) for write in writes):
            self.reject_mark = None
            raise RuntimeError("disk gone")
        return await self.inner.commit(writes, context)


def _marks(write: Any, id: Any) -> bool:
    return write["type"] == "task" and write["value"]["id"] == id and bool(write["value"].get("abortRequested"))


async def test_applies_marks_found_through_an_edge_loaded_after_reopen_once_a_failed_cascade_commit_is_reopened(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    tree = await cancelled_owner_closed(path)

    storage = RejectMark(await open_sqlite_storage(path))
    opened = await open_harness(storage)
    child = await opened.harness.conversation(tree.child, context)

    async def create(tx: Any) -> Any:
        id = await tx.create_task(Hold, {"name": "late"}, CONVERSATION)
        storage.reject_mark = id
        return id

    late = await child.commit(create, context)
    # The failed cascade commit fails the Harness. `closed` settles once its invocations have ended, and the owner's
    # abort handler ignores its signal: let it return.
    open_gate("abort.owner", "completed")
    assert (await opened.harness.closed)["reason"] == "failed"
    await opened.harness.close(context)
    # Reopening derives the mark again.
    opened = await open_harness(await open_sqlite_storage(path))
    assert await outcome(opened.harness, late) == "aborted"
    await opened.harness.close(context)


async def test_fails_the_harness_on_a_failed_cascade_commit_and_reopening_applies_the_cascade() -> None:
    failure = RuntimeError("disk gone")
    fail_mark: list[Any] = [None]

    class Failing(ControlledStorage):
        async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
            if any(_marks(write, fail_mark[0]) for write in writes):
                fail_mark[0] = None
                raise failure
            return await super().commit(writes, context)

    storage = Failing(MemoryStorage(), STORAGE_BACKENDS["memory"])
    harness, _registry, reports, root = await open_harness(storage)
    tree = await owned_child(root, "owner")
    fail_mark[0] = tree.inner
    open_gate("owner", "failed")
    assert await harness.closed == {"reason": "failed", "error": failure}
    assert reports == [failure]
    await harness.close(context)
    reopened = await open_harness(await storage.reopen())
    assert await outcome(reopened.harness, tree.inner) == "aborted"
    await reopened.harness.close(context)


# ─── owned conversations from tools and supervisors ─────────────────────────


def text(value: str) -> Any:
    return {"output": [TextContent(value)]}


def call(name: str, id: str, args: Any = None) -> Any:
    return faux_assistant_message([faux_tool_call(name, args or {}, id=id)], stop_reason="toolUse")


async def create_child(api: Any, call_context: Context, *, configured: bool = True) -> Any:
    async def change(tx: Any) -> Any:
        created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
        if configured:
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
        return created["id"]

    return await api.commit(change, call_context)


async def test_gives_a_tool_an_invocation_bound_handle_whose_submissions_stay_durable_after_the_call_ends() -> None:
    setup = chat_setup()
    seen: dict[str, Any] = {"missing": object()}

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        child = await create_child(api, call_context)
        seen["missing"] = await api.conversation(99_999, call_context)
        handle = await api.conversation(child, call_context)
        seen["handle"] = handle
        submission = await handle.submit({"type": "input", "content": "child task", "requestId": "child"}, call_context)
        seen["submission"] = submission
        result = await submission.wait(call_context)
        return text(result["status"])

    add_tool(
        setup.registry,
        define_tool(name="delegate", description="Delegates", parameters=Type.Object({}), execute=execute),
    )
    setup.faux.set_responses(
        [
            call("delegate", "c1"),
            faux_assistant_message([faux_text("child answer")]),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert seen["missing"] is None
    handle, submission = seen["handle"], seen["submission"]
    # The call ended: the handle and its submission reject, while the submission record remains.
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.submit({"type": "input", "content": "again"}, context)
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.wait_for_idle(context)
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.abort(context)
    with pytest.raises(Exception, match="invocation has ended"):
        await submission.wait(context)
    # Nothing was admitted or marked after the call ended.
    child_conversation = await harness.conversation(handle.id, context)
    child_entries = (await child_conversation.entries({}, 100, None, context))["items"]
    assert len([entry for entry in child_entries if entry["kind"] == "pi.user"]) == 1
    child_tasks = [
        task for task in (await harness.inspect(context))["tasks"] if task["record"]["conversationId"] == handle.id
    ]
    assert child_tasks == []
    assert (await (await harness.submission(submission.id, context)).status(context))["status"] == "done"
    await harness.close(context)


async def test_rejects_a_handle_operation_queued_on_the_line_when_the_invocation_ends_before_it_runs() -> None:
    setup = chat_setup()
    ready = deferred()
    go = deferred()
    queued = deferred()

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        child = await create_child(api, call_context, configured=False)
        handle = await api.conversation(child, call_context)
        resolve(ready, (child, api.task_id))
        await go
        # Called while the invocation is alive, with a context that is not the call's: the handle binds it to the
        # invocation. It reaches the line only after the abort mark.
        submitting = asyncio.ensure_future(handle.submit({"type": "input", "content": "late"}, context))
        submitting.add_done_callback(lambda future: future.cancelled() or future.exception())
        resolve(queued, submitting)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    add_tool(
        setup.registry,
        define_tool(name="delegate", description="Delegates late", parameters=Type.Object({}), execute=execute),
    )
    setup.faux.set_responses([call("delegate", "c1")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.submit({"type": "input", "content": "go"}, context)
    child, task_id = await ready
    # Hold the Session line, queue the abort mark behind it, then let the tool queue its submit.
    release = deferred()

    async def hold(_tx: Any) -> None:
        await release

    holding = root.commit(hold, context)
    aborting = harness.abort_task(task_id, context)
    resolve(go)
    submitting = await queued
    resolve(release)
    await holding
    await aborting
    with pytest.raises(BaseException):  # noqa: B017 - upstream: rejects with any error
        await submitting
    child_conversation = await harness.conversation(child, context)
    assert (await child_conversation.entries({}, 10, None, context))["items"] == []
    assert [
        record for record in (await harness.inspect(context))["submissions"] if record["conversationId"] == child
    ] == []
    await harness.close(context)


async def test_aborts_a_subagent_run_with_its_parents_conversation_rejecting_the_waiting_tool() -> None:
    setup = chat_setup()
    seen: dict[str, Any] = {}
    tool_wait = deferred()

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        child = await create_child(api, call_context)
        seen["child"] = child
        submission = await (await api.conversation(child, call_context)).submit(
            {"type": "input", "content": "child task"}, call_context
        )
        try:
            return text((await submission.wait(call_context))["status"])
        except BaseException as error:
            resolve(tool_wait, str(error) or type(error).__name__)
            raise

    add_tool(
        setup.registry,
        define_tool(name="delegate", description="Delegates", parameters=Type.Object({}), execute=execute),
    )
    child_run = gated(faux_assistant_message([faux_text("never")]))
    setup.faux.set_responses([call("delegate", "c1"), child_run.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "go"}, context)
    await child_run.reached
    await root.abort(context)
    assert match_object(await input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert await tool_wait
    child_conversation = await harness.conversation(seen["child"], context)
    await child_conversation.wait_for_idle(context)
    assert await harness.snapshot(LiveDoc, seen["child"], context) == {}
    child_inputs = [s for s in (await harness.inspect(context))["submissions"] if s["conversationId"] == seen["child"]]
    assert child_inputs == []
    await harness.wait_for_idle(context)
    await harness.close(context)


async def test_lets_a_running_tool_abort_its_owned_child_and_continue() -> None:
    setup = chat_setup()
    child_run = gated(faux_assistant_message([faux_text("never")]))

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        child = await create_child(api, call_context)
        handle = await api.conversation(child, call_context)
        submission = await handle.submit({"type": "input", "content": "child task"}, call_context)
        await child_run.reached
        await handle.abort(call_context)
        settled_child = await submission.wait(call_context)
        return text(f"child {settled_child['status']}")

    add_tool(
        setup.registry,
        define_tool(name="delegate", description="Delegates and cancels", parameters=Type.Object({}), execute=execute),
    )
    setup.faux.set_responses([call("delegate", "c1"), child_run.step, faux_assistant_message([faux_text("done")])])
    opened = await open_chat(MemoryStorage(), setup)
    result = await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert result["status"] == "done"
    await opened.harness.close(context)


async def test_aborts_the_children_of_a_tool_call_that_throws_or_is_interrupted_while_the_run_continues(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    children: list[Any] = []
    tool_running = deferred()
    setup = chat_setup()
    add_task(setup.registry, Hold)

    async def execute(args: Any, api: Any, call_context: Context) -> Any:
        async def change(tx: Any) -> Any:
            created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            return await tx.create_task(Hold, {"name": f"child.{api.call_id}"}, in_conversation(created["id"]))

        children.append(await api.commit(change, call_context))
        if args["mode"] == "throw":
            raise RuntimeError("spawn failed")
        resolve(tool_running)
        return await aborted(call_context.abort_signal)  # type: ignore[arg-type]

    add_tool(
        setup.registry,
        define_tool(
            name="spawn",
            description="Starts owned work, then throws or hangs",
            parameters=Type.Object({"mode": Type.String()}),
            execute=execute,
        ),
    )
    setup.faux.set_responses(
        [
            call("spawn", "c1", {"mode": "throw"}),
            faux_assistant_message([faux_text("after throw")]),
            call("spawn", "c2", {"mode": "hang"}),
        ]
    )
    opened = await open_chat(await open_sqlite_storage(path), setup)
    result = await (await opened.root.submit({"type": "input", "content": "one"}, context)).wait(context)
    assert result["status"] == "done"
    assert await outcome(opened.harness, children[0]) == "aborted"

    # The second call hangs until the process stops; on reopen it is interrupted.
    second = (await opened.root.submit({"type": "input", "content": "two"}, context)).id
    await tool_running
    await opened.harness.close(context)
    setup.faux.set_responses([faux_assistant_message([faux_text("after interrupt")])])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    assert await outcome(opened.harness, children[1]) == "aborted"
    assert (await (await opened.harness.submission(second, context)).wait(context))["status"] == "done"
    tools = (await opened.harness.commit(lambda tx: tx.scan_tasks({"kind": "pi.tool"}, 10), context))["items"]
    assert [task["state"]["status"] == "terminal" and task["state"]["outcome"]["status"] for task in tools] == [
        "failed",
        "failed",
    ]
    await opened.harness.close(context)


async def test_reruns_a_replay_safe_subagent_tool_after_a_restart_with_the_same_child_and_submission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    children: list[Any] = []
    setup = chat_setup()

    async def execute(_args: Any, api: Any, call_context: Context) -> Any:
        async def change(tx: Any) -> Any:
            existing = (await tx.scan_conversations({"ownerTaskId": api.task_id}, 1))["items"]
            if existing:
                return existing[0]["id"]
            created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
            return created["id"]

        child = await api.commit(change, call_context)
        children.append(child)
        request = {"type": "input", "content": "child task", "requestId": f"subagent:{api.task_id}"}
        submission = await (await api.conversation(child, call_context)).submit(request, call_context)
        result = await submission.wait(call_context)
        return text(result["status"])

    add_tool(
        setup.registry,
        define_tool(
            name="subagent", description="Delegates", parameters=Type.Object({}), replay="safe", execute=execute
        ),
    )
    child_run = gated(faux_assistant_message([faux_text("never")]))
    setup.faux.set_responses([call("subagent", "c1"), child_run.step])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    input = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await child_run.reached
    await opened.harness.close(context)

    setup.faux.set_responses(
        [faux_assistant_message([faux_text("child answer")]), faux_assistant_message([faux_text("done")])]
    )
    opened = await open_chat(await open_sqlite_storage(path), setup)
    assert (await (await opened.harness.submission(input, context)).wait(context))["status"] == "done"
    assert len(children) == 2
    assert children[1] == children[0]
    child = await opened.harness.conversation(children[0], context)
    entries = (await child.entries({}, 100, None, context))["items"]
    assert len([entry for entry in entries if entry["kind"] == "pi.user"]) == 1
    results = [message for message in (await opened.root.context(context))["messages"] if message.role == "toolResult"]
    assert [message.is_error for message in results] == [False]
    await opened.harness.close(context)


async def test_lets_a_background_supervisor_resubmit_after_a_restart_without_submitting_twice(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    Children = define_doc(
        kind="test.children", version=1, scope="conversation", history="latest", fork="initial", initial=lambda: {}
    )

    async def run(task: Any, runtime: Any, ctx: Context) -> None:
        id = (await runtime.snapshot(Children, task["input"]["parent"], ctx))["child"]
        child = await runtime.conversation(id, ctx)
        submission = await child.submit({"type": "input", "content": "work", "requestId": "stable"}, ctx)
        result = await submission.wait(ctx)
        if result["status"] != "done" or result["type"] != "input":
            raise RuntimeError(result["status"])
        answer = result["answer"]
        await runtime.commit(_terminal({"status": "completed", "result": {"answer": answer}}), ctx)

    Supervisor = define_task(
        name="test.supervisor", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_abort
    )
    setup = chat_setup()
    add_task(setup.registry, Supervisor)
    first = gated(faux_assistant_message([faux_text("never")]))
    setup.faux.set_responses([first.step, faux_assistant_message([faux_text("answer")])])
    opened = await open_chat(await open_sqlite_storage(path), setup)
    root = opened.root

    async def create(tx: Any) -> tuple[Any, Any]:
        supervisor = await tx.create_task(
            Supervisor, {"parent": root.id}, {"ownership": {"kind": "conversation"}, "background": True}
        )
        created = await tx.create_conversation(owned_by(supervisor))
        await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
        (await tx.doc(Children, root.id))["child"] = created["id"]
        return supervisor, created["id"]

    supervisor, child = await root.commit(create, context)
    opened.harness.resume()
    # The submission is admitted and its generation requested; then the process stops.
    await first.reached
    await opened.harness.close(context)

    opened = await open_chat(await open_sqlite_storage(path), setup)
    opened.harness.resume()
    done = await opened.harness.wait_for_task(supervisor, context)
    assert done["state"]["outcome"]["status"] == "completed"
    child_conversation = await opened.harness.conversation(child, context)
    entries = (await child_conversation.entries({}, 100, None, context))["items"]
    assert len([entry for entry in entries if entry["kind"] == "pi.user"]) == 1
    # The root never waited for the background supervisor.
    await opened.root.wait_for_idle(context)
    await opened.harness.close(context)
