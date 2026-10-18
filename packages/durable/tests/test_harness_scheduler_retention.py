"""Port of ``test/harness-scheduler-retention.test.ts``.

Scheduler memory is bounded by live work: an ended task or a conversation edge is kept only while live work is below
it, and a chain dropped that way is loaded again from storage when work appears below it.

Adaptations:

- Upstream rebinds storage methods on the SQLite storage instance; a :class:`StorageSpy` subclass intercepts them.
- SQLite runs on executor threads here, so waits poll in real time (``wait_for``); upstream's ``node:sqlite`` is
  synchronous, so a fixed number of turns (``eventually``) suffices there.
- The index sizes are ``Harness.scheduler_index_sizes`` (upstream's ``SCHEDULER_INDEX_SIZES`` symbol).
- "forgets a 20,000-deep chain of ended tasks without recursion" builds a chain twice Python's recursion limit
  deep (each task costs SQLite commits on executor threads and a wake scan linear in live tasks, so 20,000 would
  take minutes here).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from durable_harness_support import (
    StorageSpy,
    aborted,
    match_object,
    open_tasks,
    race_aborted,
    settled,
    wait_for,
)
from durable_session_support import context, deferred, resolve

from pi_chord.context import Context
from pi_durable import LiveDoc, define_task
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

gates: dict[str, asyncio.Future[Any]] = {}
#: How often each named task started its phase.
runs: dict[str, int] = {}
#: Rejected commits it caught.
rejections: list[BaseException] = []


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
    rejections.clear()


def _terminal(outcome: Any) -> Any:
    return lambda _tx, _current: {"status": "terminal", "outcome": outcome}


COMPLETED: Any = {"status": "completed", "result": None}


async def _abort(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(_terminal({"status": "aborted"}), ctx)


async def _hold(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    runs[name] = runs.get(name, 0) + 1
    await race_aborted(gate(name), runtime.signal)
    await runtime.commit(_terminal(COMPLETED), ctx)


#: Holds until its gate opens, then completes; its abort handler ends it ``aborted``.
Hold = define_task(
    name="test.retention.hold",
    version=1,
    initial=lambda _input: {"phase": "hold"},
    phases={"hold": _hold},
    abort=_abort,
)


def in_conversation(conversation_id: Any, **extra: Any) -> Any:
    return {"ownership": {"kind": "conversation"}, "conversationId": conversation_id, **extra}


def owned_by(task_id: Any) -> Any:
    return {"ownership": {"kind": "task", "taskId": task_id}}


CONVERSATION: Any = {"ownership": {"kind": "conversation"}}


async def _spawn(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    await race_aborted(gate(name), runtime.signal)

    async def change(tx: Any, _current: Any) -> Any:
        await tx.create_task(Hold, {"name": f"inside.{name}"}, in_conversation(task["input"]["into"]))
        return {"status": "terminal", "outcome": COMPLETED}

    await runtime.commit(change, ctx)


#: Holds until its gate opens, then creates a held task ``inside.<name>`` in conversation ``input.into`` and ends.
Spawner = define_task(
    name="test.retention.spawner",
    version=1,
    initial=lambda _input: {"phase": "hold"},
    phases={"hold": _spawn},
    abort=_abort,
)


async def _rejected(task: Any, runtime: Any, ctx: Context) -> None:
    await race_aborted(gate("rejected"), runtime.signal)

    async def change(tx: Any, _current: Any) -> Any:
        await tx.create_task(Hold, {"name": "staged"}, in_conversation(task["input"]["into"]))
        await tx.create_task(Hold, {"name": "child"}, owned_by(task["id"]))
        return {"status": "terminal", "outcome": COMPLETED}

    try:
        await runtime.commit(change, ctx)
    except Exception as error:
        rejections.append(error)
    await aborted(runtime.signal)


#: Holds until ``rejected`` opens, then tries one terminal commit that creates a task in conversation ``into`` and a
#: child of its own, which the Transaction rejects as the task is ending. Catches the rejection and holds.
Rejected = define_task(
    name="test.retention.rejected",
    version=1,
    initial=lambda _input: {"phase": "hold"},
    phases={"hold": _rejected},
    abort=_abort,
)


async def _wait(task: Any, runtime: Any, ctx: Context) -> None:
    on = task["input"]["on"]
    await runtime.commit(
        lambda _tx, _current: {"status": "waiting", "checkpoint": {"phase": "done"}, "on": on, "policy": "allSettled"},
        ctx,
    )


async def _done(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(_terminal(COMPLETED), ctx)


#: Waits on the tasks in its input, then completes.
Waiter = define_task(
    name="test.retention.waiter",
    version=1,
    initial=lambda _input: {"phase": "wait"},
    phases={"wait": _wait, "done": _done},
    abort=_abort,
)


def sizes(harness: Any) -> dict[str, int]:
    return dict(harness.scheduler_index_sizes)


class CountReads(StorageSpy):
    """Counts the task and conversation reads a storage serves; can hold reads of one conversation."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.tasks = 0
        self.conversations = 0
        self.hold: Any = None

    async def task(self, id: Any, context: Context) -> Any:
        self.tasks += 1
        return await self.inner.task(id, context)

    async def conversation(self, id: Any, context: Context) -> Any:
        self.conversations += 1
        if self.hold is not None:
            await self.hold(id)
        return await self.inner.conversation(id, context)


class Opened(NamedTuple):
    harness: Any
    registry: Any
    reports: list[BaseException]
    root: Any


async def open(tmp_path: Path, storage: Any = None) -> Opened:
    if storage is None:
        storage = await open_sqlite_storage(tmp_path / "session.sqlite")
    opened = await open_tasks(storage, [Hold, Spawner, Rejected])
    root = await opened.harness.root(context)
    opened.harness.resume()
    return Opened(opened.harness, opened.registry, opened.reports, root)


async def status(harness: Any, id: Any) -> Any:
    record = await harness.get_task(id, context)
    assert record is not None
    return record["state"]


async def is_status(harness: Any, id: Any, expected: str) -> bool:
    return (await status(harness, id))["status"] == expected


async def subagent_cycle(harness: Any, root: Any, parent: Any, n: int) -> Any:
    """Under live ``parent``: tool task T owning conversation C, a task in C; all of them end. Returns C."""

    async def create(tx: Any) -> tuple[Any, Any, Any]:
        tool = await tx.create_task(Hold, {"name": f"tool{n}"}, owned_by(parent))
        child = await tx.create_conversation(owned_by(tool))
        inner = await tx.create_task(Hold, {"name": f"inner{n}"}, in_conversation(child["id"]))
        return tool, inner, child["id"]

    tool, inner, child = await root.commit(create, context)
    resolve(gate(f"inner{n}"))
    resolve(gate(f"tool{n}"))
    await harness.wait_for_task(inner, context)
    await harness.wait_for_task(tool, context)
    return child


async def set_task(harness: Any, id: Any, **changes: Any) -> None:
    async def change(tx: Any) -> None:
        record = await tx.task(id)
        tx.set_task({**record, **changes})

    await harness.commit(change, context)


async def test_forgets_ended_subagent_chains_once_their_work_ends(tmp_path: Path) -> None:
    harness, _registry, _reports, root = await open(tmp_path)
    parent = await root.commit(lambda tx: tx.create_task(Hold, {"name": "parent"}, CONVERSATION), context)
    await subagent_cycle(harness, root, parent, 0)
    baseline = sizes(harness)
    for n in range(1, 201):
        await subagent_cycle(harness, root, parent, n)
    assert sizes(harness) == baseline
    resolve(gate("parent"))
    await harness.wait_for_task(parent, context)
    await harness.close(context)


async def _p_b_a_c(tx: Any) -> dict[str, Any]:
    p = await tx.create_task(Hold, {"name": "p"}, CONVERSATION)
    b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
    a = await tx.create_task(Hold, {"name": "a"}, owned_by(b))
    c = await tx.create_conversation(owned_by(a))
    return {"p": p, "b": b, "a": a, "c": c["id"]}


async def test_holds_a_live_owner_for_work_later_created_in_a_forgotten_conversation_loading_its_chain_again(
    tmp_path: Path,
) -> None:
    storage = CountReads(await open_sqlite_storage(tmp_path / "session.sqlite"))
    harness, _registry, _reports, root = await open(tmp_path, storage)
    # Live P -> ended B -> ended A -> conversation C, empty: forgotten.
    tree = await root.commit(_p_b_a_c, context)
    resolve(gate("a"))
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    before = sizes(harness)
    loaded = storage.tasks + storage.conversations
    c = await harness.conversation(tree["c"], context)
    d = await c.commit(lambda tx: tx.create_task(Hold, {"name": "d"}, CONVERSATION), context)
    resolve(gate("p"))
    await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    assert storage.tasks + storage.conversations > loaded
    resolve(gate("d"))
    assert (await harness.wait_for_task(d, context))["state"]["outcome"]["status"] == "completed"
    assert (await harness.wait_for_task(tree["p"], context))["state"]["outcome"]["status"] == "completed"
    assert sizes(harness)["settled"] <= before["settled"]
    await harness.close(context)


async def test_holds_an_owner_for_work_its_own_terminal_commit_creates_below_a_forgotten_chain(tmp_path: Path) -> None:
    harness, _registry, _reports, root = await open(tmp_path)

    # Live P (a spawner) -> ended B -> ended A -> conversation C, empty: forgotten. P's terminal commit creates D in C.
    async def create(tx: Any) -> dict[str, Any]:
        p = await tx.create_task(Spawner, {"name": "p", "into": 0}, CONVERSATION)
        b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
        a = await tx.create_task(Hold, {"name": "a"}, owned_by(b))
        c = await tx.create_conversation(owned_by(a))
        return {"p": p, "b": b, "c": c["id"]}

    tree = await root.commit(create, context)
    # Point the spawner at C, now that its ID is known.
    await set_task(harness, tree["p"], input={"name": "p", "into": tree["c"]})
    resolve(gate("a"))
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    resolve(gate("p"))

    async def not_running() -> bool:
        return not await is_status(harness, tree["p"], "running")

    await wait_for(not_running)
    assert (await status(harness, tree["p"]))["status"] == "completing"
    resolve(gate("inside.p"))
    assert (await harness.wait_for_task(tree["p"], context))["state"]["outcome"]["status"] == "completed"
    await harness.close(context)


async def _p_b_c_keep(tx: Any) -> dict[str, Any]:
    p = await tx.create_task(Hold, {"name": "p"}, CONVERSATION)
    b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
    c = await tx.create_conversation(owned_by(b))
    # A live child keeps P holding its abort until the input is withdrawn and the child ends.
    keep = await tx.create_task(Hold, {"name": "keep"}, owned_by(p))
    return {"p": p, "b": b, "c": c["id"], "keep": keep}


async def make_busy(conversation: Any, conversation_id: Any, task: Any) -> None:
    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["run"] = {"taskId": task, "inputs": []}

    await conversation.commit(change, context)


async def submission_status(harness: Any, id: Any) -> Any:
    return await (await harness.submission(id, context)).status(context)


async def test_withdraws_an_input_queued_in_a_forgotten_conversation_below_an_abort_marked_owner(
    tmp_path: Path,
) -> None:
    harness, _registry, _reports, root = await open(tmp_path)
    # Live P -> ended B -> conversation C with no task; C gets a queued input while P is cancelling.
    tree = await root.commit(_p_b_c_keep, context)
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    c = await harness.conversation(tree["c"], context)
    # Busy, so the input queues instead of starting a run.
    await make_busy(c, tree["c"], tree["keep"])
    queued = (await c.submit({"type": "input", "content": "later"}, context)).id
    await harness.abort_task(tree["p"], context)

    async def unanswered() -> bool:
        return (await submission_status(harness, queued))["status"] == "unanswered"

    await wait_for(unanswered)
    assert match_object(await submission_status(harness, queued), {"reason": "aborted"})
    await harness.wait_for_task(tree["p"], context)
    await harness.close(context)


async def test_aborts_work_created_in_a_forgotten_conversation_below_an_abort_marked_owner_before_it_runs(
    tmp_path: Path,
) -> None:
    harness, _registry, _reports, root = await open(tmp_path)
    tree = await root.commit(_p_b_c_keep, context)
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    # P cancels; its live child keeps it holding.
    await set_task(harness, tree["p"], abortRequested=True)

    async def keep_marked() -> bool:
        return bool(((await harness.get_task(tree["keep"], context)) or {}).get("abortRequested"))

    await wait_for(keep_marked)
    c = await harness.conversation(tree["c"], context)
    late = await c.commit(lambda tx: tx.create_task(Hold, {"name": "late"}, CONVERSATION), context)
    assert (await harness.wait_for_task(late, context))["state"]["outcome"]["status"] == "aborted"
    assert runs.get("late", 0) == 0
    await harness.close(context)


async def test_keeps_a_scope_busy_while_the_chain_of_new_work_below_a_forgotten_conversation_loads(
    tmp_path: Path,
) -> None:
    storage = CountReads(await open_sqlite_storage(tmp_path / "session.sqlite"))
    harness, _registry, _reports, root = await open(tmp_path, storage)

    async def create(tx: Any) -> dict[str, Any]:
        b = await tx.create_task(Hold, {"name": "b"}, CONVERSATION)
        c = await tx.create_conversation(owned_by(b))
        return {"b": b, "c": c["id"]}

    tree = await root.commit(create, context)
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    # Hold the chain load's read of C: the second read once holding, after the creating commit's own.
    held = deferred()
    reached = deferred()
    reads_of_c = [0]

    async def hold(id: Any) -> None:
        if id == tree["c"]:
            reads_of_c[0] += 1
            if reads_of_c[0] == 2:
                resolve(reached)
                await held

    c = await harness.conversation(tree["c"], context)
    storage.hold = hold
    await c.commit(lambda tx: tx.create_task(Hold, {"name": "d"}, CONVERSATION), context)
    await reached
    assert await settled(harness.wait_for_idle(context)) is False
    resolve(held)
    resolve(gate("d"))
    await harness.wait_for_idle(context)
    await harness.close(context)


async def test_keeps_a_chain_when_one_commit_ends_its_last_task_and_creates_new_work_below_it(tmp_path: Path) -> None:
    harness, _registry, _reports, root = await open(tmp_path)

    # Live P -> ended B -> conversation C with background task `old`, which keeps the chain known without holding B.
    async def create(tx: Any) -> dict[str, Any]:
        p = await tx.create_task(Hold, {"name": "p"}, CONVERSATION)
        b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
        c = await tx.create_conversation(owned_by(b))
        old = await tx.create_task(Hold, {"name": "old"}, in_conversation(c["id"], background=True))
        return {"p": p, "b": b, "c": c["id"], "old": old}

    tree = await root.commit(create, context)
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)

    # One commit: the old task ends, which leaves nothing live below B, and a replacement starts in C.
    async def replace(tx: Any) -> Any:
        old = await tx.task(tree["old"])
        id = await tx.create_task(Hold, {"name": "new"}, in_conversation(tree["c"]))
        tx.set_task({**old, "state": {"status": "terminal", "outcome": COMPLETED}})
        return id

    replacement = await harness.commit(replace, context)
    # The chain stayed known through the publication, so P holds for the replacement.
    resolve(gate("p"))
    await wait_for(lambda: is_status(harness, tree["p"], "completing"))
    resolve(gate("new"))
    await harness.wait_for_task(replacement, context)
    assert (await harness.wait_for_task(tree["p"], context))["state"]["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_drops_what_a_rejected_commits_chain_load_read_without_a_later_publication(tmp_path: Path) -> None:
    harness, _registry, _reports, root = await open(tmp_path)

    # Live P owns ended B, which owns conversation C: forgotten. P's terminal commit stages work in C, which loads
    # C's chain, and a child of P, which the Transaction rejects, as P is ending.
    async def create(tx: Any) -> dict[str, Any]:
        p = await tx.create_task(Rejected, {"into": 0}, CONVERSATION)
        b = await tx.create_task(Hold, {"name": "b"}, owned_by(p))
        c = await tx.create_conversation(owned_by(b))
        return {"p": p, "b": b, "c": c["id"]}

    tree = await root.commit(create, context)
    await set_task(harness, tree["p"], input={"into": tree["c"]})
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    baseline = sizes(harness)
    resolve(gate("rejected"))
    await wait_for(lambda: len(rejections) > 0)
    # The rejected commit published nothing; the sweep job still cleans up what its chain load read.
    await wait_for(lambda: sizes(harness)["dropQueue"] == 0)
    assert sizes(harness) == baseline
    await harness.close(context)


async def test_drops_what_a_read_loaded_after_reopen_without_a_later_publication(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open(tmp_path, await open_sqlite_storage(path))

    # A live task D in conversation C, owned by ended B: after reopen D's chain is not loaded.
    async def create(tx: Any) -> dict[str, Any]:
        b = await tx.create_task(Hold, {"name": "b"}, CONVERSATION)
        c = await tx.create_conversation(owned_by(b))
        d = await tx.create_task(Hold, {"name": "d"}, in_conversation(c["id"]))
        return {"b": b, "d": d}

    tree = await opened.root.commit(create, context)
    resolve(gate("b"))
    first = opened.harness
    await wait_for(lambda: is_status(first, tree["b"], "completing"))
    await first.close(context)
    gates.clear()
    opened = await open(tmp_path, await open_sqlite_storage(path))
    harness = opened.harness
    await harness.inspect(context)
    # D is still live, so its chain stays; once D ends, everything below the root goes, published or not.
    assert sizes(harness)["unloaded"] == 0
    resolve(gate("d"))
    await harness.wait_for_task(tree["d"], context)
    await harness.wait_for_task(tree["b"], context)
    await wait_for(lambda: sizes(harness)["dropQueue"] == 0)
    assert match_object(sizes(harness), {"settled": 0, "below": 0, "unloaded": 0})
    await harness.close(context)


async def test_withdraws_an_input_queued_two_ended_levels_below_an_owner_whose_intent_appears_later(
    tmp_path: Path,
) -> None:
    harness, _registry, _reports, root = await open(tmp_path)

    # Live P -> ended B -> ended A -> conversation C with no task, and a live child keeping P holding.
    async def create(tx: Any) -> dict[str, Any]:
        tree = await _p_b_a_c(tx)
        tree["keep"] = await tx.create_task(Hold, {"name": "keep"}, owned_by(tree["p"]))
        return tree

    tree = await root.commit(create, context)
    resolve(gate("a"))
    resolve(gate("b"))
    await harness.wait_for_task(tree["b"], context)
    c = await harness.conversation(tree["c"], context)
    await make_busy(c, tree["c"], tree["keep"])
    # Queued while nothing cancels: no withdrawal scan.
    queued = (await c.submit({"type": "input", "content": "later"}, context)).id
    await wait_for(lambda: sizes(harness)["dropQueue"] == 0)
    # P fails while its child runs: it holds the failed outcome, which is cancellation intent.
    await set_task(
        harness, tree["p"], state={"status": "completing", "outcome": {"status": "failed", "error": {"message": "x"}}}
    )

    async def unanswered() -> bool:
        return (await submission_status(harness, queued))["status"] == "unanswered"

    await wait_for(unanswered)
    assert match_object(await submission_status(harness, queued), {"reason": "aborted"})
    await harness.close(context)


async def test_forgets_a_deep_chain_of_ended_tasks_without_recursion(tmp_path: Path) -> None:
    harness, _registry, _reports, root = await open(tmp_path)
    depth = 2 * sys.getrecursionlimit()
    baseline = sizes(harness)

    async def create(tx: Any) -> tuple[list[Any], Any]:
        owner = await tx.create_task(Hold, {"name": "link0"}, CONVERSATION)
        links = [owner]
        for n in range(1, depth):
            owner = await tx.create_task(Hold, {"name": f"link{n}"}, owned_by(owner))
            links.append(owner)
        leaf = await tx.create_task(Hold, {"name": "leaf"}, owned_by(owner))
        return links, leaf

    links, leaf = await root.commit(create, context)
    for n in range(depth):
        resolve(gate(f"link{n}"))
    resolve(gate("leaf"))
    await harness.wait_for_task(leaf, context)
    await harness.wait_for_task(links[0], context)
    await harness.wait_for_idle(context)
    assert sizes(harness) == baseline
    await harness.close(context)


async def test_keeps_no_more_after_all_work_ended_than_a_fresh_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open(tmp_path, await open_sqlite_storage(path))
    parent = await opened.root.commit(lambda tx: tx.create_task(Hold, {"name": "parent"}, CONVERSATION), context)
    for n in range(20):
        await subagent_cycle(opened.harness, opened.root, parent, n)
    resolve(gate("parent"))
    await opened.harness.wait_for_task(parent, context)
    await opened.harness.wait_for_idle(context)
    ended = sizes(opened.harness)
    await opened.harness.close(context)
    opened = await open(tmp_path, await open_sqlite_storage(path))
    await opened.harness.wait_for_idle(context)
    assert ended == sizes(opened.harness)
    await opened.harness.close(context)


class CountQueuedScans(StorageSpy):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.scans = 0

    async def scan_submissions(self, query: Any, limit: int, cursor: Any, context: Context) -> Any:
        if query.get("status") == "queued":
            self.scans += 1
        return await self.inner.scan_submissions(query, limit, cursor, context)


async def test_does_not_scan_queued_submissions_for_an_input_when_nothing_cancels(tmp_path: Path) -> None:
    storage = CountQueuedScans(await open_sqlite_storage(tmp_path / "session.sqlite"))
    harness, _registry, _reports, root = await open(tmp_path, storage)
    busy = await root.commit(lambda tx: tx.create_task(Hold, {"name": "busy"}, CONVERSATION), context)
    await make_busy(root, root.id, busy)
    before = storage.scans
    await root.submit({"type": "input", "content": "queued"}, context)
    # Let any reconcile it would cause run.
    await asyncio.sleep(0.02)
    assert storage.scans == before
    resolve(gate("busy"))
    await harness.close(context)


async def test_lets_a_task_wait_on_a_forgotten_ended_owner(tmp_path: Path) -> None:
    """A wait on an ended task closes no cycle (#10411), also when the ended task owns the waiter's conversation and
    the scheduler forgot it: the wait resumes at once."""
    opened = await open_tasks(await open_sqlite_storage(tmp_path / "session.sqlite"), [Hold, Waiter])
    root = await opened.harness.root(context)
    opened.harness.resume()

    async def create(tx: Any) -> dict[str, Any]:
        b = await tx.create_task(Hold, {"name": "b"}, CONVERSATION)
        c = await tx.create_conversation(owned_by(b))
        return {"b": b, "c": c["id"]}

    tree = await root.commit(create, context)
    resolve(gate("b"))
    await opened.harness.wait_for_task(tree["b"], context)
    c = await opened.harness.conversation(tree["c"], context)
    waiter = await c.commit(lambda tx: tx.create_task(Waiter, {"on": [tree["b"]]}, CONVERSATION), context)
    outcome = (await opened.harness.wait_for_task(waiter, context))["state"]["outcome"]
    assert match_object(outcome, {"status": "completed"})
    await opened.harness.close(context)
