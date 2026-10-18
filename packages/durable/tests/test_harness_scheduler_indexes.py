"""Port of ``test/harness-scheduler-indexes.test.ts``.

The scheduler's in-memory indexes: cases where a wrong index changes behaviour, each against what the committed
records imply (spec §5.4, §5.5).

Adaptation: upstream rebinds ``storage.commit``/``storage.conversation`` on the SQLite storage instance; the Python
storage is slotted, so a :class:`StorageSpy` subclass intercepts them.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import StorageSpy, open_tasks, race_aborted, settled, wait_for
from durable_session_support import context, deferred, resolve

from pi_chord.context import Context
from pi_durable import create_session, define_task
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Seq, StorageWrite

gates: dict[str, asyncio.Future[Any]] = {}


def gate(name: str) -> asyncio.Future[Any]:
    found = gates.get(name)
    if found is None:
        found = deferred()
        gates[name] = found
    return found


def _complete(runtime: Any, ctx: Context) -> Any:
    return runtime.commit(
        lambda _tx, _current: {"status": "terminal", "outcome": {"status": "completed", "result": None}}, ctx
    )


async def _abort(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)


async def _hold(task: Any, runtime: Any, ctx: Context) -> None:
    await race_aborted(gate(task["input"]["name"]), runtime.signal)
    await _complete(runtime, ctx)


#: Holds until its gate opens, then completes; its abort handler ends it ``aborted``.
Hold = define_task(
    name="test.index.hold", version=1, initial=lambda _input: {"phase": "hold"}, phases={"hold": _hold}, abort=_abort
)

#: What ``owned_tasks()`` returned to each run of a ``Probe``, by its name, and the runtime of its last run.
owned: dict[str, list[Any]] = {}
runtimes: dict[str, Any] = {}


async def _look(task: Any, runtime: Any, ctx: Context) -> None:
    name = task["input"]["name"]
    runtimes[name] = runtime
    await race_aborted(gate(f"look.{name}"), runtime.signal)
    owned[name] = [record["id"] for record in await runtime.owned_tasks(ctx)]
    await race_aborted(gate(name), runtime.signal)
    await _complete(runtime, ctx)


#: Records its owned tasks once its ``look`` gate opens, then holds until its own gate opens.
Probe = define_task(
    name="test.index.probe", version=1, initial=lambda _input: {"phase": "look"}, phases={"look": _look}, abort=_abort
)


@pytest.fixture(autouse=True)
def _clear() -> Iterator[None]:
    yield
    gates.clear()
    owned.clear()
    runtimes.clear()


async def state(harness: Any, id: Any) -> Any:
    record = await harness.get_task(id, context)
    assert record is not None
    return record


def owned_by(task_id: Any) -> Any:
    return {"ownership": {"kind": "task", "taskId": task_id}}


CONVERSATION: Any = {"ownership": {"kind": "conversation"}}


class RecordLeafMarks(StorageSpy):
    def __init__(self, inner: Any, leaf: Any) -> None:
        super().__init__(inner)
        self.leaf = leaf
        self.reasons: list[Any] = []

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        for write in writes:
            value: Any = write.get("value")
            if write["type"] == "task" and value["id"] == self.leaf and value.get("abortRequested"):
                self.reasons.append(value.get("abortReason"))
        return await self.inner.commit(writes, context)


async def test_takes_a_cascades_reason_from_the_nearest_cancelling_owner(tmp_path: Path) -> None:
    # top (abort requested) -> middle (restart-marked) -> leaf: the leaf gets middle's `restart` first, as the
    # nearest cancelling owner, and the request only once middle carries it.
    path = tmp_path / "session.sqlite"
    session = create_session(await open_sqlite_storage(path))

    async def create(tx: Any) -> dict[str, Any]:
        conversation = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        top = await tx.create_task(
            Hold, {"name": "top"}, {"conversationId": conversation["id"], "ownership": {"kind": "conversation"}}
        )
        middle = await tx.create_task(Hold, {"name": "middle"}, owned_by(top))
        leaf = await tx.create_task(Hold, {"name": "leaf"}, owned_by(middle))
        return {"top": top, "middle": middle, "leaf": leaf}

    ids = await session.commit(create, context)

    async def mark(tx: Any) -> None:
        top = await tx.task(ids["top"])
        middle = await tx.task(ids["middle"])
        tx.set_task({**top, "abortRequested": True})
        tx.set_task({**middle, "abortRequested": True, "abortReason": "restart"})

    await session.commit(mark, context)
    await session.close(context)
    # Record every mark of the leaf, in commit order.
    storage = RecordLeafMarks(await open_sqlite_storage(path), ids["leaf"])
    harness = (await open_tasks(storage, [Hold])).harness
    harness.resume()
    for id in [ids["leaf"], ids["middle"], ids["top"]]:
        await harness.wait_for_task(id, context)
    assert storage.reasons[0] == "restart"
    await harness.close(context)


class RecordTerminals(StorageSpy):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        #: The commit, by count, that wrote each terminal record.
        self.ended_in: dict[Any, int] = {}
        self.commits = 0

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        self.commits += 1
        for write in writes:
            value: Any = write.get("value")
            if write["type"] == "task" and value["state"]["status"] == "terminal":
                self.ended_in[value["id"]] = self.commits
        return await self.inner.commit(writes, context)


async def test_finalizes_a_chain_of_held_owners_in_the_commit_that_frees_the_last_one(tmp_path: Path) -> None:
    storage = RecordTerminals(await open_sqlite_storage(tmp_path / "session.sqlite"))
    harness = (await open_tasks(storage, [Hold])).harness
    root = await harness.root(context)
    harness.resume()

    async def create(tx: Any) -> tuple[Any, Any, Any]:
        top = await tx.create_task(Hold, {"name": "top"}, CONVERSATION)
        middle = await tx.create_task(Hold, {"name": "middle"}, owned_by(top))
        leaf = await tx.create_task(Hold, {"name": "leaf"}, owned_by(middle))
        return top, middle, leaf

    top, middle, leaf = await root.commit(create, context)
    resolve(gate("top"))
    resolve(gate("middle"))

    async def completing(id: Any) -> bool:
        return (await state(harness, id))["state"]["status"] == "completing"

    await wait_for(lambda: completing(top))
    await wait_for(lambda: completing(middle))
    resolve(gate("leaf"))
    assert (await harness.wait_for_task(top, context))["state"]["outcome"]["status"] == "completed"
    # Both owners end in one finalize commit, after the leaf's.
    assert storage.ended_in[middle] == storage.ended_in[top]
    assert storage.ended_in[middle] > storage.ended_in[leaf]
    await harness.close(context)


class HoldConversationRead(StorageSpy):
    def __init__(self, inner: Any, conversation_id: Any) -> None:
        super().__init__(inner)
        self.conversation_id = conversation_id
        self.held = deferred()

    async def conversation(self, id: Any, context: Context) -> Any:
        if id == self.conversation_id:
            await self.held
        return await self.inner.conversation(id, context)


async def test_keeps_a_scope_busy_after_reopen_for_work_below_an_ended_task_whose_chain_is_not_loaded_yet(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_tasks(await open_sqlite_storage(path), [Hold])
    root = await opened.harness.root(context)
    opened.harness.resume()

    # root conversation -> owner (ends) -> conversation C -> inner (created after owner ended), as a subagent call's.
    async def create(tx: Any) -> tuple[Any, Any]:
        owner = await tx.create_task(Hold, {"name": "owner"}, CONVERSATION)
        child = await tx.create_conversation(owned_by(owner))
        return owner, child["id"]

    owner, child = await root.commit(create, context)
    resolve(gate("owner"))
    await opened.harness.wait_for_task(owner, context)
    conversation = await opened.harness.conversation(child, context)
    assert conversation is not None
    inner = await conversation.commit(lambda tx: tx.create_task(Hold, {"name": "inner"}, CONVERSATION), context)
    await opened.harness.close(context)
    gates.clear()
    # Hold the read of C's record, so the chain above `inner` stays unloaded while idle is decided.
    storage = HoldConversationRead(await open_sqlite_storage(path), child)
    opened = await open_tasks(storage, [Hold])
    # A task whose chain is not loaded counts as inside every scope. (The held read blocks the Session line, so the
    # root handle comes after.)
    assert await settled(opened.harness.wait_for_idle(context)) is False
    resolve(storage.held)
    root = await opened.harness.root(context)
    opened.harness.resume()
    resolve(gate("inner"))
    await opened.harness.wait_for_task(inner, context)
    await root.wait_for_idle(context)
    await opened.harness.close(context)


# ─── TaskRuntime.owned_tasks ────────────────────────────────────────────────


async def test_lists_the_live_tasks_a_task_owns_directly_in_id_order_also_after_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_tasks(await open_sqlite_storage(path), [Hold, Probe])
    root = await opened.harness.root(context)
    opened.harness.resume()

    # The probe owns: two held tasks, one that ends, a grandchild below the first, and a conversation with a task.
    async def create(tx: Any) -> dict[str, Any]:
        probe = await tx.create_task(Probe, {"name": "probe"}, CONVERSATION)
        first = await tx.create_task(Hold, {"name": "first"}, owned_by(probe))
        ends = await tx.create_task(Hold, {"name": "ends"}, owned_by(probe))
        second = await tx.create_task(Hold, {"name": "second"}, owned_by(probe))
        await tx.create_task(Hold, {"name": "grandchild"}, owned_by(first))
        child = await tx.create_conversation(owned_by(probe))
        await tx.create_task(
            Hold, {"name": "inside"}, {"ownership": {"kind": "conversation"}, "conversationId": child["id"]}
        )
        return {"probe": probe, "first": first, "ends": ends, "second": second}

    ids = await root.commit(create, context)
    resolve(gate("ends"))
    await opened.harness.wait_for_task(ids["ends"], context)
    # The first ends too, but holds its outcome as `completing` while its grandchild runs: still live, so listed.
    resolve(gate("first"))
    harness = opened.harness

    async def first_completing() -> bool:
        return (await state(harness, ids["first"]))["state"]["status"] == "completing"

    await wait_for(first_completing)
    resolve(gate("look.probe"))
    await wait_for(lambda: "probe" in owned)
    assert owned["probe"] == [ids["first"], ids["second"]]
    await opened.harness.close(context)

    # After a reopen the rerun sees the same tasks, which an earlier invocation created.
    gates.clear()
    owned.clear()
    opened = await open_tasks(await open_sqlite_storage(path), [Hold, Probe])
    opened.harness.resume()
    resolve(gate("look.probe"))
    await wait_for(lambda: "probe" in owned)
    assert owned["probe"] == [ids["first"], ids["second"]]
    await opened.harness.close(context)


async def test_rejects_once_its_invocation_has_ended(tmp_path: Path) -> None:
    opened = await open_tasks(await open_sqlite_storage(tmp_path / "session.sqlite"), [Hold, Probe])
    root = await opened.harness.root(context)
    opened.harness.resume()
    probe = await root.commit(lambda tx: tx.create_task(Probe, {"name": "probe"}, CONVERSATION), context)
    await wait_for(lambda: "probe" in runtimes)
    runtime = runtimes["probe"]
    # Aborting joins the run invocation, which has ended by then.
    await opened.harness.abort_task(probe, context)
    with pytest.raises(Exception, match="invocation has ended"):
        await runtime.owned_tasks(context)
    await opened.harness.close(context)
