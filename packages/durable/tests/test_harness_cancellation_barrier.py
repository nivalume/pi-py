"""Port of ``test/harness-cancellation-barrier.test.ts``."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from durable_harness_support import aborted, eventually, open_tasks
from durable_session_support import context, deferred, flush, resolve

from pi_chord.context import Context
from pi_durable import MemoryStorage, create_session, define_task
from pi_durable.types import Seq, StorageWrite, Task


async def _work(_task: Any, runtime: Any, _ctx: Context) -> None:
    await aborted(runtime.signal)


async def _abort(_task: Any, runtime: Any, abort_context: Context) -> None:
    await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, abort_context)


#: Runs until aborted; its abort handler ends it ``aborted``.
Dormant = define_task(
    name="test.dormant", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": _work}, abort=_abort
)


class HoldMarks(MemoryStorage):
    """Holds every commit that abort-marks ``target``, while set, until released."""

    def __init__(self) -> None:
        super().__init__()
        self.target: Any = None
        self.held = deferred()
        self.release = deferred()

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        if any(
            write["type"] == "task"
            and write["value"]["id"] == self.target  # type: ignore[typeddict-item]
            and write["value"].get("abortRequested")  # type: ignore[typeddict-item]
            for write in writes
        ):
            resolve(self.held)
            await self.release
        return await super().commit(writes, context)


async def test_does_not_start_its_next_phase_while_the_cascade_that_marks_it_has_not_committed() -> None:
    storage = HoldMarks()
    entered = deferred()
    advance = deferred()
    ran_next = [False]

    async def first(_task: Any, runtime: Any, phase_context: Context) -> None:
        resolve(entered)
        await advance
        await runtime.commit(
            lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "next"}}, phase_context
        )

    async def next_phase(_task: Any, runtime: Any, _ctx: Context) -> None:
        ran_next[0] = True
        await aborted(runtime.signal)

    Child = define_task(
        name="test.child",
        version=1,
        initial=lambda _input: {"phase": "first"},
        phases={"first": first, "next": next_phase},
        abort=_abort,
    )
    harness = (await open_tasks(storage, [Dormant, Child])).harness
    root = await harness.root(context)
    parent = await root.commit(
        lambda tx: tx.create_task(Dormant, None, {"ownership": {"kind": "conversation"}}), context
    )
    child = await root.commit(
        lambda tx: tx.create_task(Child, None, {"ownership": {"kind": "task", "taskId": parent}}), context
    )
    storage.target = child
    try:
        harness.resume()
        await entered
        aborting = harness.abort_task(parent, context)
        await storage.held
        # The parent's request is durable, the cascade's mark of the child is not: its phase must not advance.
        resolve(advance)
        for _ in range(5):
            await flush()
        assert ran_next[0] is False
        resolve(storage.release)
        assert (await harness.wait_for_task(child, context))["state"]["outcome"]["status"] == "aborted"
        await aborting
        assert ran_next[0] is False
    finally:
        resolve(advance)
        resolve(storage.release)
        await harness.close(context)


async def _ownerless_conversation(session: Any) -> Any:
    return await session.commit(lambda tx: tx.create_conversation({"ownership": {"kind": "ownerless"}}), context)


async def test_lets_a_fail_fast_request_in_the_same_pass_override_a_restart_mark() -> None:
    storage = MemoryStorage()
    session = create_session(storage)
    conversation = await _ownerless_conversation(session)
    parent = await session.commit(
        lambda tx: tx.create_task(
            Dormant, None, {"conversationId": conversation["id"], "ownership": {"kind": "conversation"}}
        ),
        context,
    )
    failed = await session.commit(
        lambda tx: tx.create_task(Dormant, None, {"ownership": {"kind": "task", "taskId": parent}}), context
    )
    child = await session.commit(
        lambda tx: tx.create_task(Dormant, None, {"ownership": {"kind": "task", "taskId": parent}}), context
    )
    # A restart-abandoned owner that waits failFast on a failed child and a live one.
    owner: Any = await storage.task(parent, context)
    failed_record: Any = await storage.task(failed, context)
    waiting = {"status": "waiting", "checkpoint": {"phase": "work"}, "on": [failed, child], "policy": "failFast"}
    failure = {"status": "failed", "error": {"message": "failed"}}
    await storage.commit(
        [
            {
                "type": "task",
                "value": {**owner, "abortRequested": True, "abortReason": "restart", "state": waiting},
            },
            {"type": "task", "value": {**failed_record, "state": {"status": "terminal", "outcome": failure}}},
        ],
        context,
    )
    # The child's definition is missing: a restart mark would keep it waiting, a request orphans it.
    harness = (await open_tasks(storage, [])).harness
    try:
        harness.resume()

        async def child_terminal() -> bool:
            return ((await harness.get_task(child, context)) or {}).get("state", {}).get("status") == "terminal"

        await eventually(child_terminal)
        record: Any = await harness.get_task(child, context)
        assert record["state"]["outcome"]["status"] == "orphaned"
        assert record["state"]["outcome"]["reason"] == "missing_task"
    finally:
        await harness.close(context)


async def test_upgrades_restart_marks_down_the_tree_when_an_abort_is_requested_above_them() -> None:
    storage = MemoryStorage()
    session = create_session(storage)
    conversation = await _ownerless_conversation(session)
    top = await session.commit(
        lambda tx: tx.create_task(
            Dormant, None, {"conversationId": conversation["id"], "ownership": {"kind": "conversation"}}
        ),
        context,
    )
    middle = await session.commit(
        lambda tx: tx.create_task(
            Dormant, None, {"ownership": {"kind": "task", "taskId": top}, "abandonOnRestart": True}
        ),
        context,
    )
    Missing = Task(dataclasses.replace(Dormant.definition, name="test.missing"))
    leaf = await session.commit(
        lambda tx: tx.create_task(Missing, None, {"ownership": {"kind": "task", "taskId": middle}}), context
    )
    # Only Dormant is registered: the leaf waits for its definition under the restart cascade.
    harness = (await open_tasks(storage, [Dormant])).harness
    try:
        harness.resume()

        async def leaf_restart_marked() -> bool:
            return ((await harness.get_task(leaf, context)) or {}).get("abortReason") == "restart"

        async def leaf_blocked() -> bool:
            inspected = next(
                (task for task in (await harness.inspect(context))["tasks"] if task["record"]["id"] == leaf), None
            )
            return inspected is not None and inspected["state"]["kind"] == "blocked"

        await eventually(leaf_restart_marked)
        await eventually(leaf_blocked)
        record: Any = await harness.get_task(leaf, context)
        assert record["state"]["status"] != "terminal"
        # A request on the top task reaches the leaf two levels down, which is then orphaned.
        await harness.abort_task(top, context)
        for id, status in [(leaf, "orphaned"), (middle, "aborted"), (top, "aborted")]:
            assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == status
    finally:
        await harness.close(context)


async def test_marks_a_flagged_task_that_holds_its_outcome_aborting_its_owned_work_and_keeps_the_outcome() -> None:
    storage = MemoryStorage()
    session = create_session(storage)
    conversation = await _ownerless_conversation(session)
    holder = await session.commit(
        lambda tx: tx.create_task(
            Dormant,
            None,
            {
                "conversationId": conversation["id"],
                "ownership": {"kind": "conversation"},
                "abandonOnRestart": True,
            },
        ),
        context,
    )
    child = await session.commit(
        lambda tx: tx.create_task(Dormant, None, {"ownership": {"kind": "task", "taskId": holder}}), context
    )
    record: Any = await storage.task(holder, context)
    completing = {"status": "completing", "outcome": {"status": "completed", "result": None}}
    await storage.commit([{"type": "task", "value": {**record, "state": completing}}], context)
    harness = (await open_tasks(storage, [Dormant])).harness
    try:
        harness.resume()
        assert (await harness.wait_for_task(child, context))["state"]["outcome"]["status"] == "aborted"
        settled = await harness.wait_for_task(holder, context)
        assert settled["abortReason"] == "restart"
        assert settled["state"]["outcome"]["status"] == "completed"
    finally:
        await harness.close(context)
