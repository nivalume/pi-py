"""Port of ``test/harness-task-graph.test.ts``.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import completed, eventually, match_object, open_tasks, race_aborted, wait_for
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, flush, resolve

from pi_ai import AbortController
from pi_chord.context import Context, with_abort_signal
from pi_chord.delta import apply_immutable
from pi_durable import MemoryStorage, define_task
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


async def _nothing(*_args: Any) -> None:
    return None


class Family:
    """A parent that creates a child task and a conversation it owns, waits for the child, creates a second child,
    and completes while that child is still live, so its outcome is held as ``completing``."""

    def __init__(self, child_gate: Awaitable[Any], late_gate: Awaitable[Any]) -> None:
        async def work(task: Any, runtime: Any, ctx: Context) -> None:
            await (late_gate if task["input"]["late"] else child_gate)
            await runtime.commit(lambda _tx, _current: completed(None), ctx)

        self.Child = Child = define_task(
            name="test.graph-child",
            version=1,
            initial=lambda _input: {"phase": "work"},
            phases={"work": work},
            abort=_nothing,
        )

        async def spawn(task: Any, runtime: Any, ctx: Context) -> None:
            ownership = {"kind": "task", "taskId": task["id"]}

            # Two conversations in a commit that leaves the parent's record unchanged.
            async def conversations(tx: Any, _current: Any) -> None:
                await tx.create_conversation({"ownership": ownership})
                await tx.create_conversation({"ownership": ownership})

            await runtime.commit(conversations, ctx)

            async def wait(tx: Any, _current: Any) -> Any:
                child = await tx.create_task(Child, {"late": False}, {"ownership": ownership})
                return {
                    "status": "waiting",
                    "checkpoint": {"phase": "join", "child": child},
                    "on": [child],
                    "policy": "allSettled",
                }

            await runtime.commit(wait, ctx)

        async def join(task: Any, runtime: Any, ctx: Context) -> None:
            async def change(tx: Any, _current: Any) -> Any:
                await tx.create_task(Child, {"late": True}, {"ownership": {"kind": "task", "taskId": task["id"]}})
                return {"status": "running", "checkpoint": {"phase": "finish"}}

            await runtime.commit(change, ctx)

        async def finish(_task: Any, runtime: Any, ctx: Context) -> None:
            await runtime.commit(lambda _tx, _current: completed(None), ctx)

        self.Parent = define_task(
            name="test.graph-parent",
            version=1,
            initial=lambda _input: {"phase": "spawn"},
            phases={"spawn": spawn, "join": join, "finish": finish},
            abort=_nothing,
        )


def statuses(graph: Any) -> dict[str, str]:
    """Node states by task kind, for compact assertions."""
    return {f"{node['kind']}#{node['id']}": node["state"]["status"] for node in graph["tasks"].values()}


async def test_follows_every_live_task_through_its_statuses_owner_edges_and_owned_conversations() -> None:
    child = deferred()
    late = deferred()
    family = Family(child, late)
    Parent, Child = family.Parent, family.Child
    harness = (await open_tasks(MemoryStorage(), [Parent, Child])).harness
    root = await harness.root(context)
    seen: list[Any] = []

    async def observe() -> Any:
        opened = await harness.task_graph(context)

        def listener(value: Any, _context: Context, delivery: Any) -> None:
            if delivery.kind == "update":
                seen.append(value)

        opened.subscribe(listener)
        return opened

    graph = [await observe()]
    assert graph[0].value == {"tasks": {}}

    async def rebuild() -> None:
        """The advanced value equals a fresh build from Storage once the last observer left."""
        advanced = graph[0].value
        graph[0].dispose()
        graph[0] = await observe()
        assert graph[0].value is not advanced
        assert graph[0].value == advanced

    parent = await root.commit(
        lambda tx: tx.create_task(Parent, None, {"ownership": {"kind": "conversation"}}), context
    )
    await flush()
    assert graph[0].value["tasks"][str(parent)] == {
        "id": parent,
        "kind": "test.graph-parent",
        "conversationId": root.id,
        "background": False,
        "abortRequested": False,
        "state": {"status": "pending", "phase": "spawn"},
        "conversations": [],
    }

    def child_id() -> str:
        node = next((node for node in graph[0].value["tasks"].values() if node["kind"] == "test.graph-child"), None)
        return "" if node is None else str(node["id"])

    harness.resume()
    await eventually(
        lambda: (
            len(graph[0].value["tasks"]) == 2
            and statuses(graph[0].value).get(f"test.graph-child#{child_id()}") == "running"
        )
    )
    first = int(child_id())
    parent_node = graph[0].value["tasks"][str(parent)]
    assert parent_node["state"] == {"status": "waiting", "phase": "join", "on": [first], "policy": "allSettled"}
    assert len(parent_node["conversations"]) == 2
    assert sorted(parent_node["conversations"]) == list(parent_node["conversations"])
    owned = parent_node["conversations"][0]
    assert match_object(graph[0].value["tasks"][str(first)], {"owner": parent, "conversationId": root.id})
    assert (await harness.conversation(owned, context)) is not None
    await rebuild()

    resolve(child)
    # The parent completes while the late child lives: its outcome is held.
    await eventually(
        lambda: (graph[0].value["tasks"].get(str(parent)) or {}).get("state", {}).get("status") == "completing"
    )
    assert graph[0].value["tasks"][str(parent)]["state"] == {"status": "completing", "outcome": "completed"}
    assert str(first) not in graph[0].value["tasks"]
    # Owned conversations stay listed while the owner lives.
    assert graph[0].value["tasks"][str(parent)]["conversations"] == parent_node["conversations"]
    await rebuild()

    resolve(late)
    await harness.wait_for_task(parent, context)
    await flush()
    assert graph[0].value == {"tasks": {}}
    # Every revision is one commit that changed a node; none repeats its predecessor.
    for index in range(1, len(seen)):
        assert seen[index] != seen[index - 1]
    # The commit that created the two conversations published one revision setting them.
    assert any(len((value["tasks"].get(str(parent)) or {"conversations": []})["conversations"]) == 2 for value in seen)
    graph[0].dispose()
    await harness.close(context)


async def test_builds_from_committed_tasks_shows_surviving_tasks_as_pending_after_reopen_and_marks_aborts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    gate = deferred()

    async def work(_task: Any, runtime: Any, ctx: Context) -> None:
        await race_aborted(gate, runtime.signal)
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(
            lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted", "reason": "test"}}, ctx
        )

    Work = define_task(
        name="test.graph-work", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": work}, abort=abort
    )
    first = await open_tasks(await open_sqlite_storage(path), [Work])
    first_root = await first.harness.root(context)

    async def create(tx: Any) -> tuple[Any, Any, Any]:
        foreground = await tx.create_task(Work, None, {"ownership": {"kind": "conversation"}})
        background = await tx.create_task(Work, None, {"ownership": {"kind": "conversation"}, "background": True})
        owned = await tx.create_conversation({"ownership": {"kind": "task", "taskId": foreground}})
        return foreground, background, owned["id"]

    foreground, background, owned = await first_root.commit(create, context)
    first.harness.resume()

    async def all_running() -> bool:
        return all(task["state"]["kind"] == "running" for task in (await first.harness.inspect(context))["tasks"])

    await wait_for(all_running)
    running = await first.harness.task_graph(context)
    assert statuses(running.value) == {
        f"test.graph-work#{foreground}": "running",
        f"test.graph-work#{background}": "running",
    }
    await first.harness.close(context)

    # Acquired after reopen: built from the committed records and owner edges; open reconciled running to pending.
    harness = (await open_tasks(await open_sqlite_storage(path), [Work])).harness
    watch = await harness.watch_task_graph(context)
    assert statuses(watch.value) == {
        f"test.graph-work#{foreground}": "pending",
        f"test.graph-work#{background}": "pending",
    }
    assert watch.value["tasks"][str(foreground)]["conversations"] == [owned]
    assert watch.value["tasks"][str(background)]["background"] is True

    # Exact frames: replaying their operations from the acquisition revision gives each delivered value.
    replica: list[Any] = [watch.value]
    frames: list[list[Any]] = []

    async def listener(value: Any, ops: Any, _context: Context) -> None:
        replica[0] = apply_immutable(replica[0], ops)
        assert replica[0] == value
        frames.append(list(ops))

    watch.start(listener)
    harness.resume()
    await wait_for(
        lambda: (watch.value["tasks"].get(str(background)) or {}).get("state", {}).get("status") == "running"
    )
    assert await harness.abort_task(background, context) == "marked"
    await harness.wait_for_task(background, context)
    await flush()

    def marks_abort(frame: list[Any]) -> bool:
        if len(frame) != 1:
            return False
        [op] = frame
        return (
            op[0] == "s"
            and tuple(op[1]) == ("tasks", str(background))
            and isinstance(op[2], dict)
            and op[2].get("abortRequested") is True
        )

    assert any(marks_abort(frame) for frame in frames)
    assert [(op[0], tuple(op[1])) for op in frames[-1]] == [("d", ("tasks", str(background)))]
    assert len(frames[-1][0]) == 2
    assert list(replica[0]["tasks"].keys()) == [str(foreground)]
    await watch.stop()
    resolve(gate)
    await harness.wait_for_task(foreground, context)
    await harness.close(context)


async def test_lists_owned_conversations_in_id_order_whatever_order_one_commit_creates_them_in() -> None:
    gate = deferred()
    created: list[int] = []

    async def spawn(task: Any, runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> None:
            ownership = {"kind": "task", "taskId": task["id"]}
            forked, fresh = await asyncio.gather(
                tx.fork_conversation(runtime.conversation_id, task["input"]["at"], {"ownership": ownership}),
                tx.create_conversation({"ownership": ownership}),
            )
            created.extend([forked["id"], fresh["id"]])

        await runtime.commit(change, ctx)
        await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Spawner = define_task(
        name="test.graph-spawner",
        version=1,
        initial=lambda _input: {"phase": "spawn"},
        phases={"spawn": spawn},
        abort=_nothing,
    )
    harness = (await open_tasks(MemoryStorage(), [Spawner])).harness
    root = await harness.root(context)
    graph = await harness.task_graph(context)

    async def create(tx: Any) -> Any:
        entry = await tx.append_entry(root.id, {"kind": "note"})
        return await tx.create_task(Spawner, {"at": entry["id"]}, {"ownership": {"kind": "conversation"}})

    id = await root.commit(create, context)
    harness.resume()
    await eventually(lambda: len((graph.value["tasks"].get(str(id)) or {"conversations": []})["conversations"]) == 2)
    advanced = graph.value
    assert advanced["tasks"][str(id)]["conversations"] == sorted(created)
    graph.dispose()
    rebuilt = await harness.task_graph(context)
    assert rebuilt.value == advanced
    rebuilt.dispose()
    resolve(gate)
    await harness.wait_for_task(id, context)
    await harness.close(context)


async def test_registers_nothing_for_an_acquisition_cancelled_while_it_waits_for_the_line() -> None:
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])
    harness = (await open_tasks(storage, [])).harness
    root = await harness.root(context)
    never = asyncio.get_running_loop().create_future()
    await root.commit(
        lambda tx: tx.create_task(Family(never, never).Child, {"late": False}, {"ownership": {"kind": "conversation"}}),
        context,
    )
    held = storage.hold_commits()

    async def block(tx: Any) -> None:
        await tx.append_entry(root.id, {"kind": "blocker"})

    blocking = harness.commit(block, context)
    await held.entered
    controller = AbortController()
    cancelled = harness.watch_task_graph(with_abort_signal(controller.signal, context))
    controller.abort(RuntimeError("cancelled"))
    held.release()
    await blocking
    with pytest.raises(Exception, match="cancelled"):
        await cancelled
    # No observer kept the mount: each new observer builds a new revision.
    first = await harness.task_graph(context)
    value = first.value
    first.dispose()
    second = await harness.task_graph(context)
    assert second.value is not value
    assert second.value == value
    second.dispose()
    await harness.close(context)


async def test_publishes_no_revision_for_a_commit_that_changes_no_node_and_shares_one_mount_between_observers() -> None:
    gate = deferred()
    reached = deferred()

    async def work(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.memo("seen", True, ctx)
        resolve(reached)
        await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Memo = define_task(
        name="test.graph-memo",
        version=1,
        initial=lambda _input: {"phase": "work"},
        phases={"work": work},
        abort=_nothing,
    )
    harness = (await open_tasks(MemoryStorage(), [Memo])).harness
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Memo, None, {"ownership": {"kind": "conversation"}}), context)
    first = await harness.task_graph(context)
    second = await harness.task_graph(context)
    assert second.value is first.value
    updates: list[str] = []

    def listener(value: Any, _context: Context, delivery: Any) -> None:
        if delivery.kind == "update":
            node = value["tasks"].get(str(id))
            updates.append("gone" if node is None else node["state"]["status"])

    first.subscribe(listener)
    harness.resume()
    await reached
    await flush()
    # Reservation changed the node; the memo commit did not.
    assert updates == ["running"]
    resolve(gate)
    await harness.wait_for_task(id, context)
    await flush()
    assert updates == ["running", "gone"]
    last = first.value
    first.dispose()
    second.dispose()
    # No observer is left, so the mount was dropped: a new observer builds a new revision.
    rebuilt = await harness.task_graph(context)
    assert rebuilt.value == last
    assert rebuilt.value is not last
    rebuilt.dispose()
    await harness.close(context)
