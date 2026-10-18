"""Port of ``test/harness-blocked.test.ts``.

Adaptation: upstream wraps the SQLite storage in a JavaScript ``Proxy`` whose methods hang once killed (or whose
``commit`` fails once); here a delegating wrapper does the same.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from durable_harness_support import completed, match_object, open_tasks, wait_for
from durable_session_support import context, deferred, flush, resolve

from pi_chord.context import Context
from pi_durable import create_registry, define_extension, define_task
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage
from pi_durable.types import Seq, StorageWrite, Task

#: Storages of killed Harnesses, never closed by them: closed after each test.
killed: list[Any] = []


@pytest.fixture(autouse=True)
async def _close_killed() -> AsyncIterator[None]:
    yield
    for storage in killed:
        with contextlib.suppress(BaseException):
            await storage.close(context)
    killed.clear()


class CrashState:
    def __init__(self, crashing: bool = True) -> None:
        self.crashing = crashing
        self.started = 0


async def _nothing(*_args: Any) -> None:
    return None


def crash_loop(state: CrashState) -> Any:
    """A task that hangs in its phase while ``crashing``, as one killed every time would, and completes otherwise."""

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        state.started += 1
        if state.crashing:
            await runtime.signal.wait()
        else:
            await runtime.commit(lambda _tx, _current: completed(None), ctx)

    return define_task(
        name="test.crash-loop", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )


class Killable:
    """Delegates to a storage until killed; then every call hangs, so it writes nothing more."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.dead = False

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.inner, name)
        if not callable(value):
            return value

        def call(*args: Any) -> Any:
            if self.dead:
                return asyncio.get_running_loop().create_future()
            return value(*args)

        return call


class Opened(NamedTuple):
    harness: Any
    wakes: list[Any]
    infos: list[Any]
    storage: Killable

    def kill(self) -> None:
        self.storage.dead = True
        killed.append(self.storage.inner)


async def open_with(path: Path, registry: Any, settings: Any = None) -> Opened:
    """Open a Harness over ``path``. ``kill()`` simulates a crash: every later call of its storage hangs, so it
    writes nothing more, not even the ``pending`` records a close writes for the tasks it stops."""
    wakes: list[Any] = []
    storage = Killable(await open_sqlite_storage(path))
    opened = await open_tasks(
        storage,  # type: ignore[arg-type]
        [],
        registry=registry,
        settings=settings if settings is not None else {},
        on_wake=wakes.append,
    )
    return Opened(opened.harness, wakes, opened.infos, storage)


def registry_with(name: str, *tasks: Any) -> Any:
    registry = create_registry()
    registry.install(define_extension(name, tasks=list(tasks)))
    return registry


CONVERSATION: Any = {"ownership": {"kind": "conversation"}}


async def create(harness: Any, task: Any, options: Any = CONVERSATION) -> Any:
    return await (await harness.root(context)).commit(lambda tx: tx.create_task(task, None, options), context)


async def get(harness: Any, id: Any) -> Any:
    return (await harness.get_task(id, context)) or {}


# ─── recovery at open ───────────────────────────────────────────────────────


class FailFirstCommit:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.fail_next = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        if not self.fail_next:
            return await self.inner.commit(writes, context)
        self.fail_next = False
        raise RuntimeError("reset")


async def test_keeps_the_hosts_wake_active_until_opens_recovery_pass_has_landed(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    Held = define_task(
        name="test.held", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": _nothing}, abort=_nothing
    )
    registry = registry_with("held", Held)
    first = await open_with(path, registry)
    id = await create(first.harness, Held)
    await first.harness.close(context)
    # A crash left the task holding its outcome: the next open's recovery pass makes it terminal.
    raw = await open_sqlite_storage(path)
    record: Any = await raw.task(id, context)
    rest = {key: value for key, value in record.items() if key != "memos"}
    completing = {**rest, "state": {"status": "completing", "outcome": {"status": "completed", "result": None}}}
    await raw.commit([{"type": "task", "value": completing}], context)
    await raw.close(context)

    # That pass's commit fails, as on a platform reset.
    failing = FailFirstCommit(await open_sqlite_storage(path))
    wakes: list[Any] = []
    harness = (await open_tasks(failing, [], registry=registry, on_wake=wakes.append)).harness  # type: ignore[arg-type]
    assert (await harness.closed)["reason"] == "failed"
    # The host was told work is underway, and never that it ended: its recovery alarm stays.
    assert wakes == [{"active": True}]
    with contextlib.suppress(BaseException):
        await failing.inner.close(context)


# ─── tasks that cannot run ──────────────────────────────────────────────────


async def test_parks_a_task_cut_off_restart_limit_times_in_a_row_shows_why_and_runs_it_again_on_retry_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    state = CrashState()
    Loop = crash_loop(state)
    registry = registry_with("loop", Loop)
    opened = await open_with(path, registry, {"restartLimit": 3})
    root = await opened.harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Loop, None, CONVERSATION), context)
    # Each open runs the phase, and each crash cuts it off: three restarts.
    for run in range(1, 4):
        opened.harness.resume()
        await wait_for(lambda run=run: state.started == run)
        opened.kill()  # a crash: a close would not count
        opened = await open_with(path, registry, {"restartLimit": 3})
    harness, infos = opened.harness, opened.infos
    assert (await get(harness, id)).get("restarts") == 3
    assert infos == [
        {
            "type": "task_parked",
            "task": {"id": id, "kind": "test.crash-loop", "conversationId": root.id},
            "extension": "loop",
        }
    ]
    inspection = await harness.inspect(context)
    assert next(task["state"] for task in inspection["tasks"] if task["record"]["id"] == id) == {
        "kind": "blocked",
        "reason": "restart_limit",
    }
    graph = await harness.task_graph(context)
    assert graph.value["tasks"][str(id)].get("blocked") == "restart_limit"
    # Parked work needs no process: nothing is active.
    await wait_for(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False}
    harness.resume()
    # A scheduling pass runs within a few turns; it leaves the parked task out.
    for _ in range(5):
        await flush()
    assert state.started == 3

    state.crashing = False
    await harness.retry_task(id, context)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    assert state.started == 4
    graph.dispose()
    await harness.close(context)


async def test_counts_no_restart_for_a_task_a_close_stopped(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    state = CrashState()
    Loop = crash_loop(state)
    registry = registry_with("loop", Loop)
    opened = await open_with(path, registry, {"restartLimit": 1})
    id = await create(opened.harness, Loop)
    opened.harness.resume()
    await wait_for(lambda: state.started == 1)
    await opened.harness.close(context)
    opened = await open_with(path, registry, {"restartLimit": 1})
    assert match_object(await opened.harness.get_task(id, context), {"state": {"status": "pending"}})
    assert "restarts" not in await get(opened.harness, id)
    state.crashing = False
    assert (await opened.harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    await opened.harness.close(context)


async def test_keeps_an_abort_mark_admitted_before_close_in_the_record_close_writes_back(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    state = CrashState()
    Loop = crash_loop(state)
    registry = registry_with("loop", Loop)
    harness = (await open_with(path, registry)).harness
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Loop, None, CONVERSATION), context)
    harness.resume()
    await wait_for(lambda: state.started == 1)
    # Hold the line; the abort mark is admitted behind it, then close begins.
    gate = deferred()

    async def hold(_tx: Any) -> None:
        await gate

    holding = root.commit(hold, context)
    aborting = harness.abort_task(id, context)
    closing = harness.close(context)
    # The invocation ends while the line is held; its write-back waits behind the abort mark.
    await flush()
    resolve(gate)
    await holding
    assert await aborting == "marked"
    await closing
    raw = await open_sqlite_storage(path)
    assert match_object(await raw.task(id, context), {"abortRequested": True, "state": {"status": "pending"}})
    await raw.close(context)


async def test_faults_a_task_whose_abort_handler_is_cut_off_restart_limit_times(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    aborts = [0]

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        await runtime.signal.wait()

    async def abort(_task: Any, runtime: Any, _ctx: Context) -> None:
        aborts[0] += 1
        await runtime.signal.wait()

    Stuck = define_task(
        name="test.stuck-abort", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort
    )
    registry = registry_with("stuck", Stuck)
    opened = await open_with(path, registry, {"restartLimit": 1})
    id = await create(opened.harness, Stuck)
    opened.harness.resume()
    asyncio.ensure_future(opened.harness.abort_task(id, context)).add_done_callback(_consume)
    await wait_for(lambda: aborts[0] == 1)
    opened.kill()  # a crash in the abort handler
    opened = await open_with(path, registry, {"restartLimit": 1})
    settled = await opened.harness.wait_for_task(id, context)
    assert match_object(
        settled["state"]["outcome"], {"status": "faulted", "error": {"detail": {"reason": "restart_limit"}}}
    )
    # The receipt keeps no restart count.
    assert "restarts" not in settled
    assert aborts[0] == 1
    await wait_for(lambda: len(opened.infos) > 0)
    assert match_object(opened.infos, [{"type": "task_faulted", "task": {"id": id}, "reason": "restart_limit"}])
    await opened.harness.close(context)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def test_runs_a_parked_task_once_a_raised_restart_limit_no_longer_parks_it(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    state = CrashState()
    Loop = crash_loop(state)
    registry = registry_with("loop", Loop)
    settings: dict[str, Any] = {"restartLimit": 1}
    opened = await open_with(path, registry, settings)
    id = await create(opened.harness, Loop)
    opened.harness.resume()
    await wait_for(lambda: state.started == 1)
    opened.kill()  # a crash: a close would not count
    opened = await open_with(path, registry, settings)
    graph = await opened.harness.task_graph(context)
    opened.harness.resume()
    await wait_for(lambda: (graph.value["tasks"].get(str(id)) or {}).get("blocked") == "restart_limit")
    # Let a scheduling pass leave the parked task out.
    for _ in range(5):
        await flush()
    state.crashing = False
    # The host's live settings change; the next pass runs the task.
    settings["restartLimit"] = math.inf
    opened.harness.resume()
    assert (await opened.harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    assert state.started == 2
    graph.dispose()
    await opened.harness.close(context)


async def test_still_runs_the_abort_handler_of_a_parked_task(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    state = CrashState()
    aborted: list[Any] = []

    async def abort(task: Any, runtime: Any, ctx: Context) -> None:
        aborted.append(task["id"])
        await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)

    Loop = Task(dataclasses.replace(crash_loop(state).definition, abort=abort))
    registry = registry_with("loop", Loop)
    opened = await open_with(path, registry, {"restartLimit": 1})
    id = await create(opened.harness, Loop)
    opened.harness.resume()
    await wait_for(lambda: state.started == 1)
    opened.kill()  # a crash: a close would not count
    opened = await open_with(path, registry, {"restartLimit": 1})
    await opened.harness.abort_task(id, context)
    assert (await opened.harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "aborted"
    assert aborted == [id]
    assert state.started == 1
    await opened.harness.close(context)


async def test_does_not_count_the_same_checkpoint_committed_again_as_progress(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    started = [0]

    async def run(task: Any, runtime: Any, ctx: Context) -> None:
        started[0] += 1
        # Intent before an effect, then the effect never returns: the checkpoint is unchanged.
        checkpoint = task["state"]["checkpoint"]
        await runtime.commit(lambda _tx, _current: {"status": "running", "checkpoint": checkpoint}, ctx)
        await runtime.signal.wait()

    Intent = define_task(
        name="test.intent", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )
    registry = registry_with("intent", Intent)
    opened = await open_with(path, registry, {"restartLimit": 2})
    id = await create(opened.harness, Intent)
    for run_number in range(1, 3):
        opened.harness.resume()
        await wait_for(lambda run_number=run_number: started[0] == run_number)
        harness = opened.harness

        async def running(harness: Any = harness) -> bool:
            return (await get(harness, id)).get("state", {}).get("status") == "running"

        await wait_for(running)
        opened.kill()  # a crash: a close would not count
        opened = await open_with(path, registry, {"restartLimit": 2})
    assert (await get(opened.harness, id)).get("restarts") == 2
    assert (await opened.harness.inspect(context))["tasks"][0]["state"] == {
        "kind": "blocked",
        "reason": "restart_limit",
    }
    await opened.harness.close(context)


async def test_starts_the_count_over_when_the_task_makes_progress(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    crashing = [True]
    started = [0]

    async def one(_task: Any, runtime: Any, ctx: Context) -> None:
        started[0] += 1
        if crashing[0]:
            await runtime.signal.wait()
        else:
            await runtime.commit(lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "two"}}, ctx)

    async def two(_task: Any, runtime: Any, _ctx: Context) -> None:
        started[0] += 1
        await runtime.signal.wait()

    Steps = define_task(
        name="test.steps",
        version=1,
        initial=lambda _input: {"phase": "one"},
        phases={"one": one, "two": two},
        abort=_nothing,
    )
    registry = registry_with("steps", Steps)
    opened = await open_with(path, registry, {"restartLimit": 2})
    id = await create(opened.harness, Steps)
    opened.harness.resume()
    await wait_for(lambda: started[0] == 1)
    opened.kill()  # a crash: a close would not count
    opened = await open_with(path, registry, {"restartLimit": 2})
    assert (await get(opened.harness, id)).get("restarts") == 1
    # Progress: phase one commits phase two, which resets the count.
    crashing[0] = False
    opened.harness.resume()
    await wait_for(lambda: started[0] == 3)
    assert "restarts" not in await get(opened.harness, id)
    await opened.harness.close(context)


async def test_tells_the_graph_and_the_host_when_a_migration_fails(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"

    def cannot_migrate(*_args: Any) -> Any:
        raise RuntimeError("cannot migrate")

    V1 = define_task(
        name="test.migrating",
        version=1,
        initial=lambda _input: {"phase": "run"},
        phases={"run": _nothing},
        abort=_nothing,
    )
    V2 = define_task(
        name="test.migrating",
        version=2,
        initial=lambda _input: {"phase": "run"},
        phases={"run": _nothing},
        abort=_nothing,
        migrate=cannot_migrate,
    )
    opened = await open_with(path, registry_with("v1", V1))
    # Created, never run: scheduling stays off.
    id = await create(opened.harness, V1)
    await opened.harness.close(context)
    opened = await open_with(path, registry_with("v2", V2))
    graph = await opened.harness.task_graph(context)
    opened.harness.resume()
    await wait_for(lambda: (graph.value["tasks"].get(str(id)) or {}).get("blocked") == "migration_failed")
    await wait_for(lambda: not opened.harness.wake()["active"])
    assert opened.harness.wake() == {"active": False}
    await wait_for(lambda: len(opened.wakes) > 0 and opened.wakes[-1]["active"] is False)
    graph.dispose()
    await opened.harness.close(context)


async def test_shows_an_abandoned_task_waiting_for_its_definition_as_blocked(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    Abandoned = define_task(
        name="test.abandoned",
        version=1,
        initial=lambda _input: {"phase": "run"},
        phases={"run": _nothing},
        abort=_nothing,
    )
    opened = await open_with(path, registry_with("abandoned", Abandoned))
    id = await create(opened.harness, Abandoned, {"ownership": {"kind": "conversation"}, "abandonOnRestart": True})
    await opened.harness.close(context)
    # The next open marks it `restart`; without its definition, it waits for one to clean up.
    opened = await open_with(path, create_registry())
    graph = await opened.harness.task_graph(context)
    opened.harness.resume()

    async def restart_marked() -> bool:
        return (await get(opened.harness, id)).get("abortReason") == "restart"

    await wait_for(restart_marked)
    await wait_for(lambda: (graph.value["tasks"].get(str(id)) or {}).get("blocked") == "missing_task")
    graph.dispose()
    await opened.harness.close(context)


async def test_shows_a_task_without_a_definition_as_blocked_and_wakes_the_host_once_a_registry_change_unblocks_it(
    tmp_path: Path,
) -> None:
    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Missing = define_task(
        name="test.missing", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )
    registry = create_registry()
    opened = await open_with(tmp_path / "session.sqlite", registry)
    harness, wakes = opened.harness, opened.wakes
    graph = await harness.task_graph(context)
    id = await create(harness, Missing)
    harness.resume()
    await wait_for(lambda: (graph.value["tasks"].get(str(id)) or {}).get("blocked") == "missing_task")
    await wait_for(lambda: not harness.wake()["active"])
    registry.install(define_extension("late", tasks=[Missing]))
    # The graph follows the registry; the host hears it can run work now.
    await wait_for(lambda: "blocked" not in (graph.value["tasks"].get(str(id)) or {}))
    await wait_for(lambda: any(wake["active"] for wake in wakes))
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    graph.dispose()
    await harness.close(context)
