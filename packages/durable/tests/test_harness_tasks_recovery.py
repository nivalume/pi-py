"""Port of ``test/harness-tasks-recovery.test.ts``.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from durable_harness_support import (
    aborted,
    aborted_with,
    add_task,
    add_tool,
    completed,
    counting_reader,
    eventually,
    match_object,
    open_tasks,
    wait_for,
)
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, flush, resolve

from pi_ai import Type, create_models
from pi_chord.context import Context
from pi_durable import Harness, MemoryStorage, create_registry, define_doc, define_task, define_tool
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


def controlled() -> ControlledStorage:
    return ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])


async def create_in(harness: Any, task: Any) -> Any:
    root = await harness.root(context)
    return await root.commit(lambda tx: tx.create_task(task, None, {"ownership": {"kind": "conversation"}}), context)


async def never() -> None:
    """``await new Promise(() => {})``."""
    await asyncio.get_running_loop().create_future()


class TransferService:
    """Fake external service whose operations are idempotent by request key."""

    def __init__(self) -> None:
        self.applied: dict[str, int] = {}
        self.calls = 0

    async def apply(self, key: str, amount: int) -> int:
        self.calls += 1
        if key not in self.applied:
            self.applied[key] = amount * 10
        return self.applied[key]


@dataclass(slots=True)
class Interrupt:
    first: bool


def transfer_task(service: TransferService, interrupt: Interrupt) -> Any:
    """Intent/effect/outcome task: ``prepare`` commits the intent, ``apply`` performs the effect and commits the outcome.

    While ``interrupt.first`` is set, the first ``apply`` blocks after the effect until the invocation is signalled.
    """

    async def prepare(task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.memo("requested", task["input"]["amount"], ctx)
        key = f"transfer-{task['id']}"
        await runtime.commit(
            lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "apply", "key": key}}, ctx
        )

    async def apply(task: Any, runtime: Any, ctx: Context) -> None:
        receipt = await service.apply(task["state"]["checkpoint"]["key"], task["input"]["amount"])
        if interrupt.first:
            interrupt.first = False
            await aborted(runtime.signal)
        await runtime.commit(lambda _tx, _current: completed({"receipt": receipt}), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)

    return define_task(
        name="test.transfer",
        version=1,
        initial=lambda _input: {"phase": "prepare"},
        phases={"prepare": prepare, "apply": apply},
        abort=abort,
    )


def versioned(version: int, result: str, migrate: Callable[[Any, Any, int], Any] | None = None) -> Any:
    """A versioned one-phase task that completes with ``result``, optionally migrating older records."""

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(result), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: aborted_with(result), ctx)

    return define_task(
        name="test.versioned",
        version=version,
        initial=lambda _input: {"phase": "run"},
        phases={"run": run},
        abort=abort,
        migrate=migrate,
    )


async def _nothing(*_args: Any) -> None:
    return None


# ─── task recovery ──────────────────────────────────────────────────────────


async def test_resumes_an_intent_effect_outcome_task_interrupted_after_its_intent_across_close_and_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    service = TransferService()
    Transfer = transfer_task(service, Interrupt(True))

    first = await open_tasks(await open_sqlite_storage(path), [Transfer])
    root = await first.harness.root(context)
    id = await root.commit(
        lambda tx: tx.create_task(Transfer, {"amount": 7}, {"ownership": {"kind": "conversation"}}), context
    )
    first.harness.resume()
    await wait_for(lambda: service.calls == 1)
    await first.harness.close(context)

    second = await open_tasks(await open_sqlite_storage(path), [Transfer])
    # Open reconciled `running` to `pending` and kept the checkpoint and memos; nothing ran yet.
    assert match_object(
        await second.harness.get_task(id, context),
        {
            "abortRequested": False,
            "memos": {"requested": 7},
            "state": {"status": "pending", "checkpoint": {"phase": "apply", "key": f"transfer-{id}"}},
        },
    )
    assert service.calls == 1
    second.harness.resume()
    receipt = await second.harness.wait_for_task(id, context)
    assert receipt["state"]["outcome"] == {"status": "completed", "result": {"receipt": 70}}
    assert service.calls == 2
    assert len(service.applied) == 1
    await second.harness.close(context)

    third = await open_tasks(await open_sqlite_storage(path), [Transfer])
    assert await third.harness.get_task(id, context) == receipt
    await third.harness.close(context)


async def test_keeps_a_tasks_started_at_across_close_and_reopen_and_stamps_ended_at_when_it_settles(
    tmp_path: Path,
) -> None:
    """#10549"""
    path = tmp_path / "session.sqlite"
    service = TransferService()
    Transfer = transfer_task(service, Interrupt(True))
    clock = [2_000]

    def now() -> float:
        return clock[0]

    first = await open_tasks(await open_sqlite_storage(path), [Transfer], now=now)
    root = await first.harness.root(context)
    id = await root.commit(
        lambda tx: tx.create_task(Transfer, {"amount": 7}, {"ownership": {"kind": "conversation"}}), context
    )
    first.harness.resume()
    await wait_for(lambda: service.calls == 1)
    clock[0] = 3_000
    await first.harness.close(context)

    clock[0] = 9_000
    second = await open_tasks(await open_sqlite_storage(path), [Transfer], now=now)
    reopened: Any = await second.harness.get_task(id, context)
    assert [reopened["state"]["status"], reopened.get("startedAt"), reopened.get("endedAt")] == ["pending", 2_000, None]
    second.harness.resume()
    receipt = await second.harness.wait_for_task(id, context)
    assert [receipt.get("startedAt"), receipt.get("endedAt")] == [2_000, 9_000]
    await second.harness.close(context)


async def test_resumes_abort_work_after_close_at_every_direct_task_abort_stage(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    log: list[str] = []
    abort_gate = {"block": True}
    abort_reached = deferred()
    run_release = deferred()

    async def run(*_args: Any) -> None:
        log.append("run")
        # Ignores the abort signal until released, so the mark is durable while the run is active.
        await run_release

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        log.append("abort")
        if abort_gate["block"]:
            resolve(abort_reached)
            await aborted(runtime.signal)
        await runtime.commit(lambda _tx, _current: aborted_with("stop"), ctx)

    Abortable = define_task(
        name="test.abortable", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort
    )

    # Stage 1: the mark is committed while the run invocation is still active.
    opened = await open_tasks(await open_sqlite_storage(path), [Abortable])
    id = await create_in(opened.harness, Abortable)
    opened.harness.resume()
    await wait_for(lambda: len(log) == 1)
    aborting = opened.harness.abort_task(id, context)
    while not ((await opened.harness.get_task(id, context)) or {}).get("abortRequested"):
        await flush()
    closing = opened.harness.close(context)
    resolve(run_release)
    await closing
    assert await aborting == "marked"
    assert log == ["run"]

    # Stage 2: reopen dispatches the abort invocation, never the run; close while it is active.
    opened = await open_tasks(await open_sqlite_storage(path), [Abortable])
    assert match_object(
        await opened.harness.get_task(id, context), {"abortRequested": True, "state": {"status": "pending"}}
    )
    opened.harness.resume()
    await abort_reached
    await opened.harness.close(context)
    assert log == ["run", "abort"]

    # Stage 3: a fresh abort invocation settles the task.
    abort_gate["block"] = False
    opened = await open_tasks(await open_sqlite_storage(path), [Abortable])
    opened.harness.resume()
    assert (await opened.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "aborted",
        "reason": "stop",
    }
    await opened.harness.close(context)
    assert log == ["run", "abort", "abort"]

    # Stage 4: the terminal receipt survives reopen and nothing runs again.
    opened = await open_tasks(await open_sqlite_storage(path), [Abortable])
    opened.harness.resume()
    await flush()
    assert await opened.harness.abort_task(id, context) == "terminal"
    await opened.harness.close(context)
    assert log == ["run", "abort", "abort"]


# ─── task crash recovery ────────────────────────────────────────────────────
#
# Crash simulation: the crashed Harness is abandoned without close, its held storage commit never lands, and its
# blocked handlers never return. A new Harness then opens the same storage.


def crash_task(log: list[str], *, block_abort: bool) -> Any:
    """A task whose run ignores its signal and whose abort handler blocks while ``block_abort`` is set."""

    async def run(*_args: Any) -> None:
        log.append("run")
        await never()

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        log.append("abort")
        if block_abort:
            await never()
        await runtime.commit(lambda _tx, _current: aborted_with("recovered"), ctx)

    return define_task(
        name="test.crash", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort
    )


class Recovered(NamedTuple):
    harness: Any
    id: Any


async def recover(storage: ControlledStorage, log: list[str]) -> Recovered:
    storage.crash()
    harness = (await open_tasks(storage, [crash_task(log, block_abort=False)])).harness
    page = await harness.commit(lambda tx: tx.scan_tasks({"kind": "test.crash"}, 1), context)
    return Recovered(harness, page["items"][0]["id"])


class Crashed(NamedTuple):
    storage: ControlledStorage
    harness: Any
    id: Any


async def crashed_run(log: list[str]) -> Crashed:
    storage = controlled()
    harness = (await open_tasks(storage, [crash_task(log, block_abort=True)])).harness
    id = await create_in(harness, crash_task(log, block_abort=True))
    harness.resume()
    await eventually(lambda: len(log) == 1)
    return Crashed(storage, harness, id)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def test_crash_while_the_mark_commit_is_in_storage_the_run_resumes() -> None:
    log: list[str] = []
    storage, harness, id = await crashed_run(log)
    held = storage.hold_commits()
    asyncio.ensure_future(harness.abort_task(id, context)).add_done_callback(_consume)
    await held.entered
    recovered = await recover(storage, log)
    assert match_object(
        await recovered.harness.get_task(id, context), {"abortRequested": False, "state": {"status": "pending"}}
    )
    recovered.harness.resume()
    await eventually(lambda: len(log) == 2)
    assert log == ["run", "run"]


async def test_crash_after_the_mark_before_the_run_joins_only_the_abort_handler_runs() -> None:
    log: list[str] = []
    storage, harness, id = await crashed_run(log)
    # The run ignores its signal, so abort_task() never finishes joining it.
    joining = asyncio.ensure_future(harness.abort_task(id, context))
    joining.add_done_callback(_consume)
    while not ((await harness.get_task(id, context)) or {}).get("abortRequested"):
        await flush()
    await flush()
    assert not joining.done()
    recovered = await recover(storage, log)
    assert match_object(
        await recovered.harness.get_task(id, context), {"abortRequested": True, "state": {"status": "pending"}}
    )
    recovered.harness.resume()
    assert (await recovered.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "aborted",
        "reason": "recovered",
    }
    assert log == ["run", "abort"]
    await recovered.harness.close(context)


async def test_crash_while_the_abort_handler_runs_a_fresh_abort_invocation_settles_the_task() -> None:
    log: list[str] = []
    storage = controlled()
    Crash = crash_task(log, block_abort=True)
    harness = (await open_tasks(storage, [Crash])).harness
    id = await create_in(harness, Crash)
    await harness.abort_task(id, context)
    harness.resume()
    await eventually(lambda: len(log) == 1)
    assert log == ["abort"]
    assert match_object(await harness.get_task(id, context), {"state": {"status": "running"}})
    recovered = await recover(storage, log)
    recovered.harness.resume()
    assert (await recovered.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "aborted",
        "reason": "recovered",
    }
    assert log == ["abort", "abort"]
    await recovered.harness.close(context)


async def test_crash_while_the_abort_outcome_is_in_storage_the_abort_handler_runs_again() -> None:
    log: list[str] = []
    storage = controlled()
    reached = deferred()
    proceed = deferred()

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        log.append("abort")
        resolve(reached)
        await proceed
        await runtime.commit(lambda _tx, _current: aborted_with("lost"), ctx)

    Crash = define_task(
        name="test.crash", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": _nothing}, abort=abort
    )
    harness = (await open_tasks(storage, [Crash])).harness
    id = await create_in(harness, Crash)
    await harness.abort_task(id, context)
    harness.resume()
    await reached
    held = storage.hold_commits()
    resolve(proceed)
    await held.entered
    recovered = await recover(storage, log)
    recovered.harness.resume()
    assert (await recovered.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "aborted",
        "reason": "recovered",
    }
    assert log == ["abort", "abort"]
    await recovered.harness.close(context)


async def test_crash_after_the_terminal_outcome_nothing_runs_again() -> None:
    log: list[str] = []
    storage = controlled()
    Crash = crash_task(log, block_abort=False)
    harness = (await open_tasks(storage, [Crash])).harness
    id = await create_in(harness, Crash)
    await harness.abort_task(id, context)
    harness.resume()
    await harness.wait_for_task(id, context)
    recovered = await recover(storage, log)
    recovered.harness.resume()
    await flush()
    assert log == ["abort"]
    assert ((await recovered.harness.get_task(id, context)) or {}).get("state") == {
        "status": "terminal",
        "outcome": {"status": "aborted", "reason": "recovered"},
    }
    await recovered.harness.close(context)


async def test_crash_while_the_reservation_commit_is_in_storage_the_task_is_still_pending() -> None:
    log: list[str] = []
    storage = controlled()
    Crash = crash_task(log, block_abort=False)
    harness = (await open_tasks(storage, [Crash])).harness
    id = await create_in(harness, Crash)
    held = storage.hold_commits()
    harness.resume()
    await held.entered
    recovered = await recover(storage, log)
    assert ((await recovered.harness.get_task(id, context)) or {})["state"]["status"] == "pending"
    assert log == []
    await recovered.harness.abort_task(id, context)
    recovered.harness.resume()
    await recovered.harness.wait_for_task(id, context)
    assert log == ["abort"]
    await recovered.harness.close(context)


# ─── blocked tasks ──────────────────────────────────────────────────────────


async def test_keeps_a_task_with_a_missing_definition_pending_and_live_for_idle_waits_until_registration() -> None:
    V1 = versioned(1, "v1")
    opened = await open_tasks(MemoryStorage(), [])
    harness, registry = opened.harness, opened.registry
    id = await create_in(harness, V1)
    harness.resume()
    await flush()
    assert ((await harness.get_task(id, context)) or {})["state"]["status"] == "pending"
    idle = harness.wait_for_idle(context)
    await flush()
    add_task(registry, V1)
    await idle
    assert ((await harness.get_task(id, context)) or {}).get("state") == {
        "status": "terminal",
        "outcome": {"status": "completed", "result": "v1"},
    }
    await harness.close(context)


async def test_keeps_a_task_stored_by_a_newer_version_pending_until_a_fitting_definition_is_registered() -> None:
    registry = create_registry()
    add_task(registry, versioned(1, "old"))
    harness = (await open_tasks(MemoryStorage(), [], registry=registry)).harness
    id = await create_in(harness, versioned(2, "new"))
    harness.resume()
    await flush()
    assert ((await harness.get_task(id, context)) or {})["state"]["status"] == "pending"
    # The same extension name replaces the old one in place.
    add_task(registry, versioned(2, "new"))
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "completed", "result": "new"}
    await harness.close(context)


async def test_migrates_at_reservation_leaves_the_record_unchanged_when_migration_fails_and_retries_only_for_a_new_definition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_tasks(await open_sqlite_storage(path), [])
    id = await create_in(opened.harness, versioned(1, "v1"))
    await opened.harness.close(context)

    registry = create_registry()
    failures = [0]

    def failing(*_args: Any) -> Any:
        failures[0] += 1
        raise RuntimeError("cannot migrate")

    add_task(registry, versioned(2, "v2", failing))
    opened = await open_tasks(await open_sqlite_storage(path), [], registry=registry)
    opened.harness.resume()
    await wait_for(lambda: len(opened.reports) == 1)

    # An unrelated registry change wakes the scheduler without retrying the same failed definition.
    async def unrelated(*_args: Any) -> Any:
        return {}

    add_tool(
        registry,
        define_tool(name="unrelated", description="unrelated", parameters=Type.Object({}), execute=unrelated),
    )
    await flush()
    assert failures[0] == 1
    assert len(opened.reports) == 1
    assert "cannot migrate" in str(opened.reports[0])
    assert match_object(
        await opened.harness.get_task(id, context),
        {"version": 1, "state": {"status": "pending", "checkpoint": {"phase": "run"}}},
    )

    migrations: list[int] = []

    def migrate(_input: Any, checkpoint: Any, from_version: int) -> Any:
        migrations.append(from_version)
        return {"input": None, "checkpoint": checkpoint}

    # The same extension name replaces the old one in place.
    add_task(registry, versioned(2, "v2", migrate))
    receipt = await opened.harness.wait_for_task(id, context)
    assert match_object(receipt, {"version": 2, "state": {"outcome": {"status": "completed", "result": "v2"}}})
    assert migrations == [1]
    await opened.harness.close(context)


async def test_blocks_an_older_record_whose_newer_definition_has_no_migration(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_tasks(await open_sqlite_storage(path), [])
    id = await create_in(opened.harness, versioned(1, "v1"))
    await opened.harness.close(context)
    opened = await open_tasks(await open_sqlite_storage(path), [versioned(2, "v2")])
    opened.harness.resume()
    await wait_for(lambda: len(opened.reports) == 1)
    assert "has no migration from 1" in str(opened.reports[0])
    assert match_object(await opened.harness.get_task(id, context), {"version": 1, "state": {"status": "pending"}})
    await opened.harness.close(context)


async def test_settles_an_aborted_blocked_task_as_orphaned_and_retires_its_documents() -> None:
    Scratch = define_doc(kind="test.orphan-scratch", version=1, scope="task", initial=lambda: {"n": 0})
    harness = (await open_tasks(MemoryStorage(), [])).harness
    root = await harness.root(context)

    async def create(tx: Any) -> Any:
        created = await tx.create_task(versioned(1, "x"), None, {"ownership": {"kind": "conversation"}})
        (await tx.doc(Scratch, created))["n"] = 1
        return created

    id = await root.commit(create, context)
    # Before resume: the marking commit settles the blocked task directly.
    assert await harness.abort_task(id, context) == "marked"
    assert ((await harness.get_task(id, context)) or {}).get("state") == {
        "status": "terminal",
        "outcome": {"status": "orphaned", "reason": "missing_task"},
    }
    assert await harness.snapshot(Scratch, id, context) is None
    await harness.close(context)


async def test_orphans_a_marked_task_whose_definition_disappeared_while_its_run_was_active() -> None:
    reached = deferred()

    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        resolve(reached)
        await aborted(runtime.signal)

    async def abort(*_args: Any) -> None:
        raise RuntimeError("must not run")

    Running = define_task(
        name="test.vanishing", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=abort
    )
    registry = create_registry()
    registration = add_task(registry, Running)
    harness = (await open_tasks(MemoryStorage(), [], registry=registry)).harness
    id = await create_in(harness, Running)
    harness.resume()
    await reached
    registration.dispose()
    # An active run means the mark does not orphan directly; the scheduler orphans once the run has ended.
    assert await harness.abort_task(id, context) == "marked"
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "missing_task",
    }
    await harness.close(context)


async def test_orphans_a_reopened_abort_marked_task_without_a_definition_once_scheduling_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.sqlite"
    opened = await open_tasks(await open_sqlite_storage(path), [versioned(1, "x")])
    id = await create_in(opened.harness, versioned(1, "x"))
    assert await opened.harness.abort_task(id, context) == "marked"
    await opened.harness.close(context)

    opened = await open_tasks(await open_sqlite_storage(path), [])
    assert match_object(
        await opened.harness.get_task(id, context), {"abortRequested": True, "state": {"status": "pending"}}
    )
    idle = opened.harness.wait_for_idle(context)
    opened.harness.resume()
    await idle
    assert ((await opened.harness.get_task(id, context)) or {}).get("state") == {
        "status": "terminal",
        "outcome": {"status": "orphaned", "reason": "missing_task"},
    }
    await opened.harness.close(context)


# ─── definition handover ────────────────────────────────────────────────────


@dataclass(slots=True)
class Gates:
    a: asyncio.Future[Any] | None = None
    b: asyncio.Future[Any] | None = None
    on_end: Callable[[], None] | None = None


def handover_task(
    label: str, version: int, log: list[str], *, gates: Gates | None = None, migrate: Callable[..., Any] | None = None
) -> Any:
    def advance(phase: str, following: Any) -> Any:
        async def handler(_task: Any, runtime: Any, ctx: Context) -> None:
            log.append(f"{label}:{phase} start")
            gate = None if gates is None else getattr(gates, phase)
            if gate is not None:
                await gate
            await runtime.commit(lambda _tx, _current: {"status": "running", "checkpoint": following}, ctx)
            # Leave room for a wrongly dispatched successor before this invocation ends.
            await flush()
            if gates is not None and gates.on_end is not None:
                gates.on_end()
            log.append(f"{label}:{phase} end")

        return handler

    async def c(_task: Any, runtime: Any, ctx: Context) -> None:
        log.append(f"{label}:c")
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        log.append(f"{label}:abort")
        await runtime.commit(lambda _tx, _current: aborted_with(label), ctx)

    return define_task(
        name="test.handover",
        version=version,
        initial=lambda _input: {"phase": "a"},
        phases={"a": advance("a", {"phase": "b"}), "b": advance("b", {"phase": "c"}), "c": c},
        abort=abort,
        migrate=None if migrate is None else (lambda *_args: migrate()),
    )


class Handover(NamedTuple):
    harness: Any
    registry: Any
    reports: list[BaseException]
    infos: list[Any]
    old: Any
    id: Any


async def start_handover(log: list[str], gates: Gates, storage: Any = None) -> Handover:
    registry = create_registry()
    old = add_task(registry, handover_task("old", 1, log, gates=gates))
    opened = await open_tasks(storage if storage is not None else MemoryStorage(), [], registry=registry)
    id = await create_in(opened.harness, handover_task("old", 1, log))
    opened.harness.resume()
    await eventually(lambda: len(log) == 1)
    return Handover(opened.harness, opened.registry, opened.reports, opened.infos, old, id)


async def test_hands_over_at_the_next_phase_boundary_to_a_same_version_replacement_without_overlap() -> None:
    log: list[str] = []
    gate = deferred()
    opened = await start_handover(log, Gates(a=gate))
    # The same extension name replaces the old one in place.
    add_task(opened.registry, handover_task("new", 1, log))
    resolve(gate)
    await opened.harness.wait_for_task(opened.id, context)
    assert log == ["old:a start", "old:a end", "new:b start", "new:b end", "new:c"]
    await opened.harness.close(context)


async def test_keeps_the_memos_across_a_handover() -> None:
    log: list[str] = []
    gate = deferred()
    reached = deferred()

    def memo_task(label: str) -> Any:
        async def a(_task: Any, runtime: Any, ctx: Context) -> None:
            await runtime.memo("picked", label, ctx)
            resolve(reached)
            await gate
            await runtime.commit(lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "b"}}, ctx)

        async def b(_task: Any, runtime: Any, ctx: Context) -> None:
            # The candidate loses to the memo the old definition stored.
            picked = await runtime.memo("picked", label, ctx)
            log.append(f"{label}:b {picked}")
            await runtime.commit(lambda _tx, _current: completed(picked), ctx)

        return define_task(
            name="test.handover-memo",
            version=1,
            initial=lambda _input: {"phase": "a"},
            phases={"a": a, "b": b},
            abort=_nothing,
        )

    registry = create_registry()
    add_task(registry, memo_task("old"))
    harness = (await open_tasks(MemoryStorage(), [], registry=registry)).harness
    id = await create_in(harness, memo_task("old"))
    harness.resume()
    await reached
    add_task(registry, memo_task("new"))
    resolve(gate)
    receipt = await harness.wait_for_task(id, context)
    assert receipt["state"]["outcome"] == {"status": "completed", "result": "old"}
    assert log == ["new:b old"]
    await harness.close(context)


async def test_hands_over_to_a_newer_version_with_a_migration() -> None:
    log: list[str] = []
    gate = deferred()
    opened = await start_handover(log, Gates(a=gate))
    # The same extension name replaces the old one in place.
    add_task(
        opened.registry, handover_task("v2", 2, log, migrate=lambda: {"input": None, "checkpoint": {"phase": "c"}})
    )
    resolve(gate)
    receipt = await opened.harness.wait_for_task(opened.id, context)
    assert receipt["version"] == 2
    assert log == ["old:a start", "old:a end", "v2:c"]
    await opened.harness.close(context)


async def test_hands_over_to_a_newer_version_whose_migration_fails_and_leaves_the_task_blocked() -> None:
    log: list[str] = []
    gate = deferred()
    opened = await start_handover(log, Gates(a=gate))

    def broken() -> Any:
        raise RuntimeError("broken migration")

    # The same extension name replaces the old one in place.
    add_task(opened.registry, handover_task("broken", 2, log, migrate=broken))
    resolve(gate)
    await eventually(lambda: len(opened.infos) == 1)
    assert match_object(opened.infos, [{"type": "migration_failed", "task": {"id": opened.id}}])
    await flush()
    assert match_object(
        await opened.harness.get_task(opened.id, context),
        {"version": 1, "state": {"status": "pending", "checkpoint": {"phase": "b"}}},
    )
    assert log == ["old:a start", "old:a end"]
    await opened.harness.close(context)


async def test_keeps_running_under_the_old_definition_when_the_replacement_is_missing_or_cannot_take_the_task() -> None:
    log: list[str] = []
    gate_a = deferred()
    gate_b = deferred()
    opened = await start_handover(log, Gates(a=gate_a, b=gate_b))
    opened.old.dispose()
    resolve(gate_a)
    await eventually(lambda: "old:b start" in log)
    # A newer definition without a migration cannot take the task either.
    add_task(opened.registry, handover_task("incompatible", 2, log))
    resolve(gate_b)
    await opened.harness.wait_for_task(opened.id, context)
    assert log == ["old:a start", "old:a end", "old:b start", "old:b end", "old:c"]
    assert match_object(
        opened.infos,
        [
            {"type": "task_outdated", "task": {"id": opened.id}, "reason": "missing_task"},
            {"type": "task_outdated", "task": {"id": opened.id}, "reason": "incompatible_task"},
        ],
    )
    await opened.harness.close(context)


async def test_rejects_a_runtime_commit_of_the_old_invocation_queued_behind_its_handover_commit() -> None:
    log: list[str] = []
    gate = deferred()
    storage = controlled()
    refs: dict[str, Any] = {}
    registry = create_registry()

    async def a(task: Any, runtime: Any, ctx: Context) -> None:
        refs["runtime"] = runtime
        await gate
        await runtime.commit(lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "b"}}, ctx)
        # Hold the line with an unrelated commit so the handover commit queues behind it.
        refs["held"] = storage.hold_commits()

        async def blocker(tx: Any) -> None:
            await tx.append_entry(task["conversationId"], {"kind": "blocker"})

        asyncio.ensure_future(refs["harness"].commit(blocker, ctx)).add_done_callback(_consume)

    Old = define_task(
        name="test.handover",
        version=1,
        initial=lambda _input: {"phase": "a"},
        phases={"a": a, "b": _nothing, "c": _nothing},
        abort=_nothing,
    )
    add_task(registry, Old)
    harness = (await open_tasks(storage, [], registry=registry)).harness
    refs["harness"] = harness
    id = await create_in(harness, Old)
    harness.resume()
    await eventually(lambda: "runtime" in refs)
    # The same extension name replaces the old one in place.
    add_task(registry, handover_task("new", 1, log))
    resolve(gate)
    await eventually(lambda: "held" in refs)
    await refs["held"].entered
    await flush()
    # Queued behind the handover commit while the invocation has not ended yet.
    late = asyncio.ensure_future(refs["runtime"].commit(lambda _tx, _current: completed(None), context))
    refs["held"].release()
    with pytest.raises(Exception, match="invocation has ended"):
        await late
    await harness.wait_for_task(id, context)
    assert log == ["new:b start", "new:b end", "new:c"]
    await harness.close(context)


async def test_preserves_an_abort_mark_that_races_the_handover_commit_the_new_definition_aborts() -> None:
    log: list[str] = []
    gate = deferred()
    storage = controlled()
    held: list[Any] = []

    # The progress commit landed; hold the next commit, the handover, and queue the abort mark behind it.
    def on_end() -> None:
        if not held:
            held.append(storage.hold_commits())

    opened = await start_handover(log, Gates(a=gate, on_end=on_end), storage)
    harness, id = opened.harness, opened.id
    # The same extension name replaces the old one in place.
    add_task(opened.registry, handover_task("new", 1, log))
    resolve(gate)
    await eventually(lambda: len(held) > 0)
    await held[0].entered
    aborting = harness.abort_task(id, context)
    held[0].release()
    assert await aborting == "marked"
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "new"}
    states = [
        f"{write['value']['state']['status']}{'+mark' if write['value'].get('abortRequested') else ''}"
        for writes in storage.commits
        for write in writes
        if write["type"] == "task" and write["value"]["id"] == id  # type: ignore[typeddict-item]
    ]
    # created, reserved, progress, handover, mark, abort reservation, aborted
    assert states == [
        "pending",
        "running",
        "running",
        "pending",
        "pending+mark",
        "running+mark",
        "terminal+mark",
    ]
    assert log == ["old:a start", "old:a end", "new:abort"]
    await harness.close(context)


# ─── Harness open ───────────────────────────────────────────────────────────


async def test_releases_its_registry_subscription_and_closes_the_session_when_open_fails() -> None:
    storage = controlled()
    log: list[str] = []

    async def run(*_args: Any) -> None:
        log.append("run")
        await never()

    Stuck = define_task(
        name="test.stuck", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )
    # Leave a running task behind so open has a reconciliation commit to fail.
    first = await open_tasks(storage, [Stuck])
    await create_in(first.harness, Stuck)
    first.harness.resume()
    await eventually(lambda: len(log) == 1)

    registry = create_registry()
    reader = counting_reader(registry)
    storage.fail_next_commit(RuntimeError("disk full"))
    with pytest.raises(Exception, match="disk full"):
        await Harness.open(storage, {"models": create_models(), "registry": reader}, context)
    assert reader.subscriptions() == 0
    with pytest.raises(Exception):  # noqa: B017 - upstream: rejects with any error
        await storage.task(1, context)
