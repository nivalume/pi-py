"""Port of ``test/harness-inspect.test.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from durable_harness_support import chat_setup, completed, eventually, open_chat, open_tasks, unanswered, wait_for
from durable_session_support import context, deferred, resolve

from pi_chord.context import Context
from pi_durable import MemoryStorage, define_task


async def _nothing(*_args: Any) -> None:
    return None


def task(
    name: str,
    version: int,
    *,
    gate: asyncio.Future[Any] | None = None,
    migrate: Callable[[], None] | None = None,
) -> Any:
    """A one-phase task that completes once ``gate`` resolves; ``migrate`` runs when an older stored version migrates."""

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        if gate is not None:
            await gate
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    migration: Any = None
    if migrate is not None:

        def migration(*_args: Any) -> Any:
            assert migrate is not None
            migrate()
            return {"input": None, "checkpoint": {"phase": "run"}}

    return define_task(
        name=name,
        version=version,
        initial=lambda _input: {"phase": "run"},
        phases={"run": run},
        abort=_nothing,
        migrate=migration,
    )


def state_of(tasks: Sequence[Any], id: Any) -> Any:
    return next((entry["state"] for entry in tasks if entry["record"]["id"] == id), None)


def without_error(state: Any) -> Any:
    return {key: value for key, value in state.items() if key != "error"}


async def test_derives_every_live_tasks_state_without_running_task_code() -> None:
    gate = deferred()
    Gate = task("test.gate", 1, gate=gate)
    gate_id: list[Any] = []

    async def wait(_task: Any, runtime: Any, ctx: Context) -> None:
        checkpoint = {"phase": "done"}
        await runtime.commit(
            lambda _tx, _current: {
                "status": "waiting",
                "checkpoint": checkpoint,
                "on": [gate_id[0]],
                "policy": "allSettled",
            },
            ctx,
        )

    async def done(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Dependent = define_task(
        name="test.dependent",
        version=1,
        initial=lambda _input: {"phase": "wait"},
        phases={"wait": wait, "done": done},
        abort=_nothing,
    )
    migrations = [0]

    def count_migration() -> None:
        migrations[0] += 1

    def failing_migration() -> None:
        raise RuntimeError("cannot migrate")

    registered = [
        Gate,
        Dependent,
        task("test.migrating", 2, migrate=count_migration),
        task("test.no-migration", 2),
        task("test.failing", 2, migrate=failing_migration),
        task("test.too-old", 1),
    ]
    opened = await open_tasks(MemoryStorage(), registered)
    harness = opened.harness
    root = await harness.root(context)
    conversation = {"ownership": {"kind": "conversation"}}

    async def create(tx: Any) -> dict[str, Any]:
        gate_id.append(await tx.create_task(Gate, None, conversation))
        return {
            "gate": gate_id[0],
            "dependent": await tx.create_task(Dependent, None, conversation),
            # Stored by definitions other than the registered ones.
            "migrating": await tx.create_task(task("test.migrating", 1), None, conversation),
            "noMigration": await tx.create_task(task("test.no-migration", 1), None, conversation),
            "failing": await tx.create_task(task("test.failing", 1), None, conversation),
            "tooOld": await tx.create_task(task("test.too-old", 2), None, conversation),
            "missing": await tx.create_task(task("test.missing", 1), None, conversation),
        }

    ids = await root.commit(create, context)

    paused = await harness.inspect(context)
    assert paused["scheduling"] == "paused"
    assert [entry["record"]["id"] for entry in paused["tasks"]] == list(ids.values())
    assert state_of(paused["tasks"], ids["gate"]) == {"kind": "ready", "migrates": False}
    assert state_of(paused["tasks"], ids["dependent"]) == {"kind": "ready", "migrates": False}
    assert state_of(paused["tasks"], ids["migrating"]) == {"kind": "ready", "migrates": True}
    # A migration that was never tried is not run to find out.
    assert state_of(paused["tasks"], ids["failing"]) == {"kind": "ready", "migrates": True}
    no_migration = state_of(paused["tasks"], ids["noMigration"])
    assert without_error(no_migration) == {"kind": "blocked", "reason": "migration_failed"}
    assert str(no_migration["error"]) == "Task test.no-migration version 2 has no migration from 1"
    assert state_of(paused["tasks"], ids["tooOld"]) == {"kind": "blocked", "reason": "task_too_old"}
    assert state_of(paused["tasks"], ids["missing"]) == {"kind": "blocked", "reason": "missing_task"}
    assert migrations[0] == 0
    assert (await harness.inspect(context))["scheduling"] == "paused"

    harness.resume()
    await eventually(lambda: migrations[0] == 1)
    await harness.wait_for_task(ids["migrating"], context)
    running: list[Any] = [await harness.inspect(context)]

    async def gate_running() -> bool:
        running[0] = await harness.inspect(context)
        tasks = running[0]["tasks"]
        return (state_of(tasks, ids["gate"]) or {}).get("kind") == "running" and (
            state_of(tasks, ids["dependent"]) or {}
        ).get("kind") == "waiting"

    await wait_for(gate_running)
    tasks = running[0]["tasks"]
    assert state_of(tasks, ids["dependent"]) == {"kind": "waiting", "on": [ids["gate"]]}
    assert running[0]["scheduling"] == "running"
    assert state_of(tasks, ids["gate"]) == {"kind": "running"}
    assert state_of(tasks, ids["migrating"]) is None
    failing = state_of(tasks, ids["failing"])
    assert without_error(failing) == {"kind": "blocked", "reason": "migration_failed"}
    assert type(failing["error"]) is RuntimeError and str(failing["error"]) == "cannot migrate"

    resolve(gate)
    await harness.wait_for_task(ids["dependent"], context)
    settled = await harness.inspect(context)
    assert [entry["record"]["id"] for entry in settled["tasks"]] == [
        ids["noMigration"],
        ids["failing"],
        ids["tooOld"],
        ids["missing"],
    ]
    await harness.close(context)


async def test_lists_unsettled_submissions() -> None:
    setup = chat_setup()
    busy = unanswered()
    setup.faux.set_responses([busy.step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    await other.submit({"type": "write", "entry": {"kind": "note"}}, context)
    input = await root.submit({"type": "input", "content": "hi"}, context)
    await busy.reached

    inspection = await harness.inspect(context)
    assert inspection["submissions"] == [await input.status(context)]
    assert [(entry["record"]["kind"], entry["state"]["kind"]) for entry in inspection["tasks"]] == [
        ("pi.generation", "running")
    ]
    await harness.close(context)
