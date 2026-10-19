"""Port of ``test/harness-sleep.test.ts``.

Adaptation: SQLite runs on executor threads here, so waits on it poll in real time (``wait_for``); upstream's
``node:sqlite`` is synchronous, so a fixed number of turns (``eventually``) suffices there.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from durable_harness_support import aborted, aborted_with, completed, eventually, match_object, open_tasks, wait_for
from durable_session_support import context, flush

from pi_chord.context import Context
from pi_durable import MemoryStorage, define_extension, define_task
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


@dataclass(slots=True)
class Runs:
    after: int = 0


def nap_task(delay: float, runs: Runs | None = None) -> Any:
    """Sleeps ``delay`` ms on the Harness clock, then completes with the time it resumed at."""
    counts = runs if runs is not None else Runs()

    async def start(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.sleep_until(runtime.now() + delay, {"phase": "after", "slept": delay}, ctx)

    async def after(_task: Any, runtime: Any, ctx: Context) -> None:
        counts.after += 1
        resumed = runtime.now()
        await runtime.commit(lambda _tx, _current: completed(resumed), ctx)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: aborted_with("test"), ctx)

    return define_task(
        name="test.nap",
        version=1,
        initial=lambda _input: {"phase": "start"},
        phases={"start": start, "after": after},
        abort=abort,
    )


async def start_nap(harness: Any, task: Any) -> Any:
    root = await harness.root(context)
    return await root.commit(lambda tx: tx.create_task(task, None, {"ownership": {"kind": "conversation"}}), context)


async def state_of(harness: Any, id: Any) -> Any:
    record = await harness.get_task(id, context)
    return None if record is None else record["state"]


async def status_of(harness: Any, id: Any) -> Any:
    state = await state_of(harness, id)
    return None if state is None else state["status"]


@dataclass(slots=True, eq=False)
class FakeTimer:
    delay_ms: float
    callback: Callable[[], None]
    disarmed: bool = False


@dataclass(slots=True)
class FakeTimers:
    """A host ``set_timer`` that records the timers that pace work and declines background ones."""

    armed: list[FakeTimer] = field(default_factory=list)

    def set_timer(self, callback: Callable[[], None], delay_ms: float, options: Any) -> Any:
        if options["background"]:
            return None
        timer = FakeTimer(delay_ms, callback)
        self.armed.append(timer)

        def disarm() -> None:
            timer.disarmed = True

        return disarm

    def live(self) -> list[FakeTimer]:
        return [timer for timer in self.armed if not timer.disarmed]


async def settle() -> None:
    """Let callbacks and a few timer turns run."""
    await flush()
    await asyncio.sleep(0.02)
    await flush()


class Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


async def _nothing(*_args: Any) -> None:
    return None


async def test_sleeps_without_an_invocation_and_resumes_at_its_checkpoint_once_due() -> None:
    clock = Clock(1_000)
    wakes: list[Any] = []
    runs = Runs()
    Nap = nap_task(1_000, runs)
    harness = (await open_tasks(MemoryStorage(), [Nap], now=clock, on_wake=wakes.append)).harness
    graph = await harness.task_graph(context)
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: _is(status_of(harness, id), "sleeping"))
    assert await state_of(harness, id) == {
        "status": "sleeping",
        "checkpoint": {"phase": "after", "slept": 1_000},
        "until": 2_000,
    }
    inspection = await harness.inspect(context)
    assert next(task["state"] for task in inspection["tasks"] if task["record"]["id"] == id) == {
        "kind": "sleeping",
        "until": 2_000,
    }
    assert graph.value["tasks"][str(id)]["state"] == {"status": "sleeping", "phase": "after", "until": 2_000}
    await eventually(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False, "at": 2_000}
    await settle()
    assert wakes[-1] == {"active": False, "at": 2_000}

    # Early and repeated wakes run nothing.
    clock.value = 1_999
    harness.resume()
    harness.resume()
    await settle()
    assert runs.after == 0
    assert await status_of(harness, id) == "sleeping"

    clock.value = 2_000
    harness.resume()
    harness.resume()
    receipt = await harness.wait_for_task(id, context)
    assert receipt["state"]["outcome"] == {"status": "completed", "result": 2_000}
    assert runs.after == 1
    await settle()
    await eventually(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False}
    assert wakes[-1] == {"active": False}
    # Only changes: idle at open, running, asleep until 2000, running again, idle.
    for index in range(1, len(wakes)):
        assert wakes[index] != wakes[index - 1]
    assert {"active": True} in wakes
    assert {"active": False, "at": 2_000} in wakes
    graph.dispose()
    await harness.close(context)


async def _is(awaitable: Any, expected: Any) -> bool:
    return (await awaitable) == expected


async def test_keeps_a_sleep_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    clock = Clock(1_000)
    Nap = nap_task(60_000)

    async def open(storage: Any) -> Any:
        return (await open_tasks(storage, [Nap], now=clock, on_wake=lambda _wake: None)).harness

    harness = await open(await open_sqlite_storage(path))
    id = await start_nap(harness, Nap)
    harness.resume()
    await wait_for(lambda: _is(status_of(harness, id), "sleeping"))
    await harness.close(context)

    harness = await open(await open_sqlite_storage(path))
    # Known from storage before scheduling starts, once open's recovery pass is done.
    await wait_for(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False, "at": 61_000}
    harness.resume()
    await settle()
    assert await status_of(harness, id) == "sleeping"
    await harness.close(context)

    clock.value = 61_000
    harness = await open(await open_sqlite_storage(path))
    assert harness.wake() == {"active": True}
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "completed", "result": 61_000}
    await harness.close(context)


async def test_resumes_a_due_sleep_with_its_timer_without_on_wake_with_background_timers_declined() -> None:
    clock = Clock(1_000)
    timers = FakeTimers()
    Nap = nap_task(500)
    harness = (await open_tasks(MemoryStorage(), [Nap], now=clock, set_timer=timers.set_timer)).harness
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: any(timer.delay_ms == 500 for timer in timers.live()))
    assert await status_of(harness, id) == "sleeping"
    clock.value = 1_500
    next(timer for timer in timers.live() if timer.delay_ms == 500).callback()
    # Polled: `wait_for_task()` would call `resume()`, which wakes due sleeps itself.
    await eventually(lambda: _is(status_of(harness, id), "terminal"))
    assert match_object(await state_of(harness, id), {"outcome": {"status": "completed", "result": 1_500}})
    await harness.close(context)


async def test_runs_a_due_sleep_at_any_scheduling_pass_before_its_timer_fires() -> None:
    clock = Clock(1_000)
    timers = FakeTimers()
    Nap = nap_task(500)
    opened = await open_tasks(MemoryStorage(), [Nap], now=clock, set_timer=timers.set_timer)
    harness, registry = opened.harness, opened.registry
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: _is(status_of(harness, id), "sleeping"))
    # The deadline passes, and an unrelated change starts a pass before the timer fires.
    clock.value = 1_500
    registry.install(define_extension("unrelated"))
    # Polled: `wait_for_task()` would call `resume()`, which wakes due sleeps itself.
    await eventually(lambda: _is(status_of(harness, id), "terminal"))
    assert match_object(await state_of(harness, id), {"outcome": {"status": "completed", "result": 1_500}})
    await harness.close(context)


async def test_disarms_the_sleep_timer_when_the_sleep_is_aborted_and_when_the_harness_closes() -> None:
    timers = FakeTimers()
    Short = nap_task(500)
    harness = (await open_tasks(MemoryStorage(), [Short], now=lambda: 1_000, set_timer=timers.set_timer)).harness
    aborted_id = await start_nap(harness, Short)
    harness.resume()
    await eventually(lambda: any(timer.delay_ms == 500 for timer in timers.live()))
    await harness.abort_task(aborted_id, context)
    await harness.wait_for_task(aborted_id, context)
    await eventually(lambda: len(timers.live()) == 0)
    kept = await start_nap(harness, Short)
    await eventually(lambda: _is(status_of(harness, kept), "sleeping"))
    assert [timer.delay_ms for timer in timers.live()] == [500]
    await harness.close(context)
    assert timers.live() == []


async def test_disarms_an_invocations_timers_when_it_ends_and_a_late_fire_runs_nothing() -> None:
    fired: list[str] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        runtime.set_timer(lambda: fired.append("late"), 60_000)
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Timed = define_task(
        name="test.timed", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )
    timers = FakeTimers()
    harness = (await open_tasks(MemoryStorage(), [Timed], set_timer=timers.set_timer)).harness
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Timed, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await harness.wait_for_task(id, context)
    timer = next(armed for armed in timers.armed if armed.delay_ms == 60_000)
    await eventually(lambda: timer.disarmed)
    # A host timer that fires anyway: the ended invocation's callback does not run.
    timer.callback()
    assert fired == []
    await harness.close(context)


async def test_arms_no_timer_for_sleeps_with_on_wake() -> None:
    clock = Clock(1_000)
    timers = FakeTimers()
    Nap = nap_task(500)
    harness = (
        await open_tasks(MemoryStorage(), [Nap], now=clock, set_timer=timers.set_timer, on_wake=lambda _wake: None)
    ).harness
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: _is(status_of(harness, id), "sleeping"))
    await settle()
    assert timers.armed == []
    clock.value = 1_500
    assert harness.wake() == {"active": True}
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_arms_a_foreground_timer_the_host_declines_with_the_loop_timer_reporting_it_once() -> None:
    Nap = nap_task(10)
    opened = await open_tasks(MemoryStorage(), [Nap], set_timer=lambda *_args: None)
    harness, reports = opened.harness, opened.reports
    first = await start_nap(harness, Nap)
    second = await start_nap(harness, Nap)
    harness.resume()
    assert (await harness.wait_for_task(first, context))["state"]["outcome"]["status"] == "completed"
    assert (await harness.wait_for_task(second, context))["state"]["outcome"]["status"] == "completed"
    assert [str(error) for error in reports] == [
        "HarnessOptions.setTimer declined a timer that paces work; using the event loop's call_later"
    ]
    await harness.close(context)


async def test_counts_neither_a_wait_on_a_sleeping_task_nor_an_outcome_held_over_one_as_active() -> None:
    Child = nap_task(500)

    async def start(task: Any, runtime: Any, ctx: Context) -> None:
        async def change(tx: Any, _current: Any) -> Any:
            child = await tx.create_task(Child, None, {"ownership": {"kind": "task", "taskId": task["id"]}})
            # Held: completes next, holding its outcome over the sleeping child. Otherwise waits for it.
            if task["input"]["hold"]:
                return {"status": "running", "checkpoint": {"phase": "joined", "child": child}}
            return {
                "status": "waiting",
                "checkpoint": {"phase": "joined", "child": child},
                "on": [child],
                "policy": "allSettled",
            }

        await runtime.commit(change, ctx)

    async def joined(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Parent = define_task(
        name="test.parent",
        version=1,
        initial=lambda _input: {"phase": "start"},
        phases={"start": start, "joined": joined},
        abort=_nothing,
    )
    for hold in [False, True]:
        harness = (
            await open_tasks(MemoryStorage(), [Parent, Child], now=lambda: 1_000, on_wake=lambda _wake: None)
        ).harness
        root = await harness.root(context)
        id = await root.commit(
            lambda tx, hold=hold: tx.create_task(Parent, {"hold": hold}, {"ownership": {"kind": "conversation"}}),
            context,
        )
        harness.resume()
        parked = "completing" if hold else "waiting"
        await eventually(lambda id=id, parked=parked, harness=harness: _is(status_of(harness, id), parked))
        await eventually(lambda harness=harness: harness.wake().get("at") == 1_500)
        await eventually(lambda harness=harness: not harness.wake()["active"])
        assert harness.wake() == {"active": False, "at": 1_500}
        await harness.close(context)


async def test_runs_the_abort_handler_of_a_sleeping_task_without_waiting_for_its_time() -> None:
    Nap = nap_task(60_000)
    harness = (await open_tasks(MemoryStorage(), [Nap], now=lambda: 1_000, on_wake=lambda _wake: None)).harness
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: _is(status_of(harness, id), "sleeping"))
    await harness.abort_task(id, context)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"] == {"status": "aborted", "reason": "test"}
    await harness.close(context)


async def test_rejects_a_sleep_until_something_that_is_not_a_time_and_a_sleep_in_an_abort_handler() -> None:
    errors: list[str] = []

    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        for until in [math.nan, -1, math.inf]:
            try:
                await runtime.sleep_until(until, {"phase": "next"}, ctx)
            except Exception as error:
                errors.append(str(error))
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Invalid = define_task(
        name="test.invalid-sleep",
        version=1,
        initial=lambda _input: {"phase": "run"},
        phases={"run": run, "next": _nothing},
        abort=_nothing,
    )

    async def wait_aborted(_task: Any, runtime: Any, _ctx: Context) -> None:
        await aborted(runtime.signal)

    async def abort(_task: Any, runtime: Any, ctx: Context) -> None:
        try:
            await runtime.sleep_until(runtime.now() + 10, {"phase": "run"}, ctx)
        except Exception as error:
            errors.append(str(error))
        await runtime.commit(lambda _tx, _current: aborted_with("test"), ctx)

    Aborting = define_task(
        name="test.sleep-in-abort",
        version=1,
        initial=lambda _input: {"phase": "run"},
        phases={"run": wait_aborted},
        abort=abort,
    )
    harness = (await open_tasks(MemoryStorage(), [Invalid, Aborting])).harness
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Invalid, None, {"ownership": {"kind": "conversation"}}), context)
    aborting = await root.commit(
        lambda tx: tx.create_task(Aborting, None, {"ownership": {"kind": "conversation"}}), context
    )
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "completed"
    await harness.abort_task(aborting, context)
    assert (await harness.wait_for_task(aborting, context))["state"]["outcome"]["status"] == "aborted"
    assert errors == [
        f"Task {id} cannot sleep until NaN: not a time in milliseconds",
        f"Task {id} cannot sleep until -1: not a time in milliseconds",
        f"Task {id} cannot sleep until Infinity: not a time in milliseconds",
        f"Abort handler of task {aborting} cannot sleep",
    ]
    await harness.close(context)


async def test_does_not_count_a_task_without_a_definition_as_active() -> None:
    Missing = nap_task(10)
    harness = (await open_tasks(MemoryStorage(), [])).harness
    root = await harness.root(context)
    # Created through a token the registry does not hold: nothing can run it.
    await root.commit(lambda tx: tx.create_task(Missing, None, {"ownership": {"kind": "conversation"}}), context)
    await eventually(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False}
    harness.resume()
    await settle()
    await eventually(lambda: not harness.wake()["active"])
    assert harness.wake() == {"active": False}
    await harness.close(context)


async def test_reports_a_throwing_on_wake_and_offers_the_same_wake_again_at_the_next_pass() -> None:
    clock = Clock(1_000)
    wakes: list[Any] = []
    failure = RuntimeError("alarm failed")
    failing = [True]

    def on_wake(wake: Any) -> None:
        wakes.append(wake)
        if failing[0] and wake.get("at") is not None:
            raise failure

    Nap = nap_task(1_000)
    opened = await open_tasks(MemoryStorage(), [Nap], now=clock, on_wake=on_wake)
    harness, registry, reports = opened.harness, opened.registry, opened.reports
    id = await start_nap(harness, Nap)
    harness.resume()
    await eventually(lambda: any(wake.get("at") == 2_000 and not wake["active"] for wake in wakes))
    assert failure in reports
    # The host recovers; an unrelated change offers the same, undelivered wake again.
    failing[0] = False
    offered = len(wakes)
    registry.install(define_extension("unrelated"))
    await eventually(lambda: len(wakes) > offered)
    assert wakes[-1] == {"active": False, "at": 2_000}
    clock.value = 2_000
    harness.resume()
    await harness.wait_for_task(id, context)
    await eventually(lambda: wakes[-1]["active"] is False and wakes[-1].get("at") is None)
    await harness.close(context)
