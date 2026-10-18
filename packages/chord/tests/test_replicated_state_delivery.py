"""Port of chord ``test/replicated-state-delivery.test.ts``.

Awaitable listener results are asyncio futures or coroutines; coroutines start eagerly
(Python 3.12+), like JavaScript async functions running to their first ``await``.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

import pytest

from pi_chord import BACKGROUND_CONTEXT, Context, with_cancel
from pi_chord import state as state_module
from pi_chord.state import (
    ReplicatedStateDelivery,
    ReplicatedStateSnapshot,
    ReplicatedStateSourceFrame,
    ReplicatedStateSourceOptions,
    replicated_state,
)

KINDS = ["mutable", "attached"]


def fixture(kind: str, on_error: Callable[[Exception], None] = lambda _error: None) -> tuple[Any, Any]:
    if kind == "mutable":
        mutable = replicated_state({"value": 0})
        return mutable, lambda value, context: mutable.replace(context, value)

    receive: list[Callable[[ReplicatedStateSourceFrame[Any]], None]] = []

    class Attachment:
        snapshot = ReplicatedStateSnapshot({"value": 0}, 0)

        def activate(self, listener: Callable[[ReplicatedStateSourceFrame[Any]], None]) -> None:
            receive.append(listener)

        def dispose(self) -> None:
            pass

    class Source:
        def attach(self) -> Attachment:
            return Attachment()

    attached = replicated_state(Source(), ReplicatedStateSourceOptions(on_error=on_error))

    def publish(value: Any, context: Context) -> None:
        receive[0](ReplicatedStateSourceFrame(value["value"], value, [("s", ("value",), value["value"])], context))

    return attached, publish


async def wait_for(check: Callable[[], bool]) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0)
    assert check()


eager = pytest.mark.skipif(sys.version_info < (3, 12), reason="coroutines start eagerly on Python 3.12+")


@eager
@pytest.mark.parametrize("kind", KINDS)
async def test_awaits_hydration_and_each_update_independently_of_other_subscribers(kind: str) -> None:
    state, publish = fixture(kind)
    loop = asyncio.get_running_loop()
    hydration: asyncio.Future[None] = loop.create_future()
    update: asyncio.Future[None] = loop.create_future()
    events: list[str] = []
    fast: list[int] = []

    async def slow(value: Any, _context: Context, _delivery: Any) -> None:
        events.append(f"start:{value['value']}")
        if value["value"] == 0:
            await hydration
        if value["value"] == 1:
            await update
        events.append(f"end:{value['value']}")

    state.subscribe(slow)
    state.subscribe(lambda value, _context, _delivery: fast.append(value["value"]))
    publish({"value": 1}, BACKGROUND_CONTEXT)
    publish({"value": 2}, BACKGROUND_CONTEXT)
    assert events == ["start:0"]
    assert fast == [0, 1, 2]
    assert state.value == {"value": 2}
    hydration.set_result(None)
    await wait_for(lambda: events == ["start:0", "end:0", "start:1"])
    update.set_result(None)
    await wait_for(lambda: events == ["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("count", [100, 101, 102, 201, 202])
async def test_bounds_pending_deliveries(kind: str, count: int) -> None:
    state, publish = fixture(kind)
    hydration: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    received: list[int] = []

    def listener(value: Any, _context: Context, _delivery: Any) -> Any:
        received.append(value["value"])
        return hydration if value["value"] == 0 else None

    state.subscribe(listener)
    for value in range(1, count + 1):
        publish({"value": value}, BACKGROUND_CONTEXT)
    assert received == [0]
    hydration.set_result(None)
    await asyncio.sleep(0)  # one loop turn, like awaiting the settled promise upstream
    first = (count - 1) // 100 * 100 + 1
    assert received == [0, *range(first, count + 1)]


@pytest.mark.parametrize("kind", KINDS)
async def test_excludes_a_running_update_from_overflow_and_retains_exact_value_context_delivery(kind: str) -> None:
    state, publish = fixture(kind)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    context = with_cancel(BACKGROUND_CONTEXT).context
    received: list[tuple[Any, Context, ReplicatedStateDelivery]] = []

    def listener(value: Any, context: Context, delivery: ReplicatedStateDelivery) -> Any:
        received.append((value, context, delivery))
        return gate if value["value"] == 1 else None

    state.subscribe(listener)
    for value in range(1, 102):
        publish({"value": value}, BACKGROUND_CONTEXT)
    newest = {"value": 102}
    publish(newest, context)
    adopted = state.value
    publish({"value": 103}, BACKGROUND_CONTEXT)
    assert [value["value"] for value, _, _ in received] == [0, 1]
    gate.set_result(None)
    await asyncio.sleep(0)
    assert [value["value"] for value, _, _ in received] == [0, 1, 102, 103]
    assert received[2][0] is adopted
    assert received[2][1] is context
    assert received[2][2] == ReplicatedStateDelivery("update", 102)


@pytest.mark.parametrize("kind", KINDS)
def test_serializes_reentrant_hydration_and_update_callbacks(kind: str) -> None:
    state, publish = fixture(kind)
    events: list[str] = []

    def listener(value: Any, _context: Context, _delivery: Any) -> None:
        events.append(f"start:{value['value']}")
        if value["value"] < 2:
            publish({"value": value["value"] + 1}, BACKGROUND_CONTEXT)
        events.append(f"end:{value['value']}")

    state.subscribe(listener)
    assert events == ["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"]


@pytest.mark.parametrize("kind", KINDS)
async def test_treats_two_subscriptions_of_the_same_callback_independently(kind: str) -> None:
    state, publish = fixture(kind)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    received: list[int] = []

    def listener(value: Any, _context: Context, _delivery: Any) -> Any:
        received.append(value["value"])
        return gate if value["value"] == 0 else None

    stop_first = state.subscribe(listener)
    stop_second = state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    stop_first()
    stop_first()
    gate.set_result(None)
    await asyncio.sleep(0)
    assert received == [0, 0, 1]
    stop_second()


@eager
@pytest.mark.parametrize("kind", KINDS)
async def test_unsubscribe_drops_queued_callbacks_without_joining_or_aborting_the_running_callback(kind: str) -> None:
    state, publish = fixture(kind)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    context = with_cancel(BACKGROUND_CONTEXT).context
    received: list[int] = []
    completed = False

    async def listener(value: Any, _context: Context, _delivery: Any) -> None:
        nonlocal completed
        received.append(value["value"])
        if value["value"] == 1:
            await gate
            completed = True

    stop = state.subscribe(listener)
    await asyncio.sleep(0)
    publish({"value": 1}, context)
    publish({"value": 2}, context)
    assert stop() is None
    publish({"value": 3}, context)
    assert context.abort_signal is not None
    assert not context.abort_signal.aborted
    assert not completed
    gate.set_result(None)
    await wait_for(lambda: completed)
    assert received == [0, 1]


# ─── attached listener errors ────────────────────────────────────────────────


def test_isolates_a_synchronous_hydration_failure_without_removing_the_subscription() -> None:
    errors: list[Exception] = []
    state, publish = fixture("attached", errors.append)
    received: list[int] = []
    failure = RuntimeError("sync hydration")

    def listener(value: Any, _context: Context, _delivery: Any) -> None:
        received.append(value["value"])
        if value["value"] == 0:
            raise failure

    state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    assert errors == [failure]
    assert received == [0, 1]


async def test_observes_hydration_rejection_synchronous_raise_and_update_rejection_while_continuing() -> None:
    errors: list[Exception] = []
    state, publish = fixture("attached", errors.append)
    loop = asyncio.get_running_loop()
    gate: asyncio.Future[None] = loop.create_future()
    received: list[int] = []
    fast: list[int] = []

    def listener(value: Any, _context: Context, _delivery: Any) -> Any:
        received.append(value["value"])
        if value["value"] == 0:
            return gate
        if value["value"] == 1:
            raise RuntimeError("sync update")
        if value["value"] == 2:
            rejected: asyncio.Future[None] = loop.create_future()
            rejected.set_exception(RuntimeError("async update"))
            return rejected
        return None

    state.subscribe(listener)
    state.subscribe(lambda value, _context, _delivery: fast.append(value["value"]))
    for value in range(1, 4):
        publish({"value": value}, BACKGROUND_CONTEXT)
    gate.set_exception(RuntimeError("async hydration"))
    await wait_for(lambda: received == [0, 1, 2, 3])
    assert [str(error) for error in errors] == ["async hydration", "sync update", "async update"]
    assert fast == [0, 1, 2, 3]


async def test_still_observes_a_callback_rejection_after_unsubscribe() -> None:
    errors: list[Exception] = []
    state, publish = fixture("attached", errors.append)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    received: list[int] = []

    def listener(value: Any, _context: Context, _delivery: Any) -> Any:
        received.append(value["value"])
        return gate

    stop = state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    stop()
    failure = RuntimeError("stopped callback")
    gate.set_exception(failure)
    await wait_for(lambda: errors == [failure])
    assert received == [0]


async def test_mutable_state_reports_rejected_callbacks_through_its_default_error_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports: list[Exception] = []
    monkeypatch.setattr(state_module, "_report_error_async", reports.append)
    state = replicated_state({"value": 0})
    failure = RuntimeError("async mutable hydration")
    received: list[int] = []

    async def listener(value: Any, _context: Context, _delivery: Any) -> None:
        received.append(value["value"])
        if value["value"] == 0:
            raise failure

    state.subscribe(listener)
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    await wait_for(lambda: received == [0, 1])
    assert reports == [failure]


async def test_default_reporter_hands_errors_to_the_loop_exception_handler() -> None:
    loop = asyncio.get_running_loop()
    seen: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    try:
        state = replicated_state({"value": 0})

        def failing(_value: Any, _context: Context, delivery: Any) -> None:
            if delivery.kind == "update":
                raise RuntimeError("reported")

        state.subscribe(failing)
        state.replace(BACKGROUND_CONTEXT, {"value": 1})
    finally:
        loop.set_exception_handler(previous)
    assert [str(entry["exception"]) for entry in seen] == ["reported"]


def test_reports_async_listeners_without_a_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    reports: list[Exception] = []
    monkeypatch.setattr(state_module, "_report_error_async", reports.append)
    state = replicated_state({"value": 0})
    received: list[int] = []

    async def listener(value: Any, _context: Context, _delivery: Any) -> None:
        received.append(value["value"])

    state.subscribe(listener)
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    assert len(reports) == 2
    assert all("running event loop" in str(report) for report in reports)
