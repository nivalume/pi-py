"""Port of chord ``test/context.test.ts``."""

from __future__ import annotations

import asyncio

import pytest

from pi_ai import AbortController, AbortError
from pi_chord import (
    BACKGROUND_CONTEXT,
    TODO_CONTEXT,
    await_with_context,
    create_context_key,
    with_abort_signal,
    with_cancel,
    with_context_value,
    without_abort_signal,
)


def test_provides_distinct_empty_root_contexts() -> None:
    key = create_context_key("value")
    assert TODO_CONTEXT is not BACKGROUND_CONTEXT
    assert TODO_CONTEXT.abort_signal is None
    assert TODO_CONTEXT.value(key) is None
    assert str(BACKGROUND_CONTEXT) == "[Context BACKGROUND_CONTEXT]"
    assert str(TODO_CONTEXT) == "[Context TODO_CONTEXT]"


def test_layers_typed_values_without_modifying_parents() -> None:
    first_key = create_context_key("first")
    second_key = create_context_key("second")
    first = with_context_value(first_key, "one", BACKGROUND_CONTEXT)
    second = with_context_value(second_key, 2, first)
    replaced = with_context_value(first_key, "updated", second)

    assert BACKGROUND_CONTEXT.value(first_key) is None
    assert first.value(first_key) == "one"
    assert first.value(second_key) is None
    assert second.value(first_key) == "one"
    assert second.value(second_key) == 2
    assert replaced.value(first_key) == "updated"
    assert second.value(first_key) == "one"
    assert str(replaced) == "[Context BACKGROUND_CONTEXT].WithValue(first).WithValue(second).WithValue(first)"


def test_keys_with_the_same_description_are_distinct() -> None:
    a = create_context_key("same")
    b = create_context_key("same")
    context = with_context_value(a, 1, BACKGROUND_CONTEXT)
    assert context.value(a) == 1
    assert context.value(b) is None


def test_inherits_parent_cancellation_and_isolates_child_cancellation() -> None:
    parent_controller = AbortController()
    parent = with_abort_signal(parent_controller.signal, BACKGROUND_CONTEXT)
    child = with_cancel(parent)
    sibling = with_cancel(parent)
    calls: list[object] = []
    assert child.context.abort_signal is not None
    child.context.abort_signal.add_listener(lambda: calls.append(1))

    child.cancel("child")
    assert child.context.abort_signal.aborted
    assert str(child.context.abort_signal.reason) == "child"
    assert sibling.context.abort_signal is not None and not sibling.context.abort_signal.aborted
    assert parent.abort_signal is not None and not parent.abort_signal.aborted
    assert calls == [1]

    parent_controller.abort("parent")
    assert sibling.context.abort_signal.aborted
    assert str(sibling.context.abort_signal.reason) == "parent"
    with pytest.raises(AbortError, match="parent"):
        sibling.context.throw_if_aborted()


def test_masks_caller_cancellation_for_mandatory_cleanup() -> None:
    controller = AbortController()
    key = create_context_key("value")
    context = with_context_value(key, "preserved", with_abort_signal(controller.signal, BACKGROUND_CONTEXT))
    cleanup = without_abort_signal(context)

    controller.abort()
    assert context.abort_signal is not None and context.abort_signal.aborted
    assert cleanup.abort_signal is None
    assert cleanup.value(key) == "preserved"
    cleanup.throw_if_aborted()


async def test_stops_waiting_when_the_invocation_is_cancelled() -> None:
    controller = AbortController()
    context = with_abort_signal(controller.signal, BACKGROUND_CONTEXT)
    loop = asyncio.get_running_loop()
    work: asyncio.Future[str] = loop.create_future()
    waiting = asyncio.ensure_future(await_with_context(work, context))
    await asyncio.sleep(0)
    cancellation = RuntimeError("cancelled")

    controller.abort(cancellation)
    with pytest.raises(RuntimeError) as raised:
        await waiting
    assert raised.value is cancellation
    work.set_result("completed later")
    assert await work == "completed later"

    async def completed() -> str:
        return "completed"

    assert await await_with_context(completed(), BACKGROUND_CONTEXT) == "completed"


async def test_await_with_context_keeps_the_abandoned_coroutine_running() -> None:
    controller = AbortController()
    context = with_abort_signal(controller.signal, BACKGROUND_CONTEXT)
    finished = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.01)
        finished.set()

    asyncio.get_running_loop().call_soon(controller.abort)
    with pytest.raises(AbortError):
        await await_with_context(work(), context)
    await asyncio.wait_for(finished.wait(), 1)


async def test_await_with_context_raises_immediately_for_cancelled_contexts() -> None:
    controller = AbortController()
    controller.abort()

    async def never() -> None:  # closed without running
        raise AssertionError

    with pytest.raises(AbortError):
        await await_with_context(never(), with_abort_signal(controller.signal, BACKGROUND_CONTEXT))
