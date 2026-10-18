from __future__ import annotations

import asyncio

import pytest

from pi_ai import AssistantMessage, AssistantMessageEventStream, DoneEvent, ErrorEvent, EventStream, Usage, now_ms


def _int_stream(complete_on: int | None = None) -> EventStream[int, int]:
    return EventStream(lambda event: complete_on is not None and event == complete_on, lambda event: event)


# Regression tests for https://github.com/earendil-works/pi/issues/9055
async def test_drains_buffered_events_and_ignores_events_after_completion() -> None:
    stream = _int_stream(3)
    for value in (1, 2, 3, 4):
        stream.push(value)
    assert await stream.result() == 3
    assert [event async for event in stream] == [1, 2, 3]


async def test_preserves_order_when_events_arrive_after_draining_starts() -> None:
    stream = _int_stream()
    stream.push(1)
    stream.push(2)
    iterator = aiter(stream)
    assert await anext(iterator) == 1
    stream.push(3)
    assert await anext(iterator) == 2
    assert await anext(iterator) == 3
    stream.end(3)
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


async def test_delivers_events_to_waiting_consumers_in_registration_order() -> None:
    stream = _int_stream()
    first = asyncio.ensure_future(anext(aiter(stream)))
    second = asyncio.ensure_future(anext(aiter(stream)))
    await asyncio.sleep(0)
    stream.push(1)
    stream.push(2)
    assert await first == 1
    assert await second == 2


async def test_drains_buffered_events_after_end_and_resolves_explicit_result() -> None:
    stream: EventStream[int, str] = EventStream(lambda _event: False, lambda event: str(event))
    stream.push(1)
    stream.push(2)
    stream.end("complete")
    assert await stream.result() == "complete"
    assert [event async for event in stream] == [1, 2]


async def test_wakes_all_waiting_consumers_when_ended_without_result() -> None:
    stream = _int_stream()
    first = asyncio.ensure_future(anext(aiter(stream)))
    second = asyncio.ensure_future(anext(aiter(stream)))
    await asyncio.sleep(0)
    stream.end()
    for waiter in (first, second):
        with pytest.raises(StopAsyncIteration):
            await waiter


async def test_result_raises_when_ended_without_result() -> None:
    stream = _int_stream()
    stream.end()
    with pytest.raises(RuntimeError, match="without a result"):
        await stream.result()


async def test_producer_crash_surfaces_from_result_and_iteration() -> None:
    stream = _int_stream()

    async def produce() -> None:
        stream.push(1)
        raise ValueError("boom")

    stream.start(produce())
    assert await anext(aiter(stream)) == 1
    with pytest.raises(ValueError, match="boom"):
        await anext(aiter(stream))
    with pytest.raises(ValueError, match="boom"):
        await stream.result()


def test_stream_created_outside_event_loop_starts_on_consumption() -> None:
    stream = _int_stream(2)

    async def produce() -> None:
        stream.push(1)
        stream.push(2)

    stream.start(produce())

    async def consume() -> list[int]:
        return [event async for event in stream]

    assert asyncio.run(consume()) == [1, 2]


async def test_aclose_cancels_producer() -> None:
    stream = _int_stream()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def produce() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stream.start(produce())
    await started.wait()
    await stream.aclose()
    assert cancelled.is_set()
    assert [event async for event in stream] == []


async def test_forward_to_moves_queued_and_future_events() -> None:
    inner = _int_stream(3)
    outer = _int_stream(3)
    inner.push(1)
    inner.forward_to(outer)
    inner.push(2)
    inner.push(3)
    assert [event async for event in outer] == [1, 2, 3]
    assert await outer.result() == 3


def _message(timestamp: int, duration_ms: int | None = None) -> AssistantMessage:
    return AssistantMessage(
        [],
        api="openai-responses",
        provider="openai",
        model="m",
        usage=Usage(),
        timestamp=timestamp,
        duration_ms=duration_ms,
    )


async def test_sets_duration_on_final_message_of_a_response_it_saw_start() -> None:
    done = AssistantMessageEventStream()
    answer = _message(now_ms())
    await asyncio.sleep(0.02)
    done.push(DoneEvent("stop", answer))
    assert answer.duration_ms is not None and answer.duration_ms >= 15
    assert (await done.result()).duration_ms == answer.duration_ms

    failed = AssistantMessageEventStream()
    error = _message(now_ms())
    error.stop_reason = "error"
    failed.push(ErrorEvent("error", error))
    assert error.duration_ms is not None and error.duration_ms >= 0

    ended = AssistantMessageEventStream()
    result = _message(now_ms())
    ended.end(result)
    assert result.duration_ms is not None and result.duration_ms >= 0


async def test_keeps_existing_duration_so_forwarding_keeps_inner_measurement() -> None:
    outer = AssistantMessageEventStream()
    await asyncio.sleep(0.02)
    inner = AssistantMessageEventStream()
    answer = _message(now_ms())
    inner.push(DoneEvent("stop", answer))
    measured = answer.duration_ms
    outer.push(DoneEvent("stop", answer))
    assert answer.duration_ms == measured
    assert measured is not None and measured < 20

    preset = _message(now_ms(), 1234)
    AssistantMessageEventStream().push(DoneEvent("stop", preset))
    assert preset.duration_ms == 1234


def test_leaves_message_untimed_when_it_started_before_the_stream() -> None:
    fetched = _message(now_ms() - 60_000)
    AssistantMessageEventStream().push(DoneEvent("stop", fetched))
    assert fetched.duration_ms is None


def test_does_not_time_message_pushed_after_completion() -> None:
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent("stop", _message(now_ms())))
    late = _message(now_ms())
    stream.push(DoneEvent("stop", late))
    assert late.duration_ms is None
