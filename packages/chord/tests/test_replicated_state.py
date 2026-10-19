"""Port of chord ``test/replicated-state.test.ts`` (freezing checks are JavaScript-only)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import pytest

from pi_chord import BACKGROUND_CONTEXT, Context
from pi_chord import state as state_module
from pi_chord.delta import Op
from pi_chord.state import (
    ReplicatedStateSnapshot,
    ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame,
    ReplicatedStateSourceOptions,
    replicated_state,
)

T = TypeVar("T")


def test_publishes_one_immutable_structurally_shared_revision() -> None:
    initial = {"changed": {"value": 1}, "retained": {"value": 2}}
    state = replicated_state(initial)
    previous = state.value
    deliveries: list[int] = []
    state.subscribe(lambda _value, _context, delivery: deliveries.append(delivery.sequence))

    def mutate(draft: Any) -> None:
        draft["changed"]["value"] = 3
        draft["changed"]["value"] = 4

    state.change(BACKGROUND_CONTEXT, mutate)
    assert state.value == {"changed": {"value": 4}, "retained": {"value": 2}}
    assert state.value is not previous
    assert state.value["changed"] is not previous["changed"]
    assert state.value["retained"] is previous["retained"]
    assert deliveries == [0, 1]


def test_rolls_back_callback_failures_and_revokes_escaped_drafts() -> None:
    state = replicated_state({"nested": {"value": 1}})
    previous = state.value
    escaped: list[Any] = []

    def mutate(draft: Any) -> None:
        escaped.append(draft["nested"])
        draft["nested"]["value"] = 2
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        state.change(BACKGROUND_CONTEXT, mutate)
    assert state.value is previous
    with pytest.raises(TypeError):
        escaped[0]["value"]


async def test_rejects_awaitable_callbacks_and_aborts_their_draft() -> None:
    state = replicated_state({"value": 0})
    previous = state.value

    async def mutate(draft: Any) -> None:
        draft["value"] = 1

    with pytest.raises(TypeError, match="synchronous"):
        state.change(BACKGROUND_CONTEXT, mutate)
    assert state.value is previous

    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def returns_future(draft: Any) -> asyncio.Future[None]:
        draft["value"] = 1
        return future

    with pytest.raises(TypeError, match="synchronous"):
        state.change(BACKGROUND_CONTEXT, returns_future)
    future.set_exception(RuntimeError("ignored"))
    await asyncio.sleep(0)
    assert state.value is previous


def test_rejects_nested_changes_and_replacements_without_losing_the_outer_rollback() -> None:
    state = replicated_state({"left": 0, "right": 0})

    def nested_change(draft: Any) -> None:
        draft["left"] = 1
        state.change(BACKGROUND_CONTEXT, lambda nested: nested.__setitem__("right", 2))

    with pytest.raises(RuntimeError, match="reentrantly"):
        state.change(BACKGROUND_CONTEXT, nested_change)
    assert state.value == {"left": 0, "right": 0}

    def nested_replace(_draft: Any) -> None:
        state.replace(BACKGROUND_CONTEXT, {"left": 1, "right": 2})

    with pytest.raises(RuntimeError, match="change callback"):
        state.change(BACKGROUND_CONTEXT, nested_replace)
    assert state.value == {"left": 0, "right": 0}


def test_queues_listener_triggered_changes_in_sequence_order() -> None:
    state = replicated_state({"value": 0})
    deliveries: list[dict[str, int]] = []
    late: list[Any] = []
    nested = False

    def first(_value: Any, _context: Context, delivery: Any) -> None:
        nonlocal nested
        if delivery.kind != "update" or nested:
            return
        nested = True
        state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 2))
        state.subscribe(lambda _value, _context, delivery: late.append(delivery))

    def second(value: Any, _context: Context, delivery: Any) -> None:
        if delivery.kind == "update":
            deliveries.append({"value": value["value"], "sequence": delivery.sequence})

    state.subscribe(first)
    state.subscribe(second)
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 1))
    assert deliveries == [{"value": 1, "sequence": 1}, {"value": 2, "sequence": 2}]
    assert [(entry.kind, entry.sequence) for entry in late] == [("hydrate", 2)]


def test_isolates_listener_failures_after_committing_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    reports: list[Exception] = []
    monkeypatch.setattr(state_module, "_report_error_async", reports.append)
    state = replicated_state({"value": 0})
    received: list[int] = []

    def failing(_value: Any, _context: Context, delivery: Any) -> None:
        if delivery.kind == "update":
            raise RuntimeError("listener failed")

    state.subscribe(failing)
    state.subscribe(lambda _value, _context, delivery: received.append(delivery.sequence))
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 1))
    assert state.value == {"value": 1}
    assert received == [0, 1]
    assert len(reports) == 1
    assert str(reports[0]) == "listener failed"


def test_copies_assigned_values_by_value() -> None:
    external = {"value": 1}
    state = replicated_state({"left": None, "right": None})

    def mutate(draft: Any) -> None:
        draft["left"] = external
        draft["right"] = external
        draft["left"]["value"] = 2

    state.change(BACKGROUND_CONTEXT, mutate)
    assert external["value"] == 1
    assert state.value == {"left": {"value": 2}, "right": {"value": 1}}
    assert state.value["left"] is not state.value["right"]


def test_takes_immutable_ownership_of_alias_free_replacements() -> None:
    state = replicated_state({"left": {"value": 1}, "right": {"value": 2}})
    replacement = {"left": {**state.value["left"]}, "right": {**state.value["left"]}}
    state.replace(BACKGROUND_CONTEXT, replacement)
    assert state.value is replacement
    assert state.value["left"] is not state.value["right"]
    state.change(BACKGROUND_CONTEXT, lambda draft: draft["left"].__setitem__("value", 9))
    assert state.value == {"left": {"value": 9}, "right": {"value": 1}}


def test_replaces_atomically_and_ignores_deeply_equal_replacements() -> None:
    state = replicated_state({"value": {"nested": 1}, "retained": {"nested": 2}})
    previous = state.value
    state.replace(BACKGROUND_CONTEXT, {"value": {"nested": 1}, "retained": {"nested": 2}})
    assert state.value is previous
    state.replace(BACKGROUND_CONTEXT, {**state.value, "value": {"nested": 2}})
    assert state.value == {"value": {"nested": 2}, "retained": {"nested": 2}}
    assert state.value["retained"] is previous["retained"]


def test_does_not_publish_noop_changes() -> None:
    state = replicated_state({"value": 1})
    deliveries: list[int] = []
    state.subscribe(lambda _value, _context, delivery: deliveries.append(delivery.sequence))
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 1))
    assert deliveries == [0]


# ─── authoritative sources ───────────────────────────────────────────────────


class TestSourceAttachment(Generic[T]):
    __test__ = False

    def __init__(self, snapshot: ReplicatedStateSnapshot[T], on_dispose: Callable[[], None]) -> None:
        self._snapshot = snapshot
        self._on_dispose = on_dispose
        self._buffer: list[ReplicatedStateSourceFrame[T]] = []
        self._listener: Callable[[ReplicatedStateSourceFrame[T]], None] | None = None
        self._activated = False
        self._disposed = False

    @property
    def snapshot(self) -> ReplicatedStateSnapshot[T]:
        return self._snapshot

    def activate(self, listener: Callable[[ReplicatedStateSourceFrame[T]], None]) -> None:
        if self._activated:
            raise RuntimeError("attachment is already active")
        self._activated = True
        self._listener = listener
        buffered, self._buffer = self._buffer, []
        for frame in buffered:
            listener(frame)

    def publish(self, frame: ReplicatedStateSourceFrame[T]) -> None:
        if self._disposed:
            return
        if self._listener is None:
            self._buffer.append(frame)
        else:
            self._listener(frame)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._buffer.clear()
        self._on_dispose()


class TestSource(Generic[T]):
    __test__ = False

    def __init__(self, value: T, cursor: int = 0) -> None:
        self.attachments: set[TestSourceAttachment[T]] = set()
        self.on_attach: Callable[[], None] | None = None
        self._value = value
        self._cursor = cursor

    def attach(self) -> ReplicatedStateSourceAttachment[T]:
        attachment: TestSourceAttachment[T] = TestSourceAttachment(
            ReplicatedStateSnapshot(self._value, self._cursor), lambda: self.attachments.discard(attachment)
        )
        self.attachments.add(attachment)
        if self.on_attach is not None:
            self.on_attach()
        return attachment

    def commit(
        self, value: T, ops: Sequence[Op], context: Context = BACKGROUND_CONTEXT, cursor: int | None = None
    ) -> None:
        self._value = value
        self._cursor = self._cursor + 1 if cursor is None else cursor
        frame = ReplicatedStateSourceFrame(self._cursor, value, ops, context)
        for attachment in list(self.attachments):
            attachment.publish(frame)


def test_captures_before_activation_and_drains_queued_commits_in_order() -> None:
    initial = {"value": 0}
    first = {"value": 1}
    second = {"value": 2}
    source = TestSource(initial, 10)

    def on_attach() -> None:
        source.commit(first, [("s", ("value",), 1)])
        source.commit(second, [("s", ("value",), 2)])

    source.on_attach = on_attach
    state = replicated_state(source)
    deliveries: list[tuple[Any, int]] = []
    state.subscribe(lambda value, _context, delivery: deliveries.append((value, delivery.sequence)))
    assert state.value is second
    assert deliveries == [(second, 2)]
    assert deliveries[0][0] is second


def test_hydrates_at_sequence_zero_when_attaching_after_existing_commits() -> None:
    source = TestSource({"value": 0}, 40)
    current = {"value": 1}
    source.commit(current, [("s", ("value",), 1)])
    state = replicated_state(source)
    deliveries: list[int] = []
    state.subscribe(lambda _value, _context, delivery: deliveries.append(delivery.sequence))
    assert state.value is current
    assert deliveries == [0]


def test_publishes_exact_source_values_without_applying_or_rediffing() -> None:
    source = TestSource({"value": 0})
    state = replicated_state(source)
    following = {"value": 1}
    published: list[Any] = []
    state.subscribe(lambda value, _context, delivery: published.append(value) if delivery.kind == "update" else None)
    # Deliberately not the operation that produces `following`: the source value is published as is.
    source.commit(following, [("s", ("value",), 7)])
    assert state.value is following
    assert published[0] is following


def test_buffers_reentrant_frames_and_skips_updates_covered_by_a_late_hydration() -> None:
    source = TestSource({"value": 0})
    state = replicated_state(source)
    received: list[int] = []
    late: list[tuple[str, int, int]] = []
    nested = False

    def first(_value: Any, _context: Context, delivery: Any) -> None:
        nonlocal nested
        if delivery.kind != "update" or delivery.sequence != 1 or nested:
            return
        nested = True
        source.commit({"value": 2}, [("s", ("value",), 2)])
        state.subscribe(
            lambda value, _context, delivery: late.append((delivery.kind, delivery.sequence, value["value"]))
        )

    def second(value: Any, _context: Context, delivery: Any) -> None:
        if delivery.kind == "update":
            received.append(value["value"])

    state.subscribe(first)
    state.subscribe(second)
    source.commit({"value": 1}, [("s", ("value",), 1)])
    assert received == [1, 2]
    assert late == [("hydrate", 2, 2)]


def test_reports_listener_failures_without_raising_them_into_the_source() -> None:
    source = TestSource({"value": 0})
    errors: list[Exception] = []
    state = replicated_state(source, ReplicatedStateSourceOptions(on_error=errors.append))
    received: list[int] = []

    def failing(_value: Any, _context: Context, delivery: Any) -> None:
        if delivery.kind == "update":
            raise RuntimeError("listener failed")

    state.subscribe(failing)
    state.subscribe(
        lambda value, _context, delivery: received.append(value["value"]) if delivery.kind == "update" else None
    )
    source.commit({"value": 1}, [("s", ("value",), 1)])
    source.commit({"value": 2}, [("s", ("value",), 2)])
    assert received == [1, 2]
    assert [str(error) for error in errors] == ["listener failed", "listener failed"]


def test_reports_cursor_gaps_disposes_the_broken_attachment_and_ignores_later_frames() -> None:
    source = TestSource({"value": 0}, 5)
    errors: list[Exception] = []
    state = replicated_state(source, ReplicatedStateSourceOptions(on_error=errors.append))
    source.commit({"value": 2}, [("s", ("value",), 2)], BACKGROUND_CONTEXT, 7)
    assert "expected 6, received 7" in str(errors[0])
    assert len(source.attachments) == 0
    assert state.value == {"value": 0}
    source.commit({"value": 3}, [("s", ("value",), 3)], BACKGROUND_CONTEXT, 8)
    assert state.value == {"value": 0}


def test_rejects_unsafe_cursors() -> None:
    source = TestSource({"value": 0}, 5)
    errors: list[Exception] = []
    replicated_state(source, ReplicatedStateSourceOptions(on_error=errors.append))
    source.commit({"value": 1}, [], BACKGROUND_CONTEXT, 2**60)
    assert "safe integer" in str(errors[0])
    with pytest.raises(TypeError, match="safe integer"):
        replicated_state(TestSource({"value": 0}, True))  # type: ignore[arg-type]


def test_keeps_attachments_independent_and_disposes_each_idempotently() -> None:
    source = TestSource({"value": 0})
    first = replicated_state(source)
    second = replicated_state(source)
    assert len(source.attachments) == 2
    source.commit({"value": 1}, [("s", ("value",), 1)])
    assert first.value["value"] == 1
    assert second.value["value"] == 1

    first.dispose()
    first.dispose()
    assert len(source.attachments) == 1
    source.commit({"value": 2}, [("s", ("value",), 2)])
    assert first.value["value"] == 1
    assert second.value["value"] == 2
    second.dispose()
    assert len(source.attachments) == 0


def test_disposes_the_attachment_when_activation_fails() -> None:
    disposed: list[bool] = []

    class Attachment:
        snapshot = ReplicatedStateSnapshot({"value": 0}, 0)

        def activate(self, listener: Any) -> None:
            raise RuntimeError("activation failed")

        def dispose(self) -> None:
            disposed.append(True)

    class Source:
        def attach(self) -> Attachment:
            return Attachment()

    with pytest.raises(RuntimeError, match="activation failed"):
        replicated_state(Source())
    assert disposed == [True]


def test_listeners_may_declare_fewer_arguments_like_javascript() -> None:
    state = replicated_state({"value": 0})
    seen: list[Any] = []
    state.subscribe(lambda value: seen.append(value["value"]))
    state.subscribe(lambda value, context: seen.append(context is BACKGROUND_CONTEXT))
    state.subscribe(lambda *arguments: seen.append(len(arguments)))
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 1))
    assert seen[-3:] == [1, True, 3]
