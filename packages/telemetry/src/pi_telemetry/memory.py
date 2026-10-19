"""The backend-neutral reference adapter (port of ``src/memory.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ._util import (
    automatic_error_status,
    copy_attributes,
    copy_status,
    merge_attributes,
    read_span_options,
    run_callback,
)
from .noop import NOOP_TELEMETRY_CONTEXT
from .types import AttributeValue, SpanAttributes, SpanOptions, SpanStatus, TelemetrySpan

__all__ = ["InMemoryTelemetryContext", "RecordedTelemetryEvent", "RecordedTelemetrySpan"]

T = TypeVar("T")


def _detached(attributes: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
    detached: dict[str, AttributeValue] = {}
    for name, value in attributes.items():
        detached[name] = list(value) if isinstance(value, list) else value  # type: ignore[assignment]
    return detached


def _detached_status(status: SpanStatus) -> SpanStatus:
    error = status.get("error")
    if status["status"] == "error" and error is not None:
        return {"status": "error", "error": {"name": error["name"], "message": error["message"]}}
    return {"status": status["status"]}  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RecordedTelemetryEvent:
    """A normalized captured event: its name and attributes (``None`` values dropped)."""

    name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordedTelemetrySpan:
    """A normalized captured span snapshot.

    ``id`` and ``end_sequence`` are deterministic counters starting at 1; ``parent_id`` is ``None`` for a root
    span; ``end_sequence`` is ``None`` while the span is open. No timestamps are recorded.
    """

    id: int
    parent_id: int | None
    name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    events: list[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=lambda: {"status": "ok"})
    settled: bool = False
    end_sequence: int | None = None


class _MutableSpan:
    __slots__ = (
        "attributes",
        "end_sequence",
        "events",
        "explicit_status",
        "id",
        "name",
        "parent_id",
        "settled",
        "status",
    )

    def __init__(self, id: int, parent_id: int | None, name: str, attributes: dict[str, AttributeValue]) -> None:
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.attributes = attributes
        self.events: list[tuple[str, dict[str, AttributeValue]]] = []
        self.status: SpanStatus = {"status": "ok"}
        self.explicit_status = False
        self.settled = False
        self.end_sequence: int | None = None


class _State:
    __slots__ = ("next_end_sequence", "next_span_id", "spans")

    def __init__(self) -> None:
        self.spans: list[_MutableSpan] = []
        self.next_span_id = 1
        self.next_end_sequence = 1


class _InMemorySpan:
    """The callback's span: records into its ``_MutableSpan`` until settlement."""

    __slots__ = ("_recorded", "_state")

    def __init__(self, state: _State, recorded: _MutableSpan) -> None:
        self._state = state
        self._recorded = recorded

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        return _start_in_memory_span(self._state, self._recorded, options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        recorded = self._recorded
        if recorded.settled:
            return
        try:
            if not isinstance(name, str):
                raise TypeError("Telemetry event names must be strings")
            recorded.events.append((name, copy_attributes(attributes)))
        except Exception:
            # Recording is passive. Ignore malformed or unreadable telemetry payloads.
            pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        recorded = self._recorded
        if recorded.settled:
            return
        try:
            recorded.attributes = merge_attributes(recorded.attributes, attributes)
        except Exception:
            pass

    def set_status(self, status: SpanStatus) -> None:
        recorded = self._recorded
        if recorded.settled:
            return
        try:
            recorded.status = copy_status(status)
            recorded.explicit_status = True
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"InMemoryTelemetrySpan(id={self._recorded.id}, name={self._recorded.name!r})"


def _settle(state: _State, recorded: _MutableSpan, failed: bool, error: BaseException | None) -> None:
    if recorded.settled:
        return
    if failed and not recorded.explicit_status:
        recorded.status = automatic_error_status(error)
    recorded.settled = True
    recorded.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


def _start_in_memory_span(
    state: _State,
    parent: _MutableSpan | None,
    options: SpanOptions,
    callback: Callable[[TelemetrySpan], T | Awaitable[T]],
) -> Awaitable[T]:
    if parent is not None and parent.settled:
        return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)
    try:
        name, attributes = read_span_options(options)
        recorded = _MutableSpan(state.next_span_id, None if parent is None else parent.id, name, attributes)
        state.next_span_id += 1
        state.spans.append(recorded)
    except Exception:
        return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)
    span = _InMemorySpan(state, recorded)
    return run_callback(callback, span, lambda failed, error: _settle(state, recorded, failed, error))


class InMemoryTelemetryContext:
    """Backend-neutral reference implementation that records spans in process memory.

    Storage is unbounded and process-local. Create a fresh instance to isolate tests or independent recording
    scopes, and do not capture sensitive attributes unless the caller's data policy allows them.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state = _State()

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        return _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> list[RecordedTelemetrySpan]:
        """Detached snapshots in span-start order."""
        return [
            RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_detached(span.attributes),
                events=[RecordedTelemetryEvent(name, _detached(attributes)) for name, attributes in span.events],
                status=_detached_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            for span in self._state.spans
        ]

    def clear(self) -> None:
        """Forget every recorded span; IDs and end sequences keep counting."""
        self._state.spans.clear()

    def __repr__(self) -> str:
        return f"InMemoryTelemetryContext(spans={len(self._state.spans)})"
