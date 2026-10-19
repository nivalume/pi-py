"""Runner-independent adapter conformance cases (port of ``src/testing/conformance.ts``).

Upstream's JS-specific cases translate as follows: rejection values are exceptions (identity is checked with
``is``), ``undefined`` rejections have no Python analog, and the ``Proxy`` objects that throw on every read are
:func:`unreadable_mapping`, :func:`unreadable_sequence` and :class:`UnreadableError`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from ..memory import RecordedTelemetrySpan
from ..types import SpanAttributes, SpanOptions, SpanStatus, TelemetrySpan
from .types import TelemetryAdapterConformanceCase, TelemetryAdapterFixture, TelemetryAdapterFixtureFactory

__all__ = [
    "create_telemetry_adapter_conformance",
    "UnreadableError",
    "unreadable_mapping",
    "unreadable_sequence",
]


def _fail(what: str) -> NoReturn:
    raise RuntimeError(what)


class _UnreadableMapping(Mapping[str, Any]):
    """A mapping whose every read raises (upstream's throwing ``Proxy``)."""

    def __getitem__(self, key: str) -> Any:
        _fail("read")

    def __iter__(self) -> Iterator[str]:
        _fail("enumerate")

    def __len__(self) -> int:
        _fail("inspect")

    def get(self, key: str, default: Any = None) -> Any:
        _fail("read")

    def items(self) -> Any:
        _fail("enumerate")

    def keys(self) -> Any:
        _fail("enumerate")

    def values(self) -> Any:
        _fail("enumerate")

    def __contains__(self, key: object) -> bool:
        _fail("inspect")


class _UnreadableList(list[Any]):
    """A list whose every read raises."""

    def __iter__(self) -> Iterator[Any]:
        _fail("enumerate")

    def __len__(self) -> int:
        _fail("inspect")

    def __getitem__(self, index: Any) -> Any:
        _fail("read")


def unreadable_mapping() -> Any:
    """A mapping (options, attributes, status) whose reads all raise."""
    return _UnreadableMapping()


def unreadable_sequence() -> Any:
    """A list whose reads all raise, as an attribute value."""
    return _UnreadableList()


class UnreadableError(Exception):
    """An exception whose text cannot be read."""

    def __str__(self) -> str:
        _fail("read")

    def __repr__(self) -> str:
        _fail("read")

    @property
    def args(self) -> Any:  # type: ignore[override]
        _fail("read")


@dataclass(frozen=True, slots=True)
class _Case:
    group: str
    name: str
    _run: Callable[[], Awaitable[None]]

    async def run(self) -> None:
        await self._run()


ConformanceTest = Callable[[TelemetryAdapterFixture], Awaitable[None]]


def _create_case(factory: TelemetryAdapterFixtureFactory, group: str, name: str, test: ConformanceTest) -> _Case:
    async def run() -> None:
        fixture = await factory()
        try:
            await test(fixture)
        finally:
            await fixture.aclose()

    return _Case(group, name, run)


def _find_span(spans: Sequence[RecordedTelemetrySpan], name: str) -> RecordedTelemetrySpan:
    for span in spans:
        if span.name == name:
            return span
    raise AssertionError(f"Expected recorded span {name}")


async def _raises_same(operation: Awaitable[Any], expected: BaseException) -> None:
    try:
        await operation
    except BaseException as error:
        assert error is expected, f"Expected {expected!r}, got {error!r}"
        return
    raise AssertionError("Expected operation to raise")


def _does_not_raise(call: Callable[[], Any]) -> None:
    try:
        call()
    except Exception as error:  # pragma: no cover - reported as a failure
        raise AssertionError(f"Recording call raised {error!r}") from error


async def _lifecycle_success(fixture: TelemetryAdapterFixture) -> None:
    admitted = False
    calls = 0
    expected = {"value": 42}

    def callback(_span: TelemetrySpan) -> Any:
        nonlocal admitted, calls
        admitted = True
        calls += 1
        return expected

    result = fixture.context.start_span({"name": "success"}, callback)
    assert admitted is True
    assert calls == 1
    assert (await result) is expected
    span = _find_span(await fixture.get_spans(), "success")
    assert span.status == {"status": "ok"}
    assert span.settled is True


async def _lifecycle_rejections(fixture: TelemetryAdapterFixture) -> None:
    context = fixture.context
    sync_error = RuntimeError("sync")

    def raise_sync(_span: TelemetrySpan) -> None:
        raise sync_error

    await _raises_same(context.start_span({"name": "sync-error"}, raise_sync), sync_error)

    async_error = ValueError("async")

    async def raise_async(_span: TelemetrySpan) -> None:
        raise async_error

    await _raises_same(context.start_span({"name": "async-error"}, raise_async), async_error)

    future_error = KeyError("future")

    def rejected_future(_span: TelemetrySpan) -> Awaitable[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        future.set_exception(future_error)
        return future

    await _raises_same(context.start_span({"name": "future-error"}, rejected_future), future_error)

    unreadable_error = UnreadableError()

    def raise_unreadable(_span: TelemetrySpan) -> None:
        raise unreadable_error

    await _raises_same(context.start_span({"name": "unreadable-error"}, raise_unreadable), unreadable_error)

    async_unreadable_error = UnreadableError()

    async def raise_async_unreadable(_span: TelemetrySpan) -> None:
        raise async_unreadable_error

    await _raises_same(
        context.start_span({"name": "async-unreadable-error"}, raise_async_unreadable), async_unreadable_error
    )

    spans = await fixture.get_spans()
    for name in ("sync-error", "async-error", "future-error", "unreadable-error", "async-unreadable-error"):
        assert _find_span(spans, name).status["status"] == "error", name


async def _status_last_wins(fixture: TelemetryAdapterFixture) -> None:
    context = fixture.context

    def last(span: TelemetrySpan) -> None:
        span.set_status({"status": "error", "error": {"name": "Expected", "message": "first"}})
        span.set_status({"status": "ok"})

    await context.start_span({"name": "last-status"}, last)

    thrown = RuntimeError("after explicit status")

    def explicit_before_throw(span: TelemetrySpan) -> None:
        span.set_status({"status": "ok"})
        raise thrown

    await _raises_same(context.start_span({"name": "explicit-before-throw"}, explicit_before_throw), thrown)

    rejected = RuntimeError("after async explicit status")

    async def explicit_before_rejection(span: TelemetrySpan) -> None:
        span.set_status({"status": "error", "error": {"name": "Expected", "message": "async failure"}})
        raise rejected

    await _raises_same(context.start_span({"name": "explicit-before-rejection"}, explicit_before_rejection), rejected)

    def expected_failure(span: TelemetrySpan) -> dict[str, bool]:
        span.set_status({"status": "error", "error": {"name": "Expected", "message": "returned failure"}})
        return {"ok": False}

    await context.start_span({"name": "expected-failure"}, expected_failure)

    spans = await fixture.get_spans()
    assert _find_span(spans, "last-status").status == {"status": "ok"}
    assert _find_span(spans, "explicit-before-throw").status == {"status": "ok"}
    assert _find_span(spans, "explicit-before-rejection").status == {
        "status": "error",
        "error": {"name": "Expected", "message": "async failure"},
    }
    assert _find_span(spans, "expected-failure").status == {
        "status": "error",
        "error": {"name": "Expected", "message": "returned failure"},
    }


async def _recording_merges(fixture: TelemetryAdapterFixture) -> None:
    def record(span: TelemetrySpan) -> None:
        span.set_attributes({"count": 1, "overwrite": "middle"})
        span.set_attributes({"count": None, "overwrite": "end"})
        span.add_event("first", {"index": 1, "ignored": None})
        span.add_event("second", {"index": 2})

    await fixture.context.start_span(
        {"name": "recording", "attributes": {"start": "value", "overwrite": "start", "ignored": None}}, record
    )
    span = _find_span(await fixture.get_spans(), "recording")
    assert span.attributes == {"start": "value", "overwrite": "end", "count": 1}, span.attributes
    assert [(event.name, event.attributes) for event in span.events] == [
        ("first", {"index": 1}),
        ("second", {"index": 2}),
    ]


async def _recording_atomic_attributes(fixture: TelemetryAdapterFixture) -> None:
    def record(span: TelemetrySpan) -> None:
        attributes: SpanAttributes = {"partial": "must not survive", "unreadable": unreadable_sequence()}
        _does_not_raise(lambda: span.set_attributes(attributes))
        invalid: Any = {"partial": "must not survive", "object": {"not": "a primitive"}}
        _does_not_raise(lambda: span.set_attributes(invalid))

    await fixture.context.start_span({"name": "atomic-attributes", "attributes": {"retained": "value"}}, record)
    assert _find_span(await fixture.get_spans(), "atomic-attributes").attributes == {"retained": "value"}


async def _recording_inert_after_settlement(fixture: TelemetryAdapterFixture) -> None:
    settled_span: list[TelemetrySpan] = []
    await fixture.context.start_span(
        {"name": "settled", "attributes": {"value": "initial"}}, lambda span: settled_span.append(span)
    )
    captured = settled_span[0]
    _does_not_raise(lambda: captured.set_attributes({"value": "late"}))
    _does_not_raise(lambda: captured.add_event("late", {"value": True}))
    _does_not_raise(lambda: captured.set_status({"status": "error"}))
    child_admitted = False

    def child(_span: TelemetrySpan) -> int:
        nonlocal child_admitted
        child_admitted = True
        return 7

    child_result = captured.start_span({"name": "late-child"}, child)
    assert child_admitted is True
    assert (await child_result) == 7

    spans = await fixture.get_spans()
    assert len(spans) == 1, spans
    assert spans[0].attributes == {"value": "initial"}
    assert spans[0].events == []
    assert spans[0].status == {"status": "ok"}


async def _parentage(fixture: TelemetryAdapterFixture) -> None:
    release_first = asyncio.Event()

    async def parent_callback(parent: TelemetrySpan) -> None:
        async def first_child(_span: TelemetrySpan) -> None:
            await release_first.wait()

        first = asyncio.ensure_future(parent.start_span({"name": "first-child"}, first_child))
        second = parent.start_span({"name": "second-child"}, lambda _span: "done")
        assert (await second) == "done"
        release_first.set()
        await first

    await fixture.context.start_span({"name": "parent"}, parent_callback)

    spans = await fixture.get_spans()
    parent = _find_span(spans, "parent")
    first = _find_span(spans, "first-child")
    second = _find_span(spans, "second-child")
    assert parent.parent_id is None
    assert first.parent_id == parent.id
    assert second.parent_id == parent.id
    assert second.end_sequence is not None and first.end_sequence is not None and parent.end_sequence is not None
    assert second.end_sequence < first.end_sequence
    assert first.end_sequence < parent.end_sequence


async def _passivity_unreadable(fixture: TelemetryAdapterFixture) -> None:
    calls = 0

    def callback(_span: TelemetrySpan) -> int:
        nonlocal calls
        calls += 1
        return 9

    options: SpanOptions = unreadable_mapping()
    result = fixture.context.start_span(options, callback)
    assert calls == 1
    assert (await result) == 9
    assert list(await fixture.get_spans()) == []

    def record(span: TelemetrySpan) -> None:
        attributes: SpanAttributes = unreadable_mapping()
        status: SpanStatus = unreadable_mapping()
        _does_not_raise(lambda: span.set_attributes(attributes))
        _does_not_raise(lambda: span.add_event("unreadable-event", attributes))
        _does_not_raise(lambda: span.set_status(status))
        bad_status: Any = {"status": "maybe"}
        _does_not_raise(lambda: span.set_status(bad_status))

    await fixture.context.start_span({"name": "unreadable-recording"}, record)
    recorded = await fixture.get_spans()
    assert len(recorded) == 1
    assert recorded[0].attributes == {}
    assert recorded[0].events == []
    assert recorded[0].status == {"status": "ok"}


async def _passivity_status(fixture: TelemetryAdapterFixture) -> None:
    rejection = RuntimeError("rejected after unreadable status")

    async def callback(span: TelemetrySpan) -> None:
        status: SpanStatus = unreadable_mapping()
        _does_not_raise(lambda: span.set_status(status))
        raise rejection

    await _raises_same(fixture.context.start_span({"name": "unreadable-status"}, callback), rejection)
    assert _find_span(await fixture.get_spans(), "unreadable-status").status["status"] == "error"


def create_telemetry_adapter_conformance(
    factory: TelemetryAdapterFixtureFactory,
) -> list[TelemetryAdapterConformanceCase]:
    """Creates runner-independent cases for the callback telemetry adapter contract.

    Each case creates a fresh fixture with ``factory`` and closes it afterwards. Register them with pytest::

        CASES = create_telemetry_adapter_conformance(make_fixture)

        @pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.group}: {case.name}")
        async def test_conformance(case):
            await case.run()
    """
    return [
        _create_case(
            factory, "callback lifecycle", "admits once synchronously and preserves the result", _lifecycle_success
        ),
        _create_case(
            factory,
            "callback lifecycle",
            "preserves synchronous and asynchronous rejection values",
            _lifecycle_rejections,
        ),
        _create_case(factory, "status", "uses last explicit status without automatic overwrite", _status_last_wins),
        _create_case(factory, "recording", "merges attributes and records ordered events", _recording_merges),
        _create_case(factory, "recording", "ignores failed attribute calls atomically", _recording_atomic_attributes),
        _create_case(factory, "recording", "makes calls after settlement inert", _recording_inert_after_settlement),
        _create_case(factory, "parentage", "records nested and concurrent child relationships", _parentage),
        _create_case(factory, "passivity", "suppresses unreadable telemetry payload failures", _passivity_unreadable),
        _create_case(factory, "passivity", "ignores failed status calls atomically", _passivity_status),
    ]
