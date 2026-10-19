"""Port of ``test/conformance.test.ts``: the in-memory reference adapter against the conformance suite."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from pi_telemetry import InMemoryTelemetryContext, RecordedTelemetrySpan, TelemetryContext, TelemetrySpan
from pi_telemetry.testing import TelemetryAdapterFixture, create_telemetry_adapter_conformance


class _Fixture:
    def __init__(self) -> None:
        self._context = InMemoryTelemetryContext()

    @property
    def context(self) -> TelemetryContext:
        return self._context

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]:
        return self._context.get_spans()

    async def aclose(self) -> None:
        pass


async def _factory() -> TelemetryAdapterFixture:
    return _Fixture()


CASES = create_telemetry_adapter_conformance(_factory)


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.group}: {case.name}")
async def test_in_memory_conformance(case: Any) -> None:
    await case.run()


def test_cases_cover_every_group() -> None:
    assert {case.group for case in CASES} == {"callback lifecycle", "status", "recording", "parentage", "passivity"}


async def test_returns_detached_snapshots_without_exposing_mutable_recording_state() -> None:
    context = InMemoryTelemetryContext()
    open_settled: list[bool] = []
    open_end_sequence: list[int | None] = []

    def callback(span: TelemetrySpan) -> None:
        span.add_event("event", {"value": 1})
        [open_span] = context.get_spans()
        open_settled.append(open_span.settled)
        open_end_sequence.append(open_span.end_sequence)

    await context.start_span({"name": "snapshot", "attributes": {"tags": ["initial"]}}, callback)

    assert open_settled == [False]
    assert open_end_sequence == [None]
    [first] = context.get_spans()
    assert first.settled is True
    assert first.end_sequence == 1
    first.attributes["tags"] = ["mutated"]
    first.events[0].attributes["value"] = 2
    tags: Any = context.get_spans()[0].attributes["tags"]
    tags.append("mutated")

    [second] = context.get_spans()
    assert second.attributes == {"tags": ["initial"]}
    assert [(event.name, event.attributes) for event in second.events] == [("event", {"value": 1})]
