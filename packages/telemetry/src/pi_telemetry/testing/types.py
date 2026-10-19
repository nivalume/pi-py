"""Fixture and case shapes of the adapter conformance suite (port of ``src/testing/types.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from ..memory import RecordedTelemetrySpan
from ..types import TelemetryContext

__all__ = ["TelemetryAdapterFixture", "TelemetryAdapterFixtureFactory", "TelemetryAdapterConformanceCase"]


class TelemetryAdapterFixture(Protocol):
    """A fresh adapter instance and normalized snapshot reader owned by one conformance case.

    ``get_spans()`` may flush an asynchronous exporter before returning; ``aclose()`` releases the adapter
    (upstream's ``Symbol.asyncDispose``).
    """

    @property
    def context(self) -> TelemetryContext: ...

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]: ...

    async def aclose(self) -> None: ...


#: Creates an isolated adapter fixture for one conformance case.
TelemetryAdapterFixtureFactory = Callable[[], Awaitable[TelemetryAdapterFixture]]


class TelemetryAdapterConformanceCase(Protocol):
    """A runner-independent conformance case that any test framework can register."""

    @property
    def group(self) -> str: ...

    @property
    def name(self) -> str: ...

    async def run(self) -> None: ...
