"""The shared passive context (port of ``src/noop.ts``)."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeGuard, TypeVar

from ._util import Settled
from .types import SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan

__all__ = ["NOOP_TELEMETRY_CONTEXT", "is_telemetry_enabled"]

T = TypeVar("T")


class _NoopTelemetrySpan:
    """One inert span: it never inspects or retains names, attributes, events or statuses."""

    __slots__ = ()

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        try:
            result = callback(self)
        except BaseException as error:
            return Settled(None, error)
        if inspect.isawaitable(result):
            return result
        return Settled(result)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        pass

    def set_status(self, status: SpanStatus) -> None:
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("NOOP_TELEMETRY_CONTEXT is frozen")

    def __repr__(self) -> str:
        return "NOOP_TELEMETRY_CONTEXT"

    def __copy__(self) -> _NoopTelemetrySpan:
        return self

    def __deepcopy__(self, memo: Any) -> _NoopTelemetrySpan:
        return self


#: Shared telemetry context used when an application does not provide one. It is also its own (frozen) span,
#: so nested spans reuse it.
NOOP_TELEMETRY_CONTEXT: Final[TelemetryContext] = _NoopTelemetrySpan()


def is_telemetry_enabled(context: TelemetryContext | None) -> TypeGuard[TelemetryContext]:
    """Whether ``context`` records anything: ``False`` for ``None`` and :data:`NOOP_TELEMETRY_CONTEXT`.

    Instrumentation uses it to skip building attributes on the disabled path.
    """
    return context is not None and context is not NOOP_TELEMETRY_CONTEXT
