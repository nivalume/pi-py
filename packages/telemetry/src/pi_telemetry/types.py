"""The callback telemetry contract (port of ``@earendil-works/pi-telemetry`` ``src/index.ts``, core types).

A :class:`TelemetryContext` starts a span around a callback. The callback receives a :class:`TelemetrySpan`,
which records attributes, events and a status and is also the explicit parent context for child spans. There
is no ambient current span: parents are always passed as arguments.

Python mapping of the TypeScript types:

- ``startSpan(options, callback): Promise<T>`` is :meth:`TelemetryContext.start_span`, which returns an
  awaitable. The callback runs synchronously at the call, exactly once. When it returns an awaitable (an
  ``async def`` callback returns a coroutine), the span stays open until that awaitable settles, so the
  returned awaitable must be awaited, as any coroutine must.
- ``undefined`` attribute values are ``None``: they are ignored, never recorded.
- ``SpanOptions`` and ``SpanStatus`` are ``TypedDict`` s with upstream's keys.
- Rejection values are exceptions; the same exception object is raised again.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
    from .content import ContentCapture

__all__ = [
    "AttributeScalar",
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanErrorInfo",
    "OkSpanStatus",
    "ErrorSpanStatus",
    "SpanStatus",
    "SpanCallback",
    "TelemetryContext",
    "TelemetrySpan",
]

T = TypeVar("T")

#: A primitive attribute value. ``bool`` is not a number.
AttributeScalar: TypeAlias = str | int | float | bool

#: A primitive scalar or a homogeneous sequence of one primitive type (stored as a list).
AttributeValue: TypeAlias = str | int | float | bool | Sequence[str] | Sequence[int | float] | Sequence[bool]

#: An open attribute bag; ``None`` values (JS ``undefined``) are ignored.
SpanAttributes: TypeAlias = Mapping[str, "AttributeValue | None"]


class SpanOptions(TypedDict):
    """The span name and optional start attributes."""

    name: str
    attributes: NotRequired[SpanAttributes]


class SpanErrorInfo(TypedDict):
    name: str
    message: str


class OkSpanStatus(TypedDict):
    status: Literal["ok"]


class ErrorSpanStatus(TypedDict):
    status: Literal["error"]
    error: NotRequired[SpanErrorInfo]


#: An explicit ``ok`` or ``error`` status; the last one set wins.
SpanStatus: TypeAlias = OkSpanStatus | ErrorSpanStatus

#: The callback a span runs around: a plain function or an ``async def``.
SpanCallback: TypeAlias = "Callable[[TelemetrySpan], T | Awaitable[T]]"


class TelemetryContext(Protocol):
    """Starts callback-managed child spans."""

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        """Run ``callback`` inside a new child span; the span settles when the callback's result settles.

        The callback is invoked synchronously, exactly once. The returned awaitable yields the callback's value
        or raises the exception it raised, including one raised synchronously. Normal completion is ``ok``,
        a raise is ``error``, unless the callback set a status explicitly.
        """
        ...


class TelemetrySpan(TelemetryContext, Protocol):
    """Records attributes, events and status; also acts as the parent context of child spans.

    Recording methods are synchronous, passive and never raise. Calls after settlement are ignored.
    """

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None:
        """Merge ``attributes``; later values replace earlier ones and ``None`` values are ignored."""
        ...

    def set_status(self, status: SpanStatus) -> None: ...


class ContentCaptureCarrier(Protocol):
    """A context that carries an explicit content-capture opt-in (see :func:`pi_telemetry.with_content_capture`)."""

    @property
    def content_capture(self) -> ContentCapture | None: ...
