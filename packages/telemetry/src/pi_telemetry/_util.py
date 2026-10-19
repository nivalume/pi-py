"""Shared helpers for adapters: payload copies that fail atomically, settled awaitables, and error statuses."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Generator, Mapping
from typing import Any, Generic, TypeVar

from .types import AttributeValue, SpanAttributes, SpanOptions, SpanStatus

__all__ = [
    "Settled",
    "copy_attribute_value",
    "copy_attributes",
    "merge_attributes",
    "copy_status",
    "read_span_options",
    "automatic_error_status",
    "run_callback",
]

T = TypeVar("T")

_SCALARS = (str, int, float)


class Settled(Generic[T]):
    """An already settled result: awaiting it returns the value or raises the error (``Promise.resolve/reject``)."""

    __slots__ = ("_error", "_value")

    def __init__(self, value: T | None = None, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error

    def __await__(self) -> Generator[Any, None, T]:
        if self._error is not None:
            raise self._error
        return self._value  # type: ignore[return-value]
        yield  # pragma: no cover - makes this a generator


def copy_attribute_value(value: Any) -> AttributeValue:
    """A detached copy of one attribute value; raises ``TypeError`` for anything but primitives and arrays.

    Arrays (lists and tuples) become lists and must hold one primitive type: strings, booleans, or numbers.
    """
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, list | tuple):
        items = list(value)
        kind: type | None = None
        for item in items:
            if isinstance(item, bool):
                item_kind: type = bool
            elif isinstance(item, str):
                item_kind = str
            elif isinstance(item, int | float):
                item_kind = float
            else:
                raise TypeError(f"Unsupported telemetry array element {type(item).__name__}")
            if kind is None:
                kind = item_kind
            elif kind is not item_kind:
                raise TypeError("Telemetry arrays must hold one primitive type")
        return items
    raise TypeError(f"Unsupported telemetry attribute value {type(value).__name__}")


def copy_attributes(attributes: SpanAttributes | None) -> dict[str, AttributeValue]:
    """A detached copy without ``None`` values. Raises when the payload cannot be read or holds invalid values,
    so a caller can ignore the whole recording call."""
    copy: dict[str, AttributeValue] = {}
    if attributes is None:
        return copy
    for name, value in attributes.items():
        if not isinstance(name, str):
            raise TypeError("Telemetry attribute names must be strings")
        if value is not None:
            copy[name] = copy_attribute_value(value)
    return copy


def merge_attributes(current: Mapping[str, AttributeValue], attributes: SpanAttributes) -> dict[str, AttributeValue]:
    """``current`` with ``attributes`` merged over it; raises (leaving ``current`` untouched) on a bad payload."""
    merged = dict(current)
    merged.update(copy_attributes(attributes))
    return merged


def copy_status(status: SpanStatus) -> SpanStatus:
    """A detached, validated copy; raises for an unreadable or malformed status."""
    kind = status["status"]
    if kind == "ok":
        return {"status": "ok"}
    if kind != "error":
        raise ValueError(f"Unknown telemetry status {kind!r}")
    error = status.get("error")
    if error is None:
        return {"status": "error"}
    name, message = error["name"], error["message"]
    if not isinstance(name, str) or not isinstance(message, str):
        raise TypeError("Telemetry error name and message must be strings")
    return {"status": "error", "error": {"name": name, "message": message}}


def read_span_options(options: SpanOptions) -> tuple[str, dict[str, AttributeValue]]:
    """The name and copied start attributes; raises for an unreadable or malformed payload."""
    name = options["name"]
    if not isinstance(name, str):
        raise TypeError("Telemetry span names must be strings")
    return name, copy_attributes(options.get("attributes"))


def automatic_error_status(error: BaseException | None) -> SpanStatus:
    """The status of a span whose callback raised: the exception's class name and message.

    Error inspection is passive: an exception whose text cannot be read gives an error status without details.
    """
    if error is not None:
        try:
            return {"status": "error", "error": {"name": type(error).__name__, "message": str(error)}}
        except Exception:
            pass
    return {"status": "error"}


def run_callback(
    callback: Callable[[Any], T | Awaitable[T]],
    span: Any,
    settle: Callable[[bool, BaseException | None], None],
) -> Awaitable[T]:
    """Invoke ``callback(span)`` synchronously and call ``settle(failed, error)`` once its result settles.

    The returned awaitable preserves the value and the raised exception object. A synchronous raise settles at
    once and is raised again when the result is awaited.
    """
    try:
        result = callback(span)
    except BaseException as error:
        settle(True, error)
        return Settled(None, error)
    if inspect.isawaitable(result):
        return _settle_after(result, settle)
    settle(False, None)
    return Settled(result)


async def _settle_after(result: Awaitable[T], settle: Callable[[bool, BaseException | None], None]) -> T:
    try:
        value = await result
    except BaseException as error:
        settle(True, error)
        raise
    settle(False, None)
    return value
