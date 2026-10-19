"""Explicit opt-in content capture (not part of upstream pi-telemetry; see the README's data policy).

By default pi instrumentation records no content: no prompts, completions, thinking, tool arguments or output,
file contents, payloads, headers, credentials, or free-form error details. :func:`with_content_capture` wraps a
context so that instrumentation running under it, and under every span started from it, records such content as
``pi.content`` span events, truncated to ``max_length`` characters after an optional ``redact`` callback::

    telemetry = with_content_capture(InMemoryTelemetryContext(), max_length=2000, redact=scrub_secrets)

The opt-in travels with the context, explicitly, like any parent: nothing global changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from .noop import NOOP_TELEMETRY_CONTEXT, is_telemetry_enabled
from .schema import TelemetryEventDefinition
from .types import SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan

__all__ = [
    "CONTENT_EVENT_NAME",
    "CONTENT_EVENT",
    "ContentCapture",
    "ContentRedactor",
    "with_content_capture",
    "content_capture_of",
    "record_content",
]

T = TypeVar("T")

#: ``redact(kind, text)``: the text to record, or ``None`` to record nothing.
ContentRedactor = Callable[[str, str], "str | None"]

CONTENT_EVENT_NAME: Final = "pi.content"

#: The event definition every pi span that may capture content declares.
CONTENT_EVENT: Final[TelemetryEventDefinition] = {
    "description": "Captured content. Recorded only under an explicit with_content_capture() opt-in.",
    "attributes": {
        "pi.content.kind": {
            "type": "string",
            "required": True,
            "description": "What the text is",
            "cardinality": "low",
            "examples": ["system", "user", "assistant", "thinking", "tool_call", "tool_arguments", "tool_output"],
        },
        "pi.content.text": {
            "type": "string",
            "required": True,
            "sensitive": True,
            "cardinality": "high",
            "description": "The redacted text, truncated to the capture's max_length characters",
        },
        "pi.content.length": {
            "type": "number",
            "required": True,
            "description": "Characters in the redacted text before truncation",
        },
        "pi.content.truncated": {
            "type": "boolean",
            "required": True,
            "description": "Whether the text was cut at max_length",
        },
        "pi.content.name": {
            "type": "string",
            "required": False,
            "description": "The tool or message name the text belongs to",
        },
        "pi.content.index": {
            "type": "number",
            "required": False,
            "description": "Position of the item among its siblings (for example the message index)",
        },
    },
}


@dataclass(frozen=True, slots=True)
class ContentCapture:
    """The content-capture policy a context carries."""

    #: Characters kept per item; longer text is cut and marked ``truncated``. ``-1`` keeps everything.
    max_length: int = 4096
    redact: ContentRedactor | None = None

    def prepare(self, kind: str, text: str) -> tuple[str, int, bool] | None:
        """``(text, length, truncated)`` after redaction and truncation, or ``None`` to record nothing.

        A raising redactor records nothing: capture is passive.
        """
        try:
            redact = self.redact
            if redact is not None:
                redacted = redact(kind, text)
                if redacted is None:
                    return None
                text = str(redacted)
            length = len(text)
            limit = self.max_length
            if 0 <= limit < length:
                return text[:limit], length, True
            return text, length, False
        except Exception:
            return None


class _CapturingContext:
    __slots__ = ("_inner", "content_capture")

    def __init__(self, inner: TelemetryContext, capture: ContentCapture) -> None:
        self._inner = inner
        self.content_capture: ContentCapture | None = capture

    def start_span(self, options: SpanOptions, callback: Callable[[TelemetrySpan], T | Awaitable[T]]) -> Awaitable[T]:
        capture: Any = self.content_capture
        return self._inner.start_span(options, lambda span: callback(_CapturingSpan(span, capture)))

    def __repr__(self) -> str:
        return f"with_content_capture({self._inner!r})"


class _CapturingSpan(_CapturingContext):
    __slots__ = ()

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        self._inner.add_event(name, attributes)  # type: ignore[attr-defined]

    def set_attributes(self, attributes: SpanAttributes) -> None:
        self._inner.set_attributes(attributes)  # type: ignore[attr-defined]

    def set_status(self, status: SpanStatus) -> None:
        self._inner.set_status(status)  # type: ignore[attr-defined]


def with_content_capture(
    context: TelemetryContext, *, max_length: int = 4096, redact: ContentRedactor | None = None
) -> TelemetryContext:
    """``context`` with content capture enabled for every span started from it, directly or nested.

    A disabled context (``None`` or :data:`NOOP_TELEMETRY_CONTEXT`) stays disabled.
    """
    if not is_telemetry_enabled(context):
        return NOOP_TELEMETRY_CONTEXT
    return _CapturingContext(context, ContentCapture(max_length, redact))


def content_capture_of(context: object) -> ContentCapture | None:
    """The capture policy ``context`` carries, or ``None`` when content must not be recorded."""
    try:
        capture = getattr(context, "content_capture", None)
    except Exception:
        return None
    return capture if isinstance(capture, ContentCapture) else None


def record_content(span: Any, kind: str, text: str, *, name: str | None = None, index: int | None = None) -> None:
    """Add a ``pi.content`` event to ``span`` when its context opted in; otherwise do nothing."""
    capture = content_capture_of(span)
    if capture is None:
        return
    prepared = capture.prepare(kind, text)
    if prepared is None:
        return
    value, length, truncated = prepared
    attributes: dict[str, Any] = {
        "pi.content.kind": kind,
        "pi.content.text": value,
        "pi.content.length": length,
        "pi.content.truncated": truncated,
    }
    if name is not None:
        attributes["pi.content.name"] = name
    if index is not None:
        attributes["pi.content.index"] = index
    try:
        span.add_event(CONTENT_EVENT_NAME, attributes)
    except Exception:
        pass
