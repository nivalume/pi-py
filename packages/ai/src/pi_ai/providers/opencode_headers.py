"""OpenCode's per-conversation routing header."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..event_stream import AssistantMessageEventStream
from ..types import Model, TranscriptContext

__all__ = ["OPENCODE_SESSION_HEADER", "with_opencode_session_header"]

OPENCODE_SESSION_HEADER = "x-opencode-session"


def _has_header(headers: Mapping[str, Any] | None, name: str) -> bool:
    # Any caller entry counts, including a ``None`` suppression marker.
    expected = name.lower()
    return any(key.lower() == expected for key in headers or {})


def _with_session_header(options: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not options or not options.get("session_id") or _has_header(options.get("headers"), OPENCODE_SESSION_HEADER):
        return options
    return {
        **options,
        "headers": {**(options.get("headers") or {}), OPENCODE_SESSION_HEADER: options["session_id"]},
    }


class _OpenCodeStreams:
    """Delegating ``ProviderStreams`` that keeps the wrapped deferred capabilities."""

    __slots__ = ("_streams", "cancel_deferred", "fetch_deferred")

    def __init__(self, streams: Any) -> None:
        self._streams = streams
        self.fetch_deferred = getattr(streams, "fetch_deferred", None)
        self.cancel_deferred = getattr(streams, "cancel_deferred", None)

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream(model, context, _with_session_header(options))

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self._streams.stream_simple(model, context, _with_session_header(options))

    def __repr__(self) -> str:
        return f"with_opencode_session_header({self._streams!r})"


def with_opencode_session_header(streams: Any) -> Any:
    """Add OpenCode's required ``x-opencode-session`` header from ``session_id`` before API dispatch.

    A caller-supplied header with the same name (any case, including ``None``) wins.
    """
    return _OpenCodeStreams(streams)
