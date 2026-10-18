"""Process-wide fallback stream function."""

from __future__ import annotations

from .types import StreamFn

__all__ = ["set_default_stream_fn", "get_default_stream_fn"]

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """Configure the fallback used by ``Agent`` and the low-level loops when ``stream_fn`` is omitted.

    Hosts that provide a default model runtime install its stream function here, e.g.
    ``set_default_stream_fn(models.stream_simple)``.
    """
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn
