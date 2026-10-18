"""Cleanup hooks for per-session resources such as pooled WebSocket connections."""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["register_session_resource_cleanup", "cleanup_session_resources"]

SessionResourceCleanup = Callable[[str | None], None]

_cleanups: list[SessionResourceCleanup] = []


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    _cleanups.append(cleanup)

    def unregister() -> None:
        if cleanup in _cleanups:
            _cleanups.remove(cleanup)

    return unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    """Release resources held for ``session_id`` (or every session when omitted)."""
    errors: list[Exception] = []
    for cleanup in tuple(_cleanups):
        try:
            cleanup(session_id)
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Failed to cleanup session resources", errors)
