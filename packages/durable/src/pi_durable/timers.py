"""Host timers (port of ``timers.ts``).

Timers run on the asyncio event loop: :func:`native_timer` arms ``loop.call_later`` on the running
loop, so it must be called from the loop's thread. asyncio has neither a maximum delay nor
referenced timers (a pending timer never keeps a process alive by itself), so unlike upstream's
``setTimeout`` wrapper no long delay is split into steps and no background timer is declined.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .types import SetTimer, TimerOptions

__all__ = ["Timers", "host_timers", "native_timer"]


def native_timer(callback: Callable[[], None], delay_ms: float, options: TimerOptions) -> Callable[[], None]:
    """The default ``HarnessOptions.setTimer``: ``loop.call_later`` on the running event loop.

    Returns the function that disarms the timer. ``options["background"]`` changes nothing here.
    """
    loop = asyncio.get_running_loop()
    handle = loop.call_later(max(0.0, delay_ms) / 1000, callback)
    return handle.cancel


class Timers:
    """Timers the Session and Harness arm through the host's ``setTimer`` (built by :func:`host_timers`)."""

    __slots__ = ("_arm", "_report", "_reported")

    def __init__(self, set_timer: SetTimer, report: Callable[[BaseException], None]) -> None:
        self._arm = set_timer
        self._report = report
        self._reported = False

    def _fall_back(self, error: BaseException) -> None:
        if self._reported:
            return
        self._reported = True
        self._report(error)

    def foreground(self, callback: Callable[[], None], delay_ms: float) -> Callable[[], None]:
        """A timer that paces or resumes work; when the host raises or declines it, the native timer arms it."""
        try:
            disarm = self._arm(callback, delay_ms, {"background": False})
            if disarm is not None:
                return disarm
            self._fall_back(RuntimeError("HarnessOptions.setTimer declined a timer that paces work; using setTimeout"))
        except Exception as error:
            self._fall_back(error)
        return native_timer(callback, delay_ms, {"background": False})

    def background(self, callback: Callable[[], None], delay_ms: float) -> Callable[[], None] | None:
        """A timer that only frees memory; ``None`` when the host declines it."""
        try:
            return self._arm(callback, delay_ms, {"background": True})
        except Exception as error:
            self._report(error)
            return None


def host_timers(set_timer: SetTimer | None, report: Callable[[BaseException], None]) -> Timers:
    """Wrap a host ``setTimer``.

    A foreground timer must fire, or work stalls: one the host raises on or declines is armed with
    :func:`native_timer`, and the first such failure is reported. A background one the host raises
    on is reported and declined.
    """
    return Timers(set_timer if set_timer is not None else native_timer, report)
