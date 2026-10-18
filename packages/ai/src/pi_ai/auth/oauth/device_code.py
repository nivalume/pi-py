"""RFC 8628 device authorization polling shared by the device-code login flows."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

from ...abort import AbortError, AbortSignal
from . import _common
from ._common import OAuthError

__all__ = [
    "DeviceCodePending",
    "DeviceCodeSlowDown",
    "DeviceCodeFailed",
    "DeviceCodeComplete",
    "DeviceCodePollResult",
    "abortable_sleep",
    "poll_oauth_device_code_flow",
]

T = TypeVar("T")

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL "
    "or VM environments. Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
# RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


@dataclass(frozen=True, slots=True)
class DeviceCodePending:
    status: ClassVar[str] = "pending"


@dataclass(frozen=True, slots=True)
class DeviceCodeSlowDown:
    #: New minimum interval reported by the server, if any.
    interval_seconds: float | None = None
    status: ClassVar[str] = "slow_down"


@dataclass(frozen=True, slots=True)
class DeviceCodeFailed:
    message: str
    status: ClassVar[str] = "failed"


@dataclass(frozen=True, slots=True)
class DeviceCodeComplete(Generic[T]):
    value: T
    status: ClassVar[str] = "complete"


DeviceCodePollResult: TypeAlias = "DeviceCodePending | DeviceCodeSlowDown | DeviceCodeFailed | DeviceCodeComplete[Any]"


async def abortable_sleep(ms: float, signal: AbortSignal, cancel_message: str) -> None:
    """Sleep ``ms`` milliseconds; raise :class:`AbortError` with ``cancel_message`` on abort."""
    if signal.aborted:
        raise AbortError(cancel_message)
    try:
        await _common.sleep(ms / 1000, signal)
    except AbortError:
        raise AbortError(cancel_message) from None


async def poll_oauth_device_code_flow(
    *,
    poll: Callable[[], Awaitable[DeviceCodePollResult]],
    signal: AbortSignal,
    interval_seconds: float | None = None,
    expires_in_seconds: float | None = None,
    wait_before_first_poll: bool = False,
) -> Any:
    """Poll until ``poll`` completes, fails, or the device code expires.

    Polls immediately unless ``wait_before_first_poll``; ``slow_down`` raises the
    interval to the server-provided value, or by five seconds.
    """
    deadline = _common.now_ms() + expires_in_seconds * 1000 if _common.is_number(expires_in_seconds) else math.inf
    interval_ms = max(
        MINIMUM_INTERVAL_MS,
        math.floor((interval_seconds if interval_seconds is not None else DEFAULT_POLL_INTERVAL_SECONDS) * 1000),
    )

    slow_down_responses = 0
    if wait_before_first_poll:
        remaining_ms = deadline - _common.now_ms()
        if remaining_ms > 0:
            await abortable_sleep(min(interval_ms, remaining_ms), signal, CANCEL_MESSAGE)

    while _common.now_ms() < deadline:
        if signal.aborted:
            raise AbortError(CANCEL_MESSAGE)

        result = await poll()
        if isinstance(result, DeviceCodeComplete):
            return result.value
        if isinstance(result, DeviceCodeFailed):
            raise OAuthError(result.message)
        if isinstance(result, DeviceCodeSlowDown):
            slow_down_responses += 1
            # Use the server-provided interval when given (GitHub reports the new required minimum
            # in `interval`); trusting only a client-tracked value risks polling early forever under
            # WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5: increase by 5 seconds.
            server_interval = result.interval_seconds
            if _common.is_number(server_interval) and math.isfinite(server_interval) and server_interval > 0:
                interval_ms = max(MINIMUM_INTERVAL_MS, math.floor(server_interval * 1000))
            else:
                interval_ms = max(MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS)

        remaining_ms = deadline - _common.now_ms()
        if remaining_ms <= 0:
            break

        await abortable_sleep(min(interval_ms, remaining_ms), signal, CANCEL_MESSAGE)

    raise OAuthError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)
