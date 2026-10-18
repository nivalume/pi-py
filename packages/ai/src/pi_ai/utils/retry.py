"""Retry classification and a policy-driven retry loop for assistant calls."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..abort import AbortError, AbortSignal, sleep
from ..types import AssistantMessage
from .callbacks import maybe_await

__all__ = [
    "RetryPolicy",
    "RetryCallbacks",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "retry_delay_ms",
    "retry_assistant_call",
    "is_retryable_assistant_error",
]


def _pattern(parts: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.IGNORECASE)


_NON_RETRYABLE = _pattern(
    [
        "GoUsageLimitError",
        "FreeUsageLimitError",
        "Monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
        "subscription_sharing_usage_limit_exceeded",
    ]
)

_RETRYABLE = _pattern(
    [
        "overloaded",
        "server_busy",
        "servers are currently busy",
        "currently experiencing high demand",
        "model is at capacity",
        "rate.?limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "524",
        r"service.?(?:is.?)?(?:temporarily.?)?unavailable",
        "server.?error",
        "internal.?error",
        r"provider.?returned.?(?:an.?)?error",
        "exceeded request buffer limit while retrying upstream",
        "unable to forward upstream",
        "upstream.?request.?failed",
        "upstream_stream_read_error",
        "stream was interrupted",
        "http/?2 stream failed",
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?lost",
        "other side closed",
        "fetch failed",
        r"request to \S+ failed, reason:",
        "getaddrinfo",
        "ENOTFOUND",
        "EAI_AGAIN",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        "websocket.?closed",
        "websocket.?error",
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        "pending stream has been canceled",
        "retry delay",
        "you can retry your request",
        "try your request again",
        "please retry your request",
        "ResourceExhausted",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
        # Python transport errors (httpx/httpcore wording)
        "remote ?protocol ?error",
        "server disconnected",
        "connection reset",
        "read ?error",
        "connect ?error",
    ]
)

DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


@dataclass(slots=True)
class RetryPolicy:
    """Bounded attempts with exponential backoff (``base_delay_ms * 2^(attempt-1)``)."""

    enabled: bool = True
    #: Max retry attempts (0 = no retries). The initial call never counts as a retry.
    max_retries: int = 3
    base_delay_ms: float = 2000
    #: Cap for each computed delay. Defaults to 60 seconds.
    max_agent_delay_ms: float | None = None


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> float:
    delay = policy.base_delay_ms * 2 ** max(0, attempt - 1)
    cap = policy.max_agent_delay_ms if policy.max_agent_delay_ms is not None else DEFAULT_MAX_AGENT_RETRY_DELAY_MS
    return min(delay, cap)


@dataclass(slots=True)
class RetryCallbacks:
    #: Before the backoff sleep of each retry attempt (1-indexed): (attempt, max_attempts, delay_ms, error_message).
    on_retry_scheduled: Callable[[int, int, float, str], Any] | None = None
    #: After the backoff sleep, immediately before the retried call starts.
    on_retry_attempt_start: Callable[[], Any] | None = None
    #: Once when the loop ends: (success, attempt, final_error).
    on_retry_finished: Callable[[bool, int, str | None], Any] | None = None


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    signal: AbortSignal | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Run an assistant-producing call with bounded retry on transient errors.

    Aborts are terminal and never retried; aborts during the backoff sleep are
    normalized to an aborted message. Non-retryable errors (quota, billing) return
    immediately.
    """
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0
    callbacks = callbacks or RetryCallbacks()
    attempt = 0
    last_retry: tuple[int, str] | None = None
    while True:
        response = await produce()
        if response.stop_reason == "aborted":
            if last_retry and callbacks.on_retry_finished:
                await maybe_await(callbacks.on_retry_finished(False, last_retry[0], None))
            return response
        if response.stop_reason != "error":
            if last_retry and callbacks.on_retry_finished:
                await maybe_await(callbacks.on_retry_finished(True, last_retry[0], None))
            return response
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry and callbacks.on_retry_finished:
                await maybe_await(callbacks.on_retry_finished(False, last_retry[0], response.error_message))
            return response

        attempt += 1
        last_retry = (attempt, response.error_message or "Unknown error")
        assert policy is not None
        delay = retry_delay_ms(policy, attempt)
        if callbacks.on_retry_scheduled:
            await maybe_await(callbacks.on_retry_scheduled(attempt, max_attempts, delay, last_retry[1]))
        try:
            await sleep(delay / 1000, signal)
        except AbortError:
            if callbacks.on_retry_finished:
                await maybe_await(callbacks.on_retry_finished(False, attempt, last_retry[1]))
            return dataclasses.replace(response, stop_reason="aborted", error_message=None)
        if callbacks.on_retry_attempt_start:
            await maybe_await(callbacks.on_retry_attempt_start())


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Whether a failed assistant message looks like a transient provider/transport error."""
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE.search(message.error_message):
        return False
    return _RETRYABLE.search(message.error_message) is not None
