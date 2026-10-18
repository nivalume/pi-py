"""Request options for pi-ai (no upstream module: TypeScript passes its camelCase options straight through).

``ConversationStreamOptions`` and ``ConversationRetryPolicy`` are stored in checkpoints as JSON with
upstream's camelCase keys; pi-ai takes snake_case keyword options and a :class:`pi_ai.utils.retry.RetryPolicy`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pi_ai.utils.retry import RetryPolicy, retry_delay_ms

__all__ = ["retry_delay", "stream_options"]

#: ``ConversationStreamOptions`` keys and the pi-ai option each one sets.
_STREAM_KEYS: Final = {
    "transport": "transport",
    "timeoutMs": "timeout_ms",
    "maxRetries": "max_retries",
    "maxRetryDelayMs": "max_retry_delay_ms",
    "headers": "headers",
    "metadata": "metadata",
    "cacheRetention": "cache_retention",
    "deferred": "deferred",
}


def stream_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """pi-ai request options from stored ``ConversationStreamOptions``; unknown keys are dropped."""
    converted: dict[str, Any] = {}
    for key, value in options.items():
        name = _STREAM_KEYS.get(key)
        if name is None or value is None:
            continue
        converted[name] = dict(value) if isinstance(value, Mapping) else value
    return converted


def retry_delay(policy: Mapping[str, Any], attempt: int) -> float:
    """``retryDelayMs(policy, attempt)`` of pi-ai for a resolved ``ConversationRetryPolicy``."""
    return retry_delay_ms(
        RetryPolicy(
            enabled=policy["enabled"],
            max_retries=policy["maxRetries"],
            base_delay_ms=policy["baseDelayMs"],
            max_agent_delay_ms=policy.get("maxAgentDelayMs"),
        ),
        attempt,
    )
