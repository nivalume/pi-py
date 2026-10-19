"""Shared option handling for ``stream_simple`` implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..types import Model, TranscriptContext
from ..utils.estimate import estimate_context_tokens

__all__ = [
    "CONTEXT_SAFETY_TOKENS",
    "MIN_ANSWER_TOKENS",
    "DEFAULT_THINKING_BUDGETS",
    "clamp_max_tokens_to_context",
    "resolve_sampling_params",
    "build_base_options",
    "clamp_reasoning",
    "thinking_budget_for_level",
    "clamp_thinking_budget_to_answer_room",
    "adjust_max_tokens_for_thinking",
]

CONTEXT_SAFETY_TOKENS = 4096
_MIN_MAX_TOKENS = 1
#: Tokens always left for the answer when a thinking budget shares the response ceiling.
MIN_ANSWER_TOKENS = 1024
DEFAULT_THINKING_BUDGETS: dict[str, int] = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}

_BASE_OPTION_KEYS = (
    "temperature",
    "signal",
    "telemetry_context",
    "http_client",
    "transport",
    "cache_retention",
    "session_id",
    "headers",
    "on_payload",
    "on_response",
    "on_provider_stream_event",
    "timeout_ms",
    "websocket_connect_timeout_ms",
    "max_retries",
    "max_retry_delay_ms",
    "metadata",
    "env",
)


def clamp_max_tokens_to_context(model: Model, context: TranscriptContext, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(_MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(_MIN_MAX_TOKENS, available))


def resolve_sampling_params(
    model: Model, thinking_level: str, request_params: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    from ..models import clamp_thinking_level

    effective = clamp_thinking_level(model, thinking_level)  # type: ignore[arg-type]
    level_params = (model.sampling_params_by_thinking_level or {}).get(effective)
    if not model.sampling_params and not level_params and not request_params:
        return None
    return {**(model.sampling_params or {}), **(level_params or {}), **(request_params or {})}


def build_base_options(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """Translate simple options into the API-agnostic stream options."""
    options = options or {}
    base: dict[str, Any] = {key: options[key] for key in _BASE_OPTION_KEYS if options.get(key) is not None}
    sampling = resolve_sampling_params(model, options.get("reasoning") or "off", options.get("sampling_params"))
    if sampling is not None:
        base["sampling_params"] = sampling
    max_tokens = options.get("max_tokens")
    base["max_tokens"] = clamp_max_tokens_to_context(
        model, context, max_tokens if max_tokens is not None else model.max_tokens
    )
    key = api_key or options.get("api_key")
    if key:
        base["api_key"] = key
    return base


def clamp_reasoning(effort: str | None) -> str | None:
    return "high" if effort in ("xhigh", "max") else effort


def thinking_budget_for_level(reasoning_level: str, custom_budgets: Mapping[str, int] | None = None) -> int:
    budgets = {**DEFAULT_THINKING_BUDGETS, **(custom_budgets or {})}
    level = clamp_reasoning(reasoning_level) or "medium"
    return budgets[level]


def clamp_thinking_budget_to_answer_room(thinking_budget: int, ceiling: int) -> int:
    """Cap a thinking budget so at least ``MIN_ANSWER_TOKENS`` remain under a shared ceiling."""
    return min(thinking_budget, max(0, ceiling - MIN_ANSWER_TOKENS))


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: str,
    custom_budgets: Mapping[str, int] | None = None,
) -> tuple[int, int]:
    """Return ``(max_tokens, thinking_budget)`` with thinking fitted inside the response ceiling.

    ``base_max_tokens=None`` means no explicit caller cap: use the model cap.
    """
    thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets)
    max_tokens = (
        model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    )
    if max_tokens <= thinking_budget:
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens)
    return max_tokens, thinking_budget
