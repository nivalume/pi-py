"""OpenRouter model-list shapes and price conversion."""

from __future__ import annotations

import math
import re
from typing import Any, TypedDict

from ..types import ModelCost, ModelCostTier

__all__ = [
    "OpenRouterPricing",
    "OpenRouterPricingOverride",
    "OpenRouterTopProvider",
    "OpenRouterModel",
    "openrouter_cost",
]


class OpenRouterPricingOverride(TypedDict, total=False):
    """A conditional price.

    ``min_prompt_tokens`` selects prompt-length pricing; ``utc_*`` fields select
    time-of-day or weekday pricing. Missing rates keep the base price.
    """

    min_prompt_tokens: int
    utc_start: int
    utc_end: int
    utc_days: list[str]
    prompt: str
    completion: str
    input_cache_read: str
    input_cache_write: str


class OpenRouterPricing(TypedDict, total=False):
    """OpenRouter's per-token prices, as decimal strings."""

    prompt: str
    completion: str
    input_cache_read: str
    input_cache_write: str
    overrides: list[OpenRouterPricingOverride]


class OpenRouterTopProvider(TypedDict, total=False):
    context_length: int | None
    max_completion_tokens: int | None


class OpenRouterModel(TypedDict, total=False):
    """The fields pi reads from a model in OpenRouter's ``GET /models`` and ``GET /models/user``."""

    id: str
    context_length: int | None
    top_provider: OpenRouterTopProvider
    pricing: OpenRouterPricing


_FLOAT_PREFIX = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Infinity)")


def _parse_float(value: Any) -> float:
    """``Number.parseFloat``: the longest numeric prefix, ``nan`` when there is none."""
    if isinstance(value, bool):
        value = str(value).lower()
    if isinstance(value, (int, float)):
        return float(value)
    match = _FLOAT_PREFIX.match(str(value) if value is not None else "")
    if match is None:
        return math.nan
    return float(match.group(1).replace("Infinity", "inf"))


def _per_million(value: Any, fallback: float) -> float:
    """Convert one $/token price to $/1M tokens. 0 means free.

    Missing, invalid and negative prices use the fallback: OpenRouter reports -1 for
    router models such as ``openrouter/auto``, whose price depends on the model they pick.
    """
    per_token = _parse_float(value)
    return round(per_token * 1_000_000, 6) if per_token >= 0 else fallback


def openrouter_cost(pricing: OpenRouterPricing | None) -> ModelCost:
    """Convert OpenRouter's $/token prices to pi's $/1M tokens.

    Missing, invalid and negative base prices are 0. Prompt-length overrides become
    request-wide tiers; a rate a tier does not list keeps the base price. Time-of-day
    overrides are skipped because :class:`ModelCost` cannot express them.
    """
    pricing = pricing or {}
    base_input = _per_million(pricing.get("prompt"), 0)
    base_output = _per_million(pricing.get("completion"), 0)
    base_cache_read = _per_million(pricing.get("input_cache_read"), 0)
    base_cache_write = _per_million(pricing.get("input_cache_write"), 0)
    tiers: list[ModelCostTier] = []
    for override in pricing.get("overrides") or []:
        if (
            override.get("min_prompt_tokens") is None
            or "utc_start" in override
            or "utc_end" in override
            or "utc_days" in override
        ):
            continue
        tiers.append(
            ModelCostTier(
                input_tokens_above=override["min_prompt_tokens"],  # type: ignore[typeddict-item]
                input=_per_million(override.get("prompt"), base_input),
                output=_per_million(override.get("completion"), base_output),
                cache_read=_per_million(override.get("input_cache_read"), base_cache_read),
                cache_write=_per_million(override.get("input_cache_write"), base_cache_write),
            )
        )
    return ModelCost(
        input=base_input,
        output=base_output,
        cache_read=base_cache_read,
        cache_write=base_cache_write,
        tiers=tiers or None,
    )
