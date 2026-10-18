"""The usage ledger document (port of ``harness/usage.ts``).

Usage is stored as its JSON (``pi_ai.serialization``: ``cacheRead``, ``totalTokens``, ``cost.total``, ...),
exactly what TypeScript pi writes. The helpers accept a :class:`pi_ai.Usage` or its JSON.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, TypedDict

from pi_ai.serialization import to_json_value
from pi_chord.json import copy_json

from ..documents import define_doc
from ..types import ConversationDocToken, ConversationId, Tx

__all__ = ["UsageDoc", "UsageState", "add_usage", "add_usage_state", "record_usage", "usage_json"]


class UsageState(TypedDict):
    """Ledger of one conversation's own spend: its entries, and compaction summarization attempts, which have none."""

    #: Assistant entries and summarization attempts, keyed ``provider/modelId`` (``Usage`` JSON).
    models: dict[str, Any]
    #: Tool results, keyed by tool name; their usage has no model identity.
    tools: dict[str, Any]


UsageDoc: ConversationDocToken[UsageState] = define_doc(
    kind="pi.usage",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"models": {}, "tools": {}},
    checkpoint_when=lambda *_args: True,
)


def usage_json(usage: Any) -> dict[str, Any]:
    """A usage as strict JSON: a :class:`pi_ai.Usage` serialized, or a JSON usage copied."""
    if isinstance(usage, dict):
        return copy_json(usage)  # type: ignore[return-value]
    return copy_json(to_json_value(usage))  # type: ignore[return-value]


async def record_usage(
    tx: Tx, conversation_id: ConversationId, bucket: Literal["models", "tools"], key: str, usage: Any
) -> None:
    """Add ``usage`` to one bucket of the conversation's ``pi.usage``, in the commit that records the response."""
    totals = (await tx.doc(UsageDoc, conversation_id))[bucket]
    total = totals.get(key)
    if total is None:
        totals[key] = usage_json(usage)
    else:
        add_usage(total, usage)


def add_usage(total: Any, usage: Any) -> None:
    """Add every counter of ``usage`` to the JSON ``total``; optional counters are added once either side reports them."""
    value = usage_json(usage)
    for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
        total[key] = total.get(key, 0) + value.get(key, 0)
    for key in ("cacheWrite1h", "reasoning"):
        if key in value:
            total[key] = total.get(key, 0) + value[key]
    cost = total.get("cost")
    if cost is None:
        cost = {}
        total["cost"] = cost
        cost = total["cost"]
    added = value.get("cost", {})
    for key in ("input", "output", "cacheRead", "cacheWrite", "total"):
        cost[key] = cost.get(key, 0) + added.get(key, 0)


def add_usage_state(total: UsageState, state: UsageState) -> None:
    """Add every bucket of ``state`` into ``total``."""
    for bucket in ("models", "tools"):
        sums = total[bucket]
        for key, usage in state[bucket].items():
            existing = sums.get(key)
            if existing is not None:
                add_usage(existing, usage)
            else:
                sums[key] = copy.deepcopy(usage)
