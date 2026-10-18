"""Context derivation: the active transcript and the model context of a conversation (port of ``harness/context.ts``).

Python does not freeze values: a :class:`ContextRange` outlives a read, so its entries and messages are
immutable by contract, like the values a Storage returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

from pi_ai.types import AssistantMessage, Message, SystemMessage, TextContent, ToolCall, ToolResultMessage
from pi_chord.context import Context

from ..entries import SystemEntry
from ..ids import id_from_number
from ..types import ContextEdit, ConversationId, Cursor, EntryId, EntryRecord, HeadMarker, Storage
from .types import ContextView

__all__ = [
    "ContextBounds",
    "ContextRange",
    "active_entries",
    "capture_context_bounds",
    "derive_context",
    "empty_view",
    "order_tool_results",
    "read_context",
    "read_context_from",
]

_SCAN_PAGE_SIZE: Final = 256
_EXCLUDED_STOP_REASONS: Final = frozenset({"aborted", "error", "deferred"})
_MISSING_RESULT_TEXT: Final = "Tool result unavailable: history ends before this call completed."


@dataclass(frozen=True, slots=True)
class ContextBounds:
    """Head marker and newest visible entry that fix one committed context range."""

    head: HeadMarker | None
    tail: EntryId


async def capture_context_bounds(
    storage: Storage, conversation_id: ConversationId, context: Context, at: EntryId | None = None
) -> ContextBounds | None:
    """Capture the bounds of the current context, or of the context cut off at the visible entry ``at``.

    Two O(1) reads; run this on the Session line. Entries at or below the tail are immutable, so
    ``derive_context()`` can then scan them off the line.
    """
    if at is None:
        page = await storage.scan_entries({"conversationId": conversation_id}, 1, None, context)
        if len(page["items"]) == 0:
            return None
        tail = page["items"][0]["id"]
    else:
        if await storage.visible_entry(conversation_id, at, context) is None:
            raise RuntimeError(f"Entry {at} is not visible from conversation {conversation_id}")
        tail = at
    return ContextBounds(await storage.find_latest_head_marker(conversation_id, tail, context), tail)


#: A baseline's system message, ``None`` when the range cannot have one, or ``_OPEN`` while a later entry may be one.
_OPEN: Final[Any] = object()
Baseline: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class ContextRange:
    """One context read: the visible entries it scanned, oldest first, and the view derived from them.

    Entries at or below the tail never change, so a later read with the same head marker extends both.
    """

    bounds: ContextBounds
    entries: Sequence[EntryRecord]
    view: ContextView
    #: Targets of the edits in ``entries``.
    edited: frozenset[EntryId]
    #: ``view.messages`` from the contributed messages before the last assistant message.
    settled: Sequence[Message]
    #: Contributed messages from the last assistant message on, before tool result ordering.
    open: Sequence[Message]
    #: The baseline's system message; see ``_find_baseline()``.
    baseline: Baseline


async def read_context_from(
    session: Any,
    storage: Storage,
    conversation_id: ConversationId,
    context: Context,
    at: EntryId | None,
    previous: ContextRange | None,
) -> tuple[ContextView, ContextRange | None]:
    """``read_context()`` that reuses ``previous``, an earlier range of the same conversation.

    With the same head marker, only entries after its tail are scanned. Returns the view and the range to
    pass to the next read; the returned view has its own lists.
    """
    bounds = await session.read_on_line(lambda: capture_context_bounds(storage, conversation_id, context, at))
    if bounds is None:
        return empty_view(), None
    head_id = None if bounds.head is None else bounds.head["id"]
    previous_head = None if previous is None or previous.bounds.head is None else previous.bounds.head["id"]
    if previous is None or previous_head != head_id:
        entries = await _scan_range(storage, conversation_id, _range_query(bounds), context)
        current = _derive_range(bounds, entries)
    elif bounds.tail == previous.bounds.tail:
        current = previous
    elif bounds.tail < previous.bounds.tail:
        current = _derive_range(bounds, [entry for entry in previous.entries if entry["id"] <= bounds.tail])
    else:
        min_entry_id: EntryId = id_from_number(previous.bounds.tail + 1)
        added = await _scan_range(
            storage, conversation_id, {"minEntryId": min_entry_id, "maxEntryId": bounds.tail}, context
        )
        current = _extend_range(previous, bounds, added)
    view = current.view
    return (
        {
            "head": view["head"],
            "entries": list(view["entries"]),
            "contributions": list(view["contributions"]),
            "messages": list(view["messages"]),
        },
        current,
    )


def read_context(
    session: Any, storage: Storage, conversation_id: ConversationId, context: Context, at: EntryId | None = None
) -> asyncio.Future[ContextView]:
    """Committed context of one conversation: bounds captured on the Session line, entries derived off it.

    The bounds read joins the line at the call, as upstream's async function reaches it synchronously.
    """
    bounds = session.read_on_line(lambda: capture_context_bounds(storage, conversation_id, context, at))

    async def derive() -> ContextView:
        return await derive_context(storage, conversation_id, await bounds, context)

    return asyncio.ensure_future(derive())


async def derive_context(
    storage: Storage, conversation_id: ConversationId, bounds: ContextBounds | None, context: Context
) -> ContextView:
    """Derive the active transcript and model context of one conversation within captured bounds.

    H = newest visible head marker; the range runs from ``H.head`` (or transcript start) through the tail. Per
    target, the newest edit in the range wins. Context entries are H followed by the range's non-head entries.
    """
    if bounds is None:
        return empty_view()
    entries = await _scan_range(storage, conversation_id, _range_query(bounds), context)
    return _derive_range(bounds, entries).view


def empty_view() -> ContextView:
    return {"head": None, "entries": [], "contributions": [], "messages": []}


def _derive_range(bounds: ContextBounds, entries: Sequence[EntryRecord]) -> ContextRange:
    edits: dict[EntryId, ContextEdit] = {}
    # Edits of every entry in the range count, including older head markers that ``_select_active()`` drops.
    for entry in entries:
        for edit in entry.get("edits", ()):
            edits[edit["target"]] = edit
    active = _select_active(bounds.head, entries)
    contributions = [_contribute(entry, edits.get(entry["id"])) for entry in active]
    settled, open_messages = _settle([], [message for messages in contributions for message in messages])
    baseline = _find_baseline(bounds.head, active)
    messages = _lead_with_system([*settled, *order_tool_results(open_messages)], baseline)
    view: ContextView = {"head": bounds.head, "entries": active, "contributions": contributions, "messages": messages}
    return ContextRange(bounds, entries, view, frozenset(edits), settled, open_messages, baseline)


def _extend_range(previous: ContextRange, bounds: ContextBounds, added: Sequence[EntryRecord]) -> ContextRange:
    """``previous`` extended by the entries ``added`` after its tail under the same head marker.

    Only their contributions and the open messages are derived. An added edit can change an earlier entry, so
    it derives the whole range again.
    """
    entries = [*previous.entries, *added]
    if any("edits" in entry or "head" in entry or entry["id"] in previous.edited for entry in added):
        return _derive_range(bounds, entries)
    contributions = [_contribute(entry, None) for entry in added]
    settled, open_messages = _settle(
        previous.settled, [*previous.open, *(message for messages in contributions for message in messages)]
    )
    baseline = previous.baseline if previous.baseline is not _OPEN else _find_baseline(bounds.head, added)
    view: ContextView = {
        "head": bounds.head,
        "entries": [*previous.view["entries"], *added],
        "contributions": [*previous.view["contributions"], *contributions],
        "messages": _lead_with_system([*settled, *order_tool_results(open_messages)], baseline),
    }
    return ContextRange(bounds, entries, view, previous.edited, settled, open_messages, baseline)


def _find_baseline(head: HeadMarker | None, entries: Sequence[EntryRecord]) -> Baseline:
    """The baseline among ``entries``, oldest first.

    The system message of the first ``pi.system`` entry after the head marker, or of the transcript without
    one, when generation wrote it with every section and tool in effect. After a head marker it always does.
    Without one it does when it saw no system message before, which holds unless an earlier entry carries one,
    even if an edit omits it later. An edit that targets the baseline contributes other messages, or none, so
    none leads; a later entry does not take its place.
    """
    for entry in entries:
        if head is not None and entry["id"] <= head["id"]:
            continue
        if SystemEntry.is_(entry):
            return next((message for message in entry.get("model", ()) if isinstance(message, SystemMessage)), None)
        if head is None and _carries_system_message(entry):
            return None
    return _OPEN


def _carries_system_message(entry: EntryRecord) -> bool:
    """Whether an entry's own messages, or the replacement messages of its edits, include a system message."""
    if any(isinstance(message, SystemMessage) for message in entry.get("model", ())):
        return True
    return any(
        edit["action"] == "replace" and any(isinstance(message, SystemMessage) for message in edit["messages"])
        for edit in entry.get("edits", ())
    )


def _lead_with_system(messages: list[Message], baseline: Baseline) -> list[Message]:
    """Move the first system message to the front when only user messages precede it, or when it is the baseline's.

    A run's input is committed before generation renders the system prompt, so a transcript starts with the
    input followed by the baseline, and the range after a compaction with the kept history. Providers treat
    only a leading system message as the initial prompt and tool set. Only the first system message moves, so
    replaying system messages in order gives the same sections and tools. Any other system message, before an
    assistant message, stays where it was committed.
    """
    index = next((at for at, message in enumerate(messages) if isinstance(message, SystemMessage)), -1)
    if index <= 0:
        return messages
    system = messages[index]
    if system is not baseline and any(message.role != "user" for message in messages[:index]):
        return messages
    return [system, *messages[:index], *messages[index + 1 :]]


def _contribute(entry: EntryRecord, edit: ContextEdit | None) -> Sequence[Message]:
    """One active entry's model messages after its edit and excluded stop reasons, before tool result ordering."""
    if edit is not None and edit["action"] == "omit":
        return ()
    contributed = edit["messages"] if edit is not None and edit["action"] == "replace" else entry.get("model", ())
    return tuple(
        message
        for message in contributed
        if not isinstance(message, AssistantMessage) or message.stop_reason not in _EXCLUDED_STOP_REASONS
    )


def _settle(settled: Sequence[Message], open_messages: Sequence[Message]) -> tuple[list[Message], list[Message]]:
    """Move the ordered messages before the last assistant message of ``open`` to ``settled``.

    ``order_tool_results()`` of a sequence equals the concatenation over its parts when each later part starts
    with an assistant message.
    """
    last = -1
    for index in range(len(open_messages) - 1, -1, -1):
        if isinstance(open_messages[index], AssistantMessage):
            last = index
            break
    if last <= 0:
        return list(settled), list(open_messages)
    return [*settled, *order_tool_results(open_messages[:last])], list(open_messages[last:])


async def active_entries(
    storage: Storage, conversation_id: ConversationId, bounds: ContextBounds | None, context: Context
) -> list[EntryRecord]:
    """The raw active entries within captured bounds, without deriving model context."""
    if bounds is None:
        return []
    return list(_select_active(bounds.head, await _scan_range(storage, conversation_id, _range_query(bounds), context)))


def _range_query(bounds: ContextBounds) -> dict[str, Any]:
    """The visible range of ``bounds``: from the head marker's head, or transcript start, through the tail."""
    head = bounds.head
    if head is None:
        return {"maxEntryId": bounds.tail}
    return {"minEntryId": head["head"], "maxEntryId": bounds.tail}


async def _scan_range(
    storage: Storage, conversation_id: ConversationId, bounds: dict[str, Any], context: Context
) -> list[EntryRecord]:
    """Visible entries within ``bounds``, oldest first."""
    entries: list[EntryRecord] = []
    cursor: Cursor | None = None
    while True:
        page = await storage.scan_entries(
            {"conversationId": conversation_id, **bounds},  # type: ignore[typeddict-item]
            _SCAN_PAGE_SIZE,
            cursor,
            context,
        )
        entries.extend(page["items"])
        cursor = page.get("next")
        if cursor is None:
            break
    entries.reverse()
    return entries


def _select_active(head: HeadMarker | None, entries: Sequence[EntryRecord]) -> Sequence[EntryRecord]:
    """The head marker followed by the range's non-head entries, or the whole range without a marker."""
    if head is None:
        return list(entries)
    return [head, *(entry for entry in entries if "head" not in entry)]


def order_tool_results(messages: Sequence[Message]) -> list[Message]:
    """Place each assistant's tool results directly after it in call order.

    Results are taken from the messages before the next assistant; a missing result is synthesized and
    unmatched results are dropped.
    """
    ordered: list[Message] = []
    for index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            continue
        ordered.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        calls = [content for content in message.content if isinstance(content, ToolCall)]
        if len(calls) == 0:
            continue
        results: dict[str, int] = {}
        following = index + 1
        while following < len(messages) and not isinstance(messages[following], AssistantMessage):
            candidate = messages[following]
            if isinstance(candidate, ToolResultMessage) and candidate.tool_call_id not in results:
                results[candidate.tool_call_id] = following
            following += 1
        for call in calls:
            result_index = results.get(call.id)
            ordered.append(_missing_result(call, message.timestamp) if result_index is None else messages[result_index])
    return ordered


def _missing_result(call: ToolCall, timestamp: int) -> ToolResultMessage:
    return ToolResultMessage(
        call.id,
        call.name,
        [TextContent(_MISSING_RESULT_TEXT)],
        True,
        details={"reason": "missing_result"},
        timestamp=timestamp,
    )
