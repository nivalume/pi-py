"""The inbox document and the boundary rules that place queued submissions (port of ``harness/inbox.ts``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from pi_ai.types import UserMessage
from pi_chord.json import copy_json

from ..documents import define_doc
from ..entries import UserEntry
from ..types import ConversationDocToken, ConversationId, EntryDraft, EntryId, SubmissionId, Tx
from .types import InputEntry, QueueMode, Settings, UserInput

__all__ = [
    "Boundary",
    "BoundaryResult",
    "InboxDoc",
    "InboxItem",
    "InboxItemInput",
    "InboxItemWrite",
    "InboxState",
    "QueueModes",
    "append_input",
    "draft_json",
    "apply_boundary",
    "has_queued_reset",
    "input_content",
    "is_display_only",
    "is_stale",
    "pick_input_entry",
    "prepare_boundary",
    "remove_inbox_item",
    "user_content_json",
    "withdraw_queued_inputs",
]


class InboxItemInput(TypedDict):
    """Queued user input for a run, with the entry it is placed as when not ``pi.user``."""

    id: SubmissionId
    mode: Literal["steer", "followUp"]
    #: ``UserInput`` as JSON: a string, or content blocks in ``pi_ai.serialization`` form.
    content: Any
    entry: NotRequired[InputEntry]


class InboxItemWrite(TypedDict):
    """A queued entry write; ``entry`` is an ``EntryDraft`` stored as plain JSON (messages serialized)."""

    id: SubmissionId
    mode: Literal["write"]
    entry: dict[str, Any]


#: A queued submission.
InboxItem: TypeAlias = InboxItemInput | InboxItemWrite


class InboxState(TypedDict):
    """Built-in queue of one conversation's submissions waiting for a boundary, in ID order."""

    items: list[InboxItem]


InboxDoc: ConversationDocToken[InboxState] = define_doc(
    kind="pi.inbox",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"items": []},
    checkpoint_when=lambda value, *_args: len(value["items"]) == 0,
)

#: The settings a boundary reads, on the Session line (``Pick<Settings, "steeringMode" | "followUpMode">``).
QueueModes: TypeAlias = Settings


@dataclass(slots=True)
class Boundary:
    """What a boundary reads before the commit's first table write, and the newest head it has seen so far."""

    conversation_id: ConversationId
    inbox: Any
    steering_mode: QueueMode
    follow_up_mode: QueueMode
    #: Start of the active range, the newest head marker's ``head``; advanced by heads written in this commit.
    head: EntryId | None


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """Selected user items, in ID order, and whether a ``head: "self"`` write (a reset) was placed."""

    users: list[SubmissionId]
    reset: bool


async def prepare_boundary(tx: Tx, conversation_id: ConversationId, modes: Any) -> Boundary:
    """Read what a boundary needs. Table reads must precede the commit's first table write, so callers prepare
    the boundary at the start of their commit."""
    marker = await tx.latest_head_marker(conversation_id)
    inbox = await tx.doc(InboxDoc, conversation_id)
    return Boundary(
        conversation_id=conversation_id,
        inbox=inbox,
        steering_mode=modes["steeringMode"],
        follow_up_mode=modes["followUpMode"],
        head=None if marker is None else marker["head"],
    )


def user_content_json(content: UserInput) -> Any:
    """``UserInput`` as JSON: a string stays, content blocks are serialized with ``pi_ai.serialization``."""
    from pi_ai.serialization import to_json_value

    if isinstance(content, str):
        return content
    return copy_json(to_json_value(list(content)))


def input_content(content: Any) -> UserInput:
    """``UserInput`` from its JSON (or from content blocks already decoded)."""
    if isinstance(content, str):
        return content
    from pi_ai.serialization import content_from_dict

    return [item if not isinstance(item, dict) else content_from_dict(dict(item)) for item in content]


def _draft_json(entry: Any) -> dict[str, Any]:
    """An ``EntryDraft`` as strict JSON for the inbox: messages and edit messages serialized."""
    from pi_ai.serialization import to_json_value

    return copy_json(to_json_value(entry), omit_undefined_properties=True)  # type: ignore[return-value]


def _draft_from_json(stored: Any) -> EntryDraft:
    """The ``EntryDraft`` an inbox write holds, with its messages decoded."""
    from pi_ai.serialization import message_from_dict

    draft: dict[str, Any] = copy_json(stored)  # type: ignore[assignment]
    if "model" in draft:
        draft["model"] = [message_from_dict(message) for message in draft["model"]]
    if "edits" in draft:
        draft["edits"] = [
            {**edit, "messages": [message_from_dict(message) for message in edit["messages"]]}
            if edit.get("action") == "replace"
            else edit
            for edit in draft["edits"]
        ]
    return draft  # type: ignore[return-value]


async def apply_boundary(tx: Tx, boundary: Boundary, at: Literal["postTools", "final"], now: float) -> BoundaryResult:
    """Place the queued items a boundary selects (spec §6).

    Every write, the first or all steers, and at ``final`` the first or all follow-ups. A selected reset turns
    a ``postTools`` boundary into ``final``. Writes are placed first and user items after them, each in ID
    order, so user items queued before a reset run in the new context. A write whose head targets an entry
    before the active range, including a range started earlier in this commit, is stale. Selected and stale
    items are removed positionally.
    """
    conversation_id = boundary.conversation_id
    items = boundary.inbox["items"]
    reset = has_queued_reset(items)
    final = at == "final" or reset

    def pick(mode: str, queue_mode: QueueMode) -> list[int]:
        indexes = [index for index, item in enumerate(items) if item["mode"] == mode]
        return indexes if queue_mode == "all" else indexes[:1]

    writes = [index for index, item in enumerate(items) if item["mode"] == "write"]
    users = [*pick("steer", boundary.steering_mode), *(pick("followUp", boundary.follow_up_mode) if final else [])]
    users.sort()

    # ``append_entry()`` copies the drafts' values; the items are removed only afterwards.
    for index in writes:
        item = items[index]
        draft = _draft_from_json(item["entry"])
        if is_stale(boundary, draft):
            tx.settle_submission(item["id"], {"status": "unanswered", "reason": "stale"})
            continue
        entry = await tx.append_entry(conversation_id, draft)
        head = draft.get("head")
        if head is not None:
            boundary.head = entry["id"] if head == "self" else head
        tx.place_submission(item["id"], entry["id"])
    placed: list[SubmissionId] = []
    for index in users:
        item = items[index]
        stored_entry = item.get("entry")
        entry = await append_input(
            tx,
            conversation_id,
            input_content(item["content"]),
            None if stored_entry is None else copy_json(stored_entry),  # type: ignore[arg-type]
            now,
        )
        tx.place_submission(item["id"], entry["id"])
        placed.append(item["id"])
    for index in sorted([*writes, *users], reverse=True):
        del items[index]
    return BoundaryResult(placed, reset)


def append_input(
    tx: Tx, conversation_id: ConversationId, content: UserInput, entry: InputEntry | None, now: float
) -> Any:
    """Append the entry an input is placed as: ``entry``'s kind and data, or ``pi.user``, with the user message
    of ``content``. Returns the ``append_entry()`` future."""
    message = UserMessage(content if isinstance(content, str) else list(content), timestamp=now)  # type: ignore[arg-type]
    if entry is None:
        return tx.append_entry(UserEntry, conversation_id, {"model": [message]})
    return tx.append_entry(conversation_id, {**pick_input_entry(entry), "model": [message]})  # type: ignore[typeddict-item]


def pick_input_entry(entry: InputEntry) -> InputEntry:
    """The kind and data of an input's entry as strict JSON, without other fields an untyped caller might pass."""
    picked: dict[str, Any] = {"kind": entry["kind"]}
    if "data" in entry:
        picked["data"] = entry["data"]
    return copy_json(picked, omit_undefined_properties=True)  # type: ignore[return-value]


def is_display_only(entry: EntryDraft) -> bool:
    """Whether a write is display-only: it has no ``model``, ``head``, or ``edits`` (present counts, even if
    empty), so it changes nothing the model sees and needs no boundary."""
    return "model" not in entry and "head" not in entry and "edits" not in entry


def has_queued_reset(items: Sequence[Any]) -> bool:
    """Whether a reset (``head: "self"`` write) is queued; entries appended before it leave the active range when
    it lands."""
    return any(item["mode"] == "write" and item["entry"].get("head") == "self" for item in items)


def is_stale(boundary: Boundary, entry: Any) -> bool:
    """Whether a head write targets an entry before the active range, so placing it would bring back cut history."""
    head = entry.get("head")
    return isinstance(head, int) and not isinstance(head, bool) and boundary.head is not None and head < boundary.head


async def remove_inbox_item(tx: Tx, conversation_id: ConversationId, id: SubmissionId) -> None:
    """Remove a withdrawn submission's item; the caller settles the submission."""
    items = (await tx.doc(InboxDoc, conversation_id))["items"]
    for index, item in enumerate(items):
        if item["id"] == id:
            del items[index]
            return


async def withdraw_queued_inputs(tx: Tx, conversation_id: ConversationId, reason: str = "aborted") -> None:
    """Withdraw every queued input of a conversation, as ``Conversation.abort()`` and abort cascades do.

    Each settles ``unanswered`` with ``reason`` (default ``aborted``) and leaves the inbox; queued writes stay
    for later placement.
    """
    items = (await tx.doc(InboxDoc, conversation_id))["items"]
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if item["mode"] == "write":
            continue
        tx.settle_submission(item["id"], {"status": "unanswered", "reason": reason})
        del items[index]


def draft_json(entry: EntryDraft) -> dict[str, Any]:
    """An ``EntryDraft`` as the strict JSON a queued write stores (messages serialized)."""
    return _draft_json(entry)
