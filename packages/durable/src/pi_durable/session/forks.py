"""Document copies of a conversation fork (port of ``session/forks.ts``)."""

from __future__ import annotations

from typing import Literal, NamedTuple

from pi_chord.context import Context

from ..documents import address_id
from ..types import (
    ConversationId,
    Cursor,
    DocumentCopySource,
    DocumentCreate,
    DocumentId,
    DocumentPoint,
    DocumentScopeConversation,
    EntryId,
    Storage,
)

__all__ = ["ForkDocumentCopy", "prepare_fork_document_copies"]

_SCAN_PAGE_SIZE = 256

_ForkPolicy = Literal["asOf", "current"]


class ForkDocumentCopy(NamedTuple):
    """One definition-free document copy to create with a forked conversation."""

    record: DocumentCreate
    source: DocumentCopySource


async def prepare_fork_document_copies(
    storage: Storage,
    parent_conversation_id: ConversationId,
    at: EntryId,
    child_conversation_id: ConversationId,
    context: Context,
) -> list[ForkDocumentCopy]:
    """Select every persisted conversation document copied by one fork.

    ``asOf`` documents come from the conversation owning the fork entry, as of that entry's commit;
    ``current`` documents from the parent's current state.
    """
    entry = await storage.visible_entry(parent_conversation_id, at, context)
    if entry is None:
        raise RuntimeError(f"Entry {at} is not visible from conversation {parent_conversation_id}")

    copies: list[ForkDocumentCopy] = []
    copied_addresses: set[str] = set()
    await _collect_copies(
        storage,
        {"kind": "conversation", "conversationId": entry["entry"]["conversationId"]},
        entry["commitSeq"],
        "asOf",
        child_conversation_id,
        copies,
        copied_addresses,
        context,
    )
    await _collect_copies(
        storage,
        {"kind": "conversation", "conversationId": parent_conversation_id},
        "current",
        "current",
        child_conversation_id,
        copies,
        copied_addresses,
        context,
    )
    return copies


async def _collect_copies(
    storage: Storage,
    scope: DocumentScopeConversation,
    at: DocumentPoint,
    policy: _ForkPolicy,
    child_conversation_id: ConversationId,
    copies: list[ForkDocumentCopy],
    copied_addresses: set[str],
    context: Context,
) -> None:
    cursor: Cursor | None = None
    while True:
        page = await storage.scan_documents({"scope": scope, "at": at}, _SCAN_PAGE_SIZE, cursor, context)
        for source in page["items"]:
            if source["scope"]["kind"] != "conversation" or source.get("fork") != policy:
                continue
            id = DocumentId(await storage.mint_id())
            record: dict[str, object] = {"id": id, "kind": source["kind"]}
            if "key" in source:
                record["key"] = source["key"]
            record["scope"] = {"kind": "conversation", "conversationId": child_conversation_id}
            record["history"] = "latest" if source.get("history") == "latest" else "rewindable"
            record["fork"] = source.get("fork")
            create: DocumentCreate = record  # type: ignore[assignment]
            copy_address = address_id(create)  # type: ignore[arg-type]
            if copy_address in copied_addresses:
                member = create["kind"] if "key" not in create else f"{create['kind']}/{create['key']}"
                raise RuntimeError(f"Fork selects multiple source documents for {member}")
            copied_addresses.add(copy_address)
            copies.append(ForkDocumentCopy(create, {"id": source["id"], "at": at}))
        cursor = page.get("next")
        if cursor is None:
            break
