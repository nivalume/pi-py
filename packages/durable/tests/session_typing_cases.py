"""Static typing cases of the Session surface (the compile-time half of upstream ``test/session-definitions.test.ts``).

``test_session_definitions.py`` runs pyright on this file with ``reportUnnecessaryTypeIgnoreComment``:
every line marked ``# pyright: ignore`` must be a type error, and every ``assert_type`` must hold.
Nothing here runs at import time except definitions.

Not expressible in Python and so not checked: a typed entry kind whose data is required
(``TypedEntryDraft.data`` is ``NotRequired``), and an entry kind without data rejecting ``data``.
"""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict, assert_type

from pi_chord.context import Context
from pi_chord.state import AttachedReplicatedState
from pi_durable import (
    ROOT_CONVERSATION_ID,
    Entry,
    EntryRecord,
    Session,
    Tx,
    TypedEntry,
    WatchHandle,
    define_doc,
    define_doc_family,
    define_entry,
)
from pi_durable.types import (
    ConversationDocFamilyToken,
    ConversationDocToken,
    ConversationId,
    EntryId,
    RewindableConversationDocFamilyToken,
    RewindableConversationDocToken,
    SessionDocFamilyToken,
    SessionDocToken,
    TaskDocFamilyToken,
    TaskDocToken,
    TaskId,
)


class State(TypedDict):
    value: int


def initial() -> State:
    return {"value": 0}


def seeded(seed: int) -> State:
    return {"value": seed}


SessionDoc: SessionDocToken[State] = define_doc(kind="t.session", version=1, scope="session", initial=initial)
LatestDoc: ConversationDocToken[State] = define_doc(
    kind="t.latest", version=1, scope="conversation", history="latest", fork="current", initial=initial
)
RewindableDoc: RewindableConversationDocToken[State] = define_doc(
    kind="t.rewindable", version=1, scope="conversation", history="rewindable", fork="asOf", initial=initial
)
TaskDoc: TaskDocToken[State] = define_doc(kind="t.task", version=1, scope="task", initial=initial)
SessionFamily: SessionDocFamilyToken[State, int] = define_doc_family(
    kind="t.session-family", version=1, scope="session", initial=seeded
)
ConversationFamily: RewindableConversationDocFamilyToken[State, int] = define_doc_family(
    kind="t.conversation-family", version=1, scope="conversation", history="rewindable", fork="initial", initial=seeded
)
LatestFamily: ConversationDocFamilyToken[State, int] = define_doc_family(
    kind="t.latest-family", version=1, scope="conversation", history="latest", fork="initial", initial=seeded
)
TaskFamily: TaskDocFamilyToken[State, int] = define_doc_family(
    kind="t.task-family", version=1, scope="task", initial=seeded
)


class NoteData(TypedDict):
    text: str


Note: Entry[NoteData] = define_entry("t.note")

StateOf: TypeAlias = AttachedReplicatedState[State | None] | None
WatchOf: TypeAlias = WatchHandle[State | None] | None


async def types_every_owner_key_and_seed_overload(
    tx: Tx, conversation_id: ConversationId, task_id: TaskId[Any]
) -> None:
    assert_type(await tx.doc(SessionDoc), State)
    assert_type(await tx.doc(LatestDoc, conversation_id), State)
    assert_type(await tx.doc(RewindableDoc, conversation_id), State)
    assert_type(await tx.doc(TaskDoc, task_id), State)
    assert_type(await tx.doc(SessionFamily, "k", 1), State)
    assert_type(await tx.doc(ConversationFamily, conversation_id, "k", 1), State)
    assert_type(await tx.doc(TaskFamily, task_id, "k", 1), State)
    await tx.retire_doc(SessionDoc)
    await tx.retire_doc(LatestDoc, conversation_id)
    await tx.retire_doc(TaskDoc, task_id)
    await tx.retire_doc(SessionFamily, "k")
    await tx.retire_doc(ConversationFamily, conversation_id, "k")
    await tx.retire_doc(TaskFamily, task_id, "k")

    # Session documents take no owner.
    await tx.doc(SessionDoc, conversation_id)  # pyright: ignore
    # Conversation documents require a conversation ID.
    await tx.doc(LatestDoc)  # pyright: ignore
    # Family access requires a creation seed.
    await tx.doc(SessionFamily, "k")  # pyright: ignore
    # Family seeds are typed.
    await tx.doc(SessionFamily, "k", "seed")  # pyright: ignore
    # Family retirement takes no seed.
    await tx.retire_doc(TaskFamily, task_id, "k", 1)  # pyright: ignore


async def types_snapshots(session: Session, context: Context, task_id: TaskId[Any]) -> None:
    assert_type(await session.snapshot(SessionDoc, context), State | None)
    assert_type(await session.snapshot(LatestDoc, ROOT_CONVERSATION_ID, context), State | None)
    assert_type(await session.snapshot(TaskDoc, task_id, context), State | None)
    assert_type(await session.snapshot(SessionFamily, "k", context), State | None)
    assert_type(await session.snapshot(ConversationFamily, ROOT_CONVERSATION_ID, "k", context), State | None)
    assert_type(await session.snapshot(TaskFamily, task_id, "k", context), State | None)
    # Snapshots never take a creation seed.
    await session.snapshot(SessionFamily, "k", 1, context)  # pyright: ignore


async def types_historical_reads_states_watches_and_typed_entries(
    session: Session, tx: Tx, conversation_id: ConversationId, task_id: TaskId[Any], at: EntryId, context: Context
) -> None:
    assert_type(await session.snapshot_as_of(RewindableDoc, conversation_id, at, context), State | None)
    assert_type(await session.snapshot_as_of(ConversationFamily, conversation_id, "k", at, context), State | None)
    # Latest conversation documents keep no history.
    await session.snapshot_as_of(LatestDoc, conversation_id, at, context)  # pyright: ignore
    await session.snapshot_as_of(LatestFamily, conversation_id, "k", at, context)  # pyright: ignore
    # Session and task documents keep no history.
    await session.snapshot_as_of(SessionDoc, conversation_id, at, context)  # pyright: ignore
    await session.snapshot_as_of(TaskDoc, task_id, at, context)  # pyright: ignore

    assert_type(await session.document_state(SessionDoc, context), StateOf)
    assert_type(await session.document_state(LatestDoc, conversation_id, context), StateOf)
    assert_type(await session.document_state(TaskDoc, task_id, context), StateOf)
    assert_type(await session.document_state(SessionFamily, "k", context), StateOf)
    assert_type(await session.document_state(ConversationFamily, conversation_id, "k", context), StateOf)
    assert_type(await session.document_state(TaskFamily, task_id, "k", context), StateOf)
    assert_type(await session.watch_doc(SessionDoc, context), WatchOf)
    assert_type(await session.watch_doc(LatestDoc, conversation_id, context), WatchOf)
    assert_type(await session.watch_doc(TaskDoc, task_id, context), WatchOf)
    assert_type(await session.watch_doc(SessionFamily, "k", context), WatchOf)
    assert_type(await session.watch_doc(ConversationFamily, conversation_id, "k", context), WatchOf)
    assert_type(await session.watch_doc(TaskFamily, task_id, "k", context), WatchOf)
    # A watch of a Session document takes no owner.
    await session.watch_doc(SessionDoc, conversation_id, context)  # pyright: ignore

    assert_type(await tx.entry(at), EntryRecord | None)
    assert_type(await tx.entry(Note, at), TypedEntry[NoteData] | None)
    assert_type(await tx.append_entry(conversation_id, {"kind": "t.raw"}), EntryRecord)
    note = await tx.append_entry(Note, conversation_id, {"data": {"text": "x"}})
    assert_type(note, TypedEntry[NoteData])
    assert_type(note["data"]["text"], str)
    # Data is typed by the token.
    await tx.append_entry(Note, conversation_id, {"data": {"text": 1}})  # pyright: ignore
    # The token supplies the kind.
    await tx.append_entry(Note, conversation_id, {"kind": "t.note", "data": {"text": "x"}})  # pyright: ignore
