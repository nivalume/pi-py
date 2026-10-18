"""Port of ``test/session-documents.test.ts``."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime
from typing import Any

import pytest
from durable_session_support import (
    StorageBackend,
    context,
    create_conversation,
    document_changes,
    flush,
    open_test_session,
)

from pi_chord.json import UNDEFINED
from pi_durable import SessionFailed, define_doc, define_doc_family
from pi_durable.ids import id_from_number

live_init_count = 0


def _live_initial() -> dict[str, Any]:
    global live_init_count
    live_init_count += 1
    return {"items": [], "nested": {"count": 0}, "other": {"label": "x"}}


LiveDoc = define_doc(
    kind="test.live", version=1, scope="conversation", history="latest", fork="initial", initial=_live_initial
)
RewindableLiveDoc = define_doc(
    kind="test.live",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: {"items": [], "nested": {"count": 0}, "other": {"label": "x"}},
)
LiveDocV2 = define_doc(
    kind="test.live",
    version=2,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"items": [], "nested": {"count": 0}, "other": {"label": "x"}},
)
CounterDoc = define_doc(kind="test.counter", version=1, scope="session", initial=lambda: {"count": 0})

seeds: list[str] = []


def _member_initial(seed: str) -> dict[str, Any]:
    seeds.append(seed)
    return {"seed": seed, "hits": 0}


MemberDoc = define_doc_family(kind="test.member", version=1, scope="session", initial=_member_initial)


async def setup_live(backend: StorageBackend) -> Any:
    opened = await open_test_session(backend)
    conversation_id = await create_conversation(opened.session)

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["items"].extend(["a", "b"])

    await opened.session.commit(change, context)
    return (*opened, conversation_id)


async def test_creates_an_initial_base_on_first_access_and_adopts_it_after_storage_success(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["message"] = "hello"

    await session.commit(change, context)
    writes = storage.commits[-1]
    assert len(writes) == 1
    write: Any = writes[0]
    assert write["type"] == "document.create"
    assert write["record"]["kind"] == "test.live"
    assert write["record"]["scope"] == {"kind": "conversation", "conversationId": conversation_id}
    assert write["record"]["history"] == "latest"
    assert write["record"]["fork"] == "initial"
    assert write["content"]["kind"] == "base" and write["content"]["version"] == 1
    assert write["content"]["value"]["message"] == "hello"
    assert write["content"]["value"]["items"] == [] and write["content"]["value"]["nested"] == {"count": 0}
    snapshot = await session.snapshot(LiveDoc, conversation_id, context)
    assert snapshot == {"message": "hello", "items": [], "nested": {"count": 0}, "other": {"label": "x"}}
    await flush()
    publication = publications[-1]
    published = document_changes(publication)[0]
    assert published["record"]["createdAt"] == publication["seq"]
    assert published["value"] is snapshot
    assert published["conversationId"] == conversation_id
    create: Any = next(w for w in storage.admitted_commits[-1] if w["type"] == "document.create")
    assert create["content"]["value"] is published["value"]


async def test_never_creates_on_snapshot_and_returns_none_when_absent(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    before = len(storage.commits)
    assert await session.snapshot(LiveDoc, conversation_id, context) is None
    assert await session.snapshot(CounterDoc, context) is None
    assert await session.snapshot(MemberDoc, "k", context) is None
    assert len(storage.commits) == before
    assert storage.mint_count == 1


async def test_returns_shared_immutable_snapshots_and_keeps_prior_revisions_stable(
    storage_backend: StorageBackend,
) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    first = await session.snapshot(LiveDoc, conversation_id, context)
    assert await session.snapshot(LiveDoc, conversation_id, context) is first

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["nested"]["count"] = 1

    await session.commit(change, context)
    second = await session.snapshot(LiveDoc, conversation_id, context)
    assert second is not first
    assert first["nested"]["count"] == 0
    assert second["nested"]["count"] == 1
    # Unchanged subtrees are structurally shared between immutable revisions.
    assert second["items"] is first["items"]
    assert second["other"] is first["other"]


async def test_adopts_by_pointer_swap_and_shares_operation_payloads_with_the_published_revision(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["other"] = {"label": "y"}
        live["items"].append("c")

    await session.commit(change, context)
    await flush()
    snapshot = await session.snapshot(LiveDoc, conversation_id, context)
    published = document_changes(publications[-1])[0]
    assert published["value"] is snapshot
    admitted: Any = next(w for w in storage.admitted_commits[-1] if w["type"] == "document.change")
    assert admitted["content"]["kind"] == "delta"
    assert published["ops"] is admitted["content"]["ops"]
    set_op = next(op for op in published["ops"] if op[0] == "s")
    assert set_op == ("s", ("other",), {"label": "y"})
    # Trusted immutability: the Session makes no second copy of operation payloads.
    assert set_op[2] is snapshot["other"]


async def test_copies_assigned_values_per_placement(storage_backend: StorageBackend) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    value = {"label": "shared"}

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["other"] = value
        live["copy"] = value
        value["label"] = "mutated"

    await session.commit(change, context)
    snapshot = await session.snapshot(LiveDoc, conversation_id, context)
    assert snapshot["other"] == {"label": "shared"}
    assert snapshot["copy"] == {"label": "shared"}
    assert snapshot["copy"] is not snapshot["other"]
    assert snapshot["other"] is not value


async def test_suppresses_writes_and_publications_for_empty_batches(storage_backend: StorageBackend) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    await flush()
    commits = len(storage.commits)
    published = len(publications)

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["nested"]["count"] = 0
        live["items"].append("z")
        live["items"].pop()

    await session.commit(change, context)
    await flush()
    assert len(storage.commits) == commits
    assert len(publications) == published


async def test_writes_and_publishes_replayable_nonempty_structural_no_ops(storage_backend: StorageBackend) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    before = await session.snapshot(LiveDoc, conversation_id, context)

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        first = live["items"].pop(0)
        live["items"].insert(0, first)

    await session.commit(change, context)
    await flush()
    write: Any = storage.commits[-1][0]
    assert write["type"] == "document.change" and write["content"]["kind"] == "delta"
    after = await session.snapshot(LiveDoc, conversation_id, context)
    assert after == before
    assert after is not before
    assert len(document_changes(publications[-1])[0]["ops"]) > 0


async def test_revokes_escaped_drafts_when_the_callback_settles(storage_backend: StorageBackend) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    escaped: list[Any] = []

    async def change(tx: Any) -> None:
        draft = await tx.doc(LiveDoc, conversation_id)
        escaped.extend([draft, draft["items"]])
        draft["message"] = "inside"

    await session.commit(change, context)
    draft, items = escaped
    with pytest.raises(TypeError):
        draft["message"]
    with pytest.raises(TypeError):
        draft["message"] = "outside"
    with pytest.raises(TypeError):
        len(items)
    with pytest.raises(TypeError):
        items.append("outside")
    assert (await session.snapshot(LiveDoc, conversation_id, context))["message"] == "inside"

    returned = await session.commit(lambda tx: tx.doc(LiveDoc, conversation_id), context)
    with pytest.raises(TypeError):
        returned["message"]


async def test_aborts_every_change_when_the_callback_fails(storage_backend: StorageBackend) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)
    before = await session.snapshot(LiveDoc, conversation_id, context)
    commits = len(storage.commits)

    async def failing(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        counter = await tx.doc(CounterDoc)
        live["message"] = "lost"
        counter["count"] = 5
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await session.commit(failing, context)
    assert len(storage.commits) == commits
    assert await session.snapshot(LiveDoc, conversation_id, context) is before
    assert await session.snapshot(CounterDoc, context) is None

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "kept"

    await session.commit(change, context)
    assert (await session.snapshot(LiveDoc, conversation_id, context))["message"] == "kept"


async def test_memoizes_concurrent_duplicate_acquisition_and_initializes_once(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    init_count = live_init_count
    mints = storage.mint_count

    async def change(tx: Any) -> None:
        first, second = await asyncio.gather(tx.doc(LiveDoc, conversation_id), tx.doc(LiveDoc, conversation_id))
        assert first is second
        assert await tx.doc(LiveDoc, conversation_id) is first
        first["message"] = "once"

    await session.commit(change, context)
    assert live_init_count == init_count + 1
    assert storage.mint_count == mints + 1
    assert len([w for w in storage.commits[-1] if w["type"] == "document.create"]) == 1


async def test_uses_the_first_family_seed_and_ignores_seeds_for_existing_members(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await open_test_session(storage_backend)
    seeds.clear()

    async def first_change(tx: Any) -> None:
        first = await tx.doc(MemberDoc, "k", "first")
        second = await tx.doc(MemberDoc, "k", "second")
        assert second is first
        first["hits"] += 1

    await session.commit(first_change, context)

    async def second_change(tx: Any) -> None:
        (await tx.doc(MemberDoc, "k", "third"))["hits"] += 1
        (await tx.doc(MemberDoc, "other", "fourth"))["hits"] += 1

    await session.commit(second_change, context)
    assert seeds == ["first", "fourth"]
    assert await session.snapshot(MemberDoc, "k", context) == {"seed": "first", "hits": 2}
    assert await session.snapshot(MemberDoc, "other", context) == {"seed": "fourth", "hits": 1}


async def test_rejects_a_callback_that_succeeds_with_a_pending_acquisition_and_drains_it(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)
    await session.unload_documents()
    gate = storage.hold_find_document()
    commits = len(storage.commits)
    pending: list[Any] = []
    commit = session.commit(lambda tx: pending.append(tx.doc(LiveDoc, conversation_id)), context)
    await gate.entered
    settled: list[bool] = []
    commit.add_done_callback(lambda _future: settled.append(True))
    await flush()
    # The line stays held until the late acquisition settles.
    assert settled == []
    gate.release()
    with pytest.raises(RuntimeError, match="pending Tx operations"):
        await commit
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        await pending[0]
    assert len(storage.commits) == commits

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "after"

    await session.commit(change, context)
    assert (await session.snapshot(LiveDoc, conversation_id, context))["message"] == "after"


async def test_does_not_initialize_or_mint_for_an_absent_acquisition_that_finishes_after_settlement(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    initialized: list[int] = []

    def initial() -> dict[str, Any]:
        initialized.append(1)
        return {"count": 0}

    LateDoc = define_doc(kind="test.late", version=1, scope="session", initial=initial)
    gate = storage.hold_find_document()
    mints = storage.mint_count
    pending: list[Any] = []
    commit = session.commit(lambda tx: pending.append(tx.doc(LateDoc)), context)
    await gate.entered
    gate.release()
    with pytest.raises(RuntimeError, match="pending Tx operations"):
        await commit
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        await pending[0]
    assert initialized == []
    assert storage.mint_count == mints


async def test_rejects_with_the_callback_error_when_it_fails_with_a_pending_acquisition(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)
    await session.unload_documents()
    gate = storage.hold_find_document()
    pending: list[Any] = []

    def failing(tx: Any) -> None:
        pending.append(tx.doc(LiveDoc, conversation_id))
        raise RuntimeError("callback failed")

    commit = session.commit(failing, context)
    await gate.entered
    gate.release()
    with pytest.raises(RuntimeError, match="callback failed"):
        await commit
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        await pending[0]


async def test_rejects_tx_use_after_the_callback_settles(storage_backend: StorageBackend) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    captured: list[Any] = []
    await session.commit_with(lambda tx: captured.append(tx), context)
    tx = captured[0]
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        await tx.doc(LiveDoc, conversation_id)
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        await tx.conversation(conversation_id)
    with pytest.raises(RuntimeError, match="Transaction has settled"):
        tx.set_task({})


async def test_rejects_tokens_whose_semantics_or_version_disagree_with_the_stored_incarnation(
    storage_backend: StorageBackend,
) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    with pytest.raises(TypeError, match="does not match the supplied definition semantics"):
        await session.snapshot(RewindableLiveDoc, conversation_id, context)

    async def rewindable(tx: Any) -> None:
        await tx.doc(RewindableLiveDoc, conversation_id)

    with pytest.raises(TypeError, match="does not match the supplied definition semantics"):
        await session.commit(rewindable, context)

    async def newer(tx: Any) -> None:
        await tx.doc(LiveDocV2, conversation_id)

    with pytest.raises(RuntimeError, match="requires migration from version 1"):
        await session.commit(newer, context)


async def test_rejects_non_json_initializer_values_and_draft_placements_before_storage_admission(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    DateDoc = define_doc(kind="test.date", version=1, scope="session", initial=lambda: {"at": datetime.now()})

    async def date(tx: Any) -> None:
        await tx.doc(DateDoc)

    with pytest.raises(TypeError, match="strict JSON"):
        await session.commit(date, context)
    commits = len(storage.commits)

    async def undefined(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["items"].append(UNDEFINED)

    with pytest.raises(TypeError, match="strict JSON"):
        await session.commit(undefined, context)
    assert len(storage.commits) == commits
    assert await session.snapshot(LiveDoc, conversation_id, context) is None


async def test_rolls_back_prepared_documents_when_batch_assembly_fails(storage_backend: StorageBackend) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)

    async def count_one(tx: Any) -> None:
        (await tx.doc(CounterDoc))["count"] = 1

    await session.commit(count_one, context)
    live = await session.snapshot(LiveDoc, conversation_id, context)
    counter = await session.snapshot(CounterDoc, context)
    commits = len(storage.commits)

    async def failing(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "lost"
        (await tx.doc(CounterDoc))["count"] = 2
        # Replacing a missing task fails during assembly, after every change was prepared.
        tx.set_task(
            {
                "id": id_from_number(999),
                "conversationId": conversation_id,
                "kind": "missing",
                "version": 1,
                "input": None,
                "background": False,
                "abortRequested": False,
                "state": {"status": "pending", "checkpoint": {"phase": "start"}},
            }
        )

    with pytest.raises(RuntimeError, match="Task 999 does not exist"):
        await session.commit_with(failing, context)
    assert len(storage.commits) == commits
    assert await session.snapshot(LiveDoc, conversation_id, context) is live
    assert await session.snapshot(CounterDoc, context) is counter

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "next"
        (await tx.doc(CounterDoc))["count"] = 3

    await session.commit(change, context)
    assert (await session.snapshot(CounterDoc, context))["count"] == 3


async def test_fails_the_session_after_a_storage_failure_and_publishes_nothing(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    await flush()
    before = await session.snapshot(LiveDoc, conversation_id, context)
    published = len(publications)
    storage.fail_next_commit(RuntimeError("disk vanished"))

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "uncertain"

    with pytest.raises(RuntimeError, match="disk vanished"):
        await session.commit(change, context)
    await flush()
    assert len(publications) == published
    assert "message" not in before
    with pytest.raises(SessionFailed):
        await session.snapshot(LiveDoc, conversation_id, context)
    with pytest.raises(SessionFailed):
        await session.commit(lambda tx: None, context)
    await session.close(context)


async def test_keeps_the_previous_revision_unchanged_through_storage_settlement(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)
    before = await session.snapshot(LiveDoc, conversation_id, context)
    snapshot_copy = copy.deepcopy(before)
    gate = storage.hold_commits()

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["items"].append("c")
        live["nested"]["count"] = 9

    commit = session.commit(change, context)
    await gate.entered
    assert await session.snapshot(LiveDoc, conversation_id, context) is before
    assert before == snapshot_copy
    gate.release()
    await commit
    after = await session.snapshot(LiveDoc, conversation_id, context)
    assert after["items"] == ["a", "b", "c"]
    assert before == snapshot_copy


async def test_retires_documents_and_creates_a_new_incarnation_at_the_same_address(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    await flush()
    old_id = document_changes(publications[-1])[0]["record"]["id"]

    async def change(tx: Any) -> None:
        live = await tx.doc(LiveDoc, conversation_id)
        live["message"] = "final"
        await tx.retire_doc(LiveDoc, conversation_id)
        replacement = await tx.doc(LiveDoc, conversation_id)
        assert replacement is not live
        replacement["message"] = "new"

    await session.commit(change, context)
    writes: list[Any] = storage.commits[-1]
    assert len(writes) == 3
    assert any(w["type"] == "document.change" and w["id"] == old_id for w in writes)
    assert {"type": "document.retire", "id": old_id} in writes
    assert any(w["type"] == "document.create" and w["record"]["kind"] == "test.live" for w in writes)
    await flush()
    publication = publications[-1]
    retired, created = document_changes(publication)
    assert retired["record"]["id"] == old_id and retired["value"] is None and retired["ops"] == []
    assert retired["record"]["retiredAt"] == publication["seq"]
    assert created["record"]["id"] != old_id
    assert created["record"]["createdAt"] == publication["seq"]
    assert created["value"]["message"] == "new" and created["value"]["items"] == [] and created["ops"] == []
    assert await session.snapshot(LiveDoc, conversation_id, context) is created["value"]

    await session.commit(lambda tx: tx.retire_doc(LiveDoc, conversation_id), context)
    assert await session.snapshot(LiveDoc, conversation_id, context) is None
    await session.unload_documents()
    assert await session.snapshot(LiveDoc, conversation_id, context) is None
    # Retiring an absent address is a no-op.
    commits = len(storage.commits)
    await session.commit(lambda tx: tx.retire_doc(LiveDoc, conversation_id), context)
    assert len(storage.commits) == commits


async def test_retires_without_acquisition_and_recreates_both_existing_and_absent_addresses(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    await flush()
    old_id = document_changes(publications[-1])[0]["record"]["id"]
    await session.unload_documents()

    async def existing(tx: Any) -> None:
        retired = tx.retire_doc(LiveDoc, conversation_id)
        replacement = tx.doc(LiveDoc, conversation_id)
        live, _ = await asyncio.gather(replacement, retired)
        live["message"] = "replacement"

    await session.commit(existing, context)
    existing_writes: list[Any] = storage.commits[-1]
    assert len(existing_writes) == 2
    assert {"type": "document.retire", "id": old_id} in existing_writes
    assert any(w["type"] == "document.create" and w["record"]["kind"] == "test.live" for w in existing_writes)

    async def absent(tx: Any) -> None:
        retired = tx.retire_doc(MemberDoc, "absent")
        replacement = tx.doc(MemberDoc, "absent", "seed")
        member, _ = await asyncio.gather(replacement, retired)
        member["hits"] = 1

    await session.commit(absent, context)
    absent_writes = storage.commits[-1]
    assert len([w for w in absent_writes if w["type"] == "document.retire"]) == 0
    assert len([w for w in absent_writes if w["type"] == "document.create"]) == 1
    assert await session.snapshot(MemberDoc, "absent", context) == {"seed": "seed", "hits": 1}


async def test_retires_the_existing_incarnation_when_retirement_races_a_pending_acquisition(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications, conversation_id = await setup_live(storage_backend)
    await flush()
    old_id = document_changes(publications[-1])[0]["record"]["id"]

    async def first(tx: Any) -> None:
        acquired = tx.doc(LiveDoc, conversation_id)
        retired = tx.retire_doc(LiveDoc, conversation_id)
        (await acquired)["message"] = "final"
        await retired

    await session.commit(first, context)
    first_writes: list[Any] = storage.commits[-1]
    assert len(first_writes) == 2
    assert any(
        w["type"] == "document.change" and w["id"] == old_id and w["content"]["kind"] == "delta" for w in first_writes
    )
    assert {"type": "document.retire", "id": old_id} in first_writes
    assert await session.snapshot(LiveDoc, conversation_id, context) is None

    async def second(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "second"

    await session.commit(second, context)
    await flush()
    second_id = document_changes(publications[-1])[0]["record"]["id"]

    async def third(tx: Any) -> None:
        acquired = tx.doc(LiveDoc, conversation_id)
        retired = tx.retire_doc(LiveDoc, conversation_id)
        recreated = tx.doc(LiveDoc, conversation_id)
        old, fresh, _ = await asyncio.gather(acquired, recreated, retired)
        assert fresh is not old
        fresh["message"] = "third"

    await session.commit(third, context)
    writes: list[Any] = storage.commits[-1]
    assert len(writes) == 2
    assert {"type": "document.retire", "id": second_id} in writes
    assert any(w["type"] == "document.create" and w["record"]["kind"] == "test.live" for w in writes)
    assert (await session.snapshot(LiveDoc, conversation_id, context))["message"] == "third"


async def test_reloads_an_unloaded_document_from_storage(storage_backend: StorageBackend) -> None:
    _, session, _, conversation_id = await setup_live(storage_backend)
    loaded = await session.snapshot(LiveDoc, conversation_id, context)
    await session.unload_documents()
    reloaded = await session.snapshot(LiveDoc, conversation_id, context)
    assert reloaded is not loaded
    assert reloaded == loaded

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["items"].append("c")

    await session.commit(change, context)
    await session.unload_documents()
    assert (await session.snapshot(LiveDoc, conversation_id, context))["items"] == ["a", "b", "c"]


async def test_delivers_complete_publications_synchronously_after_adoption(storage_backend: StorageBackend) -> None:
    _, session, publications, conversation_id = await setup_live(storage_backend)
    published = len(publications)
    contexts: list[Any] = []
    unsubscribe = session.subscribe_commits(lambda _publication, delivered: contexts.append(delivered))

    async def change(tx: Any) -> str:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "m"
        return "done"

    result = await session.commit(change, context)
    assert result == "done"
    assert contexts == [context]
    assert contexts[0] is context
    assert len(publications) == published + 1
    unsubscribe()


async def test_publishes_close_synchronously_and_supports_unsubscription(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    calls: list[str] = []
    session.subscribe_close(lambda: calls.append("active"))
    unsubscribe = session.subscribe_close(lambda: calls.append("removed"))
    unsubscribe()
    await session.close(context)
    assert calls == ["active"]


async def test_settles_admitted_commits_before_close_and_rejects_later_admission(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _, conversation_id = await setup_live(storage_backend)
    gate = storage.hold_commits()

    async def admitted(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "admitted"

    async def queued_change(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["message"] = "queued"

    commit = session.commit(admitted, context)
    await gate.entered
    queued = session.commit(queued_change, context)
    admitted_snapshot = session.snapshot(MemberDoc, "absent", context)
    closed = session.close(context)
    with pytest.raises(RuntimeError, match="closed"):
        await session.commit(lambda tx: None, context)
    with pytest.raises(RuntimeError, match="closed"):
        await session.snapshot(LiveDoc, conversation_id, context)
    gate.release()
    await commit
    await queued
    assert await admitted_snapshot is None
    await closed
    with pytest.raises(RuntimeError):
        await storage.find_document(
            {"kind": "test.live", "scope": {"kind": "conversation", "conversationId": conversation_id}},
            "current",
            context,
        )
