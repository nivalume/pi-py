"""Port of ``test/session-checkpoints-migrations.test.ts``.

Dropped: "calls checkpointWhen with its definition as the receiver" (JavaScript ``this`` binding; a
Python ``checkpoint_when`` is a plain callable).
"""

from __future__ import annotations

import asyncio
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

from pi_durable import define_doc, define_doc_family


def document_writes(writes: Any) -> list[Any]:
    return [w for w in writes if w["type"] in ("document.create", "document.change", "document.retire")]


def acquire(token: Any, *args: Any) -> Any:
    async def change(tx: Any) -> None:
        await tx.doc(token, *args)

    return change


def session_doc(kind: str, initial: Any, **options: Any) -> Any:
    return define_doc(kind=kind, scope="session", initial=initial, **{"version": 1, **options})


# ─── Session document checkpoints ────────────────────────────────────────────


async def test_selects_bases_only_for_nonempty_ordinary_batches_and_passes_the_exact_prepared_revision_and_ops(
    storage_backend: StorageBackend,
) -> None:
    calls: list[tuple[Any, Any]] = []
    false_calls: list[tuple[Any, Any]] = []

    def base_when(value: Any, ops: Any, _info: Any) -> bool:
        calls.append((value, ops))
        return True

    def delta_when(value: Any, ops: Any, _info: Any) -> bool:
        false_calls.append((value, ops))
        return False

    BaseDoc = session_doc("checkpoint.base", lambda: {"items": []}, checkpoint_when=base_when)
    DeltaDoc = session_doc("checkpoint.delta", lambda: {"count": 0}, checkpoint_when=delta_when)
    DefaultDoc = session_doc("checkpoint.default", lambda: {"count": 0})
    storage, session, publications = await open_test_session(storage_backend)

    async def create(tx: Any) -> None:
        await tx.doc(BaseDoc)
        await tx.doc(DeltaDoc)
        await tx.doc(DefaultDoc)

    await session.commit(create, context)
    assert calls == [] and false_calls == []
    assert all(
        w["type"] == "document.create" and w["content"]["kind"] == "base" for w in document_writes(storage.commits[-1])
    )

    async def change(tx: Any) -> None:
        (await tx.doc(BaseDoc))["items"].append("x")
        (await tx.doc(DeltaDoc))["count"] += 1
        (await tx.doc(DefaultDoc))["count"] += 1

    await session.commit(change, context)
    await flush()
    writes: list[Any] = [w for w in storage.admitted_commits[-1] if w["type"] == "document.change"]
    assert [w["content"]["kind"] for w in writes] == ["base", "delta", "delta"]
    assert len(calls) == 1
    assert len(false_calls) == 1
    snapshot = await session.snapshot(BaseDoc, context)
    delta_snapshot = await session.snapshot(DeltaDoc, context)
    published, published_delta, *_ = document_changes(publications[-1])
    assert calls[0][0] is snapshot
    assert calls[0][1] is published["ops"]
    assert false_calls[0][0] is delta_snapshot
    assert false_calls[0][1] is published_delta["ops"]
    assert writes[1]["content"]["kind"] == "delta"
    assert writes[1]["content"]["ops"] is false_calls[0][1]
    assert writes[0]["content"]["kind"] == "base"
    assert writes[0]["content"]["value"] is snapshot


async def test_passes_the_stored_delta_count_since_the_newest_base_including_after_unload_and_version_bases(
    storage_backend: StorageBackend,
) -> None:
    seen: list[int] = []

    def v1_when(_value: Any, _ops: Any, info: Any) -> bool:
        seen.append(info["deltasSinceBase"])
        return info["deltasSinceBase"] >= 2

    def v2_when(_value: Any, _ops: Any, info: Any) -> bool:
        seen.append(info["deltasSinceBase"])
        return False

    V1 = session_doc("checkpoint.deltas-since-base", lambda: {"count": 0}, checkpoint_when=v1_when)
    V2 = session_doc(
        "checkpoint.deltas-since-base",
        lambda: {"count": 0},
        version=2,
        migrate=lambda value, _version: {"count": value["count"]},
        checkpoint_when=v2_when,
    )
    storage, session, _ = await open_test_session(storage_backend)

    async def increment(token: Any) -> None:
        async def change(tx: Any) -> None:
            (await tx.doc(token))["count"] += 1

        await session.commit(change, context)

    await session.commit(acquire(V1), context)
    await increment(V1)
    await increment(V1)
    await increment(V1)
    await session.unload_documents()
    await increment(V1)
    assert seen == [0, 1, 2, 0]
    kinds = [
        writes[0]["content"]["kind"] if writes[0]["type"] == "document.change" else writes[0]["type"]
        for writes in storage.commits[-4:]
    ]
    assert kinds == ["delta", "delta", "base", "delta"]

    # A required version base resets the count without calling the predicate.
    await increment(V2)
    await increment(V2)
    assert seen == [0, 1, 2, 0, 0]
    content: Any = storage.commits[-2][0]["content"]
    assert content["kind"] == "base" and content["version"] == 2


async def test_skips_the_predicate_for_empty_batches_but_calls_it_for_nonempty_structural_no_ops(
    storage_backend: StorageBackend,
) -> None:
    calls: list[int] = []

    def when(*_args: Any) -> bool:
        calls.append(1)
        return False

    Doc = session_doc("checkpoint.no-op", lambda: {"items": ["a", "b"]}, checkpoint_when=when)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Doc), context)
    commits = len(storage.commits)

    async def empty(tx: Any) -> None:
        value = await tx.doc(Doc)
        value["items"].append("x")
        value["items"].pop()

    await session.commit(empty, context)
    assert calls == []
    assert len(storage.commits) == commits

    async def rotate(tx: Any) -> None:
        value = await tx.doc(Doc)
        first = value["items"].pop(0)
        value["items"].insert(0, first)

    await session.commit(rotate, context)
    assert len(calls) == 1
    write: Any = storage.commits[-1][0]
    assert write["type"] == "document.change" and write["content"]["kind"] == "delta"


async def test_rolls_back_every_prepared_document_when_a_checkpoint_predicate_throws(
    storage_backend: StorageBackend,
) -> None:
    throw_checkpoint = [True]
    first_calls: list[int] = []

    def first_when(*_args: Any) -> bool:
        first_calls.append(1)
        return False

    def second_when(*_args: Any) -> bool:
        if throw_checkpoint[0]:
            raise RuntimeError("checkpoint failed")
        return False

    First = session_doc("checkpoint.rollback.first", lambda: {"count": 0}, checkpoint_when=first_when)
    Second = session_doc("checkpoint.rollback.second", lambda: {"count": 0}, checkpoint_when=second_when)
    storage, session, publications = await open_test_session(storage_backend)

    async def create(tx: Any) -> None:
        await tx.doc(First)
        await tx.doc(Second)

    await session.commit(create, context)
    await flush()
    first = await session.snapshot(First, context)
    second = await session.snapshot(Second, context)
    commits = len(storage.commits)
    published = len(publications)

    def counts(a: int, b: int) -> Any:
        async def change(tx: Any) -> None:
            (await tx.doc(First))["count"] = a
            (await tx.doc(Second))["count"] = b

        return change

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await session.commit(counts(1, 2), context)
    await flush()
    assert len(first_calls) == 1
    assert len(storage.commits) == commits
    assert len(publications) == published
    assert await session.snapshot(First, context) is first
    assert await session.snapshot(Second, context) is second

    throw_checkpoint[0] = False
    await session.commit(counts(3, 4), context)
    assert await session.snapshot(First, context) == {"count": 3}
    assert await session.snapshot(Second, context) == {"count": 4}


async def test_persists_repeated_false_decisions_as_deltas_and_replays_the_complete_tail(
    storage_backend: StorageBackend,
) -> None:
    calls: list[int] = []

    def when(*_args: Any) -> bool:
        calls.append(1)
        return False

    Doc = session_doc("checkpoint.tail", lambda: {"values": []}, checkpoint_when=when)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Doc), context)
    for value in range(1, 9):

        async def push(tx: Any, value: int = value) -> None:
            (await tx.doc(Doc))["values"].append(value)

        await session.commit(push, context)
    changes: list[Any] = [w for writes in storage.commits for w in writes if w["type"] == "document.change"]
    assert len(changes) == 8
    assert all(w["content"]["kind"] == "delta" for w in changes)
    assert len(calls) == 8
    await session.unload_documents()
    assert await session.snapshot(Doc, context) == {"values": [1, 2, 3, 4, 5, 6, 7, 8]}


async def test_keeps_a_prepared_root_replacement_as_a_delta_when_the_predicate_is_false(
    storage_backend: StorageBackend,
) -> None:
    initial = {f"field{index}": 0 for index in range(4_100)}
    Doc = session_doc("checkpoint.root-replacement", lambda: initial, checkpoint_when=lambda *_args: False)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Doc), context)

    async def change(tx: Any) -> None:
        value = await tx.doc(Doc)
        for index in range(4_100):
            value[f"field{index}"] = 1

    await session.commit(change, context)
    write: Any = storage.commits[-1][0]
    assert write["type"] == "document.change"
    assert write["content"]["kind"] == "delta"
    assert len(write["content"]["ops"]) == 1
    assert write["content"]["ops"][0][0] == "r"
    await session.unload_documents()
    assert (await session.snapshot(Doc, context))["field4099"] == 1


async def test_uses_ordinary_checkpoint_selection_before_retirement(storage_backend: StorageBackend) -> None:
    calls: list[int] = []

    def when(*_args: Any) -> bool:
        calls.append(1)
        return True

    Doc = session_doc("checkpoint.retire", lambda: {"count": 0}, checkpoint_when=when)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Doc), context)

    async def change(tx: Any) -> None:
        (await tx.doc(Doc))["count"] = 1
        await tx.retire_doc(Doc)

    await session.commit(change, context)
    assert len(calls) == 1
    writes = document_writes(storage.commits[-1])
    assert len(writes) == 2
    assert writes[0]["type"] == "document.change" and writes[0]["content"]["kind"] == "base"
    assert writes[1]["type"] == "document.retire"


# ─── Session document migrations ─────────────────────────────────────────────


async def test_migrates_read_only_once_per_cold_load_copies_the_callback_result_and_writes_nothing(
    storage_backend: StorageBackend,
) -> None:
    Old = session_doc("migration.read-only", lambda: {"count": 2})
    calls: list[int] = []
    retained: list[Any] = []

    def migrate(value: Any, from_version: int) -> Any:
        assert from_version == 1
        calls.append(1)
        result = {"count": value["count"], "labels": ["migrated"]}
        retained.append(result)
        return result

    Current = session_doc("migration.read-only", lambda: {"count": 0, "labels": []}, version=3, migrate=migrate)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Old), context)
    await session.unload_documents()
    commits = len(storage.commits)

    first = await session.snapshot(Current, context)
    assert await session.snapshot(Current, context) is first
    assert len(calls) == 1
    assert len(storage.commits) == commits
    retained[0]["count"] = 99
    retained[0]["labels"].append("mutated")
    assert first == {"count": 2, "labels": ["migrated"]}

    await session.unload_documents()
    second = await session.snapshot(Current, context)
    assert second is not first
    assert second == first
    assert len(calls) == 2
    assert len(storage.commits) == commits


async def test_writes_the_required_base_on_the_first_successful_transaction_then_writes_deltas(
    storage_backend: StorageBackend,
) -> None:
    Old = session_doc("migration.transition", lambda: {"count": 4})
    checkpoints: list[int] = []

    def when(*_args: Any) -> bool:
        checkpoints.append(1)
        return False

    Current = session_doc(
        "migration.transition",
        lambda: {"count": 0},
        version=3,
        migrate=lambda value, from_version: {"count": value["count"] + from_version - 1},
        checkpoint_when=when,
    )
    storage, session, publications = await open_test_session(storage_backend)
    await session.commit(acquire(Old), context)
    await session.unload_documents()
    # An observer of the older shape.
    watch = await session.watch_doc(Old, context)
    frames: list[Any] = []

    async def old_listener(value: Any, ops: Any, _context: Any) -> None:
        frames.append({"value": value, "ops": ops})

    watch.start(old_listener)
    assert await session.snapshot(Current, context) == {"count": 4}
    snapshot = await session.snapshot(Current, context)
    # An observer of the new shape: the migration changes nothing it sees.
    current = await session.watch_doc(Current, context)
    current_frames: list[Any] = []

    async def current_listener(value: Any, _ops: Any, _context: Any) -> None:
        current_frames.append(value)

    current.start(current_listener)

    await session.commit(acquire(Current), context)
    await flush()
    write: Any = storage.commits[-1][0]
    assert write["type"] == "document.change"
    assert write["content"] == {"kind": "base", "version": 3, "value": {"count": 4}}
    assert checkpoints == []
    # The migration-only base is published, so the older-shape watch receives the new value as a root replacement.
    changes = document_changes(publications[-1])
    assert len(changes) == 1
    assert changes[0]["version"] == 3 and changes[0]["value"] == {"count": 4} and changes[0]["ops"] == []
    assert frames == [{"value": {"count": 4}, "ops": [("r", {"count": 4})]}]
    assert current_frames == []
    await watch.stop()
    await current.stop()
    assert await session.snapshot(Current, context) is snapshot

    async def edit(tx: Any) -> None:
        (await tx.doc(Current))["count"] = 7

    await session.commit(edit, context)
    write = storage.commits[-1][0]
    assert write["type"] == "document.change"
    assert write["content"]["kind"] == "delta" and write["content"]["version"] == 3
    assert len(checkpoints) == 1
    await session.unload_documents()
    assert await session.snapshot(Current, context) == {"count": 7}


async def test_rolls_migration_and_edits_back_with_the_callback_then_coalesces_later_edits_into_one_base(
    storage_backend: StorageBackend,
) -> None:
    Old = session_doc("migration.rollback", lambda: {"count": 1})
    migrations: list[int] = []

    def migrate(value: Any, _version: int) -> Any:
        migrations.append(1)
        return {"count": value["count"], "migrated": True}

    Current = session_doc("migration.rollback", lambda: {"count": 0, "migrated": False}, version=2, migrate=migrate)
    storage, session, publications = await open_test_session(storage_backend)
    await session.commit(acquire(Old), context)
    await session.unload_documents()
    commits = len(storage.commits)

    async def failing(tx: Any) -> None:
        (await tx.doc(Current))["count"] = 8
        raise RuntimeError("rollback")

    with pytest.raises(RuntimeError, match="rollback"):
        await session.commit(failing, context)
    assert len(storage.commits) == commits
    assert await session.snapshot(Current, context) == {"count": 1, "migrated": True}
    assert len(migrations) == 1

    async def edit(tx: Any) -> None:
        value = await tx.doc(Current)
        value["count"] = 9
        value["migrated"] = False

    await session.commit(edit, context)
    writes = document_writes(storage.commits[-1])
    assert len(writes) == 1
    assert writes[0]["type"] == "document.change"
    assert writes[0]["content"] == {"kind": "base", "version": 2, "value": {"count": 9, "migrated": False}}
    await flush()
    published = document_changes(publications[-1])[0]
    admitted: Any = storage.admitted_commits[-1][0]
    assert admitted["type"] == "document.change" and admitted["content"]["kind"] == "base"
    assert published["value"] is await session.snapshot(Current, context)
    assert published["value"] is admitted["content"]["value"]
    assert len(published["ops"]) > 0
    assert len(migrations) == 1


async def test_strict_checks_migration_results_before_tracker_ownership_and_remains_usable_after_rejection(
    storage_backend: StorageBackend,
) -> None:
    Old = session_doc("migration.invalid", lambda: {"count": 1})
    Invalid = session_doc(
        "migration.invalid", lambda: {}, version=2, migrate=lambda _value, _version: {"invalid": datetime.now()}
    )
    Other = session_doc("migration.invalid.other", lambda: {"ok": True})
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Old), context)
    await session.unload_documents()
    commits = len(storage.commits)
    with pytest.raises(TypeError, match="strict JSON"):
        await session.snapshot(Invalid, context)
    with pytest.raises(TypeError, match="strict JSON"):
        await session.commit(acquire(Invalid), context)
    assert len(storage.commits) == commits
    await session.commit(acquire(Other), context)
    assert await session.snapshot(Other, context) == {"ok": True}


async def test_rejects_newer_stored_versions_and_older_versions_without_migration_for_snapshots_and_transactions(
    storage_backend: StorageBackend,
) -> None:
    V2 = session_doc("migration.compatibility", lambda: {"count": 2}, version=2)
    V1 = session_doc("migration.compatibility", lambda: {"count": 1})
    V3WithoutMigration = session_doc("migration.compatibility", lambda: {"count": 3}, version=3)
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(V2), context)
    await session.unload_documents()
    commits = len(storage.commits)

    with pytest.raises(RuntimeError, match="newer version 2 than 1"):
        await session.snapshot(V1, context)
    with pytest.raises(RuntimeError, match="newer version 2 than 1"):
        await session.commit(acquire(V1), context)
    with pytest.raises(RuntimeError, match="requires migration from version 2"):
        await session.snapshot(V3WithoutMigration, context)
    with pytest.raises(RuntimeError, match="requires migration from version 2"):
        await session.commit(acquire(V3WithoutMigration), context)
    assert len(storage.commits) == commits


async def test_persists_a_required_migration_base_before_retirement_without_consulting_the_checkpoint_predicate(
    storage_backend: StorageBackend,
) -> None:
    Old = session_doc("migration.retire", lambda: {"count": 1})

    def must_not_run(*_args: Any) -> bool:
        raise RuntimeError("must not run")

    Current = session_doc(
        "migration.retire",
        lambda: {"count": 0},
        version=2,
        migrate=lambda value, _version: {"count": value["count"]},
        checkpoint_when=must_not_run,
    )
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(acquire(Old), context)
    await session.unload_documents()

    async def change(tx: Any) -> None:
        await tx.doc(Current)
        await tx.retire_doc(Current)

    await session.commit(change, context)
    writes = document_writes(storage.commits[-1])
    assert len(writes) == 2
    assert writes[0]["type"] == "document.change"
    assert writes[0]["content"] == {"kind": "base", "version": 2, "value": {"count": 1}}
    assert writes[1]["type"] == "document.retire"


async def test_leaves_unaccessed_older_documents_and_unavailable_definitions_untouched(
    storage_backend: StorageBackend,
) -> None:
    FirstV1 = session_doc("migration.lazy.first", lambda: {"count": 1})
    SecondV1 = session_doc("migration.lazy.second", lambda: {"count": 2})
    second_migrations: list[int] = []
    FirstV2 = session_doc(
        "migration.lazy.first",
        lambda: {"count": 0},
        version=2,
        migrate=lambda value, _version: {"count": value["count"]},
    )

    def second_migrate(value: Any, _version: int) -> Any:
        second_migrations.append(1)
        return {"count": value["count"]}

    session_doc("migration.lazy.second", lambda: {"count": 0}, version=2, migrate=second_migrate)
    storage, session, _ = await open_test_session(storage_backend)

    async def create(tx: Any) -> None:
        await tx.doc(FirstV1)
        await tx.doc(SecondV1)

    await session.commit(create, context)
    await session.unload_documents()
    commits = len(storage.commits)
    assert await session.snapshot(FirstV2, context) == {"count": 1}
    assert len(storage.commits) == commits
    assert second_migrations == []
    second_record = await storage.find_document(
        {"kind": "migration.lazy.second", "scope": {"kind": "session"}}, "current", context
    )
    assert (await storage.document(second_record["id"], "current", context))["version"] == 1


# ─── Session historical document snapshots ───────────────────────────────────


async def test_migrates_current_and_historical_rewindable_values_independently_and_follows_fork_ancestry(
    storage_backend: StorageBackend,
) -> None:
    V1 = define_doc(
        kind="history.migration",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"count": 0},
    )
    Family = define_doc_family(
        kind="history.family",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda seed: {"seed": seed, "count": 0},
    )
    migrations: list[int] = []

    def migrate(value: Any, from_version: int) -> Any:
        migrations.append(from_version)
        return {"count": value["count"], "version": 3}

    V3 = define_doc(
        kind="history.migration",
        version=3,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"count": 0, "version": 3},
        migrate=migrate,
    )
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def first(tx: Any) -> None:
        ids["first"] = (await tx.append_entry(conversation_id, {"kind": "first"}))["id"]
        (await tx.doc(V1, conversation_id))["count"] = 1
        (await tx.doc(Family, conversation_id, "member", "seed"))["count"] = 1

    async def second(tx: Any) -> None:
        ids["second"] = (await tx.append_entry(conversation_id, {"kind": "second"}))["id"]
        (await tx.doc(V1, conversation_id))["count"] = 2

    await session.commit(first, context)
    await session.commit(second, context)
    await session.unload_documents()
    commits = len(storage.commits)
    assert await session.snapshot(V3, conversation_id, context) == {"count": 2, "version": 3}
    assert migrations == [1]
    assert len(storage.commits) == commits

    async def third(tx: Any) -> None:
        ids["third"] = (await tx.append_entry(conversation_id, {"kind": "third"}))["id"]
        await tx.doc(V3, conversation_id)

    await session.commit(third, context)
    assert any(
        w["type"] == "document.change" and w["content"]["kind"] == "base" and w["content"]["version"] == 3
        for w in storage.commits[-1]
    )

    assert await session.snapshot_as_of(V3, conversation_id, ids["first"], context) == {"count": 1, "version": 3}
    assert await session.snapshot_as_of(V3, conversation_id, ids["second"], context) == {"count": 2, "version": 3}
    assert await session.snapshot_as_of(V3, conversation_id, ids["third"], context) == {"count": 2, "version": 3}
    assert migrations == [1, 1, 1]
    assert await session.snapshot_as_of(Family, conversation_id, "member", ids["first"], context) == {
        "seed": "seed",
        "count": 1,
    }

    async def fork(tx: Any) -> Any:
        return (await tx.fork_conversation(conversation_id, ids["second"], {"ownership": {"kind": "ownerless"}}))["id"]

    child_id = await session.commit(fork, context)
    assert await session.snapshot_as_of(V3, child_id, ids["first"], context) == {"count": 1, "version": 3}
    assert await session.snapshot_as_of(V3, child_id, ids["second"], context) == {"count": 2, "version": 3}
    assert await session.snapshot_as_of(Family, child_id, "member", ids["first"], context) == {
        "seed": "seed",
        "count": 1,
    }
    with pytest.raises(RuntimeError, match=f"Entry {ids['third']} is not visible"):
        await session.snapshot_as_of(V3, child_id, ids["third"], context)


async def test_selects_the_incarnation_alive_at_the_entry_commit_across_retirement_and_recreation(
    storage_backend: StorageBackend,
) -> None:
    Doc = define_doc(
        kind="history.incarnation",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"value": "initial"},
    )
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    def entry(kind: str, then: Any = None) -> Any:
        async def change(tx: Any) -> Any:
            created = (await tx.append_entry(conversation_id, {"kind": kind}))["id"]
            if then is not None:
                await then(tx)
            return created

        return change

    async def create_old(tx: Any) -> None:
        (await tx.doc(Doc, conversation_id))["value"] = "old"

    async def retire(tx: Any) -> None:
        await tx.retire_doc(Doc, conversation_id)

    async def create_new(tx: Any) -> None:
        (await tx.doc(Doc, conversation_id))["value"] = "new"

    before_creation = await session.commit(entry("before"), context)
    created_at = await session.commit(entry("create", create_old), context)
    retired_at = await session.commit(entry("retire", retire), context)
    recreated_at = await session.commit(entry("recreate", create_new), context)

    assert await session.snapshot_as_of(Doc, conversation_id, before_creation, context) is None
    assert await session.snapshot_as_of(Doc, conversation_id, created_at, context) == {"value": "old"}
    assert await session.snapshot_as_of(Doc, conversation_id, retired_at, context) is None
    assert await session.snapshot_as_of(Doc, conversation_id, recreated_at, context) == {"value": "new"}
    await session.close(context)
    with pytest.raises(RuntimeError, match="closed"):
        await session.snapshot_as_of(Doc, conversation_id, recreated_at, context)


# ─── Session tracker cache across definition versions ────────────────────────

V1Doc = session_doc("cache.versioned", lambda: {"name": "first"})
V2Doc = session_doc(
    "cache.versioned", lambda: {"names": []}, version=2, migrate=lambda value, _version: {"names": [value["name"]]}
)


async def test_migrates_a_document_cached_by_an_older_token_without_unloading(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    await session.commit(lambda tx: tx.doc(V1Doc), context)
    assert await session.snapshot(V1Doc, context) == {"name": "first"}

    # Reloaded extension code accesses the still-cached document with a newer token.
    assert await session.snapshot(V2Doc, context) == {"names": ["first"]}

    async def change(tx: Any) -> None:
        (await tx.doc(V2Doc))["names"].append("second")

    await session.commit(change, context)
    write = document_writes(storage.commits[-1])[0]
    assert write["type"] == "document.change"
    assert write["content"] == {"version": 2, "kind": "base", "value": {"names": ["first", "second"]}}
    with pytest.raises(RuntimeError, match="newer version 2"):
        await session.snapshot(V1Doc, context)


async def test_sends_observers_of_an_older_shape_a_root_replacement_after_a_newer_token_writes(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await open_test_session(storage_backend)
    await session.commit(lambda tx: tx.doc(V1Doc), context)
    state = await session.document_state(V1Doc, context)
    watch = await session.watch_doc(V1Doc, context)
    frames: list[Any] = []
    delivered = asyncio.get_running_loop().create_future()

    async def listener(_value: Any, ops: Any, _context: Any) -> None:
        frames.append(ops)
        if not delivered.done():
            delivered.set_result(None)

    watch.start(listener)

    async def change(tx: Any) -> None:
        (await tx.doc(V2Doc))["names"].append("second")

    await session.commit(change, context)
    await delivered
    assert state.value == {"names": ["first", "second"]}
    assert watch.value == {"names": ["first", "second"]}
    assert frames == [[("r", {"names": ["first", "second"]})]]
    state.dispose()
    await watch.stop()


async def test_serves_an_older_token_from_storage_after_a_newer_token_migrated_only_in_memory(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await open_test_session(storage_backend)
    await session.commit(lambda tx: tx.doc(V1Doc), context)
    await session.unload_documents()
    assert await session.snapshot(V2Doc, context) == {"names": ["first"]}
    assert await session.snapshot(V1Doc, context) == {"name": "first"}

    async def rename(tx: Any) -> None:
        (await tx.doc(V1Doc))["name"] = "renamed"

    await session.commit(rename, context)
    assert await session.snapshot(V2Doc, context) == {"names": ["renamed"]}
