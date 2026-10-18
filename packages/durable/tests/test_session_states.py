"""Port of ``test/session-states.test.ts``."""

from __future__ import annotations

from typing import Any

from durable_session_support import (
    StorageBackend,
    context,
    create_conversation,
    document_changes,
    flush,
    open_test_session,
)

from pi_durable import define_doc, define_doc_family

StateDoc = define_doc(
    kind="state.state", version=1, scope="session", initial=lambda: {"value": 0, "retained": {"label": "stable"}}
)
FamilyDoc = define_doc_family(
    kind="state.family", version=1, scope="session", initial=lambda seed: {"value": len(seed)}
)


async def create_state(backend: StorageBackend) -> Any:
    opened = await open_test_session(backend)
    await opened.session.commit(_acquire, context)
    await flush()
    return opened


async def _acquire(tx: Any) -> None:
    await tx.doc(StateDoc)


def set_value(value: int) -> Any:
    async def change(tx: Any) -> None:
        (await tx.doc(StateDoc))["value"] = value

    return change


async def test_never_creates_an_absent_document(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    commits = len(storage.commits)
    assert await session.document_state(StateDoc, context) is None
    assert await session.document_state(FamilyDoc, "missing", context) is None
    assert len(storage.commits) == commits
    assert storage.mint_count == 0


async def test_returns_an_immediately_hydrated_read_only_state_with_contiguous_chord_deliveries(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    baseline = await session.snapshot(StateDoc, context)
    state = await session.document_state(StateDoc, context)
    deliveries: list[tuple[Any, int]] = []
    state.subscribe(lambda value, _context, delivery: deliveries.append((value, delivery.sequence)))
    assert state.value is baseline

    await session.commit(set_value(1), context)
    await session.commit(set_value(2), context)
    await flush()

    assert [sequence for _, sequence in deliveries] == [0, 1, 2]
    assert [value["value"] for value, _ in deliveries] == [0, 1, 2]
    assert state.value is await session.snapshot(StateDoc, context)
    state.dispose()


async def test_creates_independent_disposable_states_for_one_incarnation(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    first = await session.document_state(StateDoc, context)
    second = await session.document_state(StateDoc, context)
    assert second is not first
    await session.commit(set_value(1), context)
    await flush()
    assert first.value["value"] == 1
    assert second.value["value"] == 1

    first.dispose()
    await session.commit(set_value(2), context)
    await flush()
    assert first.value["value"] == 1
    assert second.value["value"] == 2
    second.dispose()


async def test_shares_exact_committed_value_references_with_chord(storage_backend: StorageBackend) -> None:
    _, session, publications = await create_state(storage_backend)
    state = await session.document_state(StateDoc, context)
    await session.commit(set_value(4), context)
    await flush()
    published = document_changes(publications[-1])[0]
    assert state.value is published["value"]
    assert state.value["retained"] is (await session.snapshot(StateDoc, context))["retained"]
    state.dispose()


async def test_captures_a_late_baseline_without_redelivering_an_already_covered_commit(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    await session.commit(set_value(1), context)
    state = await session.document_state(StateDoc, context)
    deliveries: list[int] = []
    state.subscribe(lambda _value, _context, delivery: deliveries.append(delivery.sequence))
    await flush()
    assert state.value["value"] == 1
    assert deliveries == [0]
    state.dispose()


async def test_publishes_none_retirement_and_never_follows_a_replacement_incarnation(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    old_state = await session.document_state(StateDoc, context)

    async def retire_and_recreate(tx: Any) -> None:
        await tx.retire_doc(StateDoc)
        (await tx.doc(StateDoc))["value"] = 10

    await session.commit(retire_and_recreate, context)
    await flush()
    assert old_state.value is None

    replacement = await session.document_state(StateDoc, context)
    assert replacement.value["value"] == 10
    await session.commit(set_value(11), context)
    await flush()
    assert old_state.value is None
    assert replacement.value["value"] == 11
    old_state.dispose()
    replacement.dispose()


async def test_cold_loads_a_definition_free_fork_copy(storage_backend: StorageBackend) -> None:
    Copied = define_doc(
        kind="state.copied",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda: {"value": 0},
    )
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)

    async def point(tx: Any) -> Any:
        created = await tx.append_entry(parent_id, {"kind": "point"})
        (await tx.doc(Copied, parent_id))["value"] = 7
        return created

    entry = await session.commit(point, context)

    async def fork(tx: Any) -> Any:
        return (await tx.fork_conversation(parent_id, entry["id"], {"ownership": {"kind": "ownerless"}}))["id"]

    child_id = await session.commit(fork, context)
    reads = storage.document_read_count
    state = await session.document_state(Copied, child_id, context)
    assert state.value == {"value": 7}
    assert storage.document_read_count > reads
    state.dispose()


async def test_hydrates_a_migrated_tracker_without_writing_and_skips_an_equal_version_base_update(
    storage_backend: StorageBackend,
) -> None:
    Old = define_doc(kind="state.migration", version=1, scope="session", initial=lambda: {"value": 3})
    Current = define_doc(
        kind="state.migration",
        version=2,
        scope="session",
        initial=lambda: {"value": 0, "migrated": False},
        migrate=lambda value, _version: {"value": value["value"], "migrated": True},
    )
    storage, session, _ = await open_test_session(storage_backend)

    def acquire(token: Any) -> Any:
        async def change(tx: Any) -> None:
            await tx.doc(token)

        return change

    await session.commit(acquire(Old), context)
    await session.unload_documents()
    commits = len(storage.commits)
    state = await session.document_state(Current, context)
    assert state.value == {"value": 3, "migrated": True}
    assert len(storage.commits) == commits
    baseline = state.value

    await session.commit(acquire(Current), context)
    await flush()
    assert len(storage.commits) == commits + 1
    assert state.value is baseline

    async def change(tx: Any) -> None:
        (await tx.doc(Current))["value"] = 4

    await session.commit(change, context)
    await flush()
    assert state.value == {"value": 4, "migrated": True}
    state.dispose()


async def test_continues_from_exact_committed_values_after_the_tracker_cache_unloads(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await create_state(storage_backend)
    state = await session.document_state(StateDoc, context)
    baseline = state.value
    reads = storage.document_read_count
    await session.unload_documents()
    reloaded = await session.snapshot(StateDoc, context)
    assert reloaded == baseline
    assert reloaded is not baseline
    assert storage.document_read_count > reads
    await session.commit(set_value(6), context)
    await flush()
    assert state.value["value"] == 6
    state.dispose()


async def test_exposes_trusted_shared_immutable_values_without_freezing(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    snapshot = await session.snapshot(StateDoc, context)
    state = await session.document_state(StateDoc, context)
    assert state.value is snapshot
    # Plain dicts, not frozen or proxied copies (upstream: `Object.isFrozen()` is false).
    assert type(state.value) is dict
    assert type(state.value["retained"]) is dict
    state.dispose()
