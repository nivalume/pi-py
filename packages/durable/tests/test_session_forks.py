"""Port of ``test/session-forks.test.ts``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from durable_session_support import (
    StorageBackend,
    context,
    create_conversation,
    document_changes,
    document_copy_changes,
    flush,
    open_test_session,
)

from pi_durable import SessionFailed, define_doc, define_doc_family, define_task

OWNERLESS: Any = {"ownership": {"kind": "ownerless"}}


def document_creates(writes: Any) -> list[Any]:
    return [write for write in writes if write["type"] == "document.create"]


def document_copies(writes: Any) -> list[Any]:
    return [write for write in writes if write["type"] == "document.copy"]


def fork(parent_id: Any, at: Any) -> Any:
    return lambda tx: tx.fork_conversation(parent_id, at, OWNERLESS)


def rewindable(kind: str, initial: Any) -> Any:
    return define_doc(kind=kind, version=1, scope="conversation", history="rewindable", fork="asOf", initial=initial)


def latest(kind: str, initial: Any, fork_policy: Any = "current") -> Any:
    return define_doc(kind=kind, version=1, scope="conversation", history="latest", fork=fork_policy, initial=initial)


async def test_copies_as_of_and_current_singleton_and_family_bases_while_leaving_initial_documents_absent(
    storage_backend: StorageBackend,
) -> None:
    AsOf = rewindable("fork.policies.as-of", lambda: {"value": "as-initial"})
    Current = latest("fork.policies.current", lambda: {"value": "current-initial"})
    Initial = latest("fork.policies.initial", lambda: {"value": "fresh"}, "initial")
    AsOfFamily = define_doc_family(
        kind="fork.policies.as-of-family",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda seed: {"value": seed},
    )
    CurrentFamily = define_doc_family(
        kind="fork.policies.current-family",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda seed: {"value": seed},
    )
    storage, session, publications = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def at_fork(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "fork-point"}))["id"])
        (await tx.doc(AsOf, parent_id))["value"] = "as-at-fork"
        (await tx.doc(Current, parent_id))["value"] = "current-at-fork"
        (await tx.doc(Initial, parent_id))["value"] = "parent-only"
        (await tx.doc(AsOfFamily, parent_id, "a", "unused"))["value"] = "family-as-a"
        (await tx.doc(AsOfFamily, parent_id, "b", "unused"))["value"] = "family-as-b"
        (await tx.doc(CurrentFamily, parent_id, "a", "unused"))["value"] = "family-current-a"

    await session.commit(at_fork, context)

    async def after_fork(tx: Any) -> None:
        (await tx.doc(AsOf, parent_id))["value"] = "as-after-fork"
        (await tx.doc(Current, parent_id))["value"] = "current-when-copied"
        (await tx.doc(AsOfFamily, parent_id, "a", "unused"))["value"] = "family-as-after"
        (await tx.doc(CurrentFamily, parent_id, "a", "unused"))["value"] = "family-current-when-copied"

    await session.commit(after_fork, context)
    await flush()
    document_reads = storage.document_read_count

    child = await session.commit(fork(parent_id, fork_at[0]), context)
    await flush()
    assert storage.document_read_count == document_reads

    assert await session.snapshot(AsOf, child["id"], context) == {"value": "as-at-fork"}
    assert await session.snapshot(Current, child["id"], context) == {"value": "current-when-copied"}
    assert await session.snapshot(Initial, child["id"], context) is None
    assert await session.snapshot(AsOfFamily, child["id"], "a", context) == {"value": "family-as-a"}
    assert await session.snapshot(AsOfFamily, child["id"], "b", context) == {"value": "family-as-b"}
    assert await session.snapshot(CurrentFamily, child["id"], "a", context) == {"value": "family-current-when-copied"}

    copies = document_copies(storage.admitted_commits[-1])
    assert len(copies) == 5
    assert all(write["record"]["scope"]["kind"] == "conversation" for write in copies)
    assert all(write["record"]["scope"]["conversationId"] == child["id"] for write in copies)
    publication = publications[-1]
    copied = document_copy_changes(publication)
    assert len(copied) == 5
    for change in copied:
        assert change["record"]["createdAt"] == publication["seq"]
        assert change["conversationId"] == child["id"]
        write = next(candidate for candidate in copies if candidate["record"]["id"] == change["record"]["id"])
        assert change["source"] == write["source"]

    parent_record = await storage.find_document(
        {"kind": AsOf.definition.kind, "scope": {"kind": "conversation", "conversationId": parent_id}},
        "current",
        context,
    )
    child_record = await storage.find_document(
        {"kind": AsOf.definition.kind, "scope": {"kind": "conversation", "conversationId": child["id"]}},
        "current",
        context,
    )
    assert child_record["id"] != parent_record["id"]

    async def independent(tx: Any) -> None:
        (await tx.doc(AsOf, child["id"]))["value"] = "child-independent"

    await session.commit(independent, context)
    assert await session.snapshot(AsOf, parent_id, context) == {"value": "as-after-fork"}
    assert await session.snapshot(AsOf, child["id"], context) == {"value": "child-independent"}

    async def create_initial(tx: Any) -> None:
        (await tx.doc(Initial, child["id"]))["value"] = "child-created"

    await session.commit(create_initial, context)
    assert await session.snapshot(Initial, child["id"], context) == {"value": "child-created"}


async def test_uses_final_document_state_from_the_fork_entry_commit_while_excluding_later_same_commit_entries(
    storage_backend: StorageBackend,
) -> None:
    Doc = rewindable("fork.same-commit", lambda: {"value": "initial"})
    _, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def change(tx: Any) -> None:
        ids["fork_at"] = (await tx.append_entry(parent_id, {"kind": "included"}))["id"]
        (await tx.doc(Doc, parent_id))["value"] = "final-state-of-commit"
        ids["excluded"] = (await tx.append_entry(parent_id, {"kind": "excluded"}))["id"]

    await session.commit(change, context)
    child = await session.commit(fork(parent_id, ids["fork_at"]), context)
    assert await session.snapshot(Doc, child["id"], context) == {"value": "final-state-of-commit"}
    visible = await session.commit(lambda tx: tx.scan_entries({"conversationId": child["id"]}, 10), context)
    visible_ids = [item["id"] for item in visible["items"]]
    assert ids["fork_at"] in visible_ids
    assert ids["excluded"] not in visible_ids


async def test_selects_the_entry_owning_ancestor_for_as_of_copies_and_the_immediate_parent_for_current_copies(
    storage_backend: StorageBackend,
) -> None:
    AsOf = rewindable("fork.ancestry.as-of", lambda: {"value": "initial"})
    Current = latest("fork.ancestry.current", lambda: {"value": "initial"})
    _, session, _ = await open_test_session(storage_backend)
    root_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def at_root(tx: Any) -> None:
        ids["inherited"] = (await tx.append_entry(root_id, {"kind": "root"}))["id"]
        (await tx.doc(AsOf, root_id))["value"] = "root-at-entry"
        (await tx.doc(Current, root_id))["value"] = "root-current"

    await session.commit(at_root, context)
    parent = await session.commit(fork(root_id, ids["inherited"]), context)

    async def at_parent(tx: Any) -> None:
        ids["parent_entry"] = (await tx.append_entry(parent["id"], {"kind": "parent"}))["id"]
        (await tx.doc(AsOf, parent["id"]))["value"] = "parent-at-own-entry"
        (await tx.doc(Current, parent["id"]))["value"] = "parent-current"

    await session.commit(at_parent, context)

    inherited_fork = await session.commit(fork(parent["id"], ids["inherited"]), context)
    assert await session.snapshot(AsOf, inherited_fork["id"], context) == {"value": "root-at-entry"}
    assert await session.snapshot(Current, inherited_fork["id"], context) == {"value": "parent-current"}

    own_entry_fork = await session.commit(fork(parent["id"], ids["parent_entry"]), context)
    assert await session.snapshot(AsOf, own_entry_fork["id"], context) == {"value": "parent-at-own-entry"}
    assert await session.snapshot(Current, own_entry_fork["id"], context) == {"value": "parent-current"}


async def test_copies_the_stored_value_and_version_without_consulting_migration_definitions_or_migrated_caches(
    storage_backend: StorageBackend,
) -> None:
    V1 = rewindable("fork.stored-version", lambda: {"count": 1})
    migrations: list[int] = []

    def migrate(value: Any, from_version: int) -> Any:
        assert from_version == 1
        migrations.append(from_version)
        return {"count": value["count"], "migrated": True}

    V3 = define_doc(
        kind="fork.stored-version",
        version=3,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"count": 0, "migrated": False},
        migrate=migrate,
    )
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        await tx.doc(V1, parent_id)

    await session.commit(change, context)
    await session.unload_documents()
    assert await session.snapshot(V3, parent_id, context) == {"count": 1, "migrated": True}
    assert len(migrations) == 1

    child = await session.commit(fork(parent_id, fork_at[0]), context)
    assert len(migrations) == 1
    child_record = await storage.find_document(
        {"kind": V1.definition.kind, "scope": {"kind": "conversation", "conversationId": child["id"]}},
        "current",
        context,
    )
    child_stored = await storage.document(child_record["id"], "current", context)
    assert child_stored["version"] == 1 and child_stored["value"] == {"count": 1}
    assert await session.snapshot(V3, child["id"], context) == {"count": 1, "migrated": True}
    assert len(migrations) == 2


async def test_coalesces_a_typed_migration_and_override_into_the_copied_creation_base(
    storage_backend: StorageBackend,
) -> None:
    V1 = rewindable("fork.override", lambda: {"count": 2})
    checkpoints: list[int] = []

    def checkpoint_when(*_args: Any) -> bool:
        checkpoints.append(1)
        return False

    V2 = define_doc(
        kind="fork.override",
        version=2,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda: {"count": 0, "migrated": False},
        migrate=lambda value, _version: {"count": value["count"], "migrated": True},
        checkpoint_when=checkpoint_when,
    )
    storage, session, publications = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        await tx.doc(V1, parent_id)

    await session.commit(change, context)

    document_reads = storage.document_read_count

    async def fork_and_override(tx: Any) -> Any:
        created = await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS)
        value = await tx.doc(V2, created["id"])
        value["count"] = 9
        return created

    child = await session.commit(fork_and_override, context)
    await flush()
    assert storage.document_read_count == document_reads + 1
    creates = document_creates(storage.admitted_commits[-1])
    assert len(creates) == 1
    content = creates[0]["content"]
    assert content["kind"] == "base" and content["version"] == 2
    assert content["value"] == {"count": 9, "migrated": True}
    assert checkpoints == []
    changes = document_changes(publications[-1])
    assert len(changes) == 1 and changes[0]["version"] == 2
    assert await session.snapshot(V2, child["id"], context) == {"count": 9, "migrated": True}
    parent_record = await storage.find_document(
        {"kind": V1.definition.kind, "scope": {"kind": "conversation", "conversationId": parent_id}},
        "current",
        context,
    )
    assert (await storage.document(parent_record["id"], "current", context))["version"] == 1


async def test_copies_the_incarnation_alive_at_the_fork_point_across_retirement_and_recreation(
    storage_backend: StorageBackend,
) -> None:
    Doc = rewindable("fork.incarnations", lambda: {"value": "initial"})
    _, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def old(tx: Any) -> None:
        ids["old"] = (await tx.append_entry(parent_id, {"kind": "old"}))["id"]
        (await tx.doc(Doc, parent_id))["value"] = "old"

    async def retired(tx: Any) -> None:
        ids["retired"] = (await tx.append_entry(parent_id, {"kind": "retired"}))["id"]
        await tx.retire_doc(Doc, parent_id)

    async def new(tx: Any) -> None:
        ids["new"] = (await tx.append_entry(parent_id, {"kind": "new"}))["id"]
        (await tx.doc(Doc, parent_id))["value"] = "new"

    await session.commit(old, context)
    await session.commit(retired, context)
    await session.commit(new, context)

    old_child = await session.commit(fork(parent_id, ids["old"]), context)
    empty_child = await session.commit(fork(parent_id, ids["retired"]), context)
    new_child = await session.commit(fork(parent_id, ids["new"]), context)
    assert await session.snapshot(Doc, old_child["id"], context) == {"value": "old"}
    assert await session.snapshot(Doc, empty_child["id"], context) is None
    assert await session.snapshot(Doc, new_child["id"], context) == {"value": "new"}


async def test_rejects_invisible_fork_points_before_admission_and_remains_usable(
    storage_backend: StorageBackend,
) -> None:
    storage, session, publications = await open_test_session(storage_backend)
    root_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def change(tx: Any) -> None:
        ids["visible"] = (await tx.append_entry(root_id, {"kind": "visible"}))["id"]
        ids["hidden"] = (await tx.append_entry(root_id, {"kind": "hidden"}))["id"]

    await session.commit(change, context)
    parent = await session.commit(fork(root_id, ids["visible"]), context)
    await flush()
    commits = len(storage.commits)
    published = len(publications)
    with pytest.raises(RuntimeError, match=f"Entry {ids['hidden']} is not visible"):
        await session.commit(fork(parent["id"], ids["hidden"]), context)
    await flush()
    assert len(storage.commits) == commits
    assert len(publications) == published
    independent = await session.commit(lambda tx: tx.create_conversation(OWNERLESS), context)
    assert await storage.conversation(independent["id"], context) == independent


async def test_rejects_duplicate_as_of_and_current_selections_for_one_child_address_before_admission(
    storage_backend: StorageBackend,
) -> None:
    AsOf = rewindable("fork.duplicate-policy", lambda: {"value": "as-of"})
    Current = latest("fork.duplicate-policy", lambda: {"value": "current"})
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    old_at: list[Any] = []

    async def old(tx: Any) -> None:
        old_at.append((await tx.append_entry(parent_id, {"kind": "old"}))["id"])
        await tx.doc(AsOf, parent_id)

    async def current(tx: Any) -> None:
        await tx.doc(Current, parent_id)

    await session.commit(old, context)
    await session.commit(lambda tx: tx.retire_doc(AsOf, parent_id), context)
    await session.commit(current, context)
    commits = len(storage.commits)
    with pytest.raises(RuntimeError, match="Fork selects multiple source documents"):
        await session.commit(fork(parent_id, old_at[0]), context)
    assert len(storage.commits) == commits
    independent = await session.commit(lambda tx: tx.create_conversation(OWNERLESS), context)
    assert await storage.conversation(independent["id"], context) == independent


async def test_rejects_current_and_as_of_source_writes_in_the_fork_transaction(
    storage_backend: StorageBackend,
) -> None:
    Current = latest("fork.same-transaction-current", lambda: {"value": "committed"})
    AsOf = rewindable("fork.same-transaction-as-of", lambda: {"value": "committed"})
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        await tx.doc(Current, parent_id)
        await tx.doc(AsOf, parent_id)

    await session.commit(change, context)
    commits = len(storage.commits)

    async def before_fork(tx: Any) -> None:
        (await tx.doc(Current, parent_id))["value"] = "before-fork"
        await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS)

    async def after_fork(tx: Any) -> None:
        await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS)
        (await tx.doc(Current, parent_id))["value"] = "after-fork"

    async def as_of_write(tx: Any) -> None:
        (await tx.doc(AsOf, parent_id))["value"] = "as-of-write"
        await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS)

    for callback in (before_fork, after_fork, as_of_write):
        with pytest.raises(RuntimeError, match="Cannot change fork source document"):
            await session.commit(callback, context)
    assert len(storage.commits) == commits
    assert await session.snapshot(Current, parent_id, context) == {"value": "committed"}
    child = await session.commit(fork(parent_id, fork_at[0]), context)
    assert await session.snapshot(Current, child["id"], context) == {"value": "committed"}


async def test_rolls_every_copied_base_back_when_later_pre_admission_assembly_fails(
    storage_backend: StorageBackend,
) -> None:
    Copied = rewindable("fork.rollback.copied", lambda: {"value": "copied"})
    fail_checkpoint = [True]

    def checkpoint_when(*_args: Any) -> bool:
        if fail_checkpoint[0]:
            raise RuntimeError("checkpoint failed")
        return False

    Failure = define_doc(
        kind="fork.rollback.failure",
        version=1,
        scope="session",
        initial=lambda: {"count": 0},
        checkpoint_when=checkpoint_when,
    )
    storage, session, publications = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        await tx.doc(Copied, parent_id)
        await tx.doc(Failure)

    await session.commit(change, context)
    await flush()
    commits = len(storage.commits)
    published = len(publications)
    child_ids: list[Any] = []

    async def failing(tx: Any) -> None:
        child_ids.append((await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS))["id"])
        (await tx.doc(Failure))["count"] = 1

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await session.commit(failing, context)
    await flush()
    assert len(storage.commits) == commits
    assert len(publications) == published
    assert await storage.conversation(child_ids[0], context) is None
    fail_checkpoint[0] = False

    async def count_two(tx: Any) -> None:
        (await tx.doc(Failure))["count"] = 2

    await session.commit(count_two, context)
    assert await session.snapshot(Failure, context) == {"count": 2}
    assert await session.snapshot(Copied, parent_id, context) == {"value": "copied"}


async def test_fails_the_session_on_a_rejected_fork_commit_which_leaves_no_effect(
    storage_backend: StorageBackend,
) -> None:
    Doc = rewindable("fork.storage-rejected", lambda: {"value": "source"})
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        await tx.doc(Doc, parent_id)

    await session.commit(change, context)
    rejected_child: list[Any] = []
    rejection = RuntimeError("copy rejected")
    storage.fail_next_commit(rejection)

    async def rejected_fork(tx: Any) -> None:
        rejected_child.append((await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS))["id"])

    with pytest.raises(RuntimeError) as raised:
        await session.commit(rejected_fork, context)
    assert raised.value is rejection
    with pytest.raises(SessionFailed) as failed:
        await session.commit(lambda tx: tx.create_conversation(OWNERLESS), context)
    assert failed.value.cause is rejection
    # The failed Session closed its Storage; reopened, it holds nothing of the batch.
    assert await session.closed == {"reason": "failed", "error": rejection}
    assert await (await storage.reopen()).conversation(rejected_child[0], context) is None


async def test_retires_a_copied_document_and_can_recreate_the_address_in_the_fork_transaction(
    storage_backend: StorageBackend,
) -> None:
    Doc = rewindable("fork.retire-recreate", lambda: {"value": "fresh"})
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    fork_at: list[Any] = []

    async def change(tx: Any) -> None:
        fork_at.append((await tx.append_entry(parent_id, {"kind": "point"}))["id"])
        (await tx.doc(Doc, parent_id))["value"] = "copied"

    await session.commit(change, context)

    async def fork_retire_recreate(tx: Any) -> Any:
        created = await tx.fork_conversation(parent_id, fork_at[0], OWNERLESS)
        await tx.retire_doc(Doc, created["id"])
        (await tx.doc(Doc, created["id"]))["value"] = "replacement"
        return created

    child = await session.commit(fork_retire_recreate, context)
    writes = storage.commits[-1]
    assert len([w for w in writes if w["type"] == "document.copy"]) == 1
    assert len([w for w in writes if w["type"] == "document.create"]) == 1
    assert len([w for w in writes if w["type"] == "document.retire"]) == 1
    assert await session.snapshot(Doc, child["id"], context) == {"value": "replacement"}


async def test_copies_only_conversation_documents_leaving_session_and_task_documents_in_their_original_scopes(
    storage_backend: StorageBackend,
) -> None:
    Copied = latest("fork.scope.conversation", lambda: {"value": "conversation"})
    SessionOnly = define_doc(
        kind="fork.scope.session", version=1, scope="session", initial=lambda: {"value": "session"}
    )
    TaskOnly = define_doc(kind="fork.scope.task", version=1, scope="task", initial=lambda: {"value": "task"})

    async def noop(*_args: Any) -> None:
        return None

    Work = define_task(
        name="fork.scope.work", version=1, initial=lambda _input: {"phase": "start"}, phases={"start": noop}, abort=noop
    )
    storage, session, publications = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)
    ids: dict[str, Any] = {}

    async def change(tx: Any) -> None:
        ids["fork_at"] = (await tx.append_entry(parent_id, {"kind": "point"}))["id"]
        ids["task"] = await tx.create_task(
            Work, None, {"ownership": {"kind": "conversation"}, "conversationId": parent_id}
        )
        await tx.doc(Copied, parent_id)
        await tx.doc(SessionOnly)
        await tx.doc(TaskOnly, ids["task"])

    await session.commit(change, context)
    child = await session.commit(fork(parent_id, ids["fork_at"]), context)
    await flush()
    copies = document_copies(storage.commits[-1])
    assert [write["record"]["kind"] for write in copies] == [Copied.definition.kind]
    assert [change["record"]["kind"] for change in document_copy_changes(publications[-1])] == [Copied.definition.kind]
    assert await session.snapshot(Copied, child["id"], context) == {"value": "conversation"}
    assert await session.snapshot(SessionOnly, context) == {"value": "session"}
    assert await session.snapshot(TaskOnly, ids["task"], context) == {"value": "task"}


async def test_copies_every_family_member_across_storage_scan_pages(storage_backend: StorageBackend) -> None:
    Family = define_doc_family(
        kind="fork.pagination",
        version=1,
        scope="conversation",
        history="latest",
        fork="current",
        initial=lambda seed: {"value": seed},
    )
    storage, session, _ = await open_test_session(storage_backend)
    parent_id = await create_conversation(session)

    async def point(tx: Any) -> Any:
        return (await tx.append_entry(parent_id, {"kind": "point"}))["id"]

    fork_at = await session.commit(point, context)

    async def members(tx: Any) -> None:
        await asyncio.gather(*(tx.doc(Family, parent_id, f"member-{index}", index) for index in range(260)))

    await session.commit(members, context)
    child = await session.commit(fork(parent_id, fork_at), context)
    assert len(document_copies(storage.commits[-1])) == 260
    assert await session.snapshot(Family, child["id"], "member-0", context) == {"value": 0}
    assert await session.snapshot(Family, child["id"], "member-259", context) == {"value": 259}
