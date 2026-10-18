"""Port of ``test/session-watches.test.ts``."""

from __future__ import annotations

from typing import Any

import pytest
from durable_session_support import (
    StorageBackend,
    context,
    deferred,
    document_changes,
    flush,
    open_test_session,
    resolve,
)

from pi_ai.abort import AbortController
from pi_chord.context import (
    BACKGROUND_CONTEXT,
    create_context_key,
    with_abort_signal,
    with_cancel,
    with_context_value,
)
from pi_chord.delta import apply_immutable
from pi_durable import define_doc

StateDoc = define_doc(
    kind="watch.state",
    version=1,
    scope="session",
    initial=lambda: {"value": 0, "items": ["a", "b"], "retained": {"label": "stable"}},
)


async def _acquire(tx: Any) -> None:
    await tx.doc(StateDoc)


async def create_state(backend: StorageBackend) -> Any:
    opened = await open_test_session(backend)
    await opened.session.commit(_acquire, context)
    return opened


def set_value(value: int) -> Any:
    async def change(tx: Any) -> None:
        (await tx.doc(StateDoc))["value"] = value

    return change


async def retire(tx: Any) -> None:
    await tx.retire_doc(StateDoc)


async def test_never_creates_an_absent_document(storage_backend: StorageBackend) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    commits = len(storage.commits)
    assert await session.watch_doc(StateDoc, context) is None
    assert len(storage.commits) == commits
    assert storage.mint_count == 0


async def test_keeps_the_acquisition_revision_until_start_and_delivers_exact_committed_frames(
    storage_backend: StorageBackend,
) -> None:
    _, session, publications = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    initial = watch.value
    await session.commit(set_value(1), context)
    await flush()
    first_published = document_changes(publications[-1])[0]
    await session.commit(set_value(2), context)
    await flush()
    second_published = document_changes(publications[-1])[0]
    assert watch.value is initial

    inline = [True]
    deliveries: list[tuple[Any, Any]] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        assert inline[0] is False
        assert watch.value is value
        deliveries.append((value, ops))

    watch.start(listener)
    inline[0] = False
    assert deliveries == []
    await flush()
    assert [value["value"] for value, _ in deliveries] == [1, 2]
    assert deliveries[0][0] is first_published["value"]
    assert deliveries[0][1] is first_published["ops"]
    assert deliveries[1][0] is second_published["value"]
    assert deliveries[1][1] is second_published["ops"]
    assert initial["value"] == 0
    await watch.stop()


async def test_serializes_callbacks_and_buffers_exact_frames_committed_while_one_is_in_flight(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    entered = deferred()
    release = deferred()
    values: list[int] = []
    active = [0, 0]

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        active[0] += 1
        active[1] = max(active[1], active[0])
        values.append(value["value"] if value is not None else -1)
        if len(values) == 1:
            resolve(entered)
            await release
        active[0] -= 1

    watch.start(listener)
    await session.commit(set_value(1), context)
    await entered
    for value in range(2, 21):
        await session.commit(set_value(value), context)
    assert values == [1]
    resolve(release)
    await flush()
    assert active[1] == 1
    assert values == list(range(1, 21))
    await watch.stop()


async def test_allows_a_listener_to_initiate_a_later_session_commit(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    completed = deferred()
    values: list[int] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        current = value["value"] if value is not None else -1
        values.append(current)
        if current == 1:
            await session.commit(set_value(2), context)
        else:
            resolve(completed)

    watch.start(listener)
    await session.commit(set_value(1), context)
    await completed
    assert values == [1, 2]
    await watch.stop()


async def test_collapses_101_pending_commits_to_one_root_replacement(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    for value in range(1, 102):
        await session.commit(set_value(value), context)
    deliveries: list[dict[str, Any]] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        deliveries.append({"value": value["value"] if value is not None else -1, "ops": ops})

    watch.start(listener)
    await flush()
    assert deliveries == [{"value": 101, "ops": [("r", watch.value)]}]
    await watch.stop()


async def test_never_folds_the_in_flight_frame_into_an_overflow_reset(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    entered = deferred()
    release = deferred()
    deliveries: list[dict[str, Any]] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        deliveries.append({"value": value["value"] if value is not None else -1, "ops": ops})
        if len(deliveries) == 1:
            resolve(entered)
            await release

    watch.start(listener)
    await session.commit(set_value(1), context)
    await entered
    for value in range(2, 103):
        await session.commit(set_value(value), context)
    resolve(release)
    await flush()
    assert deliveries[0]["value"] == 1
    assert deliveries[1] == {"value": 102, "ops": [("r", watch.value)]}
    assert len(deliveries) == 2
    await watch.stop()


async def test_folds_retirement_into_an_overflow_reset_and_then_closes(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    for value in range(1, 101):
        await session.commit(set_value(value), context)
    await session.commit(retire, context)
    deliveries: list[dict[str, Any]] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        deliveries.append({"value": value, "ops": ops})

    watch.start(listener)
    assert await watch.closed == {"reason": "retired"}
    assert deliveries == [{"value": None, "ops": [("r", None)]}]


async def test_delivers_replayable_structural_no_op_commits_instead_of_suppressing_them(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    initial = watch.value

    async def rotate(tx: Any) -> None:
        state = await tx.doc(StateDoc)
        first = state["items"].pop(0)
        state["items"].insert(0, first)

    await session.commit(rotate, context)
    newest = await session.snapshot(StateDoc, context)
    assert newest is not initial
    assert newest == initial
    batches: list[Any] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        assert value is newest
        batches.append(ops)

    watch.start(listener)
    await flush()
    assert len(batches) == 1
    assert len(batches[0]) > 0
    await watch.stop()


async def test_preserves_commit_context_values_without_inheriting_producer_cancellation(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    key = create_context_key("watch-test")
    parent_controller = AbortController()
    commit_context = with_context_value(
        key, "newest-commit", with_abort_signal(parent_controller.signal, BACKGROUND_CONTEXT)
    )
    entered = deferred()
    release = deferred()
    delivered: list[Any] = []

    async def listener(_value: Any, _ops: Any, delivery_context: Any) -> None:
        delivered.append(delivery_context)
        resolve(entered)
        await release

    watch.start(listener)
    await session.commit(set_value(1), commit_context)
    await entered
    assert delivered[0].value(key) == "newest-commit"
    assert delivered[0].abort_signal is None
    parent_controller.abort("caller finished")
    stopped = watch.stop()
    assert await stopped == {"reason": "stopped"}
    assert delivered[0].abort_signal is None
    resolve(release)


async def test_keeps_earlier_immutable_revisions_stable(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    initial = watch.value
    delivered: list[Any] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        delivered.append(value)

    watch.start(listener)
    await session.commit(set_value(1), context)
    await flush()
    assert delivered[-1] is watch.value
    assert delivered[-1] is not initial
    assert delivered[-1]["retained"] is initial["retained"]
    assert initial["value"] == 0
    await watch.stop()


async def test_delivers_retirement_and_does_not_follow_recreation(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    old_watch = await session.watch_doc(StateDoc, context)
    values: list[Any] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        values.append(value)

    old_watch.start(listener)

    async def retire_and_recreate(tx: Any) -> None:
        await tx.retire_doc(StateDoc)
        (await tx.doc(StateDoc))["value"] = 10

    await session.commit(retire_and_recreate, context)
    assert await old_watch.closed == {"reason": "retired"}
    assert values == [None]
    assert old_watch.value is None

    replacement = await session.watch_doc(StateDoc, context)
    assert replacement.value["value"] == 10
    await session.commit(set_value(11), context)
    await flush()
    assert old_watch.value is None
    await replacement.stop()


async def test_session_close_discards_retirement_buffered_before_start(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    baseline = watch.value
    await session.commit(retire, context)
    await session.close(context)
    assert await watch.closed == {"reason": "session_closed"}
    assert watch.value is baseline

    async def listener(*_args: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match="stopped"):
        watch.start(listener)


async def test_session_close_discards_retirement_behind_an_in_flight_callback(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    entered = deferred()
    release = deferred()
    values: list[Any] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        values.append(value["value"] if value is not None else None)
        resolve(entered)
        await release

    watch.start(listener)
    await session.commit(set_value(1), context)
    await entered
    await session.commit(retire, context)
    await session.close(context)
    assert await watch.closed == {"reason": "session_closed"}
    resolve(release)
    await flush()
    assert values == [1]


async def test_supports_idempotent_stop_and_rejects_repeated_or_late_start(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)

    async def listener(*_args: Any) -> None:
        return None

    started = await session.watch_doc(StateDoc, context)
    started.start(listener)
    with pytest.raises(RuntimeError, match="already started"):
        started.start(listener)
    first = started.stop()
    second = started.stop()
    assert second is first
    assert await first == {"reason": "stopped"}

    stopped = await session.watch_doc(StateDoc, context)
    await stopped.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        stopped.start(listener)


async def test_cancels_acquisition_without_leaking_a_registered_watch(storage_backend: StorageBackend) -> None:
    storage, session, _ = await create_state(storage_backend)
    await session.unload_documents()
    gate = storage.hold_find_document()
    child = with_cancel(context)
    acquisition = session.watch_doc(StateDoc, child.context)
    await gate.entered
    child.cancel(RuntimeError("cancel acquisition"))
    gate.release()
    with pytest.raises(RuntimeError, match="cancel acquisition"):
        await acquisition
    await session.close(context)


async def test_cancels_future_delivery_without_aborting_an_in_flight_callback(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    child = with_cancel(context)
    watch = await session.watch_doc(StateDoc, child.context)
    entered = deferred()
    release = deferred()
    delivered: list[Any] = []

    async def listener(_value: Any, _ops: Any, delivery_context: Any) -> None:
        delivered.append(delivery_context)
        resolve(entered)
        await release

    watch.start(listener)
    await session.commit(set_value(1), context)
    await entered
    child.cancel()
    assert await watch.closed == {"reason": "cancelled"}
    assert delivered[0].abort_signal is None
    resolve(release)


async def test_session_close_stops_future_delivery_without_joining_an_in_flight_callback(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    entered = deferred()
    release = deferred()
    delivered: list[Any] = []

    async def listener(_value: Any, _ops: Any, delivery_context: Any) -> None:
        delivered.append(delivery_context)
        resolve(entered)
        await release

    watch.start(listener)
    await session.commit(set_value(1), context)
    await entered
    await session.close(context)
    assert delivered[0].abort_signal is None
    assert await watch.closed == {"reason": "session_closed"}
    resolve(release)


async def test_settles_listener_failure_on_only_the_affected_watch(storage_backend: StorageBackend) -> None:
    _, session, _ = await create_state(storage_backend)
    failed = await session.watch_doc(StateDoc, context)
    healthy = await session.watch_doc(StateDoc, context)

    async def failing(*_args: Any) -> None:
        raise RuntimeError("listener failed")

    failed.start(failing)
    healthy_calls = [0]

    async def counting(*_args: Any) -> None:
        healthy_calls[0] += 1

    healthy.start(counting)
    await session.commit(set_value(1), context)
    end = await failed.closed
    assert end["reason"] == "listener_error"
    assert str(end["error"]) == "listener failed"
    await flush()
    assert healthy_calls[0] == 1
    await healthy.stop()


async def test_hydrates_migration_without_writing_and_observes_the_later_exact_edit(
    storage_backend: StorageBackend,
) -> None:
    Old = define_doc(kind="watch.migration", version=1, scope="session", initial=lambda: {"value": 3})
    Current = define_doc(
        kind="watch.migration",
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
    watch = await session.watch_doc(Current, context)
    assert watch.value == {"value": 3, "migrated": True}
    assert len(storage.commits) == commits
    values: list[int] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        values.append(value["value"] if value is not None else -1)

    watch.start(listener)
    await session.commit(acquire(Current), context)
    await flush()
    assert values == []

    async def edit(tx: Any) -> None:
        (await tx.doc(Current))["value"] = 4

    await session.commit(edit, context)
    await flush()
    assert values == [4]
    await watch.stop()


async def test_can_replay_every_delivered_exact_operation_batch_from_the_acquisition_revision(
    storage_backend: StorageBackend,
) -> None:
    _, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    replica = [watch.value]

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        replica[0] = apply_immutable(replica[0], ops)
        assert replica[0] == value

    watch.start(listener)

    async def change(tx: Any) -> None:
        state = await tx.doc(StateDoc)
        state["value"] = 3
        del state["items"][0]

    await session.commit(change, context)
    await flush()
    assert replica[0] == watch.value
    await watch.stop()


async def test_continues_after_the_tracker_cache_unloads(storage_backend: StorageBackend) -> None:
    storage, session, _ = await create_state(storage_backend)
    watch = await session.watch_doc(StateDoc, context)
    baseline = watch.value
    reads = storage.document_read_count
    await session.unload_documents()
    reloaded = await session.snapshot(StateDoc, context)
    assert reloaded == baseline
    assert reloaded is not baseline
    assert storage.document_read_count > reads

    async def listener(*_args: Any) -> None:
        return None

    watch.start(listener)
    await session.commit(set_value(7), context)
    await flush()
    assert watch.value["value"] == 7
    await watch.stop()
