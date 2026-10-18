"""Port of ``test/session-document-retention.test.ts``.

The Session's document cache is bounded by use: unattached documents leave it after the retention
period and load from Storage again; documents with a watch or a document state never leave it.

Deferred to stage 3 (they need the Harness): "follows a Harness settings change at the next use" and
"gives the same events with documents evicted right after use as with documents kept".
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any

import pytest
from durable_session_support import (
    StorageBackend,
    context,
    controlled_storage,
    create_conversation,
    deferred,
    eventually,
    resolve,
    settle_line,
)

from pi_durable import define_doc, define_doc_family
from pi_durable.session import SessionImpl, date_now, set_default_document_retention

CounterDoc = define_doc(
    kind="test.retention.counter",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"count": 0},
)
RewindableDoc = define_doc(
    kind="test.retention.rewindable",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: {"count": 0},
)
#: A third-party conversation family: one member per event.
EventDoc = define_doc_family(
    kind="test.retention.event",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda seq: {"seq": seq},
)
SessionCounterDoc = define_doc(
    kind="test.retention.session-counter", version=1, scope="session", initial=lambda: {"count": 0}
)


def _failing_checkpoint(*_args: Any) -> bool:
    raise RuntimeError("checkpoint failed")


#: A session document whose checkpoint predicate raises, so the commit that changes it fails while it settles.
Failing = define_doc(
    kind="test.retention.failing",
    version=1,
    scope="session",
    initial=lambda: {"count": 0},
    checkpoint_when=_failing_checkpoint,
)

#: The default retention of this run, as the environment set it.
DEFAULT_FOR_RUN = float(os.environ.get("PI_DURABLE_DOCUMENT_RETENTION_MS", 600_000))


class Clock:
    def __init__(self) -> None:
        self.now: float = 1_000


async def open_session(backend: StorageBackend, retention_ms: Any) -> Any:
    """A Session with a controllable clock and retention."""
    clock = Clock()
    storage = await controlled_storage(backend)
    session = SessionImpl(storage, lambda: clock.now, lambda: retention_ms)
    return storage, session, clock, lambda: session.document_cache_sizes


async def bump(session: SessionImpl, conversation_id: Any, by: int = 1) -> None:
    async def change(tx: Any) -> None:
        (await tx.doc(CounterDoc, conversation_id))["count"] += by

    await session.commit(change, context)


async def touch_session_counter(tx: Any) -> None:
    await tx.doc(SessionCounterDoc)


async def nothing(_tx: Any) -> None:
    return None


@pytest.fixture
def restore_default_retention() -> Any:
    # The default as shipped, also in a run that forces another.
    set_default_document_retention(600_000)
    yield
    set_default_document_retention(DEFAULT_FOR_RUN)


async def test_evicts_an_unattached_document_once_its_retention_passed_and_loads_it_again_with_the_same_value(
    storage_backend: StorageBackend,
) -> None:
    storage, session, clock, sizes = await open_session(storage_backend, 60_000)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id, 3)
    assert sizes().cached > 0
    # Not yet: inside the retention period.
    clock.now += 59_000
    await bump(session, conversation_id, 0)
    await settle_line(session)
    assert sizes().cached > 0
    # Past it, with no use since: the next commit evicts it.
    clock.now += 61_000
    await session.commit(nothing, context)
    await session.commit(touch_session_counter, context)
    await settle_line(session)
    reads = storage.document_read_count
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 3}
    assert storage.document_read_count > reads
    await session.close(context)


async def test_evicts_with_retention_0_right_after_use_on_the_line(storage_backend: StorageBackend) -> None:
    _, session, _, sizes = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)

    async def evicted() -> bool:
        await settle_line(session)
        return sizes().cached == 0

    await eventually(evicted)
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 1}
    await session.close(context)


async def test_never_evicts_a_document_a_watch_or_a_document_state_observes_and_evicts_it_once_the_last_detaches(
    storage_backend: StorageBackend,
) -> None:
    storage, session, clock, sizes = await open_session(storage_backend, 1_000)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    watch = await session.watch_doc(CounterDoc, conversation_id, context)
    state = await session.document_state(CounterDoc, conversation_id, context)
    seen: list[int] = []

    async def listener(value: Any, _ops: Any, _context: Any) -> None:
        seen.append(value["count"] if value is not None else -1)

    watch.start(listener)
    clock.now += 1_000_000
    await bump(session, conversation_id)
    await settle_line(session)
    # Still cached: a read long after its retention passed loads nothing from Storage.
    assert state.value["count"] == 2
    await eventually(lambda: 2 in seen)
    reads = storage.document_read_count
    clock.now += 1_000_000
    await session.commit(touch_session_counter, context)
    await settle_line(session)
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 2}
    assert storage.document_read_count == reads
    await watch.stop()
    clock.now += 1_000_000
    await bump(session, conversation_id)
    await settle_line(session)
    assert state.value["count"] == 3
    state.dispose()
    clock.now += 1_000_000
    await session.commit(nothing, context)
    await session.commit(touch_session_counter, context)
    await settle_line(session)
    # The session document was just used; the counter is gone.
    assert sizes().cached == 1
    await session.close(context)


async def test_never_takes_a_tracker_from_a_transaction_that_holds_it(storage_backend: StorageBackend) -> None:
    _, session, clock, _ = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    inside = deferred()
    release = deferred()

    async def holding(tx: Any) -> None:
        draft = await tx.doc(CounterDoc, conversation_id)
        resolve(inside)
        await release
        draft["count"] += 10

    commit = session.commit(holding, context)
    await inside
    # Retention passes while the transaction holds the document; eviction waits on the line behind it.
    clock.now += 1_000_000
    resolve(release)
    await commit
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 11}
    await bump(session, conversation_id)
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 12}
    await session.close(context)


async def test_writes_after_a_reload_continue_the_stored_deltas_and_bases(storage_backend: StorageBackend) -> None:
    storage, session, _, _ = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    writes: list[list[Any]] = []
    commit = storage.commit

    async def recording(batch: Any, commit_context: Any) -> Any:
        writes.append(list(batch))
        return await commit(batch, commit_context)

    storage.commit = recording  # type: ignore[method-assign]
    for _ in range(5):
        await bump(session, conversation_id)
        await settle_line(session)
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 5}
    # Every bump after the first is a change of the same incarnation, never a second creation.
    creations = [
        write
        for batch in writes
        for write in batch
        if write["type"] == "document.create" and write["record"]["kind"] == CounterDoc.definition.kind
    ]
    assert len(creations) == 1
    await session.close(context)
    # And a fresh Session over the same storage reads the same value.
    reopened = SessionImpl(await storage.reopen(), date_now, lambda: 0)
    assert await reopened.snapshot(CounterDoc, conversation_id, context) == {"count": 5}
    await reopened.close(context)


async def test_keeps_history_and_forks_of_rewindable_documents_after_eviction(storage_backend: StorageBackend) -> None:
    _, session, _, _ = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)

    def set_count(count: int) -> Any:
        async def change(tx: Any) -> Any:
            (await tx.doc(RewindableDoc, conversation_id))["count"] = count
            return (await tx.append_entry(conversation_id, {"kind": "note", "data": None}))["id"]

        return change

    first = await session.commit(set_count(1), context)
    await settle_line(session)
    second = await session.commit(set_count(2), context)
    await settle_line(session)
    assert await session.snapshot_as_of(RewindableDoc, conversation_id, first, context) == {"count": 1}
    assert await session.snapshot_as_of(RewindableDoc, conversation_id, second, context) == {"count": 2}

    async def fork(tx: Any) -> Any:
        return (await tx.fork_conversation(conversation_id, first, {"ownership": {"kind": "ownerless"}}))["id"]

    forked = await session.commit(fork, context)
    await settle_line(session)
    assert await session.snapshot(RewindableDoc, forked, context) == {"count": 1}
    assert await session.snapshot(RewindableDoc, conversation_id, context) == {"count": 2}
    await session.close(context)


async def test_bounds_a_third_party_family_with_a_member_per_event(storage_backend: StorageBackend) -> None:
    _, session, _, sizes = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    for seq in range(200):

        async def change(tx: Any, seq: int = seq) -> None:
            await tx.doc(EventDoc, conversation_id, str(seq), seq)

        await session.commit(change, context)

    async def evicted() -> bool:
        await settle_line(session)
        return sizes().cached == 0

    await eventually(evicted)
    assert await session.snapshot(EventDoc, conversation_id, "123", context) == {"seq": 123}
    await session.close(context)


async def test_keeps_the_cache_of_a_long_lived_session_bounded_across_many_conversations(
    storage_backend: StorageBackend,
) -> None:
    _, session, clock, sizes = await open_session(storage_backend, 10_000)
    for _ in range(300):
        conversation_id = await create_conversation(session)
        await bump(session, conversation_id)
        clock.now += 1_000
    await settle_line(session)
    # Only what was used in the last 10 s, about ten conversations' worth.
    assert sizes().cached <= 12
    await session.close(context)


async def test_never_changes_the_outcome_of_a_commit_when_the_clock_throws(storage_backend: StorageBackend) -> None:
    clock = {"now": 1_000, "fail": False}

    def now() -> float:
        if clock["fail"]:
            raise RuntimeError("clock")
        return clock["now"]

    storage = await controlled_storage(storage_backend)
    session = SessionImpl(storage, now, lambda: 0)
    conversation_id = await create_conversation(session)
    clock["fail"] = True
    await bump(session, conversation_id, 42)
    clock["fail"] = False
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 42}
    await session.close(context)


async def test_evicts_after_a_commit_whose_settlement_fails(storage_backend: StorageBackend) -> None:
    _, session, clock, sizes = await open_session(storage_backend, 10_000)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)

    async def create(tx: Any) -> None:
        # Created first: the predicate runs only for changes to an existing document.
        await tx.doc(Failing)

    await session.commit(create, context)
    await settle_line(session)

    async def failing(tx: Any) -> None:
        (await tx.doc(Failing))["count"] += 1
        clock.now += 1_000_000

    # Retention passes inside a commit whose checkpoint predicate raises while it settles.
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await session.commit(failing, context)
    assert sizes().cached == 0
    await session.close(context)


async def test_evicts_promptly_once_the_retention_is_shortened(storage_backend: StorageBackend) -> None:
    retention = {"ms": 600_000}
    storage = await controlled_storage(storage_backend)
    clock = Clock()
    session = SessionImpl(storage, lambda: clock.now, lambda: retention["ms"])
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    retention["ms"] = 0
    clock.now += 2_000
    # A read, no commit: the sweep pending for ten minutes from now moves up.
    await session.snapshot(CounterDoc, conversation_id, context)

    async def evicted() -> bool:
        await settle_line(session)
        return session.document_cache_sizes.cached == 0

    await eventually(evicted)
    await session.close(context)


async def test_does_not_keep_a_replacement_cached_for_an_observer_of_the_retired_incarnation(
    storage_backend: StorageBackend,
) -> None:
    _, session, _, sizes = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    state = await session.document_state(CounterDoc, conversation_id, context)
    await session.commit(lambda tx: tx.retire_doc(CounterDoc, conversation_id), context)
    assert state.value is None
    await bump(session, conversation_id, 5)

    async def evicted() -> bool:
        await settle_line(session)
        return sizes().cached == 0

    await eventually(evicted)
    assert await session.snapshot(CounterDoc, conversation_id, context) == {"count": 5}
    state.dispose()
    await session.close(context)


async def test_leaves_no_pin_behind_when_an_observers_load_loses_a_race_with_close(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _, _ = await open_session(storage_backend, 0)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)

    async def evicted() -> bool:
        await settle_line(session)
        return session.document_cache_sizes.cached == 0

    await eventually(evicted)
    gate = storage.hold_find_document()
    attaching = session.document_state(CounterDoc, conversation_id, context)
    await gate.entered
    closing = session.close(context)
    gate.release()
    with pytest.raises(Exception):  # noqa: B017 - upstream asserts only that it rejects
        await attaching
    await closing
    sizes = session.document_cache_sizes
    assert (sizes.cached, sizes.idle, sizes.attached) == (0, 0, 0)


async def test_evicts_on_its_timer_without_a_commit_and_tolerates_a_detach_called_twice(
    storage_backend: StorageBackend,
) -> None:
    storage = await controlled_storage(storage_backend)
    session = SessionImpl(storage, date_now, lambda: 20)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    state = await session.document_state(CounterDoc, conversation_id, context)
    state.dispose()
    state.dispose()
    until = time.monotonic() + 2
    while session.document_cache_sizes.cached > 0 and time.monotonic() < until:
        await asyncio.sleep(0.01)
    assert session.document_cache_sizes.cached == 0
    await session.close(context)


async def test_migrates_a_document_again_after_its_eviction(storage_backend: StorageBackend) -> None:
    V1 = define_doc(kind="test.retention.migrating", version=1, scope="session", initial=lambda: {"count": 1})
    migrations: list[int] = []

    def migrate(value: Any, _version: int) -> Any:
        migrations.append(1)
        return {"count": value["count"], "migrated": True}

    V2 = define_doc(
        kind="test.retention.migrating",
        version=2,
        scope="session",
        initial=lambda: {"count": 0, "migrated": True},
        migrate=migrate,
    )
    _, session, _, _ = await open_session(storage_backend, 0)

    async def create(tx: Any) -> None:
        await tx.doc(V1)

    await session.commit(create, context)
    await settle_line(session)
    assert await session.snapshot(V2, context) == {"count": 1, "migrated": True}
    await settle_line(session)
    assert await session.snapshot(V2, context) == {"count": 1, "migrated": True}
    # Not written back, so each cold load migrates again.
    assert len(migrations) == 2
    await session.close(context)


async def test_treats_a_retention_that_is_not_a_number_of_milliseconds_as_the_default_and_infinity_as_never(
    storage_backend: StorageBackend, restore_default_retention: Any
) -> None:
    for retention in (math.nan, -5, "soon"):
        _, session, clock, sizes = await open_session(storage_backend, retention)
        conversation_id = await create_conversation(session)
        await bump(session, conversation_id)
        clock.now += 60_000
        await session.commit(nothing, context)
        await settle_line(session)
        # Not dropped at once, as a retention of 0 would: the ten-minute default holds it.
        assert sizes().cached == 1
        await session.close(context)
    _, session, clock, sizes = await open_session(storage_backend, math.inf)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    clock.now += 1e15
    await session.commit(nothing, context)
    await settle_line(session)
    assert sizes().cached == 1
    await session.close(context)


async def test_keeps_no_timer_where_the_host_declines_background_timers_and_evicts_on_commits(
    storage_backend: StorageBackend,
) -> None:
    # Upstream patches `setTimeout` to return a plain number, as in Cloudflare Workers; here the host's
    # `set_timer` declines the background timer by returning None.
    timers: list[float] = []

    def set_timer(_callback: Any, delay_ms: float, _options: Any) -> None:
        timers.append(delay_ms)
        return None

    clock = Clock()
    storage = await controlled_storage(storage_backend)
    session = SessionImpl(storage, lambda: clock.now, lambda: 1_000, set_timer)
    conversation_id = await create_conversation(session)
    await bump(session, conversation_id)
    await bump(session, conversation_id)
    # Tried once, then never again.
    assert len(timers) <= 1
    clock.now += 10_000
    await session.commit(nothing, context)
    await settle_line(session)
    assert session.document_cache_sizes.cached == 0
    await session.close(context)


async def test_keeps_a_reloaded_newer_version_cached_while_an_observer_of_the_older_one_is_attached(
    storage_backend: StorageBackend,
) -> None:
    V1 = define_doc(kind="test.retention.versioned", version=1, scope="session", initial=lambda: {"count": 3})
    V2 = define_doc(
        kind="test.retention.versioned",
        version=2,
        scope="session",
        initial=lambda: {"count": 0, "v2": True},
        migrate=lambda value, _version: {"count": value["count"], "v2": True},
    )
    _, session, _, sizes = await open_session(storage_backend, 0)

    async def create(tx: Any) -> None:
        await tx.doc(V1)

    await session.commit(create, context)
    state = await session.document_state(V1, context)
    assert await session.snapshot(V2, context) == {"count": 3, "v2": True}
    await settle_line(session)
    assert sizes().cached == 1
    state.dispose()

    async def evicted() -> bool:
        await settle_line(session)
        return sizes().cached == 0

    await eventually(evicted)
    await session.close(context)


async def test_queues_one_sweep_job_not_one_per_document_while_a_commit_holds_the_line(
    storage_backend: StorageBackend,
) -> None:
    _, session, _, _ = await open_session(storage_backend, 0)
    conversation_ids: list[Any] = []
    for _ in range(50):
        conversation_id = await create_conversation(session)
        await bump(session, conversation_id)
        conversation_ids.append(conversation_id)

    async def evicted() -> bool:
        await settle_line(session)
        return session.document_cache_sizes.cached == 0

    await eventually(evicted)
    before = session.document_cache_sizes.sweeps

    # One commit loads 50 evicted documents from storage, each touched as it loads; the sweep job stays
    # pending while the line is held.
    async def load_all(tx: Any) -> None:
        for conversation_id in conversation_ids:
            (await tx.doc(CounterDoc, conversation_id))["count"] += 1
            await asyncio.sleep(0)

    await session.commit(load_all, context)
    await settle_line(session)
    assert session.document_cache_sizes.cached == 0
    assert session.document_cache_sizes.sweeps - before <= 2
    await session.close(context)
