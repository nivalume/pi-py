"""Port of ``test/harness-storage-failure.test.ts``.

Adaptations:

- "unhandled rejections" are what the event loop's exception handler receives (an unretrieved task exception,
  reported when the task is collected); the tests install a collecting handler and run ``gc.collect()``.
- "a fulfilled thenable that calls only its fulfillment handler" becomes a custom awaitable (an object with
  ``__await__``, neither a coroutine nor a future) that completes.
- "closes the backend only after a paged read underway" lets close reach the Storage guard before it releases the
  first page (JavaScript's microtasks interleave them; a Python coroutine asks for the next page without yielding).
- "a thrown value that cannot be formatted" is an exception whose ``__str__`` raises (JavaScript throws a
  prototype-less object, formatted ``[object Object]``); ``error_message`` falls back to ``object.__repr__``.

Dropped: "contains a rejected promise from another realm" (``node:vm`` realms have no Python counterpart).
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Generator
from typing import Any, NamedTuple

import pytest
from durable_harness_support import aborted, open_tasks
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, flush, resolve

from pi_ai import AbortController, create_models
from pi_chord.context import Context, with_abort_signal
from pi_durable import (
    Harness,
    LiveDoc,
    MemoryStorage,
    SessionFailed,
    StorageRequestError,
    create_registry,
    define_extension,
    define_task,
    watch_events,
)
from pi_durable.harness.harness import safe_now
from pi_durable.harness.view import conversation_views
from pi_durable.session import contained

#: What each ``Waiting`` run saw of its signal.
signals: list[str] = []


async def _work(_task: Any, runtime: Any, _ctx: Context) -> None:
    try:
        await aborted(runtime.signal)
    except BaseException:
        pass
    signals.append("aborted")


async def _abort(_task: Any, runtime: Any, abort_context: Context) -> None:
    await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, abort_context)


async def _nothing(*_args: Any) -> None:
    return None


#: Waits for its signal, records that it saw it, and ends ``aborted`` through its abort handler.
Waiting = define_task(
    name="test.waiting", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": _work}, abort=_abort
)


def controlled() -> ControlledStorage:
    return ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])


class FailingCalls(ControlledStorage):
    """Fails each Storage method named in ``failing`` once, with ``error``.

    A scan of entries with a limit above 1 is the context read's scan of its range, off the Session line;
    ``held_range`` holds it until released.
    """

    def __init__(self) -> None:
        super().__init__(MemoryStorage(), STORAGE_BACKENDS["memory"])
        self.error = RuntimeError("disk gone")
        self.failing: set[str] = set()
        self.held_range: tuple[asyncio.Future[Any], asyncio.Future[Any]] | None = None

    def _fail(self, method: str) -> None:
        if method in self.failing:
            self.failing.discard(method)
            raise self.error

    async def task(self, id: Any, context: Context) -> Any:
        self._fail("task")
        return await super().task(id, context)

    async def scan_entries(self, query: Any, limit: int, cursor: Any, context: Context) -> Any:
        if limit > 1:
            held = self.held_range
            if held is not None:
                resolve(held[0])
                await held[1]
            self._fail("scanRange")
        return await super().scan_entries(query, limit, cursor, context)

    async def mint_id(self) -> int:
        self._fail("mintId")
        return await super().mint_id()


class Started(NamedTuple):
    harness: Harness
    registry: Any
    reports: list[BaseException]
    root: Any


async def started(storage: Any) -> Started:
    signals.clear()
    opened = await open_tasks(storage, [Waiting])
    root = await opened.harness.root(context)
    return Started(opened.harness, opened.registry, opened.reports, root)


async def running(harness: Harness, id: Any) -> None:
    while ((await harness.get_task(id, context)) or {}).get("state", {}).get("status") != "running":
        await asyncio.sleep(0.001)


def note(root: Any) -> Any:
    return root.commit(lambda tx: tx.append_entry(root.id, {"kind": "note"}), context)


def session_failed_by(error: BaseException, cause: BaseException) -> bool:
    return isinstance(error, SessionFailed) and error.cause is cause


async def rejection(awaitable: Any) -> BaseException:
    try:
        await awaitable
    except BaseException as error:
        return error
    raise AssertionError("Expected a rejection")


# ─── a failed Storage call ──────────────────────────────────────────────────


async def test_ends_everything_with_the_first_error_the_call_later_calls_waits_watches_streams_running_tasks() -> None:
    storage = controlled()
    harness, _registry, reports, root = await started(storage)
    id = await root.commit(lambda tx: tx.create_task(Waiting, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await running(harness, id)
    task = asyncio.ensure_future(harness.wait_for_task(id, context))
    idle = asyncio.ensure_future(harness.wait_for_idle(context))
    watch = await harness.watch_doc(LiveDoc, root.id, context)
    assert watch is not None
    stream = await watch_events(harness, root.id, context)
    stream.start(_nothing)

    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    # The call that hit it gets the storage error itself.
    assert await rejection(note(root)) is failure

    assert session_failed_by(await rejection(task), failure)
    assert session_failed_by(await rejection(idle), failure)
    assert await watch.closed == {"reason": "session_failed", "error": failure}
    assert await stream.closed == {"reason": "session_failed", "error": failure}
    # Every later call, and a wait that starts now.
    assert session_failed_by(await rejection(harness.get_task(id, context)), failure)
    assert session_failed_by(await rejection(harness.wait_for_idle(context)), failure)
    assert session_failed_by(await rejection(root.submit({"type": "input", "content": "x"}, context)), failure)
    assert await harness.closed == {"reason": "failed", "error": failure}
    assert signals == ["aborted"]
    assert reports == [failure]
    await harness.close(context)


async def test_fails_it_from_a_read_off_the_session_line_as_a_context_read_makes() -> None:
    storage = FailingCalls()
    harness, _registry, reports, root = await started(storage)
    await note(root)
    storage.failing.add("scanRange")
    assert await rejection(root.context(context)) is storage.error
    assert await harness.closed == {"reason": "failed", "error": storage.error}
    assert reports == [storage.error]
    await harness.close(context)


async def test_ends_storage_calls_underway_with_session_failed_and_closes_once_they_have_settled() -> None:
    storage = FailingCalls()
    harness, _registry, _reports, root = await started(storage)
    release = deferred()
    await note(root)
    entered = deferred()
    storage.held_range = (entered, release)
    reading = asyncio.ensure_future(root.context(context))
    await entered
    # Another call fails the Session while the context read is underway.
    storage.failing.add("mintId")
    assert await rejection(note(root)) is storage.error
    closed = [False]
    harness.closed.add_done_callback(lambda _future: closed.__setitem__(0, True))
    for _ in range(5):
        await asyncio.sleep(0)
    assert closed[0] is False
    # The read succeeds at the backend, yet its caller gets the failure: nothing it read is used.
    resolve(release)
    assert session_failed_by(await rejection(reading), storage.error)
    assert await harness.closed == {"reason": "failed", "error": storage.error}
    await harness.close(context)


async def test_closes_the_backend_only_after_a_paged_read_underway_which_cannot_start_another_page() -> None:
    pages: list[str] = []
    first_page = deferred()
    release_first = deferred()

    class Paged(ControlledStorage):
        async def scan_entries(self, query: Any, limit: int, cursor: Any, context: Context) -> Any:
            if limit > 1:
                pages.append("first" if cursor is None else "next")
                if cursor is None:
                    resolve(first_page)
                    await release_first
            return await super().scan_entries(query, limit, cursor, context)

        async def close(self, context: Context) -> None:
            pages.append("close")
            await super().close(context)

    storage = Paged(MemoryStorage(), STORAGE_BACKENDS["memory"])
    harness, _registry, _reports, root = await started(storage)

    # More entries than one page of the context read.
    async def many(tx: Any) -> None:
        for _ in range(300):
            await tx.append_entry(root.id, {"kind": "note"})

    await root.commit(many, context)
    reading = asyncio.ensure_future(root.context(context))
    await first_page
    closing = asyncio.ensure_future(harness.close(context))
    # Python: let close reach the Storage guard first. JavaScript's microtasks interleave close with the end of the
    # first page; a Python coroutine finishes the page and asks for the next without yielding.
    await flush()
    resolve(release_first)
    assert "is closed" in str(await rejection(reading))
    await closing
    assert pages == ["first", "close"]
    assert await harness.closed == {"reason": "closed"}


async def test_fails_it_when_storage_cannot_close_an_earlier_failure_stays_the_cause() -> None:
    close_error = RuntimeError("close failed")

    class FailingClose(ControlledStorage):
        async def close(self, context: Context) -> None:
            await super().close(context)
            raise close_error

    healthy = await started(FailingClose(MemoryStorage(), STORAGE_BACKENDS["memory"]))
    assert await rejection(healthy.harness.close(context)) is close_error
    assert await healthy.harness.closed == {"reason": "failed", "error": close_error}
    assert healthy.reports == [close_error]

    storage = FailingClose(MemoryStorage(), STORAGE_BACKENDS["memory"])
    failed = await started(storage)
    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    assert await rejection(note(failed.root)) is failure
    assert await failed.harness.closed == {"reason": "failed", "error": failure}
    assert failed.reports == [failure]


async def test_rejects_the_harnesss_own_calls_with_session_failed_after_a_failure() -> None:
    storage = controlled()
    harness, _registry, _reports, root = await started(storage)
    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    assert await rejection(note(root)) is failure
    with pytest.raises(SessionFailed) as raised:
        harness.resume()
    assert raised.value.cause is failure
    assert session_failed_by(await rejection(harness.root(context)), failure)
    assert session_failed_by(await rejection(harness.conversation(root.id, context)), failure)
    assert session_failed_by(
        await rejection(harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)), failure
    )
    await harness.close(context)


async def test_ends_what_a_task_holds_with_session_failed_not_as_cancelled_by_its_signal() -> None:
    ends: list[Any] = []
    watching = deferred()

    async def work(_task: Any, runtime: Any, phase_context: Context) -> None:
        doc = await runtime.watch_doc(LiveDoc, runtime.conversation_id, phase_context)
        handle = await runtime.conversation(runtime.conversation_id, phase_context)

        async def idle_outcome() -> str:
            try:
                await handle.wait_for_idle(phase_context)
            except BaseException as error:
                return type(error).__name__
            return "resolved"

        idle = asyncio.ensure_future(idle_outcome())
        resolve(watching)
        ends.append(await doc.closed)
        ends.append(await idle)

    Watcher = define_task(
        name="test.watcher", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": work}, abort=_nothing
    )
    storage = controlled()
    opened = await open_tasks(storage, [Watcher])
    root = await opened.harness.root(context)
    await root.commit(lambda tx: tx.create_task(Watcher, None, {"ownership": {"kind": "conversation"}}), context)
    opened.harness.resume()
    await watching
    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    assert await rejection(note(root)) is failure
    await opened.harness.closed
    while len(ends) < 2:
        await asyncio.sleep(0)
    assert ends == [{"reason": "session_failed", "error": failure}, "SessionFailed"]
    await opened.harness.close(context)


async def test_fails_it_from_a_failed_id_mint() -> None:
    storage = FailingCalls()
    harness, _registry, _reports, root = await started(storage)
    storage.failing.add("mintId")
    assert await rejection(note(root)) is storage.error
    assert await harness.closed == {"reason": "failed", "error": storage.error}
    await harness.close(context)


async def test_does_not_commit_what_a_callback_did_after_it_caught_a_failed_read() -> None:
    storage = FailingCalls()
    harness, _registry, _reports, root = await started(storage)
    written = len(storage.admitted_commits)
    storage.failing.add("task")

    async def change(tx: Any) -> None:
        try:
            await tx.task(1)
        except BaseException:
            pass
        await tx.append_entry(root.id, {"kind": "note"})

    assert session_failed_by(await rejection(root.commit(change, context)), storage.error)
    assert len(storage.admitted_commits) == written
    await harness.close(context)


async def test_fails_it_for_a_storage_request_error_from_a_commit_whose_effect_is_unknown() -> None:
    storage = controlled()
    harness, _registry, _reports, root = await started(storage)
    error = StorageRequestError("bad batch")
    storage.fail_next_commit(error)
    assert await rejection(note(root)) is error
    assert await harness.closed == {"reason": "failed", "error": error}
    await harness.close(context)


async def test_fails_only_the_call_for_an_invalid_request() -> None:
    harness, _registry, reports, root = await started(MemoryStorage())
    entry = await note(root)

    async def wrong_order() -> Any:
        page = await root.entries({"order": "ascending"}, 1, None, context)
        cursor = page.get("next") or {"after": 1, "order": "ascending"}
        return await root.entries({"order": "descending"}, 1, cursor, context)

    requests = [
        lambda: root.entries({}, 10, {"after": "x"}, context),
        wrong_order,
        lambda: root.commit(lambda tx: tx.scan_entries({"conversationId": 999_999}, 10), context),
        # `pi.live` keeps no history.
        lambda: harness.snapshot_as_of(LiveDoc, root.id, entry["id"], context),
    ]
    for request in requests:
        assert isinstance(await rejection(request()), StorageRequestError)
    assert await note(root) is not None
    ended = [False]
    harness.closed.add_done_callback(lambda _future: ended.__setitem__(0, True))
    await asyncio.sleep(0)
    assert ended[0] is False
    assert reports == []
    await harness.close(context)


async def test_fails_it_for_a_throw_in_its_own_commit_listeners_which_would_leave_memory_behind_storage() -> None:
    storage = controlled()
    harness, _registry, reports, root = await started(storage)
    bug = RuntimeError("listener bug")

    def listener(*_args: Any) -> None:
        raise bug

    harness.observe_commits(listener)
    # The commit itself is durable and resolves; the Session fails behind it.
    assert await note(root) is not None
    assert await harness.closed == {"reason": "failed", "error": bug}
    assert reports == [bug]
    await harness.close(context)


async def test_fails_it_for_a_storage_error_during_a_host_close_before_the_backend_closes() -> None:
    storage = controlled()
    harness, _registry, reports, root = await started(storage)
    held = storage.hold_commits()
    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    committing = asyncio.ensure_future(note(root))
    await held.entered
    closing = asyncio.ensure_future(harness.close(context))
    held.release()
    assert await rejection(committing) is failure
    await closing
    assert await harness.closed == {"reason": "failed", "error": failure}
    assert reports == [failure]


async def test_ends_an_abort_that_joins_a_run_ignoring_its_signal_and_a_reentrant_report_finds_the_session_failed() -> (
    None
):
    release = deferred()
    entered = deferred()

    async def work(*_args: Any) -> None:
        resolve(entered)
        await release

    Stubborn = define_task(
        name="test.stubborn", version=1, initial=lambda _input: {"phase": "work"}, phases={"work": work}, abort=_nothing
    )
    storage = controlled()
    registry = create_registry()
    registry.install(define_extension("tasks", tasks=[Stubborn]))
    reentered: list[Any] = []
    harness_box: list[Harness] = []

    def on_report(*_args: Any) -> None:
        if not reentered:
            reentered.append(asyncio.ensure_future(harness_box[0].wait_for_idle(context)))

    harness = await Harness.open(
        storage, {"models": create_models(), "registry": registry, "onReport": on_report}, context
    )
    harness_box.append(harness)
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Stubborn, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await entered
    held = storage.hold_commits()
    aborting = asyncio.ensure_future(harness.abort_task(id, context))
    await held.entered
    held.release()
    # The abort mark commits and the abort joins the run, which ignores its signal; then the Session fails.
    failure = RuntimeError("disk gone")
    await asyncio.sleep(0.005)
    storage.fail_next_commit(failure)
    assert await rejection(note(root)) is failure
    assert session_failed_by(await rejection(aborting), failure)
    assert session_failed_by(await rejection(reentered[0]), failure)
    resolve(release)
    await harness.close(context)


async def test_does_not_fail_it_for_a_read_its_caller_cancelled() -> None:
    controller = AbortController()
    cancelled = RuntimeError("caller gave up")

    class CancelledRead(MemoryStorage):
        async def scan_entries(self, query: Any, limit: int, cursor: Any, context: Context) -> Any:
            if context.abort_signal is controller.signal:
                controller.abort(cancelled)
                raise cancelled
            return await super().scan_entries(query, limit, cursor, context)

    harness, _registry, _reports, root = await started(CancelledRead())
    cancelling = with_abort_signal(controller.signal, context)
    assert await rejection(root.context(cancelling)) is cancelled
    assert await note(root) is not None
    await harness.close(context)


# ─── listener and report errors ─────────────────────────────────────────────


class LoopErrors:
    """Collects what reaches the event loop's exception handler, such as an unretrieved task exception."""

    def __init__(self) -> None:
        self.errors: list[Any] = []
        self._loop = asyncio.get_running_loop()
        self._previous = self._loop.get_exception_handler()
        self._loop.set_exception_handler(lambda _loop, info: self.errors.append(info))

    def restore(self) -> None:
        self._loop.set_exception_handler(self._previous)


async def test_reports_a_listener_or_on_report_whose_awaitable_fails_without_an_unhandled_error() -> None:
    unhandled = LoopErrors()
    try:
        harness, _registry, reports, root = await started(MemoryStorage())

        async def async_throw(*_args: Any) -> None:
            raise RuntimeError("async listener")

        harness.subscribe_commits(async_throw)
        harness.subscribe_close(async_throw)
        await note(root)
        await harness.close(context)
        await asyncio.sleep(0.01)
        assert [str(error) for error in reports] == ["async listener", "async listener"]

        rejecting = await Harness.open(
            MemoryStorage(),
            {"models": create_models(), "registry": create_registry(), "onReport": async_throw},
            context,
        )

        def sync_throw(*_args: Any) -> None:
            raise RuntimeError("sync listener")

        rejecting.subscribe_commits(sync_throw)
        again = await rejecting.root(context)
        await again.commit(lambda tx: tx.append_entry(again.id, {"kind": "note"}), context)
        await rejecting.close(context)
        await asyncio.sleep(0.01)
        gc.collect()
        await asyncio.sleep(0)
        assert unhandled.errors == []
    finally:
        unhandled.restore()


async def test_reports_a_throwing_commit_or_close_listener_and_runs_the_others_the_commit_stands() -> None:
    harness, _registry, reports, root = await started(MemoryStorage())
    seen: list[str] = []

    def commit_throw(*_args: Any) -> None:
        raise RuntimeError("commit listener")

    def close_throw(*_args: Any) -> None:
        raise RuntimeError("close listener")

    harness.subscribe_commits(commit_throw)
    harness.subscribe_commits(lambda *_args: seen.append("commit"))
    harness.subscribe_close(close_throw)
    harness.subscribe_close(lambda *_args: seen.append("close"))
    assert await note(root) is not None
    await harness.close(context)
    assert seen == ["commit", "close"]
    assert [str(error) for error in reports] == ["commit listener", "close listener"]


async def test_keeps_feeding_the_other_observers_of_a_view_when_one_throws() -> None:
    harness, _registry, reports, root = await started(MemoryStorage())
    views = conversation_views(harness)
    failure = RuntimeError("observer broke")

    class Broken:
        def publication(self, *_args: Any) -> None:
            raise failure

        def close_session(self, _failure: Any) -> None:
            return None

    async def create(*_args: Any) -> Any:
        return Broken()

    await views.attach(root.id, create, context)
    watch = await root.watch(context)
    seen: list[Any] = []

    async def listener(value: Any, *_rest: Any) -> None:
        seen.append(value)

    watch.start(listener)
    await note(root)
    while len(seen) == 0:
        await asyncio.sleep(0.001)
    assert reports == [failure]
    await harness.close(context)
    assert await watch.closed == {"reason": "session_closed"}


class _Completes:
    """An awaitable that is neither a coroutine nor a future, and completes."""

    def __await__(self) -> Generator[Any, None, None]:
        yield from asyncio.sleep(0).__await__()


async def test_contains_a_custom_awaitable_that_completes() -> None:
    unhandled = LoopErrors()
    try:
        errors: list[BaseException] = []
        contained(lambda: _Completes(), errors.append)
        await asyncio.sleep(0.01)
        gc.collect()
        await asyncio.sleep(0)
        assert errors == []
        assert unhandled.errors == []
    finally:
        unhandled.restore()


async def test_reports_a_throwing_watch_listener_which_ends_that_watch() -> None:
    harness, _registry, reports, root = await started(MemoryStorage())
    watch = await harness.watch_doc(LiveDoc, root.id, context)
    assert watch is not None
    failure = RuntimeError("listener broke")

    async def listener(*_args: Any) -> None:
        raise failure

    watch.start(listener)

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, root.id))["compactions"] = []

    await root.commit(change, context)
    assert await watch.closed == {"reason": "listener_error", "error": failure}
    assert reports == [failure]
    await harness.close(context)


class Unformattable(Exception):
    def __str__(self) -> str:
        raise TypeError("cannot format")


async def test_faults_only_the_task_and_ends_only_the_watch_for_a_raised_value_that_cannot_be_formatted() -> None:
    unformattable = Unformattable()

    async def work(*_args: Any) -> None:
        raise unformattable

    Throws = define_task(
        name="test.throws-unformattable",
        version=1,
        initial=lambda _input: {"phase": "work"},
        phases={"work": work},
        abort=_nothing,
    )
    opened = await open_tasks(MemoryStorage(), [Throws])
    root = await opened.harness.root(context)
    watch = await opened.harness.watch_doc(LiveDoc, root.id, context)
    assert watch is not None

    async def listener(*_args: Any) -> None:
        raise unformattable

    watch.start(listener)
    id = await root.commit(lambda tx: tx.create_task(Throws, None, {"ownership": {"kind": "conversation"}}), context)
    opened.harness.resume()
    assert (await opened.harness.wait_for_task(id, context))["state"]["outcome"] == {
        "status": "faulted",
        "error": {"message": object.__repr__(unformattable)},
    }

    async def change(tx: Any) -> None:
        (await tx.doc(LiveDoc, root.id))["compactions"] = []

    await root.commit(change, context)
    assert (await watch.closed)["reason"] == "listener_error"
    assert await note(root) is not None
    await opened.harness.close(context)


def test_reports_a_throwing_clock_once_also_when_the_report_handler_reads_the_clock_again() -> None:
    reports: list[BaseException] = []
    clock_box: list[Any] = []

    def broken() -> float:
        raise RuntimeError("clock broke")

    def report(error: BaseException, _info: Any) -> None:
        reports.append(error)
        # Reentry, as a handler that commits does: the clock is read again before this report returns.
        if len(reports) < 5:
            clock_box[0]()

    clock = safe_now(broken, report)
    clock_box.append(clock)
    assert isinstance(clock(), (int, float))
    clock()
    assert [str(error) for error in reports] == ["clock broke"]


async def test_keeps_working_when_on_report_throws_and_falls_back_to_date_now_when_the_clock_throws() -> None:
    registry = create_registry()
    registry.install(define_extension("tasks", tasks=[Waiting]))

    def broken_clock() -> float:
        raise RuntimeError("clock broke")

    def broken_report(*_args: Any) -> None:
        raise RuntimeError("report broke")

    harness = await Harness.open(
        MemoryStorage(),
        {"models": create_models(), "registry": registry, "now": broken_clock, "onReport": broken_report},
        context,
    )
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Waiting, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await running(harness, id)
    started_at = ((await harness.get_task(id, context)) or {}).get("startedAt")
    assert isinstance(started_at, (int, float))
    await harness.abort_task(id, context)
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)
