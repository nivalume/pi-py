"""Port of ``test/harness-view.test.ts``."""

from __future__ import annotations

import asyncio
from typing import Any, NamedTuple, TypedDict

import pytest
from durable_harness_support import aborted, all_entries, assert_match, chat_setup, open_chat, wait_for
from durable_session_support import context, deferred, flush, resolve

from pi_ai import AbortController, faux_assistant_message, faux_text
from pi_ai.types import AssistantMessage
from pi_chord.context import with_abort_signal
from pi_chord.delta import apply_immutable
from pi_durable import (
    AgentDoc,
    ConversationDocToken,
    Harness,
    InboxDoc,
    LiveDoc,
    MemoryStorage,
    ProviderDoc,
    UsageDoc,
    define_doc,
)

MOUNTED = ["pi.agent", "pi.inbox", "pi.live", "pi.provider", "pi.usage"]


class Frame(NamedTuple):
    value: Any
    ops: Any


class Recorded(NamedTuple):
    initial: Any
    frames: list[Frame]
    watch: Any

    async def stop(self) -> None:
        await self.watch.stop()


async def record(conversation: Any) -> Recorded:
    """Start a watch of ``conversation`` that records its acquisition revision and every delivered frame."""
    watch = await conversation.watch(context)
    initial = watch.value
    frames: list[Frame] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        frames.append(Frame(value, ops))

    watch.start(listener)
    return Recorded(initial, frames, watch)


async def fresh(conversation: Any) -> Any:
    """A freshly built view of ``conversation``, for comparing with an advanced one."""
    state = await conversation.view_state(context)
    value = state.value
    state.dispose()
    return value


async def committed(harness: Harness, conversation: Any, record: Any) -> Any:
    """The view as committed state defines it, read without any mount: the active entries and the built-in documents."""
    docs: dict[str, Any] = {}
    for token in (AgentDoc, LiveDoc, InboxDoc, ProviderDoc, UsageDoc):
        value = await harness.snapshot(token, conversation.id, context)  # type: ignore[arg-type]
        if value is not None:
            docs[token.definition.kind] = value
    return {"conversation": record, "entries": (await conversation.context(context))["entries"], "docs": docs}


def replay(initial: Any, frames: list[Frame]) -> Any:
    """Replay every frame's operations from ``initial``, checking each delivered revision on the way."""
    value = initial
    for frame in frames:
        value = apply_immutable(value, frame.ops)
        assert value == frame.value
    return value


def answer(text: str) -> AssistantMessage:
    return faux_assistant_message([faux_text(text)])


async def drained() -> None:
    """Let watch callbacks, which run after the commit, catch up."""
    await flush()


class Counter:
    count = 0


def touches(harness: Harness, conversation: Any) -> Counter:
    counter = Counter()

    def listener(publication: Any, _context: Any) -> None:
        touched = any(
            (change["type"] == "entry" and change["value"]["conversationId"] == conversation.id)
            or (
                change["type"] == "document"
                and change.get("conversationId") == conversation.id
                and change["record"]["kind"] in MOUNTED
                and len(change["ops"]) > 0
            )
            for change in publication["changes"]
        )
        if touched:
            counter.count += 1

    harness.subscribe_commits(listener)
    return counter


async def note(conversation: Any, kind: str, **extra: Any) -> Any:
    return await conversation.commit(lambda tx: tx.append_entry(conversation.id, {"kind": kind, **extra}), context)


async def test_hydrates_the_active_entries_and_the_built_in_documents() -> None:
    setup = chat_setup()
    setup.faux.set_responses([answer("hello")])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    view = await fresh(root)
    assert view["conversation"] == {"id": root.id}
    assert list(view["entries"]) == await all_entries(root)
    assert sorted(view["docs"].keys()) == MOUNTED
    assert view["docs"]["pi.live"] == {}
    assert view["docs"]["pi.inbox"] == {"items": []}
    await harness.close(context)


async def test_publishes_one_frame_per_touching_commit_whose_operations_rebuild_every_revision() -> None:
    setup = chat_setup()
    release = deferred()

    async def step(_request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        cancelled = asyncio.ensure_future(aborted(options["signal"]))
        try:
            await asyncio.wait([release, cancelled], return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancelled.cancel()
        return answer("a longer answer")

    setup.faux.set_responses([step])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    recorded = await record(root)
    initial, frames = recorded.initial, recorded.frames
    touching = touches(harness, root)
    submission = await root.submit({"type": "input", "content": "hi"}, context)
    await wait_for(lambda: any(frame.value["docs"]["pi.live"].get("generation") for frame in frames))
    resolve(release)
    await submission.wait(context)
    await harness.wait_for_idle(context)
    await drained()
    assert len(frames) == touching.count
    assert replay(initial, frames) == await committed(harness, root, initial["conversation"])
    first_ops = frames[0].ops
    assert any(
        op[0] == "p"
        and tuple(op[1]) == ("entries",)
        and op[2:4] == (0, 0)
        and len(op[4]) == 1
        and op[4][0]["kind"] == "pi.user"
        for op in first_ops
    )
    # Document operations keep their exact shape under the mount path.
    assert ("s", ("docs", "pi.live", "generation"), {"attempt": 1}) in [
        (op[0], tuple(op[1]), *op[2:]) for frame in frames for op in frame.ops
    ]
    await recorded.stop()
    await harness.close(context)


async def test_shares_unchanged_parts_between_revisions_and_skips_commits_that_touch_nothing_mounted() -> None:
    setup = chat_setup()
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    recorded = await record(root)
    initial, frames = recorded.initial, recorded.frames
    await note(other, "note")
    await root.configure({"thinkingLevel": "high"}, context)
    await note(root, "note")
    await drained()
    assert len(frames) == 2
    assert list(frames[0].ops) == [("s", ("docs", "pi.agent", "thinkingLevel"), "high")]
    assert frames[0].value["entries"] is initial["entries"]
    assert frames[0].value["docs"]["pi.live"] is initial["docs"]["pi.live"]
    assert frames[1].value["docs"] is frames[0].value["docs"]
    await recorded.stop()
    await harness.close(context)


async def test_cuts_the_entries_at_a_head_marker_keeping_the_entries_from_its_head() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    await note(root, "a")
    b = await note(root, "b")
    await note(root, "c")
    recorded = await record(root)
    initial, frames = recorded.initial, recorded.frames
    summary = await note(root, "summary", head=b["id"])
    await note(root, "d")
    await root.reset(None, context)
    await drained()
    kinds = [[entry["kind"] for entry in frame.value["entries"]] for frame in frames]
    assert kinds == [["summary", "b", "c"], ["summary", "b", "c", "d"], ["pi.reset"]]
    assert list(frames[0].ops) == [("p", ("entries",), 0, 1, [summary])]
    assert replay(initial, frames) == await committed(harness, root, initial["conversation"])
    await recorded.stop()
    await harness.close(context)


async def test_keeps_only_mounted_entries_for_a_raw_head_write_that_targets_before_the_active_range() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    old = await note(root, "old")
    await root.reset(None, context)
    recorded = await record(root)
    await note(root, "summary", head=old["id"])
    await drained()
    # Model context now starts at `old` again, but the mount never held it (spec §12); a rebuilt mount shows it.
    assert [entry["kind"] for entry in recorded.frames[0].value["entries"]] == ["summary"]
    await recorded.stop()
    assert [entry["kind"] for entry in (await fresh(root))["entries"]] == ["summary", "old"]
    await harness.close(context)


async def test_cuts_a_forks_view_into_its_inherited_entries() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    a = await note(root, "a")
    b = await note(root, "b")
    fork = await root.fork(b["id"], {"ownership": {"kind": "ownerless"}}, context)
    recorded = await record(fork)
    initial, frames = recorded.initial, recorded.frames
    await note(fork, "summary", head=b["id"])
    await drained()
    assert [entry["id"] for entry in initial["entries"]] == [a["id"], b["id"]]
    assert [entry["kind"] for entry in frames[0].value["entries"]] == ["summary", "b"]
    assert frames[0].value == await committed(harness, fork, initial["conversation"])
    await recorded.stop()
    await harness.close(context)


async def test_shows_a_forks_inherited_entries_and_follows_only_the_forks_own_commits() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    first = await note(root, "first")
    fork = await root.fork(first["id"], {"ownership": {"kind": "ownerless"}}, context)
    recorded = await record(fork)
    initial, frames = recorded.initial, recorded.frames
    assert [entry["kind"] for entry in initial["entries"]] == ["first"]
    assert initial["conversation"]["parent"] == {"conversationId": root.id, "at": first["id"]}
    await note(root, "parent")
    await note(fork, "child")
    await drained()
    assert [[entry["kind"] for entry in frame.value["entries"]] for frame in frames] == [["first", "child"]]
    await recorded.stop()
    await harness.close(context)


async def test_unmounts_a_retired_document_and_mounts_its_recreation_whole() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    recorded = await record(root)
    await root.commit(lambda tx: tx.retire_doc(LiveDoc, root.id), context)

    async def recreate(tx: Any) -> None:
        (await tx.doc(LiveDoc, root.id))["tools"] = []

    await root.commit(recreate, context)
    await drained()
    assert [list(frame.ops) for frame in recorded.frames] == [
        [("d", ("docs", "pi.live"))],
        [("s", ("docs", "pi.live"), {"tools": []})],
    ]
    assert "pi.live" not in recorded.frames[0].value["docs"]
    await recorded.stop()
    await harness.close(context)


async def test_replaces_undelivered_frames_with_the_newest_view_after_100_pending_frames() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    watch = await root.watch(context)
    for _ in range(101):
        await note(root, "note")
    frames: list[Frame] = []

    async def listener(value: Any, ops: Any, _context: Any) -> None:
        frames.append(Frame(value, ops))

    watch.start(listener)
    await drained()
    assert len(frames) == 1
    assert list(frames[0].ops) == [("r", frames[0].value)]
    assert len(frames[0].value["entries"]) == 101
    await watch.stop()
    await harness.close(context)


async def test_keeps_states_and_watches_of_one_conversation_independent_and_remounts_after_the_last_one_detaches() -> (
    None
):
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    state = await root.view_state(context)
    recorded = await record(root)
    await note(root, "one")
    await drained()
    assert [entry["kind"] for entry in state.value["entries"]] == ["one"]
    await recorded.stop()
    assert len(recorded.frames) == 1
    await note(root, "two")
    await drained()
    assert [entry["kind"] for entry in state.value["entries"]] == ["one", "two"]
    last = state.value
    state.dispose()
    # No observer is left, so the mount was dropped: a new observer builds a new revision from committed state.
    rebuilt = await fresh(root)
    assert rebuilt == last
    assert rebuilt is not last
    await harness.close(context)


async def test_ends_states_and_watches_at_close_and_rejects_later_acquisition() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    watch = await root.watch(context)
    state = await root.view_state(context)
    await harness.close(context)
    assert await watch.closed == {"reason": "session_closed"}
    assert list(state.value["entries"]) == []
    with pytest.raises(Exception):  # noqa: B017
        await root.watch(context)
    with pytest.raises(Exception):  # noqa: B017
        await root.view_state(context)


async def test_rejects_an_acquisition_cancelled_or_closed_while_it_waits_for_the_session_line() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root

    def hold() -> tuple[Any, Any]:
        release = deferred()

        async def change(_tx: Any) -> None:
            await release

        blocking = root.commit(change, context)
        return (lambda: resolve(release)), blocking

    release, blocking = hold()
    controller = AbortController()
    cancelled = asyncio.ensure_future(root.watch(with_abort_signal(controller.signal, context)))
    controller.abort(RuntimeError("cancelled"))
    release()
    await blocking
    with pytest.raises(Exception, match="cancelled"):
        await cancelled

    release, blocking = hold()
    closed_while_queued = asyncio.ensure_future(root.watch(context))
    closing = asyncio.ensure_future(harness.close(context))
    release()
    await blocking
    with pytest.raises(Exception, match="Harness is closed"):
        await closed_while_queued
    await closing


async def test_shares_one_mount_between_concurrent_observers_and_isolates_a_failing_listener() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    failing = await root.watch(context)
    recorded = await record(root)
    shared = await root.view_state(context)
    assert failing.value is shared.value
    shared.dispose()

    async def fail(*_args: Any) -> None:
        raise RuntimeError("listener failed")

    failing.start(fail)
    await note(root, "one")
    await note(root, "two")
    await drained()
    assert_match(await failing.closed, {"reason": "listener_error"})
    assert len(recorded.frames) == 2
    await recorded.stop()
    await harness.close(context)


class OtherValue(TypedDict):
    n: int


async def test_publishes_one_frame_for_a_commit_that_appends_several_entries_and_edits_a_document() -> None:
    opened = await open_chat(MemoryStorage(), chat_setup())
    harness, root = opened.harness, opened.root
    other: ConversationDocToken[OtherValue] = define_doc(
        kind="app.other", version=1, scope="conversation", history="latest", fork="initial", initial=lambda: {"n": 0}
    )
    recorded = await record(root)
    initial, frames = recorded.initial, recorded.frames

    async def several(tx: Any) -> None:
        await tx.append_entry(root.id, {"kind": "a"})
        (await tx.doc(LiveDoc, root.id))["tools"] = []
        await tx.append_entry(root.id, {"kind": "b"})

    await root.commit(several, context)

    # A document that is not mounted publishes nothing.
    async def unmounted(tx: Any) -> None:
        (await tx.doc(other, root.id))["n"] = 1

    await root.commit(unmounted, context)
    await drained()
    assert len(frames) == 1
    assert [entry["kind"] for entry in frames[0].value["entries"]] == ["a", "b"]
    assert replay(initial, frames) == await committed(harness, root, initial["conversation"])
    await recorded.stop()
    await harness.close(context)
