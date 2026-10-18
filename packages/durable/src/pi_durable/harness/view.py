"""Conversation view mounts (port of ``harness/view.ts``)."""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, TypeVar, cast

from typing_extensions import ReadOnly

from pi_chord.context import Context, without_abort_signal
from pi_chord.delta import Op, apply_immutable
from pi_chord.json import JsonObject
from pi_chord.state import AttachedReplicatedState, ReplicatedStateSourceOptions, replicated_state

from ..session import CommittedStateSource, CommittedWatch, contained
from ..types import (
    CommitPublication,
    ConversationDocToken,
    ConversationId,
    ConversationRecord,
    DocumentId,
    EntryRecord,
    Storage,
    WatchHandle,
)
from .agent import AgentDoc
from .context import active_entries, capture_context_bounds
from .inbox import InboxDoc
from .live import LiveDoc
from .provider import ProviderDoc
from .usage import UsageDoc
from .util import closed_error

__all__ = ["ConversationView", "ConversationViews", "ViewObserver", "conversation_views"]

Out = TypeVar("Out")


class ConversationView(TypedDict):
    """Structural mount of one conversation's active transcript and built-in documents (spec §9.3)."""

    conversation: ReadOnly[ConversationRecord]
    #: Raw active entries, as ``ContextView["entries"]``: the head marker, then the non-head entries from its head.
    entries: ReadOnly[Sequence[EntryRecord]]
    #: Built-in conversation documents keyed by kind; absent documents are absent.
    docs: ReadOnly[dict[str, JsonObject]]


class ViewObserver(Protocol):
    """Receives each next revision of a mount, and the Session's close. ``advance`` and ``publication`` are optional."""

    def close_session(self, failure: Any) -> None: ...


_MOUNTED: tuple[ConversationDocToken[Any], ...] = (AgentDoc, LiveDoc, InboxDoc, ProviderDoc, UsageDoc)
_MOUNTED_KINDS = frozenset(token.definition.kind for token in _MOUNTED)


@dataclass(slots=True, eq=False)
class _Mount:
    value: ConversationView
    #: Mounted incarnation and definition version per kind; another incarnation or version is set whole.
    docs: dict[str, tuple[DocumentId, int]]
    observers: dict[Any, None] = field(default_factory=dict)


_VIEWS: weakref.WeakKeyDictionary[Any, ConversationViews] = weakref.WeakKeyDictionary()


def conversation_views(harness: object) -> ConversationViews:
    """The view mounts of a Harness, for adapters built on them."""
    views = _VIEWS.get(harness)
    if views is None:
        raise RuntimeError("Not a Harness")
    return views


class ConversationViews:
    """The Harness's conversation view mounts.

    At most one per conversation, built on the Session line by its first observer and dropped with its last.
    Each mount advances from the Session's commit publications, which are durable.
    """

    __slots__ = ("_closed", "_mounts", "_session", "_storage")

    def __init__(self, session: Any, storage: Storage) -> None:
        self._session = session
        self._storage = storage
        self._mounts: dict[ConversationId, _Mount] = {}
        self._closed = False
        _VIEWS[session] = self

        def on_commit(publication: CommitPublication, context: Context) -> None:
            for id, mount in list(self._mounts.items()):
                _advance(id, mount, publication, context, self.report)

        session.observe_commits(on_commit)

        def on_close() -> None:
            self._closed = True
            # A copy: an observer's close can detach it, which drops its mount (JavaScript Maps allow that mid-loop).
            for mount in list(self._mounts.values()):
                for observer in list(mount.observers):
                    contained(lambda observer=observer: observer.close_session(self._session.failure), self.report)
            self._mounts.clear()

        session.subscribe_close(on_close)

    def report(self, error: BaseException) -> None:
        """Report an observer's listener error through the Harness."""
        self._session.report(error)

    def state(self, id: ConversationId, context: Context) -> asyncio.Future[AttachedReplicatedState[ConversationView]]:
        """A disposable read-only Chord state of the view.

        Eager (section 5 of DESIGN.md): the attachment is admitted to the Session line at the call, as upstream's
        ``async`` body does before its first ``await``, so a ``close()`` that follows the call cannot overtake it.
        """
        attached = self.attach(id, lambda value, release, _storage: CommittedStateSource(value, release), context)

        async def finish() -> AttachedReplicatedState[ConversationView]:
            observer, detach = await attached
            try:
                return replicated_state(observer, ReplicatedStateSourceOptions(on_error=self._session.report))
            except BaseException:
                detach()
                raise

        return asyncio.ensure_future(finish())

    def watch(self, id: ConversationId, context: Context) -> asyncio.Future[WatchHandle[ConversationView]]:
        """A serialized exact-frame watch of the view; cancelling ``context`` stops it. Eager, as :meth:`state`."""
        attached = self.attach(
            id, lambda value, release, _storage: CommittedWatch(value, release, self._session.report), context
        )

        async def finish() -> WatchHandle[ConversationView]:
            observer, _detach = await attached
            signal = context.abort_signal
            if signal is not None and signal.aborted:
                observer.cancel()
                context.throw_if_aborted()
            if signal is not None:
                observer.observe_cancellation(signal)
            return observer

        return asyncio.ensure_future(finish())

    def attach(
        self,
        id: ConversationId,
        create: Callable[[ConversationView, Callable[[], None], Storage], Out | Awaitable[Out]],
        context: Context,
    ) -> Awaitable[tuple[Out, Callable[[], None]]]:
        """Register an observer created from the current revision, atomically on the Session line.

        It sees every later publication and nothing earlier. ``create`` may read committed Storage, still on
        the line. ``detach`` drops it, and the mount with its last observer.
        """

        async def job() -> tuple[Out, Callable[[], None]]:
            mount = self._mounts.get(id)
            if mount is None:
                mount = await self._build(id, context)
            holder: list[Any] = []

            def detach() -> None:
                if holder:
                    mount.observers.pop(holder[0], None)
                if len(mount.observers) == 0 and self._mounts.get(id) is mount:
                    del self._mounts[id]

            created = create(mount.value, detach, self._storage)
            observer = await created if inspect.isawaitable(created) else created
            holder.append(observer)
            # Close or cancellation may begin while the mount hydrates; register nothing then.
            if self._closed:
                raise closed_error(self._session)
            context.throw_if_aborted()
            self._mounts[id] = mount
            mount.observers[observer] = None
            return observer, detach  # type: ignore[return-value]

        return self._session.read_on_line(job)

    async def _build(self, id: ConversationId, context: Context) -> _Mount:
        conversation = await self._storage.conversation(id, context)
        if conversation is None:
            raise RuntimeError(f"Conversation {id} does not exist")
        bounds = await capture_context_bounds(self._storage, id, context)
        entries = await active_entries(self._storage, id, bounds, context)
        docs: dict[str, JsonObject] = {}
        incarnations: dict[str, tuple[DocumentId, int]] = {}
        for token in _MOUNTED:
            loaded = await self._session.conversation_document_on_line(token, id, context)
            if loaded is None:
                continue
            record, version, value = loaded
            docs[token.definition.kind] = value
            incarnations[token.definition.kind] = (record["id"], version)
        return _Mount({"conversation": conversation, "entries": entries, "docs": docs}, incarnations)


def _advance(
    id: ConversationId,
    mount: _Mount,
    publication: CommitPublication,
    context: Context,
    report: Callable[[BaseException], None],
) -> None:
    """Derive the mount's operations from one publication, apply them, and hand the revision to every observer.

    One that raises is reported through ``report``, and the others still get it.
    """
    doc_ops: list[Op] = []
    entry_ops: list[Op] = []
    entries: Sequence[EntryRecord] = mount.value["entries"]
    entries_changed = False
    # Entry writes are published in ID order.
    for change in publication["changes"]:
        if change["type"] == "entry" and change["value"]["conversationId"] == id:
            entry = change["value"]
            if "head" not in entry:
                entry_ops.append(("p", ("entries",), len(entries), 0, [entry]))  # type: ignore[arg-type]
                entries = [*entries, entry]
                entries_changed = True
                continue
            # A head marker keeps the non-head entries from its head, which are always a suffix, and goes in front.
            target = entry["head"]
            kept = next(
                (
                    index
                    for index, candidate in enumerate(entries)
                    if "head" not in candidate and candidate["id"] >= target
                ),
                len(entries),
            )
            entry_ops.append(("p", ("entries",), 0, kept, [entry]))  # type: ignore[arg-type]
            entries = [entry, *entries[kept:]]
            entries_changed = True
            continue
        if change["type"] != "document" or change.get("conversationId") != id:
            continue
        record = change["record"]
        kind = record["kind"]
        if kind not in _MOUNTED_KINDS or "key" in record:
            continue
        path = ("docs", kind)
        mounted = mount.docs.get(kind)
        if change["value"] is None:
            if mounted is None or mounted[0] != record["id"]:
                continue
            del mount.docs[kind]
            doc_ops.append(("d", path))
        elif mounted is not None and mounted[0] == record["id"] and mounted[1] == change["version"]:
            for op in change["ops"]:
                doc_ops.append(_prefixed(op, path))
        else:
            mount.docs[kind] = (record["id"], change["version"])  # type: ignore[assignment]
            doc_ops.append(("s", path, change["value"]))
    before = mount.value
    ops = [*doc_ops, *entry_ops]
    frame_context = without_abort_signal(context)
    if len(ops) > 0:
        value = mount.value if len(doc_ops) == 0 else apply_immutable(mount.value, doc_ops)
        mount.value = value if not entries_changed else {**value, "entries": entries}  # type: ignore[typeddict-item]
        for observer in list(mount.observers):
            advance = getattr(observer, "advance", None)
            if advance is not None:
                contained(lambda advance=advance: advance(mount.value, ops, frame_context), report)
    for observer in list(mount.observers):
        on_publication = getattr(observer, "publication", None)
        if on_publication is not None:
            contained(
                lambda on_publication=on_publication: on_publication(
                    before, mount.value, ops, publication, frame_context
                ),
                report,
            )


def _prefixed(op: Any, prefix: tuple[str, ...]) -> Op:
    """``op`` moved under ``prefix``; a root replacement becomes a set of the prefix."""
    verb = op[0]
    if verb == "r":
        return cast(Op, ("s", prefix, op[1]))
    path = (*prefix, *op[1])
    if verb == "p":
        return cast(Op, ("p", path, op[2], op[3], op[4]))
    if verb == "d":
        return cast(Op, ("d", path))
    return cast(Op, (verb, path, op[2]))  # "m", "s", "a", "t"
