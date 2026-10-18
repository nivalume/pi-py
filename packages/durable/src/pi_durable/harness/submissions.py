"""Admission, waits, and withdrawal of durable submissions (port of ``harness/submissions.ts``)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from pi_chord.context import Context

from ..errors import ConversationBusy
from ..session._futures import rejected, resolved
from ..types import CommitPublication, ConversationId, Storage, SubmissionId, SubmissionRecord, Tx
from .inbox import (
    InboxDoc,
    append_input,
    apply_boundary,
    draft_json,
    has_queued_reset,
    is_display_only,
    is_stale,
    pick_input_entry,
    prepare_boundary,
    remove_inbox_item,
    user_content_json,
)
from .live import LiveDoc
from .types import InputEntry, SettledSubmissionRecord, Submission, SubmissionDraft
from .util import Waiters, closed_error, eager

__all__ = ["SubmissionHandle", "Submissions", "admit_submission"]

AbortResult = Literal["aborted", "already_placed", "settled"]


class Submissions:
    """Admission, waits, and withdrawal of the durable submissions of one Harness."""

    __slots__ = ("_closed", "_now", "_queue_modes", "_resume", "_session", "_storage", "_waiters")

    def __init__(
        self,
        session: Any,
        storage: Storage,
        now: Callable[[], float],
        queue_modes: Callable[[], Any],
        resume: Callable[[], None],
    ) -> None:
        self._session = session
        self._storage = storage
        self._now = now
        #: Read at each admission, on the Session line.
        self._queue_modes = queue_modes
        #: Enable task scheduling; submitting or waiting asks for progress.
        self._resume = resume
        self._waiters: Waiters[SubmissionId, SettledSubmissionRecord] = Waiters()
        self._closed = False
        session.observe_commits(lambda publication, _context: self._observe(publication))

        def on_close() -> None:
            self._closed = True
            self._waiters.reject_all(closed_error(self._session))

        session.subscribe_close(on_close)

    def submit(
        self, conversation_id: ConversationId, draft: SubmissionDraft, context: Context
    ) -> asyncio.Future[Submission]:
        """Admit a submission in one commit; see :func:`admit_submission`. Admission happens at the call."""
        try:
            self._resume()
        except Exception as error:
            return rejected(error)
        committed = self._session.commit_with(
            lambda tx: admit_submission(tx, conversation_id, draft, self._now(), self._queue_modes()), context
        )
        return eager(committed, lambda id: SubmissionHandle(id, self))

    def get(self, id: SubmissionId, context: Context) -> asyncio.Future[Submission | None]:
        """Handle for an existing submission, or ``None``."""
        record = self._session.read_on_line(lambda: self._storage.submission(id, context))
        return eager(record, lambda found: None if found is None else SubmissionHandle(found["id"], self))

    def status(self, id: SubmissionId, context: Context) -> asyncio.Future[SubmissionRecord]:
        async def read() -> SubmissionRecord:
            record = await self._storage.submission(id, context)
            if record is None:
                raise RuntimeError(f"Submission {id} does not exist")
            return record

        return self._session.read_on_line(read)

    def wait(self, id: SubmissionId, context: Context) -> asyncio.Future[SettledSubmissionRecord]:
        try:
            self._resume()
        except Exception as error:
            return rejected(error)

        # Check and register on the line so no settling publication falls between them.
        async def check() -> Awaitable[SettledSubmissionRecord]:
            record = await self._storage.submission(id, context)
            if record is None:
                raise RuntimeError(f"Submission {id} does not exist")
            if _is_settled(record):
                return resolved(record)  # type: ignore[arg-type]
            # Close rejects registered waiters synchronously and may begin during the read.
            if self._closed:
                raise closed_error(self._session)
            return self._waiters.add(id, context)

        found = self._session.read_on_line(check)

        async def wait() -> SettledSubmissionRecord:
            return await (await found)

        return asyncio.ensure_future(wait())

    def abort(
        self, id: SubmissionId, context: Context, conversation_id: ConversationId | None = None
    ) -> asyncio.Future[AbortResult | Literal["not_found"]]:
        """Withdraw a queued submission and remove its inbox item; placed inputs and settled submissions are reported."""

        async def change(tx: Any) -> AbortResult | Literal["not_found"]:
            record = await tx.submission(id)
            if record is None or (conversation_id is not None and record["conversationId"] != conversation_id):
                return "not_found"
            if record["status"] == "queued":
                tx.settle_submission(id, {"status": "unanswered", "reason": "aborted"})
                await remove_inbox_item(tx, record["conversationId"], id)
                return "aborted"
            return "already_placed" if record["status"] == "placed" else "settled"

        return self._session.commit_with(change, context)

    def _observe(self, publication: CommitPublication) -> None:
        for change in publication["changes"]:
            if change["type"] != "submission" or not _is_settled(change["value"]):
                continue
            self._waiters.resolve(change["value"]["id"], change["value"])  # type: ignore[arg-type]


class SubmissionHandle:
    """Awaitable host object for one durably admitted submission."""

    __slots__ = ("_submissions", "id")

    def __init__(self, id: SubmissionId, submissions: Submissions) -> None:
        self.id = id
        self._submissions = submissions

    def __repr__(self) -> str:
        return f"Submission({self.id})"

    def status(self, context: Context) -> asyncio.Future[SubmissionRecord]:
        return self._submissions.status(self.id, context)

    def wait(self, context: Context) -> asyncio.Future[SettledSubmissionRecord]:
        return self._submissions.wait(self.id, context)

    def abort(self, context: Context) -> asyncio.Future[AbortResult]:
        def result(found: str) -> Any:
            if found == "not_found":
                raise RuntimeError(f"Submission {self.id} does not exist")
            return found

        return eager(self._submissions.abort(self.id, context), result)


def _is_settled(record: SubmissionRecord) -> bool:
    return record["status"] in ("done", "unanswered")


def _check_input_entry(entry: InputEntry) -> None:
    """Reject an input entry an untyped caller got wrong: its kind must be a non-empty string outside ``pi.``."""
    kind: Any = entry.get("kind") if isinstance(entry, dict) else None
    if not isinstance(kind, str) or len(kind) == 0 or kind.startswith("pi."):
        shown = "undefined" if kind is None else str(kind)
        raise TypeError(f'An input entry\'s kind must be a non-empty string not starting with "pi.", got {shown}')


async def admit_submission(
    tx: Tx, conversation_id: ConversationId, draft: SubmissionDraft, now: float, queue_modes: Any
) -> SubmissionId:
    """Admit a submission inside a commit (spec §6); ``Conversation.submit()`` and conversation-owned compactions share it.

    A known request ID returns its existing submission without writing. A display-only write (only ``kind`` and
    ``data``) is appended and settled ``done`` at once, busy or not; only behind a queued reset it queues. A busy
    conversation queues anything else in ``pi.inbox``, or rejects ``whenBusy: "reject"`` input with
    ``ConversationBusy``. An idle conversation with queued items queues it behind them and runs a final
    boundary. Otherwise idle input places its entry and starts a run, and an idle write appends its entry and
    settles ``done``, or ``stale`` when its head reaches before the active range.
    """
    from .generation import start_run

    if draft["type"] == "input" and "entry" in draft:
        _check_input_entry(draft["entry"])
    request: dict[str, Any] = {}
    if "requestId" in draft:
        request_id = draft["requestId"]
        existing = await tx.submission_by_request(conversation_id, request_id)
        if existing is not None:
            if existing["type"] != draft["type"]:
                raise RuntimeError(f"Request {request_id} already identifies a submission of type {existing['type']}")
            return existing["id"]
        request = {"requestId": request_id}
    if (
        draft["type"] == "write"
        and is_display_only(draft["entry"])
        and not has_queued_reset((await tx.doc(InboxDoc, conversation_id))["items"])
    ):
        entry = await tx.append_entry(conversation_id, draft["entry"])
        write: Any = {
            "conversationId": conversation_id,
            **request,
            "type": "write",
            "status": "done",
            "entry": entry["id"],
        }
        return (await tx.create_submission(write))["id"]
    live = await tx.doc(LiveDoc, conversation_id)
    busy = "run" in live
    if busy and draft["type"] == "input" and draft.get("whenBusy") == "reject":
        raise ConversationBusy(conversation_id)
    # A boundary reads the table, so it is prepared before the first table write; a busy one needs none.
    boundary = None if busy else await prepare_boundary(tx, conversation_id, queue_modes)
    if boundary is None or len(boundary.inbox["items"]) > 0:
        queued: Any = {"conversationId": conversation_id, **request, "type": draft["type"], "status": "queued"}
        id = (await tx.create_submission(queued))["id"]
        items = (boundary.inbox if boundary is not None else await tx.doc(InboxDoc, conversation_id))["items"]
        if draft["type"] == "write":
            items.append({"id": id, "mode": "write", "entry": draft_json(draft["entry"])})
        else:
            mode = "steer" if draft.get("whenBusy") == "steer" else "followUp"
            item: dict[str, Any] = {"id": id, "mode": mode, "content": user_content_json(draft["content"])}
            if "entry" in draft:
                item["entry"] = pick_input_entry(draft["entry"])
            items.append(cast(Any, item))
        if boundary is None:
            return id
        result = await apply_boundary(tx, boundary, "final", now)
        if len(result.users) > 0:
            await start_run(tx, conversation_id, live, result.users)
        return id
    if draft["type"] == "write":
        if is_stale(boundary, draft["entry"]):
            stale: Any = {
                "conversationId": conversation_id,
                **request,
                "type": "write",
                "status": "unanswered",
                "reason": "stale",
            }
            return (await tx.create_submission(stale))["id"]
        entry = await tx.append_entry(conversation_id, draft["entry"])
        done: Any = {
            "conversationId": conversation_id,
            **request,
            "type": "write",
            "status": "done",
            "entry": entry["id"],
        }
        return (await tx.create_submission(done))["id"]
    entry = await append_input(tx, conversation_id, draft["content"], draft.get("entry"), now)
    placed: Any = {
        "conversationId": conversation_id,
        **request,
        "type": "input",
        "status": "placed",
        "entry": entry["id"],
    }
    id = (await tx.create_submission(placed))["id"]
    await start_run(tx, conversation_id, live, [id])
    return id
