"""Experimental agent events of one conversation (port of ``harness/events.ts``, spec §9.4).

Events are dicts with upstream's keys (``{"type": "message_end", "entry": ...}``). Messages inside events are
pi-ai data classes: entries hold them so, and an in-flight partial (stored in ``pi.live`` as JSON) is decoded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, TypeAlias

from pi_ai.serialization import from_dict, message_from_dict
from pi_ai.types import Usage
from pi_chord.context import Context
from pi_chord.delta import Op

from ..session import CommittedWatch
from ..types import CommitPublication, ConversationId, EntryRecord, SubmissionRecord, TaskId, WatchEnd
from .agent import AgentDoc
from .tool import NestedCallDoc
from .usage import UsageDoc
from .util import js_same, scan_all
from .view import ConversationView, conversation_views

__all__ = ["AgentEvent", "AgentEventStream", "MessageChange", "SnapshotEvent", "ToolEventCall", "watch_events"]

#: One change to the in-flight assistant message, relative to that message (a dict with ``type``).
MessageChange: TypeAlias = dict[str, Any]
#: The ``snapshot`` event at attachment.
SnapshotEvent: TypeAlias = dict[str, Any]
#: Experimental agent event, shaped like the coding agent's session events (spec §9.4).
AgentEvent: TypeAlias = dict[str, Any]
#: Which call a tool event is about: ``toolCallId``, ``toolName``, and ``taskId``, ``parentToolCallId``,
#: ``parentTaskId`` when known.
ToolEventCall: TypeAlias = dict[str, Any]


class AgentEventStream:
    """Serialized stream of one conversation's event batches, one per commit."""

    __slots__ = ("_watch", "snapshot")

    def __init__(self, snapshot: SnapshotEvent, watch: CommittedWatch[Sequence[AgentEvent]]) -> None:
        #: The ``snapshot`` event at attachment.
        self.snapshot = snapshot
        self._watch = watch

    def start(self, listener: Callable[[Sequence[AgentEvent], Context], Awaitable[None] | None]) -> None:
        """Install the sole listener: ``listener(events, context)``, sync or async."""
        self._watch.start(lambda events, _ops, context: listener(events, context))

    def stop(self) -> asyncio.Future[WatchEnd]:
        return self._watch.stop()

    @property
    def closed(self) -> asyncio.Future[WatchEnd]:
        return self._watch.closed


def _parts(view: ConversationView) -> tuple[dict[str, Any], Any, Any, Any]:
    docs = view["docs"]
    return docs.get("pi.live", {}), docs.get("pi.inbox"), docs.get("pi.agent"), docs.get("pi.usage")


def _message(value: Any) -> Any:
    return None if value is None else message_from_dict(value)


def _snapshot_of(view: ConversationView) -> SnapshotEvent:
    live, inbox, agent, usage = _parts(view)
    event: dict[str, Any] = {"type": "snapshot", "entries": view["entries"]}
    if "run" in live:
        event["run"] = {"inputs": live["run"]["inputs"]}
    if "generation" in live:
        generation = dict(live["generation"])
        if "message" in generation:
            generation["message"] = _message(generation["message"])
        event["generation"] = generation
    event["tools"] = live.get("tools", [])
    event["nestedTools"] = live.get("nestedTools", [])
    event["compactions"] = live.get("compactions", [])
    if "compactionFailure" in live:
        event["compactionFailure"] = live["compactionFailure"]
    event["inbox"] = _queued(inbox)
    event["agent"] = agent if agent is not None else AgentDoc.definition.initial()
    event["usage"] = usage if usage is not None else UsageDoc.definition.initial()
    return event


def _queued(inbox: Any) -> list[dict[str, Any]]:
    return [{"id": item["id"], "mode": item["mode"]} for item in ([] if inbox is None else inbox["items"])]


async def watch_events(harness: Any, conversation_id: ConversationId, context: Context) -> AgentEventStream:
    """Experimental: attach to one conversation's agent events (spec §9.4).

    The snapshot and the registration for later commits are captured atomically on the Session line; overflow
    replaces undelivered batches with one snapshot.
    """
    views = conversation_views(harness)
    holder: dict[str, Any] = {}

    async def create(initial: ConversationView, release: Callable[[], None], storage: Any) -> Any:
        # Generations whose held outcome already ended their turn, read on the line with the snapshot.
        query: Any = {"conversationId": conversation_id, "kind": "pi.generation", "status": "completing"}
        completing = await scan_all(lambda cursor: storage.scan_tasks(query, 100, cursor, context))
        held: set[TaskId[Any]] = {record["id"] for record in completing}
        current = [initial]
        holder["snapshot"] = _snapshot_of(initial)
        # Batches are the watch's values; an overflow delivers a snapshot of the newest view instead.
        watch: CommittedWatch[Sequence[AgentEvent]] = CommittedWatch(
            [], release, views.report, lambda: [_snapshot_of(current[0])]
        )
        holder["watch"] = watch

        class Observer:
            def publication(
                self,
                before: ConversationView,
                after: ConversationView,
                ops: Sequence[Op],
                publication: CommitPublication,
                commit_context: Context,
            ) -> None:
                current[0] = after
                events = _translate(conversation_id, before, after, ops, publication, held)
                if len(events) > 0:
                    watch.advance(events, [], commit_context)

            def close_session(self, failure: Any) -> None:
                watch.close_session(failure)

        return Observer()

    await views.attach(conversation_id, create, context)
    watch: CommittedWatch[Sequence[AgentEvent]] = holder["watch"]
    # Like a watch, the acquisition context governs the stream's lifetime.
    signal = context.abort_signal
    if signal is not None and signal.aborted:
        watch.cancel()
        context.throw_if_aborted()
    if signal is not None:
        watch.observe_cancellation(signal)
    return AgentEventStream(holder["snapshot"], watch)


def _result_of(entries: Sequence[EntryRecord], call_id: str) -> EntryRecord | None:
    """The tool result for ``call_id`` among ``entries``."""
    for entry in entries:
        model = entry.get("model")
        message = model[0] if model else None
        if message is not None and message.role == "toolResult" and message.tool_call_id == call_id:  # type: ignore[union-attr]
            return entry
    return None


def _translate(
    conversation_id: ConversationId,
    before: ConversationView,
    after: ConversationView,
    view_ops: Sequence[Op],
    publication: CommitPublication,
    held: set[TaskId[Any]],
) -> list[AgentEvent]:
    """Every event one publication causes, in the order of spec §9.4."""
    entries: list[EntryRecord] = []
    tasks: dict[TaskId[Any], Any] = {}
    submissions: list[SubmissionRecord] = []
    # Nested results this commit stored, by nested task ID; read now, since the caller's documents may retire next.
    nested_results: dict[TaskId[Any], Any] = {}
    for change in publication["changes"]:
        if (
            change["type"] == "document"
            and change["record"]["kind"] == NestedCallDoc.definition.kind
            and change["value"] is not None
            and change["conversationId"] == conversation_id
        ):
            state: Any = change["value"]
            if "taskId" in state and "result" in state:
                nested_results[state["taskId"]] = _nested_result(state["result"])
        if change["type"] == "entry" and change["value"]["conversationId"] == conversation_id:
            entries.append(change["value"])
        if change["type"] == "task" and change["value"]["conversationId"] == conversation_id:
            tasks[change["value"]["id"]] = change["value"]
        if change["type"] == "submission" and change["value"]["conversationId"] == conversation_id:
            submissions.append(change["value"])
    if len(view_ops) == 0 and len(entries) == 0 and len(tasks) == 0 and len(submissions) == 0:
        return []
    # Entries are appended in ID order; submission records are published in the order the commit first touched them.
    submissions.sort(key=lambda record: record["id"])
    was_live, was_inbox, was_agent, was_usage = _parts(before)
    now_live, now_inbox, now_agent, now_usage = _parts(after)
    events: list[AgentEvent] = []

    # Progress: tool starts, the in-flight message, tool updates, retry and deferred state.
    slots_before = {slot["callId"]: slot for slot in was_live.get("tools", [])}
    slots = now_live.get("tools", [])
    nested_before = {slot["taskId"]: slot for slot in was_live.get("nestedTools", [])}
    nested = now_live.get("nestedTools", [])
    for slot in [*slots, *nested]:
        previous = nested_before.get(slot["taskId"]) if "parentCallId" in slot else slots_before.get(slot["callId"])
        if slot["status"] != "running" or (previous is not None and previous["status"] == "running"):
            continue
        # A nested slot carries the arguments the call runs with; a model-issued call's are in its intent checkpoint.
        if "parentCallId" in slot:
            args = slot["arguments"]
        else:
            task = tasks.get(slot["taskId"]) if "taskId" in slot else None
            checkpoint = None if task is None else task["state"].get("checkpoint")
            args = (checkpoint or {}).get("arguments", {}) if isinstance(checkpoint, dict) else {}
        events.append({"type": "tool_execution_start", **_call_of(slot), "args": args})
    was_generation = was_live.get("generation")
    now_generation = now_live.get("generation")
    partial_before = None if was_generation is None else was_generation.get("message")
    partial = None if now_generation is None else now_generation.get("message")
    if partial is not None and partial_before is None:
        events.append({"type": "message_start", "message": _message(partial)})
    elif partial is not None and partial is not partial_before:
        message = _message(partial)
        events.append(
            {"type": "message_update", "usage": message.usage, "changes": _message_changes(view_ops, message)}
        )
    for index, slot in enumerate(slots):
        previous = slots_before.get(slot["callId"])
        if slot["status"] != "running" or previous is None or previous["status"] != "running":
            continue
        update = _tool_update(view_ops, ("tools", index), slot, previous)
        if update is not None:
            events.append({"type": "tool_execution_update", **_call_of(slot), **update})
    for index, slot in enumerate(nested):
        previous = nested_before.get(slot["taskId"])
        if slot["status"] != "running" or previous is None or previous["status"] != "running":
            continue
        update = _tool_update(view_ops, ("nestedTools", index), slot, previous)
        if update is not None:
            events.append({"type": "tool_execution_update", **_call_of(slot), **update})
    retry_now = None if now_generation is None else now_generation.get("retry")
    retry_before = None if was_generation is None else was_generation.get("retry")
    if retry_now is not None and retry_before is None:
        events.append(
            {
                "type": "auto_retry_start",
                "attempt": now_generation["attempt"],  # type: ignore[index]
                "at": retry_now["at"],
                "errorMessage": retry_now["error"],
            }
        )
    if retry_before is not None and retry_now is None:
        events.append({"type": "auto_retry_end", "attempt": was_generation["attempt"]})  # type: ignore[index]
    deferred_now = None if now_generation is None else now_generation.get("deferred")
    deferred_before = None if was_generation is None else was_generation.get("deferred")
    if deferred_now is not None and deferred_now["pollAt"] != (
        None if deferred_before is None else deferred_before["pollAt"]
    ):
        events.append({"type": "deferred_poll", "pollAt": deferred_now["pollAt"]})

    # Tools that end in this commit: a slot that becomes done, one created done (a call not offered), or an
    # unfinished one that vanishes because its run ended. A done slot that vanishes ended earlier.
    tool_ends: list[AgentEvent] = []

    def end_tool(slot: Any, entry_id: Any) -> None:
        entry = next((candidate for candidate in entries if candidate["id"] == entry_id), None)
        end: dict[str, Any] = {"type": "tool_execution_end", **_call_of(slot)}
        if entry is not None:
            end["entry"] = entry
        tool_ends.append(end)

    for previous in slots_before.values():
        if previous["status"] == "done":
            continue
        slot = next((candidate for candidate in slots if candidate["callId"] == previous["callId"]), None)
        if slot is not None and slot["status"] == "done":
            end_tool(slot, slot.get("entry"))
        # A slot whose run ended in this commit may have had its result appended with it, as for unstarted calls.
        elif slot is None:
            found = _result_of(entries, previous["callId"])
            end_tool(previous, None if found is None else found["id"])
    for slot in slots:
        if slot["status"] == "done" and slot["callId"] not in slots_before:
            end_tool(slot, slot.get("entry"))

    # Nested calls that end in this commit, the same way, children before the calls that made them: the lists hold
    # parents first, so walk them backwards.
    def end_nested(slot: Any) -> None:
        result = nested_results.get(slot["taskId"])
        end: dict[str, Any] = {"type": "tool_execution_end", **_call_of(slot)}
        if result is not None:
            end["result"] = result
        events.append(end)

    nested_now = {slot["taskId"]: slot for slot in nested}
    for previous in reversed(list(nested_before.values())):
        if previous["status"] == "done":
            continue
        slot = nested_now.get(previous["taskId"])
        if slot is None or slot["status"] == "done":
            end_nested(slot if slot is not None else previous)
    for slot in reversed(nested):
        if slot["status"] == "done" and slot["taskId"] not in nested_before:
            end_nested(slot)

    # Entries in append order; a tool's end directly precedes its result's message, as in the coding agent.
    assistant_appended = False
    for entry in entries:
        events.extend(end for end in tool_ends if end.get("entry") is entry)
        model = entry.get("model")
        message = model[0] if model else None
        if message is None:
            events.append({"type": "entry_appended", "entry": entry})
            continue
        # A streamed answer already started with its first partial.
        streamed = message.role == "assistant" and partial_before is not None and not assistant_appended
        if message.role == "assistant":
            assistant_appended = True
        if not streamed:
            events.append({"type": "message_start", "message": message})
        events.append({"type": "message_end", "entry": entry})
    # Ends without a result entry: a faulted or orphaned tool, or one whose run ended.
    events.extend(end for end in tool_ends if "entry" not in end)

    # Compaction ends, task failures, then turn and run ends.
    compactions_before = was_live.get("compactions", [])
    compactions = now_live.get("compactions", [])
    failure = now_live.get("compactionFailure")
    for status in compactions_before:
        if not any(candidate["taskId"] == status["taskId"] for candidate in compactions):
            end = {"type": "compaction_end", "taskId": status["taskId"], "reason": status["reason"]}
            if failure is not None and failure["taskId"] == status["taskId"]:
                end["error"] = failure["error"]
            events.append(end)
    # A generation's turn ends when its outcome is committed: at a ``completing`` hold or at terminal, whichever
    # comes first, so a successor created at the hold starts after it.
    turn_ended = False
    for task in tasks.values():
        status_name = task["state"]["status"]
        if task["kind"] == "pi.generation" and status_name == "completing" and task["id"] not in held:
            held.add(task["id"])
            turn_ended = True
        if status_name != "terminal":
            continue
        if task["kind"] == "pi.generation":
            if task["id"] in held:
                held.discard(task["id"])
            else:
                turn_ended = True
        outcome = task["state"]["outcome"]
        if outcome["status"] in ("faulted", "orphaned"):
            text = outcome["error"]["message"] if outcome["status"] == "faulted" else outcome["reason"]
            events.append({"type": "task_failed", "taskId": task["id"], "kind": task["kind"], "message": text})
    if turn_ended:
        events.append({"type": "turn_end"})
    run = now_live.get("run")
    run_before = was_live.get("run")
    first_now = None if run is None or len(run["inputs"]) == 0 else run["inputs"][0]
    first_before = None if run_before is None or len(run_before["inputs"]) == 0 else run_before["inputs"][0]
    run_changed = first_now != first_before
    if run_before is not None and run_changed:
        events.append({"type": "run_end", "inputs": run_before["inputs"]})

    # Submissions, document state, then what began.
    for record in submissions:
        events.append({"type": "submission", "record": record})
    if now_inbox is not was_inbox:
        events.append({"type": "inbox_update", "items": _queued(now_inbox)})
    # A retired document reads as its initial value, as in a snapshot.
    if now_agent is not was_agent:
        events.append(
            {"type": "agent_changed", "agent": now_agent if now_agent is not None else AgentDoc.definition.initial()}
        )
    if now_usage is not was_usage:
        events.append(
            {"type": "usage_changed", "usage": now_usage if now_usage is not None else UsageDoc.definition.initial()}
        )
    for status in compactions:
        if not any(candidate["taskId"] == status["taskId"] for candidate in compactions_before):
            events.append(
                {
                    "type": "compaction_start",
                    "taskId": status["taskId"],
                    "reason": status["reason"],
                    "blocking": status["blocking"],
                }
            )
    if run is not None and run_changed:
        events.append({"type": "run_start", "inputs": run["inputs"]})
    if run is not None and run["taskId"] != (None if run_before is None else run_before["taskId"]):
        started = tasks.get(run["taskId"])
        if started is not None and started["kind"] == "pi.generation":
            events.append({"type": "turn_start"})
    return events


def _nested_result(stored: Any) -> Any:
    """A stored nested result as ``execute_tool()`` returns it: its usage decoded."""
    result = dict(stored)
    if "usage" in result:
        result["usage"] = from_dict(Usage, result["usage"])
    return result


_PARTIAL_PATH: Final = ("docs", "pi.live", "generation", "message")


def _message_changes(view_ops: Sequence[Op], message: Any) -> list[MessageChange]:
    """Translate the view operations on the in-flight message into message changes (spec §9.4)."""
    changes: list[MessageChange] = []
    # A block sent whole already holds every later change to it in this batch.
    whole: set[int] = set()
    for op in view_ops:
        # View operations never replace the root.
        path: tuple[Any, ...] = tuple(op[1])  # type: ignore[arg-type]
        if not _starts_with(path, _PARTIAL_PATH):
            # The whole message or generation was replaced.
            if _starts_with(_PARTIAL_PATH, path):
                return [{"type": "message", "message": message}]
            continue
        rest = path[len(_PARTIAL_PATH) :]
        if len(rest) > 0 and rest[0] == "usage":
            continue
        if len(rest) == 0 or rest[0] != "content":
            return [{"type": "message", "message": message}]
        if len(rest) == 1:
            if op[0] != "p" or op[3] != 0:  # type: ignore[misc]
                return [{"type": "message", "message": message}]
            for offset, _block in enumerate(op[4]):  # type: ignore[misc]
                content_index = op[2] + offset  # type: ignore[operator]
                block = message.content[content_index]
                kind = (
                    "text_start"
                    if block.type == "text"
                    else "thinking_start"
                    if block.type == "thinking"
                    else "toolcall_start"
                )
                changes.append({"type": kind, "contentIndex": content_index, "block": block})
            continue
        content_index = rest[1]
        field = rest[2] if len(rest) > 2 else None
        if content_index in whole:
            continue
        if op[0] == "a" and len(rest) == 3 and field in ("text", "thinking"):
            changes.append(
                {
                    "type": "text_delta" if field == "text" else "thinking_delta",
                    "contentIndex": content_index,
                    "delta": op[2],  # type: ignore[misc]
                }
            )
        elif op[0] == "a" and field == "arguments":
            changes.append(
                {"type": "toolcall_delta", "contentIndex": content_index, "path": list(rest[3:]), "delta": op[2]}  # type: ignore[misc]
            )
        else:
            whole.add(content_index)
            changes.append({"type": "block", "contentIndex": content_index, "block": message.content[content_index]})
    return changes


def _call_of(slot: Any) -> ToolEventCall:
    """The identifying fields of a slot's tool events."""
    call: dict[str, Any] = {"toolCallId": slot["callId"], "toolName": slot["name"]}
    if "taskId" in slot:
        call["taskId"] = slot["taskId"]
    if "parentCallId" in slot:
        call["parentToolCallId"] = slot["parentCallId"]
        call["parentTaskId"] = slot["parentTaskId"]
    return call


def _tool_update(view_ops: Sequence[Op], at: tuple[str, int], slot: Any, previous: Any) -> dict[str, Any] | None:
    """Output, details, and diagnostics changes of a running slot at ``at`` in ``pi.live``, from the view ops on it."""
    output_path = ("docs", "pi.live", *at, "output")
    trim_start = 0
    append = ""
    replaced = False
    for op in view_ops:
        if not _starts_with(tuple(op[1]), output_path):  # type: ignore[arg-type]
            continue
        if op[0] == "t":
            trim_start += op[2]  # type: ignore[misc]
        elif op[0] == "a":
            append += op[2]  # type: ignore[misc]
        else:
            replaced = True
    output: dict[str, Any] | None = None
    if replaced or (slot.get("output") != previous.get("output") and trim_start == 0 and append == ""):
        output = {"set": slot.get("output", "")}
    elif trim_start > 0 or append != "":
        output = {}
        if trim_start > 0:
            output["trimStart"] = trim_start
        if append != "":
            output["append"] = append
    update: dict[str, Any] = {}
    if output is not None:
        update["output"] = output
    # A safe replay clears a running slot's progress: removed details send ``None``, removed diagnostics ``[]``.
    if not js_same(slot.get("details"), previous.get("details")):
        update["details"] = slot.get("details")
    if not js_same(slot.get("diagnostics"), previous.get("diagnostics")):
        update["diagnostics"] = slot.get("diagnostics", [])
    return None if len(update) == 0 else update


def _starts_with(path: Sequence[Any], prefix: Sequence[Any]) -> bool:
    return len(prefix) <= len(path) and all(path[index] == segment for index, segment in enumerate(prefix))
