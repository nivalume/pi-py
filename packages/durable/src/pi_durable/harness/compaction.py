"""The built-in compaction task (port of ``harness/compaction.ts``, spec §8.7)."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict

from pi_ai.types import (
    AssistantMessage,
    Message,
    ModelThinkingLevel,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.estimate import calculate_context_tokens, estimate_message_tokens
from pi_ai.utils.retry import is_retryable_assistant_error
from pi_chord.context import Context

from ..entries import CompactionEntry
from ..tasks import define_task
from ..types import ConversationId, EntryId, Task, TaskId, Tx
from .live import (
    LiveDoc,
    add_compaction_status,
    compaction_status,
    fail_compaction_status,
    remove_compaction_status,
)
from .provider import ensure_provider_session_id
from .stream import retry_delay, stream_options
from .types import (
    CompactionHooks,
    CompactionPolicy,
    CompactionReason,
    CompactionResult,
    ContextView,
    ConversationStreamOptions,
    ModelRef,
    Settings,
)
from .usage import record_usage

__all__ = [
    "CompactionCheckpoint",
    "CompactionInput",
    "CompactionTask",
    "SummaryRequest",
    "compaction_policy",
    "create_compaction",
    "estimate_context",
    "select_cut",
    "serialize_conversation",
    "summarized_messages",
]


class CompactionInput(TypedDict):
    reason: CompactionReason
    instructions: NotRequired[str]


class SummaryRequest(TypedDict):
    """The pinned summarization request."""

    attempt: int
    model: ModelRef
    thinkingLevel: ModelThinkingLevel
    streamOptions: ConversationStreamOptions
    maxTokens: float
    #: Newest entry of the context the range was selected from.
    tail: EntryId
    #: First entry kept verbatim; the summary's ``head``.
    firstKept: EntryId


class CompactionCheckpointSelect(TypedDict):
    phase: Literal["select"]


class CompactionCheckpointSummarize(SummaryRequest):
    phase: Literal["summarize"]


class CompactionCheckpointRetry(SummaryRequest):
    phase: Literal["retry"]
    until: float


CompactionCheckpoint: TypeAlias = CompactionCheckpointSelect | CompactionCheckpointSummarize | CompactionCheckpointRetry

#: Longest tool result text a serialized summary source keeps.
_TOOL_RESULT_MAX_CHARS: Final = 2000

_SUMMARY_PREFIX: Final = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
_SUMMARY_SUFFIX: Final = "\n</summary>"

_SUMMARIZATION_SYSTEM_PROMPT: Final = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

_SUMMARIZATION_PROMPT: Final = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier summary, preserve its information and fold the newer messages into it.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


def _terminal_completed(result: Any) -> Any:
    return {"status": "terminal", "outcome": {"status": "completed", "result": result}}


async def _select(task: Any, runtime: Any, context: Context) -> None:
    conversation_id = runtime.conversation_id
    agent = await runtime.agent(context)
    settings = runtime.settings
    ref = agent.get("model")
    model = None if ref is None else runtime.models.get_model(ref["provider"], ref["modelId"])
    if ref is None or model is None:
        return await _fail_no_model(runtime, ref, context)
    policy = compaction_policy(settings, ref)
    view = await runtime.context(conversation_id, context)
    cut = select_cut(view, policy["keepRecentTokens"])
    if cut is None:
        return await _complete(runtime, context)
    first_kept = view["entries"][cut]["id"]
    task_input = task["input"]
    compaction: dict[str, Any] = {
        "reason": task_input["reason"],
        "entries": view["entries"][:cut],
        "messages": summarized_messages(view, cut),
        "firstKept": first_kept,
    }
    if "instructions" in task_input:
        compaction["instructions"] = task_input["instructions"]
    decision: list[Any] = [None]

    async def invoke(hook: Any) -> None:
        if decision[0] is None:
            result = hook(compaction, runtime, context)
            decision[0] = await result if inspect.isawaitable(result) else result

    await runtime.hooks.each("before_compact", invoke)
    decided = decision[0]
    if decided is not None and "decline" in decided:
        return await _complete(runtime, context)
    if decided is not None:
        return await _place(runtime, first_kept, decided["summary"], context)
    tail = first_kept
    for entry in view["entries"]:
        if entry["id"] > tail:
            tail = entry["id"]
    request: SummaryRequest = {
        "attempt": 1,
        "model": ref,
        "thinkingLevel": agent["thinkingLevel"],
        "streamOptions": settings["stream"],
        "maxTokens": min(
            math.floor(0.8 * policy["reserveTokens"]), model.max_tokens if model.max_tokens > 0 else math.inf
        ),
        "tail": tail,
        "firstKept": first_kept,
    }
    await runtime.commit(
        lambda _tx, _current: {"status": "running", "checkpoint": {"phase": "summarize", **request}}, context
    )


async def _summarize(task: Any, runtime: Any, context: Context) -> None:
    checkpoint = task["state"]["checkpoint"]
    request = {key: value for key, value in checkpoint.items() if key != "phase"}
    ref = request["model"]
    attempt = request["attempt"]
    first_kept = request["firstKept"]
    model = runtime.models.get_model(ref["provider"], ref["modelId"])
    if model is None:
        return await _fail_no_model(runtime, ref, context)
    # The context at ``tail`` is immutable, so this is the range ``select`` chose.
    view = await runtime.context(runtime.conversation_id, context, at=request["tail"])
    cut = next((index for index, entry in enumerate(view["entries"]) if entry["id"] == first_kept), -1)
    now = runtime.now()
    messages: list[Message] = [
        SystemMessage(_SUMMARIZATION_SYSTEM_PROMPT, timestamp=now),
        UserMessage(
            [TextContent(_summary_prompt(summarized_messages(view, cut), task["input"].get("instructions")))],
            timestamp=now,
        ),
    ]
    forwarded = {key: value for key, value in request["streamOptions"].items() if key != "deferred"}
    options = stream_options(forwarded)
    options["cache_retention"] = "none"
    max_tokens = request["maxTokens"]
    if max_tokens != math.inf:
        options["max_tokens"] = int(max_tokens)
    options["signal"] = runtime.signal
    options["session_id"] = await ensure_provider_session_id(runtime, context)
    thinking = request["thinkingLevel"]
    if thinking != "off":
        options["reasoning"] = thinking
    message: AssistantMessage = await runtime.models.complete_simple(model, {"messages": messages}, options)
    # An abort mark or close: the abort invocation or the reopened task handles the committed state.
    runtime.signal.throw_if_aborted()
    summary = _summary_text(message)
    policy = runtime.settings["retry"]
    retry = (
        message.stop_reason == "error"
        and is_retryable_assistant_error(message)
        and policy["enabled"]
        and attempt <= policy["maxRetries"]
    )
    until = runtime.now() + retry_delay(policy, attempt) if retry else 0

    async def change(tx: Tx, current: Any) -> Any:
        await record_usage(tx, runtime.conversation_id, "models", f"{message.provider}/{message.model}", message.usage)
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        if summary is not None:
            return await _place_summary(tx, runtime, current, live, first_kept, summary)
        if retry:
            status = compaction_status(live, runtime.task_id)
            if status is not None:
                status["retry"] = {"at": until, "error": message.error_message or ""}
            return {"status": "sleeping", "checkpoint": {"phase": "retry", **request, "until": until}, "until": until}
        text = _summary_failure(message)
        fail_compaction_status(live, runtime.task_id, text)
        return {
            "status": "terminal",
            "outcome": {"status": "failed", "error": {"message": text, "detail": {"reason": "model_error"}}},
        }

    await runtime.commit(change, context)


async def _retry(task: Any, runtime: Any, context: Context) -> None:
    checkpoint = task["state"]["checkpoint"]
    until = checkpoint["until"]
    if until > runtime.now():
        return await runtime.sleep_until(until, checkpoint, context)
    request = {key: value for key, value in checkpoint.items() if key not in ("phase", "until")}
    attempt = request["attempt"] + 1

    async def change(tx: Tx, _current: Any) -> Any:
        status = compaction_status(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
        if status is not None:
            status["attempt"] = attempt
            status.pop("retry", None)
        return {"status": "running", "checkpoint": {"phase": "summarize", **request, "attempt": attempt}}

    await runtime.commit(change, context)


async def _abort(_task: Any, runtime: Any, context: Context) -> None:
    async def change(tx: Tx, _current: Any) -> Any:
        remove_compaction_status(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
        return {"status": "terminal", "outcome": {"status": "aborted"}}

    await runtime.commit(change, context)


#: Built-in compaction task (spec §8.7): select an old prefix of the model context, summarize it, and place a
#: summary entry whose ``head`` is the first kept entry. A compaction the generation owns blocks it and appends
#: directly; a conversation-owned one places its summary through a write submission.
CompactionTask: Task[CompactionInput, CompactionCheckpoint, CompactionResult, CompactionHooks] = define_task(
    name="pi.compaction",
    version=1,
    initial=lambda _input: {"phase": "select"},
    phases={"select": _select, "summarize": _summarize, "retry": _retry},
    abort=_abort,
)


async def create_compaction(
    tx: Tx, conversation_id: ConversationId, input: CompactionInput, owner: TaskId[Any] | None = None
) -> TaskId[CompactionResult]:
    """Create a compaction task with its status in this commit.

    ``owner`` is the generation that waits for it (a blocking compaction); without one it is
    conversation-owned, and ``background`` unless it is manual.
    """
    ownership: Any = {"kind": "conversation"} if owner is None else {"kind": "task", "taskId": owner}
    background = owner is None and input["reason"] != "manual"
    task_id = await tx.create_task(
        CompactionTask, input, {"ownership": ownership, "conversationId": conversation_id, "background": background}
    )
    status: Any = {"taskId": task_id, "reason": input["reason"], "blocking": owner is not None, "attempt": 1}
    add_compaction_status(await tx.doc(LiveDoc, conversation_id), status)
    return task_id


def compaction_policy(settings: Settings, model: ModelRef) -> CompactionPolicy:
    """The compaction policy for ``model``: its ``modelOverrides`` entry over the harness-wide policy, field by field."""
    compaction: Any = settings["compaction"]
    overrides = compaction.get("modelOverrides") or {}
    override = overrides.get(f"{model['provider']}/{model['modelId']}") or {}

    def pick(key: str) -> Any:
        value = override.get(key)
        return compaction[key] if value is None else value

    return {
        "enabled": pick("enabled"),
        "reserveTokens": pick("reserveTokens"),
        "keepRecentTokens": pick("keepRecentTokens"),
        "backgroundTokens": pick("backgroundTokens"),
    }


def select_cut(view: ContextView, keep_recent_tokens: float) -> int | None:
    """Index in ``view["entries"]`` of the first entry a summary keeps, or ``None`` when there is nothing to compact.

    Walks back from the tail until ``keep_recent_tokens`` are kept, then cuts at the first candidate at or
    after that entry: an entry whose contribution starts with a user or assistant message, never a tool result,
    and never a user entry that a result of the preceding assistant's calls still follows.
    """
    contributions = view["contributions"]
    start = 0 if view["head"] is None else 1
    candidates = [index for index in range(start, len(contributions)) if _is_candidate(contributions, index)]
    kept = 0
    cut: int | None = None
    for index in range(len(contributions) - 1, start - 1, -1):
        for message in contributions[index]:
            kept += estimate_message_tokens(message)
        if kept < keep_recent_tokens:
            continue
        cut = next(
            (candidate for candidate in candidates if candidate >= index), candidates[-1] if candidates else None
        )
        break
    if cut is None:
        return None
    for index in range(start, cut):
        if len(contributions[index]) > 0:
            return cut
    return None


def _is_candidate(contributions: Sequence[Sequence[Message]], index: int) -> bool:
    messages = contributions[index]
    first = messages[0] if len(messages) > 0 else None
    if isinstance(first, AssistantMessage):
        return True
    if not isinstance(first, UserMessage):
        return False
    # A result of the preceding assistant's calls that follows this entry, before the next assistant, belongs before it.
    calls: set[str] = set()
    for before in range(index - 1, -1, -1):
        assistant = next(
            (message for message in reversed(contributions[before]) if isinstance(message, AssistantMessage)), None
        )
        if assistant is None:
            continue
        calls = {content.id for content in assistant.content if isinstance(content, ToolCall)}
        break
    if len(calls) == 0:
        return True
    for after in range(index, len(contributions)):
        for position, message in enumerate(contributions[after]):
            if isinstance(message, AssistantMessage) and (after > index or position > 0):
                return True
            if isinstance(message, ToolResultMessage) and message.tool_call_id in calls:
                return False
    return True


def summarized_messages(view: ContextView, cut: int) -> list[Message]:
    """Model messages of the entries before ``cut``: the head marker first, ordered like model context (spec §2.1)."""
    from .context import order_tool_results

    selected = view["contributions"][:cut]
    return order_tool_results([message for messages in selected for message in messages])


def estimate_context(view: ContextView, extra: Sequence[Message]) -> float:
    """Size of a request over ``view`` followed by ``extra`` (spec §8.3).

    The usage of the newest assistant appended after the head marker, whose request included the marker, plus
    estimates of the messages after it; without one, estimates of every message.
    """
    measured: AssistantMessage | None = None
    head = view["head"]
    after = -math.inf if head is None else head["id"]
    entries = view["entries"]
    for index in range(len(entries) - 1, -1, -1):
        if measured is not None:
            break
        if entries[index]["id"] <= after:
            continue
        measured = next(
            (
                message
                for message in reversed(view["contributions"][index])
                if isinstance(message, AssistantMessage) and calculate_context_tokens(message.usage) > 0
            ),
            None,
        )
    messages = view["messages"]
    start = 0
    if measured is not None:
        last = -1
        for index in range(len(messages) - 1, -1, -1):
            if messages[index] is measured:
                last = index
                break
        start = last + 1
    tokens: float = 0 if measured is None else calculate_context_tokens(measured.usage)
    for message in messages[start:]:
        tokens += estimate_message_tokens(message)
    for message in extra:
        tokens += estimate_message_tokens(message)
    return tokens


def _summary_text(message: AssistantMessage) -> str | None:
    """The summary of a clean ``stop`` with text and no tool call; anything else is not a summary."""
    if message.stop_reason != "stop" or any(isinstance(content, ToolCall) for content in message.content):
        return None
    text = "\n".join(content.text for content in message.content if isinstance(content, TextContent)).strip()
    return None if len(text) == 0 else text


def _summary_failure(message: AssistantMessage) -> str:
    if message.stop_reason in ("error", "aborted"):
        return f"Summarization failed: {message.error_message or message.stop_reason}"
    if message.stop_reason == "length":
        return "Summarization hit the token limit; the summary is incomplete"
    if any(isinstance(content, ToolCall) for content in message.content):
        return "Summarization attempted to call a tool"
    return "Summarization produced no text"


def _summary_prompt(messages: Sequence[Message], instructions: str | None) -> str:
    """The summarizer's user message: the serialized conversation, the prompt, and any instructions."""
    focus = "" if instructions is None else f"\n\nAdditional focus: {instructions}"
    return f"<conversation>\n{serialize_conversation(messages)}\n</conversation>\n\n{_SUMMARIZATION_PROMPT}{focus}"


def _js_stringify(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def serialize_conversation(messages: Sequence[Message]) -> str:
    """Messages as plain text, so the summarizer reads a transcript instead of continuing it.

    System messages and thinking are omitted; visible text, tool calls, and tool results record what happened.
    """
    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            text = _content_text(message.content)
            if len(text) > 0:
                parts.append(f"[User]: {text}")
        elif isinstance(message, AssistantMessage):
            texts = [content.text for content in message.content if isinstance(content, TextContent)]
            calls = [
                f"{content.name}({', '.join(f'{key}={_js_stringify(value)}' for key, value in content.arguments.items())})"
                for content in message.content
                if isinstance(content, ToolCall)
            ]
            if len(texts) > 0:
                parts.append(f"[Assistant]: {chr(10).join(texts)}")
            if len(calls) > 0:
                parts.append(f"[Assistant tool calls]: {'; '.join(calls)}")
        elif isinstance(message, ToolResultMessage):
            text = _content_text(message.content)
            if len(text) > 0:
                parts.append(f"[Tool result]: {_truncate(text, _TOOL_RESULT_MAX_CHARS)}")
    return "\n\n".join(parts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content if isinstance(block, TextContent))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"


async def _place(runtime: Any, first_kept: EntryId, summary: str, context: Context) -> None:
    """Place a summary supplied by a hook in its own commit."""

    async def change(tx: Tx, current: Any) -> Any:
        live = await tx.doc(LiveDoc, runtime.conversation_id)
        return await _place_summary(tx, runtime, current, live, first_kept, summary)

    await runtime.commit(change, context)


async def _place_summary(tx: Tx, runtime: Any, current: Any, live: Any, first_kept: EntryId, summary: str) -> Any:
    """Place the summary entry and complete (spec §8.7).

    A blocking compaction, owned by its generation, appends it: the generation holds the run and waits. A
    conversation-owned one admits it as a write submission, placed at once when idle, otherwise at the next
    boundary, or settled ``stale``.
    REMINDER: nothing else may append to a busy conversation, so every non-blocking summary goes through admission.
    """
    from .submissions import admit_submission

    remove_compaction_status(live, runtime.task_id)
    live.pop("compactionFailure", None)
    text = f"{_SUMMARY_PREFIX}{summary}{_SUMMARY_SUFFIX}"
    entry: Any = {
        "kind": CompactionEntry.kind,
        "head": first_kept,
        "model": [UserMessage([TextContent(text)], timestamp=runtime.now())],
        "data": {"reason": current["input"]["reason"]},
    }
    if "owner" not in current:
        submission_id = await admit_submission(
            tx,
            runtime.conversation_id,
            {"type": "write", "requestId": f"compaction:{runtime.task_id}", "entry": entry},
            runtime.now(),
            runtime.settings,
        )
        result: CompactionResult = {"submissionId": submission_id}
    else:
        result = {"entryId": (await tx.append_entry(runtime.conversation_id, entry))["id"]}
    return _terminal_completed(result)


async def _complete(runtime: Any, context: Context) -> None:
    """Remove the status and complete without a summary."""

    async def change(tx: Tx, _current: Any) -> Any:
        remove_compaction_status(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id)
        return _terminal_completed({})

    await runtime.commit(change, context)


async def _fail_no_model(runtime: Any, ref: ModelRef | None, context: Context) -> None:
    message = "No model is configured" if ref is None else f"Model {ref['provider']}/{ref['modelId']} is not available"

    async def change(tx: Tx, _current: Any) -> Any:
        fail_compaction_status(await tx.doc(LiveDoc, runtime.conversation_id), runtime.task_id, message)
        return {
            "status": "terminal",
            "outcome": {"status": "failed", "error": {"message": message, "detail": {"reason": "no_model"}}},
        }

    await runtime.commit(change, context)
