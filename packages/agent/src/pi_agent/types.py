"""Types for the agent runtime (mirrors ``@earendil-works/pi-agent-core`` ``src/types.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeAlias

from pi_ai import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    ImageContent,
    Message,
    Model,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "StreamFn",
    "ToolExecutionMode",
    "QueueMode",
    "ThinkingLevel",
    "AgentToolCall",
    "AgentMessage",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "AgentTurnContext",
    "AgentTurnDecision",
    "FinishTurn",
    "AgentLoopTurnUpdate",
    "PrepareRequestContext",
    "AgentRequestUpdate",
    "PrepareRequest",
    "PrepareNextTurnContext",
    "AgentLoopConfig",
    "AgentToolResult",
    "AgentToolCallOutcome",
    "AgentToolUpdateCallback",
    "AgentTool",
    "AgentContext",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "AgentEvent",
]

#: Stream function used by the agent loop (``models.stream_simple`` satisfies it).
#:
#: It receives a normalized transcript (prompt and tools live in system messages) and
#: must encode failures in the returned stream instead of raising.
StreamFn: TypeAlias = Callable[
    [Model, TranscriptContext, "Mapping[str, Any] | None"],
    "AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]",
]

#: ``sequential``: each tool call is prepared, executed and finalized before the next starts.
#: ``parallel``: tool calls are prepared sequentially, then allowed tools execute concurrently.
ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]

#: How many queued messages are injected at a drain point: every message, or only the oldest.
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]

ThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

AgentToolCall: TypeAlias = ToolCall

#: Any message in an agent transcript: the LLM messages plus app-defined custom
#: messages (any object with a ``role`` attribute, e.g. a dataclass).
AgentMessage: TypeAlias = Any


@dataclass(slots=True)
class BeforeToolCallResult:
    """Returned from ``before_tool_call``. ``block=True`` prevents execution with an error result."""

    block: bool = False
    #: Text of the error result when blocked. Defaults to "Tool execution was blocked".
    reason: str | None = None
    #: Hint to stop after the current tool batch when blocked (only when every result in the batch terminates).
    terminate: bool | None = None


@dataclass(slots=True)
class AfterToolCallResult:
    """Partial override returned from ``after_tool_call``; omitted (``None``) fields keep their values.

    ``content`` without ``structured_content`` drops the original structured content, because it
    may no longer match. No deep merge is performed.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    structured_content: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    #: Validated tool arguments.
    args: Any
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    #: The executed tool result before overrides are applied.
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass(slots=True)
class AgentTurnContext:
    """Context passed to completed-turn callbacks."""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    #: Agent context after the turn's assistant message and tool results were appended.
    context: AgentContext
    #: Messages this loop invocation returns if it exits now.
    new_messages: list[Any]


PrepareNextTurnContext = AgentTurnContext


@dataclass(slots=True)
class AgentTurnDecision:
    """Decision returned by ``finish_turn``: ``"continue"`` or ``"end"``.

    ``finish_turn`` may also return the plain strings or ``{"action": ...}`` dicts.
    """

    action: Literal["continue", "end"]


FinishTurn: TypeAlias = Callable[[AgentTurnContext, "AbortSignal | None"], Any]


@dataclass(slots=True)
class AgentLoopTurnUpdate:
    """Replacement runtime state used before starting another provider request."""

    context: AgentContext | None = None
    #: Messages to append before the next provider request, with normal lifecycle events.
    messages: list[Any] | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True)
class PrepareRequestContext:
    context: AgentContext
    model: Model
    thinking_level: ThinkingLevel


@dataclass(slots=True)
class AgentRequestUpdate:
    """Replacement runtime state for the provider request being prepared."""

    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


PrepareRequest: TypeAlias = Callable[[PrepareRequestContext, "AbortSignal | None"], Any]


@dataclass(slots=True, kw_only=True)
class AgentLoopConfig:
    """Configuration of the low-level agent loop.

    ``stream_options`` holds the ``SimpleStreamOptions`` forwarded to the stream function
    (``session_id``, ``transport``, ``thinking_budgets``, ``on_payload``, ``temperature``...).
    """

    model: Model
    #: Convert agent messages to LLM messages before each call. Must not raise.
    convert_to_llm: Callable[[list[Any]], list[Message] | Awaitable[list[Message]]]
    #: Requested reasoning level (``None`` means off).
    reasoning: ThinkingLevel | None = None
    api_key: str | None = None
    stream_options: dict[str, Any] = field(default_factory=dict)
    #: Optional transform applied before ``convert_to_llm`` (pruning, context injection). Must not raise.
    transform_context: Callable[[list[Any], AbortSignal | None], Awaitable[list[Any]]] | None = None
    #: Resolve an API key per call (short-lived OAuth tokens). Must not raise.
    get_api_key: Callable[[str], str | Awaitable[str | None] | None] | None = None
    finish_turn: FinishTurn | None = None
    prepare_request: PrepareRequest | None = None
    prepare_next_turn: Callable[[AgentTurnContext], Any] | None = None
    get_steering_messages: Callable[[], Awaitable[list[Any]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[Any]]] | None = None
    #: Default ``"parallel"``.
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: (
        Callable[[BeforeToolCallContext, AbortSignal | None], Awaitable[BeforeToolCallResult | None]] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext, AbortSignal | None], Awaitable[AfterToolCallResult | None]] | None
    ) = None


@dataclass(slots=True)
class AgentToolResult:
    """Final or partial result produced by a tool."""

    #: Text or image content returned to the model.
    content: list[TextContent | ImageContent] = field(default_factory=list)
    #: Arbitrary structured details for logs or UI rendering.
    details: Any = None
    _: KW_ONLY
    #: Machine-readable result matching the tool's ``output_schema``; not sent to the model.
    structured_content: Any = None
    usage: Usage | None = None
    #: Report a failure without raising; ``details``/``structured_content`` are kept.
    is_error: bool | None = None
    #: Hint to stop after the current tool batch (only when every result in the batch terminates).
    terminate: bool | None = None


@dataclass(slots=True)
class AgentToolCallOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
    #: Milliseconds ``execute()`` took; ``None`` when the tool did not run.
    duration_ms: int | None = None


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], Any]


class _Execute(Protocol):
    def __call__(
        self,
        tool_call_id: str,
        params: Any,
        signal: AbortSignal | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> Awaitable[AgentToolResult]: ...


@dataclass(slots=True, kw_only=True)
class AgentTool(Tool):
    """A tool the agent can execute.

    ``execute(tool_call_id, params, signal, on_update)`` receives validated parameters
    (a dict, or a model instance for Pydantic parameters). Raise on failure, or return a
    result with ``is_error=True``; do not only describe the failure in ``content``.
    """

    execute: _Execute
    #: Human-readable label for UI display (defaults to ``name``).
    label: str = ""
    #: Compatibility shim for raw arguments before schema validation.
    prepare_arguments: Callable[[Any], Any] | None = None
    #: JSON Schema of ``structured_content`` in successful results.
    output_schema: Any = None
    #: Recovery policy for an effect whose durable intent exists but whose outcome is unknown.
    replay: Literal["never", "safe"] | None = None
    #: Per-tool execution mode override; ``"sequential"`` forces the whole batch to run sequentially.
    execution_mode: ToolExecutionMode | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass(slots=True)
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    messages: list[Any] = field(default_factory=list)
    tools: list[AgentTool] | None = None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentStartEvent:
    type: ClassVar[Literal["agent_start"]] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    """Last event of a run. Awaited ``Agent.subscribe`` listeners still count toward settlement."""

    type: ClassVar[Literal["agent_end"]] = "agent_end"
    messages: list[Any]


@dataclass(slots=True)
class TurnStartEvent:
    type: ClassVar[Literal["turn_start"]] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    type: ClassVar[Literal["turn_end"]] = "turn_end"
    message: Any
    tool_results: list[ToolResultMessage]


@dataclass(slots=True)
class MessageStartEvent:
    type: ClassVar[Literal["message_start"]] = "message_start"
    message: Any


@dataclass(slots=True)
class MessageUpdateEvent:
    """Only emitted for assistant messages during streaming."""

    type: ClassVar[Literal["message_update"]] = "message_update"
    message: Any
    assistant_message_event: AssistantMessageEvent


@dataclass(slots=True)
class MessageEndEvent:
    type: ClassVar[Literal["message_end"]] = "message_end"
    message: Any


@dataclass(slots=True)
class ToolExecutionStartEvent:
    type: ClassVar[Literal["tool_execution_start"]] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    type: ClassVar[Literal["tool_execution_update"]] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


@dataclass(slots=True)
class ToolExecutionEndEvent:
    type: ClassVar[Literal["tool_execution_end"]] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    #: Milliseconds ``execute()`` took; ``None`` when the tool did not run.
    duration_ms: int | None = None


AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
