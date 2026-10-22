"""Harness types: extensions, tools, prompt sections, hooks, settings, and the Harness surface.

Port of ``harness/types.ts`` (types only; the Harness implementation lives in the other
``pi_durable.harness`` modules). The mapping rules of :mod:`pi_durable.types` apply:

- Plain data (``AgentState``, ``Settings``, ``ToolExecutionResult``, ``ToolDiagnostic``, option
  objects) are ``TypedDict`` s with upstream's keys; ``AgentState`` and the other stored values keep
  their camelCase keys on the wire.
- What an application defines (``Extension``, ``ToolRegistration``, ``PromptSection``, ``Wrap``,
  ``HookRegistration``) are slotted dataclasses with snake_case attributes, built by the helpers of
  :mod:`pi_durable.harness.define`.
- Hook tables (``GenerationHooks``, ``ToolHooks``, ``CompactionHooks``, a task's ``H``) are
  ``TypedDict(total=False)`` s of handlers keyed by the snake_case hook name: these are method
  tables, not data, so ``HookRunner.each("before_tool", ...)`` names them like methods.
- Interfaces with behavior (``Conversation``, ``Harness``, ``ToolExecutionApi``, ``RegistrySnapshot``)
  are ``Protocol`` s with snake_case members.

``ConversationView``, ``TaskGraph``, ``TaskGraphWatch`` and ``UsageState`` are defined by
:mod:`pi_durable.harness.view`, :mod:`pi_durable.harness.task_graph` and :mod:`pi_durable.harness.usage`
and re-exported here for type checkers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict, TypeVar, overload

from pi_ai.types import (
    AssistantMessage,
    CacheRetention,
    ImageContent,
    Message,
    ModelThinkingLevel,
    TextContent,
    Tool,
    ToolCall,
    Transport,
    Usage,
)
from pi_chord.context import Context
from pi_chord.json import JsonObject
from pi_chord.state import AttachedReplicatedState

from ..types import (
    AnyTask,
    AnyTaskRecord,
    ConversationId,
    ConversationOwnership,
    ConversationRecord,
    Cursor,
    DocumentObserver,
    DocumentReader,
    EntryDraft,
    EntryId,
    EntryRecord,
    Page,
    ReportInfo,
    ScanOrder,
    Session,
    SetTimer,
    SubmissionId,
    SubmissionRecord,
    SubmissionRecordInputDone,
    SubmissionRecordInputUnanswered,
    SubmissionRecordWriteDone,
    SubmissionRecordWriteUnanswered,
    Task,
    TaskId,
    TaskOptions,
    TaskRecord,
    TaskStateTerminal,
    Tx,
    WatchHandle,
)

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

    from pi_ai.models import Models
    from pi_telemetry import TelemetryContext

    from ..approval import (
        ApprovalPolicy,
        ApprovalRequest,
        PermissionMode,
        PermissionRules,
        PermissionsChange,
        PermissionsState,
        PermissionSubject,
        ReviewDecision,
    )
    from ..env import ExecutionEnv, ShellOutputSkip, ShellOutputWindow
    from .task_graph import TaskGraph, TaskGraphWatch
    from .usage import UsageState
    from .view import ConversationView

__all__ = [
    "ModelRef",
    "UserInput",
    "InputEntry",
    "SubmissionDraft",
    "SubmissionDraftInput",
    "SubmissionDraftWrite",
    "InputSubmissionDraft",
    "SettledSubmissionRecord",
    "Submission",
    "SettledTask",
    "AnyTask",
    "ConversationHandle",
    "ToolControl",
    "ToolDiagnostic",
    "ToolExecutionResult",
    "NestedToolExecutionResult",
    "ToolExecutionMode",
    "QueueMode",
    "RetainedOutput",
    "ToolExecutionApi",
    "ToolOutputLimits",
    "ToolRegistration",
    "PromptInput",
    "PromptSection",
    "HookRegistration",
    "ToolWrap",
    "SectionWrap",
    "Wrap",
    "Extension",
    "RegistryTool",
    "RegistrySection",
    "RegistrySnapshot",
    "RegistryReader",
    "Registry",
    "ExtensionSelectionEdit",
    "ToolSelectionEdit",
    "ModelToolsEdit",
    "AgentState",
    "ExtensionChangeEdit",
    "ToolChangeEdit",
    "ModelToolsChangeEdit",
    "AgentChange",
    "Agent",
    "ConversationInit",
    "ConversationCreateOptions",
    "DeferredStreamOptions",
    "ConversationStreamOptions",
    "ConversationRetryPolicy",
    "PartialConversationRetryPolicy",
    "CompactionPolicy",
    "PartialCompactionPolicy",
    "CompactionSettings",
    "ProgressPolicy",
    "PartialProgressPolicy",
    "CompactionReason",
    "CompactionResult",
    "HarnessSettings",
    "AfterFailedRun",
    "Settings",
    "ResolvedCompactionPolicy",
    "EnvTarget",
    "EnvBuilder",
    "HarnessOptions",
    "Wake",
    "TaskInspection",
    "TaskInspectionState",
    "TaskInspectionRunning",
    "TaskInspectionReady",
    "TaskInspectionWaiting",
    "TaskInspectionSleeping",
    "TaskInspectionCompleting",
    "TaskInspectionBlocked",
    "HarnessInspection",
    "ContextView",
    "Conversation",
    "ConversationView",
    "ConversationWatch",
    "TaskGraph",
    "TaskGraphWatch",
    "UsageState",
    "Harness",
    "HookApi",
    "HookResult",
    "GenerationRequest",
    "GenerationContinue",
    "RunInput",
    "InputBlock",
    "InputContext",
    "InputDecision",
    "GenerationHooks",
    "ToolCallParent",
    "ToolHookCall",
    "BeforeToolResult",
    "ToolHooks",
    "CompactionHookInput",
    "CompactionDecline",
    "CompactionSummary",
    "CompactionHooks",
]

T = TypeVar("T")
R = TypeVar("R")


class ModelRef(TypedDict):
    """Provider and model ID resolved through pi-ai ``Models``."""

    provider: ReadOnly[str]
    modelId: ReadOnly[str]


#: ``UserMessage["content"]``.
UserInput: TypeAlias = "str | Sequence[TextContent | ImageContent]"


class InputEntry(TypedDict):
    """The entry an input is placed as instead of ``pi.user``: an application kind, and data the transcript shows."""

    kind: ReadOnly[str]
    data: ReadOnly[NotRequired[Any]]


class SubmissionDraftInput(TypedDict):
    """User input that may start a run."""

    requestId: ReadOnly[NotRequired[str]]
    type: ReadOnly[Literal["input"]]
    content: ReadOnly[UserInput]
    whenBusy: ReadOnly[NotRequired[Literal["steer", "followUp", "reject"]]]
    #: Place the input as this entry; default ``pi.user`` without data. The kind must not start with ``pi.``.
    entry: ReadOnly[NotRequired[InputEntry]]


class SubmissionDraftWrite(TypedDict):
    """A passive entry write."""

    requestId: ReadOnly[NotRequired[str]]
    type: ReadOnly[Literal["write"]]
    entry: ReadOnly[EntryDraft]


#: Host submission: user input that may start a run, or a passive entry write.
SubmissionDraft: TypeAlias = SubmissionDraftInput | SubmissionDraftWrite
InputSubmissionDraft: TypeAlias = SubmissionDraftInput
SettledSubmissionRecord: TypeAlias = (
    SubmissionRecordInputDone
    | SubmissionRecordInputUnanswered
    | SubmissionRecordWriteDone
    | SubmissionRecordWriteUnanswered
)


class Submission(Protocol):
    """Awaitable host object for one durably admitted submission."""

    @property
    def id(self) -> SubmissionId: ...
    def status(self, context: Context) -> Awaitable[SubmissionRecord]: ...
    def wait(self, context: Context) -> Awaitable[SettledSubmissionRecord]: ...
    def abort(self, context: Context) -> Awaitable[Literal["aborted", "already_placed", "settled"]]: ...


class SettledTask(TaskRecord[Any, Any, R]):
    """A terminal task record."""

    state: ReadOnly[TaskStateTerminal[R]]


class ConversationHandle(Protocol):
    """Invocation-bound conversation operations for tasks and tools. Raises after the invocation ends."""

    @property
    def id(self) -> ConversationId: ...
    def submit(self, submission: InputSubmissionDraft, context: Context) -> Awaitable[Submission]: ...
    def abort(self, context: Context, *, background: bool = False) -> Awaitable[None]:
        """``Conversation.abort()``: withdraw queued inputs, abort the ordinary ownership scope, wait until idle."""
        ...

    def wait_for_idle(self, context: Context) -> Awaitable[None]:
        """Resolve when the conversation's ordinary ownership scope has no live non-background task."""
        ...


class ToolControl(TypedDict, total=False):
    """Post-tools controls requested by a tool result."""

    addTools: ReadOnly[Sequence[str]]
    #: End the run without another model request when every result of the round sets it to ``True``.
    terminate: ReadOnly[bool]
    handoff: ReadOnly[str]


class ToolDiagnostic(TypedDict):
    """Remark about a call for the model and the UI, such as truncation or a spill path."""

    severity: ReadOnly[Literal["info", "warn", "error"]]
    message: ReadOnly[str]
    code: ReadOnly[NotRequired[str]]


class ToolExecutionResult(TypedDict, total=False):
    """What a tool returns: ``output`` for the model, ``structuredOutput`` for programs, ``details`` for UIs."""

    #: For the model. Omitted: the retained ``output()`` text.
    output: ReadOnly[Sequence[TextContent | ImageContent]]
    #: For programs; required unless ``isError`` when the tool declares ``structured_output_schema``.
    structuredOutput: ReadOnly[Any]
    isError: ReadOnly[bool]
    #: Omitted: the last ``details()`` value becomes the details.
    details: ReadOnly[Any]
    #: Added after those recorded through ``api.diagnostic()``.
    diagnostics: ReadOnly[Sequence[ToolDiagnostic]]
    #: Spend of the execution itself, such as a model call.
    usage: ReadOnly[Usage]
    control: ReadOnly[ToolControl]


class NestedToolExecutionResult(TypedDict):
    """Result of a nested call, as ``execute_tool()`` returns it."""

    #: The nested call's tool task.
    taskId: ReadOnly[TaskId[Any]]
    structuredOutput: ReadOnly[NotRequired[Any]]
    isError: ReadOnly[bool]
    details: ReadOnly[NotRequired[Any]]
    diagnostics: ReadOnly[Sequence[ToolDiagnostic]]
    usage: ReadOnly[NotRequired[Usage]]
    #: Execution time of the attempt that produced the result; absent when it never executed.
    durationMs: ReadOnly[NotRequired[float]]


#: Whether the tools of one round run at once or one after another in call order.
ToolExecutionMode: TypeAlias = Literal["parallel", "sequential"]
#: How many queued items of one mode a boundary places: the first, or all of them.
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]


class RetainedOutput(TypedDict):
    text: ReadOnly[str]
    truncated: ReadOnly[bool]


class ToolExecutionApi(DocumentObserver, DocumentReader, Protocol):
    """Operations available to one tool invocation. Every operation raises after the invocation ends."""

    @property
    def task_id(self) -> TaskId[Any]: ...
    @property
    def conversation_id(self) -> ConversationId: ...
    @property
    def call_id(self) -> str: ...
    @property
    def registry(self) -> RegistrySnapshot:
        """The tool task's phase snapshot."""
        ...

    def agent(self, context: Context) -> Awaitable[Agent]:
        """The calling conversation's agent, as the tool task's phase resolved it."""
        ...

    @property
    def models(self) -> Models:
        """The current ``HarnessOptions.models``, read at each access."""
        ...

    @property
    def env(self) -> ExecutionEnv | None:
        """Built by ``HarnessOptions.env`` for this call; ``None`` without an environment."""
        ...

    @property
    def telemetry_context(self) -> TelemetryContext:
        """The call's ``pi.harness.tool`` span, parent of telemetry the tool produces (``NOOP_TELEMETRY_CONTEXT``
        when telemetry is off). Never persist it."""
        ...

    def now(self) -> float:
        """The Harness clock (``HarnessOptions.now``)."""
        ...

    def output(self, chunk: str | bytes, skipped: ShellOutputSkip | None = None) -> None:
        """Append running output; it becomes the result's ``output`` when the result omits it."""
        ...

    @property
    def output_window(self) -> ShellOutputWindow | None:
        """The tail this call's output keeps and the pace of its progress commits, for ``ShellExecOptions.window``."""
        ...

    def retained_output(self) -> RetainedOutput:
        """The output retained so far, as the model will see it when the result omits ``output``."""
        ...

    def diagnostic(self, diagnostic: ToolDiagnostic) -> None:
        """Record a model-visible remark about this call."""
        ...

    def details(self, value: Any, context: Context) -> Awaitable[None]:
        """Replace running details; the last value becomes the result details when the result omits them."""
        ...

    def commit(self, change: Callable[[Tx], T | Awaitable[T]], context: Context) -> Awaitable[T]: ...

    @overload
    def memo(self, name: str, context: Context, /) -> Awaitable[Any]: ...
    @overload
    def memo(self, name: str, candidate: T, context: Context, /) -> Awaitable[T]: ...

    def create_task(
        self, task: Task[Any, Any, R, Any], input: Any, options: TaskOptions, context: Context
    ) -> Awaitable[TaskId[R]]:
        """Create a task in one commit (``options`` without ``conversationId``)."""
        ...

    def get_task(self, id: TaskId[R], context: Context) -> Awaitable[TaskRecord[Any, Any, R] | None]: ...
    def wait_for_task(self, id: TaskId[R], context: Context) -> Awaitable[SettledTask[R]]: ...
    def conversation(self, id: ConversationId, context: Context) -> Awaitable[ConversationHandle | None]: ...
    def execute_tool(
        self,
        name: str,
        args: JsonObject,
        context: Context,
        *,
        key: str | None = None,
        progress: bool | None = None,
    ) -> Awaitable[NestedToolExecutionResult]:
        """Run tool ``name`` as a nested call of this one and wait for its result (see upstream for the rules)."""
        ...


class ToolOutputLimits(TypedDict, total=False):
    maxBytes: ReadOnly[int]
    maxLines: ReadOnly[int]
    retain: ReadOnly[Literal["head", "tail"]]


@dataclass(slots=True, eq=False)
class ToolRegistration(Tool):
    """Executable tool registered in a registry (``define_tool()``).

    Only the :class:`pi_ai.Tool` fields (``name``, ``description``, ``parameters``,
    ``constrained_sampling``) enter the transcript: write ``to_tool()``, never the registration, into
    messages. ``parameters`` is a JSON Schema dict (build it with :class:`pi_ai.Type`) or a Pydantic
    model, as for any pi-ai tool; the Harness validates ``args`` against it before ``execute()``.
    Compared and hashed by identity.
    """

    _: KW_ONLY
    #: ``execute(args, api, context)``.
    execute: Callable[[Any, ToolExecutionApi, Context], Awaitable[ToolExecutionResult]]
    #: Whether an interrupted execution may rerun on recovery. Default ``unsafe``.
    replay: Literal["safe", "unsafe"] | None = None
    #: Default: the settings' ``toolExecution``. One sequential call makes its whole round sequential.
    execution_mode: ToolExecutionMode | None = None
    #: Repair arguments models commonly get wrong before validation. Must be pure.
    prepare_arguments: Callable[[Any], Any] | None = None
    #: JSON Schema of ``structuredOutput``, which programs that call the tool receive.
    structured_output_schema: Mapping[str, Any] | None = None
    #: Who may call the tool: the model, other tools through ``execute_tool()``, or both (default).
    callers: Sequence[Literal["model", "tools"]] | None = None
    #: Not offered to the model until loaded (``modelTools`` names it).
    deferred: Literal[True] | None = None
    output_limits: ToolOutputLimits | None = None
    #: What permission rules match a call on: ``permission_subject(args, env, context)`` (may be async) returns a command
    #: the call runs or a path it reads or writes, absolute (``env``, ``None`` without one, resolves relative paths).
    #: Default none: only rules naming the tool match it.
    permission_subject: (
        Callable[
            [dict[str, Any], ExecutionEnv | None, Context],
            PermissionSubject | Awaitable[PermissionSubject | None] | None,
        ]
        | None
    ) = None
    #: The key a session grant (``approvedForSession``) matches: ``approval_key(args)``; ``None`` from it: never
    #: granted for the session. Default the final arguments as JSON with sorted keys.
    approval_key: Callable[[dict[str, Any]], str | None] | None = None

    def __hash__(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def to_tool(self) -> Tool:
        """The pi-ai tool this registration offers: the fields that enter the transcript."""
        return Tool(self.name, self.description, self.parameters, constrained_sampling=self.constrained_sampling)


class PromptInput(TypedDict):
    """Input to system prompt section rendering for one request preparation."""

    conversationId: ReadOnly[ConversationId]
    #: The request's resolution; ``agent["tools"]`` are the tools offered in this request.
    agent: ReadOnly[Agent]
    #: Built by ``HarnessOptions.env`` for this preparation; ``None`` without an environment.
    env: ReadOnly[ExecutionEnv | None]
    #: Sections already in effect after replaying the active transcript.
    shown: ReadOnly[Mapping[str, str]]
    #: Committed document reads.
    read: ReadOnly[DocumentReader]


@dataclass(frozen=True, slots=True, eq=False)
class PromptSection:
    """One system prompt section; the agent's sections render in order before each request (``section()``)."""

    key: str
    #: ``render(input, context)``: the text, ``None`` to omit the section; may be async.
    render: Callable[[PromptInput, Context], str | Awaitable[str | None] | None]
    #: Default ``True``: wrap the text as ``<key>\\n...\\n</key>``.
    tag: bool | None = None


@dataclass(frozen=True, slots=True, eq=False)
class HookRegistration:
    """Built by ``hook()``; matches tasks by name. ``handlers`` is keyed by snake_case hook name."""

    task: str
    handlers: Mapping[str, Any]


@dataclass(frozen=True, slots=True, eq=False)
class ToolWrap:
    """Built by ``wrap_tool()``: wrap the tool named ``tool`` wherever the wrapping extension is selected."""

    tool: str
    wrap: Callable[[ToolRegistration], ToolRegistration]


@dataclass(frozen=True, slots=True, eq=False)
class SectionWrap:
    """Built by ``wrap_section()``: wrap the section ``section`` wherever the wrapping extension is selected."""

    section: str
    wrap: Callable[[PromptSection], PromptSection]


#: A tool or section decorator; wrappers are pure.
Wrap: TypeAlias = ToolWrap | SectionWrap


@dataclass(frozen=True, slots=True, eq=False)
class Extension:
    """Named bundle of code; installed in a registry and selected by conversations by name."""

    name: str
    tools: Sequence[ToolRegistration] = ()
    sections: Sequence[PromptSection] = ()
    hooks: Sequence[HookRegistration] = ()
    #: Apply where this extension is selected, in order.
    wraps: Sequence[Wrap] = ()
    #: Resolved by name for every task, whichever conversations select this extension.
    tasks: Sequence[AnyTask] = ()


class RegistryTool(TypedDict):
    extension: ReadOnly[Extension]
    tool: ReadOnly[ToolRegistration]


class RegistrySection(TypedDict):
    extension: ReadOnly[Extension]
    section: ReadOnly[PromptSection]


class RegistrySnapshot(Protocol):
    """Immutable view of one published registry state."""

    def installed(self) -> Sequence[Extension]: ...
    def extension(self, name: str) -> Extension | None: ...
    def tools(self) -> Sequence[RegistryTool]:
        """Every installed tool with its extension, in install order. Names may repeat across extensions."""
        ...

    def sections(self) -> Sequence[RegistrySection]: ...
    def tasks(self) -> Sequence[AnyTask]:
        """Built-in and installed task definitions."""
        ...

    def task(self, name: str) -> AnyTask | None: ...


class RegistryReader(Protocol):
    """Read side of a registry consumed by a Harness."""

    def snapshot(self) -> RegistrySnapshot:
        """Immutable view of the whole current registry."""
        ...

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Called synchronously after every publication; returns the unsubscribe function."""
        ...


class Registry(RegistryReader, Protocol):
    """Application-owned registry of extensions."""

    def install(self, extension: Extension) -> None:
        """Install ``extension``, or replace the installed extension with its name in place."""
        ...

    def uninstall(self, extension: Extension) -> None:
        """Remove the installed extension with ``extension.name``, whichever object it is."""
        ...


class ExtensionSelectionEdit(TypedDict, total=False):
    add: list[str]
    remove: list[str]


class ToolSelectionEdit(TypedDict):
    remove: list[str]


class ModelToolsEdit(TypedDict, total=False):
    remove: list[str]
    add: list[str]


class AgentState(TypedDict, total=False):
    """Stored choices of one conversation (the ``pi.agent`` document); names, not objects."""

    model: ModelRef
    thinkingLevel: ModelThinkingLevel
    #: A list selects exactly these extensions, in order. An object edits the host default selection.
    extensions: list[str] | ExtensionSelectionEdit
    #: Filters the selected extensions' tools: the enabled tools.
    tools: list[str] | ToolSelectionEdit
    #: Filters the enabled tools the model may call; never widens.
    modelTools: list[str] | ModelToolsEdit
    #: Rendered after every extension section, as the section ``instructions``.
    instructions: str
    #: Directory within the environment's file system, passed to ``HarnessOptions.env``.
    cwd: str


class ExtensionChangeEdit(TypedDict, total=False):
    add: Sequence[Extension]
    remove: Sequence[Extension]


class ToolChangeEdit(TypedDict):
    remove: Sequence[ToolRegistration]


class ModelToolsChangeEdit(TypedDict, total=False):
    remove: Sequence[ToolRegistration]
    add: Sequence[ToolRegistration]


class AgentChange(TypedDict, total=False):
    """A change to ``pi.agent``: a given key replaces the stored field, ``None`` clears it, absent changes nothing."""

    model: ReadOnly[ModelRef | None]
    thinkingLevel: ReadOnly[ModelThinkingLevel | None]
    extensions: ReadOnly[Sequence[Extension] | ExtensionChangeEdit | None]
    tools: ReadOnly[Sequence[ToolRegistration] | ToolChangeEdit | None]
    #: Replaces the stored filter as a whole; tools loaded with ``control.addTools`` are in its ``add``.
    modelTools: ReadOnly[Sequence[ToolRegistration] | ModelToolsChangeEdit | None]
    instructions: ReadOnly[str | None]
    cwd: ReadOnly[str | None]


class Agent(TypedDict):
    """A conversation's agent resolved against a registry snapshot and the settings."""

    model: ReadOnly[NotRequired[ModelRef]]
    thinkingLevel: ReadOnly[ModelThinkingLevel]
    extensions: ReadOnly[Sequence[Extension]]
    #: The tools a request offers, in order: enabled, callable by the model, and selected by ``modelTools``.
    tools: ReadOnly[Sequence[ToolRegistration]]
    #: The tools nested calls (``execute_tool()``) resolve among, in order.
    callable: ReadOnly[Sequence[ToolRegistration]]
    #: Enabled deferred tools the model may call and is not offered yet, in registry order.
    deferred: ReadOnly[Sequence[ToolRegistration]]
    #: Extension sections, then ``instructions`` when set.
    sections: ReadOnly[Sequence[PromptSection]]
    instructions: ReadOnly[NotRequired[str]]
    cwd: ReadOnly[NotRequired[str]]


#: Runs inside the creating commit, after the creation hook and the ``agent`` change: ``init(tx, conversation_id)``.
ConversationInit: TypeAlias = Callable[[Tx, ConversationId], "Awaitable[None] | None"]


class ConversationCreateOptions(TypedDict):
    ownership: ReadOnly[ConversationOwnership]
    #: Applied in the creating commit after the creation hook's copy, before ``init``.
    agent: ReadOnly[NotRequired[AgentChange]]
    init: ReadOnly[NotRequired[ConversationInit]]


class DeferredStreamOptions(TypedDict, total=False):
    window: Literal["15m", "1h", "24h"]


class ConversationStreamOptions(TypedDict, total=False):
    """Curated pi-ai request options; absent fields use pi-ai defaults."""

    transport: Transport
    timeoutMs: float
    #: Provider/SDK retries inside one request attempt.
    maxRetries: int
    maxRetryDelayMs: float
    headers: Mapping[str, str]
    metadata: JsonObject
    cacheRetention: CacheRetention
    deferred: bool | DeferredStreamOptions


class ConversationRetryPolicy(TypedDict):
    """Durable generation attempt retries; the JSON shape of pi-ai ``RetryPolicy``."""

    enabled: bool
    maxRetries: int
    baseDelayMs: float
    maxAgentDelayMs: NotRequired[float]


class PartialConversationRetryPolicy(TypedDict, total=False):
    enabled: bool
    maxRetries: int
    baseDelayMs: float
    maxAgentDelayMs: float


class CompactionPolicy(TypedDict):
    """Automatic compaction thresholds (spec §8.7); manual compaction ignores ``enabled``."""

    enabled: bool
    #: Room kept free for the answer.
    reserveTokens: int
    #: Approximate size of the recent context a summary keeps verbatim.
    keepRecentTokens: int
    #: Background compaction starts ``backgroundTokens`` below the blocking threshold; ``0`` disables it.
    backgroundTokens: int


class PartialCompactionPolicy(TypedDict, total=False):
    enabled: bool
    reserveTokens: int
    keepRecentTokens: int
    backgroundTokens: int


class CompactionSettings(PartialCompactionPolicy, total=False):
    """Harness-wide compaction settings; ``modelOverrides`` is keyed by exact ``"provider/modelId"``."""

    modelOverrides: Mapping[str, PartialCompactionPolicy]


class ResolvedCompactionPolicy(CompactionPolicy):
    modelOverrides: NotRequired[Mapping[str, PartialCompactionPolicy]]


class ProgressPolicy(TypedDict):
    """How often running progress is committed."""

    #: Minimum pause between commits of the answer being generated.
    partialIntervalMs: float
    #: Minimum pause between commits of running tool output.
    outputIntervalMs: float


class PartialProgressPolicy(TypedDict, total=False):
    partialIntervalMs: float
    outputIntervalMs: float


#: Why a compaction runs: ``compact()``, a threshold in generation preparation, or a context overflow.
CompactionReason: TypeAlias = Literal["manual", "threshold", "overflow"]


class CompactionResult(TypedDict, total=False):
    """``entryId`` of a blocking compaction's summary, or the ``submissionId`` of a summary write."""

    entryId: EntryId
    submissionId: SubmissionId


#: What a failed run does with the inputs queued behind it.
AfterFailedRun: TypeAlias = Literal["keep", "continue", "withdraw"]


class HarnessSettings(TypedDict, total=False):
    """Harness-wide run policy. Read at every resolution and never copied; synchronous."""

    #: Default extension selection; absent: every installed extension, in install order.
    extensions: Sequence[Extension]
    stream: ConversationStreamOptions
    retry: PartialConversationRetryPolicy
    compaction: CompactionSettings
    progress: PartialProgressPolicy
    toolExecution: ToolExecutionMode
    steeringMode: QueueMode
    followUpMode: QueueMode
    contextRetentionMs: float
    documentRetentionMs: float
    afterFailedRun: AfterFailedRun
    restartLimit: float
    #: How long an approval request waits for a decision unless the policy says otherwise; default 15 minutes.
    approvalTimeoutMs: float
    #: How many ``on_yield`` continuations one run takes; later ones are reported and ignored. Default 8.
    maxContinuations: float


class Settings(TypedDict):
    """Resolved settings: every field over its built-in default, object fields merged."""

    extensions: ReadOnly[NotRequired[Sequence[Extension]]]
    stream: ReadOnly[ConversationStreamOptions]
    retry: ReadOnly[ConversationRetryPolicy]
    compaction: ReadOnly[ResolvedCompactionPolicy]
    progress: ReadOnly[ProgressPolicy]
    toolExecution: ReadOnly[ToolExecutionMode]
    steeringMode: ReadOnly[QueueMode]
    followUpMode: ReadOnly[QueueMode]
    contextRetentionMs: ReadOnly[float]
    documentRetentionMs: ReadOnly[float]
    afterFailedRun: ReadOnly[AfterFailedRun]
    restartLimit: ReadOnly[float]
    approvalTimeoutMs: ReadOnly[float]
    maxContinuations: ReadOnly[float]


class EnvTarget(TypedDict):
    """What ``HarnessOptions.env`` builds an environment for."""

    conversationId: ReadOnly[ConversationId]
    #: The conversation's agent ``cwd``.
    cwd: ReadOnly[NotRequired[str]]
    read: ReadOnly[DocumentReader]


#: ``HarnessOptions.env``: ``env(target, context)``; may be async.
EnvBuilder: TypeAlias = Callable[[EnvTarget, Context], "ExecutionEnv | Awaitable[ExecutionEnv | None] | None"]


class Wake(TypedDict):
    """When a Session next needs a running process (``Harness.wake()``)."""

    active: ReadOnly[bool]
    at: ReadOnly[NotRequired[float]]


class HarnessOptions(TypedDict):
    """Options of ``Harness.open()``; callables keep upstream's keys (``setTimer``, ``onWake``, ``onReport``)."""

    #: A ``Models``, or a function returning the current one, called at each use.
    models: ReadOnly[Models | Callable[[], Models]]
    registry: ReadOnly[RegistryReader]
    settings: ReadOnly[NotRequired[HarnessSettings | Callable[[], HarnessSettings | None]]]
    env: ReadOnly[NotRequired[EnvBuilder]]
    #: Decides, per tool call, whether it runs, is denied or must be approved; default every call runs
    #: (:class:`pi_durable.approval.AllowAll`). Held in memory only.
    approval: ReadOnly[NotRequired[ApprovalPolicy]]
    #: Runs in every commit that creates or forks a conversation: ``conversationCreated(tx, conversation)``.
    conversationCreated: ReadOnly[NotRequired[Callable[[Tx, ConversationRecord], Awaitable[None] | None]]]
    #: The Harness clock, epoch milliseconds; default ``time.time() * 1000`` as an ``int``.
    now: ReadOnly[NotRequired[Callable[[], float]]]
    setTimer: ReadOnly[NotRequired[SetTimer]]
    onWake: ReadOnly[NotRequired[Callable[[Wake], None]]]
    onReport: ReadOnly[NotRequired[Callable[[BaseException, ReportInfo], None]]]
    #: Parent of the Harness's spans (``pi.harness.*``, ``pi.session.commit``, and the model requests and code mode
    #: runs below them); see :mod:`pi_durable.telemetry`. Held in memory only, never persisted.
    telemetryContext: ReadOnly[NotRequired[TelemetryContext]]


class TaskInspectionRunning(TypedDict):
    """An invocation is active."""

    kind: ReadOnly[Literal["running"]]


class TaskInspectionReady(TypedDict):
    """The next scheduling pass reserves it; ``migrates`` when its definition is newer and has ``migrate``."""

    kind: ReadOnly[Literal["ready"]]
    migrates: ReadOnly[bool]


class TaskInspectionWaiting(TypedDict):
    kind: ReadOnly[Literal["waiting"]]
    on: ReadOnly[Sequence[TaskId[Any]]]


class TaskInspectionSleeping(TypedDict):
    kind: ReadOnly[Literal["sleeping"]]
    until: ReadOnly[float]


class TaskInspectionCompleting(TypedDict):
    kind: ReadOnly[Literal["completing"]]


class TaskInspectionBlocked(TypedDict):
    kind: ReadOnly[Literal["blocked"]]
    reason: ReadOnly[Literal["missing_task", "task_too_old", "migration_failed", "restart_limit"]]
    error: ReadOnly[NotRequired[BaseException]]


TaskInspectionState: TypeAlias = (
    TaskInspectionRunning
    | TaskInspectionReady
    | TaskInspectionWaiting
    | TaskInspectionSleeping
    | TaskInspectionCompleting
    | TaskInspectionBlocked
)


class TaskInspection(TypedDict):
    """Live task and what the scheduler would do with it under the current registry."""

    record: ReadOnly[AnyTaskRecord]
    state: ReadOnly[TaskInspectionState]


class HarnessInspection(TypedDict):
    """Point-in-time view of live work: unfinished tasks and submissions, read on the Session line."""

    scheduling: ReadOnly[Literal["paused", "running", "closing"]]
    tasks: ReadOnly[Sequence[TaskInspection]]
    #: Queued and placed submissions, in ID order.
    submissions: ReadOnly[Sequence[SubmissionRecord]]


class ContextView(TypedDict):
    """Raw active transcript and derived model context."""

    #: Newest applicable head marker, if any.
    head: ReadOnly[EntryRecord | None]
    #: Raw active entries: the head marker followed by non-head entries from its head through the tail.
    entries: ReadOnly[Sequence[EntryRecord]]
    #: Per entry of ``entries``, its model messages after edits and excluded stop reasons.
    contributions: ReadOnly[Sequence[Sequence[Message]]]
    #: Model context for the next provider request.
    messages: ReadOnly[Sequence[Message]]


ConversationWatch: TypeAlias = "WatchHandle[ConversationView]"


class Conversation(Protocol):
    """Stateless handle for one conversation, bound to the Harness that returned it. Compare handles by ``id``."""

    @property
    def id(self) -> ConversationId: ...
    def agent(self, context: Context) -> Awaitable[Agent]:
        """Resolved with the current registry snapshot and settings."""
        ...

    def configure(self, change: AgentChange, context: Context) -> Awaitable[None]:
        """``configure()`` in its own commit."""
        ...

    def submit(self, submission: SubmissionDraft, context: Context) -> Awaitable[Submission]:
        """Durably admit user input or a passive entry write."""
        ...

    def reset(self, handoff: str | None, context: Context) -> Awaitable[None]:
        """Admit a write of a ``pi.reset`` entry that starts a new context."""
        ...

    def compact(self, instructions: str | None, context: Context) -> Awaitable[TaskId[CompactionResult]]:
        """Admit a manual compaction task and return its ID."""
        ...

    def commit(self, change: Callable[[Tx], T | Awaitable[T]], context: Context) -> Awaitable[T]:
        """Session commit whose ``tx.create_task()`` defaults to this conversation."""
        ...

    def context(self, context: Context, *, at: EntryId | None = None) -> Awaitable[ContextView]:
        """Committed raw active transcript and model context; with ``at``, as of that visible entry."""
        ...

    def entries(
        self,
        query: EntriesQuery,
        limit: int,
        cursor: Cursor | None,
        context: Context,
    ) -> Awaitable[Page[EntryRecord]]:
        """Fork-aware history of this conversation, newest first unless ``query["order"]`` is ``ascending``."""
        ...

    def fork(self, at: EntryId, options: ConversationCreateOptions, context: Context) -> Awaitable[Conversation]: ...
    def abort(self, context: Context, *, background: bool = False) -> Awaitable[None]:
        """Withdraw queued inputs, mark every live non-background task of the scope, and resolve once idle."""
        ...

    def wait_for_idle(self, context: Context) -> Awaitable[None]: ...
    def view_state(self, context: Context) -> Awaitable[AttachedReplicatedState[ConversationView]]:
        """The structural view (spec §9.3) as a disposable read-only Chord state."""
        ...

    def watch(self, context: Context) -> Awaitable[ConversationWatch]:
        """The structural view as a serialized exact-frame watch with bounded pending frames."""
        ...

    def decide(self, id: TaskId[Any], decision: ReviewDecision, context: Context) -> Awaitable[None]:
        """Decide the pending approval request ``id`` (its tool task's id) of this conversation."""
        ...

    def pending_approvals(self, context: Context) -> Awaitable[list[ApprovalRequest]]:
        """The conversation's approval requests waiting for a decision, in id order."""
        ...

    def permissions(self, context: Context) -> Awaitable[PermissionsState]:
        """The conversation's ``pi.permissions``; ``{}`` when none were set."""
        ...

    def update_permissions(self, change: PermissionsChange, context: Context) -> Awaitable[PermissionsState]:
        """Change the conversation's mode, rules or directories in one commit; returns the new state."""
        ...

    def set_permission_mode(self, mode: PermissionMode | None, context: Context) -> Awaitable[PermissionsState]:
        """Set the conversation's permission mode; ``None`` returns it to the policy's default."""
        ...


class EntriesQuery(TypedDict, total=False):
    """``Omit<EntryQuery, "conversationId">``: the query of ``Conversation.entries()``."""

    minEntryId: ReadOnly[EntryId]
    maxEntryId: ReadOnly[EntryId]
    order: ReadOnly[ScanOrder]


class Harness(Session, Protocol):
    """Durable agent harness over one Session."""

    def resume(self) -> None:
        """Enable task scheduling. Idempotent; raises after close."""
        ...

    def wake(self) -> Wake:
        """When this Session next needs a running process, from the scheduler's memory."""
        ...

    def root(
        self,
        context: Context,
        *,
        agent: AgentChange | None = None,
        init: ConversationInit | None = None,
    ) -> Awaitable[Conversation]:
        """Return the reserved root conversation, creating it with ``agent`` and ``init`` in one commit when absent."""
        ...

    def conversation(self, id: ConversationId, context: Context) -> Awaitable[Conversation | None]: ...
    def create_conversation(self, options: ConversationCreateOptions, context: Context) -> Awaitable[Conversation]: ...
    def get_task(self, id: TaskId[R], context: Context) -> Awaitable[TaskRecord[Any, Any, R] | None]: ...
    def inspect(self, context: Context) -> Awaitable[HarnessInspection]:
        """Live tasks and unsettled submissions. Writes nothing and runs no task code."""
        ...

    def submission(self, id: SubmissionId, context: Context) -> Awaitable[Submission | None]:
        """Reacquire a submission, for example after reopen."""
        ...

    def abort_submission(
        self, id: SubmissionId, context: Context, conversation_id: ConversationId | None = None
    ) -> Awaitable[Literal["aborted", "already_placed", "settled", "not_found"]]: ...
    def abort_task(self, id: TaskId[Any], context: Context) -> Awaitable[Literal["marked", "terminal"]]: ...
    def retry_task(self, id: TaskId[Any], context: Context) -> Awaitable[None]:
        """Run a task parked by ``settings.restartLimit`` again."""
        ...

    def wait_for_task(self, id: TaskId[R], context: Context) -> Awaitable[SettledTask[R]]:
        """Resolve with the terminal receipt; cancelling ``context`` cancels only this wait."""
        ...

    def wait_for_idle(self, context: Context) -> Awaitable[None]: ...
    def usage(self, context: Context) -> Awaitable[UsageState]:
        """Session total: every conversation's ``pi.usage`` summed."""
        ...

    def decide(self, id: TaskId[Any], decision: ReviewDecision, context: Context) -> Awaitable[None]:
        """Decide the approval request ``id`` in whichever conversation its tool task runs."""
        ...

    def pending_approvals(self, context: Context) -> Awaitable[list[tuple[ConversationId, ApprovalRequest]]]:
        """Every conversation's approval requests waiting for a decision."""
        ...

    def permission_rules(self, context: Context) -> Awaitable[PermissionRules]:
        """The allow rules approvals added (``approvedWithRule``), for every conversation."""
        ...

    def remove_permission_rule(self, rule: str, context: Context) -> Awaitable[bool]:
        """Remove ``rule`` from the rules approvals added; whether it was there."""
        ...

    def task_graph(self, context: Context) -> Awaitable[AttachedReplicatedState[TaskGraph]]: ...
    def watch_task_graph(self, context: Context) -> Awaitable[TaskGraphWatch]: ...


class HookApi(DocumentReader, Protocol):
    """What a hook may use: committed reads and the asking task's memos, which hooks and the task share."""

    @property
    def task_id(self) -> TaskId[Any]: ...
    @property
    def conversation_id(self) -> ConversationId: ...
    @property
    def models(self) -> Models:
        """The current ``HarnessOptions.models``, read at each access."""
        ...

    @overload
    async def memo(self, name: str, context: Context, /) -> Any: ...
    @overload
    async def memo(self, name: str, candidate: T, context: Context, /) -> T: ...

    async def agent(self, context: Context) -> Agent:
        """The conversation's agent as the phase resolved it."""
        ...

    async def env(self, context: Context) -> ExecutionEnv | None:
        """The conversation's environment, built for this use; ``None`` without ``HarnessOptions.env``. Commands a hook
        runs go through it."""
        ...

    async def cwd(self, context: Context) -> str | None:
        """The conversation's working directory: its environment's, else the agent's ``cwd``."""
        ...

    async def conversation_record(self, context: Context) -> ConversationRecord | None:
        """The committed record of the hook's conversation (its owner, its parent)."""
        ...


#: What a hook returns: a value, ``None`` for no decision, or an awaitable of either.
HookResult: TypeAlias = "T | Awaitable[T | None] | None"


class GenerationRequest(TypedDict):
    messages: ReadOnly[Sequence[Message]]


#: What ``on_yield`` returns to continue the run: ``{"continue": input}``, appended as a user message.
GenerationContinue = TypedDict("GenerationContinue", {"continue": "UserInput"})


class RunInput(TypedDict):
    """The input a run starts with, as ``on_input`` sees it: the entries its inputs placed, in order."""

    entries: ReadOnly[Sequence[EntryRecord]]


class InputBlock(TypedDict):
    """Ends the run ``unanswered`` with reason ``blocked`` and this detail before any request; the input stays in the
    transcript."""

    block: ReadOnly[str]


class InputContext(TypedDict):
    """Context for the model, appended as a ``pi.context`` entry (one system message) before the first request."""

    context: ReadOnly[str]


#: What an ``on_input`` hook decides about a run's input.
InputDecision: TypeAlias = InputBlock | InputContext


class GenerationHooks(TypedDict, total=False):
    """Hooks of the built-in generation task.

    - ``before_request(request, api, context)``: before every request attempt; the result is used for that
      request only.
    - ``after_response(message, api, context)``: every terminal provider message, before classification.
    - ``on_input(input, api, context)``: once per run, before its first request; the first ``{"block": reason}``
      ends the run, every ``{"context": text}`` is appended for the model. A raise is reported.
    - ``on_yield(answer, api, context)``: a final answer; the first ``{"continue": input}`` continues the run, up to
      ``settings.maxContinuations`` times (the count so far is ``pi.live``'s ``run.continuations``).
    - ``after_tools(assistant, results, api, context)``: after every tool of the round is terminal.
    """

    before_request: Callable[[GenerationRequest, HookApi, Context], HookResult[GenerationRequest]]
    after_response: Callable[[AssistantMessage, HookApi, Context], Awaitable[None] | None]
    on_input: Callable[[RunInput, HookApi, Context], HookResult[InputDecision]]
    on_yield: Callable[[AssistantMessage, HookApi, Context], HookResult[GenerationContinue]]
    after_tools: Callable[[EntryId, Sequence[EntryId], HookApi, Context], Awaitable[None] | None]


class ToolCallParent(TypedDict):
    taskId: ReadOnly[TaskId[Any]]
    callId: ReadOnly[str]


@dataclass(slots=True)
class ToolHookCall(ToolCall):
    """A call as tool hooks see it; ``parent`` is set for a nested call made through ``execute_tool()``."""

    _: KW_ONLY
    parent: ToolCallParent | None = None


class BeforeToolResult(TypedDict, total=False):
    arguments: ReadOnly[JsonObject]
    block: ReadOnly[str]
    #: The hook's word on approval; the most restrictive of the chain reaches the policy. ``allow`` skips what the
    #: policy would ask by its mode (deny and ask rules still apply); ``ask`` asks even when the policy allows.
    permission: ReadOnly[Literal["allow", "ask"]]
    #: The prompt of ``permission: "ask"``.
    reason: ReadOnly[str]
    #: Text for the model, appended to the call's result as a final text item (after ``after_tool``); every hook's
    #: context is kept.
    context: ReadOnly[str]


class ToolHooks(TypedDict, total=False):
    """Hooks of the built-in tool task, for model-issued and nested calls.

    - ``before_tool(call, api, context)``: before intent; the first ``block`` wins, otherwise ``arguments``
      replace the call's arguments, the most restrictive ``permission`` reaches the approval policy, and every
      ``context`` follows the result. A raise blocks.
    - ``after_tool(call, result, api, context)``: after execution, before the result is committed; replaces it.
    """

    before_tool: Callable[[ToolHookCall, HookApi, Context], HookResult[BeforeToolResult]]
    after_tool: Callable[[ToolHookCall, ToolExecutionResult, HookApi, Context], HookResult[ToolExecutionResult]]


class CompactionHookInput(TypedDict):
    reason: ReadOnly[CompactionReason]
    #: The active entries the summary replaces, the head marker first.
    entries: ReadOnly[Sequence[EntryRecord]]
    #: Their model context, the summarizer's source.
    messages: ReadOnly[Sequence[Message]]
    #: The first entry kept verbatim.
    firstKept: ReadOnly[EntryId]
    instructions: ReadOnly[NotRequired[str]]


class CompactionDecline(TypedDict):
    decline: ReadOnly[Literal[True]]


class CompactionSummary(TypedDict):
    summary: ReadOnly[str]


class CompactionHooks(TypedDict, total=False):
    """Hooks of the built-in compaction task.

    - ``before_compact(compaction, api, context)``: after range selection, before summarizing; the first
      decision wins.
    """

    before_compact: Callable[[CompactionHookInput, HookApi, Context], HookResult[CompactionDecline | CompactionSummary]]


def __getattr__(name: str) -> Any:
    """The types later harness modules define, imported on first access (they import this module)."""
    if name == "ConversationView":
        from .view import ConversationView

        return ConversationView
    if name in ("TaskGraph", "TaskGraphWatch"):
        from . import task_graph

        return getattr(task_graph, name)
    if name == "UsageState":
        from .usage import UsageState

        return UsageState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
