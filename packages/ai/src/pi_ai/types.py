"""Core data types for pi-ai.

Every type mirrors its counterpart in ``@earendil-works/pi-ai`` (``src/types.ts``).
Python attribute names are snake_case; :mod:`pi_ai.serialization` converts to and
from pi's camelCase JSON so transcripts, models and contexts stay interchangeable
with the TypeScript implementation.

All data classes use ``__slots__`` for low memory use and fast attribute access.
Discriminators (``type`` on content blocks, events and models, ``role`` on
messages) are class-level constants, so they can be read as ``block.type`` and
matched structurally (``match event: case TextDeltaEvent(delta=d): ...``).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    import httpx

    from .abort import AbortSignal
    from .event_stream import AssistantMessageEventStream
    from .utils.diagnostics import AssistantMessageDiagnostic

__all__ = [
    # literal aliases
    "KnownApi",
    "Api",
    "KnownImageApi",
    "ImageApi",
    "KnownClassifierApi",
    "ClassifierApi",
    "KnownProvider",
    "ProviderId",
    "StopReason",
    "ThinkingLevel",
    "ModelThinkingLevel",
    "THINKING_LEVELS",
    "MODEL_THINKING_LEVELS",
    "CacheRetention",
    "Transport",
    "ToolChoice",
    "ModelType",
    "ModelInputModality",
    "JsonValue",
    "JsonObject",
    "ProviderEnv",
    "ProviderHeaders",
    "SamplingParams",
    # content
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    "TextSignatureV1",
    "AssistantContent",
    "UserContent",
    "ToolResultContent",
    # usage
    "UsageCost",
    "Usage",
    # tools
    "GrammarFormat",
    "GrammarVariants",
    "ConstrainedSamplingConfig",
    "Tool",
    "ToolReference",
    # messages
    "DeferredHandle",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "NestedToolCallRecord",
    "NestedToolCalls",
    "ToolResultMessage",
    "Message",
    "Context",
    "TranscriptContext",
    # events
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    "AssistantMessageEvent",
    # models
    "ModelCostRates",
    "ModelCostTier",
    "ModelCost",
    "ModelImageResizeOptions",
    "ModelImageInputLimits",
    "ModelInputLimits",
    "BaseModel",
    "Model",
    "ImageModel",
    "ClassifierModel",
    "AnyModel",
    # images
    "ImagesContext",
    "AssistantImages",
    # classifier
    "ClassifierChoiceQuestion",
    "ClassifierScoreQuestion",
    "ClassifierBoolQuestion",
    "ClassifierQuestion",
    "ClassifierContext",
    "ClassifierChoiceAnswer",
    "ClassifierScoreAnswer",
    "ClassifierBoolAnswer",
    "ClassifierAnswer",
    "ClassifierResult",
    # options
    "ProviderResponse",
    "ThinkingBudgets",
    "ProviderRequestOptions",
    "StreamOptions",
    "SimpleStreamOptions",
    "DeferredFetchOptions",
    "DeferredCancelOptions",
    "ImagesOptions",
    "ClassifierOptions",
    # implementation contracts
    "ProviderStreams",
    "ProviderImages",
    "ProviderClassifier",
    "StreamFunction",
    "now_ms",
]


def now_ms() -> int:
    """Current Unix time in milliseconds (the timestamp unit used throughout pi)."""
    return time.time_ns() // 1_000_000


# ---------------------------------------------------------------------------
# Literal aliases
# ---------------------------------------------------------------------------

KnownApi: TypeAlias = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
#: Any API id. Known ids are listed in :data:`KnownApi`; custom providers may use their own.
Api: TypeAlias = str

KnownImageApi: TypeAlias = Literal["openrouter-images"]
ImageApi: TypeAlias = str

KnownClassifierApi: TypeAlias = Literal[
    "typesafe-system-one",
    "cloudflare-workers-ai-system-one",
    "llama-cpp-classify",
    "openai-decisions",
]
ClassifierApi: TypeAlias = str

KnownProvider: TypeAlias = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure",
    "openai-codex",
    "radius",
    "typesafe",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "baseten",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "meta",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "qwen-token-plan-individual",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
ProviderId: TypeAlias = str

StopReason: TypeAlias = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]

#: Thinking levels accepted by ``stream_simple(reasoning=...)``.
ThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
#: Thinking levels including ``"off"``.
ModelThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")
MODEL_THINKING_LEVELS: tuple[ModelThinkingLevel, ...] = ("off", *THINKING_LEVELS)

CacheRetention: TypeAlias = Literal["none", "short", "long"]
Transport: TypeAlias = Literal["sse", "websocket", "websocket-cached", "auto"]
ToolChoice: TypeAlias = Literal["auto", "none"]
ModelType: TypeAlias = Literal["chat", "image", "classifier"]
ModelInputModality: TypeAlias = Literal["text", "image"]

JsonValue: TypeAlias = Any
JsonObject: TypeAlias = dict[str, Any]
#: Provider-scoped environment overrides. Values take precedence over ``os.environ``.
ProviderEnv: TypeAlias = dict[str, str]
#: Request headers. A ``None`` value suppresses a provider/API default header with the same name.
ProviderHeaders: TypeAlias = dict[str, "str | None"]
SamplingParams: TypeAlias = dict[str, Any]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextSignatureV1(TypedDict, total=False):
    v: Literal[1]
    id: str
    phase: Literal["commentary", "final_answer"]


@dataclass(slots=True)
class TextContent:
    """Plain text."""

    type: ClassVar[Literal["text"]] = "text"
    text: str = ""
    _: KW_ONLY
    #: Provider message metadata, e.g. OpenAI Responses message ids (legacy id string or TextSignatureV1 JSON).
    text_signature: str | None = None


@dataclass(slots=True)
class ThinkingContent:
    """Model reasoning. ``thinking_signature`` carries provider replay data."""

    type: ClassVar[Literal["thinking"]] = "thinking"
    thinking: str = ""
    _: KW_ONLY
    thinking_signature: str | None = None
    #: True when the reasoning was redacted by safety filters; the encrypted payload is in ``thinking_signature``.
    redacted: bool | None = None


@dataclass(slots=True)
class ImageContent:
    """Base64-encoded image data."""

    type: ClassVar[Literal["image"]] = "image"
    data: str
    mime_type: str


@dataclass(slots=True)
class ToolCall:
    """A tool call requested by the model."""

    type: ClassVar[Literal["toolCall"]] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    #: Google-specific opaque signature for reusing thought context.
    thought_signature: str | None = None
    #: OpenAI Responses namespace for calls to dynamically loaded or namespaced tools.
    namespace: str | None = None
    #: Set when the model's complete arguments text was not a valid JSON object; ``arguments`` is then ``{}``.
    arguments_error: str | None = None


AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCall
UserContent: TypeAlias = TextContent | ImageContent
ToolResultContent: TypeAlias = TextContent | ImageContent


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class Usage:
    """Token counts and their cost in USD."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)
    _: KW_ONLY
    #: Subset of ``cache_write`` written with 1h retention. Only Anthropic reports this split.
    cache_write_1h: int | None = None
    #: Reasoning tokens, a subset of ``output``, when the provider reports them.
    reasoning: int | None = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

GrammarFormat: TypeAlias = Literal["openai_lark", "openai_regex"]
GrammarVariants: TypeAlias = dict[str, str]


class ConstrainedSamplingConfig(TypedDict, total=False):
    """Provider-side constrained sampling for a tool.

    ``{"type": "json_schema", "strict": "prefer" | "require"}`` or
    ``{"type": "grammar", "variants": {"openai_lark": "..."}}``.
    """

    type: Literal["json_schema", "grammar"]
    strict: Literal["prefer", "require"]
    variants: GrammarVariants


@dataclass(slots=True)
class Tool:
    """A tool the model may call.

    ``parameters`` is a JSON Schema object (build it with :class:`pi_ai.Type`) or a
    Pydantic model class. Pydantic models are converted to JSON Schema for the
    provider and validated arguments are returned as model instances.
    """

    name: str
    description: str = ""
    parameters: Any = field(default_factory=lambda: {"type": "object", "properties": {}})
    _: KW_ONLY
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = None


@dataclass(slots=True)
class ToolReference:
    name: str


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeferredHandle:
    provider: str
    model_id: str
    api: str
    #: Provider token, such as a response id or batch id plus row id.
    id: str
    _: KW_ONLY
    expires_at: int | None = None
    poll_after_ms: int | None = None
    #: Provider conversion data required to reconstruct the final assistant message.
    data: Any = None


@dataclass(slots=True)
class SystemMessage:
    """System instructions and tool declarations at one point in the transcript.

    The leading system message is the system prompt. Later system messages change it:
    ``content`` adds instructions from that point on, ``sections`` replace or remove
    named prompt sections (``None`` removes one), and ``tools_added``/``tools_removed``
    change the tool set. Replaying every system message in order yields the current
    prompt and tools.
    """

    role: ClassVar[Literal["system"]] = "system"
    content: str | list[TextContent] = ""
    _: KW_ONLY
    sections: dict[str, str | None] | None = None
    tools_added: list[Tool] | None = None
    tools_removed: list[ToolReference] | None = None
    timestamp: int = field(default_factory=now_ms)


@dataclass(slots=True)
class UserMessage:
    role: ClassVar[Literal["user"]] = "user"
    content: str | list[TextContent | ImageContent] = ""
    _: KW_ONLY
    timestamp: int = field(default_factory=now_ms)


@dataclass(slots=True)
class AssistantMessage:
    role: ClassVar[Literal["assistant"]] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall] = field(default_factory=list)
    _: KW_ONLY
    api: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    #: Unix timestamp in milliseconds when the request started.
    timestamp: int = field(default_factory=now_ms)
    #: Concrete model reported by the provider when different from the requested ``model``.
    response_model: str | None = None
    #: Provider-specific response/message identifier when the upstream API exposes one.
    response_id: str | None = None
    #: Exact provider-native effort level used for this response.
    provider_thinking_level: str | None = None
    #: Pi thinking level the agent loop requested for this response.
    thinking_level: ModelThinkingLevel | None = None
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    deferred: DeferredHandle | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    #: Provider indication of whether the model explicitly ended its turn.
    end_turn: bool | None = None
    #: Milliseconds from ``timestamp`` until the response ended (monotonic clock).
    duration_ms: int | None = None

    def shallow_copy(self) -> AssistantMessage:
        """Copy the message object (not its content blocks), like ``{ ...message }`` in TypeScript."""
        copied = object.__new__(AssistantMessage)
        for name in _ASSISTANT_SLOTS:
            object.__setattr__(copied, name, getattr(self, name))
        copied.content = list(self.content)
        return copied


_ASSISTANT_SLOTS: tuple[str, ...] = AssistantMessage.__slots__  # type: ignore[attr-defined]


@dataclass(slots=True)
class NestedToolCallRecord:
    """A tool call that another tool made while it ran (for example from a codemode script)."""

    id: str
    name: str
    status: Literal["ok", "error", "unfinished"]
    _: KW_ONLY
    arguments: dict[str, Any] | None = None
    arguments_bytes: int | None = None
    duration_ms: int | None = None
    error: str | None = None


@dataclass(slots=True)
class NestedToolCalls:
    calls: list[NestedToolCallRecord] = field(default_factory=list)
    complete: bool = True


@dataclass(slots=True)
class ToolResultMessage:
    role: ClassVar[Literal["toolResult"]] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent] = field(default_factory=list)
    is_error: bool = False
    _: KW_ONLY
    details: Any = None
    #: Usage from the tool execution itself. Not part of the main LLM context accounting.
    usage: Usage | None = None
    #: Calls this tool made to other tools. Kept for the session record; not sent to the model.
    nested_calls: NestedToolCalls | None = None
    timestamp: int = field(default_factory=now_ms)
    #: Milliseconds the tool's execution took.
    duration_ms: int | None = None


Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class Context:
    """Request input accepted by the public stream entry points.

    ``system_prompt`` and ``tools`` are shorthand for a leading system message;
    :func:`pi_ai.normalize_context` folds them into one before the request reaches
    a provider.
    """

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[Tool] | None = None


class TranscriptContext:
    """Normalized request context passed to providers and API implementations.

    The prompt and tool declarations are carried by the transcript's system
    messages. Only :func:`pi_ai.normalize_context` (and the transcript helpers)
    produce this type, so a raw :class:`Context` cannot reach provider code by
    accident.
    """

    __slots__ = ("messages",)

    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    def __repr__(self) -> str:
        return f"TranscriptContext(messages={self.messages!r})"


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StartEvent:
    type: ClassVar[Literal["start"]] = "start"
    partial: AssistantMessage


@dataclass(slots=True)
class TextStartEvent:
    type: ClassVar[Literal["text_start"]] = "text_start"
    content_index: int
    partial: AssistantMessage


@dataclass(slots=True)
class TextDeltaEvent:
    type: ClassVar[Literal["text_delta"]] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(slots=True)
class TextEndEvent:
    type: ClassVar[Literal["text_end"]] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass(slots=True)
class ThinkingStartEvent:
    type: ClassVar[Literal["thinking_start"]] = "thinking_start"
    content_index: int
    partial: AssistantMessage


@dataclass(slots=True)
class ThinkingDeltaEvent:
    type: ClassVar[Literal["thinking_delta"]] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(slots=True)
class ThinkingEndEvent:
    type: ClassVar[Literal["thinking_end"]] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


@dataclass(slots=True)
class ToolCallStartEvent:
    type: ClassVar[Literal["toolcall_start"]] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


@dataclass(slots=True)
class ToolCallDeltaEvent:
    type: ClassVar[Literal["toolcall_delta"]] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


@dataclass(slots=True)
class ToolCallEndEvent:
    type: ClassVar[Literal["toolcall_end"]] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


@dataclass(slots=True)
class DoneEvent:
    type: ClassVar[Literal["done"]] = "done"
    reason: Literal["stop", "length", "toolUse", "deferred"]
    message: AssistantMessage


@dataclass(slots=True)
class ErrorEvent:
    type: ClassVar[Literal["error"]] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent: TypeAlias = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelCostRates:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(slots=True)
class ModelCostTier:
    """Request-wide pricing tier used when total request input exceeds ``input_tokens_above``."""

    input_tokens_above: int
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(slots=True)
class ModelCost:
    """USD per million tokens, with optional request-wide pricing tiers."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    tiers: list[ModelCostTier] | None = None


@dataclass(slots=True)
class ModelImageResizeOptions:
    max_width: int | None = None
    max_height: int | None = None
    #: Maximum base64-encoded payload size in bytes.
    max_bytes: int | None = None
    jpeg_quality: int | None = None


@dataclass(slots=True)
class ModelImageInputLimits:
    resize: ModelImageResizeOptions | None = None
    max_per_message: int | None = None
    max_per_request: int | None = None


@dataclass(slots=True)
class ModelInputLimits:
    #: Maximum serialized provider request size in bytes.
    max_request_bytes: int | None = None
    images: ModelImageInputLimits | None = None


@dataclass(slots=True, kw_only=True)
class BaseModel:
    """Fields shared by every catalog entry, regardless of what you can do with it."""

    type: ClassVar[ModelType] = "chat"
    id: str
    name: str = ""
    api: str
    provider: str
    base_url: str = ""
    input: list[str] = field(default_factory=lambda: ["text"])
    cost: ModelCost = field(default_factory=ModelCost)
    input_limits: ModelInputLimits | None = None
    headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id


@dataclass(slots=True, kw_only=True)
class Model(BaseModel):
    """Chat model: usable with ``stream()`` and friends.

    ``compat`` holds API-specific compatibility overrides as a plain dict using pi's
    JSON keys (``{"supportsDeveloperRole": False}``), so ``models.json`` configuration
    is interchangeable with the TypeScript implementation.
    """

    type: ClassVar[Literal["chat"]] = "chat"
    reasoning: bool = False
    context_window: int = 128_000
    max_tokens: int = 16_384
    #: Maps pi thinking levels to provider/model-specific values. ``None`` marks a level unsupported.
    thinking_level_map: dict[str, str | None] | None = None
    #: Prompt cache lifetimes in seconds per retention tier (``{"short": 300, "long": 3600}``).
    prompt_cache: dict[str, float] | None = None
    #: Default sampling parameters; per-request ``sampling_params`` override these per key.
    sampling_params: SamplingParams | None = None
    #: Sampling parameter overrides selected by the effective pi thinking level.
    sampling_params_by_thinking_level: dict[str, SamplingParams] | None = None
    #: Compatibility overrides for the model's API. Unset fields are auto-detected.
    compat: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class ImageModel(BaseModel):
    """Image-generation model: usable with ``generate_images()`` only."""

    type: ClassVar[Literal["image"]] = "image"
    #: Output modalities. Always includes ``"image"``; ``"text"`` means the model can also return text blocks.
    output: list[str] = field(default_factory=lambda: ["image"])


@dataclass(slots=True, kw_only=True)
class ClassifierModel(BaseModel):
    """Structured classifier model: usable with ``classify()`` only."""

    type: ClassVar[Literal["classifier"]] = "classifier"
    context_window: int = 0


AnyModel: TypeAlias = Model | ImageModel | ClassifierModel


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImagesContext:
    input: list[TextContent | ImageContent] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class AssistantImages:
    api: str
    provider: str
    model: str
    output: list[TextContent | ImageContent] = field(default_factory=list)
    response_id: str | None = None
    usage: Usage | None = None
    stop_reason: Literal["stop", "error", "aborted"] = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassifierChoiceQuestion:
    type: ClassVar[Literal["choice"]] = "choice"
    instructions: str
    criteria: dict[str, str]


@dataclass(slots=True)
class ClassifierScoreQuestion:
    type: ClassVar[Literal["score"]] = "score"
    instructions: str
    criteria: list[str]


@dataclass(slots=True)
class ClassifierBoolQuestion:
    type: ClassVar[Literal["bool"]] = "bool"
    instructions: str
    #: ``{"true": "...", "false": "..."}``
    criteria: dict[str, str]


ClassifierQuestion: TypeAlias = ClassifierChoiceQuestion | ClassifierScoreQuestion | ClassifierBoolQuestion


@dataclass(slots=True)
class ClassifierContext:
    state: dict[str, Any]
    questions: dict[str, ClassifierQuestion]
    _: KW_ONLY
    #: Images judged together with ``state``. Only models whose ``input`` includes ``"image"`` accept them.
    images: list[ImageContent] | None = None


@dataclass(slots=True)
class ClassifierChoiceAnswer:
    type: ClassVar[Literal["choice"]] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(slots=True)
class ClassifierScoreAnswer:
    type: ClassVar[Literal["score"]] = "score"
    score: float
    confidence: float


@dataclass(slots=True)
class ClassifierBoolAnswer:
    type: ClassVar[Literal["bool"]] = "bool"
    probability: float


ClassifierAnswer: TypeAlias = ClassifierChoiceAnswer | ClassifierScoreAnswer | ClassifierBoolAnswer


@dataclass(slots=True, kw_only=True)
class ClassifierResult:
    api: str
    provider: str
    model: str
    answers: dict[str, ClassifierAnswer] = field(default_factory=dict)
    usage: Usage | None = None
    stop_reason: Literal["stop", "error", "aborted"] = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderResponse:
    status: int
    headers: dict[str, str]


class ThinkingBudgets(TypedDict, total=False):
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int
    low: int
    medium: int
    high: int


MaybeAwaitable: TypeAlias = "Any | Awaitable[Any]"


class ProviderRequestOptions(TypedDict, total=False):
    """Authentication, HTTP transport, and lifecycle callbacks shared by provider requests.

    Every public entry point also accepts these as keyword arguments.
    """

    signal: AbortSignal
    api_key: str
    #: Custom ``httpx.AsyncClient`` for provider HTTP requests (the analog of pi's ``fetch`` option).
    http_client: httpx.AsyncClient
    #: Provider-scoped environment values; take precedence over ``os.environ``.
    env: ProviderEnv
    #: Inspect or replace the provider payload before sending. Return ``None`` to keep it unchanged.
    on_payload: Callable[[Any, Any], Any]
    #: Called after an HTTP response is received, before its body is consumed.
    on_response: Callable[[ProviderResponse, Any], Any]
    #: Extra headers merged over provider defaults. ``None`` values suppress a default header.
    headers: ProviderHeaders
    #: HTTP request timeout in milliseconds.
    timeout_ms: float
    #: Maximum client-side retry attempts for retryable HTTP failures (default 0 at this layer).
    max_retries: int
    #: Fail instead of waiting when the server requests a retry delay above this (default 60000; 0 disables).
    max_retry_delay_ms: float


class StreamOptions(ProviderRequestOptions, total=False):
    #: Observe each parsed provider stream event before pi normalizes it. Treat as read-only.
    on_provider_stream_event: Callable[[Any, Any], Any]
    temperature: float
    #: Extra sampling parameters merged into OpenAI-compatible request bodies (``top_p``, ``top_k``, ...).
    sampling_params: SamplingParams
    max_tokens: int
    transport: Transport
    #: Prompt cache retention preference. Default ``"short"``.
    cache_retention: CacheRetention
    #: Session identifier for session-based caching and routing.
    session_id: str
    websocket_connect_timeout_ms: float
    #: Request metadata. Providers extract the fields they understand (e.g. Anthropic ``user_id``).
    metadata: dict[str, Any]


class SimpleStreamOptions(StreamOptions, total=False):
    #: Provider-neutral tool selection.
    tool_choice: ToolChoice
    reasoning: ThinkingLevel
    #: Ask a capable provider to return a durable handle and continue asynchronously.
    deferred: bool | dict[str, str]
    thinking_budgets: ThinkingBudgets


class DeferredFetchOptions(ProviderRequestOptions, total=False):
    #: Maximum provider long-poll duration in milliseconds. 0 performs one status check.
    wait: float


DeferredCancelOptions: TypeAlias = ProviderRequestOptions


class ImagesOptions(ProviderRequestOptions, total=False):
    metadata: dict[str, Any]


class ClassifierOptions(ProviderRequestOptions, total=False):
    #: Divides answer logits before normalization. Values above 1 soften the distribution.
    temperature: float


# ---------------------------------------------------------------------------
# Implementation contracts
# ---------------------------------------------------------------------------

StreamFunction: TypeAlias = Callable[
    [Model, TranscriptContext, "Mapping[str, Any] | None"], "AssistantMessageEventStream"
]


class ProviderStreams(Protocol):
    """The stream contract of an API implementation module.

    Every module under ``pi_ai.api`` exposes ``stream`` and ``stream_simple``; capable
    modules also expose ``fetch_deferred`` and ``cancel_deferred``.
    """

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream: ...


class ProviderImages(Protocol):
    async def generate_images(
        self, model: ImageModel, context: ImagesContext, options: Mapping[str, Any] | None = None
    ) -> AssistantImages: ...


class ProviderClassifier(Protocol):
    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: Mapping[str, Any] | None = None
    ) -> ClassifierResult: ...
