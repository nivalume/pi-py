"""In-memory provider with scripted responses for tests and demos.

::

    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("Hello!")])
    message = await models.complete(faux.get_model(), Context(messages=[UserMessage("Hi")]))
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import random
import string
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, overload

from ..auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from ..event_stream import AssistantMessageEventStream
from ..models import Provider, create_provider
from ..types import (
    AssistantMessage,
    DeferredHandle,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    ModelInputLimits,
    ProviderResponse,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    UsageCost,
    now_ms,
)
from ..utils.callbacks import maybe_await
from ..utils.text import append_text, append_thinking, get_system_message_text

__all__ = [
    "FauxModelDefinition",
    "FauxProviderState",
    "FauxResponseFactory",
    "FauxResponseStep",
    "FauxProviderHandle",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "faux_assistant_message",
    "faux_provider",
    "create_faux_core",
]

_DEFAULT_API = "faux"
_DEFAULT_PROVIDER = "faux"
_DEFAULT_MODEL_ID = "faux-1"
_DEFAULT_MODEL_NAME = "Faux Model"
_DEFAULT_BASE_URL = "http://localhost:0"
_DEFAULT_MIN_TOKEN_SIZE = 3
_DEFAULT_MAX_TOKEN_SIZE = 5


@dataclass(slots=True)
class FauxModelDefinition:
    id: str
    name: str | None = None
    reasoning: bool = False
    input: list[str] | None = None
    input_limits: ModelInputLimits | None = None
    cost: ModelCost | None = None
    context_window: int = 128_000
    max_tokens: int = 16_384


FauxContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCall


def faux_text(text: str) -> TextContent:
    return TextContent(text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking)


def _random_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=11))
    return f"{prefix}:{now_ms()}:{suffix}"


def faux_tool_call(name: str, arguments: dict[str, Any], *, id: str | None = None) -> ToolCall:
    return ToolCall(id or _random_id("tool"), name, arguments)


def faux_assistant_message(
    content: str | FauxContentBlock | Sequence[FauxContentBlock],
    *,
    stop_reason: StopReason = "stop",
    deferred: DeferredHandle | None = None,
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    """Build a scripted assistant reply. A string becomes one text block."""
    if isinstance(content, str):
        blocks: list[Any] = [faux_text(content)]
    elif isinstance(content, (TextContent, ThinkingContent, ToolCall)):
        blocks = [content]
    else:
        blocks = list(content)
    return AssistantMessage(
        blocks,
        api=_DEFAULT_API,
        provider=_DEFAULT_PROVIDER,
        model=_DEFAULT_MODEL_ID,
        usage=Usage(),
        stop_reason=stop_reason,
        deferred=deferred,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else now_ms(),
    )


@dataclass(slots=True)
class FauxProviderState:
    call_count: int = 0
    deferred_fetch_count: int = 0
    cancelled_deferred: list[DeferredHandle] = field(default_factory=list)


FauxResponseFactory = Callable[
    [TranscriptContext, Mapping[str, Any] | None, FauxProviderState, Model],
    "AssistantMessage | Awaitable[AssistantMessage]",
]
FauxResponseStep: TypeAlias = "AssistantMessage | FauxResponseFactory"


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(content: Sequence[Any]) -> str:
    parts = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "thinking":
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{json.dumps(block.arguments, separators=(',', ':'))}")
    return "\n".join(parts)


def _tool_json(tool: Any) -> str:
    from ..serialization import to_json_value

    return json.dumps(to_json_value(tool), separators=(",", ":"))


def _message_to_text(message: Any) -> str:
    role = message.role
    if role == "system":
        parts = [get_system_message_text(message)]
        parts.extend(f"tool-:{_tool_json(tool)}" for tool in message.tools_removed or ())
        parts.extend(f"tool+:{_tool_json(tool)}" for tool in message.tools_added or ())
        return "\n".join(part for part in parts if part)
    if role == "user":
        return _content_to_text(message.content)
    if role == "assistant":
        return _assistant_content_to_text(message.content)
    if role == "toolResult":
        return "\n".join([message.tool_name, *(_content_to_text([block]) for block in message.content)])
    return ""


def _joined_length(messages: Sequence[str], count: int | None = None) -> int:
    count = len(messages) if count is None else count
    length = (count - 1) * 2 if count > 0 else 0
    for index in range(count):
        length += len(messages[index])
    return length


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def _common_prompt_prefix_length(previous: Sequence[str], current: Sequence[str]) -> int:
    index = 0
    while index < len(previous) and index < len(current) and previous[index] == current[index]:
        index += 1

    def rest(messages: Sequence[str]) -> str:
        if index == len(messages):
            return ""
        return ("\n\n" if index > 0 else "") + "\n\n".join(messages[index:])

    return _joined_length(previous, index) + _common_prefix_length(rest(previous), rest(current))


def _with_usage_estimate(
    message: AssistantMessage,
    context: TranscriptContext,
    options: Mapping[str, Any] | None,
    prompt_cache: dict[str, list[str]],
) -> AssistantMessage:
    prompt = [f"{m.role}:{_message_to_text(m)}" for m in context.messages]
    prompt_length = _joined_length(prompt)
    prompt_tokens = math.ceil(prompt_length / 4)
    output_tokens = _estimate_tokens(_assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.get("session_id") if options else None
    if session_id and (options or {}).get("cache_retention") != "none":
        previous = prompt_cache.get(session_id)
        if previous is not None:
            cached_chars = _common_prompt_prefix_length(previous, prompt)
            cache_read = math.ceil(cached_chars / 4)
            cache_write = math.ceil((prompt_length - cached_chars) / 4)
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt
    message.usage = Usage(
        input_tokens,
        output_tokens,
        cache_read,
        cache_write,
        input_tokens + output_tokens + cache_read + cache_write,
        UsageCost(),
    )
    return message


def _split_by_token_size(text: str, min_size: int, max_size: int) -> list[str]:
    chunks: list[str] = []
    index = 0
    while index < len(text):
        token_size = min_size + int(random.random() * (max_size - min_size + 1))
        char_size = max(1, token_size * 4)
        chunks.append(text[index : index + char_size])
        index += char_size
    return chunks or [""]


def _clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    if cloned.usage is None:
        cloned.usage = Usage()
    return cloned


def _deferred_message(model: Model, handle: DeferredHandle) -> AssistantMessage:
    return AssistantMessage(
        [],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="deferred",
        deferred=handle,
    )


def _error_message(error: Any, api: str, provider: str, model_id: str) -> AssistantMessage:
    return AssistantMessage(
        [],
        api=api,
        provider=provider,
        model=model_id,
        usage=Usage(),
        stop_reason="error",
        error_message=str(error),
    )


def _aborted(partial: AssistantMessage) -> AssistantMessage:
    aborted = partial.shallow_copy()
    aborted.stop_reason = "aborted"
    aborted.error_message = "Request was aborted"
    aborted.timestamp = now_ms()
    return aborted


async def _schedule_chunk(chunk: str, tokens_per_second: float | None) -> None:
    if not tokens_per_second or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    await asyncio.sleep(_estimate_tokens(chunk) / tokens_per_second)


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_size: int,
    max_size: int,
    tokens_per_second: float | None,
    signal: Any,
) -> None:
    partial = message.shallow_copy()
    partial.content = []
    partial.stop_reason = "pending"

    def abort_now() -> bool:
        if signal is not None and signal.aborted:
            aborted = _aborted(partial)
            stream.push(ErrorEvent("aborted", aborted))
            stream.end(aborted)
            return True
        return False

    if abort_now():
        return
    stream.push(StartEvent(partial.shallow_copy()))

    for index, block in enumerate(message.content):
        if abort_now():
            return
        if block.type == "thinking":
            current = ThinkingContent("")
            partial.content.append(current)
            stream.push(ThinkingStartEvent(index, partial.shallow_copy()))
            for chunk in _split_by_token_size(block.thinking, min_size, max_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if abort_now():
                    return
                append_thinking(current, chunk)
                stream.push(ThinkingDeltaEvent(index, chunk, partial.shallow_copy()))
            stream.push(ThinkingEndEvent(index, block.thinking, partial.shallow_copy()))
            continue
        if block.type == "text":
            current_text = TextContent("")
            partial.content.append(current_text)
            stream.push(TextStartEvent(index, partial.shallow_copy()))
            for chunk in _split_by_token_size(block.text, min_size, max_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if abort_now():
                    return
                append_text(current_text, chunk)
                stream.push(TextDeltaEvent(index, chunk, partial.shallow_copy()))
            stream.push(TextEndEvent(index, block.text, partial.shallow_copy()))
            continue
        current_call = ToolCall(block.id, block.name, {})
        partial.content.append(current_call)
        stream.push(ToolCallStartEvent(index, partial.shallow_copy()))
        for chunk in _split_by_token_size(json.dumps(block.arguments, separators=(",", ":")), min_size, max_size):
            await _schedule_chunk(chunk, tokens_per_second)
            if abort_now():
                return
            stream.push(ToolCallDeltaEvent(index, chunk, partial.shallow_copy()))
        current_call.arguments = block.arguments
        stream.push(ToolCallEndEvent(index, block, partial.shallow_copy()))

    if message.stop_reason == "pending":
        raise RuntimeError("Faux response ended without a stop reason")
    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(message.stop_reason, message))  # type: ignore[arg-type]
        stream.end(message)
        return
    stream.push(DoneEvent(message.stop_reason, message))  # type: ignore[arg-type]
    stream.end(message)


@dataclass(slots=True)
class _DeferredEntry:
    handle: DeferredHandle
    step: Any
    context: TranscriptContext
    options: dict[str, Any] | None
    model: Model
    pending_fetches: int
    cancelled: bool = False
    final: AssistantMessage | None = None


class _FauxCore:
    def __init__(
        self,
        *,
        api: str | None = None,
        provider: str | None = None,
        models: Sequence[FauxModelDefinition | Mapping[str, Any]] | None = None,
        deferred: Mapping[str, Any] | None = None,
        tokens_per_second: float | None = None,
        token_size: Mapping[str, int] | None = None,
    ) -> None:
        self.api = api or _random_id(_DEFAULT_API)
        self.provider = provider or _DEFAULT_PROVIDER
        sizes = token_size or {}
        max_size_option = sizes.get("max", _DEFAULT_MAX_TOKEN_SIZE)
        self.min_size = max(1, min(sizes.get("min", _DEFAULT_MIN_TOKEN_SIZE), max_size_option))
        self.max_size = max(self.min_size, max_size_option)
        self.tokens_per_second = tokens_per_second
        self.deferred_options = dict(deferred or {})
        self.state = FauxProviderState()
        self._pending: list[Any] = []
        self._prompt_cache: dict[str, list[str]] = {}
        self._deferred: dict[str, _DeferredEntry] = {}
        definitions = [
            definition if isinstance(definition, FauxModelDefinition) else FauxModelDefinition(**dict(definition))
            for definition in (models or [])
        ] or [
            FauxModelDefinition(
                id=_DEFAULT_MODEL_ID,
                name=_DEFAULT_MODEL_NAME,
                input=["text", "image"],
                context_window=128_000,
                max_tokens=16_384,
            )
        ]
        self.models: list[Model] = [
            Model(
                id=definition.id,
                name=definition.name or definition.id,
                api=self.api,
                provider=self.provider,
                base_url=_DEFAULT_BASE_URL,
                reasoning=definition.reasoning,
                input=list(definition.input or ["text", "image"]),
                input_limits=definition.input_limits,
                cost=definition.cost or ModelCost(),
                context_window=definition.context_window,
                max_tokens=definition.max_tokens,
            )
            for definition in definitions
        ]

    def get_model(self, model_id: str | None = None) -> Model | None:
        if not model_id:
            return self.models[0]
        return next((model for model in self.models if model.id == model_id), None)

    async def _resolve(
        self, step: Any, context: TranscriptContext, options: Mapping[str, Any] | None, model: Model
    ) -> AssistantMessage:
        resolved = await maybe_await(step(context, options, self.state, model)) if callable(step) else step
        return _with_usage_estimate(
            _clone_message(resolved, self.api, self.provider, model.id), context, options, self._prompt_cache
        )

    def stream(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()
        step = self._pending.pop(0) if self._pending else None
        self.state.call_count += 1
        options = dict(options) if options else None

        async def run() -> None:
            try:
                on_response = options.get("on_response") if options else None
                if on_response is not None:
                    await maybe_await(on_response(ProviderResponse(200, {}), model))
                if step is None:
                    message = _error_message("No more faux responses queued", self.api, self.provider, model.id)
                    message = _with_usage_estimate(message, context, options, self._prompt_cache)
                    outer.push(ErrorEvent("error", message))
                    outer.end(message)
                    return
                signal = options.get("signal") if options else None
                if options and options.get("deferred"):
                    handle = DeferredHandle(model.provider, model.id, model.api, _random_id("deferred"))
                    if self.deferred_options.get("poll_after_ms") is not None:
                        handle.poll_after_ms = self.deferred_options["poll_after_ms"]
                    self._deferred[handle.id] = _DeferredEntry(
                        handle=handle,
                        step=step,
                        context=context,
                        options=options,
                        model=model,
                        pending_fetches=max(0, int(self.deferred_options.get("pending_fetches", 0))),
                    )
                    await _stream_with_deltas(
                        outer,
                        _deferred_message(model, handle),
                        self.min_size,
                        self.max_size,
                        self.tokens_per_second,
                        signal,
                    )
                    return
                message = await self._resolve(step, context, options, model)
                await _stream_with_deltas(outer, message, self.min_size, self.max_size, self.tokens_per_second, signal)
            except Exception as error:
                message = _error_message(error, self.api, self.provider, model.id)
                outer.push(ErrorEvent("error", message))
                outer.end(message)

        outer.start(run())
        return outer

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        return self.stream(model, context, options)

    def fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()
        self.state.deferred_fetch_count += 1

        async def run() -> None:
            try:
                on_response = options.get("on_response") if options else None
                if on_response is not None:
                    await maybe_await(on_response(ProviderResponse(200, {}), model))
                entry = self._deferred.get(handle.id)
                if (
                    entry is None
                    or entry.handle.provider != handle.provider
                    or entry.handle.model_id != handle.model_id
                    or entry.handle.api != handle.api
                ):
                    raise RuntimeError(f"Unknown faux deferred response: {handle.id}")
                if entry.cancelled:
                    raise RuntimeError(f"Faux deferred response was cancelled: {handle.id}")
                signal = options.get("signal") if options else None
                if entry.pending_fetches > 0:
                    entry.pending_fetches -= 1
                    await _stream_with_deltas(
                        outer,
                        _deferred_message(model, entry.handle),
                        self.min_size,
                        self.max_size,
                        self.tokens_per_second,
                        signal,
                    )
                    return
                if entry.final is None:
                    submission = {
                        key: value
                        for key, value in (entry.options or {}).items()
                        if key not in ("deferred", "signal", "on_response")
                    }
                    try:
                        entry.final = await self._resolve(entry.step, entry.context, submission, entry.model)
                    except Exception as error:
                        entry.final = _error_message(error, self.api, self.provider, entry.model.id)
                await _stream_with_deltas(
                    outer, entry.final, self.min_size, self.max_size, self.tokens_per_second, signal
                )
            except Exception as error:
                message = _error_message(error, self.api, self.provider, model.id)
                outer.push(ErrorEvent("error", message))
                outer.end(message)

        outer.start(run())
        return outer

    async def cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: Mapping[str, Any] | None = None
    ) -> None:
        self.state.cancelled_deferred.append(copy.deepcopy(handle))
        entry = self._deferred.get(handle.id)
        if entry is not None:
            entry.cancelled = True
        on_response = options.get("on_response") if options else None
        if on_response is not None:
            await maybe_await(on_response(ProviderResponse(200, {}), model))

    def set_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        self._pending = list(responses)

    def append_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        self._pending.extend(responses)

    def get_pending_response_count(self) -> int:
        return len(self._pending)


def create_faux_core(**options: Any) -> _FauxCore:
    return _FauxCore(**options)


class FauxProviderHandle:
    """Handle returned by :func:`faux_provider` for scripting responses."""

    def __init__(self, core: _FauxCore, provider: Provider) -> None:
        self._core = core
        self.provider = provider
        self.api = core.api
        self.models = core.models
        self.state = core.state

    @overload
    def get_model(self) -> Model: ...

    @overload
    def get_model(self, model_id: str) -> Model | None: ...

    def get_model(self, model_id: str | None = None) -> Model | None:
        """The first faux model, or the one with ``model_id`` (``None`` if unknown)."""
        return self._core.get_model(model_id)

    def set_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        """Replace the remaining response queue."""
        self._core.set_responses(responses)

    def append_responses(self, responses: Sequence[FauxResponseStep]) -> None:
        self._core.append_responses(responses)

    def get_pending_response_count(self) -> int:
        return self._core.get_pending_response_count()


def faux_provider(
    *,
    api: str | None = None,
    provider: str | None = None,
    models: Sequence[FauxModelDefinition | Mapping[str, Any]] | None = None,
    deferred: Mapping[str, Any] | None = None,
    tokens_per_second: float | None = None,
    token_size: Mapping[str, int] | None = None,
) -> FauxProviderHandle:
    """Create an in-memory provider with scripted responses.

    Responses are consumed in request start order. When the queue is empty the
    request fails with ``"No more faux responses queued"``. Usage is estimated at
    about 4 characters per token; prompt caching is simulated when ``session_id``
    is set. ``tokens_per_second`` paces chunk delivery in real time.
    """
    core = _FauxCore(
        api=api,
        provider=provider,
        models=models,
        deferred=deferred,
        tokens_per_second=tokens_per_second,
        token_size=token_size,
    )

    async def resolve(**_: Any) -> AuthResult:
        return AuthResult(ModelAuth())

    created = create_provider(
        id=core.provider,
        auth=ProviderAuth(api_key=ApiKeyAuth("Faux", resolve)),
        models=core.models,
        api=core,
    )
    return FauxProviderHandle(core, created)
