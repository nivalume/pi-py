# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""OpenAI Chat Completions API (``openai-completions``).

Port of pi's ``api/openai-completions.ts``. Talks to ``{base_url}/chat/completions``
directly over HTTP with the JSON body and headers the OpenAI SDK would send, and
covers the many OpenAI-compatible servers through ``model.compat`` (auto-detected
from the provider id and base URL when unset).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypedDict

from ..abort import AbortError, abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ProviderResponse,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.http import ProviderHttpError, build_headers, iter_sse, send_request
from ..utils.json_parse import StreamingJsonParser, parse_tool_call_arguments
from ..utils.misc import get_pi_user_agent, get_provider_env_value, headers_to_record, sanitize_surrogates, short_hash
from ..utils.text import append_text, append_thinking, get_system_message_text, render_system_message_update
from ..utils.transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .simple_options import (
    build_base_options,
    clamp_thinking_budget_to_answer_room,
    resolve_sampling_params,
    thinking_budget_for_level,
)
from .transform_messages import transform_messages

__all__ = [
    "OpenAICompletionsOptions",
    "ConvertCompletionsMessagesOptions",
    "ResolvedOpenAICompletionsCompat",
    "stream",
    "stream_simple",
    "build_params",
    "convert_messages",
    "convert_tools",
    "detect_compat",
    "get_compat",
]


class OpenAICompletionsOptions(StreamOptions, total=False):
    """Options for :func:`stream` (pi's ``OpenAICompletionsOptions``)."""

    #: ``"auto"``, ``"none"``, ``"required"`` or a ``{"type": "function", ...}`` object.
    tool_choice: str | dict[str, Any]
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    #: Token budgets per thinking level. Used when ``compat.thinkingTokenBudgetField`` or
    #: ``compat.supportsThinkingTokenBudget`` is set, or by ``{"$var": "thinking.budget"}``.
    thinking_budgets: ThinkingBudgets


class ConvertCompletionsMessagesOptions(TypedDict, total=False):
    #: Grammar tool name -> its single string input property.
    grammar_tool_input_properties: Mapping[str, str]


#: Fully resolved ``OpenAICompletionsCompat`` (camelCase keys, as in pi).
ResolvedOpenAICompletionsCompat: TypeAlias = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
_REPLAY_REASONING_FIELDS = frozenset(("reasoning", "reasoning_content", "reasoning_text"))
# Characters removed by JavaScript's String.prototype.trim().
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


class _Undefined:
    """JavaScript ``undefined``: distinct from JSON ``null`` (``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "undefined"


_UNDEFINED: Any = _Undefined()


# ---------------------------------------------------------------------------
# Small JavaScript-semantics helpers
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    """JavaScript truthiness (empty objects and arrays are truthy)."""
    if value is None or value is False or value is _UNDEFINED:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (int, float)):
        return value != 0 and value == value
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _prop(obj: Any, key: str) -> Any:
    """``obj?.[key]`` for parsed JSON (``None`` for missing keys and non-objects)."""
    return obj.get(key) if isinstance(obj, dict) else None


def _js_trim(text: str) -> str:
    return text.strip(_JS_WHITESPACE)


def _js_string(value: Any) -> str:
    """JavaScript ``String(value)`` for parsed JSON values."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ",".join("" if item is None else _js_string(item) for item in value)
    return "[object Object]"


def _json_compact(value: Any) -> str:
    """``JSON.stringify(value)``."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _level_value(model: Model, level: str | None) -> Any:
    """``model.thinkingLevelMap?.[level]``: ``_UNDEFINED`` when absent, ``None`` for an explicit null."""
    level_map = model.thinking_level_map
    if level_map is None or level is None or level not in level_map:
        return _UNDEFINED
    return level_map[level]


def _level_value_or(model: Model, level: str | None, fallback: Any) -> Any:
    """``model.thinkingLevelMap?.[level] ?? fallback``."""
    value = _level_value(model, level)
    return fallback if value is None or value is _UNDEFINED else value


def _has_nonblank_header(headers: Mapping[str, Any] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(
        key.lower() == expected and isinstance(value, str) and bool(_js_trim(value)) for key, value in headers.items()
    )


def _get_client_api_key(provider: str, api_key: str | None, headers: Mapping[str, Any] | None) -> str:
    if api_key:
        return api_key
    if _has_nonblank_header(headers, "authorization") or _has_nonblank_header(headers, "cf-aig-authorization"):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def _has_tool_history(messages: Sequence[Message]) -> bool:
    """Whether the transcript contains tool calls or tool results.

    Anthropic (via a LiteLLM-style proxy) requires the ``tools`` param when the
    conversation includes tool calls or tool role messages.
    """
    for message in messages:
        role = message.role
        if role == "toolResult":
            return True
        if role == "assistant" and any(block.type == "toolCall" for block in message.content):
            return True
    return False


def _resolve_cache_retention(cache_retention: str | None, env: Mapping[str, str] | None) -> str:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


# ---------------------------------------------------------------------------
# reasoning_details (OpenRouter-style structured reasoning replay data)
# ---------------------------------------------------------------------------


def _is_reasoning_detail(detail: Any) -> bool:
    if not isinstance(detail, dict):
        return False
    if "id" in detail and detail["id"] is not None and not isinstance(detail["id"], str):
        return False
    if "format" in detail and not isinstance(detail["format"], str):
        return False
    if "index" in detail and not _is_number(detail["index"]):
        return False
    kind = detail.get("type")
    if kind == "reasoning.summary":
        return isinstance(detail.get("summary"), str)
    if kind == "reasoning.encrypted":
        return isinstance(detail.get("data"), str)
    if kind == "reasoning.text":
        signature = detail.get("signature")
        return isinstance(detail.get("text"), str) and (signature is None or isinstance(signature, str))
    return False


def _parse_reasoning_details(signature: str | None) -> list[dict[str, Any]] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, list) and parsed and all(_is_reasoning_detail(detail) for detail in parsed):
        return parsed
    return None


def _parse_legacy_encrypted_reasoning_detail(signature: str | None) -> dict[str, Any] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except (TypeError, ValueError):
        return None
    if (
        _is_reasoning_detail(parsed)
        and parsed["type"] == "reasoning.encrypted"
        and isinstance(parsed.get("id"), str)
        and len(parsed["id"]) > 0
        and len(parsed["data"]) > 0
    ):
        return parsed
    return None


def _assign_from(target: dict[str, Any], key: str, source: Mapping[str, Any]) -> None:
    # ``target[key] = source[key]`` where a missing source key assigns ``undefined``.
    target[key] = source.get(key, _UNDEFINED)


def _nullish(value: Any) -> bool:
    return value is None or value is _UNDEFINED


def _fill_missing_common_fields(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    if _nullish(target.get("id")):
        _assign_from(target, "id", source)
    if not _truthy(target.get("format")):
        _assign_from(target, "format", source)
    if _nullish(target.get("index")):
        _assign_from(target, "index", source)


def _append_reasoning_detail(details: list[dict[str, Any]], detail: dict[str, Any]) -> None:
    last = details[-1] if details else None
    kind = detail["type"]
    if kind == "reasoning.text" and last is not None and last.get("type") == "reasoning.text":
        last["text"] += detail["text"]
        if not _truthy(last.get("signature")):
            _assign_from(last, "signature", detail)
        _fill_missing_common_fields(last, detail)
        return
    if kind == "reasoning.summary" and last is not None and last.get("type") == "reasoning.summary":
        last["summary"] += detail["summary"]
        _fill_missing_common_fields(last, detail)
        return
    details.append(dict(detail))


def _serialize_reasoning_details(details: list[dict[str, Any]]) -> str:
    # JSON.stringify drops ``undefined`` members but keeps the key order of the rest.
    return fastjson.dumps([{key: value for key, value in d.items() if value is not _UNDEFINED} for d in details])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a Chat Completions response as pi assistant message events."""
    options = dict(options or {})
    out = AssistantMessageEventStream()
    normalized = resolve_transcript(context, get_compat(model).get("supportsMidConvoSystemMessages"))
    out.start(_run(model, normalized, options, out))
    return out


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning``, ``tool_choice``, ``thinking_budgets``).

    Raises ``RuntimeError`` synchronously when no API key (or authorization header) is given.
    """
    options = options or {}
    _get_client_api_key(model.provider, options.get("api_key"), options.get("headers"))
    base = build_base_options(model, context, options, options.get("api_key"))
    tool_choice = options.get("tool_choice")
    if tool_choice is not None:
        base["tool_choice"] = tool_choice
    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    if clamped and clamped != "off":
        base["reasoning_effort"] = clamped
    thinking_budgets = options.get("thinking_budgets")
    if thinking_budgets is not None:
        base["thinking_budgets"] = thinking_budgets
    return stream(model, context, base)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _chat_completions_url(base_url: str) -> str:
    base = base_url or _DEFAULT_BASE_URL
    return (base[:-1] if base.endswith("/") else base) + "/chat/completions"


def _build_request_headers(
    model: Model,
    context: TranscriptContext,
    api_key: str,
    options_headers: Mapping[str, str | None] | None,
    session_id: str | None,
    compat: ResolvedOpenAICompletionsCompat,
) -> dict[str, str]:
    headers: dict[str, str | None] = {"User-Agent": get_pi_user_agent()}
    if model.headers:
        headers.update(model.headers)
    if model.provider == "github-copilot":
        messages = context.messages
        headers.update(build_copilot_dynamic_headers(messages=messages, has_images=has_copilot_vision_input(messages)))
    if session_id and compat.get("sendSessionAffinityHeaders"):
        if compat.get("sessionAffinityFormat") == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.get("sessionAffinityFormat") == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id
    # Options headers last so they can override defaults.
    if options_headers:
        headers.update(options_headers)
    # Same precedence as the OpenAI SDK: SDK defaults < auth < client default headers < body headers.
    return build_headers(
        {"Accept": "application/json"},
        {"Authorization": f"Bearer {api_key}"},
        headers,
        {"Content-Type": "application/json"},
    )


def _sdk_error_message(error: Any) -> str:
    """The OpenAI SDK's ``APIError`` message for an error without HTTP status."""
    message = _prop(error, "message")
    if _truthy(message):
        text = message if isinstance(message, str) else _json_compact(message)
    elif _truthy(error):
        text = _json_compact(error)
    else:
        text = ""
    return text or "(no status code or body)"


class _CustomInput:
    __slots__ = ("buffer", "property")

    def __init__(self, property: str) -> None:
        self.property = property
        self.buffer = GrammarToolInputJsonBuffer()


class _StreamProcessor:
    """Normalizes Chat Completions chunks into ``output`` and pushes pi events.

    Streaming scratch state (argument parsers, grammar input buffers, stream
    indexes) is keyed by block identity so the content blocks stay clean.
    """

    __slots__ = (
        "custom_inputs",
        "grammar_properties",
        "has_finish_reason",
        "index_of",
        "model",
        "out",
        "output",
        "parsers",
        "reasoning_details",
        "stream_indexes",
        "text_block",
        "thinking_block",
        "tool_by_id",
        "tool_by_index",
    )

    def __init__(
        self,
        model: Model,
        out: AssistantMessageEventStream,
        output: AssistantMessage,
        grammar_properties: Mapping[str, str],
    ) -> None:
        self.model = model
        self.out = out
        self.output = output
        self.grammar_properties = grammar_properties
        self.text_block: TextContent | None = None
        self.thinking_block: ThinkingContent | None = None
        self.has_finish_reason = False
        self.tool_by_index: dict[Any, ToolCall] = {}
        self.tool_by_id: dict[Any, ToolCall] = {}
        self.index_of: dict[int, int] = {}
        self.parsers: dict[int, StreamingJsonParser] = {}
        self.custom_inputs: dict[int, _CustomInput] = {}
        self.stream_indexes: dict[int, Any] = {}
        # ``reasoning_details`` are replay metadata, not user-visible stream deltas. Keep
        # them in memory during streaming and serialize once when the block is finalized.
        self.reasoning_details: list[dict[str, Any]] | None = None

    # -- blocks ---------------------------------------------------------------

    def _add_block(self, block: Any) -> int:
        content = self.output.content
        index = len(content)
        content.append(block)
        self.index_of[id(block)] = index
        return index

    def apply_reasoning_details(self, block: ThinkingContent) -> None:
        if self.reasoning_details is not None:
            block.thinking_signature = _serialize_reasoning_details(self.reasoning_details)

    def ensure_text_block(self) -> tuple[TextContent, int]:
        block = self.text_block
        if block is None:
            block = TextContent("")
            self.text_block = block
            index = self._add_block(block)
            self.out.push(TextStartEvent(index, self.output))
            return block, index
        return block, self.index_of[id(block)]

    def ensure_thinking_block(self, signature: str) -> tuple[ThinkingContent, int]:
        block = self.thinking_block
        if block is None:
            block = ThinkingContent("", thinking_signature=signature)
            self.thinking_block = block
            index = self._add_block(block)
            self.out.push(ThinkingStartEvent(index, self.output))
            return block, index
        return block, self.index_of[id(block)]

    def _custom_input_value(self, block: ToolCall) -> str:
        custom = self.custom_inputs.get(id(block))
        if custom is None:
            return ""
        value = block.arguments.get(custom.property) if isinstance(block.arguments, dict) else None
        return value if isinstance(value, str) else ""

    def _append_custom_input(self, block: ToolCall, next_input: str, close: bool) -> str | None:
        custom = self.custom_inputs.get(id(block))
        if custom is None:
            return None
        delta = append_grammar_tool_input_json_delta(custom.buffer, custom.property, next_input, close)
        block.arguments = {custom.property: next_input}
        return delta

    def ensure_tool_call_block(self, tool_call: dict[str, Any]) -> ToolCall:
        index = tool_call.get("index")
        stream_index = index if _is_number(index) else None
        function = tool_call.get("function")
        custom = tool_call.get("custom")
        name = _prop(function, "name")
        if name is None:
            name = _prop(custom, "name")
        if name is None:
            name = ""
        tool_id = tool_call.get("id")
        is_custom = _truthy(custom) and not _truthy(function)

        block = self.tool_by_index.get(stream_index) if stream_index is not None else None
        if block is None and _truthy(tool_id):
            block = self.tool_by_id.get(tool_id)
        if block is None:
            # Note: the "input" fallback should not be taken. In case the LLM makes up a
            # tool we don't know about, we at least have a place to stash its input.
            custom_property = self.grammar_properties.get(name, "input") if is_custom else None
            block = ToolCall(tool_id if _truthy(tool_id) else "", name, {custom_property: ""} if is_custom else {})
            key = id(block)
            if custom_property is not None:
                self.custom_inputs[key] = _CustomInput(custom_property)
            else:
                self.parsers[key] = StreamingJsonParser()
            if stream_index is not None:
                self.stream_indexes[key] = stream_index
                self.tool_by_index[stream_index] = block
            if _truthy(tool_id):
                self.tool_by_id[tool_id] = block
            content_index = self._add_block(block)
            self.out.push(ToolCallStartEvent(content_index, self.output))
        key = id(block)
        if stream_index is not None and key not in self.stream_indexes:
            self.stream_indexes[key] = stream_index
            self.tool_by_index[stream_index] = block
        if _truthy(tool_id):
            self.tool_by_id[tool_id] = block
        if not block.name and name:
            block.name = name
        if is_custom and key not in self.custom_inputs:
            custom_property = self.grammar_properties.get(block.name, "input")
            block.arguments = {custom_property: ""}
            self.custom_inputs[key] = _CustomInput(custom_property)
            self.parsers.pop(key, None)
        return block

    def finish_block(self, block: Any) -> None:
        content_index = self.index_of.get(id(block))
        if content_index is None:
            return
        out = self.out
        output = self.output
        kind = block.type
        if kind == "text":
            out.push(TextEndEvent(content_index, block.text, output))
        elif kind == "thinking":
            self.apply_reasoning_details(block)
            out.push(ThinkingEndEvent(content_index, block.thinking, output))
        elif kind == "toolCall":
            key = id(block)
            if key in self.custom_inputs:
                delta = self._append_custom_input(block, self._custom_input_value(block), True)
                if delta is not None:
                    out.push(ToolCallDeltaEvent(content_index, delta, output))
            else:
                parser = self.parsers.get(key)
                arguments, arguments_error = parse_tool_call_arguments(parser.text if parser is not None else None)
                block.arguments = arguments
                if arguments_error is not None:
                    block.arguments_error = arguments_error
            # Finalize and drop the scratch buffers so replay only carries parsed arguments.
            self.parsers.pop(key, None)
            self.custom_inputs.pop(key, None)
            self.stream_indexes.pop(key, None)
            out.push(ToolCallEndEvent(content_index, block, output))

    def finish(self) -> None:
        for block in list(self.output.content):
            self.finish_block(block)

    # -- chunks ---------------------------------------------------------------

    def handle_chunk(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        output = self.output
        model = self.model
        out = self.out

        # OpenAI documents ChatCompletionChunk.id as the unique chat completion identifier,
        # and each chunk in a streamed completion carries the same id.
        if not output.response_id:
            output.response_id = chunk.get("id")
        chunk_model = chunk.get("model")
        if isinstance(chunk_model, str) and chunk_model and chunk_model != model.id and not output.response_model:
            output.response_model = chunk_model
        usage = chunk.get("usage")
        if _truthy(usage):
            output.usage = _parse_chunk_usage(usage, model)

        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            return

        # Fallback: some providers (e.g., Moonshot) return usage in choice.usage
        # instead of the standard chunk.usage.
        if not _truthy(usage):
            choice_usage = choice.get("usage")
            if _truthy(choice_usage):
                output.usage = _parse_chunk_usage(choice_usage, model)

        finish_reason = choice.get("finish_reason")
        if _truthy(finish_reason):
            output.raw_stop_reason = finish_reason
            stop_reason, error_message = _map_stop_reason(finish_reason)
            output.stop_reason = stop_reason  # type: ignore[assignment]
            if error_message:
                output.error_message = error_message
            self.has_finish_reason = True

        delta = choice.get("delta")
        if not _truthy(delta) or not isinstance(delta, dict):
            return

        # Handle reasoning before content: some proxies (e.g. LiteLLM) aggregate reasoning and
        # answer into one delta, and the thinking block must precede the text block.
        # Some endpoints return reasoning in reasoning_content (llama.cpp), or reasoning (other
        # OpenAI-compatible endpoints). Use the first non-empty reasoning field to avoid
        # duplication (e.g., chutes.ai returns both reasoning_content and reasoning).
        for field in _REASONING_FIELDS:
            reasoning = delta.get(field)
            if isinstance(reasoning, str) and reasoning:
                signature = "reasoning_content" if model.provider == "opencode-go" and field == "reasoning" else field
                thinking_block, thinking_index = self.ensure_thinking_block(signature)
                append_thinking(thinking_block, reasoning)
                out.push(ThinkingDeltaEvent(thinking_index, reasoning, output))
                break

        reasoning_details = delta.get("reasoning_details")
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                if not _is_reasoning_detail(detail):
                    continue
                self.ensure_thinking_block("")
                if self.reasoning_details is None:
                    self.reasoning_details = []
                # Keep provider replay data in the existing signature slot. OpenRouter streams
                # reasoning_details as deltas: consecutive text/summary deltas are merged into
                # logical entries, while encrypted entries remain opaque and discrete.
                _append_reasoning_detail(self.reasoning_details, detail)

        content = delta.get("content")
        if isinstance(content, str) and content:
            text_block, text_index = self.ensure_text_block()
            append_text(text_block, content)
            out.push(TextDeltaEvent(text_index, content, output))

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                self._handle_tool_call_delta(tool_call)

    def _handle_tool_call_delta(self, tool_call: dict[str, Any]) -> None:
        block = self.ensure_tool_call_block(tool_call)
        tool_id = tool_call.get("id")
        if not block.id and _truthy(tool_id):
            block.id = tool_id
            self.tool_by_id[tool_id] = block
        function = tool_call.get("function")
        custom = tool_call.get("custom")
        name = _prop(function, "name")
        if name is None:
            name = _prop(custom, "name")
        if not block.name and _truthy(name):
            block.name = name

        key = id(block)
        delta = ""
        arguments = _prop(function, "arguments")
        if _truthy(arguments):
            delta = arguments if isinstance(arguments, str) else _js_string(arguments)
            parser = self.parsers.get(key)
            if parser is None:
                parser = StreamingJsonParser()
                self.parsers[key] = parser
            parser.feed(delta)
            block.arguments = parser.value()
        else:
            custom_input = _prop(custom, "input")
            if _truthy(custom_input):
                next_input = self._custom_input_value(block) + _js_string(custom_input)
                custom_delta = self._append_custom_input(block, next_input, False)
                delta = custom_delta if custom_delta is not None else ""
        self.out.push(ToolCallDeltaEvent(self.index_of[key], delta, self.output))


async def _run(
    model: Model,
    context: TranscriptContext,
    options: dict[str, Any],
    out: AssistantMessageEventStream,
) -> None:
    output = AssistantMessage(
        [], api=model.api, provider=model.provider, model=model.id, usage=Usage(), stop_reason="pending"
    )
    signal = options.get("signal")
    processor: _StreamProcessor | None = None
    try:
        api_key = _get_client_api_key(model.provider, options.get("api_key"), options.get("headers"))
        compat = get_compat(model)
        grammar_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), bool(compat.get("supportsOpenAIGrammarTools"))
        )
        cache_retention = _resolve_cache_retention(options.get("cache_retention"), options.get("env"))
        cache_session_id = None if cache_retention == "none" else options.get("session_id")
        headers = _build_request_headers(model, context, api_key, options.get("headers"), cache_session_id, compat)
        params: Any = build_params(model, context, options, compat, cache_retention, grammar_properties)
        on_payload = options.get("on_payload")
        if on_payload is not None:
            next_params = await maybe_await(on_payload(params, model))
            if next_params is not None:
                params = next_params

        url = _chat_completions_url(model.base_url)
        try:
            async with abort_scope(signal):
                response = await send_request(
                    "POST", url, headers=headers, body=params, options=options, error_style="openai"
                )
        except AbortError as error:
            if signal is not None and signal.aborted:
                raise AbortError("Request aborted") from error
            raise

        processor = _StreamProcessor(model, out, output, grammar_properties)
        try:
            on_response = options.get("on_response")
            if on_response is not None:
                await maybe_await(
                    on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
                )
            out.push(StartEvent(output))
            on_event = options.get("on_provider_stream_event")
            try:
                async with abort_scope(signal):
                    async for sse in iter_sse(response):
                        data = sse.data
                        if data.startswith("[DONE]"):
                            break
                        chunk = sse.json()
                        event_name = sse.event
                        if event_name is not None and event_name.startswith("thread."):
                            chunk = {"event": event_name, "data": chunk}
                        elif isinstance(chunk, dict) and _truthy(chunk.get("error")):
                            error_body = chunk["error"]
                            raise ProviderHttpError(
                                _sdk_error_message(error_body), headers=response.headers, error=error_body
                            )
                        if on_event is not None:
                            await maybe_await(on_event(chunk, model))
                        processor.handle_chunk(chunk)
            except AbortError:
                # The OpenAI SDK ends the chunk iteration silently when the request is aborted.
                if signal is None or not signal.aborted:
                    raise
        finally:
            await response.aclose()

        processor.finish()
        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "aborted":
            raise RuntimeError("Request was aborted")
        supports_finish_reason = compat.get("supportsFinishReason")
        if not processor.has_finish_reason and not supports_finish_reason:
            output.stop_reason = "toolUse" if any(block.type == "toolCall" for block in output.content) else "stop"
        if output.stop_reason == "error":
            raise RuntimeError(output.error_message or "Provider returned an error stop reason")
        if (supports_finish_reason and not processor.has_finish_reason) or output.stop_reason == "pending":
            raise RuntimeError("Stream ended without finish_reason")

        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        if processor is not None and processor.thinking_block is not None:
            processor.apply_reasoning_details(processor.thinking_block)
        output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
        message = format_provider_error(normalize_provider_error(error))
        # Some providers via OpenRouter give additional information in this field.
        # normalize_provider_error already stringifies the parsed body (error.error) into the
        # message, so only append the raw metadata when it is not already present.
        raw_metadata = _prop(_prop(getattr(error, "error", None), "metadata"), "raw")
        if _truthy(raw_metadata):
            raw_text = _js_string(raw_metadata)
            if raw_text not in message:
                message += f"\n{raw_text}"
        output.error_message = message
        out.push(ErrorEvent(output.stop_reason, output))
        out.end()


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------


def build_params(
    model: Model,
    context: TranscriptContext,
    options: Mapping[str, Any] | None = None,
    compat: ResolvedOpenAICompletionsCompat | None = None,
    cache_retention: str | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the ``chat.completions.create`` request body (``stream: true``)."""
    options = options or {}
    if compat is None:
        compat = get_compat(model)
    if cache_retention is None:
        cache_retention = _resolve_cache_retention(options.get("cache_retention"), options.get("env"))
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), bool(compat.get("supportsOpenAIGrammarTools"))
        )
    transcript_tools = resolve_transcript_tools(
        context.messages,
        compat.get("supportsMidConvoSystemMessages") is True and compat.get("supportsMidConvoToolAdditions") is True,
    )
    messages = convert_messages(
        model, context, compat, {"grammar_tool_input_properties": grammar_tool_input_properties}
    )
    cache_control = _get_compat_cache_control(compat, cache_retention)
    long_retention = cache_retention == "long" and _truthy(compat.get("supportsLongCacheRetention"))

    params: dict[str, Any] = {"model": model.id, "messages": messages, "stream": True}
    if ("api.openai.com" in model.base_url and cache_retention != "none") or long_retention:
        prompt_cache_key = clamp_openai_prompt_cache_key(options.get("session_id"))
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key
    if long_retention:
        params["prompt_cache_retention"] = "24h"

    if compat.get("supportsUsageInStreaming") is not False:
        params["stream_options"] = {"include_usage": True}

    if _truthy(compat.get("supportsStore")):
        params["store"] = False

    max_tokens = options.get("max_tokens")
    if _truthy(max_tokens):
        if compat.get("maxTokensField") == "max_tokens":
            # Deprecated by OpenAI, but some OpenAI-compatible providers only accept max_tokens.
            params["max_tokens"] = max_tokens
        else:
            params["max_completion_tokens"] = max_tokens

    temperature = options.get("temperature")
    if temperature is not None:
        params["temperature"] = temperature

    if transcript_tools.request_tools:
        params["tools"] = convert_tools(transcript_tools.request_tools, compat)
        if _truthy(compat.get("zaiToolStream")):
            params["tool_stream"] = True
    elif _has_tool_history(context.messages):
        # Anthropic (via LiteLLM/proxy) requires tools param when conversation has tool_calls/tool_results.
        params["tools"] = []

    if cache_control is not None:
        _apply_anthropic_cache_control(messages, params.get("tools"), cache_control)

    tool_choice = options.get("tool_choice")
    if _truthy(tool_choice):
        params["tool_choice"] = tool_choice

    vllm_priority = compat.get("vllmPriority")
    if vllm_priority is not None:
        params["priority"] = vllm_priority

    thinking_budget_field = _resolve_thinking_token_budget_field(compat)
    thinking_budget = _resolve_clamped_thinking_budget(model, options, params)
    _apply_thinking_params(model, options, compat, params, thinking_budget)

    # Cap reasoning with a top-level budget field. Independent of thinkingFormat: the same
    # server can serve zai, qwen or chat-template models. Reasoning and the answer share
    # max_tokens here, so an uncapped reasoning phase can consume the whole response and
    # leave no answer and no tool call.
    if thinking_budget_field and thinking_budget is not None:
        params[thinking_budget_field] = thinking_budget

    model_compat = model.compat or {}
    # OpenRouter provider routing preferences
    open_router_routing = model_compat.get("openRouterRouting")
    if _truthy(open_router_routing):
        params["provider"] = open_router_routing

    # Vercel AI Gateway provider routing preferences
    vercel_routing = model_compat.get("vercelGatewayRouting")
    if _truthy(vercel_routing):
        only = _prop(vercel_routing, "only")
        order = _prop(vercel_routing, "order")
        if _truthy(only) or _truthy(order):
            gateway_options: dict[str, Any] = {}
            if _truthy(only):
                gateway_options["only"] = only
            if _truthy(order):
                gateway_options["order"] = order
            params["providerOptions"] = {"gateway": gateway_options}

    # Last so model and request sampling parameters override named request fields.
    sampling_params = resolve_sampling_params(
        model, options.get("reasoning_effort") or "off", options.get("sampling_params")
    )
    if sampling_params:
        params.update(sampling_params)

    return params


def _apply_thinking_params(
    model: Model,
    options: Mapping[str, Any],
    compat: ResolvedOpenAICompletionsCompat,
    params: dict[str, Any],
    thinking_budget: int | None,
) -> None:
    effort: str | None = options.get("reasoning_effort") or None
    thinking_format = compat.get("thinkingFormat")
    reasoning = model.reasoning
    supports_effort = _truthy(compat.get("supportsReasoningEffort"))
    level_map = model.thinking_level_map

    if thinking_format == "zai" and reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and supports_effort:
            mapped = _level_value(model, effort)
            value = effort if mapped is _UNDEFINED else mapped
            if isinstance(value, str):
                params["reasoning_effort"] = value
    elif thinking_format == "qwen" and reasoning:
        params["enable_thinking"] = bool(effort)
        if effort and supports_effort:
            value = _level_value_or(model, effort, effort)
            if isinstance(value, str):
                params["reasoning_effort"] = value
    elif thinking_format == "qwen-chat-template" and reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
    elif thinking_format == "chat-template" and reasoning:
        kwargs = _build_chat_template_values(model, options, compat.get("chatTemplateKwargs"), thinking_budget)
        if kwargs is not None:
            params["chat_template_kwargs"] = kwargs
    elif thinking_format == "baseten" and reasoning:
        args = _build_chat_template_values(model, options, compat.get("chatTemplateArgs"), thinking_budget)
        if args is not None:
            params["chat_template_args"] = args
        if supports_effort:
            mapped = _level_value(model, effort) if effort else _level_value(model, "off")
            value = effort if mapped is _UNDEFINED else mapped
            if isinstance(value, str):
                params["reasoning_effort"] = value
    elif thinking_format == "deepseek" and reasoning:
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif _level_value(model, "off") is not None:
            params["thinking"] = {"type": "disabled"}
        if effort and supports_effort:
            params["reasoning_effort"] = _level_value_or(model, effort, effort)
    elif thinking_format == "openrouter" and reasoning:
        # OpenRouter normalizes reasoning across providers via a nested reasoning object.
        if effort:
            params["reasoning"] = {"effort": _level_value_or(model, effort, effort)}
        elif _level_value(model, "off") is not None:
            params["reasoning"] = {"effort": _level_value_or(model, "off", "none")}
    elif thinking_format == "ant-ling" and reasoning and effort:
        value = _level_value(model, effort)
        if isinstance(value, str):
            params["reasoning"] = {"effort": value}
    elif thinking_format == "together" and reasoning:
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and supports_effort:
            params["reasoning_effort"] = _level_value_or(model, effort, effort)
    elif thinking_format == "string-thinking" and reasoning:
        if effort:
            params["thinking"] = _level_value_or(model, effort, effort)
        elif _level_value(model, "off") is not None:
            params["thinking"] = _level_value_or(model, "off", "none")
    elif effort and reasoning and supports_effort:
        # OpenAI-style reasoning_effort
        params["reasoning_effort"] = _level_value_or(model, effort, effort)
    elif not effort and reasoning and supports_effort:
        off_value = level_map.get("off") if level_map else None
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value


def _resolve_thinking_token_budget_field(compat: Mapping[str, Any]) -> str | None:
    field = compat.get("thinkingTokenBudgetField")
    if _truthy(field):
        return field
    if _truthy(compat.get("supportsThinkingTokenBudget")):
        return "thinking_token_budget"
    return None


def _resolve_clamped_thinking_budget(model: Model, options: Mapping[str, Any], params: Mapping[str, Any]) -> int | None:
    effort = options.get("reasoning_effort")
    if not effort or not model.reasoning:
        return None
    ceiling = params.get("max_tokens")
    if ceiling is None:
        ceiling = params.get("max_completion_tokens")
    if ceiling is None:
        ceiling = model.max_tokens
    try:
        budget = clamp_thinking_budget_to_answer_room(
            thinking_budget_for_level(effort, options.get("thinking_budgets")), ceiling
        )
    except (KeyError, TypeError):
        # Unknown levels or budgets yield NaN upstream, which never counts as a budget.
        return None
    return budget if budget > 0 else None


def _build_chat_template_values(
    model: Model, options: Mapping[str, Any], values: Any, thinking_budget: int | None
) -> dict[str, Any] | None:
    if not isinstance(values, Mapping):
        return None
    resolved: dict[str, Any] = {}
    for key, value in values.items():
        result = _resolve_chat_template_value(model, options, value, thinking_budget)
        if result is not _UNDEFINED:
            resolved[key] = result
    return resolved or None


def _resolve_chat_template_value(
    model: Model, options: Mapping[str, Any], value: Any, thinking_budget: int | None
) -> Any:
    if not isinstance(value, dict):
        return value
    effort = options.get("reasoning_effort")
    if not effort and _truthy(value.get("omitWhenOff")):
        return _UNDEFINED
    variable = value.get("$var")
    if variable == "thinking.enabled":
        return bool(effort)
    if variable == "thinking.budget":
        return thinking_budget if thinking_budget is not None else _UNDEFINED
    mapped = _level_value(model, effort) if effort else _level_value(model, "off")
    if mapped is _UNDEFINED:
        return effort if effort else _UNDEFINED
    return mapped if isinstance(mapped, str) else _UNDEFINED


# ---------------------------------------------------------------------------
# Anthropic-style cache_control markers
# ---------------------------------------------------------------------------


def _get_compat_cache_control(compat: Mapping[str, Any], cache_retention: str) -> dict[str, Any] | None:
    if compat.get("cacheControlFormat") != "anthropic" or cache_retention == "none":
        return None
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if cache_retention == "long" and _truthy(compat.get("supportsLongCacheRetention")):
        cache_control["ttl"] = "1h"
    return cache_control


def _apply_anthropic_cache_control(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, cache_control: dict[str, Any]
) -> None:
    # System prompt
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _add_cache_control_to_text_content(message, cache_control)
            break
    # Last tool definition
    if tools:
        tools[-1]["cache_control"] = cache_control
    # Last user, assistant, or tool-result text content
    for message in reversed(messages):
        if message.get("role") in ("user", "assistant", "tool") and _add_cache_control_to_text_content(
            message, cache_control
        ):
            return


def _add_cache_control_to_text_content(message: dict[str, Any], cache_control: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return True
    if not isinstance(content, list):
        return False
    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True
    return False


# ---------------------------------------------------------------------------
# Messages and tools
# ---------------------------------------------------------------------------

_INVALID_ID_CHAR = re.compile(r"[^a-zA-Z0-9_-]")


def _replace_invalid_id_char(match: re.Match[str]) -> str:
    # JavaScript regexes without the u flag replace each UTF-16 code unit.
    return "__" if ord(match.group()) > 0xFFFF else "_"


def _sanitize_id_part(text: str) -> str:
    return _INVALID_ID_CHAR.sub(_replace_invalid_id_char, text)


def _image_url(block: ImageContent) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}


def convert_messages(
    model: Model,
    context: TranscriptContext,
    compat: Mapping[str, Any],
    options: ConvertCompletionsMessagesOptions | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a pi transcript into Chat Completions ``messages``."""
    normalized = resolve_transcript(context, compat.get("supportsMidConvoSystemMessages"))
    grammar_properties: Mapping[str, str] | None = (options or {}).get("grammar_tool_input_properties")
    params: list[dict[str, Any]] = []
    provider = model.provider

    def normalize_tool_call_id(tool_call_id: str, _model: Model, _source: AssistantMessage) -> str:
        # Handle pipe-separated IDs from the OpenAI Responses API: {call_id}|{id}, where {id}
        # can be 400+ chars with special chars (+, /, =). They come from providers like
        # github-copilot, openai-codex and opencode. Multiple tool calls in the same turn can
        # share call_id but differ by item_id, so preserve item-level uniqueness when
        # replaying into Chat Completions, which requires distinct tool call ids.
        if "|" in tool_call_id:
            # Sanitize to allowed chars and truncate to 40 chars (OpenAI limit).
            separator = tool_call_id.index("|")
            call_id = _sanitize_id_part(tool_call_id[:separator])
            item_id = _sanitize_id_part(tool_call_id[separator + 1 :])
            combined = f"{call_id}_{item_id}" if item_id else call_id
            if len(combined) <= 40:
                return combined
            digest = short_hash(tool_call_id)[:8]
            prefix = call_id[: max(1, 40 - len(digest) - 1)]
            return f"{prefix}_{digest}"
        if provider == "openai":
            return tool_call_id[:40] if len(tool_call_id) > 40 else tool_call_id
        return tool_call_id

    transformed = transform_messages(normalized.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized.messages,
        compat.get("supportsMidConvoSystemMessages") is True and compat.get("supportsMidConvoToolAdditions") is True,
    )
    instruction_role = "developer" if model.reasoning and _truthy(compat.get("supportsDeveloperRole")) else "system"
    requires_assistant_after_tool_result = _truthy(compat.get("requiresAssistantAfterToolResult"))
    supports_images = "image" in model.input

    last_role: str | None = None
    i = 0
    count = len(transformed)
    while i < count:
        msg: Any = transformed[i]
        role = msg.role
        # Some providers don't allow user messages directly after tool results.
        # Insert a synthetic assistant message to bridge the gap.
        if requires_assistant_after_tool_result and last_role == "toolResult" and role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if role == "system":
            added_tools = (msg.tools_added or []) if i > 0 and transcript_tools.anchors_additions else []
            if added_tools:
                params.append({"role": "system", "tools": convert_tools(added_tools, compat)})
            text = get_system_message_text(msg) if i == 0 else render_system_message_update(msg)
            if text:
                params.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        if item.text:
                            parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        parts.append(_image_url(item))
                if not parts:
                    i += 1
                    continue
                params.append({"role": "user", "content": parts})
        elif role == "assistant":
            if not _append_assistant_message(params, msg, model, compat, grammar_properties):
                i += 1
                continue
        elif role == "toolResult":
            image_blocks: list[dict[str, Any]] = []
            j = i
            while j < count and transformed[j].role == "toolResult":
                tool_msg: ToolResultMessage = transformed[j]  # type: ignore[assignment]
                text_result = "\n".join(block.text for block in tool_msg.content if block.type == "text")
                has_images = any(block.type == "image" for block in tool_msg.content)
                # Always send tool result with text (or placeholder if only images).
                if _js_trim(text_result):
                    tool_result_text = text_result
                else:
                    tool_result_text = "(see attached image)" if has_images else "(no tool output)"
                tool_result: dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_result_text),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                # Some providers require the 'name' field in tool results.
                if _truthy(compat.get("requiresToolResultName")) and tool_msg.tool_name:
                    tool_result["name"] = tool_msg.tool_name
                params.append(tool_result)
                if has_images and supports_images:
                    image_blocks.extend(_image_url(block) for block in tool_msg.content if block.type == "image")
                j += 1
            i = j
            if image_blocks:
                if requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"
            continue

        last_role = role
        i += 1

    return params


def _append_assistant_message(
    params: list[dict[str, Any]],
    msg: AssistantMessage,
    model: Model,
    compat: Mapping[str, Any],
    grammar_properties: Mapping[str, str] | None,
) -> bool:
    # Some providers don't accept null content, use empty string instead.
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "" if _truthy(compat.get("requiresAssistantAfterToolResult")) else None,
    }
    text_parts = [
        {"type": "text", "text": sanitize_surrogates(block.text)}
        for block in msg.content
        if block.type == "text" and _js_trim(block.text)
    ]
    assistant_text = "".join(part["text"] for part in text_parts)

    thinking_blocks: list[ThinkingContent] = [block for block in msg.content if block.type == "thinking"]  # type: ignore[misc]
    tool_calls: list[ToolCall] = [block for block in msg.content if block.type == "toolCall"]  # type: ignore[misc]
    signed_details = None
    for block in thinking_blocks:
        signed_details = _parse_reasoning_details(block.thinking_signature)
        if signed_details is not None:
            break
    legacy_details = [
        detail
        for detail in (_parse_legacy_encrypted_reasoning_detail(call.thought_signature) for call in tool_calls)
        if detail is not None
    ]
    preserved_details = signed_details if signed_details is not None else (legacy_details or None)

    non_empty_thinking = [block for block in thinking_blocks if _js_trim(block.thinking)]
    if non_empty_thinking:
        if _truthy(compat.get("requiresThinkingAsText")):
            # Convert thinking blocks to plain text (no tags to avoid the model mimicking them).
            thinking_text = "\n\n".join(sanitize_surrogates(block.thinking) for block in non_empty_thinking)
            assistant["content"] = [{"type": "text", "text": thinking_text}, *text_parts]
        else:
            # Always send assistant content as a plain string (OpenAI Chat Completions API
            # standard format). Sending an array of {type:"text", text:"..."} objects is
            # non-standard and causes some models (e.g. DeepSeek V3.2 via NVIDIA NIM) to mirror
            # the content-block structure literally in their output.
            if assistant_text:
                assistant["content"] = assistant_text
            # reasoning_details is the structured alternative to a raw reasoning field.
            if preserved_details is None:
                # Use the signature from the first thinking block if available (llama.cpp server + gpt-oss).
                signature = non_empty_thinking[0].thinking_signature
                if model.provider == "opencode-go" and signature == "reasoning":
                    signature = "reasoning_content"
                if signature and signature in _REPLAY_REASONING_FIELDS:
                    assistant[signature] = "\n".join(block.thinking for block in non_empty_thinking)
    elif assistant_text:
        assistant["content"] = assistant_text

    if tool_calls:
        converted: list[dict[str, Any]] = []
        for call in tool_calls:
            custom_property = grammar_properties.get(call.name) if grammar_properties else None
            if custom_property is not None:
                converted.append(
                    {
                        "id": call.id,
                        "type": "custom",
                        "custom": {
                            "name": call.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(call.name, call.arguments, custom_property)
                            ),
                        },
                    }
                )
            else:
                converted.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": fastjson.dumps(call.arguments)},
                    }
                )
        assistant["tool_calls"] = converted
    if preserved_details is not None:
        assistant["reasoning_details"] = preserved_details
    if (
        _truthy(compat.get("requiresReasoningContentOnAssistantMessages"))
        and model.reasoning
        and "reasoning_content" not in assistant
    ):
        assistant["reasoning_content"] = ""

    # Skip assistant messages that have no content and no tool calls. Some providers require
    # "either content or tool_calls, but not none", and others reject empty assistant messages.
    # This handles aborted assistant responses that got no content.
    content = assistant["content"]
    if not content and "tool_calls" not in assistant:
        return False
    params.append(assistant)
    return True


def convert_tools(tools: Sequence[Tool], compat: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert pi tools to Chat Completions tool definitions (function or grammar ``custom`` tools)."""
    supports_strict = compat.get("supportsStrictMode") is not False
    supports_grammar = _truthy(compat.get("supportsOpenAIGrammarTools"))
    result: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_grammar)
        if grammar is not None:
            result.append(
                {
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": grammar.format, "definition": grammar.definition},
                        },
                    },
                }
            )
            continue
        strict = resolve_json_schema_strict_sampling(tool, supports_strict)
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict),
        }
        # Only include strict if the provider supports it. Some reject unknown fields.
        if supports_strict:
            function["strict"] = strict if strict is not None else False
        result.append({"type": "function", "function": function})
    return result


# ---------------------------------------------------------------------------
# Usage and stop reasons
# ---------------------------------------------------------------------------


def _parse_chunk_usage(raw_usage: Any, model: Model) -> Usage:
    raw = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = raw.get("prompt_tokens") or 0
    prompt_details = raw.get("prompt_tokens_details")
    cache_read = _prop(prompt_details, "cached_tokens")
    if cache_read is None:
        cache_read = raw.get("prompt_cache_hit_tokens")
    if cache_read is None:
        cache_read = raw.get("cached_tokens")
    if cache_read is None:
        cache_read = 0
    cache_write = _prop(prompt_details, "cache_write_tokens") or 0

    # Follow documented OpenAI/OpenRouter semantics: cached_tokens is cache-read tokens (hits).
    # Providers disagree on placement: OpenAI/OpenRouter use prompt_tokens_details.cached_tokens,
    # DeepSeek uses prompt_cache_hit_tokens, and Kimi documents top-level usage.cached_tokens on
    # the final usage chunk. OpenAI does not document or emit cache_write_tokens, but
    # OpenRouter-compatible providers can include it as a separate write count
    # (https://github.com/OpenRouterTeam/ai-sdk-provider/pull/409). Do not subtract writes from
    # cached_tokens, otherwise spec-compliant providers are under-reported
    # (DS4 mirrors this contract too: https://github.com/antirez/ds4/pull/29).
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    # OpenAI completion_tokens already includes reasoning_tokens.
    output_tokens = raw.get("completion_tokens") or 0
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        reasoning=_prop(raw.get("completion_tokens_details"), "reasoning_tokens") or 0,
    )
    calculate_cost(model, usage)
    return usage


def _map_stop_reason(reason: Any) -> tuple[str, str | None]:
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    return "error", f"Provider finish_reason: {_js_string(reason)}"


# ---------------------------------------------------------------------------
# Compatibility detection
# ---------------------------------------------------------------------------


def detect_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    """Auto-detect compatibility settings from the provider id and base URL.

    Used as the base when ``model.compat`` is not set; explicit ``model.compat``
    entries override these detected values (see :func:`get_compat`).
    """
    provider = model.provider
    base_url = model.base_url

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_open_router = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_chutes = "chutes.ai" in base_url
    is_grok = provider == "xai" or "api.x.ai" in base_url

    is_non_standard = (
        is_nvidia
        or is_cerebras
        or is_grok
        or is_together
        or is_chutes
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )
    use_max_tokens = (
        is_chutes
        or is_deepseek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )
    is_open_router_developer_role_model = is_open_router and (
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    cache_control_format = "anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_open_router:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_open_router_developer_role_model or (not is_non_standard and not is_open_router),
        "supportsReasoningEffort": not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsThinkingTokenBudget": False,
        "thinkingTokenBudgetField": None,
        # OpenAI compatibility alone does not imply strict JSON-schema tool support.
        "supportsStrictMode": False,
        "supportsOpenAIGrammarTools": False,
        "supportsMidConvoSystemMessages": False,
        "supportsMidConvoToolAdditions": False,
        "cacheControlFormat": cache_control_format,
        "sendSessionAffinityHeaders": is_open_router,
        "sessionAffinityFormat": "openrouter" if is_open_router else "openai",
        "supportsLongCacheRetention": not (
            is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
    }


_COMPAT_KEYS = (
    "supportsStore",
    "supportsDeveloperRole",
    "supportsReasoningEffort",
    "supportsUsageInStreaming",
    "supportsFinishReason",
    "maxTokensField",
    "requiresToolResultName",
    "requiresAssistantAfterToolResult",
    "requiresThinkingAsText",
    "requiresReasoningContentOnAssistantMessages",
    "thinkingFormat",
    "openRouterRouting",
    "vercelGatewayRouting",
    "chatTemplateKwargs",
    "chatTemplateArgs",
    "zaiToolStream",
    "supportsThinkingTokenBudget",
    "thinkingTokenBudgetField",
    "supportsStrictMode",
    "supportsOpenAIGrammarTools",
    "supportsMidConvoSystemMessages",
    "supportsMidConvoToolAdditions",
    "cacheControlFormat",
    "sendSessionAffinityHeaders",
    "sessionAffinityFormat",
    "supportsLongCacheRetention",
)


def get_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    """Resolved compatibility settings: auto-detection overridden by explicit ``model.compat``."""
    detected = detect_compat(model)
    model_compat = model.compat
    if not model_compat:
        return detected
    resolved: ResolvedOpenAICompletionsCompat = {}
    for key in _COMPAT_KEYS:
        value = model_compat.get(key)
        resolved[key] = value if value is not None else detected[key]
    open_router_routing = model_compat.get("openRouterRouting")
    resolved["openRouterRouting"] = open_router_routing if open_router_routing is not None else {}
    resolved["vllmPriority"] = model_compat.get("vllmPriority")
    return resolved
