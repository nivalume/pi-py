# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""Shared utilities for the Google Generative AI and Google Vertex APIs.

Like pi's TypeScript adapters (which use ``@google/genai``), both adapters send requests
through the official ``google-genai`` SDK (``client.aio.models.generate_content_stream``).
This module holds what they share: message/tool conversion, thinking configuration,
stop reasons, usage, the retry policy, SDK client construction and the stream producer.
"""

from __future__ import annotations

import base64
import itertools
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Required, TypeAlias, TypedDict, TypeVar

import httpx
from google.genai import Client
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from ..abort import AbortError, abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
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
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    now_ms,
)
from ..utils.callbacks import maybe_await
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.http import ProviderHttpError, retry_provider_request
from ..utils.misc import get_pi_user_agent, provider_headers_to_record, sanitize_surrogates
from ..utils.text import append_text, append_thinking, get_system_message_text
from ..utils.transcript import (
    collapse_system_messages,
    get_current_tools,
    get_initial_system_message,
    without_initial_system_message,
)
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

__all__ = [
    "GoogleApiThinkingLevel",
    "ResolvedGoogleThinkingLevel",
    "GoogleThinkingOptions",
    "resolve_google_thinking_level",
    "uses_google_thinking_level",
    "to_google_thinking_level",
    "to_google_sdk_thinking_level",
    "get_disabled_google_thinking_config",
    "is_thinking_part",
    "retain_thought_signature",
    "requires_tool_call_id",
    "convert_messages",
    "convert_tools",
    "supports_google_strict_tool_sampling",
    "map_tool_choice",
    "resolve_google_function_calling_mode",
    "map_stop_reason",
    "map_stop_reason_string",
    "retry_google_request",
    "parse_google_usage",
    "build_google_params",
    "build_google_http_headers",
    "google_api_error",
    "get_google_client",
    "new_google_output",
    "run_google_stream",
]

# ---------------------------------------------------------------------------
# Thinking levels
# ---------------------------------------------------------------------------

#: Thinking level for Gemini 3 models. Mirrors Google's ThinkingLevel enum values.
GoogleApiThinkingLevel: TypeAlias = Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"]
ResolvedGoogleThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high"]

_RESOLVED_LEVELS = frozenset(("minimal", "low", "medium", "high"))


class GoogleThinkingOptions(TypedDict, total=False):
    """``thinking`` option of the Google adapters."""

    enabled: Required[bool]
    #: -1 for dynamic, 0 to disable.
    budget_tokens: int
    level: GoogleApiThinkingLevel


def resolve_google_thinking_level(model: Model, level: str) -> ResolvedGoogleThinkingLevel:
    """Resolve a supported pi level or model-specific Google mapping to a standard Google level."""
    level_map = model.thinking_level_map or {}
    mapped = level_map.get(level)
    resolved = mapped.lower() if isinstance(mapped, str) else level
    if resolved in _RESOLVED_LEVELS:
        return resolved  # type: ignore[return-value]
    # String(undefined) / String(null) in the upstream message.
    mapped_text = "undefined" if level not in level_map else ("null" if mapped is None else str(mapped))
    raise ValueError(
        f"Unsupported Google thinking level mapping for {model.provider}/{model.id}: {level} -> {mapped_text}"
    )


_GEMINI_3_PRO_FLASH = re.compile(r"gemini-3(?:\.\d+)?-(?:pro|flash)")
_GEMINI_4_PLUS = re.compile(r"gemini-(?:[4-9]|[1-9]\d)")
_GEMMA_4 = re.compile(r"gemma-?4")


def uses_google_thinking_level(model: Model) -> bool:
    """Whether this model uses Gemini's discrete ``thinkingLevel`` control instead of ``thinkingBudget``.

    Supported levels come from the model's ``thinking_level_map``; this only selects the wire format.
    """
    model_id = model.id.lower()
    return bool(
        # Match Gemini 3 Pro/Flash IDs with or without a minor version, such as
        # gemini-3-flash-preview, gemini-3.1-pro-preview, and gemini-3.8-flash.
        _GEMINI_3_PRO_FLASH.search(model_id)
        # Match Gemini 4 and later, such as gemini-4-argon (#10890).
        or _GEMINI_4_PLUS.search(model_id)
        or model_id == "gemini-flash-latest"
        or model_id == "gemini-flash-lite-latest"
        # Match both hosted Gemma 4 naming forms: gemma-4-* and gemma4-*.
        or _GEMMA_4.search(model_id)
    )


def to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleApiThinkingLevel:
    return level.upper()  # type: ignore[return-value]


def to_google_sdk_thinking_level(level: GoogleApiThinkingLevel) -> GoogleApiThinkingLevel:
    """The SDK's ``ThinkingLevel`` enum value (identical strings on the wire)."""
    return level


def get_disabled_google_thinking_config(model: Model) -> dict[str, Any]:
    """``thinkingConfig`` that turns thinking off, or the lowest level for models that cannot disable it."""
    if not uses_google_thinking_level(model):
        return {"thinkingBudget": 0}
    fallback = clamp_thinking_level(model, "off")
    if fallback == "off":
        return {"thinkingBudget": 0}
    resolved = resolve_google_thinking_level(model, fallback)
    return {"thinkingLevel": to_google_sdk_thinking_level(to_google_thinking_level(resolved))}


# ---------------------------------------------------------------------------
# Thought signatures
# ---------------------------------------------------------------------------


def is_thinking_part(part: genai_types.Part | Mapping[str, Any]) -> bool:
    """Whether a streamed Gemini ``Part`` (an SDK ``Part`` or its dict form) is thinking content.

    Protocol note (Gemini / Vertex AI thought signatures):

    * ``thought: true`` is the definitive marker for thinking content (thought summaries).
    * ``thoughtSignature`` is an encrypted representation of the model's internal thought
      process used to preserve reasoning context across multi-turn interactions. It can
      appear on ANY part type (text, functionCall, ...) and does NOT mark the part as thinking.
    * For non-functionCall responses, the signature appears on the last part for context replay.
    * When persisting/replaying model outputs, signature-bearing parts must be preserved as-is;
      do not merge/move signatures across parts.

    See https://ai.google.dev/gemini-api/docs/thought-signatures
    """
    thought = part.get("thought") if isinstance(part, Mapping) else part.thought
    return thought is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """Keep the last non-empty signature for the current streamed block.

    Some backends only send ``thoughtSignature`` on the first delta for a part; later deltas
    may omit it. This does NOT move signatures across distinct response parts.
    """
    if isinstance(incoming, str) and incoming:
        return incoming
    return existing


# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE = re.compile(r"[A-Za-z0-9+/]+={0,2}")


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return _BASE64_SIGNATURE.fullmatch(signature) is not None


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    """Only keep signatures from the same provider/model and with valid base64."""
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

_GEMINI_MAJOR_VERSION = re.compile(r"^gemini(?:-live)?-(\d+)")
_INVALID_TOOL_CALL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _get_gemini_major_version(model_id: str) -> int | None:
    match = _GEMINI_MAJOR_VERSION.match(model_id.lower())
    return int(match.group(1)) if match else None


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs in function calls/responses."""
    major = _get_gemini_major_version(model_id)
    return model_id.startswith("claude-") or model_id.startswith("gpt-oss-") or (major is not None and major >= 3)


def _supports_multimodal_function_response(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


def convert_messages(model: Model, context: TranscriptContext) -> list[dict[str, Any]]:
    """Convert transcript messages to Gemini ``Content[]``."""
    # Gemini has no mid-conversation system messages; the leading prompt is sent as systemInstruction.
    conversation = without_initial_system_message(collapse_system_messages(context).messages)
    contents: list[dict[str, Any]] = []
    include_id = requires_tool_call_id(model.id)

    def normalize_tool_call_id(tool_call_id: str, _model: Model, _source: AssistantMessage) -> str:
        if not include_id:
            return tool_call_id
        return _INVALID_TOOL_CALL_ID_CHARS.sub("_", tool_call_id)[:64]

    for msg in transform_messages(conversation, model, normalize_tool_call_id):
        role = msg.role
        if role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
                continue
            parts: list[dict[str, Any]] = []
            for item in msg.content:
                if item.type == "text":
                    parts.append({"text": sanitize_surrogates(item.text)})
                else:
                    parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
            if not parts:
                continue
            contents.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts = []
            # Check if message is from same provider and model - only then keep thinking blocks
            is_same = msg.provider == model.provider and msg.model == model.id
            for block in msg.content:
                kind = block.type
                if kind == "text":
                    signature = _resolve_thought_signature(is_same, block.text_signature)
                    # Skip empty text blocks — unless they carry a thought signature. Gemini can attach
                    # the signature to a part whose visible text is empty and requires it echoed back;
                    # dropping it breaks the reasoning chain and the model intermittently ends mid-task
                    # turns with a thought-only STOP (empty completion, no tool call).
                    if (not block.text or not block.text.strip()) and not signature:
                        continue
                    part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif kind == "thinking":
                    # Only keep as thinking block if same provider AND same model.
                    # Otherwise convert to plain text (no tags to avoid model mimicking them).
                    if is_same:
                        signature = _resolve_thought_signature(is_same, block.thinking_signature)
                        # Same rule as text blocks: an empty thinking block is dropped only when it
                        # carries no signature (mirrors the anthropic converter's handling).
                        if (not block.thinking or not block.thinking.strip()) and not signature:
                            continue
                        part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                        if signature:
                            part["thoughtSignature"] = signature
                        parts.append(part)
                    else:
                        # Cross-provider/model: the signature is unusable, empty blocks stay dropped.
                        if not block.thinking or not block.thinking.strip():
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})
                elif kind == "toolCall":
                    signature = _resolve_thought_signature(is_same, block.thought_signature)
                    function_call: dict[str, Any] = {
                        "name": block.name,
                        "args": block.arguments if block.arguments is not None else {},
                    }
                    if include_id:
                        function_call["id"] = block.id
                    part = {"functionCall": function_call}
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
            if not parts:
                continue
            contents.append({"role": "model", "parts": parts})
        elif role == "toolResult":
            text_result = "\n".join(block.text for block in msg.content if block.type == "text")
            images = [block for block in msg.content if block.type == "image"] if "image" in model.input else []
            has_text = bool(text_result.strip())
            has_images = bool(images)
            # Gemini 3+ models support multimodal function responses with images nested inside
            # functionResponse.parts. Claude and other non-Gemini models behind Cloud Code Assist /
            # Gemini < 3 still need a separate user image turn.
            multimodal = _supports_multimodal_function_response(model.id)
            # Use "output" key for success, "error" key for errors as per SDK documentation
            response_value = (
                sanitize_surrogates(text_result) if has_text else "(see attached image)" if has_images else ""
            )
            image_parts = [{"inlineData": {"mimeType": image.mime_type, "data": image.data}} for image in images]
            function_response: dict[str, Any] = {
                "name": msg.tool_name,
                "response": {"error": response_value} if msg.is_error else {"output": response_value},
            }
            if has_images and multimodal:
                function_response["parts"] = image_parts
            if include_id:
                function_response["id"] = msg.tool_call_id
            function_response_part = {"functionResponse": function_response}

            # Cloud Code Assist API requires all function responses to be in a single user turn.
            # Check if the last content is already a user turn with function responses and merge.
            last = contents[-1] if contents else None
            if (
                last is not None
                and last.get("role") == "user"
                and any(part.get("functionResponse") for part in last.get("parts") or ())
            ):
                last["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            # For Gemini < 3, add images in a separate user message
            if has_images and not multimodal:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})
    return contents


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    (
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",  # pre-draft-2019-09 equivalent of $defs
    )
)


def _sanitize_for_open_api(schema: Any) -> Any:
    """Strip meta-declarations from a schema object (returns a copy)."""
    if not isinstance(schema, dict):
        return schema
    return {
        key: _sanitize_for_open_api(value) for key, value in schema.items() if key not in _JSON_SCHEMA_META_DECLARATIONS
    }


def convert_tools(
    tools: Sequence[Tool], use_parameters: bool = False, supports_strict_mode: bool = True
) -> list[dict[str, Any]] | None:
    """Convert tools to Gemini function declarations.

    By default uses ``parametersJsonSchema``, which supports full JSON Schema (``anyOf``,
    ``oneOf``, ``const``, ...). ``use_parameters=True`` uses the legacy ``parameters`` field
    (OpenAPI 3.03 Schema) instead, which Cloud Code Assist needs for Claude models because the
    API translates ``parameters`` into Anthropic's ``input_schema``.
    """
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if use_parameters:
            declaration["parameters"] = _sanitize_for_open_api(parameters)
        else:
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ enforces required function parameters in validated tool-calling modes."""
    major = _get_gemini_major_version(model_id)
    return major is not None and major >= 3


def map_tool_choice(choice: str) -> str:
    """Map a tool choice string to Gemini's ``FunctionCallingConfigMode``."""
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


def resolve_google_function_calling_mode(
    tools: Sequence[Tool], tool_choice: str | None, supports_strict_mode: bool
) -> str | None:
    use_strict_mode = any(resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools)
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


# ---------------------------------------------------------------------------
# Stop reasons, usage, retries
# ---------------------------------------------------------------------------

_ERROR_FINISH_REASONS = frozenset(
    (
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "NO_IMAGE",
    )
)


def map_stop_reason(reason: str) -> StopReason:
    """Map a Gemini ``FinishReason`` to a pi stop reason (unknown reasons raise, like upstream)."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _ERROR_FINISH_REASONS:
        return "error"
    raise ValueError(f"Unhandled stop reason: {reason}")


def map_stop_reason_string(reason: str) -> StopReason:
    """Map a string finish reason to a pi stop reason (for raw API responses)."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


T = TypeVar("T")


class _UnretriedError(Exception):
    """Carries an error past the provider retry policy unchanged."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


def google_api_error(error: genai_errors.APIError) -> ProviderHttpError:
    """The SDK's ``APIError`` as a :class:`ProviderHttpError` worded like ``@google/genai``'s ``ApiError``.

    * Error status: the message is the JSON error body. A non-JSON body becomes
      ``{"error": {"message": <text>, "code": <status>, "status": <reason>}}``.
    * Error event inside a 200 event stream: ``got status: <status>. <JSON event>``.

    Like the JS SDK error it carries the status code but no headers, so retries use plain backoff.
    """
    response = error.response
    details = error.details
    if _is_stream_error(error):
        status = error.status
        message = f"got status: {status if status is not None else 'undefined'}. {_compact_json(details)}"
    else:
        body = details
        if isinstance(response, httpx.Response) and "application/json" not in (
            response.headers.get("content-type") or ""
        ):
            body = {"error": {"message": response.text, "code": error.code, "status": response.reason_phrase}}
        message = _compact_json(body)
    converted = ProviderHttpError(message, status=error.code)
    converted.__cause__ = error
    return converted


def _is_stream_error(error: genai_errors.APIError) -> bool:
    """Whether the SDK raised for an error event received in a successful (200) event stream."""
    status_code = getattr(error.response, "status_code", None)
    return isinstance(status_code, int) and status_code < 400


async def retry_google_request(request: Callable[[], Awaitable[T]], options: Mapping[str, Any] | None = None) -> T:
    """Run a Google request with the shared provider retry policy.

    408/409/429/5xx are retried with backoff, honoring ``max_retries``/``max_retry_delay_ms``/``signal``,
    like the Anthropic and OpenAI adapters. The SDK's ``APIError`` is converted with
    :func:`google_api_error`. Upstream only retries the SDK's HTTP errors, so transport failures and
    error events inside an accepted stream are not retried.
    """
    options = options or {}

    async def attempt() -> T:
        try:
            return await request()
        except genai_errors.APIError as error:
            converted = google_api_error(error)
            if _is_stream_error(error):
                raise _UnretriedError(converted) from error
            raise converted from error
        except ProviderHttpError as error:
            if error.status is None:
                raise _UnretriedError(error) from error
            raise

    try:
        return await retry_provider_request(
            attempt,
            max_retries=options.get("max_retries") or 0,
            max_retry_delay_ms=options.get("max_retry_delay_ms"),
            signal=options.get("signal"),
            telemetry_context=options.get("telemetry_context"),
        )
    except _UnretriedError as wrapped:
        raise wrapped.error from None


def parse_google_usage(model: Model, metadata: genai_types.GenerateContentResponseUsageMetadata) -> Usage:
    """Usage from the SDK's ``usage_metadata``, with cost filled in."""
    prompt = metadata.prompt_token_count or 0
    cached = metadata.cached_content_token_count or 0
    candidates = metadata.candidates_token_count or 0
    thoughts = metadata.thoughts_token_count or 0
    usage = Usage(
        input=prompt - cached,
        output=candidates + thoughts,
        cache_read=cached,
        cache_write=0,
        total_tokens=metadata.total_token_count or 0,
        reasoning=thoughts,
    )
    calculate_cost(model, usage)
    return usage


# ---------------------------------------------------------------------------
# Request parameters (the SDK's GenerateContentParameters)
# ---------------------------------------------------------------------------


def build_google_params(model: Model, context: TranscriptContext, options: Mapping[str, Any]) -> dict[str, Any]:
    """Build ``{"model", "contents", "config"}`` exactly as pi passes them to ``generate_content_stream``.

    This is the payload handed to ``on_payload``. Keys use the API's camelCase names, which the
    SDK's parameter models accept as aliases.
    """
    contents = convert_messages(model, context)
    initial = get_initial_system_message(context.messages)
    current_tools = get_current_tools(context.messages)

    config: dict[str, Any] = {}
    if options.get("temperature") is not None:
        config["temperature"] = options["temperature"]
    if options.get("max_tokens") is not None:
        config["maxOutputTokens"] = options["max_tokens"]

    supports_strict_mode = supports_google_strict_tool_sampling(model.id)
    function_calling_mode = (
        resolve_google_function_calling_mode(current_tools, options.get("tool_choice"), supports_strict_mode)
        if current_tools
        else None
    )
    system_instruction = get_system_message_text(initial) if initial is not None else ""
    if system_instruction:
        config["systemInstruction"] = sanitize_surrogates(system_instruction)
    if current_tools:
        config["tools"] = convert_tools(current_tools, False, supports_strict_mode)
    if function_calling_mode is not None:
        config["toolConfig"] = {"functionCallingConfig": {"mode": function_calling_mode}}

    thinking = options.get("thinking")
    if thinking is not None and thinking.get("enabled") and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if thinking.get("level") is not None:
            thinking_config["thinkingLevel"] = to_google_sdk_thinking_level(thinking["level"])
        elif thinking.get("budget_tokens") is not None:
            thinking_config["thinkingBudget"] = thinking["budget_tokens"]
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking is not None and not thinking.get("enabled"):
        config["thinkingConfig"] = get_disabled_google_thinking_config(model)

    signal = options.get("signal")
    if signal is not None and signal.aborted:
        raise AbortError("Request aborted")

    return {"model": model.id, "contents": contents, "config": config}


# ---------------------------------------------------------------------------
# SDK client
# ---------------------------------------------------------------------------


def build_google_http_headers(model: Model, options: Mapping[str, Any]) -> dict[str, str] | None:
    """``httpOptions.headers``: pi's User-Agent, then model headers, then request headers.

    Names are lower-cased so they replace the SDK's own entries (``x-goog-api-key``,
    ``user-agent``) instead of duplicating them. The SDK prefixes ``user-agent`` with its
    ``google-genai-sdk/<version> gl-python/<version>`` label and sets ``x-goog-api-client``.
    """
    record = provider_headers_to_record({"User-Agent": get_pi_user_agent()}, model.headers, options.get("headers"))
    return {name.lower(): value for name, value in record.items()} if record else None


_CLIENT_CACHE_ATTR = "_pi_google_genai_clients"
_CLIENT_CACHE_SIZE = 16


async def get_google_client(
    http_client: httpx.AsyncClient, key: Hashable, factory: Callable[[], Awaitable[Client]]
) -> Client:
    """The SDK client for ``key`` on ``http_client``, built with ``factory`` on first use.

    Building a ``google.genai.Client`` takes tens of milliseconds (it creates SSL contexts and
    a sync httpx client it never uses here), and a Vertex client holds the ADC credentials and
    their access token, so clients are reused. The cache lives on the httpx client (the SDK
    client refers back to it), so both are collected together; a client that does not take
    attributes gets a fresh SDK client per request.
    """
    clients: OrderedDict[Hashable, Client] | None = getattr(http_client, _CLIENT_CACHE_ATTR, None)
    if clients is None:
        clients = OrderedDict()
        try:
            setattr(http_client, _CLIENT_CACHE_ATTR, clients)
        except (AttributeError, TypeError):
            return await factory()
    client = clients.get(key)
    if client is not None:
        clients.move_to_end(key)
        return client
    client = await factory()
    clients[key] = client
    if len(clients) > _CLIENT_CACHE_SIZE:
        clients.popitem(last=False)
    return client


def _stream_request(params: Mapping[str, Any], headers: dict[str, str] | None, options: Mapping[str, Any]) -> Any:
    """``generate_content_stream`` keyword arguments, with per-request ``http_options`` in the config."""
    http_options: dict[str, Any] = {}
    if headers:
        http_options["headers"] = headers
    timeout_ms = options.get("timeout_ms")
    if timeout_ms is not None:
        http_options["timeout"] = int(timeout_ms)
    config = params.get("config")
    if http_options:
        if isinstance(config, genai_types.GenerateContentConfig):
            config = config.model_copy(update={"http_options": genai_types.HttpOptions(**http_options)})
        else:
            config = dict(config or {})
            existing = config.pop("httpOptions", None) or config.pop("http_options", None)
            config["httpOptions"] = {**existing, **http_options} if isinstance(existing, Mapping) else http_options
    return {"model": params.get("model"), "contents": params.get("contents"), "config": config}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

# Counter for generating unique tool call IDs
_tool_call_counter = itertools.count(1)

_ChunkStream: TypeAlias = AsyncIterator[genai_types.GenerateContentResponse]


def new_google_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        [], api=model.api, provider=model.provider, model=model.id, usage=Usage(), stop_reason="pending"
    )


async def _aclose(stream: _ChunkStream) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


async def _open_stream(client: Client, request: Any) -> tuple[_ChunkStream, genai_types.GenerateContentResponse | None]:
    """Send the request and wait for the first chunk.

    The SDK's stream sends its request lazily, on the first iteration; waiting for the first
    chunk makes HTTP errors surface here, inside the retry policy, as they do upstream.
    """
    stream = await client.aio.models.generate_content_stream(**request)
    try:
        first = await anext(stream, None)
    except BaseException:
        await _aclose(stream)
        raise
    return stream, first


async def run_google_stream(
    model: Model,
    context: TranscriptContext,
    options: Mapping[str, Any],
    out: AssistantMessageEventStream,
    output: AssistantMessage,
    *,
    create_client: Callable[[], Awaitable[Client]],
    missing_finish_reason_message: str,
) -> None:
    """Producer shared by the Gemini and Vertex adapters (identical upstream apart from the client)."""
    signal = options.get("signal")
    try:
        client = await create_client()
        normalized = collapse_system_messages(context)
        params: Any = build_google_params(model, normalized, options)
        on_payload = options.get("on_payload")
        if on_payload is not None:
            replaced = await maybe_await(on_payload(params, model))
            if replaced is not None:
                params = replaced
        request = _stream_request(params, build_google_http_headers(model, options), options)
        async with abort_scope(signal):
            stream, first = await retry_google_request(lambda: _open_stream(client, request), options)
            try:
                on_response = options.get("on_response")
                if on_response is not None:
                    # The SDK exposes the response headers on each chunk; a stream that started is a 200.
                    http_response = first.sdk_http_response if first is not None else None
                    headers = dict(http_response.headers or {}) if http_response is not None else {}
                    await maybe_await(on_response(ProviderResponse(200, headers), model))
                out.push(StartEvent(output))
                if first is not None:
                    await _consume_stream(model, options, out, output, first, stream)
            except genai_errors.APIError as error:
                raise google_api_error(error) from error
            finally:
                await _aclose(stream)

        if signal is not None and signal.aborted:
            raise AbortError("Request was aborted")
        if output.stop_reason == "pending":
            raise RuntimeError(missing_finish_reason_message)
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(
                f"Provider stopped with: {output.raw_stop_reason}"
                if output.raw_stop_reason
                else "An unknown error occurred"
            )
        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        out.push(ErrorEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()


def _end_block(
    out: AssistantMessageEventStream, output: AssistantMessage, block: TextContent | ThinkingContent
) -> None:
    index = len(output.content) - 1
    if isinstance(block, TextContent):
        out.push(TextEndEvent(index, block.text, output))
    else:
        out.push(ThinkingEndEvent(index, block.thinking, output))


def _signature_text(signature: bytes | str | None) -> str | None:
    """A thought signature as the base64 text pi stores (the SDK decodes it to bytes)."""
    if not signature:
        return None
    if isinstance(signature, bytes):
        return base64.b64encode(signature).decode("ascii")
    return signature


async def _consume_stream(
    model: Model,
    options: Mapping[str, Any],
    out: AssistantMessageEventStream,
    output: AssistantMessage,
    first: genai_types.GenerateContentResponse,
    stream: _ChunkStream,
) -> None:
    on_event = options.get("on_provider_stream_event")
    blocks = output.content
    current: TextContent | ThinkingContent | None = None

    chunk: genai_types.GenerateContentResponse | None = first
    while chunk is not None:
        if on_event is not None:
            await maybe_await(on_event(chunk, model))
        # GenerateContentResponse.response_id is an output-only field identifying each response.
        # Keep the first non-empty one from the stream.
        if not output.response_id:
            output.response_id = chunk.response_id or None
        candidates = chunk.candidates
        candidate = candidates[0] if candidates else None
        if candidate is not None:
            content = candidate.content
            parts = content.parts if content is not None else None
            for part in parts or ():
                text = part.text
                if text is not None:
                    is_thinking = part.thought is True
                    if current is None or is_thinking != isinstance(current, ThinkingContent):
                        if current is not None:
                            _end_block(out, output, current)
                        if is_thinking:
                            current = ThinkingContent("")
                            blocks.append(current)
                            out.push(ThinkingStartEvent(len(blocks) - 1, output))
                        else:
                            current = TextContent("")
                            blocks.append(current)
                            out.push(TextStartEvent(len(blocks) - 1, output))
                    signature = _signature_text(part.thought_signature)
                    if isinstance(current, ThinkingContent):
                        append_thinking(current, text)
                        current.thinking_signature = retain_thought_signature(current.thinking_signature, signature)
                        out.push(ThinkingDeltaEvent(len(blocks) - 1, text, output))
                    else:
                        append_text(current, text)
                        current.text_signature = retain_thought_signature(current.text_signature, signature)
                        out.push(TextDeltaEvent(len(blocks) - 1, text, output))

                function_call = part.function_call
                if function_call is not None:
                    if current is not None:
                        _end_block(out, output, current)
                        current = None
                    # Generate unique ID if not provided or if it's a duplicate
                    provided_id = function_call.id
                    needs_new_id = not provided_id or any(
                        isinstance(block, ToolCall) and block.id == provided_id for block in blocks
                    )
                    name = function_call.name
                    tool_call_id = (
                        f"{name if name is not None else 'undefined'}_{now_ms()}_{next(_tool_call_counter)}"
                        if needs_new_id
                        else provided_id
                    )
                    args = function_call.args
                    tool_call = ToolCall(
                        tool_call_id,  # type: ignore[arg-type]
                        name or "",
                        args if args is not None else {},
                        thought_signature=_signature_text(part.thought_signature),
                    )
                    blocks.append(tool_call)
                    index = len(blocks) - 1
                    out.push(ToolCallStartEvent(index, output))
                    out.push(ToolCallDeltaEvent(index, _compact_json(tool_call.arguments), output))
                    out.push(ToolCallEndEvent(index, tool_call, output))

            finish_reason = candidate.finish_reason
            if finish_reason:
                raw = finish_reason.value if isinstance(finish_reason, Enum) else str(finish_reason)
                output.raw_stop_reason = raw
                output.stop_reason = map_stop_reason(raw)
                if output.stop_reason == "stop" and any(isinstance(block, ToolCall) for block in blocks):
                    output.stop_reason = "toolUse"

        usage_metadata = chunk.usage_metadata
        if usage_metadata is not None:
            output.usage = parse_google_usage(model, usage_metadata)

        chunk = await anext(stream, None)

    if current is not None:
        _end_block(out, output, current)


def _compact_json(value: Any) -> str:
    """``JSON.stringify`` output."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
