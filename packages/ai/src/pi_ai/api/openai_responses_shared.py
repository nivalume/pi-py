# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""Message/tool conversion and stream processing shared by the OpenAI Responses API family.

Used by ``openai-responses``, ``azure-openai-responses`` and ``openai-codex-responses``.
Request items and stream events are plain dicts in the OpenAI Responses wire format.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Literal, TypedDict

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    ImageContent,
    Model,
    StopReason,
    SystemMessage,
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
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.json_parse import StreamingJsonParser, parse_tool_call_arguments
from ..utils.misc import sanitize_surrogates, short_hash
from ..utils.text import append_text, append_thinking, get_system_message_text, render_system_message_update
from ..utils.transcript import resolve_transcript, resolve_transcript_tools
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .transform_messages import transform_messages

__all__ = [
    "OpenAIResponsesStreamOptions",
    "ConvertResponsesMessagesOptions",
    "ConvertResponsesToolsOptions",
    "convert_responses_messages",
    "convert_responses_tools",
    "process_responses_stream",
]

# =============================================================================
# Utilities
# =============================================================================


def _json_stringify(value: Any) -> str:
    """Compact JSON like ``JSON.stringify`` (non-ASCII kept as is)."""
    return fastjson.dumps(value)


def _encode_text_signature_v1(item_id: Any, phase: Any = None) -> str:
    payload: dict[str, Any] = {"v": 1}
    if item_id is not None:
        payload["id"] = item_id
    if phase:
        payload["phase"] = phase
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _parse_text_signature(signature: str | None) -> tuple[str, str | None] | None:
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            parsed = None  # Fall through to legacy plain-string handling.
        if isinstance(parsed, dict):
            version = parsed.get("v")
            if version == 1 and not isinstance(version, bool) and isinstance(parsed.get("id"), str):
                phase = parsed.get("phase")
                return parsed["id"], phase if phase in ("commentary", "final_answer") else None
    return signature, None


def _convert_tool_result_output(model: Model, content: Sequence[TextContent | ImageContent]) -> str | list[Any]:
    text_result = "\n".join(block.text for block in content if block.type == "text")
    images = [block for block in content if block.type == "image"]
    has_text = len(text_result.strip()) > 0

    if not images or "image" not in model.input:
        if has_text:
            return sanitize_surrogates(text_result)
        return "(see attached image)" if images else "(no tool output)"

    output: list[Any] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {"type": "input_image", "detail": "auto", "image_url": f"data:{image.mime_type};base64,{image.data}"}
        )
    return output


class OpenAIResponsesStreamOptions(TypedDict, total=False):
    on_provider_stream_event: Callable[[Any, Any], Any]
    service_tier: str
    #: Grammar (custom) tool name -> its single string input property.
    grammar_tool_input_properties: Mapping[str, str]
    #: ``(response_service_tier, request_service_tier) -> service tier`` used for pricing.
    resolve_service_tier: Callable[[Any, Any], Any]
    #: ``(usage, service_tier) -> None``; adjusts ``usage.cost`` in place.
    apply_service_tier_pricing: Callable[[Usage, Any], None]


class ConvertResponsesToolsOptions(TypedDict, total=False):
    #: Default ``strict`` value of function tools. Absent means ``False``; ``None`` sends ``null``.
    strict: bool | None
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    tool_search_result: bool


class ConvertResponsesMessagesOptions(TypedDict, total=False):
    include_system_prompt: bool
    grammar_tool_input_properties: Mapping[str, str]
    #: Whether later system messages are sent in place; otherwise they are folded into the leading prompt.
    supports_mid_convo_system_messages: bool
    supports_additional_tools: bool
    supports_tool_search: bool
    tool_options: ConvertResponsesToolsOptions
    #: Effort for the reply being generated. When set, the thinking level is sent as
    #: ``configuration_update`` items instead of a changing ``reasoning.effort``, which would invalidate
    #: the prompt cache. Each item sets the effort from that point on: one goes before every earlier
    #: reply with the level recorded on it (``provider_thinking_level``), and one with this effort goes
    #: at the end. Earlier input therefore never changes. OpenAI documents the item before user
    #: messages only. After tool results it was verified on gpt-6-luna (October 2026) to be accepted
    #: and to apply to the next reply.
    configuration_update_effort: str


# =============================================================================
# Message conversion
# =============================================================================

_ID_PART_INVALID = re.compile(r"[^a-zA-Z0-9_-]")
_TRAILING_UNDERSCORES = re.compile(r"_+$")


def _replace_invalid_id_char(match: re.Match[str]) -> str:
    # JavaScript replaces per UTF-16 code unit, so astral characters become two underscores.
    return "__" if ord(match.group()) > 0xFFFF else "_"


def _normalize_id_part(part: str) -> str:
    sanitized = _ID_PART_INVALID.sub(_replace_invalid_id_char, part)
    normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
    return _TRAILING_UNDERSCORES.sub("", normalized)


def _build_foreign_responses_item_id(item_id: str) -> str:
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64] if len(normalized) > 64 else normalized


def convert_responses_messages(
    model: Model,
    context: TranscriptContext,
    allowed_tool_call_providers: AbstractSet[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    """Convert a transcript into OpenAI Responses ``input`` items."""
    options = options or {}
    normalized_context = resolve_transcript(context, options.get("supports_mid_convo_system_messages"))
    messages: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(tool_call_id)
        if "|" not in tool_call_id:
            return _normalize_id_part(tool_call_id)
        parts = tool_call_id.split("|")
        call_id, item_id = parts[0], parts[1]
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id) if is_foreign_tool_call else _normalize_id_part(item_id)
        )
        # OpenAI Responses API requires item id to start with "fc"
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(normalized_context.messages, model, normalize_tool_call_id)
    supports_additional_tools = bool(options.get("supports_additional_tools"))
    supports_tool_search = bool(options.get("supports_tool_search"))
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages, supports_additional_tools or supports_tool_search
    )
    tool_options = options.get("tool_options")
    grammar_properties = options.get("grammar_tool_input_properties") or {}

    def append_system_tool_additions(message: SystemMessage, seed: str) -> None:
        tools = (message.tools_added or []) if transcript_tools.anchors_additions else []
        if not tools:
            return
        if supports_additional_tools:
            messages.append(
                {"type": "additional_tools", "role": "developer", "tools": convert_responses_tools(tools, tool_options)}
            )
            return
        if not supports_tool_search:
            return
        names = [tool.name for tool in tools]
        call_id = f"pi_tool_load_{short_hash(seed + ':' + ','.join(names))}"
        messages.append(
            {
                "type": "tool_search_call",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "arguments": {"query": " ".join(names), "limit": len(names)},
            }
        )
        messages.append(
            {
                "type": "tool_search_output",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(tools, {**(tool_options or {}), "tool_search_result": True}),
            }
        )

    include_initial_system_message = options.get("include_system_prompt")
    if include_initial_system_message is None:
        include_initial_system_message = True
    configuration_update_effort = options.get("configuration_update_effort")

    def push_configuration_update(effort: str) -> None:
        messages.append({"type": "configuration_update", "reasoning": {"effort": effort}})

    compat = model.compat or {}
    instruction_role = "developer" if model.reasoning and compat.get("supportsDeveloperRole") is not False else "system"

    msg_index = 0
    source_index = 0
    for msg in transformed_messages:
        is_leading_system_message = source_index == 0 and msg.role == "system"
        source_index += 1
        role = msg.role
        if role == "system":
            if not is_leading_system_message:
                append_system_tool_additions(msg, f"system:{msg_index}")
            if not is_leading_system_message or include_initial_system_message:
                text = get_system_message_text(msg) if is_leading_system_message else render_system_message_update(msg)
                if text:
                    messages.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {"role": "user", "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}]}
                )
            else:
                content: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if not content:
                    continue
                messages.append({"role": "user", "content": content})
        elif role == "assistant":
            output: list[dict[str, Any]] = []
            is_same_provider_and_api = msg.provider == model.provider and msg.api == model.api
            is_same_model = is_same_provider_and_api and msg.model == model.id
            is_different_model = is_same_provider_and_api and msg.model != model.id
            text_block_index = 0

            for block in msg.content:
                kind = block.type
                if kind == "thinking":
                    if block.thinking_signature:
                        output.append(fastjson.loads(block.thinking_signature))
                elif kind == "text":
                    parsed_signature = _parse_text_signature(block.text_signature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    # OpenAI requires id to be max 64 characters
                    msg_id = parsed_signature[0] if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_message_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    message_item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    phase = parsed_signature[1] if parsed_signature else None
                    if phase is not None:
                        message_item["phase"] = phase
                    output.append(message_item)
                elif kind == "toolCall":
                    id_parts = block.id.split("|")
                    call_id = id_parts[0]
                    item_id: str | None = id_parts[1] if len(id_parts) > 1 else None
                    custom_input_property = grammar_properties.get(block.name)

                    # For different-model messages, set id to undefined to avoid pairing validation.
                    # OpenAI tracks which item IDs were paired with rs_xxx reasoning items.
                    # By omitting the id, we avoid triggering that validation (like cross-provider does).
                    # Also drop ids that do not match the replayed item type: function_call ids must be fc_*
                    # and custom_tool_call ids must be ctc_*. Foreign tool call ids are normalized to fc_*, and
                    # a call can switch between the two types when grammar tool support differs.
                    item_id_prefix = "fc_" if custom_input_property is None else "ctc_"
                    if is_different_model or not (item_id is not None and item_id.startswith(item_id_prefix)):
                        item_id = None

                    call_item: dict[str, Any]
                    if custom_input_property is not None:
                        call_item = {"type": "custom_tool_call"}
                        if item_id is not None:
                            call_item["id"] = item_id
                        call_item["call_id"] = call_id
                        call_item["name"] = block.name
                        call_item["input"] = sanitize_surrogates(
                            get_grammar_tool_input(block.name, block.arguments, custom_input_property)
                        )
                    else:
                        call_item = {"type": "function_call"}
                        if item_id is not None:
                            call_item["id"] = item_id
                        call_item["call_id"] = call_id
                        call_item["name"] = block.name
                        call_item["arguments"] = _json_stringify(block.arguments)
                    if is_same_model and block.namespace is not None:
                        call_item["namespace"] = block.namespace
                    output.append(call_item)
            if not output:
                continue
            if configuration_update_effort is not None and is_same_model and msg.provider_thinking_level is not None:
                push_configuration_update(msg.provider_thinking_level)
            messages.extend(output)
        elif role == "toolResult":
            call_id = msg.tool_call_id.split("|")[0]
            tool_output = _convert_tool_result_output(model, msg.content)
            if msg.tool_name in grammar_properties:
                messages.append({"type": "custom_tool_call_output", "call_id": call_id, "output": tool_output})
            else:
                messages.append({"type": "function_call_output", "call_id": call_id, "output": tool_output})
        if not is_leading_system_message:
            msg_index += 1
    if configuration_update_effort is not None:
        push_configuration_update(configuration_update_effort)

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(
    tools: Sequence[Tool], options: ConvertResponsesToolsOptions | None = None
) -> list[dict[str, Any]]:
    """Convert pi tools to OpenAI Responses ``function``/``custom`` (grammar) tools."""
    options = options or {}
    default_strict = options.get("strict", False)
    supports_strict_mode = options.get("supports_strict_mode")
    if supports_strict_mode is None:
        supports_strict_mode = True
    supports_openai_grammar_tools = bool(options.get("supports_openai_grammar_tools"))
    tool_search_result = bool(options.get("tool_search_result"))

    result: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            custom_tool: dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {"type": "grammar", "syntax": grammar.format, "definition": grammar.definition},
            }
            if tool_search_result:
                custom_tool["defer_loading"] = True
            result.append(custom_tool)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        strict = constrained_strict if constrained_strict is not None else default_strict
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict is True),
        }
        if tool_search_result:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = strict
        result.append(function_tool)
    return result


# =============================================================================
# Stream processing
# =============================================================================

_SlotKind = Literal["thinking", "text", "toolCall"]


class _Slot:
    """Streaming state of one output item (scratch data never stored on the content block)."""

    __slots__ = ("block", "content_index", "custom_buffer", "custom_property", "kind", "parser")

    def __init__(self, kind: _SlotKind, block: Any, content_index: int) -> None:
        self.kind = kind
        self.block = block
        self.content_index = content_index
        #: Function-call argument parser; ``None`` once the call finished (pi's ``partialJson``).
        self.parser: StreamingJsonParser | None = None
        #: Grammar tool input property and JSON delta buffer; ``None`` once finished (pi's ``customInput``).
        self.custom_property: str | None = None
        self.custom_buffer: GrammarToolInputJsonBuffer | None = None


def _custom_tool_call_input(slot: _Slot) -> str:
    if slot.custom_property is None:
        return ""
    value = slot.block.arguments.get(slot.custom_property)
    return value if isinstance(value, str) else ""


def _append_custom_tool_call_input(slot: _Slot, next_input: str, close: bool) -> str | None:
    buffer = slot.custom_buffer
    prop = slot.custom_property
    if buffer is None or prop is None:
        return None
    delta = append_grammar_tool_input_json_delta(buffer, prop, next_input, close)
    slot.block.arguments = {prop: next_input}
    return delta


def _js_template(event: Mapping[str, Any], key: str) -> str:
    """Render a field the way a JavaScript template literal would (missing -> ``undefined``)."""
    if key not in event:
        return "undefined"
    value = event[key]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return "[object Object]" if isinstance(value, dict) else ",".join(str(item) for item in value)


async def process_responses_stream(
    events: AsyncIterable[Any],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """Normalize OpenAI Responses stream events into ``output`` and push pi events to ``stream``.

    Raises when the stream fails, ends before a terminal response event, or completes with an
    unfinished tool call.
    """
    options = options or {}
    on_provider_stream_event = options.get("on_provider_stream_event")
    grammar_properties = options.get("grammar_tool_input_properties") or {}
    saw_terminal_response_event = False
    output_slots: dict[Any, _Slot] = {}
    reasoning_blocks_by_id: dict[Any, ThinkingContent] = {}
    #: Tool-call blocks (by identity) whose ``output_item.done`` has not arrived yet.
    unfinished_tool_calls: set[int] = set()
    content = output.content
    push = stream.push

    def apply_message_phase_stop_reason(item: Mapping[str, Any]) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def get_slot(output_index: Any, kind: _SlotKind) -> _Slot | None:
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.kind == kind else None

    def push_tool_call_delta(slot: _Slot, delta: str | None) -> None:
        if delta is None:
            return
        push(ToolCallDeltaEvent(slot.content_index, delta, output))

    def create_slot(output_index: Any, item: Mapping[str, Any]) -> _Slot | None:
        item_type = item.get("type")
        if item_type == "reasoning":
            block: Any = ThinkingContent("")
            content.append(block)
            slot = _Slot("thinking", block, len(content) - 1)
            output_slots[output_index] = slot
            push(ThinkingStartEvent(slot.content_index, output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            block = TextContent("")
            content.append(block)
            slot = _Slot("text", block, len(content) - 1)
            output_slots[output_index] = slot
            push(TextStartEvent(slot.content_index, output))
            return slot
        if item_type == "function_call":
            block = ToolCall(
                f"{_js_template(item, 'call_id')}|{_js_template(item, 'id')}",
                item.get("name"),  # type: ignore[arg-type]
                {},
                namespace=item.get("namespace"),
            )
            content.append(block)
            slot = _Slot("toolCall", block, len(content) - 1)
            parser = StreamingJsonParser()
            parser.feed(item.get("arguments") or "")
            slot.parser = parser
            unfinished_tool_calls.add(id(block))
            output_slots[output_index] = slot
            push(ToolCallStartEvent(slot.content_index, output))
            return slot
        if item_type == "custom_tool_call":
            input_property = grammar_properties.get(item.get("name"), "input")  # type: ignore[arg-type]
            block = ToolCall(
                f"{_js_template(item, 'call_id')}|{_js_template(item, 'id')}",
                item.get("name"),  # type: ignore[arg-type]
                {input_property: item.get("input") or ""},
                namespace=item.get("namespace"),
            )
            content.append(block)
            slot = _Slot("toolCall", block, len(content) - 1)
            slot.custom_property = input_property
            slot.custom_buffer = GrammarToolInputJsonBuffer()
            unfinished_tool_calls.add(id(block))
            output_slots[output_index] = slot
            push(ToolCallStartEvent(slot.content_index, output))
            return slot
        return None

    def backfill_reasoning_signatures(response_output: Any) -> None:
        # Azure OpenAI can omit reasoning.encrypted_content from response.output_item.done
        # and provide it only in response.completed.response.output. Backfill the
        # persisted reasoning signature from the terminal response to keep store:false
        # multi-turn replay stateless. See https://github.com/earendil-works/pi/issues/6409.
        for item in response_output or ():
            if not isinstance(item, dict) or item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id"))
            if block is None or not block.thinking_signature:
                continue
            stored_item = json.loads(block.thinking_signature)
            if stored_item.get("encrypted_content"):
                continue
            block.thinking_signature = _json_stringify({**stored_item, "encrypted_content": item["encrypted_content"]})

    def finalize_response(response: Any) -> None:
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        response = response if isinstance(response, dict) else {}
        backfill_reasoning_signatures(response.get("output"))
        if response.get("id"):
            output.response_id = response["id"]
        usage = response.get("usage")
        if usage:
            input_details = usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output_details = usage.get("output_tokens_details") or {}
            output.usage = Usage(
                # OpenAI includes cached and cache-write tokens in input_tokens, so subtract both.
                input=max(0, (usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=usage.get("output_tokens") or 0,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                total_tokens=usage.get("total_tokens") or 0,
                reasoning=output_details.get("reasoning_tokens") or 0,
            )
        calculate_cost(model, output.usage)
        apply_service_tier_pricing = options.get("apply_service_tier_pricing")
        if apply_service_tier_pricing is not None:
            resolve_service_tier = options.get("resolve_service_tier")
            response_tier = response.get("service_tier")
            request_tier = options.get("service_tier")
            if resolve_service_tier is not None:
                service_tier = resolve_service_tier(response_tier, request_tier)
            else:
                service_tier = response_tier if response_tier is not None else request_tier
            apply_service_tier_pricing(output.usage, service_tier)
        # Map status to stop reason. For incomplete responses, retain the provider's
        # specific reason so max-output truncation and content filtering stay distinct.
        status = response.get("status")
        incomplete_details = response.get("incomplete_details")
        reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
        incomplete_reason = reason if isinstance(reason, str) else None
        output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        stop_reason, error_message = _map_stop_reason(status, incomplete_reason)
        output.stop_reason = stop_reason
        output.error_message = error_message
        if output.stop_reason == "stop" and any(block.type == "toolCall" for block in content):
            output.stop_reason = "toolUse"

    async for event in events:
        if on_provider_stream_event is not None:
            await maybe_await(on_provider_stream_event(event, model))
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "response.created":
            response = event.get("response")
            output.response_id = response.get("id") if isinstance(response, dict) else None
        elif event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                create_slot(event.get("output_index"), item)
        elif event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            delta = event.get("delta") or ""
            append_thinking(slot.block, delta)
            push(ThinkingDeltaEvent(slot.content_index, delta, output))
        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            append_thinking(slot.block, "\n\n")
            push(ThinkingDeltaEvent(slot.content_index, "\n\n", output))
        elif event_type in ("response.output_text.delta", "response.refusal.delta"):
            slot = get_slot(event.get("output_index"), "text")
            if slot is None:
                continue
            delta = event.get("delta") or ""
            append_text(slot.block, delta)
            push(TextDeltaEvent(slot.content_index, delta, output))
        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.parser is None:
                continue
            delta = event.get("delta") or ""
            slot.parser.feed(delta)
            slot.block.arguments = slot.parser.value()
            push_tool_call_delta(slot, delta)
        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.parser is None:
                continue
            final_arguments = event.get("arguments") or ""
            previous_partial_json = slot.parser.text
            if final_arguments.startswith(previous_partial_json):
                delta = final_arguments[len(previous_partial_json) :]
                slot.parser.feed(delta)
                slot.block.arguments = slot.parser.value()
                if delta:
                    push_tool_call_delta(slot, delta)
            else:
                parser = StreamingJsonParser()
                parser.feed(final_arguments)
                slot.parser = parser
                slot.block.arguments = parser.value()
        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.custom_buffer is None:
                continue
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(slot, _custom_tool_call_input(slot) + (event.get("delta") or ""), False),
            )
        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.custom_buffer is None:
                continue
            push_tool_call_delta(slot, _append_custom_tool_call_input(slot, event.get("input") or "", True))
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            output_index = event.get("output_index")
            apply_message_phase_stop_reason(item)
            slot = output_slots.get(output_index) or create_slot(output_index, item)
            item_type = item.get("type")

            if item_type == "reasoning" and slot is not None and slot.kind == "thinking":
                block = slot.block
                summary_text = "\n\n".join(_text_of(part) for part in item.get("summary") or ())
                content_text = "\n\n".join(_text_of(part) for part in item.get("content") or ())
                block.thinking = summary_text or content_text or block.thinking
                block.thinking_signature = _json_stringify(item)
                reasoning_blocks_by_id[item.get("id")] = block
                push(ThinkingEndEvent(slot.content_index, block.thinking, output))
                del output_slots[output_index]
            elif item_type == "message" and slot is not None and slot.kind == "text":
                block = slot.block
                block.text = "".join(
                    (part.get("text") if part.get("type") == "output_text" else part.get("refusal")) or ""
                    for part in item.get("content") or ()
                    if isinstance(part, dict)
                )
                block.text_signature = _encode_text_signature_v1(item.get("id"), item.get("phase"))
                push(TextEndEvent(slot.content_index, block.text, output))
                del output_slots[output_index]
            elif (
                item_type == "function_call"
                and slot is not None
                and slot.kind == "toolCall"
                and slot.parser is not None
            ):
                block = slot.block
                arguments, arguments_error = parse_tool_call_arguments(item.get("arguments") or slot.parser.text)
                block.arguments = arguments
                if arguments_error is not None:
                    block.arguments_error = arguments_error
                if item.get("namespace") is not None:
                    block.namespace = item["namespace"]
                # Finalize in-place and drop the scratch buffer so replay only carries parsed arguments.
                slot.parser = None
                unfinished_tool_calls.discard(id(block))
                push(ToolCallEndEvent(slot.content_index, block, output))
                del output_slots[output_index]
            elif (
                item_type == "custom_tool_call"
                and slot is not None
                and slot.kind == "toolCall"
                and slot.custom_buffer is not None
            ):
                block = slot.block
                final_input = item.get("input")
                push_tool_call_delta(
                    slot,
                    _append_custom_tool_call_input(
                        slot, final_input if final_input is not None else _custom_tool_call_input(slot), True
                    ),
                )
                if item.get("namespace") is not None:
                    block.namespace = item["namespace"]
                slot.custom_buffer = None
                slot.custom_property = None
                unfinished_tool_calls.discard(id(block))
                push(ToolCallEndEvent(slot.content_index, block, output))
                del output_slots[output_index]
        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(event.get("response"))
        elif event_type == "error":
            raise RuntimeError(f"Error Code {_js_template(event, 'code')}: {_js_template(event, 'message')}")
        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response")
            response = response if isinstance(response, dict) else {}
            output.raw_stop_reason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if error:
                message = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif isinstance(details, dict) and details.get("reason"):
                message = f"incomplete: {details['reason']}"
            else:
                message = "Unknown error (no error details in response)"
            raise RuntimeError(message)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")
    # The agent runs every tool call in the final message. Refuse to hand over calls whose
    # output_item.done never arrived: their arguments may be cut off or mixed up, e.g. when a
    # non-compliant server omits output_index. Finished calls have their scratch buffers removed.
    if output.stop_reason == "toolUse":
        for block in content:
            if block.type == "toolCall" and id(block) in unfinished_tool_calls:
                raise RuntimeError(
                    f"OpenAI Responses stream completed with an unfinished tool call: {block.name} ({block.id})"
                )


def _text_of(part: Any) -> str:
    if isinstance(part, dict):
        text = part.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _map_stop_reason(status: Any, incomplete_reason: str | None = None) -> tuple[StopReason, str | None]:
    if not status:
        return "stop", None
    if status == "completed":
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return "error", (
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason"
        )
    if status in ("failed", "cancelled"):
        return "error", None
    # These two are wonky ...
    if status in ("in_progress", "queued"):
        return "stop", None
    raise RuntimeError(f"Unhandled stop reason: {status}")
