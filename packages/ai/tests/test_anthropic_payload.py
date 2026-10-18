"""Anthropic Messages request payloads (ports the payload-capturing upstream tests).

Covers anthropic-empty-thinking-signature-compat, anthropic-error-tool-result-images,
anthropic-force-adaptive-thinking, anthropic-mid-conversation-effort, anthropic-temperature-compat,
anthropic-thinking-disable, anthropic-strict-tool-schema, anthropic-adaptive-thinking-models,
cache-retention (Anthropic), transcript-tool-changes (Anthropic), sampling-options (Anthropic),
fireworks-models (Messages payloads) and pre-generation-error (Anthropic).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Type,
    Usage,
    UserMessage,
    normalize_context,
)
from pi_ai.api import anthropic_messages
from pi_ai.api.anthropic_messages import stream, stream_simple
from pi_ai.models import get_supported_thinking_levels
from pi_ai.providers.catalog import BUILTIN_PROVIDERS, get_builtin_model, get_builtin_models


class PayloadCaptured(Exception):
    pass


def model_of(provider: str, model_id: str) -> Model:
    model = get_builtin_model(provider, model_id)
    assert model is not None, f"{provider}/{model_id}"
    return model


def custom_model(**overrides: Any) -> Model:
    fields: dict[str, Any] = {
        "id": "claude-test",
        "name": "Claude Test",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "base_url": "http://127.0.0.1:9",
        "reasoning": False,
        "input": ["text"],
        "context_window": 200_000,
        "max_tokens": 1024,
    }
    fields.update(overrides)
    return Model(**fields)


async def capture(
    model: Model, context: Any, options: dict[str, Any] | None = None, *, simple: bool = True
) -> tuple[dict[str, Any], AssistantMessage]:
    """Capture the payload passed to ``on_payload`` (the request then fails before any HTTP call)."""
    captured: list[Any] = []

    def on_payload(payload: Any, _model: Any) -> Any:
        captured.append(payload)
        raise PayloadCaptured("payload captured")

    transcript = normalize_context(context)
    function = stream_simple if simple else stream
    s = function(model, transcript, {"api_key": "fake-key", **(options or {}), "on_payload": on_payload})
    message = await s.result()
    assert captured, f"Expected payload capture before request failure: {message.error_message}"
    assert message.stop_reason == "error"
    assert message.error_message == "payload captured"
    return captured[0], message


def hello() -> Context:
    return Context(messages=[UserMessage("Hello", timestamp=1)])


def assistant_message(
    content: list[Any], *, provider: str = "anthropic", model: str = "claude-test", **kwargs: Any
) -> AssistantMessage:
    return AssistantMessage(
        content,
        api="anthropic-messages",
        provider=provider,
        model=model,
        usage=Usage(),
        stop_reason=kwargs.pop("stop_reason", "stop"),
        timestamp=1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Empty thinking signatures (anthropic-empty-thinking-signature-compat)
# ---------------------------------------------------------------------------


def xiaomi_model(allow_empty_signature: bool | None = None) -> Model:
    return custom_model(
        id="mimo-v2.6-pro",
        provider="xiaomi-token-plan-ams",
        base_url="http://127.0.0.1:9/anthropic",
        reasoning=True,
        context_window=1_048_576,
        compat=None if allow_empty_signature is None else {"allowEmptySignature": allow_empty_signature},
    )


def signature_context(
    signature: str,
    thinking: str = "internal reasoning",
    provider: str = "xiaomi-token-plan-ams",
    model: str = "mimo-v2.6-pro",
) -> Context:
    return Context(
        messages=[
            UserMessage("first"),
            assistant_message(
                [ThinkingContent(thinking, thinking_signature=signature)], provider=provider, model=model
            ),
            UserMessage("second"),
        ]
    )


def assistant_content(payload: dict[str, Any]) -> Any:
    return next(message for message in payload["messages"] if message["role"] == "assistant")["content"]


async def test_converts_empty_signature_thinking_to_text_by_default() -> None:
    payload, _ = await capture(xiaomi_model(), signature_context(""))
    assert assistant_content(payload) == [{"type": "text", "text": "internal reasoning"}]


async def test_preserves_empty_thinking_text_when_the_signature_is_present() -> None:
    payload, _ = await capture(xiaomi_model(), signature_context("signed-thinking", ""))
    assert assistant_content(payload) == [{"type": "thinking", "thinking": "", "signature": "signed-thinking"}]


async def test_preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled() -> None:
    payload, _ = await capture(xiaomi_model(True), signature_context(" "))
    assert assistant_content(payload) == [{"type": "thinking", "thinking": "internal reasoning", "signature": ""}]


def test_allows_empty_thinking_signatures_for_every_vercel_ai_gateway_model() -> None:
    # Regression for #9676: Vercel AI Gateway emits unsigned thinking for translated models.
    models = get_builtin_models("vercel-ai-gateway")
    assert models
    assert all((model.compat or {}).get("allowEmptySignature") is True for model in models)


@pytest.mark.parametrize(
    "model_id",
    [
        "accounts/fireworks/models/deepseek-v4p1-flash",
        "accounts/fireworks/models/qwen3p8-max",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
        "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    ],
)
async def test_preserves_unsigned_thinking_for_fireworks(model_id: str) -> None:
    # Regression for #9323: Fireworks emits unsigned thinking that must survive replay.
    model = model_of("fireworks", model_id)
    assert (model.compat or {}).get("allowEmptySignature") is True
    context = signature_context("", "internal reasoning", "fireworks", model_id)
    context.messages[1].content.append(TextContent("answer"))
    payload, _ = await capture(model, context)
    assert assistant_content(payload) == [
        {"type": "thinking", "thinking": "internal reasoning", "signature": ""},
        {"type": "text", "text": "answer"},
    ]


async def test_replays_redacted_thinking_and_drops_empty_blocks() -> None:
    model = custom_model(reasoning=True)
    context = Context(
        messages=[
            UserMessage("first"),
            assistant_message(
                [
                    ThinkingContent("[Reasoning redacted]", thinking_signature="opaque", redacted=True),
                    ThinkingContent("  "),
                    TextContent("   "),
                    TextContent("answer"),
                    ToolCall("toolu_1", "lookup", {"value": "x"}),
                ]
            ),
            ToolResultMessage("toolu_1", "lookup", [TextContent("result")]),
        ]
    )
    payload, _ = await capture(model, context, {"cache_retention": "none"})
    assert assistant_content(payload) == [
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "answer"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"value": "x"}},
    ]
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result", "is_error": False}],
    }


async def test_normalizes_foreign_tool_call_ids() -> None:
    model = custom_model()
    foreign = AssistantMessage(
        [ToolCall("call_1|fc/abc", "lookup", {})],
        api="openai-responses",
        provider="openai",
        model="gpt-5",
        stop_reason="toolUse",
    )
    context = Context(
        messages=[
            UserMessage("first"),
            foreign,
            ToolResultMessage("call_1|fc/abc", "lookup", [TextContent("ok")]),
        ]
    )
    payload, _ = await capture(model, context, {"cache_retention": "none"})
    assert assistant_content(payload)[0]["id"] == "call_1_fc_abc"
    assert payload["messages"][-1]["content"][0]["tool_use_id"] == "call_1_fc_abc"
    assert anthropic_messages.normalize_tool_call_id("a" * 80) == "a" * 64


# ---------------------------------------------------------------------------
# Error tool results with images (anthropic-error-tool-result-images)
# ---------------------------------------------------------------------------


def image_block(data: str) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def two_tool_calls() -> AssistantMessage:
    return assistant_message(
        [ToolCall("call-1", "codemode", {}), ToolCall("call-2", "read", {})], stop_reason="toolUse"
    )


async def test_moves_error_result_images_after_the_tool_results_of_the_turn() -> None:
    # Regression for #10794: Anthropic rejects error tool results that contain images.
    model = custom_model(input=["text", "image"])
    payload, _ = await capture(
        model,
        Context(
            messages=[
                UserMessage("Show the screenshots"),
                two_tool_calls(),
                ToolResultMessage(
                    "call-1",
                    "codemode",
                    [
                        TextContent("Script failed"),
                        ImageContent("AAAA", "image/png"),
                        ImageContent("CCCC", "image/png"),
                        TextContent("Script error"),
                    ],
                    True,
                ),
                ToolResultMessage("call-2", "read", [ImageContent("BBBB", "image/png")], False),
            ]
        ),
        {"cache_retention": "none"},
    )
    assert payload["messages"][2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "Script failed\n(image attached after the tool results)\nScript error",
                "is_error": True,
            },
            {
                "type": "tool_result",
                "tool_use_id": "call-2",
                "content": [{"type": "text", "text": "(see attached image)"}, image_block("BBBB")],
                "is_error": False,
            },
            {"type": "text", "text": "Images from the error result of tool call call-1 (codemode):"},
            image_block("AAAA"),
            image_block("CCCC"),
        ],
    }


async def test_keeps_error_tool_results_without_images_unchanged() -> None:
    model = custom_model(input=["text", "image"])
    payload, _ = await capture(
        model,
        Context(
            messages=[
                UserMessage("Run it"),
                assistant_message([ToolCall("call-1", "codemode", {})], stop_reason="toolUse"),
                ToolResultMessage("call-1", "codemode", [TextContent("Script failed")], True),
            ]
        ),
        {"cache_retention": "none"},
    )
    assert payload["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "Script failed", "is_error": True}],
    }


async def test_converts_user_images_and_skips_blank_user_messages() -> None:
    model = custom_model(input=["text", "image"])
    payload, _ = await capture(
        model,
        Context(
            messages=[
                UserMessage("   "),
                UserMessage([TextContent(" "), ImageContent("IMG", "image/jpeg"), TextContent("look")]),
                UserMessage([TextContent("")]),
            ]
        ),
        {"cache_retention": "none"},
    )
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "IMG"}},
                {"type": "text", "text": "look"},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Adaptive thinking (anthropic-force-adaptive-thinking, anthropic-thinking-disable)
# ---------------------------------------------------------------------------


def vendor_model(compat: dict[str, Any] | None = None, model_id: str = "vendor--claude-opus-latest") -> Model:
    # Id intentionally does not match any built-in adaptive substring. This
    # mirrors corporate proxy schemes such as `anthropic--claude-opus-latest`.
    return custom_model(id=model_id, provider="vendor-proxy", reasoning=True, max_tokens=32000, compat=compat)


def local(model: Model) -> Model:
    return dataclasses.replace(model, base_url="http://127.0.0.1:9")


async def test_sends_legacy_thinking_payload_for_custom_model_ids_by_default() -> None:
    payload, _ = await capture(vendor_model(), hello(), {"reasoning": "medium"})
    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


async def test_sends_adaptive_thinking_payload_when_force_adaptive_thinking_is_true() -> None:
    payload, _ = await capture(vendor_model({"forceAdaptiveThinking": True}), hello(), {"reasoning": "medium"})
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "medium"}


async def test_uses_adaptive_thinking_with_native_xhigh_effort_for_claude_fable_5() -> None:
    payload, _ = await capture(local(model_of("anthropic", "claude-fable-5")), hello(), {"reasoning": "xhigh"})
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


async def test_allows_built_in_adaptive_models_to_opt_out_with_force_adaptive_thinking_false() -> None:
    model = dataclasses.replace(
        local(model_of("anthropic", "claude-opus-4-8")), compat={"forceAdaptiveThinking": False}
    )
    payload, _ = await capture(model, hello(), {"reasoning": "medium"})
    assert payload["thinking"]["type"] == "enabled"
    assert "output_config" not in payload


async def test_preserves_disabled_thinking_when_reasoning_is_off_regardless_of_override() -> None:
    payload, _ = await capture(vendor_model({"forceAdaptiveThinking": True}), hello())
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


@pytest.mark.parametrize("model_id", ["claude-sonnet-4-5", "claude-opus-4-6", "claude-opus-4-8"])
async def test_sends_disabled_thinking_when_thinking_is_off(model_id: str) -> None:
    payload, _ = await capture(local(model_of("anthropic", model_id)), hello())
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


async def test_omits_disabled_thinking_for_claude_fable_5_when_thinking_is_off() -> None:
    payload, _ = await capture(local(model_of("anthropic", "claude-fable-5")), hello())
    assert "thinking" not in payload
    assert "output_config" not in payload


@pytest.mark.parametrize(
    ("model_id", "reasoning", "effort"),
    [("claude-opus-4-8", "high", "high"), ("claude-sonnet-5", "high", "high"), ("claude-opus-4-8", "xhigh", "xhigh")],
)
async def test_uses_adaptive_thinking_when_reasoning_is_enabled(model_id: str, reasoning: str, effort: str) -> None:
    payload, _ = await capture(local(model_of("anthropic", model_id)), hello(), {"reasoning": reasoning})
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": effort}


async def test_budget_thinking_fits_inside_max_tokens() -> None:
    model = vendor_model()
    payload, _ = await capture(model, hello(), {"reasoning": "high", "max_tokens": 4000})
    # max_tokens = min(4000 + 16384, 32000); budget stays at the high default.
    assert payload["max_tokens"] == 20384
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 16384, "display": "summarized"}
    assert "interleaved-thinking-2025-05-14" in payload["betas"]
    assert "temperature" not in payload


async def test_budget_thinking_is_clamped_to_leave_answer_room() -> None:
    model = vendor_model(model_id="vendor--small")
    model = dataclasses.replace(model, max_tokens=4096)
    payload, _ = await capture(model, hello(), {"reasoning": "high"})
    assert payload["max_tokens"] == 4096
    assert payload["thinking"]["budget_tokens"] == 4096 - 1024


async def test_thinking_display_and_interleaved_options() -> None:
    model = vendor_model()
    payload, _ = await capture(
        model,
        hello(),
        {"thinking_enabled": True, "thinking_display": "omitted", "interleaved_thinking": False},
        simple=False,
    )
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}
    assert "betas" not in payload


async def test_adaptive_thinking_without_effort_omits_output_config() -> None:
    model = vendor_model({"forceAdaptiveThinking": True})
    payload, _ = await capture(model, hello(), {"thinking_enabled": True}, simple=False)
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "output_config" not in payload
    # Adaptive models have interleaved thinking built in.
    assert "betas" not in payload


def test_marks_built_in_anthropic_messages_models_that_use_adaptive_thinking() -> None:
    expected = {
        "anthropic/claude-fable-5",
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "cloudflare-ai-gateway/claude-fable-5",
        "fireworks/accounts/fireworks/models/deepseek-v4p1-flash",
        "fireworks/accounts/fireworks/models/gpt-oss-120b",
        "fireworks/accounts/fireworks/models/qwen3p8-max",
        "opencode/claude-opus-4-8",
        "opencode/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-opus-4.8",
        "vercel-ai-gateway/anthropic/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-sonnet-5",
    }
    flagged = sorted(
        f"{model.provider}/{model.id}"
        for provider in BUILTIN_PROVIDERS
        for model in get_builtin_models(provider)
        if model.api == "anthropic-messages" and (model.compat or {}).get("forceAdaptiveThinking") is True
    )
    assert expected <= set(flagged)
    pattern = re.compile(r"(opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|haiku[-.]5[-.]5|fable[-.]5)")
    # Regression for #9323: Fireworks uses catalog effort metadata and verified fallbacks.
    assert all(model_id.startswith("fireworks/") or pattern.search(model_id) for model_id in flagged)


# ---------------------------------------------------------------------------
# Mid-conversation effort (anthropic-mid-conversation-effort)
# ---------------------------------------------------------------------------


def managed_model(provider: str = "anthropic") -> Model:
    return custom_model(
        id="claude-fable-5-1",
        provider=provider,
        reasoning=True,
        thinking_level_map={
            "off": None,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "max",
        },
        max_tokens=32000,
        compat={"forceAdaptiveThinking": True, "supportsMidConvoEffort": True},
    )


def managed_assistant(model: Model, level: str | None = None) -> AssistantMessage:
    return assistant_message(
        [ThinkingContent("reasoning", thinking_signature="signature"), TextContent("answer")],
        provider=model.provider,
        model=model.id,
        provider_thinking_level=level,
    )


async def capture_effort(model: Model, messages: list[Any], **options: Any) -> tuple[dict[str, Any], AssistantMessage]:
    return await capture(
        model,
        Context(messages=messages),
        {"cache_retention": "none", "thinking_enabled": True, **options},
        simple=False,
    )


def user(text: str, timestamp: int) -> UserMessage:
    return UserMessage(text, timestamp=timestamp)


def effort_messages(payload: dict[str, Any]) -> list[Any]:
    return [message for message in payload["messages"] if message["role"] == "system"]


async def test_reconstructs_an_exact_historical_marker_prefix_and_appends_the_current_marker() -> None:
    model = managed_model()
    first, first_message = await capture_effort(model, [user("one", 1)], effort="low")
    second, _ = await capture_effort(
        model, [user("one", 1), managed_assistant(model, "low"), user("two", 2)], effort="high"
    )
    assert first["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "system", "content": [], "output_config": {"effort": "low"}},
    ]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["messages"][-1] == {"role": "system", "content": [], "output_config": {"effort": "high"}}
    assert first["output_config"] == {"effort": "high"}
    assert second["output_config"] == {"effort": "high"}
    assert second["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert first_message.provider_thinking_level == "low"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_preserves_native_effort(effort: str) -> None:
    payload, message = await capture_effort(managed_model(), [user("one", 1)], effort=effort)
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": effort}}]
    assert message.provider_thinking_level == effort


async def test_defaults_omitted_effort_to_high_and_still_enables_drop_block() -> None:
    payload, message = await capture_effort(managed_model(), [user("one", 1)])
    assert payload["messages"][-1] == {"role": "system", "content": [], "output_config": {"effort": "high"}}
    assert payload["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"
    assert message.provider_thinking_level == "high"


async def test_does_not_invent_markers_for_legacy_or_other_provider_assistants() -> None:
    model = managed_model()
    legacy = managed_assistant(model)
    other_provider = dataclasses.replace(managed_assistant(model, "low"), provider="other-provider")
    payload, _ = await capture_effort(
        model,
        [user("one", 1), legacy, user("two", 2), other_provider, user("three", 3)],
        effort="medium",
    )
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": "medium"}}]


async def test_sends_plain_disabled_thinking_without_effort_when_off_is_supported() -> None:
    model = managed_model()
    model.thinking_level_map = {"minimal": None, "low": "low", "medium": "medium", "high": "high", "max": "max"}
    # Anthropic rejects block_binding with disabled thinking, and rejects effort markers
    # that change level (here low, then medium) while thinking is disabled.
    payload, message = await capture_effort(
        model,
        [
            user("one", 1),
            managed_assistant(model, "low"),
            user("two", 2),
            managed_assistant(model, "medium"),
            user("three", 3),
        ],
        thinking_enabled=False,
    )
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload
    assert effort_messages(payload) == []
    assert message.provider_thinking_level is None


async def test_keeps_adaptive_thinking_when_off_is_unsupported() -> None:
    payload, message = await capture_effort(managed_model(), [user("one", 1)], thinking_enabled=False)
    assert payload["thinking"]["type"] == "adaptive"
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": "high"}}]
    assert message.provider_thinking_level == "high"


async def test_leaves_unsupported_models_on_top_level_effort() -> None:
    model = managed_model()
    model.compat = {"forceAdaptiveThinking": True}
    payload, message = await capture_effort(model, [user("one", 1)], effort="low")
    assert payload["messages"] == [{"role": "user", "content": "one"}]
    assert payload["output_config"] == {"effort": "low"}
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert message.provider_thinking_level is None


async def test_managed_effort_models_drop_temperature() -> None:
    payload, _ = await capture_effort(managed_model(), [user("one", 1)], thinking_enabled=False, temperature=0.2)
    assert "temperature" not in payload


def test_generates_exact_model_and_transport_gates() -> None:
    direct = model_of("anthropic", "claude-fable-5-1")
    open_router = model_of("openrouter", "anthropic/claude-fable-5.1")
    unsupported = model_of("anthropic", "claude-opus-4-8")
    assert direct.compat["supportsMidConvoEffort"] is True
    assert "off" in direct.thinking_level_map and direct.thinking_level_map["off"] is None
    assert open_router.api == "anthropic-messages"
    assert open_router.base_url == "https://openrouter.ai/api"
    assert open_router.compat["supportsMidConvoEffort"] is True
    assert unsupported.compat.get("supportsMidConvoEffort") is None
    # OpenRouter rejects configuration_update on Opus 5 but accepts it on Fable 5.1.
    assert (model_of("openrouter", "anthropic/claude-opus-5").compat or {}).get("supportsMidConvoEffort") is None
    assert model_of("anthropic", "claude-opus-5").compat.get("allowedFallbackModels") is None


def test_supports_off_only_on_managed_models_that_accept_disabled_thinking() -> None:
    for model_id in ("claude-opus-5", "claude-haiku-5-5"):
        model = model_of("anthropic", model_id)
        assert model.compat["supportsMidConvoEffort"] is True
        assert (model.thinking_level_map or {}).get("off", "unset") is not None
    for model_id in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"):
        level_map = model_of("anthropic", model_id).thinking_level_map
        assert level_map is not None and "off" in level_map and level_map["off"] is None


# ---------------------------------------------------------------------------
# Temperature (anthropic-temperature-compat)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "temperature", "expected"),
    [
        ("claude-opus-4-7", 0, None),
        ("claude-opus-4-8", 0, None),
        ("claude-opus-4-7", 1, None),
        ("claude-opus-4-6", 0, 0),
        ("claude-sonnet-4-6", 0, 0),
    ],
)
async def test_temperature_compat(model_id: str, temperature: float, expected: float | None) -> None:
    payload, _ = await capture(local(model_of("anthropic", model_id)), hello(), {"temperature": temperature})
    assert payload.get("temperature") == expected
    assert ("temperature" in payload) == (expected is not None)


async def test_omits_temperature_for_custom_models_with_supports_temperature_disabled() -> None:
    model = vendor_model({"supportsTemperature": False}, "vendor--claude-opus-4-7")
    payload, _ = await capture(model, hello(), {"temperature": 0})
    assert "temperature" not in payload


async def test_omits_temperature_with_extended_thinking() -> None:
    payload, _ = await capture(vendor_model(), hello(), {"temperature": 0.5, "reasoning": "low"})
    assert "temperature" not in payload


def test_sampling_params_are_ignored() -> None:
    model = custom_model()
    built = anthropic_messages.build_params(
        model, normalize_context(hello()), False, {"sampling_params": {"top_p": 0.9, "top_k": 40}}
    )
    assert "top_p" not in built.params
    assert "top_k" not in built.params


# ---------------------------------------------------------------------------
# Strict tool schemas (anthropic-strict-tool-schema)
# ---------------------------------------------------------------------------


def strict_model() -> Model:
    return custom_model(
        id="claude-opus-4-8",
        provider="test-anthropic",
        reasoning=True,
        max_tokens=32000,
        compat={"forceAdaptiveThinking": True, "supportsStrictTools": True},
    )


async def capture_first_tool(tool: Tool) -> dict[str, Any]:
    payload, _ = await capture(
        strict_model(),
        Context(messages=[UserMessage("Use the tool")], tools=[tool]),
        {"cache_retention": "none"},
        simple=False,
    )
    return payload["tools"][0]


def lookup_tool(parameters: Any, constrained_sampling: Any = None) -> Tool:
    return Tool("lookup", "Look up a value", parameters, constrained_sampling=constrained_sampling)


def strict_tool(parameters: Any) -> Tool:
    return lookup_tool(parameters, {"type": "json_schema", "strict": "prefer"})


async def test_only_sends_the_full_input_schema_for_strict_json_schema_tools() -> None:
    legacy_parameters = Type.Object({"value": Type.String()}, additional_properties=False, title="LookupInput")
    legacy = await capture_first_tool(lookup_tool(legacy_parameters))
    assert "strict" not in legacy
    assert legacy["input_schema"] == {
        "type": "object",
        "properties": legacy_parameters["properties"],
        "required": legacy_parameters["required"],
    }
    assert legacy["eager_input_streaming"] is True

    strict = await capture_first_tool(
        strict_tool(
            Type.Object({"value": Type.String(), "optional": Type.Optional(Type.Number())}, title="StrictLookupInput")
        )
    )
    assert strict["strict"] is True
    schema = strict["input_schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["value", "optional"]
    assert schema["properties"]["optional"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert schema["title"] == "StrictLookupInput"
    assert list(strict) == ["name", "description", "eager_input_streaming", "strict", "input_schema"]


async def test_sends_prefer_tools_non_strict_when_they_use_keywords_anthropic_strict_mode_rejects() -> None:
    # https://github.com/earendil-works/pi/issues/9953
    unsupported = [
        Type.Object({"timeoutMs": Type.Optional(Type.Integer(minimum=1, maximum=300000))}),
        Type.Object({"options": Type.Object({"tags": Type.Array(Type.String(), min_items=2)})}),
        Type.Object({"expression": Type.String(format="regex")}),
    ]
    for parameters in unsupported:
        tool = await capture_first_tool(strict_tool(parameters))
        assert "strict" not in tool

    supported = await capture_first_tool(
        strict_tool(
            Type.Object(
                {
                    "code": Type.String(min_length=1, max_length=1000, pattern="^[a-z]+$"),
                    "url": Type.String(format="uri"),
                    "tags": Type.Array(Type.String(), min_items=1),
                }
            )
        )
    )
    assert supported["strict"] is True


async def test_required_strict_tool_without_support_fails() -> None:
    model = custom_model()
    tool = lookup_tool(Type.Object({"value": Type.String()}), {"type": "json_schema", "strict": "require"})
    s = stream(model, normalize_context(Context(messages=[UserMessage("x")], tools=[tool])), {"api_key": "k"})
    result = await s.result()
    assert result.stop_reason == "error"
    assert "requires JSON-schema constrained sampling" in result.error_message


# ---------------------------------------------------------------------------
# Cache retention (cache-retention, Anthropic provider)
# ---------------------------------------------------------------------------

CACHE_CONTEXT = Context(system_prompt="You are a helpful assistant.", messages=[UserMessage("Hello")])


@pytest.fixture
def no_cache_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("PI_CACHE_RETENTION", raising=False)
    return monkeypatch


async def test_uses_default_cache_ttl_without_pi_cache_retention(no_cache_env: pytest.MonkeyPatch) -> None:
    payload, _ = await capture(local(model_of("anthropic", "claude-haiku-4-5")), CACHE_CONTEXT, simple=False)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_adds_ttl_for_non_anthropic_base_url_with_pi_cache_retention_long(
    no_cache_env: pytest.MonkeyPatch,
) -> None:
    no_cache_env.setenv("PI_CACHE_RETENTION", "long")
    proxy = dataclasses.replace(model_of("anthropic", "claude-haiku-4-5"), base_url="https://my-proxy.example.com/v1")
    payload, _ = await capture(proxy, CACHE_CONTEXT, simple=False)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_pi_cache_retention_can_come_from_provider_env(no_cache_env: pytest.MonkeyPatch) -> None:
    payload, _ = await capture(
        local(model_of("anthropic", "claude-haiku-4-5")), CACHE_CONTEXT, {"env": {"PI_CACHE_RETENTION": "long"}}
    )
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_omits_ttl_when_supports_long_cache_retention_is_false(no_cache_env: pytest.MonkeyPatch) -> None:
    proxy = dataclasses.replace(
        model_of("anthropic", "claude-haiku-4-5"),
        base_url="https://my-proxy.example.com/v1",
        compat={"supportsLongCacheRetention": False},
    )
    payload, _ = await capture(proxy, CACHE_CONTEXT, {"cache_retention": "long"}, simple=False)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_omits_cache_control_when_cache_retention_is_none(no_cache_env: pytest.MonkeyPatch) -> None:
    payload, _ = await capture(
        local(model_of("anthropic", "claude-haiku-4-5")), CACHE_CONTEXT, {"cache_retention": "none"}, simple=False
    )
    assert "cache_control" not in payload["system"][0]
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]


async def test_adds_cache_control_to_string_user_messages(no_cache_env: pytest.MonkeyPatch) -> None:
    payload, _ = await capture(local(model_of("anthropic", "claude-haiku-4-5")), CACHE_CONTEXT, simple=False)
    last = payload["messages"][-1]
    assert isinstance(last["content"], list)
    assert last["content"][-1] == {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}


async def test_sets_1h_cache_ttl_when_cache_retention_is_long(no_cache_env: pytest.MonkeyPatch) -> None:
    payload, _ = await capture(
        local(model_of("anthropic", "claude-haiku-4-5")), CACHE_CONTEXT, {"cache_retention": "long"}, simple=False
    )
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_cache_control_on_last_tool_and_tool_result(no_cache_env: pytest.MonkeyPatch) -> None:
    tools = [Tool("a", "A", Type.Object({})), Tool("b", "B", Type.Object({}))]
    context = Context(
        messages=[
            UserMessage("go"),
            assistant_message([ToolCall("t1", "a", {})], stop_reason="toolUse"),
            ToolResultMessage("t1", "a", [TextContent("done")]),
        ],
        tools=tools,
    )
    payload, _ = await capture(custom_model(), context, simple=False)
    assert "cache_control" not in payload["tools"][0]
    assert payload["tools"][1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["input_schema"] == {"type": "object", "properties": {}, "required": []}
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Transcript system messages and tool changes (transcript-tool-changes, Anthropic)
# ---------------------------------------------------------------------------


def empty_tool(name: str) -> Tool:
    return Tool(name, f"{name} tool", Type.Object({}))


BASE_TOOL = empty_tool("base_tool")
LATE_TOOL = empty_tool("late_tool")


def tool_change_context() -> Context:
    return Context(
        messages=[
            SystemMessage(
                "base prompt",
                sections={"rules": "<rules>\nold rules\n</rules>", "docs": "<docs>\nread docs\n</docs>"},
                tools_added=[BASE_TOOL],
                timestamp=0,
            ),
            UserMessage("before", timestamp=1),
            SystemMessage(
                "updated guidance",
                sections={"rules": "<rules>\nnew rules\n</rules>", "docs": None},
                tools_removed=[ToolReference("base_tool")],
                tools_added=[LATE_TOOL],
                timestamp=2,
            ),
        ]
    )


def native_model(**compat: Any) -> Model:
    return custom_model(
        id="claude-opus-5",
        reasoning=True,
        max_tokens=1000,
        context_window=100_000,
        compat=compat or {"supportsMidConvoSystemMessages": True, "supportsMidConvoToolChanges": True},
    )


async def test_sends_anthropic_updates_and_tool_changes_in_native_system_messages() -> None:
    payload, _ = await capture(native_model(), tool_change_context())

    assert "inline-tools-2026-09-15" in payload["betas"]
    assert "mid-conversation-tool-changes-2026-07-01" not in payload["betas"]
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>"
    ]
    # The top-level list holds only the initial tools (with the cache breakpoint) and the
    # deferred placeholder; later tools never appear there.
    tools = payload["tools"]
    assert len(tools) == 2
    assert tools[0]["name"] == "base_tool"
    assert tools[0]["cache_control"] == {"type": "ephemeral"}
    assert "defer_loading" not in tools[0]
    assert tools[1]["name"] == "__pi_deferred_placeholder__"
    assert tools[1]["defer_loading"] is True
    assert "cache_control" not in tools[1]

    update = payload["messages"][-1]
    assert update["role"] == "system"
    content = update["content"]
    assert [block["type"] for block in content] == ["text", "tool_removal", "tool_addition"]
    assert "cache_control" not in content[0]
    assert content[1]["tool"] == {"type": "tool_reference", "name": "base_tool"}
    assert content[2]["tool"]["type"] == "tool_definition"
    definition = content[2]["tool"]["definition"]
    assert definition["name"] == "late_tool"
    assert definition["description"] == "late_tool tool"
    assert content[2]["cache_control"] == {"type": "ephemeral"}
    # Cache control goes on the block, never on the inline definition; nothing is deferred.
    assert "cache_control" not in definition
    assert "defer_loading" not in definition
    assert "updated guidance" in content[0]["text"]
    assert "<rules>\nnew rules\n</rules>" in content[0]["text"]
    assert 'Removed system prompt section "docs"' in content[0]["text"]

    # The placeholder is declared before any change so its scaffolding is cached from request one.
    initial, _ = await capture(native_model(), Context(messages=tool_change_context().messages[:2]))
    assert [tool["name"] for tool in initial["tools"]] == ["base_tool", "__pi_deferred_placeholder__"]


async def test_redefines_an_anthropic_tool_by_value_under_the_same_name() -> None:
    redefined = dataclasses.replace(BASE_TOOL, description="changed")
    payload, _ = await capture(
        native_model(),
        Context(
            messages=[
                SystemMessage("base prompt", tools_added=[BASE_TOOL], timestamp=0),
                UserMessage("before", timestamp=1),
                SystemMessage("", tools_removed=[ToolReference("base_tool")], tools_added=[redefined], timestamp=2),
            ]
        ),
    )
    assert "inline-tools-2026-09-15" in payload["betas"]
    # The top-level declaration keeps the original definition, so the cached prefix survives.
    assert [(tool["name"], tool["description"]) for tool in payload["tools"]] == [
        ("base_tool", "base_tool tool"),
        ("__pi_deferred_placeholder__", "Reserved placeholder. Never available. Never call this."),
    ]
    # The new definition replaces the old one in place; no removal block is sent for it.
    last = payload["messages"][-1]
    assert last["role"] == "system"
    assert len(last["content"]) == 1
    assert last["content"][0]["type"] == "tool_addition"
    assert last["content"][0]["tool"]["definition"]["name"] == "base_tool"
    assert last["content"][0]["tool"]["definition"]["description"] == "changed"


async def test_sends_the_current_anthropic_tool_list_without_an_initial_tool() -> None:
    # Anthropic rejects a tool list where every tool (here: the placeholder) is deferred.
    payload, _ = await capture(
        native_model(),
        Context(
            messages=[
                SystemMessage("base prompt", timestamp=0),
                UserMessage("before", timestamp=1),
                SystemMessage("updated guidance", tools_added=[LATE_TOOL], timestamp=2),
            ]
        ),
    )
    assert "inline-tools-2026-09-15" not in payload.get("betas", [])
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "late_tool"
    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert "defer_loading" not in payload["tools"][0]
    assert [block["type"] for block in payload["messages"][-1]["content"]] == ["text"]


async def test_folds_anthropic_updates_into_the_system_prompt_without_native_support() -> None:
    model = custom_model(id="claude-sonnet-4-5", reasoning=True, max_tokens=1000, context_window=100_000)
    payload, _ = await capture(model, tool_change_context())
    assert "inline-tools-2026-09-15" not in payload.get("betas", [])
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"
    ]
    assert [tool["name"] for tool in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


async def test_requires_both_anthropic_capabilities_for_native_tool_changes() -> None:
    payload, _ = await capture(native_model(supportsMidConvoToolChanges=True), tool_change_context())
    assert "inline-tools-2026-09-15" not in payload.get("betas", [])
    assert [tool["name"] for tool in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


async def test_mid_conversation_system_messages_are_emitted_before_the_next_assistant() -> None:
    model = native_model(supportsMidConvoSystemMessages=True)
    payload, _ = await capture(
        model,
        Context(
            messages=[
                SystemMessage("base", timestamp=0),
                UserMessage("one", timestamp=1),
                assistant_message([ToolCall("t1", "lookup", {})], stop_reason="toolUse"),
                SystemMessage("between tool call and result", timestamp=2),
                ToolResultMessage("t1", "lookup", [TextContent("ok")]),
                SystemMessage("later", timestamp=3),
                UserMessage("two", timestamp=4),
                assistant_message([TextContent("answer")]),
            ]
        ),
        {"cache_retention": "none"},
    )
    roles = [
        (message["role"], None if isinstance(message["content"], str) else message["content"][0].get("type"))
        for message in payload["messages"]
    ]
    assert roles == [
        ("user", None),
        ("assistant", "tool_use"),
        ("user", "tool_result"),
        ("user", None),
        ("system", "text"),
        ("system", "text"),
        ("assistant", "text"),
    ]
    assert payload["messages"][4]["content"][0]["text"] == "between tool call and result"
    assert payload["messages"][5]["content"][0]["text"] == "later"


# ---------------------------------------------------------------------------
# OAuth (Claude Code) request shaping
# ---------------------------------------------------------------------------


async def test_oauth_tokens_use_claude_code_identity_and_tool_names() -> None:
    tools = [empty_tool("read"), empty_tool("todowrite"), empty_tool("find"), empty_tool("my_custom_tool")]
    context = Context(
        system_prompt="System prompt.",
        messages=[
            UserMessage("go"),
            assistant_message([ToolCall("t1", "read", {})], stop_reason="toolUse"),
            ToolResultMessage("t1", "read", [TextContent("ok")]),
        ],
        tools=tools,
    )
    payload, _ = await capture(custom_model(), context, {"api_key": "sk-ant-oat01-test"}, simple=False)
    assert [block["text"] for block in payload["system"]] == [
        "You are Claude Code, Anthropic's official CLI for Claude.",
        "System prompt.",
    ]
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in payload["system"])
    assert payload["betas"] == ["claude-code-20250219", "oauth-2025-04-20"]
    # "find" is not a Claude Code tool, so it is not mapped to "Glob".
    assert [tool["name"] for tool in payload["tools"]] == ["Read", "TodoWrite", "find", "my_custom_tool"]
    assert assistant_content(payload)[0]["name"] == "Read"


async def test_oauth_without_system_prompt_still_sends_claude_code_identity() -> None:
    payload, _ = await capture(custom_model(), hello(), {"api_key": "sk-ant-oat01-test", "cache_retention": "none"})
    assert payload["system"] == [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}]


def test_claude_code_name_round_trip() -> None:
    tools = [empty_tool("todowrite"), empty_tool("read")]
    assert anthropic_messages.to_claude_code_name("todowrite") == "TodoWrite"
    assert anthropic_messages.from_claude_code_name("TodoWrite", tools) == "todowrite"
    assert anthropic_messages.from_claude_code_name("Read", tools) == "read"
    assert anthropic_messages.to_claude_code_name("find") == "find"
    assert anthropic_messages.from_claude_code_name("Glob", tools) == "Glob"
    assert anthropic_messages.from_claude_code_name("Glob") == "Glob"


# ---------------------------------------------------------------------------
# Misc request fields
# ---------------------------------------------------------------------------


async def test_metadata_tool_choice_and_max_tokens() -> None:
    tools = [empty_tool("lookup")]
    payload, _ = await capture(
        custom_model(),
        Context(messages=[UserMessage("x")], tools=tools),
        {
            "metadata": {"user_id": "u-1", "other": 1},
            "tool_choice": {"type": "tool", "name": "lookup"},
            "max_tokens": 77,
        },
        simple=False,
    )
    assert payload["metadata"] == {"user_id": "u-1"}
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert payload["max_tokens"] == 77
    assert list(payload)[:4] == ["model", "messages", "max_tokens", "stream"]


async def test_simple_tool_choice_maps_to_anthropic_type() -> None:
    payload, _ = await capture(custom_model(), hello(), {"tool_choice": "none"})
    assert payload["tool_choice"] == {"type": "none"}
    assert payload["max_tokens"] == 1024


async def test_non_string_metadata_user_id_is_ignored() -> None:
    payload, _ = await capture(custom_model(), hello(), {"metadata": {"user_id": 5}}, simple=False)
    assert "metadata" not in payload


async def test_eager_tool_input_streaming_compat_uses_legacy_beta() -> None:
    model = custom_model(compat={"supportsEagerToolInputStreaming": False})
    payload, _ = await capture(model, Context(messages=[UserMessage("x")], tools=[empty_tool("lookup")]), simple=False)
    assert "eager_input_streaming" not in payload["tools"][0]
    assert payload["betas"] == ["fine-grained-tool-streaming-2025-05-14"]
    no_tools, _ = await capture(model, hello(), simple=False)
    assert "betas" not in no_tools


# ---------------------------------------------------------------------------
# Fireworks Messages payloads (fireworks-models)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "levels"),
    [
        ("accounts/fireworks/models/deepseek-v4p1-flash", ["off", "low", "high", "max"]),
        ("accounts/fireworks/models/qwen3p8-max", ["off", "low", "medium", "xhigh"]),
        ("accounts/fireworks/models/qwen3p8-2p4t-a95b", ["off", "low", "medium", "xhigh"]),
    ],
)
async def test_sends_native_messages_effort_levels_for_fireworks(model_id: str, levels: list[str]) -> None:
    # Regression for #9323: native effort must reach Messages without budget-based fallback.
    model = model_of("fireworks", model_id)
    assert model.api == "anthropic-messages"
    assert model.compat["forceAdaptiveThinking"] is True
    assert get_supported_thinking_levels(model) == levels
    for level in levels:
        payload, _ = await capture(
            model,
            Context(messages=[UserMessage("test", timestamp=0)]),
            {"reasoning": None if level == "off" else level},
        )
        if level == "off":
            assert payload["thinking"] == {"type": "disabled"}
            assert "output_config" not in payload
        else:
            assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert payload["output_config"] == {"effort": level}


async def test_keeps_toggle_only_fireworks_messages_models_on_budget_based_thinking() -> None:
    model = model_of("fireworks", "accounts/fireworks/models/nemotron-3-ultra-nvfp4")
    assert model.compat.get("forceAdaptiveThinking") is None
    payload, _ = await capture(model, Context(messages=[UserMessage("test", timestamp=0)]), {"reasoning": "high"})
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 16384, "display": "summarized"}
    assert "output_config" not in payload


def test_fireworks_compat_for_session_affinity_and_unsupported_tool_fields() -> None:
    model = model_of("fireworks", "accounts/fireworks/models/nemotron-3-ultra-nvfp4")
    assert model.compat["sendSessionAffinityHeaders"] is True
    assert model.compat["supportsEagerToolInputStreaming"] is False
    assert model.compat["supportsCacheControlOnTools"] is False
    assert model.compat["supportsLongCacheRetention"] is False
    assert model.compat["allowEmptySignature"] is True


# ---------------------------------------------------------------------------
# Authentication (pre-generation-error)
# ---------------------------------------------------------------------------


def test_stream_simple_throws_synchronously_when_auth_is_missing() -> None:
    model = custom_model(id="test-model", provider="test-provider", base_url="https://example.invalid")
    with pytest.raises(ValueError, match="No API key for provider: test-provider"):
        stream_simple(model, normalize_context(Context(messages=[])), {})


def test_stream_simple_accepts_header_owned_auth() -> None:
    model = custom_model(provider="test-provider")
    for headers in ({"Authorization": "Bearer x"}, {"x-api-key": "k"}, {"cf-aig-authorization": "Bearer y"}):
        s = stream_simple(model, normalize_context(hello()), {"headers": headers})
        assert s is not None
    with pytest.raises(ValueError):
        stream_simple(model, normalize_context(hello()), {"headers": {"Authorization": "  "}})


async def test_stream_reports_missing_auth_as_an_error_event() -> None:
    model = custom_model(provider="test-provider")
    s = stream(model, normalize_context(hello()), {})
    events = [event.type async for event in s]
    result = await s.result()
    assert events == ["error"]
    assert result.stop_reason == "error"
    assert result.error_message == "No API key for provider: test-provider"
