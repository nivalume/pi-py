"""Offline tests for pi_ai.api.google_shared (ports of upstream test/google-shared-*.test.ts)."""

from __future__ import annotations

import json
from typing import Any

import pytest

import pi_ai.utils.http as http_module
from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    normalize_context,
)
from pi_ai.api.google_shared import (
    build_google_params,
    build_google_request_body,
    convert_messages,
    convert_tools,
    get_disabled_google_thinking_config,
    google_api_error,
    google_model_path,
    is_thinking_part,
    map_stop_reason,
    map_stop_reason_string,
    map_tool_choice,
    requires_tool_call_id,
    resolve_google_function_calling_mode,
    resolve_google_thinking_level,
    retain_thought_signature,
    retry_google_request,
    supports_google_strict_tool_sampling,
    uses_google_thinking_level,
)
from pi_ai.utils.http import ProviderHttpError

VALID_SIG = "AAAAAAAAAAAAAAAAAAAAAA=="


def make_model(
    id: str = "gemini-3-pro-preview",
    api: str = "google-generative-ai",
    provider: str = "google",
    input: list[str] | None = None,
    thinking_level_map: dict[str, str | None] | None = None,
) -> Model:
    return Model(
        id=id,
        name=id,
        api=api,
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=input or ["text"],
        context_window=128000,
        max_tokens=8192,
        thinking_level_map=thinking_level_map,
    )


def assistant(model: Any, content: list[Any], stop_reason: str = "toolUse") -> AssistantMessage:
    return AssistantMessage(
        content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


def make_tool(parameters: dict[str, Any]) -> Tool:
    return Tool("test_tool", "A test tool", parameters)


def all_parts(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [part for content in contents for part in content.get("parts") or []]


# ---------------------------------------------------------------------------
# convertTools (google-shared-convert-tools.test.ts)
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_strips_meta_keys_when_use_parameters(self) -> None:
        tools = [
            make_tool(
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$id": "urn:bash-tool",
                    "$comment": "A bash tool for demonstration",
                    "$defs": {"commandDef": {"type": "string"}},
                    "definitions": {"legacyDef": {"type": "number"}},
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                }
            )
        ]
        result = convert_tools(tools, True)
        assert result is not None
        decl = result[0]["functionDeclarations"][0]
        assert decl["parameters"] == {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        for key in ("$schema", "$id", "$comment", "$defs", "definitions"):
            assert key not in decl["parameters"]

    def test_recursively_strips_nested_meta_keys(self) -> None:
        tools = [
            make_tool(
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "deep": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "$id": "urn:nested",
                            "type": "string",
                        }
                    },
                }
            )
        ]
        decl = convert_tools(tools, True)[0]["functionDeclarations"][0]  # type: ignore[index]
        assert decl["parameters"] == {"type": "object", "properties": {"deep": {"type": "string"}}}

    def test_preserves_ref(self) -> None:
        tools = [
            make_tool(
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {"refProp": {"$ref": "#/$defs/someDef", "type": "string"}},
                }
            )
        ]
        decl = convert_tools(tools, True)[0]["functionDeclarations"][0]  # type: ignore[index]
        assert decl["parameters"] == {
            "type": "object",
            "properties": {"refProp": {"$ref": "#/$defs/someDef", "type": "string"}},
        }

    def test_does_not_mutate_original_parameters(self) -> None:
        original = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        snapshot = json.loads(json.dumps(original))
        convert_tools([make_tool(original)], True)
        assert original == snapshot

    def test_preserves_schema_in_parameters_json_schema(self) -> None:
        params = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        decl = convert_tools([make_tool(params)], False)[0]["functionDeclarations"][0]  # type: ignore[index]
        assert decl["parametersJsonSchema"] == params
        assert decl["name"] == "test_tool"
        assert decl["description"] == "A test tool"

    def test_handles_tools_without_schema(self) -> None:
        params = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        decl = convert_tools([make_tool(params)], True)[0]["functionDeclarations"][0]  # type: ignore[index]
        assert decl["parameters"] == params

    def test_validated_function_calling_for_strict_tools_on_gemini_3(self) -> None:
        tool = make_tool({"type": "object", "properties": {}})
        tool.constrained_sampling = {"type": "json_schema", "strict": "require"}
        assert supports_google_strict_tool_sampling("gemini-3.1-pro-preview") is True
        assert supports_google_strict_tool_sampling("gemini-2.5-pro") is False
        assert resolve_google_function_calling_mode([tool], None, True) == "VALIDATED"
        with pytest.raises(ValueError, match='Tool "test_tool" requires JSON-schema constrained sampling'):
            resolve_google_function_calling_mode([tool], None, False)

    def test_strict_tool_parameters_and_explicit_choices(self) -> None:
        tool = make_tool({"type": "object", "properties": {"a": {"type": "string"}}})
        tool.constrained_sampling = {"type": "json_schema", "strict": "prefer"}
        decl = convert_tools([tool], False, True)[0]["functionDeclarations"][0]  # type: ignore[index]
        assert decl["parametersJsonSchema"]["additionalProperties"] is False
        assert decl["parametersJsonSchema"]["required"] == ["a"]
        assert resolve_google_function_calling_mode([tool], "none", True) == "NONE"
        assert resolve_google_function_calling_mode([tool], "any", True) == "ANY"
        assert resolve_google_function_calling_mode([tool], "auto", True) == "VALIDATED"
        # Not strict when the provider cannot honor it and the tool only prefers it.
        assert resolve_google_function_calling_mode([tool], None, False) is None
        assert resolve_google_function_calling_mode([tool], "auto", False) == "AUTO"
        assert map_tool_choice("whatever") == "AUTO"

    def test_returns_none_for_empty_tool_list(self) -> None:
        assert convert_tools([]) is None
        assert convert_tools([], True) is None


# ---------------------------------------------------------------------------
# Gemini 3 unsigned tool calls (google-shared-gemini3-unsigned-tool-call.test.ts)
# ---------------------------------------------------------------------------


def unsigned_tool_call_context(model: Any, thought_signature: str | None = None) -> Context:
    return Context(
        messages=[
            UserMessage("Hi"),
            assistant(
                model,
                [
                    ToolCall("call_1", "bash", {"command": "echo hi"}, thought_signature=thought_signature),
                    ToolCall("call_2", "bash", {"command": "ls -la"}),
                ],
            ),
            ToolResultMessage("call_1", "bash", [TextContent("hi")], False),
            ToolResultMessage("call_2", "bash", [TextContent("files")], False),
        ]
    )


class _ModelRef:
    def __init__(self, api: str, provider: str, id: str) -> None:
        self.api, self.provider, self.id = api, provider, id


class TestGemini3UnsignedToolCalls:
    @pytest.mark.parametrize(
        "model",
        [
            make_model("gemini-3-pro-preview", "google-generative-ai", "google"),
            make_model("gemini-3.6-flash", "google-generative-ai", "google"),
            make_model("gemini-3-pro-preview", "google-vertex", "google-vertex"),
        ],
        ids=lambda m: f"{m.id}-{m.api}",
    )
    def test_preserves_tool_call_ids(self, model: Model) -> None:
        contents = convert_messages(model, normalize_context(unsigned_tool_call_context(model)))
        parts = all_parts(contents)
        call_ids = [p["functionCall"]["id"] for p in parts if p.get("functionCall", {}).get("id")]
        response_ids = [p["functionResponse"]["id"] for p in parts if p.get("functionResponse", {}).get("id")]
        assert call_ids == ["call_1", "call_2"]
        assert response_ids == ["call_1", "call_2"]

    def test_no_skip_validator_for_unsigned_google_tool_calls(self) -> None:
        model = make_model()
        other = _ModelRef(model.api, model.provider, "other-model")
        contents = convert_messages(model, normalize_context(unsigned_tool_call_context(other)))
        model_turn = next(c for c in contents if c["role"] == "model")
        calls = [p for p in model_turn["parts"] if "functionCall" in p]
        assert len(calls) == 2
        assert "thoughtSignature" not in calls[0]
        assert "thoughtSignature" not in calls[1]
        assert "skip_thought_signature_validator" not in json.dumps(model_turn)
        texts = [p for p in model_turn["parts"] if "text" in p and "Historical context" in p["text"]]
        assert texts == []

    def test_no_skip_validator_for_unsigned_vertex_tool_calls(self) -> None:
        model = make_model("gemini-3-pro-preview", "google-vertex", "google-vertex")
        contents = convert_messages(model, normalize_context(unsigned_tool_call_context(model)))
        model_turn = next(c for c in contents if c["role"] == "model")
        calls = [p for p in model_turn["parts"] if "functionCall" in p]
        assert len(calls) == 2
        assert all("thoughtSignature" not in call for call in calls)
        assert "skip_thought_signature_validator" not in json.dumps(model_turn)

    def test_preserves_valid_signature_for_same_provider_and_model(self) -> None:
        model = make_model()
        contents = convert_messages(model, normalize_context(unsigned_tool_call_context(model, VALID_SIG)))
        model_turn = next(c for c in contents if c["role"] == "model")
        calls = [p for p in model_turn["parts"] if "functionCall" in p]
        assert len(calls) == 2
        assert calls[0]["thoughtSignature"] == VALID_SIG
        assert "thoughtSignature" not in calls[1]

    def test_drops_invalid_base64_signature(self) -> None:
        model = make_model()
        for signature in ("not base64!", "AAA", "AAAA\n"):
            contents = convert_messages(model, normalize_context(unsigned_tool_call_context(model, signature)))
            model_turn = next(c for c in contents if c["role"] == "model")
            assert "thoughtSignature" not in model_turn["parts"][0]

    def test_no_signature_or_ids_for_non_gemini_3(self) -> None:
        model = make_model("gemini-2.5-flash")
        other = _ModelRef(model.api, model.provider, "other-model")
        contents = convert_messages(model, normalize_context(unsigned_tool_call_context(other)))
        model_turn = next(c for c in contents if c["role"] == "model")
        calls = [p for p in model_turn["parts"] if "functionCall" in p]
        responses = [p for p in all_parts(contents) if "functionResponse" in p]
        assert len(calls) == 2
        assert all("id" not in p["functionCall"] for p in calls)
        assert all("thoughtSignature" not in p for p in calls)
        assert len(responses) == 2
        assert all("id" not in p["functionResponse"] for p in responses)

    @pytest.mark.parametrize(
        ("expected", "model_id"),
        [(False, "gemini-2.5-flash"), (True, "gemini-3.6-flash"), (True, "claude-sonnet-4-5"), (True, "gpt-oss-120b")],
    )
    def test_requires_tool_call_id(self, expected: bool, model_id: str) -> None:
        assert requires_tool_call_id(model_id) is expected

    def test_normalizes_foreign_tool_call_ids(self) -> None:
        model = make_model("gemini-3-pro-preview")
        source = _ModelRef("openai-responses", "openai", "gpt-5")
        long_id = "call|" + "x" * 80
        context = Context(
            messages=[
                UserMessage("Hi"),
                assistant(source, [ToolCall(long_id, "bash", {})]),
                ToolResultMessage(long_id, "bash", [TextContent("ok")], False),
            ]
        )
        parts = all_parts(convert_messages(model, normalize_context(context)))
        call_id = next(p["functionCall"]["id"] for p in parts if "functionCall" in p)
        response_id = next(p["functionResponse"]["id"] for p in parts if "functionResponse" in p)
        assert call_id == response_id == ("call_" + "x" * 80)[:64]


# ---------------------------------------------------------------------------
# Image tool result routing (google-shared-image-tool-result-routing.test.ts)
# ---------------------------------------------------------------------------


def image_routing_context(model: Model) -> Context:
    return Context(
        messages=[
            UserMessage("read the files"),
            assistant(
                model,
                [
                    ToolCall("call_a", "read", {"path": "a.txt"}),
                    ToolCall("call_img", "read", {"path": "image.png"}),
                    ToolCall("call_b", "read", {"path": "b.txt"}),
                ],
            ),
            ToolResultMessage("call_a", "read", [TextContent("alpha text")], False),
            ToolResultMessage("call_img", "read", [ImageContent("abc", "image/png")], False),
            ToolResultMessage("call_b", "read", [TextContent("beta text")], False),
        ]
    )


class TestImageToolResultRouting:
    def test_separate_image_turn_for_gemini_2(self) -> None:
        model = make_model("gemini-2.5-flash", input=["text", "image"])
        contents = convert_messages(model, normalize_context(image_routing_context(model)))
        assert len(contents) == 5
        assert all("functionResponse" in part for part in contents[2]["parts"])
        assert contents[3]["parts"][0]["text"] == "Tool result image:"
        assert contents[3]["parts"][1]["inlineData"] == {"mimeType": "image/png", "data": "abc"}
        assert "functionResponse" in contents[4]["parts"][0]

    def test_nests_images_for_gemini_3(self) -> None:
        model = make_model("gemini-3-pro-preview", input=["text", "image"])
        contents = convert_messages(model, normalize_context(image_routing_context(model)))
        assert len(contents) == 3
        tool_turn = contents[2]
        assert len(tool_turn["parts"]) == 3
        image_response = tool_turn["parts"][1]["functionResponse"]
        assert len(image_response["parts"]) == 1
        assert image_response["parts"][0]["inlineData"] == {"mimeType": "image/png", "data": "abc"}
        assert image_response["response"] == {"output": "(see attached image)"}

    def test_whitespace_only_tool_result_text_counts_as_no_text(self) -> None:
        # https://github.com/earendil-works/pi/issues/8720
        model = make_model("gemini-3-pro-preview", input=["text", "image"])
        context = image_routing_context(model)
        for message in context.messages:
            if not isinstance(message, ToolResultMessage):
                continue
            if message.tool_call_id == "call_a":
                message.content = [TextContent("\r\n")]
            if message.tool_call_id == "call_img":
                message.content.insert(0, TextContent(" \n"))
        contents = convert_messages(model, normalize_context(context))
        responses = [part["functionResponse"]["response"] for part in contents[2]["parts"]]
        assert responses == [{"output": ""}, {"output": "(see attached image)"}, {"output": "beta text"}]

    def test_error_results_use_error_key(self) -> None:
        model = make_model("gemini-2.5-flash")
        context = Context(
            messages=[
                UserMessage("go"),
                assistant(model, [ToolCall("c1", "bash", {})]),
                ToolResultMessage("c1", "bash", [TextContent("boom")], True),
            ]
        )
        contents = convert_messages(model, normalize_context(context))
        assert contents[-1]["parts"][0]["functionResponse"] == {"name": "bash", "response": {"error": "boom"}}


# ---------------------------------------------------------------------------
# Signed empty blocks (google-shared-signed-empty-blocks.test.ts)
# ---------------------------------------------------------------------------


def signed_context(model: Any, content: list[Any]) -> Context:
    return Context(messages=[UserMessage("Hi"), assistant(model, content)])


class TestSignedEmptyBlocks:
    def test_keeps_signed_empty_thinking_block(self) -> None:
        model = make_model()
        contents = convert_messages(
            model,
            normalize_context(
                signed_context(
                    model,
                    [ThinkingContent("", thinking_signature=VALID_SIG), ToolCall("call_1", "bash", {"command": "ls"})],
                )
            ),
        )
        model_turn = next(c for c in contents if c["role"] == "model")
        signed = [p for p in model_turn["parts"] if p.get("thoughtSignature") == VALID_SIG]
        assert len(signed) == 1
        assert signed[0]["thought"] is True

    def test_keeps_signed_empty_text_block(self) -> None:
        model = make_model()
        contents = convert_messages(
            model,
            normalize_context(
                signed_context(
                    model, [TextContent("", text_signature=VALID_SIG), ToolCall("call_1", "bash", {"command": "ls"})]
                )
            ),
        )
        model_turn = next(c for c in contents if c["role"] == "model")
        assert len([p for p in model_turn["parts"] if p.get("thoughtSignature") == VALID_SIG]) == 1

    def test_drops_unsigned_empty_blocks(self) -> None:
        model = make_model()
        contents = convert_messages(
            model,
            normalize_context(
                signed_context(
                    model,
                    [ThinkingContent(""), TextContent("   "), ToolCall("call_1", "bash", {"command": "ls"})],
                )
            ),
        )
        model_turn = next(c for c in contents if c["role"] == "model")
        assert len(model_turn["parts"]) == 1
        assert "functionCall" in model_turn["parts"][0]

    def test_drops_signed_empty_blocks_from_other_model(self) -> None:
        model = make_model()
        other = _ModelRef(model.api, model.provider, "other-model")
        contents = convert_messages(
            model,
            normalize_context(
                signed_context(
                    other,
                    [
                        ThinkingContent("", thinking_signature=VALID_SIG),
                        TextContent("", text_signature=VALID_SIG),
                        ToolCall("call_1", "bash", {"command": "ls"}),
                    ],
                )
            ),
        )
        model_turn = next(c for c in contents if c["role"] == "model")
        assert len(model_turn["parts"]) == 1
        assert "functionCall" in model_turn["parts"][0]
        assert VALID_SIG not in json.dumps(model_turn)


class TestConvertMessagesMisc:
    def test_user_content_and_same_model_thinking(self) -> None:
        model = make_model(input=["text", "image"])
        context = Context(
            system_prompt="system",
            messages=[
                UserMessage("plain"),
                UserMessage([TextContent("look"), ImageContent("b64", "image/jpeg")]),
                assistant(
                    model,
                    [ThinkingContent("hmm", thinking_signature=VALID_SIG), TextContent("answer")],
                    stop_reason="stop",
                ),
            ],
        )
        contents = convert_messages(model, normalize_context(context))
        assert contents == [
            {"role": "user", "parts": [{"text": "plain"}]},
            {"role": "user", "parts": [{"text": "look"}, {"inlineData": {"mimeType": "image/jpeg", "data": "b64"}}]},
            {
                "role": "model",
                "parts": [{"thought": True, "text": "hmm", "thoughtSignature": VALID_SIG}, {"text": "answer"}],
            },
        ]

    def test_errored_assistant_turns_are_skipped(self) -> None:
        model = make_model()
        context = Context(messages=[UserMessage("a"), assistant(model, [TextContent("partial")], stop_reason="error")])
        assert convert_messages(model, normalize_context(context)) == [{"role": "user", "parts": [{"text": "a"}]}]


# ---------------------------------------------------------------------------
# Thought signatures (google-thinking-signature.test.ts)
# ---------------------------------------------------------------------------


class TestThinkingSignature:
    def test_thought_true_is_thinking(self) -> None:
        assert is_thinking_part({"thought": True}) is True
        assert is_thinking_part({"thought": True, "thoughtSignature": "opaque-signature"}) is True

    def test_signature_alone_is_not_thinking(self) -> None:
        assert is_thinking_part({"thoughtSignature": "opaque-signature"}) is False
        assert is_thinking_part({"thought": False, "thoughtSignature": "opaque-signature"}) is False

    def test_missing_signature_is_not_thinking(self) -> None:
        assert is_thinking_part({}) is False
        assert is_thinking_part({"thought": False, "thoughtSignature": ""}) is False

    def test_preserves_existing_signature(self) -> None:
        first = retain_thought_signature(None, "sig-1")
        assert first == "sig-1"
        second = retain_thought_signature(first, None)
        assert second == "sig-1"
        assert retain_thought_signature(second, "") == "sig-1"

    def test_updates_signature(self) -> None:
        assert retain_thought_signature("sig-1", "sig-2") == "sig-2"


# ---------------------------------------------------------------------------
# Thinking levels (google-thinking-level-map.test.ts, resolver part)
# ---------------------------------------------------------------------------


class TestThinkingLevels:
    def test_resolves_supported_levels_and_mappings(self) -> None:
        for level in ("minimal", "low", "medium", "high"):
            assert resolve_google_thinking_level(make_model("gemini-3.7-flash", thinking_level_map={}), level) == level
        mapped = {
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "MINIMAL": "minimal",
            "LOW": "low",
            "MEDIUM": "medium",
            "HIGH": "high",
        }
        for value, expected in mapped.items():
            model = make_model("gemini-3.7-flash", thinking_level_map={"high": value, "xhigh": value, "max": value})
            for level in ("high", "xhigh", "max"):
                assert resolve_google_thinking_level(model, level) == expected

        invalid = make_model("gemini-3.7-flash", thinking_level_map={"xhigh": "extreme"})
        invalid.provider = "test-google"
        with pytest.raises(
            ValueError,
            match="Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme",
        ):
            resolve_google_thinking_level(invalid, "xhigh")
        plain = make_model("gemini-3.7-flash", thinking_level_map={})
        plain.provider = "test-google"
        with pytest.raises(
            ValueError,
            match="Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined",
        ):
            resolve_google_thinking_level(plain, "max")

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("gemini-3-flash-preview", True),
            ("gemini-3.1-pro-preview", True),
            ("gemini-3.8-flash", True),
            ("gemini-4-argon", True),
            ("gemini-12-x", True),
            ("gemini-flash-latest", True),
            ("gemini-flash-lite-latest", True),
            ("gemma-4-31b-it", True),
            ("gemma4-e2b", True),
            ("gemini-2.5-flash", False),
            ("gemini-2.5-pro", False),
            ("gemini-3-image", False),
        ],
    )
    def test_uses_google_thinking_level(self, model_id: str, expected: bool) -> None:
        assert uses_google_thinking_level(make_model(model_id)) is expected

    def test_disabled_thinking_config(self) -> None:
        assert get_disabled_google_thinking_config(make_model("gemini-2.5-flash")) == {"thinkingBudget": 0}
        assert get_disabled_google_thinking_config(make_model("gemini-3-flash-preview", thinking_level_map={})) == {
            "thinkingBudget": 0
        }
        no_off = make_model("gemini-3-pro-preview", thinking_level_map={"off": None, "minimal": None})
        assert get_disabled_google_thinking_config(no_off) == {"thinkingLevel": "LOW"}


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------


def test_map_stop_reason() -> None:
    assert map_stop_reason("STOP") == "stop"
    assert map_stop_reason("MAX_TOKENS") == "length"
    for reason in ("SAFETY", "MALFORMED_FUNCTION_CALL", "OTHER", "TOO_MANY_TOOL_CALLS", "NO_IMAGE"):
        assert map_stop_reason(reason) == "error"
    with pytest.raises(ValueError, match="Unhandled stop reason: SOMETHING_NEW"):
        map_stop_reason("SOMETHING_NEW")
    assert map_stop_reason_string("STOP") == "stop"
    assert map_stop_reason_string("MAX_TOKENS") == "length"
    assert map_stop_reason_string("SOMETHING_NEW") == "error"


# ---------------------------------------------------------------------------
# Retries (google-shared-retry.test.ts)
# ---------------------------------------------------------------------------


def google_api_error_like(status: int) -> ProviderHttpError:
    """Shaped like @google/genai's ApiError: has a status, but no headers."""
    return ProviderHttpError(f"got status: {status}", status=status)


class _Request:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(seconds: float, signal: Any = None) -> None:
        delays.append(seconds)

    monkeypatch.setattr(http_module, "sleep", fake_sleep)
    return delays


class TestRetryGoogleRequest:
    async def test_retries_headerless_error_with_retryable_status(self, no_backoff: list[float]) -> None:
        request = _Request([google_api_error_like(429), "ok"])
        assert await retry_google_request(request, {"max_retries": 1}) == "ok"
        assert request.calls == 2
        assert len(no_backoff) == 1 and 0.3 < no_backoff[0] <= 0.5

    async def test_does_not_retry_without_max_retries(self) -> None:
        error = google_api_error_like(429)
        request = _Request([error])
        with pytest.raises(ProviderHttpError) as info:
            await retry_google_request(request)
        assert info.value is error
        assert request.calls == 1

    async def test_does_not_retry_non_retryable_status(self) -> None:
        error = google_api_error_like(400)
        request = _Request([error])
        with pytest.raises(ProviderHttpError) as info:
            await retry_google_request(request, {"max_retries": 2})
        assert info.value is error
        assert request.calls == 1

    async def test_does_not_retry_transport_errors(self, no_backoff: list[float]) -> None:
        error = ProviderHttpError("Connection error: refused")
        request = _Request([error, "ok"])
        with pytest.raises(ProviderHttpError) as info:
            await retry_google_request(request, {"max_retries": 2})
        assert info.value is error
        assert request.calls == 1


# ---------------------------------------------------------------------------
# SDK wire format
# ---------------------------------------------------------------------------


class TestRequestBody:
    def test_model_paths(self) -> None:
        assert google_model_path("gemini-2.5-flash", False) == "models/gemini-2.5-flash"
        assert google_model_path("models/x", False) == "models/x"
        assert google_model_path("tunedModels/x", False) == "tunedModels/x"
        assert google_model_path("gemini-2.5-flash", True) == "publishers/google/models/gemini-2.5-flash"
        assert google_model_path("meta/llama-3", True) == "publishers/meta/models/llama-3"
        assert google_model_path("projects/p/locations/l/endpoints/e", True) == "projects/p/locations/l/endpoints/e"
        with pytest.raises(ValueError, match="invalid model parameter"):
            google_model_path("a?b", False)

    def test_params_and_body_split_like_the_sdk(self) -> None:
        model = make_model("gemini-3-pro-preview")
        tool = Tool("bash", "Run bash", {"type": "object", "properties": {"command": {"type": "string"}}})
        context = normalize_context(Context(system_prompt="Be terse.", messages=[UserMessage("hi")], tools=[tool]))
        params = build_google_params(
            model,
            context,
            {
                "temperature": 0.2,
                "max_tokens": 100,
                "tool_choice": "any",
                "thinking": {"enabled": True, "level": "HIGH"},
            },
        )
        assert params["model"] == "gemini-3-pro-preview"
        assert params["config"] == {
            "temperature": 0.2,
            "maxOutputTokens": 100,
            "systemInstruction": "Be terse.",
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "bash",
                            "description": "Run bash",
                            "parametersJsonSchema": {"type": "object", "properties": {"command": {"type": "string"}}},
                        }
                    ]
                }
            ],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "HIGH"},
        }
        model_path, body = build_google_request_body(params, vertex=False)
        assert model_path == "models/gemini-3-pro-preview"
        assert body == {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "Be terse."}]},
            "tools": params["config"]["tools"],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 100,
                "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "HIGH"},
            },
        }

    def test_empty_config_and_errors(self) -> None:
        _, body = build_google_request_body({"model": "m", "contents": "hello", "config": {}}, vertex=True)
        assert body == {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {}}
        with pytest.raises(ValueError, match="contents are required"):
            build_google_request_body({"model": "m", "contents": [], "config": {}}, vertex=False)
        with pytest.raises(ValueError, match="labels parameter is only supported in Gemini Enterprise Agent Platform"):
            build_google_request_body({"model": "m", "contents": "x", "config": {"labels": {"a": "b"}}}, vertex=False)
        _, body = build_google_request_body(
            {"model": "m", "contents": "x", "config": {"labels": {"a": "b"}, "modelSelectionConfig": {"x": 1}}},
            vertex=True,
        )
        assert body["labels"] == {"a": "b"}
        assert body["generationConfig"] == {"modelConfig": {"x": 1}}

    def test_api_error_wording(self) -> None:
        body = {"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}}
        error = google_api_error(403, json.dumps(body), "application/json; charset=UTF-8")
        assert str(error) == json.dumps(body, separators=(",", ":"))
        assert error.status == 403
        assert error.headers is None
        text_error = google_api_error(502, "bad gateway", "text/html")
        assert json.loads(str(text_error)) == {
            "error": {"message": "bad gateway", "code": 502, "status": "Bad Gateway"}
        }
