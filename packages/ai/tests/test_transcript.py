from __future__ import annotations

from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    Type,
    UserMessage,
    collapse_system_messages,
    content_text,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_system_message_text,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    render_system_message_update,
    resolve_transcript_tools,
)


def tool(name: str, description: str | None = None) -> Tool:
    return Tool(name, description if description is not None else f"{name} tool", Type.Object({}))


def _transcript():
    return normalize_context(
        Context(
            messages=[
                SystemMessage(
                    "base", sections={"a": "<a>1</a>", "b": "<b>1</b>"}, tools_added=[tool("first")], timestamp=10
                ),
                UserMessage("hello", timestamp=11),
                SystemMessage("also do this", timestamp=12),
                AssistantMessage([TextContent("ok")], api="x", provider="y", model="z", timestamp=13),
                SystemMessage(
                    "",
                    sections={"a": "<a>2</a>", "b": None, "c": "<c>1</c>"},
                    tools_removed=[ToolReference("first")],
                    tools_added=[tool("second")],
                    timestamp=14,
                ),
            ]
        )
    )


def test_replays_content_sections_and_tools_into_one_leading_message() -> None:
    transcript = _transcript()
    current = get_current_system_message(transcript.messages)
    assert current == SystemMessage(
        "base\n\nalso do this",
        sections={"a": "<a>2</a>", "c": "<c>1</c>"},
        tools_added=[tool("second")],
        timestamp=10,
    )
    assert get_current_system_prompt(transcript.messages) == "base\n\nalso do this\n\n<a>2</a>\n\n<c>1</c>"


def test_collapse_keeps_only_non_system_messages_after_replayed_head() -> None:
    collapsed = collapse_system_messages(_transcript())
    assert [message.role for message in collapsed.messages] == ["system", "user", "assistant"]
    assert collapse_system_messages(collapsed).messages == collapsed.messages


def test_replay_without_system_messages_is_empty() -> None:
    context = normalize_context(Context(messages=[UserMessage("hi", timestamp=1)]))
    assert get_current_system_message(context.messages) is None
    assert get_current_system_prompt(context.messages) == ""
    assert collapse_system_messages(context).messages == context.messages


def test_late_full_patch_without_leading_message_replays_as_prompt() -> None:
    context = normalize_context(
        Context(
            messages=[
                UserMessage("old session", timestamp=1),
                SystemMessage("", sections={"preamble": "You are pi."}, tools_added=[tool("x")], timestamp=2),
            ]
        )
    )
    assert get_current_system_prompt(context.messages) == "You are pi."
    head = collapse_system_messages(context).messages[0]
    assert head.role == "system" and head.tools_added == [tool("x")]


def test_renders_complete_prompts_and_framed_updates() -> None:
    transcript = _transcript()
    assert get_system_message_text(transcript.messages[0]) == "base\n\n<a>1</a>\n\n<b>1</b>"
    assert render_system_message_update(transcript.messages[4]) == "\n\n".join(
        [
            'Updated system prompt section "a":\n\n<a>2</a>',
            'Removed system prompt section "b".',
            'Updated system prompt section "c":\n\n<c>1</c>',
        ]
    )


def test_normalizes_legacy_prompt_and_tool_fields() -> None:
    messages = [UserMessage("hi", timestamp=1)]
    assert normalize_context(Context(messages=messages)).messages == messages
    assert normalize_context(Context(messages=messages, system_prompt="", tools=[])).messages == messages
    assert normalize_context(Context(messages=messages, system_prompt="be brief", tools=[tool("a")])).messages == [
        SystemMessage("be brief", tools_added=[tool("a")], timestamp=0),
        *messages,
    ]


def test_normalize_accepts_plain_dicts_in_pi_json_format() -> None:
    context = normalize_context(
        {
            "systemPrompt": "sys",
            "messages": [
                {"role": "user", "content": "hi", "timestamp": 1},
                {
                    "role": "toolResult",
                    "toolCallId": "c1",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "x"}],
                    "isError": False,
                    "timestamp": 2,
                },
            ],
        }
    )
    assert [message.role for message in context.messages] == ["system", "user", "toolResult"]
    assert context.messages[2].tool_call_id == "c1"


def test_compares_tool_declarations() -> None:
    assert declarations_equal(tool("a"), tool("a"))
    assert not declarations_equal(tool("a"), tool("a", "changed"))


def test_tool_state_changes_treat_changed_definitions_as_removal_plus_addition() -> None:
    changes = get_tool_state_changes([tool("a"), tool("b")], [tool("b", "changed"), tool("c")])
    assert changes.tools_added == [tool("b", "changed"), tool("c")]
    assert changes.tools_removed == [ToolReference("a"), ToolReference("b")]
    unchanged = get_tool_state_changes([tool("a")], [tool("a")])
    assert unchanged.tools_added == [] and unchanged.tools_removed == []


def test_detects_non_additive_tool_history_and_redefinitions() -> None:
    transcript = _transcript()
    assert has_non_additive_tool_changes(transcript.messages)
    assert not has_tool_redefinitions(transcript.messages)
    additive = [
        SystemMessage("", tools_added=[tool("a")], timestamp=1),
        SystemMessage("", tools_added=[tool("b")], timestamp=2),
    ]
    assert not has_non_additive_tool_changes(additive)
    redeclared = [
        SystemMessage("", tools_added=[tool("a")], timestamp=1),
        SystemMessage("", tools_added=[tool("a", "changed")], timestamp=2),
    ]
    assert has_non_additive_tool_changes(redeclared)
    assert has_tool_redefinitions(redeclared)


def test_resolve_transcript_tools_anchors_only_additive_histories() -> None:
    additive = [
        SystemMessage("", tools_added=[tool("a")], timestamp=1),
        SystemMessage("", tools_added=[tool("b")], timestamp=2),
    ]
    anchored = resolve_transcript_tools(additive, True)
    assert anchored.anchors_additions and [t.name for t in anchored.request_tools] == ["a"]
    assert [t.name for t in resolve_transcript_tools(additive, False).request_tools] == ["a", "b"]
    assert not resolve_transcript_tools(_transcript().messages, True).anchors_additions


def test_content_text() -> None:
    content = [
        ThinkingContent("reasoning"),
        TextContent("first"),
        ToolCall("1", "read", {}),
        TextContent("second"),
    ]
    assert content_text(content) == "first\nsecond"
    assert content_text(content, "") == "firstsecond"
    assert content_text("hello") == "hello"
    tool_result = [TextContent("first"), ImageContent("...", "image/png"), TextContent("second")]
    assert content_text(tool_result, "") == "firstsecond"
