"""Port of ``test/harness-prompt.test.ts``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from durable_harness_support import add_section, open_harness, user
from durable_session_support import context

from pi_ai import AbortController, Type
from pi_ai.types import SystemMessage, Tool
from pi_ai.utils.transcript import get_current_tools, to_tool_declaration
from pi_chord.context import with_abort_signal
from pi_durable import MemoryStorage, PromptSection, SystemEntry, create_registry, define_extension, wrap_section
from pi_durable.harness.agent import resolve_agent, resolve_settings
from pi_durable.harness.prompt import plan_system_entries, render_sections, replay_sections


async def apply(conversation: Any, desired: dict[str, str]) -> list[dict[str, Any]]:
    """Plan against the current context, append the plan, and check that replay then yields ``desired`` in order."""
    drafts = plan_system_entries(await conversation.context(context), dict(desired), [], 7)

    async def append(tx: Any) -> None:
        for draft in drafts:
            await tx.append_entry(SystemEntry, conversation.id, draft)

    await conversation.commit(append, context)
    replayed = replay_sections((await conversation.context(context))["messages"])
    assert list(replayed.items()) == list(desired.items())
    planned: list[dict[str, Any]] = []
    for draft in drafts:
        message = draft["model"][0]
        assert isinstance(message, SystemMessage)
        assert message.content == "" and message.timestamp == 7
        edits = draft.get("edits")
        if edits is None:
            planned.append({"sections": message.sections})
        else:
            assert all(edit["action"] == "omit" for edit in edits)
            planned.append({"sections": message.sections, "omit": [edit["target"] for edit in edits]})
    return planned


async def root() -> Any:
    opened = await open_harness(MemoryStorage())
    return await opened.harness.root(context)


async def last_system_id(conversation: Any) -> Any:
    page = await conversation.entries({}, 100, None, context)
    return next(entry for entry in page["items"] if entry["kind"] == "pi.system")["id"]


async def marker(conversation: Any, head: Any) -> Any:
    async def change(tx: Any) -> Any:
        return (await tx.append_entry(conversation.id, {"kind": "summary", "head": head, "model": [user("summary")]}))[
            "id"
        ]

    return await conversation.commit(change, context)


async def _nothing(*_args: Any) -> None:
    return None


class _Read:
    snapshot = staticmethod(_nothing)
    snapshot_as_of = staticmethod(_nothing)


prompt_input: Any = {
    "conversationId": 1,
    "agent": {"thinkingLevel": "off", "extensions": [], "tools": [], "callable": [], "deferred": [], "sections": []},
    "env": None,
    "shown": {},
    "read": _Read(),
}


async def test_renders_sections_in_order_with_tags_omissions_wrappers_and_failures() -> None:
    registry = create_registry()
    add_section(registry, "preamble", lambda *_: "You are helpful.", tag=False)

    async def cwd(*_args: Any) -> str:
        return "/repo"

    def failing(*_args: Any) -> str:
        raise RuntimeError("render failed")

    def new_failing(*_args: Any) -> str:
        raise RuntimeError("also failed")

    add_section(registry, "cwd", cwd)
    add_section(registry, "skipped", lambda *_: None)
    add_section(registry, "failing", failing)
    add_section(registry, "new-failing", new_failing)

    def wrap_cwd(inner: PromptSection) -> PromptSection:
        async def render(value: Any, ctx: Any) -> str:
            return f"{await inner.render(value, ctx)} (git)"  # type: ignore[misc]

        return dataclasses.replace(inner, render=render)

    registry.install(define_extension("git", wraps=[wrap_section("cwd", wrap_cwd)]))
    reports: list[BaseException] = []
    shown = {"failing": "<failing>\nold\n</failing>", "cwd": "stale"}

    def report(error: BaseException, _extension: str) -> None:
        raise error

    agent = resolve_agent(None, registry.snapshot(), resolve_settings(None), report)
    desired = await render_sections(
        agent["sections"], prompt_input, shown, lambda error, _s: reports.append(error), context
    )
    assert list(desired.items()) == [
        ("preamble", "You are helpful."),
        ("cwd", "<cwd>\n/repo (git)\n</cwd>"),
        ("failing", "<failing>\nold\n</failing>"),
    ]
    assert [str(error) for error in reports] == ["render failed", "also failed"]


async def test_propagates_section_errors_after_cancellation() -> None:
    controller = AbortController()
    controller.abort(RuntimeError("cancelled"))
    cancelled = with_abort_signal(controller.signal, context)

    def fail(*_args: Any) -> str:
        raise RuntimeError("cancelled")

    failing = PromptSection(key="a", render=fail)
    with pytest.raises(RuntimeError, match="cancelled"):
        await render_sections([failing], prompt_input, {}, lambda *_: None, cancelled)


def test_replays_sections_in_place_deletes_on_none_and_appends_re_additions() -> None:
    def system(sections: dict[str, str | None]) -> SystemMessage:
        return SystemMessage("", sections=sections, timestamp=1)

    shown = replay_sections(
        [system({"a": "1", "b": "2", "c": "3"}), user("x"), system({"b": "20", "a": None}), system({"a": "10"})]
    )
    assert list(shown.items()) == [("b", "20"), ("c", "3"), ("a", "10")]


async def test_emits_minimal_value_patches_removals_and_additions() -> None:
    conversation = await root()
    assert await apply(conversation, {"a": "1", "b": "2", "c": "3"}) == [{"sections": {"a": "1", "b": "2", "c": "3"}}]
    assert await apply(conversation, {"a": "1", "b": "20", "c": "3", "d": "4"}) == [{"sections": {"b": "20", "d": "4"}}]
    assert await apply(conversation, {"a": "1", "c": "3", "d": "4"}) == [{"sections": {"b": None}}]
    assert await apply(conversation, {"a": "1", "c": "3", "d": "4"}) == []
    assert await apply(conversation, {}) == [{"sections": {"a": None, "c": None, "d": None}}]


async def test_rewrites_order_only_changes_and_re_additions_as_two_entries() -> None:
    conversation = await root()
    await apply(conversation, {"a": "1", "b": "2"})
    first = await apply(conversation, {"b": "2", "a": "1"})
    assert first == [{"sections": {"a": None, "b": None}}, {"sections": {"b": "2", "a": "1"}}]
    assert list(first[1]["sections"]) == ["b", "a"]
    readded = await root()
    await apply(readded, {"a": "1", "b": "2", "c": "3"})
    assert await apply(readded, {"a": "1", "c": "3"}) == [{"sections": {"b": None}}]
    # Patching would append ``b`` after ``c``.
    assert await apply(readded, {"a": "1", "b": "2", "c": "3"}) == [
        {"sections": {"a": None, "c": None}},
        {"sections": {"a": "1", "b": "2", "c": "3"}},
    ]


async def test_rebaselines_after_a_head_marker_omitting_retained_deltas_on_both_sides_of_it() -> None:
    conversation = await root()
    await apply(conversation, {"a": "1", "b": "2"})
    await conversation.commit(
        lambda tx: tx.append_entry(conversation.id, {"kind": "pi.user", "model": [user("hi")]}), context
    )
    await apply(conversation, {"a": "1", "b": "20"})
    delta = await last_system_id(conversation)
    # The head keeps the delta but cuts its baseline: replay alone would show only ``b``.
    await marker(conversation, delta)
    assert await apply(conversation, {"a": "1", "b": "20"}) == [{"sections": {"a": "1", "b": "20"}, "omit": [delta]}]
    baseline = await last_system_id(conversation)
    # A system entry follows the marker now, so later changes are ordinary patches.
    assert await apply(conversation, {"a": "1", "b": "21"}) == [{"sections": {"b": "21"}}]
    after = await last_system_id(conversation)
    # A second marker keeps deltas from both sides of the first one.
    await marker(conversation, delta)
    assert await apply(conversation, {"a": "1", "b": "21"}) == [
        {"sections": {"a": "1", "b": "21"}, "omit": [delta, baseline, after]}
    ]


async def test_writes_a_complete_post_head_baseline_even_when_replay_already_matches() -> None:
    conversation = await root()
    await apply(conversation, {"a": "1"})
    baseline = await last_system_id(conversation)
    await marker(conversation, baseline)
    assert await apply(conversation, {"a": "1"}) == [{"sections": {"a": "1"}, "omit": [baseline]}]
    await marker(conversation, "self")
    assert await apply(conversation, {}) == [{"sections": {}}]
    assert await apply(conversation, {}) == []


def declaration(name: str, description: str | None = None) -> Tool:
    return Tool(name, name if description is None else description, Type.Object({}))


async def apply_tools(
    conversation: Any, tools: list[Tool], sections: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Plan tools only, append the plan, check that replay offers ``tools`` in order, and return each change."""
    drafts = plan_system_entries(await conversation.context(context), dict(sections or {}), tools, 7)

    async def append(tx: Any) -> None:
        for draft in drafts:
            await tx.append_entry(SystemEntry, conversation.id, draft)

    await conversation.commit(append, context)
    offered = get_current_tools((await conversation.context(context))["messages"])
    assert offered == [to_tool_declaration(tool) for tool in tools]
    changes: list[dict[str, Any]] = []
    for draft in drafts:
        message = draft["model"][0]
        change: dict[str, Any] = {}
        if message.tools_removed is not None:
            change["removed"] = [tool.name for tool in message.tools_removed]
        if message.tools_added is not None:
            change["added"] = [tool.name for tool in message.tools_added]
        if message.sections is not None:
            change["sections"] = message.sections
        changes.append(change)
    return changes


async def test_adds_removes_replaces_changed_declarations_and_rewrites_the_order_when_needed() -> None:
    conversation = await root()
    a, b, c = declaration("a"), declaration("b"), declaration("c")
    assert await apply_tools(conversation, [a, b]) == [{"added": ["a", "b"]}]
    assert await apply_tools(conversation, [a, b]) == []
    assert await apply_tools(conversation, [a, b, c]) == [{"added": ["c"]}]
    assert await apply_tools(conversation, [a, c]) == [{"removed": ["b"]}]
    # A changed declaration at the end is removed and re-added in place.
    c2 = declaration("c", "changed")
    assert await apply_tools(conversation, [a, c2]) == [{"removed": ["c"], "added": ["c"]}]
    # A changed declaration in the middle would move to the end, so the whole order is rewritten.
    a2 = declaration("a", "changed")
    assert await apply_tools(conversation, [a2, c2]) == [{"removed": ["a", "c"], "added": ["a", "c"]}]
    # Order-only change.
    assert await apply_tools(conversation, [c2, a2]) == [{"removed": ["a", "c"], "added": ["c", "a"]}]
    assert await apply_tools(conversation, []) == [{"removed": ["c", "a"]}]


async def test_puts_tool_changes_on_the_last_section_entry_and_re_declares_every_tool_after_a_head_cut() -> None:
    conversation = await root()
    a, b = declaration("a"), declaration("b")
    assert await apply_tools(conversation, [a], {"x": "1", "y": "2"}) == [
        {"added": ["a"], "sections": {"x": "1", "y": "2"}}
    ]
    # Section order changes need two entries; the tool change rides on the second.
    assert await apply_tools(conversation, [a, b], {"y": "2", "x": "1"}) == [
        {"sections": {"x": None, "y": None}},
        {"added": ["b"], "sections": {"y": "2", "x": "1"}},
    ]
    await marker(conversation, "self")
    assert await apply_tools(conversation, [a, b], {"y": "2", "x": "1"}) == [
        {"added": ["a", "b"], "sections": {"y": "2", "x": "1"}}
    ]
