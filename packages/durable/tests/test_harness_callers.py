"""Port of ``test/harness-callers.test.ts``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from durable_harness_support import ChatSetup, add_tool, chat_setup, open_chat
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.serialization import to_json_value
from pi_ai.types import AssistantMessage, TextContent
from pi_chord.context import Context
from pi_durable import AgentDoc, MemoryStorage, define_tool


def tool(name: str, **extra: Any) -> Any:
    async def execute(_args: Any, _api: Any, _context: Context) -> Any:
        return {"output": [TextContent(f"{name} ran")]}

    options: dict[str, Any] = {"execute": execute, **extra}
    return define_tool(name=name, description=name, parameters=Type.Object({}), **options)


DONE = faux_assistant_message([faux_text("done")])


def call(name: str, id: str = "c1") -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, {}, id=id)], stop_reason="toolUse")


class Prober:
    """A tool that calls each of ``names`` as a nested call and records what it got back."""

    def __init__(self, names: Sequence[str]) -> None:
        self.received: dict[str, Any] = {}

        async def execute(_args: Any, api: Any, call_context: Context) -> Any:
            for name in names:
                self.received[name] = await api.execute_tool(name, {}, call_context)
            return {}

        self.registration = define_tool(name="probe", description="probe", parameters=Type.Object({}), execute=execute)


async def offered(root: Any) -> list[str]:
    return [each.name for each in (await root.agent(context))["tools"]]


async def callable_tools(root: Any) -> list[str]:
    return [each.name for each in (await root.agent(context))["callable"]]


def setup_with(tools: Sequence[Any]) -> ChatSetup:
    setup = chat_setup()
    for each in tools:
        add_tool(setup.registry, each)
    return setup


async def go(root: Any) -> None:
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)


async def test_offers_the_model_only_tools_it_may_call_and_lets_tools_call_only_tools_they_may_call() -> None:
    script = tool("script", callers=["model"])
    hidden = tool("hidden", callers=["tools"])
    both = tool("both")
    probe = Prober(["script", "hidden", "both"])
    setup = setup_with([script, hidden, both, probe.registration])
    setup.faux.set_responses([call("probe"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    assert await offered(root) == ["script", "both", "probe"]
    assert await callable_tools(root) == ["hidden", "both", "probe"]
    await go(root)
    assert probe.received["script"]["diagnostics"][0].get("code") == "tool_unavailable"
    assert probe.received["hidden"]["isError"] is False
    assert probe.received["hidden"]["structuredOutput"] == "hidden ran"
    assert probe.received["both"]["isError"] is False
    assert probe.received["both"]["structuredOutput"] == "both ran"
    await harness.close(context)


async def test_narrows_what_the_model_is_offered_with_model_tools_leaving_the_rest_callable_by_tools() -> None:
    read = tool("read")
    bash = tool("bash")
    probe = Prober(["read", "bash"])
    setup = setup_with([read, bash, probe.registration])
    setup.faux.set_responses([call("probe"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"modelTools": [probe.registration]}, context)
    assert await offered(root) == ["probe"]
    assert await callable_tools(root) == ["read", "bash", "probe"]
    await go(root)
    assert probe.received["read"]["isError"] is False
    assert probe.received["bash"]["isError"] is False

    await root.configure({"modelTools": {"remove": [bash]}}, context)
    assert await offered(root) == ["read", "probe"]
    assert ((await harness.snapshot(AgentDoc, root.id, context)) or {}).get("modelTools") == {"remove": ["bash"]}
    await root.configure({"modelTools": None}, context)
    assert await offered(root) == ["read", "bash", "probe"]
    await harness.close(context)


async def test_never_offers_a_tool_that_is_not_enabled_nor_lets_tools_call_it() -> None:
    read = tool("read")
    bash = tool("bash")
    probe = Prober(["bash"])
    setup = setup_with([read, bash, probe.registration])
    setup.faux.set_responses([call("probe"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    # modelTools cannot bring back what tools disabled.
    await root.configure({"tools": {"remove": [bash]}, "modelTools": [bash, probe.registration]}, context)
    assert await offered(root) == ["probe"]
    assert await callable_tools(root) == ["read", "probe"]
    await go(root)
    assert probe.received["bash"]["diagnostics"][0].get("code") == "tool_unavailable"
    await harness.close(context)


async def test_answers_a_model_issued_call_of_a_tool_it_was_not_offered_as_unavailable() -> None:
    read = tool("read")
    bash = tool("bash")
    setup = setup_with([read, bash])
    setup.faux.set_responses([call("bash"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"modelTools": [read]}, context)
    await go(root)
    entries = await root.entries({}, 100, None, context)
    result = next((entry for entry in entries["items"] if entry["kind"] == "pi.tool-result"), None)
    assert result is not None
    assert "Tool bash is not available" in json.dumps(to_json_value(result["model"]))
    await harness.close(context)


async def test_applies_add_tools_to_model_tools_too_so_the_model_is_offered_the_added_tools() -> None:
    async def grow_execute(_args: Any, _api: Any, _context: Context) -> Any:
        return {"output": [], "control": {"addTools": ["extra"]}}

    extra = tool("extra")
    grow = tool("grow", execute=grow_execute)
    setup = setup_with([extra, grow])
    setup.faux.set_responses([call("grow"), DONE])
    opened = await open_chat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.configure({"modelTools": [grow]}, context)
    await go(root)
    assert ((await harness.snapshot(AgentDoc, root.id, context)) or {}).get("modelTools") == ["grow", "extra"]
    assert await offered(root) == ["grow", "extra"]
    await harness.close(context)
