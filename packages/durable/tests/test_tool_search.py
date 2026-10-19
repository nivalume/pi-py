"""Port of ``test/tool-search.test.ts``: deferred tools loaded by ``tool_search``."""

from __future__ import annotations

from typing import Any

from durable_harness_support import ChatSetup, add_tool, all_entries, chat_setup, context, open_chat

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, SystemMessage, TextContent, ToolResultMessage
from pi_durable import AgentDoc, MemoryStorage, define_extension, define_tool
from pi_durable.tool_search import create_tool_search_extension


def tool(name: str, description: str, *, deferred: bool = False) -> Any:
    async def execute(args: Any, api: Any, call_context: Any) -> Any:
        return {"output": [TextContent(f"{name} ran")]}

    return define_tool(
        name=name,
        description=description,
        parameters=Type.Object({"number": Type.Optional(Type.Number(description="Issue number"))}),
        execute=execute,
        deferred=True if deferred else None,
    )


DONE = faux_assistant_message([faux_text("done")])


def call(name: str, args: dict[str, Any] | None = None, id: str = "c1") -> AssistantMessage:
    return faux_assistant_message([faux_tool_call(name, args or {}, id=id)], stop_reason="toolUse")


def setup() -> ChatSetup:
    """``alpha``, the deferred ``github_get_issue`` and ``github_list_repos``, then ``omega``, and ``tool_search`` last."""
    chat = chat_setup()
    add_tool(chat.registry, tool("alpha", "Alpha tool"))
    add_tool(chat.registry, tool("github_get_issue", "Get a GitHub issue by number", deferred=True))
    add_tool(chat.registry, tool("github_list_repos", "List the repositories of a user", deferred=True))
    add_tool(chat.registry, tool("omega", "Omega tool"))
    chat.registry.install(create_tool_search_extension())
    return chat


async def offered(root: Any) -> list[str]:
    return [each.name for each in (await root.agent(context))["tools"]]


async def deferred_names(root: Any) -> list[str]:
    return [each.name for each in (await root.agent(context))["deferred"]]


def system_messages(entries: list[Any]) -> list[SystemMessage]:
    """The model messages of the ``pi.system`` entries, oldest first."""
    return [entry["model"][0] for entry in entries if entry["kind"] == "pi.system"]


def tool_result_texts(entries: list[Any]) -> list[str]:
    texts: list[str] = []
    for entry in entries:
        if entry["kind"] == "pi.tool-result":
            message = entry["model"][0]
            assert isinstance(message, ToolResultMessage)
            texts.append("".join(getattr(item, "text", "") or "" for item in message.content))
    return texts


async def model_tools(harness: Any, root: Any) -> Any:
    return ((await harness.snapshot(AgentDoc, root.id, context)) or {}).get("modelTools")


async def submit(root: Any, text: str) -> None:
    await (await root.submit({"type": "input", "content": text}, context)).wait(context)


async def test_offers_a_deferred_tool_only_once_tool_search_loads_it_by_appending_its_declaration() -> None:
    chat = setup()
    chat.faux.set_responses(
        [
            call("tool_search", {"query": "github issue", "limit": 1}),
            call("github_get_issue", {"number": 42}, "c2"),
            DONE,
        ]
    )
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    assert await offered(root) == ["alpha", "omega", "tool_search"]
    assert await deferred_names(root) == ["github_get_issue", "github_list_repos"]
    # Deferred tools stay callable by other tools, such as code mode scripts.
    assert "github_get_issue" in [each.name for each in (await root.agent(context))["callable"]]
    await submit(root, "go")
    entries = await all_entries(root)
    search, issue = tool_result_texts(entries)
    assert search == (
        "Loaded 1 tool. They are available from your next call:\n- github_get_issue: Get a GitHub issue by number"
    )
    assert issue == "github_get_issue ran"
    assert await model_tools(harness, root) == {"add": ["github_get_issue"]}
    assert await offered(root) == ["alpha", "omega", "tool_search", "github_get_issue"]
    # A later search sees only the tools not loaded yet.
    assert await deferred_names(root) == ["github_list_repos"]
    # The load appended the declaration: no tool was removed and declared again, so the cached prefix holds.
    baseline, load = system_messages(entries)
    assert [each.name for each in baseline.tools_added or []] == ["alpha", "omega", "tool_search"]
    assert [each.name for each in load.tools_added or []] == ["github_get_issue"]
    assert load.tools_removed is None
    await harness.close(context)


async def test_appends_each_loaded_tool_in_load_order_not_registry_order() -> None:
    chat = setup()
    chat.faux.set_responses(
        [
            call("tool_search", {"query": "repositories", "limit": 1}),
            call("tool_search", {"query": "issue", "limit": 1}, "c2"),
            DONE,
        ]
    )
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    await submit(root, "go")
    assert await offered(root) == ["alpha", "omega", "tool_search", "github_list_repos", "github_get_issue"]
    loads = system_messages(await all_entries(root))[1:]
    assert [[each.name for each in message.tools_added or []] for message in loads] == [
        ["github_list_repos"],
        ["github_get_issue"],
    ]
    assert all(message.tools_removed is None for message in loads)
    await harness.close(context)


async def test_records_loads_by_name_unknown_and_non_deferred_names_change_nothing_remove_wins_and_a_load_lifts_it() -> (
    None
):
    chat = setup()

    async def grow_execute(args: Any, api: Any, call_context: Any) -> Any:
        return {"output": [], "control": {"addTools": ["github_get_issue", "alpha", "ghost"]}}

    grow = define_tool(name="grow", description="grow", parameters=Type.Object({}), execute=grow_execute)
    add_tool(chat.registry, grow)
    chat.faux.set_responses([call("grow"), DONE, call("grow"), DONE])
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    await submit(root, "go")
    assert await model_tools(harness, root) == {"add": ["github_get_issue", "alpha", "ghost"]}
    assert await offered(root) == ["alpha", "omega", "tool_search", "grow", "github_get_issue"]
    # A name in both remove and add is not offered.
    get_issue = next(each for each in (await root.agent(context))["callable"] if each.name == "github_get_issue")
    await root.configure({"modelTools": {"remove": [get_issue], "add": [get_issue]}}, context)
    assert "github_get_issue" not in await offered(root)
    # Loading it again drops it from remove.
    await submit(root, "again")
    assert await model_tools(harness, root) == {"add": ["github_get_issue", "alpha", "ghost"]}
    assert "github_get_issue" in await offered(root)
    await harness.close(context)


async def test_keeps_a_loaded_tool_when_its_extension_is_gone_for_a_while_and_appends_to_an_explicit_array() -> None:
    chat = setup()
    issue = tool("github_get_issue", "Get a GitHub issue by number", deferred=True)
    chat.faux.set_responses([call("tool_search", {"query": "github issue", "limit": 1}), DONE])
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    await submit(root, "go")
    # The server goes away: its extension is installed again without the tool.
    chat.registry.install(define_extension("tool:github_get_issue", tools=[]))
    assert "github_get_issue" not in await offered(root)
    chat.registry.install(define_extension("tool:github_get_issue", tools=[issue]))
    assert "github_get_issue" in await offered(root)
    # With an explicit array, a load appends to it as before.
    tools = (await root.agent(context))["tools"]
    alpha = next(each for each in tools if each.name == "alpha")
    search = next(each for each in tools if each.name == "tool_search")
    await root.configure({"modelTools": [alpha, search]}, context)
    chat.faux.set_responses([call("tool_search", {"query": "repositories", "limit": 1}), DONE])
    await submit(root, "more")
    assert await model_tools(harness, root) == ["alpha", "tool_search", "github_list_repos"]
    await harness.close(context)


async def test_keeps_loaded_tools_per_branch_and_in_the_baseline_after_a_reset() -> None:
    chat = setup()
    chat.faux.set_responses([call("tool_search", {"query": "github issue", "limit": 1}), DONE])
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    await submit(root, "go")
    entries = await all_entries(root)
    before = await root.fork(entries[0]["id"], {"ownership": {"kind": "ownerless"}}, context)
    after = await root.fork(entries[-1]["id"], {"ownership": {"kind": "ownerless"}}, context)
    assert "github_get_issue" not in await offered(before)
    assert "github_get_issue" in await offered(after)
    await root.reset(None, context)
    chat.faux.set_responses([DONE])
    await submit(root, "fresh")
    baseline = system_messages(await all_entries(root))[-1]
    assert [each.name for each in baseline.tools_added or []] == ["alpha", "omega", "tool_search", "github_get_issue"]
    await harness.close(context)


async def test_finds_nothing_for_an_unrelated_query_and_loads_nothing() -> None:
    chat = setup()
    chat.faux.set_responses([call("tool_search", {"query": "weather forecast"}), DONE])
    opened = await open_chat(MemoryStorage(), chat)
    harness, root = opened.harness, opened.root
    await submit(root, "go")
    assert tool_result_texts(await all_entries(root)) == ["No matching tools found."]
    assert await model_tools(harness, root) is None
    await harness.close(context)
