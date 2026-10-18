"""Port of ``test/harness-conversations.test.ts``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import add_tool, open_harness, tool, user
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context

from pi_ai import create_models
from pi_durable import (
    ROOT_CONVERSATION_ID,
    AgentDoc,
    Harness,
    LiveDoc,
    MemoryStorage,
    ProviderDoc,
    configure,
    create_registry,
    create_session,
    define_doc,
    define_entry,
    define_extension,
    define_task,
)
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

UUID_V7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

NoteDoc = define_doc(
    kind="test.note",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: {"text": ""},
)

Message = define_entry("message")


def controlled() -> ControlledStorage:
    return ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])


async def append(conversation: Any, text: str) -> Any:
    return await conversation.commit(
        lambda tx: tx.append_entry(conversation.id, {"kind": "message", "model": [user(text)]}), context
    )


async def all_entries(conversation: Any, order: str | None = None) -> list[str]:
    texts: list[str] = []
    cursor = None
    while True:
        page = await conversation.entries({} if order is None else {"order": order}, 2, cursor, context)
        texts.extend(entry["model"][0].content for entry in page["items"])
        cursor = page.get("next")
        if cursor is None:
            return texts


def _unreachable(*_args: Any) -> None:
    raise AssertionError("unreachable")


async def test_creates_the_root_lazily_with_its_agent_change_and_init_in_one_commit() -> None:
    storage = controlled()
    harness = (await open_harness(storage, ["read", "bash"])).harness
    assert len(storage.commits) == 0

    async def init(tx: Any, id: Any) -> None:
        # The agent change applied before init.
        assert (await tx.doc(AgentDoc, id))["thinkingLevel"] == "high"
        (await tx.doc(NoteDoc, id))["text"] = "root note"

    root = await harness.root(context, agent={"thinkingLevel": "high"}, init=init)
    assert root.id == ROOT_CONVERSATION_ID
    assert len(storage.commits) == 1
    # Conversation, five built-in documents, and the init note.
    assert [write["type"] for write in storage.commits[0]] == ["conversation", *["document.create"] * 6]
    assert await harness.snapshot(LiveDoc, root.id, context) == {}
    provider = await harness.snapshot(ProviderDoc, root.id, context)
    assert provider is not None and UUID_V7.match(provider["sessionId"])
    assert await harness.snapshot(AgentDoc, root.id, context) == {"thinkingLevel": "high"}
    agent = await root.agent(context)
    assert agent["thinkingLevel"] == "high"
    assert [each.name for each in agent["tools"]] == ["read", "bash"]
    assert await harness.snapshot(NoteDoc, root.id, context) == {"text": "root note"}

    again = await harness.root(context, agent={"thinkingLevel": "low"}, init=_unreachable)
    assert again.id == root.id
    assert len(storage.commits) == 1
    await harness.close(context)


async def test_keeps_root_and_conversation_identity_and_state_across_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "session.sqlite")
    first = await open_harness(await open_sqlite_storage(path), ["read"])
    root = await first.harness.root(context)
    await root.configure({"model": {"provider": "anthropic", "modelId": "claude"}, "cwd": "/repo"}, context)
    entry = await append(root, "hello")
    child = await first.harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    fork = await root.fork(entry["id"], {"ownership": {"kind": "ownerless"}}, context)

    async def session_id(harness: Harness, id: Any) -> Any:
        state = await harness.snapshot(ProviderDoc, id, context)
        return None if state is None else state["sessionId"]

    provider_session_ids = [await session_id(first.harness, conversation.id) for conversation in (root, child, fork)]
    # Regression coverage for #10424: a fork must not inherit its parent's provider identity.
    for value in provider_session_ids:
        assert UUID_V7.match(value)
    assert len(set(provider_session_ids)) == 3
    await first.harness.close(context)
    with pytest.raises(RuntimeError, match="is closed"):
        await root.agent(context)

    # The new process installs nothing: the stored choices survive, the tools do not resolve.
    first = await open_harness(await open_sqlite_storage(path), [])
    reopened = await first.harness.root(context, init=_unreachable)
    assert reopened.id == ROOT_CONVERSATION_ID
    agent = await reopened.agent(context)
    assert agent["model"] == {"provider": "anthropic", "modelId": "claude"}
    assert agent["cwd"] == "/repo"
    assert list(agent["tools"]) == []
    assert await all_entries(reopened) == ["hello"]
    found = await first.harness.conversation(child.id, context)
    assert found is not None and found.id == child.id
    assert [await session_id(first.harness, id) for id in (root.id, child.id, fork.id)] == provider_session_ids
    reopened_fork = await first.harness.conversation(fork.id, context)
    assert reopened_fork is not None
    assert await all_entries(reopened_fork) == ["hello"]
    assert (await reopened_fork.agent(context))["model"] == {"provider": "anthropic", "modelId": "claude"}
    assert await first.harness.conversation(999, context) is None  # type: ignore[arg-type]
    await first.harness.close(context)


async def test_creates_independent_conversations_atomically_with_init_and_rolls_back_failures() -> None:
    storage = controlled()
    opened = await open_harness(storage, ["read"])
    harness, registry = opened.harness, opened.registry

    async def seed(tx: Any, id: Any) -> None:
        await tx.append_entry(id, {"kind": "message", "model": [user("seed")]})

    created = await harness.create_conversation({"ownership": {"kind": "ownerless"}, "init": seed}, context)
    assert len(storage.commits) == 1
    assert await all_entries(created) == ["seed"]

    before = len(storage.commits)

    def fail(_tx: Any, _id: Any) -> None:
        raise RuntimeError("init failed")

    with pytest.raises(RuntimeError, match="init failed"):
        await harness.create_conversation(
            {"ownership": {"kind": "ownerless"}, "agent": {"thinkingLevel": "high"}, "init": fail}, context
        )
    assert len(storage.commits) == before

    # A conversation on the default selection follows installs live.
    add_tool(registry, tool("bash"))
    assert [each.name for each in (await created.agent(context))["tools"]] == ["read", "bash"]
    await harness.close(context)


async def test_forks_at_a_concrete_entry_with_the_as_of_agent_and_applies_agent_and_init_overrides() -> None:
    opened = await open_harness(MemoryStorage(), ["read"])
    harness, registry = opened.harness, opened.registry
    legacy = tool("legacy")
    add_tool(registry, legacy)
    root = await harness.root(context)
    await root.configure({"thinkingLevel": "low"}, context)
    at = await append(root, "one")
    await root.configure({"thinkingLevel": "high", "tools": [tool("read")]}, context)
    await append(root, "two")

    child = await root.fork(at["id"], {"ownership": {"kind": "ownerless"}}, context)
    assert await harness.snapshot(AgentDoc, child.id, context) == {"thinkingLevel": "low"}
    assert await all_entries(child) == ["one"]

    async def init(tx: Any, id: Any) -> None:
        assert (await tx.doc(AgentDoc, id))["thinkingLevel"] == "minimal"

    overridden = await root.fork(
        at["id"],
        {"ownership": {"kind": "ownerless"}, "agent": {"thinkingLevel": "minimal", "tools": [legacy]}, "init": init},
        context,
    )
    assert await harness.snapshot(AgentDoc, overridden.id, context) == {"thinkingLevel": "minimal", "tools": ["legacy"]}
    assert await harness.snapshot(AgentDoc, root.id, context) == {"thinkingLevel": "high", "tools": ["read"]}

    unrelated = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    assert await root.fork(at["id"], {"ownership": {"kind": "ownerless"}}, context) is not None
    with pytest.raises(Exception, match="is not visible"):
        await unrelated.fork(at["id"], {"ownership": {"kind": "ownerless"}}, context)
    await harness.close(context)


async def test_paginates_fork_aware_history_through_deep_ancestor_caps_and_same_commit_prefixes() -> None:
    harness = (await open_harness(MemoryStorage())).harness
    root = await harness.root(context)
    await append(root, "r1")

    async def two(tx: Any) -> Any:
        return [
            await tx.append_entry(root.id, {"kind": "message", "model": [user("r2")]}),
            await tx.append_entry(root.id, {"kind": "message", "model": [user("r3")]}),
        ]

    r2, _r3 = await root.commit(two, context)
    child = await root.fork(r2["id"], {"ownership": {"kind": "ownerless"}}, context)
    c1 = await append(child, "c1")
    await append(child, "c2")
    grandchild = await child.fork(c1["id"], {"ownership": {"kind": "ownerless"}}, context)
    await append(grandchild, "g1")

    assert await all_entries(root) == ["r3", "r2", "r1"]
    assert await all_entries(child) == ["c2", "c1", "r2", "r1"]
    assert await all_entries(grandchild) == ["g1", "c1", "r2", "r1"]
    # #10546
    assert await all_entries(grandchild, "ascending") == ["r1", "r2", "c1", "g1"]
    bounded = await grandchild.entries(
        {"minEntryId": r2["id"], "maxEntryId": c1["id"], "conversationId": root.id},  # type: ignore[typeddict-unknown-key]
        10,
        None,
        context,
    )
    assert [entry["id"] for entry in bounded["items"]] == [c1["id"], r2["id"]]
    assert Message.is_(bounded["items"][0]) is True
    assert Message.is_(None) is False
    await harness.close(context)


async def _nothing(*_args: Any) -> None:
    return None


async def test_binds_commits_and_task_creation_to_the_conversation() -> None:
    harness = (await open_harness(MemoryStorage())).harness
    conversation = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    task = define_task(
        name="test.work", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": _nothing}, abort=_nothing
    )
    task_id = await conversation.commit(
        lambda tx: tx.create_task(task, {"n": 1}, {"ownership": {"kind": "conversation"}}), context
    )
    record = await harness.commit(lambda tx: tx.task(task_id), context)
    assert record is not None and record["conversationId"] == conversation.id
    with pytest.raises(Exception, match="requires options.conversationId"):
        await harness.commit(
            lambda tx: tx.create_task(task, {"n": 2}, {"ownership": {"kind": "conversation"}}), context
        )
    await harness.close(context)


async def test_runs_conversation_created_in_every_creating_commit_after_the_built_ins_and_before_agent_and_init() -> (
    None
):
    seen: list[str] = []

    async def created(tx: Any, conversation: Any) -> None:
        agent = await tx.doc(AgentDoc, conversation["id"])
        kind = "new" if "parent" not in conversation else "fork"
        seen.append(f"{conversation['id']}:{kind}:{agent.get('cwd', 'undefined')}")
        note = await tx.doc(NoteDoc, conversation["id"])
        note["text"] = note["text"] or "created"
        if agent.get("cwd") == "/fail":
            raise RuntimeError("no")

    harness = await Harness.open(
        MemoryStorage(),
        {"models": create_models(), "registry": create_registry(), "conversationCreated": created},
        context,
    )

    async def init(tx: Any, id: Any) -> None:
        seen.append(f"init:{(await tx.doc(NoteDoc, id))['text']}")

    root = await harness.root(context, agent={"cwd": "/root"}, init=init)

    # A raw creation in a commit, as in a tool, and a fork, which already has the asOf copies.
    async def raw_creation(tx: Any) -> Any:
        return (await tx.create_conversation({"ownership": {"kind": "ownerless"}}))["id"]

    raw = await harness.commit(raw_creation, context)
    entry = await append(root, "hello")
    fork = await root.fork(entry["id"], {"ownership": {"kind": "ownerless"}}, context)
    assert seen == [f"{root.id}:new:undefined", "init:created", f"{raw}:new:undefined", f"{fork.id}:fork:/root"]
    assert await harness.snapshot(NoteDoc, raw, context) == {"text": "created"}
    # A raise fails the creating commit.
    await root.configure({"cwd": "/fail"}, context)
    assert await root.fork(entry["id"], {"ownership": {"kind": "ownerless"}}, context) is not None
    failing = await append(root, "after")
    with pytest.raises(RuntimeError, match="no"):
        await root.fork(failing["id"], {"ownership": {"kind": "ownerless"}}, context)
    await harness.close(context)


async def test_replaces_whole_fields_clears_them_with_none_and_leaves_absent_fields_alone() -> None:
    opened = await open_harness(MemoryStorage())
    harness, registry = opened.harness, opened.registry
    read, bash, edit = tool("read"), tool("bash"), tool("edit")
    registry.install(define_extension("coding", tools=[read, bash, edit]))
    root = await harness.root(context)

    async def stored() -> Any:
        return await harness.snapshot(AgentDoc, root.id, context)

    async def offered() -> list[str]:
        return [each.name for each in (await root.agent(context))["tools"]]

    agent = await root.agent(context)
    assert agent["thinkingLevel"] == "off" and list(agent["tools"]) == [read, bash, edit]
    assert "model" not in agent

    await root.configure({"model": {"provider": "openai", "modelId": "gpt"}, "thinkingLevel": "medium"}, context)
    await root.configure({"instructions": "Be terse."}, context)
    assert await stored() == {
        "model": {"provider": "openai", "modelId": "gpt"},
        "thinkingLevel": "medium",
        "instructions": "Be terse.",
    }
    await root.configure({"model": None, "instructions": None}, context)
    assert await stored() == {"thinkingLevel": "medium"}

    await root.configure({"tools": {"remove": [edit]}}, context)
    assert await offered() == ["read", "bash"]
    # A new filter replaces the old one: edit is offered again.
    await root.configure({"tools": {"remove": [bash]}}, context)
    assert await offered() == ["read", "edit"]
    await root.configure({"tools": [edit, read]}, context)
    assert await offered() == ["edit", "read"]
    await root.configure({"tools": None}, context)
    assert await offered() == ["read", "bash", "edit"]
    # Names are stored without checking the registry.
    await root.configure({"tools": [tool("missing"), read]}, context)
    assert (await stored())["tools"] == ["missing", "read"]
    assert await offered() == ["read"]
    await harness.close(context)


async def test_gives_conversations_created_through_tx_their_documents() -> None:
    harness = (await open_harness(MemoryStorage(), ["read"])).harness
    root = await harness.root(
        context, agent={"model": {"provider": "faux", "modelId": "m"}, "instructions": "Main role.", "cwd": "/repo"}
    )
    owner = define_task(
        name="test.owner",
        version=1,
        initial=lambda _input: {"phase": "never"},
        phases={"never": _nothing},
        abort=_nothing,
    )

    async def create(tx: Any) -> Any:
        task_id = await tx.create_task(owner, {}, {"ownership": {"kind": "conversation"}})
        plain = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        owned = await tx.create_conversation({"ownership": {"kind": "task", "taskId": task_id}})
        # The copy exists when create_conversation() returns, so a configure() in the same callback overrides it.
        assert (await tx.doc(AgentDoc, owned["id"]))["cwd"] == "/repo"
        await configure(tx, owned["id"], {"cwd": "/worktree"})
        return {"taskId": task_id, "plain": plain["id"], "owned": owned["id"]}

    ids = await root.commit(create, context)
    assert await harness.snapshot(AgentDoc, ids["plain"], context) == {}
    assert await harness.snapshot(LiveDoc, ids["plain"], context) == {}
    provider = await harness.snapshot(ProviderDoc, ids["plain"], context)
    assert provider is not None and UUID_V7.match(provider["sessionId"])
    owned_provider = await harness.snapshot(ProviderDoc, ids["owned"], context)
    root_provider = await harness.snapshot(ProviderDoc, root.id, context)
    assert owned_provider is not None and root_provider is not None
    assert owned_provider["sessionId"] != root_provider["sessionId"]
    assert await harness.snapshot(AgentDoc, ids["owned"], context) == {
        "model": {"provider": "faux", "modelId": "m"},
        "instructions": "Main role.",
        "cwd": "/worktree",
    }

    # A later owner change does not reach the child.
    await root.configure({"thinkingLevel": "high"}, context)
    owned_agent = await harness.snapshot(AgentDoc, ids["owned"], context)
    assert owned_agent is not None and "thinkingLevel" not in owned_agent

    # A task-owned fork keeps its as-of copy of its fork parent, not its owner's agent.
    async def note(tx: Any) -> Any:
        return (await tx.append_entry(ids["plain"], {"kind": "note"}))["id"]

    at = await root.commit(note, context)

    async def fork_it(tx: Any) -> Any:
        return (await tx.fork_conversation(ids["plain"], at, {"ownership": {"kind": "task", "taskId": ids["taskId"]}}))[
            "id"
        ]

    fork = await root.commit(fork_it, context)
    assert await harness.snapshot(AgentDoc, fork, context) == {}
    assert await harness.snapshot(LiveDoc, fork, context) == {}
    fork_provider = await harness.snapshot(ProviderDoc, fork, context)
    plain_provider = await harness.snapshot(ProviderDoc, ids["plain"], context)
    assert fork_provider is not None and plain_provider is not None
    assert fork_provider["sessionId"] != plain_provider["sessionId"]
    await harness.close(context)


async def test_reads_an_absent_agent_for_conversations_a_plain_session_created_without_writing() -> None:
    storage = controlled()

    async def create(tx: Any) -> Any:
        return (await tx.create_conversation({"ownership": {"kind": "ownerless"}}))["id"]

    id = await create_session(storage).commit(create, context)
    harness = (await open_harness(storage, ["read"])).harness
    raw = await harness.conversation(id, context)
    assert raw is not None
    assert await harness.snapshot(LiveDoc, id, context) is None
    commits = len(storage.commits)
    agent = await raw.agent(context)
    assert agent["thinkingLevel"] == "off" and [each.name for each in agent["tools"]] == ["read"]
    assert len(storage.commits) == commits
    await raw.configure({"thinkingLevel": "low"}, context)
    assert await harness.snapshot(AgentDoc, id, context) == {"thinkingLevel": "low"}
    await harness.close(context)


async def test_returns_stateless_handles_and_rejects_operations_after_close() -> None:
    harness = (await open_harness(MemoryStorage())).harness
    root = await harness.root(context)
    again = await harness.root(context)
    assert again is not root
    assert again.id == root.id
    await harness.close(context)
    with pytest.raises(Exception, match="is closed"):
        await harness.root(context)
    with pytest.raises(Exception, match="is closed"):
        await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    with pytest.raises(Exception, match="is closed"):
        await harness.conversation(root.id, context)


async def test_forwards_generic_session_document_apis() -> None:
    harness = (await open_harness(MemoryStorage())).harness
    root = await harness.root(context)

    async def first(tx: Any) -> Any:
        (await tx.doc(NoteDoc, root.id))["text"] = "first"
        return await tx.append_entry(root.id, {"kind": "message", "model": [user("m")]})

    entry = await root.commit(first, context)

    async def second(tx: Any) -> None:
        (await tx.doc(NoteDoc, root.id))["text"] = "second"

    await harness.commit(second, context)
    assert await harness.snapshot(NoteDoc, root.id, context) == {"text": "second"}
    assert await harness.snapshot_as_of(NoteDoc, root.id, entry["id"], context) == {"text": "first"}
    state = await harness.document_state(NoteDoc, root.id, context)
    assert state is not None and state.value == {"text": "second"}
    state.dispose()
    await harness.close(context)
