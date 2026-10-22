"""``RulesPolicy`` over the coding tools: rules, modes, rules added by approvals, how permissions follow forks and owned
conversations, validation, and the memoized decision."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from durable_harness_support import ChatSetup, aborted, all_entries, chat_setup, wait_for
from durable_session_support import context

from pi_ai import faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import TextContent, ToolResultMessage
from pi_chord.context import Context, await_with_context
from pi_durable import (
    Harness,
    MemoryStorage,
    PermissionDefaults,
    RuleError,
    RulesPolicy,
    ToolResultEntry,
    define_extension,
    define_task,
    watch_events,
)
from pi_durable.approval import ApprovalCall
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool


def policy(mode: Any = "default", allow: tuple[str, ...] = (), ask: tuple[str, ...] = (), deny: tuple[str, ...] = ()):
    rules: Any = {"allow": list(allow), "ask": list(ask), "deny": list(deny)}
    return RulesPolicy(PermissionDefaults(mode=mode, rules=rules))


class Workspace:
    """A Harness with the coding tools in a temporary directory and an approval policy."""

    def __init__(self, directory: Path, setup: ChatSetup, harness: Harness, root: Any) -> None:
        self.directory = directory
        self.setup = setup
        self.harness = harness
        self.root = root

    @staticmethod
    async def open(directory: Path, approval: Any, storage: Any = None) -> Workspace:
        setup = chat_setup()
        setup.registry.install(
            define_extension(
                "coding",
                tools=[create_read_tool(), create_write_tool(), create_edit_tool(), create_bash_tool()],
            )
        )
        harness, root = await open_harness(directory, setup, approval, storage)
        return Workspace(directory, setup, harness, root)

    def queue(self, id: str, tool: str, args: dict[str, Any]) -> None:
        self.setup.faux.set_responses(
            [faux_assistant_message([faux_tool_call(tool, args, id=id)], stop_reason="toolUse"), DONE]
        )

    async def call(self, tool: str, args: dict[str, Any], conversation: Any = None) -> ToolResultMessage:
        """One call of ``tool``, run to its result."""
        conversation = conversation if conversation is not None else self.root
        self.queue("c1", tool, args)
        submission = await conversation.submit({"type": "input", "content": "go"}, context)
        await submission.wait(context)
        return [entry["model"][0] for entry in await all_entries(conversation) if ToolResultEntry.is_(entry)][-1]

    async def asking(self, id: str, tool: str, args: dict[str, Any]) -> tuple[Any, Any]:
        """Submit one call of ``tool`` and wait until its request is pending."""
        self.queue(id, tool, args)
        submission = await self.root.submit({"type": "input", "content": "go"}, context)
        found: list[Any] = []

        async def check() -> bool:
            found[:] = await self.root.pending_approvals(context)
            return len(found) == 1

        await wait_for(check)
        return submission, found[0]

    async def close(self) -> None:
        await self.harness.close(context)


DONE = faux_assistant_message([faux_text("Done")])


async def open_harness(directory: Path, setup: ChatSetup, approval: Any, storage: Any = None) -> tuple[Harness, Any]:
    options: Any = {
        "models": setup.models,
        "registry": setup.registry,
        "settings": setup.settings,
        "env": lambda _target, _context: LocalExecutionEnv(cwd=str(directory)),
        "approval": approval,
        "onReport": lambda error, _info: setup.reports.append(error),
    }
    harness = await Harness.open(storage if storage is not None else MemoryStorage(), options, context)
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    return harness, root


def text(message: ToolResultMessage) -> str:
    return "".join(item.text for item in message.content if isinstance(item, TextContent))


def blocked(message: ToolResultMessage) -> str:
    assert message.is_error, text(message)
    return text(message).removeprefix("<harness>\n[error] ").removesuffix("\n</harness>")


async def test_deny_rules_beat_allow_rules(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy(allow=("Bash",), deny=("Bash(rm *)",)))
    assert text(await workspace.call("bash", {"command": "echo ok"})) == "ok\n"
    denied = await workspace.call("bash", {"command": "echo ok && rm -rf build"})
    assert blocked(denied) == "Tool call blocked: approval denied: denied by rule Bash(rm *)"
    await workspace.close()


async def test_ask_rules_ask_without_a_proposal(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy(allow=("Bash",), ask=("Bash(git push:*)",)))
    submission, request = await workspace.asking("c1", "bash", {"command": "git push origin"})
    assert request["prompt"] == "Bash(git push:*) requires approval"
    assert "proposedRule" not in request
    await workspace.root.decide(request["id"], {"type": "denied"}, context)
    await submission.wait(context)
    await workspace.close()


async def test_the_default_mode_reads_inside_and_asks_for_the_rest(tmp_path: Path) -> None:
    inside = tmp_path / "work"
    inside.mkdir()
    workspace = await Workspace.open(inside, policy())
    (inside / "inside.txt").write_text("in\n")
    assert text(await workspace.call("read", {"path": "inside.txt"})) == "in\n"

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "far.txt").write_text("far\n")
    submission, request = await workspace.asking("c1", "read", {"path": str(outside / "far.txt")})
    assert request["proposedRule"] == f"Read(//{str(outside).lstrip('/')}/**)"
    await workspace.root.decide(request["id"], {"type": "approved"}, context)
    assert (await submission.wait(context))["status"] == "done"

    submission, request = await workspace.asking("c2", "write", {"path": "new.txt", "content": "x"})
    assert request["proposedRule"] == f"Edit(//{str(inside).lstrip('/')}/**)"
    await workspace.root.decide(request["id"], {"type": "denied"}, context)
    await submission.wait(context)
    assert not (inside / "new.txt").exists()

    submission, request = await workspace.asking("c3", "bash", {"command": "git status --short"})
    assert request["proposedRule"] == "Bash(git status:*)"
    await workspace.root.decide(request["id"], {"type": "denied"}, context)
    await submission.wait(context)
    await workspace.close()


async def test_accept_edits_writes_inside(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy())
    await workspace.root.set_permission_mode("acceptEdits", context)
    written = await workspace.call("write", {"path": "a/b.txt", "content": "edited"})
    assert not written.is_error, text(written)
    assert (tmp_path / "a" / "b.txt").read_text() == "edited"
    edited = await workspace.call("edit", {"path": "a/b.txt", "edits": [{"oldText": "edited", "newText": "changed"}]})
    assert not edited.is_error, text(edited)
    assert (tmp_path / "a" / "b.txt").read_text() == "changed"
    await workspace.close()


async def test_plan_denies_commands_and_writes(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy("plan", allow=("Bash",)))
    (tmp_path / "notes.txt").write_text("plan\n")
    assert text(await workspace.call("read", {"path": "notes.txt"})) == "plan\n"
    bash = await workspace.call("bash", {"command": "echo hi"})
    assert blocked(bash) == "Tool call blocked: approval denied: plan mode: bash may not run"
    write = await workspace.call("write", {"path": "notes.txt", "content": "x"})
    assert blocked(write) == "Tool call blocked: approval denied: plan mode: write may not run"
    assert (tmp_path / "notes.txt").read_text() == "plan\n"
    await workspace.close()


async def test_dont_ask_denies_what_would_ask(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy("dontAsk", allow=("Bash(echo *)",)))
    assert text(await workspace.call("bash", {"command": "echo fine"})) == "fine\n"
    other = await workspace.call("bash", {"command": "pwd"})
    assert blocked(other) == "Tool call blocked: approval denied: bash would need approval (dontAsk mode)"
    await workspace.close()


async def test_bypass_keeps_deny_and_ask_rules(tmp_path: Path) -> None:
    workspace = await Workspace.open(
        tmp_path, policy("bypassPermissions", ask=("Bash(git push:*)",), deny=("Read(./secret.txt)",))
    )
    assert text(await workspace.call("bash", {"command": "echo free"})) == "free\n"
    (tmp_path / "secret.txt").write_text("hidden")
    secret = await workspace.call("read", {"path": "secret.txt"})
    assert blocked(secret) == "Tool call blocked: approval denied: denied by rule Read(./secret.txt)"
    # A Read deny covers writes too.
    write = await workspace.call("write", {"path": "secret.txt", "content": "x"})
    assert blocked(write) == "Tool call blocked: approval denied: denied by rule Read(./secret.txt)"
    submission, request = await workspace.asking("c1", "bash", {"command": "git push"})
    await workspace.root.decide(request["id"], {"type": "abort"}, context)
    assert (await submission.wait(context))["status"] == "unanswered"
    await workspace.close()


async def test_conversation_rules_and_directories(tmp_path: Path) -> None:
    work = tmp_path / "work"
    data = tmp_path / "data"
    work.mkdir()
    data.mkdir()
    (data / "d.txt").write_text("data\n")
    workspace = await Workspace.open(work, policy())
    state = await workspace.root.update_permissions(
        {"add": {"allow": ["Bash(echo *)"], "deny": ["Bash(echo secret*)"]}, "directories": [str(data)]}, context
    )
    assert state == {"rules": {"allow": ["Bash(echo *)"], "deny": ["Bash(echo secret*)"]}, "directories": [str(data)]}
    assert text(await workspace.call("bash", {"command": "echo hi"})) == "hi\n"
    assert blocked(await workspace.call("bash", {"command": "echo secret"})).endswith(
        "denied by rule Bash(echo secret*)"
    )
    assert text(await workspace.call("read", {"path": str(data / "d.txt")})) == "data\n"
    assert await workspace.root.permissions(context) == state
    await workspace.close()


async def test_an_approved_rule_applies_to_every_conversation(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy())
    submission, request = await workspace.asking("c1", "bash", {"command": "echo one"})
    rule = request["proposedRule"]
    assert rule == "Bash(echo one:*)"
    await workspace.root.decide(request["id"], {"type": "approvedWithRule", "rule": rule}, context)
    assert (await submission.wait(context))["status"] == "done"
    assert await workspace.harness.permission_rules(context) == {"allow": [rule]}
    entries = [entry for entry in await all_entries(workspace.root) if entry["kind"] == "pi.approval"]
    assert entries[0]["data"]["decision"] == {"type": "approvedWithRule", "rule": rule}

    # Another conversation runs the same command without asking.
    other = await workspace.harness.create_conversation(
        {"ownership": {"kind": "ownerless"}, "agent": {"model": {"provider": "faux", "modelId": "faux-1"}}}, context
    )
    assert text(await workspace.call("bash", {"command": "echo one two"}, other)) == "one two\n"
    assert await other.pending_approvals(context) == []

    assert await workspace.harness.remove_permission_rule(rule, context) is True
    assert await workspace.harness.remove_permission_rule(rule, context) is False
    assert await workspace.harness.permission_rules(context) == {}
    await workspace.close()


async def _forever(_task: Any, runtime: Any, _ctx: Context) -> None:
    await aborted(runtime.signal)


async def _forever_abort(_task: Any, runtime: Any, ctx: Context) -> None:
    await runtime.commit(lambda _tx, _current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)


Forever = define_task(
    name="test.forever",
    version=1,
    initial=lambda _input: {"phase": "hold"},
    phases={"hold": _forever},
    abort=_forever_abort,
)


async def test_forks_and_owned_conversations_take_the_permissions(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy())
    workspace.setup.registry.install(define_extension("forever", tasks=[Forever]))
    harness, root = workspace.harness, workspace.root
    await root.set_permission_mode("plan", context)
    stream = await watch_events(harness, root.id, context)
    batches: list[Any] = []
    stream.start(lambda events, _context: batches.append(list(events)))
    await root.update_permissions({"add": {"deny": ["Read(./.env)"]}}, context)
    await wait_for(lambda: any(event["type"] == "permissions_changed" for batch in batches for event in batch))
    changed = [event for batch in batches for event in batch if event["type"] == "permissions_changed"]
    assert changed[-1]["permissions"] == {"mode": "plan", "rules": {"deny": ["Read(./.env)"]}}
    await stream.stop()

    workspace.setup.faux.set_responses([faux_assistant_message([faux_text("Hello")])])
    await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    at = (await all_entries(root))[-1]["id"]
    fork = await root.fork(at, {"ownership": {"kind": "ownerless"}}, context)
    await root.set_permission_mode("acceptEdits", context)
    assert await fork.permissions(context) == {"mode": "plan", "rules": {"deny": ["Read(./.env)"]}}

    # A conversation owned by a task of this one starts with its permissions.
    task_id = await root.commit(
        lambda tx: tx.create_task(Forever, {}, {"ownership": {"kind": "conversation"}}), context
    )
    owned = await harness.create_conversation({"ownership": {"kind": "task", "taskId": task_id}}, context)
    assert (await owned.permissions(context)).get("mode") == "acceptEdits"
    await harness.abort_task(task_id, context)
    await workspace.close()


async def test_rules_are_validated(tmp_path: Path) -> None:
    workspace = await Workspace.open(tmp_path, policy())
    with pytest.raises(RuleError):
        await workspace.root.update_permissions({"add": {"allow": ["Bash(ls"]}}, context)
    with pytest.raises(ValueError, match="Unknown permission mode"):
        await workspace.root.set_permission_mode("yolo", context)  # type: ignore[arg-type]
    assert await workspace.root.permissions(context) == {}

    submission, request = await workspace.asking("c1", "bash", {"command": "ls"})
    with pytest.raises(RuleError):
        await workspace.root.decide(request["id"], {"type": "approvedWithRule", "rule": "fetch(x)"}, context)
    assert len(await workspace.root.pending_approvals(context)) == 1
    await workspace.root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    await workspace.close()

    with pytest.raises(RuleError) as raised:
        RulesPolicy(PermissionDefaults(rules={"deny": ["Read(/secrets)"]}))
    assert raised.value.code == "no_root"
    RulesPolicy(PermissionDefaults(rules={"deny": ["Read(/secrets)"]}, root=str(tmp_path)))
    with pytest.raises(RuleError):
        RulesPolicy(PermissionDefaults(rules={"allow": ["x(y)"]}))


class HangOnce:
    """The policy under test, which hangs after its first decision so the Harness can close between the decision and
    the request."""

    def __init__(self, inner: RulesPolicy) -> None:
        self.inner = inner
        self.decisions: list[Any] = []
        self.calls: list[ApprovalCall] = []

    async def decide(self, call: ApprovalCall, api: Any, call_context: Context) -> Any:
        decision = await self.inner.decide(call, api, call_context)
        self.decisions.append(decision)
        self.calls.append(call)
        if len(self.decisions) == 1:
            await await_with_context(asyncio.get_running_loop().create_future(), call_context)
        return decision


async def test_a_rerun_reuses_the_decision(tmp_path: Path) -> None:
    hang = HangOnce(policy())
    storage = MemoryStorage()
    workspace = await Workspace.open(tmp_path, hang, storage)
    workspace.queue("c1", "bash", {"command": "echo once"})
    await workspace.root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: len(hang.decisions) == 1)
    assert hang.calls[0].subject == {"type": "command", "command": "echo once"}
    assert hang.calls[0].cwd == str(tmp_path)
    await workspace.harness.close(context)

    # Bypass would allow the command, but the rerun reuses the decision it memoized.
    storage._closed = False  # pyright: ignore[reportAttributeAccessIssue]
    harness, root = await open_harness(tmp_path, workspace.setup, hang, storage)
    await root.set_permission_mode("bypassPermissions", context)
    harness.resume()
    found: list[Any] = []

    async def check() -> bool:
        found[:] = await root.pending_approvals(context)
        return len(found) == 1

    await wait_for(check)
    assert len(hang.decisions) == 2
    assert hang.decisions[0] == hang.decisions[1]
    assert hang.decisions[1]["type"] == "ask"
    await harness.close(context)
