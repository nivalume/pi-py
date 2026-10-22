"""Claude Code hooks over a Harness with the coding tools and the faux model: command handlers through the local
environment, HTTP handlers through ``httpx.MockTransport``."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from durable_harness_support import aborted, all_entries, chat_setup, wait_for
from durable_session_support import context

from pi_ai import faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import SystemMessage, TextContent, ToolResultMessage
from pi_chord.context import Context, await_with_context
from pi_durable import (
    AllowAll,
    AskAll,
    ContextEntry,
    Harness,
    MemoryStorage,
    ToolResultEntry,
    define_extension,
    define_task,
)
from pi_durable.claude_hooks import ClaudeHooks, ClaudeHooksOptions, ConfigError, HooksConfig, Matcher
from pi_durable.claude_hooks.protocol import arguments_from, claude_tool_input, claude_tool_name, output_block
from pi_durable.claude_hooks.runner import interpolate, parse_output
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool

MODEL: Any = {"model": {"provider": "faux", "modelId": "faux-1"}}


class Setup:
    def __init__(self, harness: Harness, root: Any, faux: Any, errors: list[tuple[str, str]], directory: Path) -> None:
        self.harness = harness
        self.root = root
        self.faux = faux
        self.errors = errors
        self.directory = directory


async def open_setup(
    directory: Path,
    config: Any,
    *,
    policy: Callable[[ClaudeHooks], Any] | None = None,
    settings: dict[str, Any] | None = None,
    extensions: tuple[Any, ...] = (),
    storage: Any = None,
    client: httpx.AsyncClient | None = None,
    setup: Any = None,
    errors: list[tuple[str, str]] | None = None,
) -> Setup:
    setup = setup if setup is not None else chat_setup()
    errors = errors if errors is not None else []
    if setup.registry.snapshot().extension("coding") is None:
        tools = [create_read_tool(), create_write_tool(), create_edit_tool(), create_bash_tool()]
        setup.registry.install(define_extension("coding", tools=tools))
        for extension in extensions:
            setup.registry.install(extension)
    hooks = ClaudeHooks(
        HooksConfig.from_value(config),
        ClaudeHooksOptions(client=client, on_error=lambda event, error: errors.append((event, error))),
    )
    setup.registry.install(hooks.extension())
    setup.settings.update(settings or {})
    options: Any = {
        "models": setup.models,
        "registry": setup.registry,
        "settings": setup.settings,
        "env": lambda _target, _context: LocalExecutionEnv(cwd=str(directory)),
        "approval": AllowAll() if policy is None else policy(hooks),
        "onReport": lambda error, _info: setup.reports.append(error),
    }
    harness = await Harness.open(storage if storage is not None else MemoryStorage(), options, context)
    root = await harness.root(context, agent=MODEL)
    return Setup(harness, root, setup.faux, errors, directory)


def bash(command: str) -> Any:
    return faux_assistant_message([faux_tool_call("bash", {"command": command})], stop_reason="toolUse")


def text(value: str) -> Any:
    return faux_assistant_message([faux_text(value)])


def command(event: str, matcher: str, script: str) -> dict[str, Any]:
    return {event: [{"matcher": matcher, "hooks": [{"type": "command", "command": script}]}]}


async def submit(conversation: Any, content: str) -> Any:
    return await (await conversation.submit({"type": "input", "content": content}, context)).wait(context)


async def tool_texts(conversation: Any) -> list[list[str]]:
    results: list[ToolResultMessage] = [
        entry["model"][0] for entry in await all_entries(conversation) if ToolResultEntry.is_(entry)
    ]
    return [[item.text for item in result.content if isinstance(item, TextContent)] for result in results]


# ─── units ──────────────────────────────────────────────────────────────────


def test_matchers_take_all_names_or_patterns() -> None:
    assert Matcher.parse(None).matches(["Bash"])
    assert Matcher.parse("*").matches(["anything"])
    names = Matcher.parse("Edit|Write")
    assert names.matches(["Write"])
    assert not names.matches(["WriteFile"])
    assert Matcher.parse("mcp__.*").matches(["mcp__github__search"])
    with pytest.raises(ConfigError):
        Matcher.parse("(")


def test_configurations_load_and_name_unsupported_events() -> None:
    config = HooksConfig.from_value(
        {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "check.sh", "timeout": 5}]},
                {
                    "hooks": [
                        {
                            "type": "http",
                            "url": "http://localhost/hook",
                            "headers": {"Authorization": "Bearer $TOKEN"},
                            "allowedEnvVars": ["TOKEN"],
                        }
                    ]
                },
            ],
            "SessionStart": [{"hooks": [{"type": "command", "command": "hello"}]}],
        }
    )
    assert len(config.handlers("PreToolUse", ["Bash"])) == 2
    assert len(config.handlers("PreToolUse", ["Read"])) == 1
    assert config.handlers("PreToolUse", ["Bash"])[0]["timeout"] == 5
    assert config.unsupported() == ["SessionStart"]
    assert HooksConfig.from_settings({}).events == {}
    with pytest.raises(ConfigError):
        HooksConfig.from_value({"Stop": [{"hooks": [{"type": "mcp_tool"}]}]})


def test_outputs_and_tool_inputs() -> None:
    assert output_block({"decision": "block", "reason": "Run the tests"}) == "Run the tests"
    assert output_block({"continue": False, "stopReason": "Done for today"}) == "Done for today"
    assert output_block({}) is None
    assert claude_tool_name("bash") == "Bash"
    assert claude_tool_name("mcp__x") == "mcp__x"
    arguments = {"path": "a.txt", "content": "x"}
    assert claude_tool_input(arguments)["file_path"] == "a.txt"
    assert arguments_from({"file_path": "b.txt", "content": "y"}, arguments) == {"path": "b.txt", "content": "y"}
    assert parse_output(' {"decision":"block","reason":"no"} \n') == {
        "kind": "output",
        "output": {"decision": "block", "reason": "no"},
    }
    assert parse_output("plain words\n") == {"kind": "output", "output": {}, "text": "plain words"}
    assert parse_output("") == {"kind": "output", "output": {}}
    home = os.environ.get("HOME", "")
    assert interpolate("Bearer ${HOME} $HOME", ["HOME"]) == f"Bearer {home} {home}"
    assert interpolate("Bearer $OTHER", ["HOME"]) == "Bearer $OTHER"


# ─── events ─────────────────────────────────────────────────────────────────


async def test_pre_tool_use_exit_2_blocks_with_stderr_and_gets_the_input(tmp_path: Path) -> None:
    script = 'cat > "$CLAUDE_PROJECT_DIR/input.json"; echo "no commands today" >&2; exit 2'
    setup = await open_setup(tmp_path, command("PreToolUse", "Bash", script))
    setup.faux.set_responses([bash("echo hi"), text("ok")])
    await submit(setup.root, "go")
    assert await tool_texts(setup.root) == [["<harness>\n[error] Tool call blocked: no commands today\n</harness>"]]
    received = json.loads((tmp_path / "input.json").read_text())
    assert received["hook_event_name"] == "PreToolUse"
    assert received["tool_name"] == "Bash"
    assert received["tool_input"]["command"] == "echo hi"
    assert received["session_id"] == str(setup.root.id)
    assert received["permission_mode"] == "default"
    assert received["cwd"] == str(tmp_path)
    await setup.harness.close(context)


async def test_pre_tool_use_replaces_the_input_and_adds_context(tmp_path: Path) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": {"command": "echo changed"},
            "additionalContext": "the hook changed the command",
        }
    }
    config = command("PreToolUse", "Bash", f"cat > /dev/null; printf '%s' '{json.dumps(output)}'")
    # A matcher that does not match runs nothing.
    config["PreToolUse"].append({"matcher": "Read", "hooks": [{"type": "command", "command": "exit 2"}]})
    setup = await open_setup(tmp_path, config)
    setup.faux.set_responses([bash("echo original"), text("ok")])
    await submit(setup.root, "go")
    assert await tool_texts(setup.root) == [["changed\n", "the hook changed the command"]]
    await setup.harness.close(context)


async def test_http_handlers_deny_and_fail_without_blocking(tmp_path: Path) -> None:
    received: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        received.append(request)
        if request.url.path == "/deny":
            deny = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "remote says no",
                }
            }
            return httpx.Response(200, json=deny)
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    config = {
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "http",
                        "url": "http://hooks.test/deny",
                        "headers": {"X-Home": "${HOME}"},
                        "allowedEnvVars": ["HOME"],
                    }
                ],
            }
        ],
        "PostToolUse": [{"hooks": [{"type": "http", "url": "http://hooks.test/broken"}]}],
    }
    setup = await open_setup(tmp_path, config, client=client)
    setup.faux.set_responses([bash("echo hi"), text("ok")])
    await submit(setup.root, "go")
    assert await tool_texts(setup.root) == [["<harness>\n[error] Tool call blocked: remote says no\n</harness>"]]
    body = json.loads(received[0].content)
    assert body["tool_input"]["command"] == "echo hi"
    assert received[0].headers["x-home"] == os.environ.get("HOME", "")

    # A failing handler does not block: the next call runs and the failure is reported.
    (tmp_path / "here.txt").write_text("here\n")
    setup.faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("read", {"path": "here.txt"})], stop_reason="toolUse"),
            text("ok"),
        ]
    )
    await submit(setup.root, "again")
    assert (await tool_texts(setup.root))[1] == ["here\n"]
    assert any(event == "PostToolUse" and "500" in error for event, error in setup.errors), setup.errors
    await setup.harness.close(context)
    await client.aclose()


async def test_post_tool_use_replaces_the_output(tmp_path: Path) -> None:
    output = {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": "[redacted]"}}
    script = f"cat > \"$CLAUDE_PROJECT_DIR/post.json\"; printf '%s' '{json.dumps(output)}'"
    setup = await open_setup(tmp_path, command("PostToolUse", "Bash", script))
    setup.faux.set_responses([bash("echo secret"), text("ok")])
    await submit(setup.root, "go")
    assert await tool_texts(setup.root) == [["[redacted]"]]
    received = json.loads((tmp_path / "post.json").read_text())
    assert received["tool_response"]["output"] == "secret\n"
    assert received["tool_response"]["structured"]["exitCode"] == 0
    await setup.harness.close(context)


async def test_post_tool_use_failure_appends_the_reason(tmp_path: Path) -> None:
    script = 'cat > "$CLAUDE_PROJECT_DIR/failure.json"; echo "look at the logs" >&2; exit 2'
    setup = await open_setup(tmp_path, command("PostToolUseFailure", "Bash", script))
    setup.faux.set_responses([bash("exit 3"), text("ok")])
    await submit(setup.root, "go")
    [texts] = await tool_texts(setup.root)
    assert texts[-2] == "look at the logs"
    received = json.loads((tmp_path / "failure.json").read_text())
    assert received["hook_event_name"] == "PostToolUseFailure"
    assert "3" in received["error"]
    await setup.harness.close(context)


async def test_stop_continues_the_run_once(tmp_path: Path) -> None:
    script = (
        "input=$(cat); if printf '%s' \"$input\" | grep -q '\"stop_hook_active\":true'; then exit 0; fi; "
        'printf \'{"decision":"block","reason":"run the tests too"}\''
    )
    setup = await open_setup(tmp_path, command("Stop", "", script))
    setup.faux.set_responses([text("first"), text("second")])
    assert (await submit(setup.root, "go"))["status"] == "done"
    users = [
        entry["model"][0].content
        for entry in await all_entries(setup.root)
        if entry.get("model") and entry["model"][0].role == "user"
    ]
    assert users == ["go", "run the tests too"]
    assert setup.faux.state.call_count == 2
    await setup.harness.close(context)


async def test_user_prompt_submit_blocks_or_adds_context(tmp_path: Path) -> None:
    script = (
        'input=$(cat); if printf \'%s\' "$input" | grep -q secret; then echo "no secrets" >&2; exit 2; fi; '
        'echo "extra context"'
    )
    setup = await open_setup(tmp_path, command("UserPromptSubmit", "", script))
    blocked = await submit(setup.root, "tell me the secret")
    assert (blocked["status"], blocked["reason"], blocked["detail"]) == ("unanswered", "blocked", "no secrets")

    setup.faux.set_responses([text("hello")])
    await submit(setup.root, "hi")
    contexts = [
        entry["model"][0].content for entry in await all_entries(setup.root) if entry["kind"] == ContextEntry.kind
    ]
    assert contexts == ["extra context"]
    assert all(
        isinstance(entry["model"][0], SystemMessage)
        for entry in await all_entries(setup.root)
        if entry["kind"] == ContextEntry.kind
    )
    await setup.harness.close(context)


async def test_permission_request_answers_what_the_policy_asks(tmp_path: Path) -> None:
    allow = {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}}
    deny = {
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": {"behavior": "deny", "message": "not that one"},
        }
    }
    script = (
        "input=$(cat); if printf '%s' \"$input\" | grep -q forbidden; then "
        f"printf '%s' '{json.dumps(deny)}'; else printf '%s' '{json.dumps(allow)}'; fi"
    )
    setup = await open_setup(
        tmp_path,
        command("PermissionRequest", "Bash", script),
        policy=lambda hooks: hooks.permission_policy(AskAll()),
    )
    setup.faux.set_responses([bash("echo allowed"), text("ok")])
    await submit(setup.root, "go")
    setup.faux.set_responses([bash("echo forbidden"), text("ok")])
    await submit(setup.root, "again")
    assert await tool_texts(setup.root) == [
        ["allowed\n"],
        ["<harness>\n[error] Tool call blocked: approval denied: not that one\n</harness>"],
    ]
    assert await setup.root.pending_approvals(context) == []
    await setup.harness.close(context)


async def test_pre_compact_declines_a_manual_compaction(tmp_path: Path) -> None:
    compaction = {"enabled": False, "reserveTokens": 1000, "keepRecentTokens": 10, "backgroundTokens": 0}
    setup = await open_setup(
        tmp_path,
        command("PreCompact", "manual", 'cat > "$CLAUDE_PROJECT_DIR/compact.json"; exit 2'),
        settings={"compaction": compaction},
    )
    setup.faux.set_responses([text("long answer " * 500), text("second")])
    await submit(setup.root, "first")
    await submit(setup.root, "second")
    task = await setup.root.compact("keep the decisions", context)
    await setup.harness.wait_for_task(task, context)
    received = json.loads((tmp_path / "compact.json").read_text())
    assert (received["trigger"], received["custom_instructions"]) == ("manual", "keep the decisions")
    assert all(entry["kind"] != "pi.compaction" for entry in await all_entries(setup.root))
    await setup.harness.close(context)


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


async def test_a_task_owned_conversation_runs_subagent_stop(tmp_path: Path) -> None:
    config = {
        **command("SubagentStop", "", 'cat > /dev/null; touch "$CLAUDE_PROJECT_DIR/subagent"'),
        **command("Stop", "", 'cat > /dev/null; touch "$CLAUDE_PROJECT_DIR/stop"'),
    }
    setup = await open_setup(tmp_path, config, extensions=(define_extension("forever", tasks=[Forever]),))
    task = await setup.root.commit(
        lambda tx: tx.create_task(Forever, {}, {"ownership": {"kind": "conversation"}}), context
    )
    sub = await setup.harness.create_conversation(
        {"ownership": {"kind": "task", "taskId": task}, "agent": MODEL}, context
    )
    setup.faux.set_responses([text("sub answer")])
    await submit(sub, "work")
    assert (tmp_path / "subagent").exists()
    assert not (tmp_path / "stop").exists()
    await setup.harness.abort_task(task, context)
    await setup.harness.close(context)


class HangOnce:
    """Hangs on its first decision, so the Harness can stop between the hooks and the intent."""

    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, _call: Any, _api: Any, call_context: Context) -> Any:
        self.calls += 1
        if self.calls == 1:
            import asyncio

            await await_with_context(asyncio.get_running_loop().create_future(), call_context)
        return {"type": "allow"}


async def test_a_rerun_does_not_run_handlers_again(tmp_path: Path) -> None:
    policy = HangOnce()
    config = command("PreToolUse", "Bash", 'cat > /dev/null; echo run >> "$CLAUDE_PROJECT_DIR/count"')
    storage = MemoryStorage()
    setup = await open_setup(tmp_path, config, policy=lambda _hooks: policy, storage=storage)
    setup.faux.set_responses([bash("echo once"), text("ok")])
    await setup.root.submit({"type": "input", "content": "go"}, context)
    await wait_for(lambda: policy.calls == 1)
    await setup.harness.close(context)

    storage._closed = False  # pyright: ignore[reportAttributeAccessIssue]
    chat = chat_setup()
    chat.faux = setup.faux
    chat.models = setup.harness._options["models"]  # pyright: ignore[reportAttributeAccessIssue]
    reopened = await open_setup(tmp_path, config, policy=lambda _hooks: policy, storage=storage, setup=chat)
    reopened.harness.resume()
    await reopened.root.wait_for_idle(context)
    assert policy.calls == 2
    assert await tool_texts(reopened.root) == [["once\n"]]
    assert (tmp_path / "count").read_text() == "run\n"
    await reopened.harness.close(context)
