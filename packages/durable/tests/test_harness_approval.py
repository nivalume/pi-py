"""Approvals end to end: asking, deciding (approve, deny, session grants, abort), timeouts, refusals, recovery,
``before_tool`` permissions, events and the approval policy's seams."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from durable_harness_support import ChatSetup, all_entries, chat_setup, wait_for
from durable_session_support import context

from pi_ai import Type, faux_assistant_message, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage, TextContent, ToolResultMessage
from pi_durable import (
    AllowAll,
    ApprovalNotPending,
    ApprovalsDoc,
    AskAll,
    DenyAll,
    Harness,
    InvalidDecision,
    LiveDoc,
    MemoryStorage,
    RuleError,
    ToolResultEntry,
    ToolTask,
    UnknownApproval,
    define_extension,
    define_tool,
    hook,
    watch_events,
)
from pi_durable.approval import APPROVAL_ENTRY, ApprovalCall
from pi_durable.types import Storage

EchoParameters = Type.Object({"text": Type.String()})


class Echo:
    """An ``echo`` tool that counts its runs."""

    def __init__(self) -> None:
        self.runs = 0
        self.tool = define_tool(name="echo", description="Echo text", parameters=EchoParameters, execute=self._run)

    async def _run(self, args: Any, _api: Any, _context: Any) -> Any:
        self.runs += 1
        return {"output": [TextContent(args["text"])]}


def calls(*items: tuple[str, str]) -> AssistantMessage:
    """One ``echo`` call per ``(id, text)``."""
    return faux_assistant_message(
        [faux_tool_call("echo", {"text": text}, id=id) for id, text in items], stop_reason="toolUse"
    )


def done(text: str = "Done") -> AssistantMessage:
    return faux_assistant_message([faux_text(text)])


async def open_harness(
    setup: ChatSetup,
    policy: Any,
    storage: Storage | None = None,
    *,
    set_timer: Callable[..., Any] | None = None,
    now: Callable[[], float] | None = None,
) -> tuple[Harness, Any]:
    options: Any = {
        "models": setup.models,
        "registry": setup.registry,
        "settings": setup.settings,
        "onReport": lambda error, _info: setup.reports.append(error),
    }
    if policy is not None:
        options["approval"] = policy
    if set_timer is not None:
        options["setTimer"] = set_timer
    if now is not None:
        options["now"] = now
    harness = await Harness.open(storage if storage is not None else MemoryStorage(), options, context)
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    return harness, root


def setup_with(echo: Echo) -> ChatSetup:
    setup = chat_setup()
    setup.registry.install(define_extension("echo", tools=[echo.tool]))
    return setup


async def requests(conversation: Any, count: int) -> list[Any]:
    """The conversation's pending requests once there are ``count`` of them."""
    found: list[Any] = []

    async def check() -> bool:
        found[:] = await conversation.pending_approvals(context)
        return len(found) >= count

    await wait_for(check)
    return found


def results(entries: list[Any]) -> list[ToolResultMessage]:
    return [entry["model"][0] for entry in entries if ToolResultEntry.is_(entry)]


def text(message: ToolResultMessage) -> str:
    return "".join(item.text for item in message.content if isinstance(item, TextContent))


async def decisions(conversation: Any) -> list[Any]:
    return [entry["data"]["decision"] for entry in await all_entries(conversation) if entry["kind"] == APPROVAL_ENTRY]


async def submit(conversation: Any, content: str = "go") -> Any:
    return await conversation.submit({"type": "input", "content": content}, context)


async def test_an_approved_call_runs_with_the_requested_arguments() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done("Echoed")])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert (request["callId"], request["name"], request["arguments"]) == ("c1", "echo", {"text": "hi"})
    assert request["deadline"] - request["requestedAt"] == 15 * 60 * 1000
    assert request["approvalKey"] == '{"text":"hi"}'
    assert echo.runs == 0
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    assert live["tools"][0]["status"] == "pending"
    task = await harness.get_task(request["id"], context)
    assert task is not None
    assert task["kind"] == ToolTask.definition.name
    assert task["state"]["checkpoint"]["phase"] == "approval"
    assert await harness.pending_approvals(context) == [(root.id, request)]

    await harness.decide(request["id"], {"type": "approved"}, context)
    assert (await submission.wait(context))["status"] == "done"
    entries = await all_entries(root)
    assert text(results(entries)[0]) == "hi"
    assert echo.runs == 1
    assert await decisions(root) == [{"type": "approved"}]
    assert await harness.snapshot(ApprovalsDoc, root.id, context) == {"requests": [], "grants": []}
    # The decision entry is display-only: not model context.
    view = await root.context(context)
    roles = [message.role for message in view["messages"]]
    assert roles == ["system", "user", "assistant", "toolResult", "assistant"]
    await harness.close(context)


async def test_a_denied_call_is_blocked() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "rm -rf"), ("c2", "ls")), done("Blocked")])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    pending = await requests(root, 2)
    by_call = {request["callId"]: request for request in pending}
    await root.decide(by_call["c1"]["id"], {"type": "denied", "rejection": "too dangerous"}, context)
    await root.decide(by_call["c2"]["id"], {"type": "denied"}, context)
    assert (await submission.wait(context))["status"] == "done"
    found = {message.tool_call_id: message for message in results(await all_entries(root))}
    assert text(found["c1"]) == "<harness>\n[error] Tool call blocked: approval denied: too dangerous\n</harness>"
    assert text(found["c2"]) == "<harness>\n[error] Tool call blocked: approval denied\n</harness>"
    assert all(message.is_error for message in found.values())
    assert echo.runs == 0
    assert await root.pending_approvals(context) == []
    assert await harness.snapshot(ApprovalsDoc, root.id, context) == {"requests": [], "grants": []}
    await harness.close(context)


async def test_approval_for_the_session_skips_later_prompts() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "status")), calls(("c2", "status")), calls(("c3", "deploy")), done()])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    [first] = await requests(root, 1)
    await root.decide(first["id"], {"type": "approvedForSession"}, context)
    # ``c2`` has the same key and runs without a request; ``c3`` asks.
    [third] = await requests(root, 1)
    assert third["callId"] == "c3"
    await root.decide(third["id"], {"type": "approved"}, context)
    assert (await submission.wait(context))["status"] == "done"
    assert [text(message) for message in results(await all_entries(root))] == ["status", "status", "deploy"]
    state = await harness.snapshot(ApprovalsDoc, root.id, context)
    assert state is not None
    assert state["grants"] == [{"tool": "echo", "key": '{"text":"status"}', "grantedBy": first["id"]}]
    assert await decisions(root) == [{"type": "approvedForSession"}, {"type": "approved"}]
    await harness.close(context)


async def test_an_abort_decision_aborts_the_run() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi"))])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    [request] = await requests(root, 1)
    await root.decide(request["id"], {"type": "abort"}, context)
    settled = await submission.wait(context)
    assert (settled["status"], settled.get("reason")) == ("unanswered", "aborted")
    assert text(results(await all_entries(root))[0]) == "<harness>\n[error] Tool echo was aborted\n</harness>"
    assert echo.runs == 0
    state = await harness.snapshot(ApprovalsDoc, root.id, context)
    assert state is not None
    assert state["requests"] == []
    live = await harness.snapshot(LiveDoc, root.id, context)
    assert live is not None
    assert "run" not in live
    await harness.close(context)


async def test_decide_refuses_what_it_cannot_decide() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    [request] = await requests(root, 1)
    with pytest.raises(InvalidDecision):
        await root.decide(request["id"], {"type": "timedOut"}, context)
    with pytest.raises(InvalidDecision):
        await root.decide(request["id"], {"type": "maybe"}, context)  # type: ignore[typeddict-item]
    with pytest.raises(RuleError):
        await root.decide(request["id"], {"type": "approvedWithRule", "rule": "fetch(x)"}, context)
    with pytest.raises(UnknownApproval):
        await harness.decide(999_999, {"type": "approved"}, context)  # type: ignore[arg-type]
    other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
    with pytest.raises(UnknownApproval):
        await other.decide(request["id"], {"type": "approved"}, context)
    assert len(await root.pending_approvals(context)) == 1
    await root.decide(request["id"], {"type": "approved"}, context)
    with pytest.raises(ApprovalNotPending):
        await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    assert len(await decisions(root)) == 1
    await harness.close(context)


class ApprovalTimers:
    """A ``setTimer`` that holds timers of exactly ``hold_ms`` (an approval deadline under a fixed clock) until
    ``fire()``; other timers run on the event loop."""

    def __init__(self, hold_ms: float) -> None:
        self.hold_ms = hold_ms
        self.held: list[Callable[[], None]] = []

    def set_timer(self, callback: Callable[[], None], delay_ms: float, _options: Any) -> Callable[[], None]:
        if delay_ms == self.hold_ms:
            self.held.append(callback)
            return lambda: self.held.remove(callback) if callback in self.held else None
        handle = asyncio.get_running_loop().call_later(delay_ms / 1000, callback)
        return handle.cancel

    def fire(self) -> None:
        held, self.held = self.held, []
        for callback in held:
            callback()


async def test_an_unanswered_request_times_out() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.settings["approvalTimeoutMs"] = 30_000
    setup.faux.set_responses([calls(("c1", "hi")), done("Timed out")])
    timers = ApprovalTimers(30_000)
    harness, root = await open_harness(setup, AskAll(), set_timer=timers.set_timer, now=lambda: 1_000_000)
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert request["deadline"] - request["requestedAt"] == 30_000
    await wait_for(lambda: len(timers.held) == 1)
    timers.fire()
    assert (await submission.wait(context))["status"] == "done"
    assert (
        text(results(await all_entries(root))[0])
        == "<harness>\n[error] Tool call blocked: approval timed out after 30s\n</harness>"
    )
    assert await decisions(root) == [{"type": "timedOut"}]
    with pytest.raises(ApprovalNotPending):
        await root.decide(request["id"], {"type": "approved"}, context)
    assert echo.runs == 0
    await harness.close(context)


async def test_a_policy_timeout_and_prompt_reach_the_request() -> None:
    class Asks:
        def decide(self, call: ApprovalCall, _api: Any, _context: Any) -> Any:
            return {"type": "ask", "timeoutMs": 5_000, "prompt": f"Echo {call.call.arguments['text']}?"}

    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, Asks())
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert request["prompt"] == "Echo hi?"
    assert request["deadline"] - request["requestedAt"] == 5_000
    await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    await harness.close(context)


async def test_policies_deny_and_fail_closed() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, DenyAll("not here"))
    await (await submit(root)).wait(context)
    assert (
        text(results(await all_entries(root))[0])
        == "<harness>\n[error] Tool call blocked: approval denied: not here\n</harness>"
    )
    await harness.close(context)

    class Broken:
        def decide(self, *_args: Any) -> Any:
            raise RuntimeError("policy exploded")

    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, Broken())
    await (await submit(root)).wait(context)
    assert (
        text(results(await all_entries(root))[0])
        == "<harness>\n[error] Tool call blocked: approval denied: policy exploded\n</harness>"
    )
    assert any(str(error) == "policy exploded" for error in setup.reports)
    assert echo.runs == 0
    await harness.close(context)


async def test_before_tool_permissions_reach_the_policy() -> None:
    echo = Echo()
    setup = setup_with(echo)
    seen: list[Any] = []

    class Records:
        async def decide(self, call: ApprovalCall, _api: Any, _context: Any) -> Any:
            seen.append(call.hook)
            return {"type": "allow"}

    def before_tool(call: Any, _api: Any, _context: Any) -> Any:
        if call.id == "c2":
            return {"permission": "ask", "reason": "Really echo?", "arguments": {"text": "changed"}}
        return {"permission": "allow"}

    setup.registry.install(define_extension("hooks", hooks=[hook(ToolTask, {"before_tool": before_tool})]))
    setup.faux.set_responses([calls(("c1", "one")), calls(("c2", "two")), done()])
    harness, root = await open_harness(setup, Records())
    submission = await submit(root)
    [request] = await requests(root, 1)
    # An ``ask`` asks even though the policy allowed the call; the request has the final arguments.
    assert (request["callId"], request["prompt"], request["arguments"]) == ("c2", "Really echo?", {"text": "changed"})
    await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    assert seen == ["allow", "ask"]
    assert [text(message) for message in results(await all_entries(root))] == ["one", "changed"]
    await harness.close(context)


async def test_a_hook_ask_asks_without_a_policy() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.registry.install(
        define_extension("hooks", hooks=[hook(ToolTask, {"before_tool": lambda *_args: {"permission": "ask"}})])
    )
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, None)
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert "prompt" not in request
    await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    assert echo.runs == 1
    await harness.close(context)


async def test_allow_all_runs_calls_without_requests() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, AllowAll())
    await (await submit(root)).wait(context)
    assert echo.runs == 1
    assert await harness.snapshot(ApprovalsDoc, root.id, context) is None
    await harness.close(context)


async def test_events_show_requests_and_permissions() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    harness, root = await open_harness(setup, AskAll())
    stream = await watch_events(harness, root.id, context)
    assert (stream.snapshot["approvals"], stream.snapshot["permissions"]) == ([], {})
    batches: list[Any] = []
    stream.start(lambda events, _context: batches.append(list(events)))
    submission = await submit(root)
    [request] = await requests(root, 1)
    await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    await root.set_permission_mode("plan", context)

    def updates() -> list[Any]:
        return [event for batch in batches for event in batch if event["type"] == "approval_update"]

    await wait_for(lambda: any(event["type"] == "permissions_changed" for batch in batches for event in batch))
    assert [[item["callId"] for item in event["requests"]] for event in updates()] == [["c1"], []]
    changed = [event for batch in batches for event in batch if event["type"] == "permissions_changed"]
    assert changed == [{"type": "permissions_changed", "permissions": {"mode": "plan"}}]
    await stream.stop()
    await harness.close(context)


async def test_a_decision_committed_while_down_is_read_by_the_rerun() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    storage = MemoryStorage()
    harness, root = await open_harness(setup, AskAll(), storage)
    submission = await submit(root)
    [request] = await requests(root, 1)
    await harness.close(context)

    storage._closed = False  # pyright: ignore[reportAttributeAccessIssue]
    harness, root = await open_harness(setup, AskAll(), storage)
    # Requests outlive the process; deciding resumes scheduling, and the rerun consumes the decision.
    assert [pending["id"] for pending in await root.pending_approvals(context)] == [request["id"]]
    await root.decide(request["id"], {"type": "approved"}, context)
    reacquired = await harness.submission(submission.id, context)
    assert reacquired is not None
    assert (await reacquired.wait(context))["status"] == "done"
    assert echo.runs == 1
    assert text(results(await all_entries(root))[0]) == "hi"
    await harness.close(context)


async def test_a_rerun_after_the_deadline_times_out_at_the_stored_deadline() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.settings["approvalTimeoutMs"] = 2_000
    clock = {"now": 1_000_000.0}
    setup.now = lambda: clock["now"]
    setup.faux.set_responses([calls(("c1", "hi")), done()])
    storage = MemoryStorage()
    timers = ApprovalTimers(2_000)

    async def open_at() -> tuple[Harness, Any]:
        options: Any = {
            "models": setup.models,
            "registry": setup.registry,
            "settings": setup.settings,
            "now": lambda: clock["now"],
            "setTimer": timers.set_timer,
            "approval": AskAll(),
        }
        harness = await Harness.open(storage, options, context)
        return harness, await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})

    harness, root = await open_at()
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert request["deadline"] == request["requestedAt"] + 2_000
    await harness.close(context)

    clock["now"] += 5_000
    timers.held.clear()
    storage._closed = False  # pyright: ignore[reportAttributeAccessIssue]
    harness, root = await open_at()
    harness.resume()
    reacquired = await harness.submission(submission.id, context)
    assert reacquired is not None
    assert (await reacquired.wait(context))["status"] == "done"
    assert (
        text(results(await all_entries(root))[0])
        == "<harness>\n[error] Tool call blocked: approval timed out after 2s\n</harness>"
    )
    assert echo.runs == 0
    await harness.close(context)


async def test_aborting_the_conversation_withdraws_the_request() -> None:
    echo = Echo()
    setup = setup_with(echo)
    setup.faux.set_responses([calls(("c1", "hi"))])
    harness, root = await open_harness(setup, AskAll())
    submission = await submit(root)
    await requests(root, 1)
    await root.abort(context)
    assert (await submission.wait(context))["status"] == "unanswered"
    state = await harness.snapshot(ApprovalsDoc, root.id, context)
    assert state is not None
    assert state["requests"] == []
    # Withdrawn without a decision entry.
    assert await decisions(root) == []
    await harness.close(context)


async def test_nested_calls_ask_too() -> None:
    echo = Echo()
    setup = setup_with(echo)

    async def outer(_args: Any, api: Any, call_context: Any) -> Any:
        nested = await api.execute_tool("echo", {"text": "inner"}, call_context)
        return {"output": [TextContent(f"outer saw {nested['isError']}")]}

    setup.registry.install(
        define_extension(
            "outer",
            tools=[define_tool(name="outer", description="Calls echo", parameters=Type.Object({}), execute=outer)],
        )
    )

    class AsksForEcho:
        def decide(self, call: ApprovalCall, _api: Any, _context: Any) -> Any:
            return {"type": "ask"} if call.call.name == "echo" else {"type": "allow"}

    setup.faux.set_responses(
        [faux_assistant_message([faux_tool_call("outer", {}, id="o1")], stop_reason="toolUse"), done()]
    )
    harness, root = await open_harness(setup, AsksForEcho())
    submission = await submit(root)
    [request] = await requests(root, 1)
    assert (request["name"], request["callId"]) == ("echo", "o1/1")
    await root.decide(request["id"], {"type": "approved"}, context)
    await submission.wait(context)
    assert echo.runs == 1
    assert text(results(await all_entries(root))[0]) == "outer saw False"
    await harness.close(context)
