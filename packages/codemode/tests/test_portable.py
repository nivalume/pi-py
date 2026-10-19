"""Port of upstream ``test/portable.test.ts`` and of the raw-worker cases of ``test/sandbox.test.ts``.

Upstream's portable tests cover the in-thread QuickJS sandbox (interrupt budget) and the Cloudflare
remote sandbox. Both are QuickJS/Cloudflare specifics and are dropped; what they test about the host
side runs here against :class:`CodemodeSandboxBase` with launchers written in the test: tools run in
the caller wherever the script runs, launcher failures are sandbox errors, an abort stops the
sandbox, and payloads that do not decode fail the run as a broken bridge. The import tests check that
the portable parts need no Monty.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import pytest

from pi_ai.abort import AbortController
from pi_codemode import CodemodeFailure, CodemodeSandboxBase, CodemodeSuccess, CodemodeTextOutput, CodemodeTool
from pi_codemode.portable import (
    CallMessage,
    CrashMessage,
    DoneMessage,
    HostToVmMessage,
    OutputMessage,
    VmChannel,
    VmEvents,
    VmStartData,
)

_NO_MONTY = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "pydantic_monty" or name.startswith("pydantic_monty."):
            raise ImportError("No module named 'pydantic_monty'")
        return None

sys.meta_path.insert(0, Block())
"""


def run_without_monty(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", _NO_MONTY + code], capture_output=True, text=True, timeout=60)


def test_portable_parts_import_without_monty() -> None:
    completed = run_without_monty(
        """
import pi_codemode, pi_codemode.portable, pi_codemode.source, pi_codemode.declarations
from pi_codemode import CodemodeSandboxBase, CodemodeTool, parse_codemode_source, render_declarations, to_codemode_identifier
assert parse_codemode_source("# @options: {}\\nreturn 1").code == "\\nreturn 1"
assert render_declarations(tools=[CodemodeTool("a-b", lambda a, c: a)]) == "class tools:\\n    async def a_b(self, args: Any = None, /) -> Any: ..."
assert to_codemode_identifier("my-tool") == "my_tool"
assert "pydantic_monty" not in sys.modules
print("ok")
"""
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_the_sandbox_names_the_extra_when_monty_is_missing() -> None:
    completed = run_without_monty(
        """
import pi_codemode
for attempt in (lambda: pi_codemode.CodemodeSandbox, lambda: __import__("pi_codemode.sandbox")):
    try:
        attempt()
    except ImportError as error:
        print(error)
    else:
        raise SystemExit("imported")
"""
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 2
    assert all("pi-ai-python[codemode]" in line for line in lines)


class FakeVm:
    """A launcher that runs a tiny "script" itself: it calls each tool named in ``code`` (one per
    line, ``name arg``), prints what each returns, and returns the last result."""

    def __init__(self, *, fail_launch: str | None = None) -> None:
        self.fail_launch = fail_launch
        self.started: list[VmStartData] = []
        self.stopped = 0

    def __call__(self, data: VmStartData, events: VmEvents) -> VmChannel:
        if self.fail_launch is not None:
            raise RuntimeError(self.fail_launch)
        self.started.append(data)
        return _FakeChannel(self, data, events)


class _FakeChannel:
    def __init__(self, vm: FakeVm, data: VmStartData, events: VmEvents) -> None:
        self.vm = vm
        self.events = events
        self.results: asyncio.Queue[HostToVmMessage] = asyncio.Queue()
        self.task = asyncio.ensure_future(self.run(data.code))

    async def run(self, code: str) -> None:
        value: Any = None
        for index, line in enumerate(code.splitlines(), start=1):
            name, _, arg = line.partition(" ")
            self.events.message(CallMessage(index, "tool", name, arg or None))
            result = await self.results.get()
            if not result.ok:
                self.events.message(DoneMessage(ok=False, error={"name": "Exception", "message": result.payload or ""}))
                return
            value = result.payload
            self.events.message(OutputMessage(CodemodeTextOutput(f"{name} -> {value}")))
        self.events.message(DoneMessage(ok=True, value=value, writes=(("k", "1"),)))

    def post(self, message: HostToVmMessage) -> None:
        self.results.put_nowait(message)

    async def stop(self) -> None:
        self.vm.stopped += 1
        self.task.cancel()


async def test_runs_tools_in_the_caller_and_the_script_wherever_the_launcher_puts_it() -> None:
    seen: list[Any] = []

    def add(args: Any, _context: Any) -> Any:
        seen.append(args)
        return args["a"] + 1

    vm = FakeVm()
    sandbox = CodemodeSandboxBase(vm, tools=[CodemodeTool("add", add)])
    result = await sandbox.execute('add {"a": 1}', store={"k": 1})
    assert isinstance(result, CodemodeSuccess)
    assert (result.value, result.output) == (2, [CodemodeTextOutput("add -> 2")])
    assert result.store_writes.set == {"k": 1}
    assert seen == [{"a": 1}]
    assert [(call.name, call.status) for call in result.calls] == [("add", "ok")]
    assert vm.started[0].tools == (("add", "add", ""),)
    assert vm.started[0].store == {"k": "1"}
    assert vm.stopped == 1
    await sandbox.close()


async def test_tells_the_sandbox_to_stop_when_the_caller_aborts() -> None:
    started = asyncio.Event()

    async def hang(_args: Any, _context: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    vm = FakeVm()
    sandbox = CodemodeSandboxBase(vm, tools=[CodemodeTool("hang", hang)])
    controller = AbortController()
    pending = asyncio.ensure_future(sandbox.execute("hang", signal=controller.signal))
    await started.wait()
    controller.abort()
    result = await pending
    assert isinstance(result, CodemodeFailure) and result.error.kind == "aborted"
    assert vm.stopped == 1
    await sandbox.close()


async def test_reports_a_failing_launcher_or_sandbox_as_a_sandbox_error() -> None:
    failing = CodemodeSandboxBase(FakeVm(fail_launch="no binding"))
    result = await failing.execute("x")
    assert isinstance(result, CodemodeFailure)
    assert (result.error.kind, result.error.message) == ("sandbox", "no binding")

    async def crashing(data: VmStartData, events: VmEvents) -> VmChannel:
        channel = FakeVm()(data, events)
        events.message(CrashMessage("object reset"))
        return channel

    crashed = await CodemodeSandboxBase(crashing).execute("x")
    assert isinstance(crashed, CodemodeFailure)
    assert (crashed.error.kind, crashed.error.message) == ("sandbox", "object reset")


async def test_queues_replies_until_the_launcher_returns_its_channel() -> None:
    gate = asyncio.Event()

    async def slow_launch(data: VmStartData, events: VmEvents) -> VmChannel:
        # A call before the channel exists: its reply waits for the channel.
        channel = _FakeChannel(FakeVm(), data, events)
        await asyncio.sleep(0.02)
        gate.set()
        return channel

    sandbox = CodemodeSandboxBase(slow_launch, tools=[CodemodeTool("echo", lambda args, _c: args)])
    result = await sandbox.execute('echo "x"')
    assert gate.is_set()
    assert isinstance(result, CodemodeSuccess) and result.value == "x"


def raw_launcher(*messages: object):
    """Posts ``messages`` as if a sandbox had sent them (upstream's raw-worker fixture)."""

    def launch(data: VmStartData, events: VmEvents) -> VmChannel:
        for message in messages:
            events.message(message)
        return _IdleChannel()

    return launch


class _IdleChannel:
    def post(self, message: HostToVmMessage) -> None:
        pass

    async def stop(self) -> None:
        pass


async def test_accepts_a_well_formed_done_message() -> None:
    result = await CodemodeSandboxBase(raw_launcher(DoneMessage(ok=True, value="42"))).execute("")
    assert isinstance(result, CodemodeSuccess) and result.value == 42


# #10444: a malformed payload must fail the run as a sandbox error instead of raising in the host's
# message handler and never settling.
@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (DoneMessage(ok=True, value="1", writes="null"), "store writes are not a list"),  # type: ignore[arg-type]
        (DoneMessage(ok=True, value="1", writes=(1,)), "store writes contain a malformed entry"),  # type: ignore[arg-type]
        (DoneMessage(ok=True, value="1", writes=(("k", "{"),)), 'store value for "k" is not valid JSON'),
        (DoneMessage(ok=True, value="{"), "return value is not valid JSON"),
        (DoneMessage(ok=False, error="5"), "script error is not an object"),  # type: ignore[arg-type]
        (DoneMessage(ok=False, error={}), "script error is malformed"),
        (CallMessage(1, "tool", "x", "{"), "call arguments is not valid JSON"),
        (OutputMessage("text"), "output item is malformed"),  # type: ignore[arg-type]
        ({"type": "nonsense"}, "unknown message from the sandbox"),
    ],
)
async def test_reports_a_broken_bridge_as_a_sandbox_error(message: object, reason: str) -> None:
    sandbox = CodemodeSandboxBase(raw_launcher(message))
    if isinstance(message, CallMessage):
        # Upstream decodes forwarded call arguments in host-controlled runs.
        run = sandbox.start("", started_at=0, seed=0, on_call=lambda _call: None)
        result = await run.result
    else:
        result = await sandbox.execute("")
    assert isinstance(result, CodemodeFailure)
    assert result.error.kind == "sandbox"
    assert result.error.message.startswith(f"Sandbox bridge broken: {reason}")


async def test_reports_a_failing_host_callback_as_a_sandbox_error() -> None:
    def on_call(_call: Any) -> None:
        raise RuntimeError("journal unavailable")

    sandbox = CodemodeSandboxBase(raw_launcher(CallMessage(1, "tool", "x", None)))
    result = await sandbox.start("", started_at=0, seed=0, on_call=on_call).result
    assert isinstance(result, CodemodeFailure)
    assert (result.error.kind, result.error.message) == ("sandbox", "Sandbox host failed: journal unavailable")
