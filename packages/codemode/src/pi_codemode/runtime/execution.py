"""Host side of an execution, shared by every place a sandbox can run (port of ``execution.ts``).

The host owns everything the script must not influence: the tool table, call records,
output, the deadline, the caller's abort signal, and the decoding of what the sandbox
reports. The sandbox itself runs wherever a :data:`VmLauncher` decides (a QuickJS VM on its own
thread for :class:`~pi_codemode.CodemodeSandbox`, a Monty worker for
:class:`~pi_codemode.MontyCodemodeSandbox`). The two sides only exchange the messages in
:mod:`pi_codemode.runtime.protocol`, so launchers differ in transport and termination, not in
behavior. The host only needs to know the scripts' language for names: which identifiers tools
get and which globals are reserved. This module imports no sandbox, so hosts that bring their
own launcher (or only declare tools) need neither wasmtime nor Monty.

Lifecycle of one execution:

1. The constructor arms the deadline and the abort listener, then asks the launcher for a
   channel.
2. Messages from the sandbox arrive through :meth:`VmEvents.message`; tool calls run here and
   their results go back through :meth:`VmChannel.post`. Replies produced before the launcher
   returned its channel are queued and flushed when it arrives.
3. The first terminal event wins: ``done``, ``crash``, a launcher failure, the deadline, or an
   abort. ``finish()`` aborts pending tool calls, asks the channel to stop, and resolves the
   result once that request completes; the sandbox may exit later. Later events are ignored.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias

from pi_ai.abort import AbortController, AbortSignal
from pi_chord.context import Context
from pi_chord.json import JsonValue
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, TelemetryContext

from ..identifier import is_codemode_identifier, to_codemode_identifier
from ..language import CodemodeLanguage, check_codemode_language
from ..types import (
    CodemodeCall,
    CodemodeCallStatus,
    CodemodeDeclaredGlobal,
    CodemodeDeclaredTool,
    CodemodeDuring,
    CodemodeError,
    CodemodeFailure,
    CodemodeHostCall,
    CodemodeImageOutput,
    CodemodeOutcome,
    CodemodeOutputItem,
    CodemodeResult,
    CodemodeStoreWrites,
    CodemodeSuccess,
    CodemodeTextOutput,
    CodemodeTool,
    CodemodeToolContext,
)
from .protocol import (
    CallMessage,
    CrashMessage,
    DoneMessage,
    HostToVmMessage,
    IdleMessage,
    OutputMessage,
    ResultMessage,
    StartedMessage,
    VmStartData,
)

__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "RESERVED_GLOBALS",
    "CodemodeRun",
    "CodemodeSandboxBase",
    "OnCall",
    "OnIdle",
    "OnStarted",
    "VmChannel",
    "VmEvents",
    "VmLauncher",
    "check_globals",
    "error_message",
]

DEFAULT_TIMEOUT_MS: Final = 300_000
#: Names a global may not take: the script's built-in globals, by language.
RESERVED_GLOBALS: Final[Mapping[CodemodeLanguage, frozenset[str]]] = {
    "javascript": frozenset({"tools", "ALL_TOOLS", "console", "text", "image", "exit", "globalThis", "store", "load"}),
    "python": frozenset({"tools", "ALL_TOOLS", "print", "text", "image", "exit", "store", "load"}),
}


class VmEvents(Protocol):
    """Callbacks a launcher uses to report what its sandbox does. Call them on the event loop thread."""

    def message(self, message: object) -> None:
        """A message from the sandbox. Validated here, so launchers can pass through anything they receive."""

    def failure(self, error: CodemodeError) -> None:
        """The sandbox failed or stopped outside the bridge, for example a crashed worker."""


class VmChannel(Protocol):
    """The host's handle on a running sandbox."""

    def post(self, message: HostToVmMessage) -> None: ...

    async def stop(self) -> None:
        """Requests sandbox shutdown. Called once; may return before the sandbox actually exits."""


#: Starts a sandbox for one execution. Exceptions become ``sandbox`` errors with their message.
#: The sandbox may start reporting messages before the returned awaitable settles.
VmLauncher: TypeAlias = Callable[[VmStartData, VmEvents], "VmChannel | Awaitable[VmChannel]"]

OnStarted: TypeAlias = Callable[[str], Any]
OnCall: TypeAlias = Callable[[CodemodeHostCall], Any]
OnIdle: TypeAlias = Callable[[int, int], Any]


def error_message(error: object) -> str:
    return str(error)


class BridgeError(Exception):
    """A payload from the sandbox side that does not decode. The execution fails as a sandbox error."""


def _parse_bridge_json(text: str, what: str) -> JsonValue:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        raise BridgeError(f"{what} is not valid JSON") from None


def _dumps(value: Any) -> str:
    """Strict JSON text: what ``JSON.stringify`` accepts, minus non-finite numbers."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _serialize_store(store: Mapping[str, Any] | None) -> dict[str, str]:
    return {key: _dumps(value) for key, value in (store or {}).items()}


def _parse_store_writes(writes: object) -> CodemodeStoreWrites:
    if not isinstance(writes, (tuple, list)):
        raise BridgeError("store writes are not a list")
    result = CodemodeStoreWrites()
    for entry in writes:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not (entry[1] is None or isinstance(entry[1], str))
        ):
            raise BridgeError("store writes contain a malformed entry")
        key, value = entry
        if value is None:
            result.delete.append(key)
        else:
            result.set[key] = _parse_bridge_json(value, f"store value for {json.dumps(key)}")
    return result


def _parse_script_error(error: object) -> CodemodeError:
    if not isinstance(error, Mapping):
        raise BridgeError("script error is not an object")
    name, message, stack = error.get("name"), error.get("message"), error.get("stack")
    if (
        not isinstance(message, str)
        or not (name is None or isinstance(name, str))
        or not (stack is None or isinstance(stack, str))
    ):
        raise BridgeError("script error is malformed")
    return CodemodeError("script", message, name, stack)


def _check_output(item: object) -> CodemodeOutputItem:
    if isinstance(item, CodemodeTextOutput) and isinstance(item.text, str):
        return item
    if isinstance(item, CodemodeImageOutput) and isinstance(item.data, str) and isinstance(item.mime_type, str):
        return item
    raise BridgeError("output item is malformed")


def check_globals(
    globals: Iterable[CodemodeDeclaredGlobal | CodemodeTool], language: CodemodeLanguage = "javascript"
) -> None:
    """Checks global names: identifiers or ``namespace.member``, not shadowing a built-in global or each other.

    In Python, names starting with ``__`` are the prelude's, so they are refused too.
    """
    reserved = RESERVED_GLOBALS[check_codemode_language(language)]
    names: set[str] = set()
    namespaces: set[str] = set()
    for global_ in globals:
        parts = global_.name.split(".")
        if (
            len(parts) > 2
            or not all(
                is_codemode_identifier(part, language) and not (language == "python" and part.startswith("__"))
                for part in parts
            )
            or parts[0] in reserved
        ):
            raise ValueError(f'Invalid global name "{global_.name}"')
        if global_.name in names:
            raise ValueError(f'Global "{global_.name}" is already registered')
        if len(parts) == 2:
            namespaces.add(parts[0])
        names.add(global_.name)
    for name in namespaces:
        if name in names:
            raise ValueError(f'Global "{name}" conflicts with the namespace "{name}"')


def _now_ms() -> float:
    return time.perf_counter() * 1000


@dataclass(slots=True)
class _PendingCall:
    record: CodemodeCall | None
    started_at: float
    controller: AbortController


@dataclass(slots=True, frozen=True)
class _Control:
    """Callbacks of a host-controlled run: its calls go to the caller, which answers them with ``deliver()``."""

    on_call: OnCall
    on_started: OnStarted | None
    on_idle: OnIdle | None


@dataclass(slots=True, frozen=True)
class _Executors:
    tools: Mapping[str, CodemodeTool]
    globals: Mapping[str, CodemodeTool]


class _Events:
    __slots__ = ("_execution",)

    def __init__(self, execution: _Execution) -> None:
        self._execution = execution

    def message(self, message: object) -> None:
        self._execution.receive(message)

    def failure(self, error: CodemodeError) -> None:
        self._execution.finish(error)


class _Execution:
    """One script run in its own sandbox."""

    def __init__(
        self,
        *,
        code: str,
        tools: Sequence[CodemodeDeclaredTool],
        globals: Sequence[CodemodeDeclaredGlobal],
        executors: _Executors,
        timeout_ms: float,
        signal: AbortSignal | None,
        memory_limit_bytes: int | None,
        store: dict[str, str],
        deterministic: tuple[float, int] | None,
        control: _Control | None,
        launch: VmLauncher,
        language: CodemodeLanguage,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self.future: asyncio.Future[CodemodeResult] = self._loop.create_future()
        self._channel: VmChannel | None = None
        #: Replies produced before the launcher returned the channel.
        self._unsent: list[HostToVmMessage] = []
        self._executors = executors
        self._control = control
        #: Result numbers ``deliver()`` used.
        self._delivered: set[int] = set()
        self._signal = signal
        self._remove_abort_listener: Callable[[], None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._output: list[CodemodeOutputItem] = []
        self._calls: list[CodemodeCall] = []
        self._pending: dict[int, _PendingCall] = {}
        self._tasks: set[asyncio.Future[Any]] = set()
        #: Whether the execution has its outcome.
        self.finished = False

        if math.isfinite(timeout_ms):
            self._timer = self._loop.call_later(
                max(timeout_ms, 0) / 1000,
                lambda: self.finish(CodemodeError("timeout", f"Execution timed out after {_format_ms(timeout_ms)} ms")),
            )
        if signal is not None:
            if signal.aborted:
                self._on_abort()
            else:
                self._remove_abort_listener = signal.add_listener(self._abort_listener)
        if self.finished:
            return

        data = VmStartData(
            code=code,
            tools=tuple(
                (tool.name, tool.js_name or to_codemode_identifier(tool.name, language), tool.description or "")
                for tool in tools
            ),
            globals=tuple((global_.name, global_.spread) for global_ in globals),
            memory_limit_bytes=memory_limit_bytes,
            store=store,
            deterministic=deterministic,
        )
        self._spawn(self._launch(launch, data))

    def _spawn(self, awaitable: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _launch(self, launch: VmLauncher, data: VmStartData) -> None:
        try:
            launched = launch(data, _Events(self))
            channel = await launched if inspect.isawaitable(launched) else launched
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.finish(CodemodeError("sandbox", error_message(error)))
            return
        if self.finished:
            try:
                await channel.stop()
            except Exception:
                pass
            return
        self._channel = channel
        unsent, self._unsent = self._unsent, []
        for message in unsent:
            channel.post(message)

    def deliver(self, seq: int, now: float, id: int, outcome: CodemodeOutcome) -> None:
        """:meth:`CodemodeRun.deliver`."""
        if self.finished:
            return
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError(f"The clock value of result {seq} is not a finite number")
        if type(seq) is not int or seq < 1 or seq in self._delivered:
            raise ValueError(f"Result number {seq} is not a positive integer used once")
        pending = self._pending.get(id)
        if pending is None:
            raise ValueError(f"Call {id} is not waiting for a result")
        if not isinstance(outcome, Mapping) or not isinstance(outcome.get("ok"), bool):
            raise ValueError(f"Result {seq} is not an outcome")
        ok = outcome["ok"]
        properties: str | None = None
        # Serialized before anything is recorded, so a value JSON cannot hold leaves the call waiting for another answer.
        if ok:
            payload = _dumps(outcome["value"]) if "value" in outcome else None
        else:
            payload = str(outcome.get("message", ""))
            if "properties" in outcome:
                raw = outcome.get("properties")
                if not isinstance(raw, Mapping):
                    raise ValueError(f"The properties of result {seq} are not an object")
                properties = _dumps(raw)
        self._delivered.add(seq)
        self._pending.pop(id)
        if pending.record is not None:
            pending.record.status = "ok" if ok else "error"
            pending.record.duration_ms = _now_ms() - pending.started_at
        self._post(ResultMessage(id, ok, payload, seq=seq, now=float(now), properties=properties))

    def abort(self, message: str) -> asyncio.Future[CodemodeResult]:
        self.finish(CodemodeError("aborted", message))
        return self.future

    def _abort_listener(self) -> None:
        if threading.get_ident() == self._loop_thread:
            self._on_abort()
        else:
            self._loop.call_soon_threadsafe(self._on_abort)

    def _on_abort(self) -> None:
        reason = self._signal.reason if self._signal is not None else None
        message = (str(reason) or "Execution aborted") if isinstance(reason, BaseException) else "Execution aborted"
        self.finish(CodemodeError("aborted", message))

    def _post(self, message: HostToVmMessage) -> None:
        if self._channel is not None:
            self._channel.post(message)
        else:
            self._unsent.append(message)

    def receive(self, message: object) -> None:
        """An exception here would escape into the launcher's transport and leave the execution unsettled."""
        try:
            self._handle_message(message)
        except BridgeError as error:
            self.finish(
                CodemodeError(
                    "sandbox",
                    f"Sandbox bridge broken: {error}. The script may have replaced the codemode helpers.",
                )
            )
        except Exception as error:
            self.finish(CodemodeError("sandbox", f"Sandbox host failed: {error_message(error)}"))

    def _handle_message(self, message: object) -> None:
        if self.finished:
            return
        if isinstance(message, OutputMessage):
            self._output.append(_check_output(message.item))
        elif isinstance(message, CallMessage):
            if message.id in self._pending:
                raise BridgeError(f"duplicate call id {message.id}")
            if self._control is not None:
                self._forward_call(message, self._control)
            else:
                self._start_call(message)
        elif isinstance(message, StartedMessage):
            if self._control is not None and self._control.on_started is not None:
                self._control.on_started(message.fingerprint)
        elif isinstance(message, IdleMessage):
            if self._control is not None and self._control.on_idle is not None:
                self._control.on_idle(message.seq, message.issued)
        elif isinstance(message, DoneMessage):
            self._handle_done(message)
        elif isinstance(message, CrashMessage):
            self.finish(CodemodeError("sandbox", message.message))
        else:
            raise BridgeError("unknown message from the sandbox")

    def _handle_done(self, message: DoneMessage) -> None:
        # Decode everything before finish(), which must not raise once it starts.
        if not message.ok:
            self.finish(_parse_script_error(message.error), during=message.during)
            return
        value = None if message.value is None else _parse_bridge_json(message.value, "return value")
        self.finish(None, value, _parse_store_writes(message.writes), message.during)

    def _forward_call(self, message: CallMessage, control: _Control) -> None:
        """A host-controlled run: record the call and hand it to the caller, which answers with ``deliver()``."""
        args = None if message.args is None else _parse_bridge_json(message.args, "call arguments")
        record = CodemodeCall(message.name, "cancelled", 0) if message.target == "tool" else None
        if record is not None:
            self._calls.append(record)
        self._pending[message.id] = _PendingCall(record, _now_ms(), AbortController())
        control.on_call(CodemodeHostCall(message.id, message.target, message.name, args))

    def _start_call(self, message: CallMessage) -> None:
        is_tool = message.target == "tool"
        record = CodemodeCall(message.name, "cancelled", 0) if is_tool else None
        if record is not None:
            self._calls.append(record)
        pending = _PendingCall(record, _now_ms(), AbortController())
        self._pending[message.id] = pending
        self._spawn(self._run_call(message, pending))

    async def _run_call(self, message: CallMessage, pending: _PendingCall) -> None:
        is_tool = message.target == "tool"
        status: CodemodeCallStatus
        try:
            tool = (self._executors.tools if is_tool else self._executors.globals).get(message.name)
            if tool is None:
                raise LookupError(f'Unknown {"tool" if is_tool else "global"} "{message.name}"')
            args = None if message.args is None else json.loads(message.args)
            value = tool.execute(args, CodemodeToolContext(pending.controller.signal))
            if inspect.isawaitable(value):
                value = await value
            reply = ResultMessage(message.id, True, None if value is None else _dumps(value))
            status = "ok"
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            reply = ResultMessage(message.id, False, error_message(error))
            status = "error"
        # Already cancelled by finish(): the record keeps "cancelled" and the sandbox is gone or going.
        if self._pending.pop(message.id, None) is None:
            return
        if pending.record is not None:
            pending.record.status = status
            pending.record.duration_ms = _now_ms() - pending.started_at
        self._post(reply)

    def finish(
        self,
        error: CodemodeError | None,
        value: JsonValue = None,
        writes: CodemodeStoreWrites | None = None,
        during: CodemodeDuring | None = None,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        if self._timer is not None:
            self._timer.cancel()
        if self._remove_abort_listener is not None:
            self._remove_abort_listener()

        now = _now_ms()
        for pending in self._pending.values():
            if pending.record is not None:
                pending.record.duration_ms = now - pending.started_at
            pending.controller.abort()
        self._pending.clear()
        self._unsent = []

        result: CodemodeResult
        if error is not None:
            result = CodemodeFailure(error, self._output, self._calls, during)
        else:
            result = CodemodeSuccess(value, self._output, self._calls, writes or CodemodeStoreWrites(), during)
        channel = self._channel
        if channel is None:
            self._resolve(result)
            return
        self._spawn(self._stop_and_resolve(channel, result))

    async def _stop_and_resolve(self, channel: VmChannel, result: CodemodeResult) -> None:
        try:
            await channel.stop()
        except Exception:
            pass
        self._resolve(result)

    def _resolve(self, result: CodemodeResult) -> None:
        if not self.future.done():
            self.future.set_result(result)


def _format_ms(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


class CodemodeRun:
    """A running :meth:`CodemodeSandboxBase.start` execution."""

    __slots__ = ("_execution", "result")

    def __init__(self, execution: _Execution, result: asyncio.Future[CodemodeResult]) -> None:
        self._execution = execution
        #: Resolves with the run's result; never raises for script failures.
        self.result = result

    @property
    def finished(self) -> bool:
        """Whether the run has its outcome: the script ended, timed out, or failed, or the run was aborted.

        Set when that is decided, before :attr:`result` settles; calls can no longer be answered.
        It does not mean the sandbox has exited.
        """
        return self._execution.finished

    def deliver(self, seq: int, now: float, id: int, outcome: CodemodeOutcome) -> None:
        """Answer call ``id`` as result number ``seq``.

        Results are numbered 1, 2, ... without gaps, and the sandbox applies them strictly in
        that order, whatever order they arrive in: after a gap it waits for the missing number.
        While it processes this one, its clock reads ``now`` (ms since the epoch), or the
        previous value if ``now`` is earlier. Raises ``ValueError`` for a call that is not
        waiting for an answer, a ``seq`` used before, or a ``now`` that is not finite, and
        ``TypeError``/``ValueError`` for a value JSON cannot hold (the call keeps waiting).
        """
        self._execution.deliver(seq, now, id, outcome)


def _signal_of(signal: AbortSignal | None, context: Context | None) -> AbortSignal | None:
    if signal is not None:
        return signal
    return None if context is None else context.abort_signal


class CodemodeSandboxBase:
    """Tool table, defaults and in-flight executions: the behavior every sandbox shares.

    Subclasses only choose where the sandbox runs by passing a :data:`VmLauncher`, and the
    language its scripts are written in.

    The script sees ``tools.<name>(...)`` for every registered tool, ``ALL_TOOLS``, the output
    helpers ``text``, ``image``, ``exit`` and ``console.*`` (JavaScript) or ``print`` (Python),
    ``store``/``load``, and the configured globals; nothing else (no filesystem, network,
    subprocesses or host modules). Each ``execute()`` gets its own sandbox session; the sandbox
    object only holds the tool table and defaults. ``close()`` aborts in-flight executions.
    """

    def __init__(
        self,
        launch: VmLauncher,
        *,
        language: CodemodeLanguage = "javascript",
        tools: Iterable[CodemodeTool] = (),
        globals: Iterable[CodemodeTool] = (),
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        memory_limit_bytes: int | None = None,
    ) -> None:
        """
        Args:
            launch: Starts the sandbox of one execution.
            language: The language of the scripts ``launch``'s sandbox runs. It decides the
                identifiers tools get and the reserved global names.
            tools: Functions scripts call as ``await tools.<name>(...)``.
            globals: Functions exposed as top-level names instead of on ``tools``, for host
                helpers such as attaching an image to the result. They behave like tools (JSON
                round trip, awaitable result) but are not recorded in ``result.calls``. Names
                must be identifiers, or ``namespace.member``, and may not shadow the built-in
                globals (:data:`RESERVED_GLOBALS`).
            timeout_ms: Overall deadline per execution, including time spent in tools.
                ``math.inf`` disables it; the execution then only ends when the script settles
                or is aborted. Default: 300000.
            memory_limit_bytes: Maximum heap the sandbox may use. Allocations beyond it fail the
                script (``InternalError: out of memory`` in JavaScript, ``MemoryError`` in
                Python). Default: no limit.
        """
        self._language: CodemodeLanguage = check_codemode_language(language)
        self._tools_by_name: dict[str, CodemodeTool] = {}
        self._globals_by_name: dict[str, CodemodeTool] = {}
        self._timeout_ms = timeout_ms
        self._memory_limit_bytes = memory_limit_bytes
        self._launch = launch
        self._running: set[_Execution] = set()
        self._closed = False
        for tool in tools:
            self.register_tool(tool)
        globals = list(globals)
        check_globals(globals, self._language)
        for global_ in globals:
            self._globals_by_name[global_.name] = global_

    @property
    def language(self) -> CodemodeLanguage:
        """The language of the scripts this sandbox runs."""
        return self._language

    @property
    def memory_limit_bytes(self) -> int | None:
        return self._memory_limit_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def register_tool(self, tool: CodemodeTool) -> None:
        """Raises ``ValueError`` if a tool with the same name is already registered."""
        if tool.name in self._tools_by_name:
            raise ValueError(f'Tool "{tool.name}" is already registered')
        self._tools_by_name[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        return self._tools_by_name.pop(name, None) is not None

    @property
    def tools(self) -> list[CodemodeTool]:
        return list(self._tools_by_name.values())

    @property
    def globals(self) -> list[CodemodeTool]:
        return list(self._globals_by_name.values())

    async def execute(
        self,
        code: str,
        *,
        signal: AbortSignal | None = None,
        context: Context | None = None,
        timeout_ms: float | None = None,
        store: Mapping[str, Any] | None = None,
        telemetry_context: TelemetryContext | None = None,
    ) -> CodemodeResult:
        """Run ``code``, the body of an async function: top-level ``await`` and ``return`` work.

        Never raises for script failures; those come back as a :class:`~pi_codemode.CodemodeFailure`.
        The script can use ``store(key, value)`` and ``load(key)`` on ``store``; its writes come
        back as ``result.store_writes``, and persisting them is up to the caller.

        Args:
            signal: Aborts the execution (``kind="aborted"``). ``context``'s abort signal is
                used when no signal is given.
            timeout_ms: Overrides the sandbox default for this execution.
            store: Values the script reads with ``load(key)``. Must be JSON-serializable.
            telemetry_context: Parent of the execution's ``pi.codemode.run`` span (see
                :mod:`pi_codemode.telemetry`); without one nothing is recorded.
        """
        if self._closed:
            raise RuntimeError("Sandbox is closed")
        if telemetry_context is None or telemetry_context is NOOP_TELEMETRY_CONTEXT:
            return await self._execute(code, signal, context, timeout_ms, store)
        from ..telemetry import record_codemode_result, trace_codemode_run

        effective_timeout = self._timeout_ms if timeout_ms is None else timeout_ms

        async def run(span: Any) -> CodemodeResult:
            result = await self._execute(code, signal, context, timeout_ms, store)
            record_codemode_result(span, result, effective_timeout)
            return result

        return await trace_codemode_run(
            telemetry_context,
            code,
            mode="execute",
            timeout_ms=effective_timeout,
            tool_count=len(self._tools_by_name),
            run=run,
        )

    async def _execute(
        self,
        code: str,
        signal: AbortSignal | None,
        context: Context | None,
        timeout_ms: float | None,
        store: Mapping[str, Any] | None,
    ) -> CodemodeResult:
        tools = dict(self._tools_by_name)
        execution = _Execution(
            code=code,
            tools=[CodemodeDeclaredTool(tool.name, None, tool.description) for tool in tools.values()],
            globals=[
                CodemodeDeclaredGlobal(global_.name, global_.spread) for global_ in self._globals_by_name.values()
            ],
            executors=_Executors(tools, dict(self._globals_by_name)),
            timeout_ms=self._timeout_ms if timeout_ms is None else timeout_ms,
            signal=_signal_of(signal, context),
            memory_limit_bytes=self._memory_limit_bytes,
            store=_serialize_store(store),
            deterministic=None,
            control=None,
            launch=self._launch,
            language=self._language,
        )
        self._running.add(execution)
        try:
            return await asyncio.shield(execution.future)
        finally:
            if execution.future.done():
                self._running.discard(execution)
            else:
                # The caller was cancelled: stop the execution, which then unregisters itself.
                execution.abort("Execution cancelled")
                execution.future.add_done_callback(lambda _future: self._running.discard(execution))

    def start(
        self,
        code: str,
        *,
        started_at: float,
        seed: int,
        on_call: OnCall,
        tools: Sequence[CodemodeDeclaredTool] = (),
        globals: Sequence[CodemodeDeclaredGlobal] = (),
        store: Mapping[str, Any] | None = None,
        signal: AbortSignal | None = None,
        context: Context | None = None,
        timeout_ms: float | None = None,
        on_started: OnStarted | None = None,
        on_idle: OnIdle | None = None,
    ) -> CodemodeRun:
        """Run ``code`` deterministically, with the caller answering its calls.

        The same script, start time, seed, tools (with their ``js_name``), globals and store,
        answered with the same results in the same ``seq`` order, makes the same calls with
        the same arguments and produces the same output, on any sandbox with the same
        fingerprint. A host can therefore replay it after a restart. The sandbox runs in UTC,
        its clock reads ``started_at`` and then the ``now`` of the result being processed
        (in Python, sleeps advance it without waiting), and its randomness comes from ``seed``;
        JavaScript has no ``WeakRef`` or ``FinalizationRegistry`` in such a run.

        The sandbox's own tools and globals are not used: the run declares its own, and every
        call goes to ``on_call``. Never raises for script failures; abort through ``signal``
        to end the run with an error the script cannot catch. Must be called with a running
        event loop.

        Args:
            started_at: The clock's start, in ms since the epoch.
            seed: Seeds ``Math.random()`` (JavaScript) or ``random`` and ``os.urandom()`` (Python).
                A 32-bit unsigned integer.
            on_call: The script made a call. Answer it with :meth:`CodemodeRun.deliver`, now or later.
            timeout_ms: Deadline of this run in ms. Default: none (not the sandbox's default).
            on_started: The sandbox's fingerprint, before the script runs and before any of its calls.
            on_idle: ``(seq, issued)``: the sandbox processed result ``seq`` (``0``: the script's
                first run, before any result) and waits again, having issued ``issued`` calls in total.
        """
        if self._closed:
            raise RuntimeError("Sandbox is closed")
        if (
            isinstance(started_at, bool)
            or not isinstance(started_at, (int, float))
            or not math.isfinite(started_at)
            or started_at < 0
        ):
            raise ValueError("started_at must be a non-negative number of milliseconds since the epoch")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be a 32-bit unsigned integer")
        tools = list(tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Tool names must be unique")
        script_names = [tool.js_name or to_codemode_identifier(tool.name, self._language) for tool in tools]
        if len(set(script_names)) != len(tools):
            raise ValueError("Tool script names must be unique")
        for name in script_names:
            if not is_codemode_identifier(name, self._language):
                raise ValueError(f'Invalid tool script name "{name}"')
        globals = list(globals)
        check_globals(globals, self._language)
        execution = _Execution(
            code=code,
            tools=tools,
            globals=globals,
            executors=_Executors({}, {}),
            timeout_ms=math.inf if timeout_ms is None else timeout_ms,
            signal=_signal_of(signal, context),
            memory_limit_bytes=self._memory_limit_bytes,
            store=_serialize_store(store),
            deterministic=(float(started_at), seed),
            control=_Control(on_call, on_started, on_idle),
            launch=self._launch,
            language=self._language,
        )
        self._running.add(execution)
        execution.future.add_done_callback(lambda _future: self._running.discard(execution))
        return CodemodeRun(execution, execution.future)

    async def close(self) -> None:
        """Aborts in-flight executions (they resolve with ``kind="aborted"``) and rejects new ones."""
        self._closed = True
        await asyncio.gather(*(execution.abort("Sandbox closed") for execution in list(self._running)))

    async def __aenter__(self) -> CodemodeSandboxBase:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
