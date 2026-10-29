"""Drives one QuickJS VM through one script (port of ``runtime/vm.ts``).

Contract with the caller, as upstream's:

- ``post`` receives every message for the host, synchronously from inside VM callbacks. It must not re-enter the VM
  (no ``deliver``, no ``dispose``); the thread launcher hands the messages to the event loop.
- :meth:`VmSession.deliver` hands the VM a tool result and runs the script until it waits again. It must only be
  called while the VM is idle, which holds for any caller that never re-enters from ``post``.
- ``should_interrupt`` is polled by QuickJS while the script runs, and here between batches of promise jobs.
  Returning ``True`` throws an uncatchable error in the script; the VM then reports ``crash`` or a failed ``done``.
  The job batches matter for a script such as ``while (true) await null``: every iteration is a separate job that
  executes only a few opcodes, so QuickJS's own polling, which counts opcodes, would let it run for a long time
  between polls.
- The VM never decides on its own when it is finished: after ``done`` or ``crash`` it stays alive until the caller
  disposes it.

Failures after the VM exists are reported as a ``crash`` message; :func:`start_vm` only raises when the VM cannot be
created at all.

Deterministic mode (``data.deterministic``, used by ``CodemodeSandboxBase.start()``): a script must make the same
calls with the same arguments every time it runs with the same results, so a host can replay it after a restart. The
VM is otherwise deterministic, because it only runs when the host delivers something and has no timers. What remains
is fixed here, as upstream fixes it:

- Time: every WASI clock reads virtual time, ``started_at`` and then the ``now`` of the result being processed, never
  going backwards (see :class:`~pi_codemode.runtime.quickjs.Clock`).
- Randomness: ``random_get`` draws from a PRNG over the seed.
- Timezone: UTC, not the host's.
- Garbage collection: ``WeakRef`` and ``FinalizationRegistry`` are not available; collection timing is not.
- Delivery order: results carry a ``seq`` and are applied strictly in that order, buffering early ones.

The VM reports ``started`` with its fingerprint before the script runs, and ``idle`` with the number of calls issued
after each drain, so the host can check a replay step by step. The fingerprint has upstream's format, so a run
started by upstream's sandbox on the same ``quickjs-wasi`` build reports the same one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import wasmtime

from ..types import CodemodeDuring, CodemodeImageOutput, CodemodeTextOutput
from .protocol import (
    CallMessage,
    CrashMessage,
    DoneMessage,
    HostToVmMessage,
    IdleMessage,
    OutputMessage,
    StartedMessage,
    VmStartData,
    VmToHostMessage,
)
from .quickjs import (
    INTRINSICS_ALL,
    INTRINSICS_WEAK_REF,
    MAX_STACK_SIZE,
    QUICKJS_WASI_VERSION,
    Clock,
    JSException,
    QuickJS,
)

__all__ = ["DETERMINISTIC_VERSION", "PRELUDE_SOURCE", "SCRIPT_FILENAME", "VmSession", "start_vm"]

#: Upstream's prelude, unchanged.
PRELUDE_SOURCE: Final = Path(__file__).with_name("prelude.js").read_text(encoding="utf-8")
#: Bumped whenever this module or the prelude changes what a deterministic script observes, beyond what the
#: fingerprint's other parts (engine versions, prelude hash, limits) already capture. Upstream's value.
DETERMINISTIC_VERSION: Final = 1
#: The file name stack traces give the script.
SCRIPT_FILENAME: Final = "codemode.js"
#: Promise jobs run between two ``should_interrupt`` polls while draining the job queue.
_JOBS_PER_POLL: Final = 50


def _fnv1a(text: str) -> str:
    """FNV-1a over UTF-16 code units, as 8 hex digits: identifies the prelude in the fingerprint."""
    value = 0x811C9DC5
    data = text.encode("utf-16-le", "surrogatepass")
    for index in range(0, len(data), 2):
        value ^= data[index] | (data[index + 1] << 8)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return f"{value:08x}"


_PRELUDE_HASH: Final = _fnv1a(PRELUDE_SOURCE)


def _describe_exception(error: JSException) -> dict[str, str | None]:
    """A script error as upstream reports it: the stack prefixed with ``Name: message`` like V8's."""
    head = f"{error.name}: {error.message}" if error.message else error.name
    stack = None if error.stack is None else error.stack.rstrip()
    return {"name": error.name, "message": error.message, "stack": f"{head}\n{stack}" if stack else head}


def _crash_message(error: BaseException) -> str:
    if isinstance(error, wasmtime.Trap):
        # The trap's text holds a wasm backtrace; its cause ("wasm trap: call stack exhausted") is the useful part.
        lines = [line.strip() for line in str(error).splitlines() if line.strip().startswith("wasm trap:")]
        return f"RuntimeError: {lines[-1] if lines else str(error).splitlines()[0]}"
    if isinstance(error, JSException):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _decode_writes(writes_json: str) -> Any:
    """The prelude's store writes (``[[key, json] | [key], ...]``) as the protocol's ``(key, json | None)`` pairs.

    Anything else is passed on as is: the host validates it and reports a broken bridge.
    """
    try:
        entries = json.loads(writes_json)
    except ValueError:
        return writes_json
    if not isinstance(entries, list):
        return entries
    decoded: list[Any] = []
    for entry in entries:
        if isinstance(entry, list) and len(entry) == 1:
            decoded.append((entry[0], None))
        elif isinstance(entry, list) and len(entry) == 2:
            decoded.append((entry[0], entry[1]))
        else:
            decoded.append(entry)
    return tuple(decoded)


def _decode_error(error_json: str) -> Any:
    try:
        return json.loads(error_json)
    except ValueError:
        return error_json


class VmSession:
    """The host's handle on a started VM: deliver results, then dispose it."""

    def __init__(self, vm: QuickJS, post: Callable[[VmToHostMessage], None], clock: Clock | None) -> None:
        self._vm = vm
        self._post = post
        self._clock = clock
        self._disposed = False
        #: Calls the script has issued, tools and globals; reported with every ``idle``.
        self.issued = 0
        #: Deterministic mode: the next result to apply, and results that arrived before it.
        self._next_seq = 1
        #: The result being processed, 0 during the script's first run; reported with ``done``.
        self.current = 0
        self._early: dict[int, HostToVmMessage] = {}
        self.api = 0
        self.settle = 0
        self.stalled = 0
        self._should_interrupt: Callable[[], bool] = lambda: False

    def deliver(self, message: HostToVmMessage) -> None:
        if self._disposed:
            return
        try:
            clock = self._clock
            if clock is None:
                self._apply(message)
                return
            seq = message.seq
            if type(seq) is not int or seq < self._next_seq or seq in self._early:
                raise RuntimeError(
                    f"Result {message.id} has sequence number {seq}, expected {self._next_seq} or later, once"
                )
            self._early[seq] = message
            following = self._early.pop(self._next_seq, None)
            while following is not None:
                # Never backwards: a resumed host's clock may be behind the one that started the run.
                if following.now is not None and math.isfinite(following.now):
                    clock.now = max(clock.now, following.now)
                self.current = self._next_seq
                self._apply(following)
                self._post(IdleMessage(self._next_seq, self.issued))
                self._next_seq += 1
                following = self._early.pop(self._next_seq, None)
        except Exception as error:
            self._post(CrashMessage(_crash_message(error)))

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._vm.dispose()

    def _apply(self, message: HostToVmMessage) -> None:
        vm = self._vm
        args = [
            vm.new_number(message.id),
            vm.true if message.ok else vm.false,
            vm.undefined if message.payload is None else vm.new_string(message.payload),
            vm.undefined if message.properties is None else vm.new_string(message.properties),
        ]
        try:
            vm.free(vm.call(self.settle, self.api, *args))
        finally:
            for value in args:
                vm.free(value)
        self.drain()

    def drain(self) -> None:
        """Run queued jobs, then fail a script that waits on nothing that can ever resume it."""
        vm = self._vm
        # Jobs run one at a time, so the loop can poll for an interrupt between them.
        jobs = 0
        while vm.is_job_pending():
            # A job only fails for uncatchable errors, such as an interrupt.
            if vm.execute_pending_job() < 0:
                raise RuntimeError("Promise job failed")
            jobs += 1
            if jobs % _JOBS_PER_POLL == 0 and self._should_interrupt():
                raise RuntimeError("Execution interrupted")
        vm.free(vm.call(self.stalled, self.api))


def start_vm(
    data: VmStartData,
    post: Callable[[VmToHostMessage], None],
    should_interrupt: Callable[[], bool],
    module: wasmtime.Module,
    *,
    max_stack_size: int = MAX_STACK_SIZE,
    interrupt_budget: int | None = None,
) -> VmSession:
    """Create the VM, run the prelude and the script until it first waits, and return the session."""
    deterministic = data.deterministic
    clock = None if deterministic is None else Clock(deterministic[0], deterministic[1])
    vm = QuickJS(
        module,
        memory_limit=data.memory_limit_bytes,
        # Without a guard, deep recursion overflows the wasm stack and traps instead of throwing a catchable RangeError.
        max_stack_size=max_stack_size,
        interrupt=should_interrupt,
        clock=clock,
        timezone_offset_minutes=0 if clock is not None else None,
        intrinsics=INTRINSICS_ALL & ~INTRINSICS_WEAK_REF if clock is not None else None,
    )
    session = VmSession(vm, post, clock)
    session._should_interrupt = should_interrupt  # pyright: ignore[reportPrivateUsage]

    def during() -> CodemodeDuring | None:
        return None if clock is None else CodemodeDuring(session.current, session.issued)

    def bridge(_this: int, *args: int) -> int:
        # Called from the prelude with primitives only.
        kind = vm.to_string(args[0]) if args else ""
        a, b, c = (args[1:] + (vm.undefined,) * 3)[:3]
        if kind in ("call", "global"):
            session.issued += 1
            post(
                CallMessage(
                    int(vm.to_number(a)),
                    "tool" if kind == "call" else "global",
                    vm.to_string(b),
                    None if vm.is_undefined(c) else vm.to_string(c),
                )
            )
        elif kind == "output":
            what = vm.to_string(a)
            if what == "image":
                post(OutputMessage(CodemodeImageOutput(vm.to_string(b), vm.to_string(c))))
            else:
                post(OutputMessage(CodemodeTextOutput(vm.to_string(b), console=what == "console")))
        elif kind == "done":
            # The host stops the VM on `done`, so the `idle` of this drain never arrives: report its numbers here.
            if vm.to_bool(a):
                value = None if vm.is_undefined(b) else vm.to_string(b)
                post(DoneMessage(ok=True, value=value, writes=_decode_writes(vm.to_string(c)), during=during()))
            else:
                post(DoneMessage(ok=False, error=_decode_error(vm.to_string(b)), during=during()))
        return vm.undefined

    try:
        bridge_function = vm.new_function("bridge", bridge)
        prelude = vm.eval(PRELUDE_SOURCE, "codemode-prelude.js")
        strings = [
            vm.new_string(
                json.dumps(
                    [
                        {"name": name, "jsName": js_name, "description": description}
                        for name, js_name, description in data.tools
                    ],
                    ensure_ascii=False,
                )
            ),
            vm.new_string(
                json.dumps([{"name": name, "spread": spread} for name, spread in data.globals], ensure_ascii=False)
            ),
            vm.new_string(json.dumps(data.store, ensure_ascii=False)),
        ]
        # These values live as long as the VM, which the caller disposes as a whole.
        session.api = vm.call(prelude, vm.undefined, bridge_function, *strings)
        for value in (prelude, bridge_function, *strings):
            vm.free(value)
        session.settle = vm.get_prop(session.api, "settle")
        run = vm.get_prop(session.api, "run")
        session.stalled = vm.get_prop(session.api, "stalled")
        if clock is not None:
            budget = "none" if interrupt_budget is None else str(interrupt_budget)
            heap = "none" if data.memory_limit_bytes is None else str(data.memory_limit_bytes)
            post(
                StartedMessage(
                    f"quickjs-wasi {QUICKJS_WASI_VERSION}, quickjs {vm.quickjs_version}, prelude {_PRELUDE_HASH}, "
                    f"deterministic {DETERMINISTIC_VERSION}, stack {max_stack_size}, heap {heap}, budget {budget}"
                )
            )
        # The prefix shares the first line with the script so reported line numbers match the script as written.
        try:
            function = vm.eval(f"(async (tools, console) => {{{data.code}\n}})", SCRIPT_FILENAME)
        except JSException as error:
            post(DoneMessage(ok=False, error=_describe_exception(error)))
            vm.free(run)
            return session
        try:
            vm.free(vm.call(run, session.api, function))
        finally:
            vm.free(function)
            vm.free(run)
        session.drain()
        if clock is not None:
            post(IdleMessage(0, session.issued))
    except Exception as error:
        post(CrashMessage(_crash_message(error)))
    return session
