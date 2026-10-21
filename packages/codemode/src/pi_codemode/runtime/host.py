"""Runs each execution's QuickJS VM on a thread of its own (the Python counterpart of upstream's worker launcher).

QuickJS runs synchronously, so a script that computes would block the thread it runs on. Each execution gets a fresh
thread and a fresh VM (its own wasm instance); wasmtime releases the GIL while wasm runs, so the host's event loop
keeps going. The thread talks to the loop only through messages: the VM's go to :meth:`VmEvents.message` in batches
(one ``call_soon_threadsafe`` per batch; output alone is batched for up to 2 ms), the host's results go into the
thread's queue.

Stopping sets the interrupt flag the VM polls and wakes the thread, which then drops the VM and exits; ``stop()``
does not wait for that. A trapped or crashed VM fails only its own execution as a ``sandbox`` error.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Final, cast

from ..types import CodemodeError
from .execution import VmChannel, VmEvents, VmLauncher, error_message
from .protocol import CrashMessage, DoneMessage, HostToVmMessage, OutputMessage, VmStartData, VmToHostMessage
from .quickjs import MAX_STACK_SIZE, compiled_module, load_module
from .vm import start_vm

__all__ = ["quickjs_launcher"]

#: Stack of a VM thread: room for wasmtime's 2 MiB wasm stack and the host frames around it, on every platform
#: (macOS gives other threads 512 KiB by default).
_THREAD_STACK_SIZE: Final = 16 * 1024 * 1024
#: How long output alone waits to be handed to the loop with more output.
_OUTPUT_BATCH_SECONDS: Final = 0.002
_thread_stack_lock = threading.Lock()
_STOP: Final = object()


def quickjs_launcher(
    *,
    wasm_path: str | os.PathLike[str] | None = None,
    max_stack_size: int = MAX_STACK_SIZE,
    interrupt_budget: int | None = None,
) -> VmLauncher:
    """A launcher that runs each execution's VM on a thread of its own.

    Args:
        wasm_path: ``quickjs.wasm`` to load (default: the bundled ``quickjs-wasi`` 3.6.2 build).
        max_stack_size: QuickJS's stack limit; deeper recursion throws a catchable ``RangeError``.
        interrupt_budget: Ends a script after this many interrupt polls (QuickJS polls about every 10,000 opcodes)
            with ``kind="timeout"``, whatever the deadline. Default: no budget. Part of the fingerprint.
    """

    async def launch(data: VmStartData, events: VmEvents) -> VmChannel:
        try:
            module = compiled_module(wasm_path) or await asyncio.to_thread(load_module, wasm_path)
        except Exception as error:
            raise RuntimeError(f"Failed to load QuickJS: {error_message(error)}") from error
        runner = _VmThread(data, events, module, max_stack_size, interrupt_budget)
        runner.start()
        return runner

    return launch


class _VmThread:
    """One execution's thread: the channel the host posts results to."""

    def __init__(
        self,
        data: VmStartData,
        events: VmEvents,
        module: object,
        max_stack_size: int,
        interrupt_budget: int | None,
    ) -> None:
        self._data = data
        self._events = events
        self._module = module
        self._max_stack_size = max_stack_size
        self._budget = interrupt_budget
        self._loop = asyncio.get_running_loop()
        self._inbox: queue.SimpleQueue[object] = queue.SimpleQueue()
        #: Polled by the VM: set by stop().
        self._stopped = threading.Event()
        self._polls = 0
        self._exhausted = False
        #: Thread side: a terminal message went out; later ones (a crash caused by interrupting it) are noise.
        self._settled = False
        #: Messages for the loop, handed over in batches: one wakeup of the loop per batch, not per message.
        self._outbox: list[VmToHostMessage] = []
        self._outbox_lock = threading.Lock()
        #: The flush the outbox waits for: ``now`` (a call or the outcome is in it), ``later`` (only output), or none.
        self._pending: str | None = None

    def start(self) -> None:
        with _thread_stack_lock:
            previous = threading.stack_size(_THREAD_STACK_SIZE)
            try:
                thread = threading.Thread(target=self._main, name="pi-codemode-quickjs", daemon=True)
                thread.start()
            finally:
                threading.stack_size(previous)

    # ── channel (event loop) ─────────────────────────────────────────────────

    def post(self, message: HostToVmMessage) -> None:
        if not self._stopped.is_set():
            self._inbox.put(message)

    async def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._inbox.put(_STOP)

    def _flush_later(self) -> None:
        self._loop.call_later(_OUTPUT_BATCH_SECONDS, self._flush)

    def _flush(self) -> None:
        with self._outbox_lock:
            messages, self._outbox = self._outbox, []
            self._pending = None
        for message in messages:
            if self._stopped.is_set():
                return
            # The interrupt surfaces as a crash or a failed done; report what caused it instead.
            if self._exhausted and isinstance(message, (DoneMessage, CrashMessage)):
                self._events.failure(
                    CodemodeError("timeout", f"Execution exceeded its interrupt budget of {self._budget}")
                )
                return
            self._events.message(message)

    # ── thread ───────────────────────────────────────────────────────────────

    def _should_interrupt(self) -> bool:
        if self._stopped.is_set() or self._exhausted:
            return True
        if self._budget is None:
            return False
        self._polls += 1
        if self._polls <= self._budget:
            return False
        self._exhausted = True
        return True

    def _post(self, message: VmToHostMessage) -> None:
        terminal = isinstance(message, (DoneMessage, CrashMessage))
        if terminal and self._settled:
            return
        if terminal:
            self._settled = True
        # Output only matters with the outcome, and goes along with the next call or the outcome, which are flushed at
        # once; on its own it waits up to a few milliseconds, so a script that prints in a loop does not wake the loop
        # (and take the GIL from this thread) for every item.
        urgent = not isinstance(message, OutputMessage)
        with self._outbox_lock:
            self._outbox.append(message)
            if self._pending == "now" or (self._pending == "later" and not urgent):
                return
            self._pending = "now" if urgent else "later"
        try:
            self._loop.call_soon_threadsafe(self._flush if urgent else self._flush_later)
        except RuntimeError:  # the loop is closed: nobody reads the result any more
            self._stopped.set()

    def _main(self) -> None:
        try:
            session = start_vm(
                self._data,
                self._post,
                self._should_interrupt,
                self._module,  # pyright: ignore[reportArgumentType]
                max_stack_size=self._max_stack_size,
                interrupt_budget=self._budget,
            )
        except BaseException as error:
            self._post(CrashMessage(f"Failed to start QuickJS: {error_message(error)}"))
            return
        try:
            while not self._stopped.is_set():
                message = self._inbox.get()
                if message is _STOP or self._stopped.is_set():
                    break
                session.deliver(cast("HostToVmMessage", message))
        finally:
            session.dispose()
