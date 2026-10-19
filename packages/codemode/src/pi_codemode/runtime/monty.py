"""Drives one Monty session through one script (the Python counterpart of ``vm.ts`` and ``host.ts``).

Each execution checks a session out of an ``AsyncMonty`` pool (a dedicated ``monty`` worker
subprocess), feeds it the prelude, then starts the script with ``feed_start`` and answers
every suspension itself:

- ``__codemode_call__`` / ``__codemode_global__``: a tool or global call. The arguments are
  checked to be JSON, the call goes to the host as a ``CallMessage``, and the sandbox gets a
  pending future at once, so the call runs eagerly and ``asyncio.gather`` over calls runs them
  concurrently on the host.
- ``__codemode_exit__`` / ``__codemode_fail__``: the script ends. The snapshot is never
  resumed; leaving the session discards the suspended feed, so nothing can catch it.
- OS calls: sleeps (real ones in ``execute()``, virtual in deterministic runs), the virtual
  clock and seeded ``os.urandom()`` in deterministic runs; everything else (files,
  environment) gets Monty's default denial.
- A future snapshot (every sandbox task waits): the host's results are handed in. A wait that
  no pending call or sleep can end fails at once instead of hanging.

Output reaches the host as records printed by the prelude; Monty delivers printed output on a
background thread, flushed before every host call, so the records are queued under a lock and
drained on the event loop before each snapshot is handled and whenever new output arrives.

Deterministic mode (``CodemodeSandboxBase.start()``): the session's ``os_policy`` routes the
clock and sleeps to the host and seeds ``random``; the clock reads ``started_at`` and then the
``now`` of the result being processed, never going backwards, and a sleep advances it by its
duration without waiting. Results carry a ``seq`` and are applied one at a time, strictly in
that order, buffering early ones. The driver reports ``started`` with its fingerprint before
the script runs and ``idle`` with the number of calls issued after each drain.

Stopping cancels the driving task: a worker that is computing is killed by Monty, and the pool
replaces it. A crashed worker (``MontyCrashedError``) ends the execution as a sandbox error
without harming the host.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import datetime as dt
import json
import math
import os
import random
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from importlib import metadata
from typing import Any, Final, Literal

try:
    import pydantic_monty
    from pydantic_monty import (
        AsyncFunctionSnapshot,
        AsyncFutureSnapshot,
        AsyncNameLookupSnapshot,
        MontyComplete,
        MontyCrashedError,
        MontyError,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
    )
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "pi_codemode's sandbox runs scripts in Monty, which is not installed. "
        "Install it with `pip install 'pi-ai-python[codemode]'` (or `pip install pydantic-monty`)."
    ) from error

from ..types import CodemodeDuring, CodemodeError, CodemodeImageOutput, CodemodeTextOutput
from .execution import VmChannel, VmEvents, VmLauncher
from .prelude import MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, OUTPUT_LIMIT_MESSAGE, PRELUDE_SOURCE, TOOLS_HINT
from .protocol import (
    CallMessage,
    CrashMessage,
    DoneMessage,
    HostToVmMessage,
    IdleMessage,
    OutputMessage,
    StartedMessage,
    VmStartData,
)

__all__ = [
    "DEFAULT_MAX_RECURSION_DEPTH",
    "DETERMINISTIC_VERSION",
    "SCRIPT_FILENAME",
    "PoolProvider",
    "fingerprint_of",
    "host_timezone",
    "monty_launcher",
]

#: Bumped whenever this module or the prelude changes what a deterministic script observes,
#: beyond what the fingerprint's other parts (Monty versions, prelude hash, limits) capture.
DETERMINISTIC_VERSION: Final = 1
#: The file name tracebacks give the script.
SCRIPT_FILENAME: Final = "codemode.py"
#: Python's default recursion limit; deeper recursion raises a catchable ``RecursionError``.
DEFAULT_MAX_RECURSION_DEPTH: Final = 1000
#: Monty counts host round trips per checkout and cannot disable the limit; this is no limit in practice.
_MAX_SUSPENSIONS: Final = 2**63 - 1
#: The feed that holds the script: the prelude is the session's first feed, the script its second.
_SCRIPT_FEED: Final = "<python-input-1>"
_MONOTONIC_CALLERS: Final = frozenset(
    {"time.monotonic", "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns"}
)
_MAX_URANDOM_BYTES: Final = 1024 * 1024
#: Longest host timer; longer sleeps never end on their own (the deadline or an abort ends them).
_MAX_TIMER_SECONDS: Final = 86_400 * 365
_IMAGE_TYPES: Final = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_STALL_MESSAGE: Final = (
    "The script is waiting on something that can never finish: no tool call or sleep is pending, "
    "and nothing else can resume it."
)
_MISSING_ATTRIBUTE: Final = re.compile(r"^'([A-Za-z_][A-Za-z0-9_]*)' object has no attribute '(.*)'$", re.S)

#: Returns the pool to check sessions out of: an entered ``AsyncMonty`` (or ``AsyncMontyWebsocket``).
PoolProvider = Callable[[], Awaitable[Any]]


def _fnv1a(text: str) -> str:
    """FNV-1a over UTF-16 code units, as 8 hex digits, like upstream's prelude hash."""
    value = 0x811C9DC5
    data = text.encode("utf-16-le")
    for index in range(0, len(data), 2):
        value ^= data[index] | (data[index + 1] << 8)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return f"{value:08x}"


def _runtime_version() -> str:
    try:
        return metadata.version("pydantic-monty-runtime")
    except metadata.PackageNotFoundError:
        return "unknown"


_FINGERPRINT_BASE: Final = (
    f"pydantic-monty {pydantic_monty.__version__}, monty {_runtime_version()}, "
    f"prelude {_fnv1a(PRELUDE_SOURCE)}, deterministic {DETERMINISTIC_VERSION}"
)


def fingerprint_of(max_recursion_depth: int, memory_limit_bytes: int | None) -> str:
    """What a deterministic run reports with ``started``: Monty versions, prelude hash and limits."""
    heap = "none" if memory_limit_bytes is None else str(memory_limit_bytes)
    return f"{_FINGERPRINT_BASE}, recursion {max_recursion_depth}, heap {heap}"


def host_timezone() -> str | dict[str, Any]:
    """The host's local zone for Monty's ``os_policy``: an IANA name when one can be found, else a fixed offset.

    ``TZ`` (as POSIX hosts read it) wins, then the ``/etc/localtime`` link; otherwise the current
    UTC offset of the host, which misses later daylight saving changes.
    """
    candidates: list[str] = []
    tz = os.environ.get("TZ")
    if tz:
        candidates.append(tz.removeprefix(":"))
    else:
        try:
            target = os.path.realpath("/etc/localtime")
        except OSError:
            target = ""
        if "zoneinfo/" in target:
            candidates.append(target.split("zoneinfo/", 1)[1])
    for name in candidates:
        if re.fullmatch(r"[A-Za-z0-9_+\-]+(/[A-Za-z0-9_+\-]+)*", name):
            try:
                from zoneinfo import ZoneInfo

                ZoneInfo(name)
            except Exception:
                continue
            return name
    local = dt.datetime.now().astimezone()
    offset = local.utcoffset() or dt.timedelta(0)
    return {"offset_seconds": int(offset.total_seconds()), "name": local.tzname() or "UTC"}


def _comparable(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _missing_member_message(label: str, names: list[str], member: str) -> str:
    """The prelude's ``__codemode_missing__``, for attribute reads Monty itself rejected."""
    wanted = _comparable(member)
    close = [name for name in names if _comparable(name) == wanted]
    if not close and wanted:
        close = [name for name in names if _comparable(name) in wanted or wanted in _comparable(name)]
    message = f"{label}.{member} does not exist."
    if close:
        message += " Did you mean " + ", ".join(f"{label}.{name}" for name in close[:5]) + "?"
    elif len(names) <= 20:
        message += " Available: " + ", ".join(names) + "."
    if label == "tools":
        message += f" {TOOLS_HINT}"
    return message + f' Check for a member with "{member}" in {label}.'


class _Ended(Exception):
    """The run has its outcome; the driver unwinds and leaves the session."""


def monty_launcher(
    pool: PoolProvider,
    *,
    max_recursion_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    timezone: str | Mapping[str, Any] | None = None,
    on_driver: Callable[[asyncio.Task[None]], None] | None = None,
) -> VmLauncher:
    """A launcher that runs each execution in a session of the pool ``pool()`` returns.

    Args:
        pool: Returns the entered pool; raising fails the execution with
            ``Failed to start Monty: <error>``.
        max_recursion_depth: Monty's recursion limit; part of the fingerprint.
        timezone: The zone of ``execute()`` runs (default: :func:`host_timezone`); deterministic runs use UTC.
        on_driver: Receives each driving task, so an owner can wait for sessions to be released.
    """

    async def launch(data: VmStartData, events: VmEvents) -> VmChannel:
        try:
            entered = await pool()
        except Exception as error:
            raise RuntimeError(f"Failed to start Monty: {error}") from error
        driver = _Driver(entered, data, events, max_recursion_depth, timezone)
        driver.start()
        if on_driver is not None and driver.task is not None:
            on_driver(driver.task)
        return driver

    return launch


class _Clock:
    """The deterministic run's clock, in ms since the epoch."""

    __slots__ = ("now", "started_at")

    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self.now = started_at


class _Driver:
    """One execution's session: implements the channel the host posts results to."""

    def __init__(
        self,
        pool: Any,
        data: VmStartData,
        events: VmEvents,
        max_recursion_depth: int,
        timezone: str | Mapping[str, Any] | None,
    ) -> None:
        self._pool = pool
        self._data = data
        self._events = events
        self._max_recursion_depth = max_recursion_depth
        self._timezone = timezone
        self._loop = asyncio.get_running_loop()
        self.task: asyncio.Task[None] | None = None
        self._stopped = False
        #: Set once the outcome is reported: the driver only leaves its session.
        self._ending = False
        deterministic = data.deterministic
        self._clock = None if deterministic is None else _Clock(deterministic[0])
        self._entropy = None if deterministic is None else random.Random(f"pi-codemode:{deterministic[1]}")
        #: Calls the script issued so far; its call ids are 1, 2, ...
        self._issued = 0
        #: Our call id -> Monty's call id of its future, and Monty ids of pending calls and sleeps.
        self._monty_ids: dict[int, int] = {}
        self._waiting_calls: set[int] = set()
        self._waiting_sleeps: set[int] = set()
        #: Results ready for the sandbox: by Monty id (``execute()``) or by seq (deterministic).
        self._ready: dict[int, HostToVmMessage] = {}
        self._settled_sleeps: list[int] = []
        self._early: dict[int, HostToVmMessage] = {}
        self._next_seq = 1
        #: The result being processed, 0 during the script's first run; reported with ``done``.
        self._current = 0
        self._idle_due = True
        self._wake = asyncio.Event()
        self._sleep_timers: list[asyncio.TimerHandle] = []
        # Output arrives on Monty's I/O thread.
        self._lock = threading.Lock()
        self._chunks: list[tuple[str, str]] = []
        self._partial: dict[str, str] = {}
        self._drain_scheduled = False
        self._output_chars = 0
        self._output_items = 0
        self._writes: dict[str, str | None] = {}
        tool_names: list[str] = []
        for _name, js_name, _description in data.tools:
            if js_name not in tool_names:
                tool_names.append(js_name)
        self._member_names: dict[str, list[str]] = {"tools": tool_names}
        for name, _spread in data.globals:
            label, dot, member = name.partition(".")
            if dot:
                self._member_names.setdefault(label, []).append(member)

    # -- channel ------------------------------------------------------------------------

    def start(self) -> None:
        self.task = asyncio.ensure_future(self._run())

    def post(self, message: HostToVmMessage) -> None:
        if self._stopped or self._ending:
            return
        try:
            if self._clock is None:
                monty_id = self._monty_ids.get(message.id)
                if monty_id is None or monty_id not in self._waiting_calls:
                    raise RuntimeError(f"Result {message.id} answers no pending call")
                self._ready[monty_id] = message
            else:
                seq = message.seq
                if type(seq) is not int or seq < self._next_seq or seq in self._early:
                    raise RuntimeError(
                        f"Result {message.id} has sequence number {seq}, expected {self._next_seq} or later, once"
                    )
                self._early[seq] = message
        except RuntimeError as error:
            self._crash(str(error))
            return
        self._wake.set()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for timer in self._sleep_timers:
            timer.cancel()
        # A driver that already reported its outcome is leaving its session; let it finish.
        if self.task is not None and not self.task.done() and not self._ending:
            self.task.cancel()

    # -- reporting ----------------------------------------------------------------------

    def _emit(self, message: object) -> None:
        if not self._stopped and not self._ending:
            self._events.message(message)

    def _end(self, message: object) -> None:
        self._drain_output()
        self._emit(message)
        self._ending = True
        raise _Ended

    def _crash(self, message: str) -> None:
        if self._stopped or self._ending:
            return
        self._ending = True
        self._events.message(CrashMessage(message))
        if self.task is not None and not self.task.done() and asyncio.current_task() is not self.task:
            self.task.cancel()

    def _bridge_broken(self, reason: str) -> None:
        if self._stopped or self._ending:
            return
        self._ending = True
        self._events.failure(
            CodemodeError(
                "sandbox", f"Sandbox bridge broken: {reason}. The script may have replaced the codemode helpers."
            )
        )
        raise _Ended

    def _during(self) -> CodemodeDuring | None:
        return None if self._clock is None else CodemodeDuring(self._current, self._issued)

    def _done_error(self, name: str, message: str, stack: str | None = None) -> None:
        error: dict[str, str | None] = {
            "name": name,
            "message": message,
            "stack": stack or (f"{name}: {message}" if message else name),
        }
        self._end(DoneMessage(ok=False, error=error, during=self._during()))

    # -- output -------------------------------------------------------------------------

    def _on_print(self, stream: Literal["stdout", "stderr"], text: str) -> None:
        """Monty's print callback, called on its I/O thread."""
        with self._lock:
            self._chunks.append((stream, text))
            if self._drain_scheduled:
                return
            self._drain_scheduled = True
        try:
            self._loop.call_soon_threadsafe(self._scheduled_drain)
        except RuntimeError:  # the loop is closed: nothing will read the output any more
            pass

    def _scheduled_drain(self) -> None:
        try:
            self._drain_output()
        except _Ended:
            if self.task is not None and not self.task.done():
                self.task.cancel()

    def _drain_output(self) -> None:
        with self._lock:
            chunks, self._chunks = self._chunks, []
            self._drain_scheduled = False
        for stream, text in chunks:
            lines = (self._partial.pop(stream, "") + text).split("\n")
            if lines[-1]:
                self._partial[stream] = lines[-1]
            for line in lines[:-1]:
                if self._stopped or self._ending:
                    return
                self._record(line)

    def _record(self, line: str) -> None:
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not isinstance(record, list) or not record or not all(isinstance(part, str) for part in record):
            self._bridge_broken("malformed output record")
            return
        kind = record[0]
        if kind in ("t", "c", "i"):
            if len(record) != (3 if kind == "i" else 2) or (kind == "i" and record[2] not in _IMAGE_TYPES):
                self._bridge_broken("malformed output record")
                return
            self._output_chars += len(record[1])
            self._output_items += 1
            if self._output_chars > MAX_OUTPUT_CHARS or self._output_items > MAX_OUTPUT_ITEMS:
                self._done_error("ValueError", OUTPUT_LIMIT_MESSAGE)
            item = (
                CodemodeImageOutput(record[1], record[2])
                if kind == "i"
                else CodemodeTextOutput(record[1], console=kind == "c")
            )
            self._emit(OutputMessage(item))
        elif kind == "s" and len(record) == 3:
            self._writes.pop(record[1], None)
            self._writes[record[1]] = record[2]
        elif kind == "d" and len(record) == 2:
            self._writes.pop(record[1], None)
            self._writes[record[1]] = None
        else:
            self._bridge_broken("malformed output record")

    # -- the run ------------------------------------------------------------------------

    def _limits(self) -> dict[str, Any]:
        limits: dict[str, Any] = {
            "max_suspensions": _MAX_SUSPENSIONS,
            "max_recursion_depth": self._max_recursion_depth,
        }
        if self._data.memory_limit_bytes is not None:
            limits["max_memory"] = self._data.memory_limit_bytes
        return limits

    def _os_policy(self) -> dict[str, Any]:
        deterministic = self._data.deterministic
        if deterministic is not None:
            return {
                "datetime": "call_host",
                "timezone": "utc",
                "sleep": "call_host",
                "random_start": {"seed": deterministic[1]},
                "process_time": "zero",
            }
        return {
            "timezone": dict(self._timezone)
            if isinstance(self._timezone, Mapping)
            else self._timezone or host_timezone()
        }

    async def _run(self) -> None:
        try:
            async with self._pool.checkout(
                script_name=SCRIPT_FILENAME, limits=self._limits(), os_policy=self._os_policy()
            ) as session:
                try:
                    await session.feed_run(
                        PRELUDE_SOURCE,
                        inputs={
                            "__codemode_tools__": [list(tool) for tool in self._data.tools],
                            "__codemode_globals__": [list(global_) for global_ in self._data.globals],
                            "__codemode_store__": dict(self._data.store),
                        },
                        print_callback=self._on_print,
                    )
                except MontyCrashedError:
                    raise
                except MontyError as error:
                    self._crash(f"Sandbox prelude failed: {_display(error)}")
                    return
                if self._data.deterministic is not None:
                    self._emit(StartedMessage(fingerprint_of(self._max_recursion_depth, self._data.memory_limit_bytes)))
                try:
                    snapshot = await session.feed_start(_script_source(self._data.code), print_callback=self._on_print)
                    await self._drive(snapshot)
                except (MontyRuntimeError, MontySyntaxError, MontyTypingError) as error:
                    self._script_error(error)
        except _Ended:
            pass
        except asyncio.CancelledError:
            if not self._stopped and not self._ending:
                raise
        except MontyCrashedError as error:
            self._crash(f"The sandbox worker crashed: {error}")
        except MontyError as error:
            self._crash(f"Sandbox failed: {_display(error)}")
        except Exception as error:
            self._crash(f"Sandbox driver failed: {type(error).__name__}: {error}")

    async def _drive(self, snapshot: Any) -> None:
        while True:
            self._drain_output()
            if self._stopped or self._ending:
                return
            if isinstance(snapshot, MontyComplete):
                self._complete(snapshot.output)
            elif isinstance(snapshot, AsyncFunctionSnapshot):
                snapshot = await self._function(snapshot)
            elif isinstance(snapshot, AsyncNameLookupSnapshot):
                # An undefined name: Monty raises its NameError.
                snapshot = await snapshot.resume()
            elif isinstance(snapshot, AsyncFutureSnapshot):
                snapshot = await self._wait(snapshot)
            else:
                self._crash(f"Unexpected snapshot {type(snapshot).__name__}")
                return

    def _complete(self, value: Any) -> None:
        try:
            value_json = None if value is None else json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            self._done_error("TypeError", str(error))
            return
        writes = tuple(self._writes.items())
        self._end(DoneMessage(ok=True, value=value_json, writes=writes, during=self._during()))

    async def _function(self, snapshot: Any) -> Any:
        if snapshot.is_os_function:
            return await self._os_call(snapshot)
        name = snapshot.function_name
        if name in ("__codemode_call__", "__codemode_global__"):
            return await self._call(snapshot, "tool" if name == "__codemode_call__" else "global")
        if name == "__codemode_exit__":
            self._end(DoneMessage(ok=True, value=None, writes=tuple(self._writes.items()), during=self._during()))
        if name == "__codemode_fail__":
            args = snapshot.args
            if len(args) != 2 or not all(isinstance(arg, str) for arg in args):
                self._bridge_broken("malformed failure")
            self._done_error(args[0], args[1])
        return await snapshot.resume({"exception": NameError(f"name '{name}' is not defined")})

    async def _call(self, snapshot: Any, target: Literal["tool", "global"]) -> Any:
        args = snapshot.args
        entries = self._data.tools if target == "tool" else self._data.globals
        if (
            len(args) != 3
            or type(args[0]) is not int
            or not 0 <= args[0] < len(entries)
            or not isinstance(args[2], dict)
        ):
            self._bridge_broken(f"malformed {target} call")
        index, positional, keywords = args
        if target == "global" and self._data.globals[index][1]:
            payload: Any = list(positional) + ([keywords] if keywords else [])
        elif keywords:
            if positional is not None:
                return await snapshot.resume(
                    {
                        "exception": TypeError(
                            "pass the arguments as one positional value or as keyword arguments, not both"
                        )
                    }
                )
            payload = keywords
        else:
            payload = positional
        try:
            args_json = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            return await snapshot.resume({"exception": TypeError(str(error))})
        self._issued += 1
        call_id = self._issued
        self._monty_ids[call_id] = snapshot.call_id
        self._waiting_calls.add(snapshot.call_id)
        self._emit(CallMessage(call_id, target, entries[index][0], args_json))
        return await snapshot.resume({"future": ...})

    async def _os_call(self, snapshot: Any) -> Any:
        name: str = snapshot.function_name
        args: tuple[Any, ...] = snapshot.args
        clock = self._clock
        if clock is not None:
            now = clock.now / 1000
            if name == "time.time":
                caller = args[0] if args else "time.time"
                value: Any = (clock.now - clock.started_at) / 1000 if caller in _MONOTONIC_CALLERS else now
                return await snapshot.resume({"return_value": value})
            if name == "datetime.now":
                zone = args[0] if args and isinstance(args[0], dt.tzinfo) else None
                instant = dt.datetime.fromtimestamp(now, dt.UTC)
                value = instant.replace(tzinfo=None) if zone is None else instant.astimezone(zone)
                return await snapshot.resume({"return_value": value})
            if name == "date.today":
                return await snapshot.resume({"return_value": dt.datetime.fromtimestamp(now, dt.UTC).date()})
            if name in ("asyncio.sleep", "time.sleep", "system.sleep", "system.async_sleep"):
                delay = _delay(args)
                if math.isfinite(delay):
                    clock.now += delay * 1000
                return await snapshot.resume({"return_value": None})
            if name == "os.urandom":
                size = _urandom_size(args)
                if isinstance(size, BaseException):
                    return await snapshot.resume({"exception": size})
                assert self._entropy is not None
                return await snapshot.resume({"return_value": self._entropy.randbytes(size)})
            return await snapshot.resume_not_handled()
        if name in ("asyncio.sleep", "system.async_sleep"):
            monty_id = snapshot.call_id
            self._waiting_sleeps.add(monty_id)
            delay = _delay(args)
            if delay < _MAX_TIMER_SECONDS:
                self._sleep_timers.append(self._loop.call_later(max(delay, 0), self._sleep_done, monty_id))
            return await snapshot.resume({"future": ...})
        if name in ("time.sleep", "system.sleep"):
            await asyncio.sleep(min(max(_delay(args), 0), _MAX_TIMER_SECONDS))
            return await snapshot.resume({"return_value": None})
        if name == "os.urandom":
            size = _urandom_size(args)
            if isinstance(size, BaseException):
                return await snapshot.resume({"exception": size})
            return await snapshot.resume({"return_value": os.urandom(size)})
        return await snapshot.resume_not_handled()

    def _sleep_done(self, monty_id: int) -> None:
        self._settled_sleeps.append(monty_id)
        self._wake.set()

    async def _wait(self, snapshot: Any) -> Any:
        pending = snapshot.pending_call_ids
        if not any(monty_id in self._waiting_calls or monty_id in self._waiting_sleeps for monty_id in pending):
            self._done_error("RuntimeError", _STALL_MESSAGE)
        if self._clock is not None:
            return await self._apply_next(snapshot)
        while not self._ready and not self._settled_sleeps:
            self._wake.clear()
            await self._wake.wait()
        results: dict[int, Any] = {}
        for monty_id, message in self._ready.items():
            self._waiting_calls.discard(monty_id)
            results[monty_id] = _external_result(message)
        for monty_id in self._settled_sleeps:
            self._waiting_sleeps.discard(monty_id)
            results[monty_id] = {"return_value": None}
        self._ready.clear()
        self._settled_sleeps.clear()
        return await snapshot.resume(results)

    async def _apply_next(self, snapshot: Any) -> Any:
        """Deterministic mode: report the drain that just ended, then apply the next result in ``seq`` order."""
        assert self._clock is not None
        if self._idle_due:
            self._idle_due = False
            self._emit(IdleMessage(self._current, self._issued))
        while self._next_seq not in self._early:
            self._wake.clear()
            await self._wake.wait()
        message = self._early.pop(self._next_seq)
        # Never backwards: a resumed host's clock may be behind the one that started the run.
        if message.now is not None and math.isfinite(message.now):
            self._clock.now = max(self._clock.now, message.now)
        self._current = self._next_seq
        self._next_seq += 1
        self._idle_due = True
        monty_id = self._monty_ids.get(message.id)
        if monty_id is None or monty_id not in self._waiting_calls:
            self._crash(f"Result {message.id} answers no pending call")
            raise _Ended
        self._waiting_calls.discard(monty_id)
        return await snapshot.resume({monty_id: _external_result(message)})

    # -- errors -------------------------------------------------------------------------

    def _script_error(self, error: MontyError) -> None:
        inner = error.exception()
        name = type(inner).__name__ if isinstance(inner, BaseException) else "Error"
        message = str(inner) if isinstance(inner, BaseException) else str(error)
        if isinstance(inner, KeyError) and len(inner.args) == 1 and isinstance(inner.args[0], str):
            message = inner.args[0]
        if name == "AttributeError":
            match = _MISSING_ATTRIBUTE.match(message)
            if match is not None and match.group(1) in self._member_names:
                message = _missing_member_message(match.group(1), self._member_names[match.group(1)], match.group(2))
        frames = error.traceback() if isinstance(error, (MontyRuntimeError, MontySyntaxError)) else []
        self._done_error(name, message, _render_stack(frames, name, message, isinstance(error, MontySyntaxError)))


def _script_source(code: str) -> str:
    """The script as fed to Monty: a session feed returns its trailing expression, but a script, like
    the body of a function, only returns what ``return`` gives. A script that parses gets a final
    ``None`` statement; one that does not is fed unchanged, so its syntax error points at its own lines."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    return _rename_shadowing_comprehension_targets(code, tree) + "\nNone"


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))


def _rename_shadowing_comprehension_targets(code: str, tree: ast.Module) -> str:
    """Work around Monty (1.2) resolving a top-level comprehension variable named like a builtin
    (``[id for id in xs]``, ``{type: n for type, n in pairs}``) to the builtin instead of the loop value.

    Such targets are renamed inside their comprehension, in place: only those names change, so line
    numbers and the rest of the script stay as written.
    """
    edits: list[tuple[int, int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _COMPREHENSIONS):
            continue
        shadowing: set[str] = set()
        for generator in node.generators:
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name) and target.id in _BUILTIN_NAMES:
                    shadowing.add(target.id)
        if not shadowing:
            continue
        # The first iterable is evaluated in the enclosing scope; everything else sees the targets.
        scoped: list[ast.AST] = [node.generators[0].target, *node.generators[0].ifs]
        for generator in node.generators[1:]:
            scoped.extend([generator.target, generator.iter, *generator.ifs])
        scoped.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
        for part in scoped:
            _collect_renames(part, shadowing, edits)
    if not edits:
        return code
    lines = code.splitlines(keepends=True)
    for line, start, end, name in sorted(set(edits), reverse=True):
        raw = lines[line - 1].encode()
        lines[line - 1] = (raw[:start] + f"__codemode_{name}".encode() + raw[end:]).decode()
    return "".join(lines)


def _collect_renames(node: ast.AST, names: set[str], edits: list[tuple[int, int, int, str]]) -> None:
    if isinstance(node, ast.Lambda):
        bound = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        bound.update(arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
        for default in (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)):
            _collect_renames(default, names, edits)
        names = names - bound
        if not names:
            return
        _collect_renames(node.body, names, edits)
        return
    if isinstance(node, ast.Name) and node.id in names and node.end_col_offset is not None:
        edits.append((node.lineno, node.col_offset, node.end_col_offset, node.id))
        return
    for child in ast.iter_child_nodes(node):
        _collect_renames(child, names, edits)


def _display(error: MontyError) -> str:
    if isinstance(error, (MontyRuntimeError, MontySyntaxError)):
        return error.display("type-msg")
    return str(error)


def _render_stack(frames: list[Any], name: str, message: str, syntax: bool) -> str:
    """A CPython-style traceback of the script's own frames, in ``codemode.py``."""
    head = f"{name}: {message}" if message else name
    lines: list[str] = []
    for frame in frames:
        if frame.filename != _SCRIPT_FEED:
            continue
        location = f'  File "{SCRIPT_FILENAME}", line {frame.line}'
        if frame.function_name is not None:
            location += f", in {frame.function_name}"
        lines.append(location)
        if frame.source_line:
            lines.append(f"    {frame.source_line.strip()}")
    if not lines:
        return head
    if syntax:
        return "\n".join([*lines, head])
    return "\n".join(["Traceback (most recent call last):", *lines, head])


def _external_result(message: HostToVmMessage) -> dict[str, Any]:
    if not message.ok:
        return {"exception": Exception(message.payload or "")}
    return {"return_value": None if message.payload is None else json.loads(message.payload)}


def _delay(args: tuple[Any, ...]) -> float:
    value = args[0] if args else 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return 0.0
    return float(value)


def _urandom_size(args: tuple[Any, ...]) -> int | BaseException:
    size = args[0] if args else None
    if type(size) is not int or size < 0:
        return TypeError("os.urandom() size must be a non-negative integer")
    if size > _MAX_URANDOM_BYTES:
        return MemoryError(f"os.urandom() size exceeds {_MAX_URANDOM_BYTES} bytes")
    return size
