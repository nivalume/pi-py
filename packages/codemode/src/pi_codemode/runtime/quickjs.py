"""QuickJS in WebAssembly, on wasmtime (the parts of ``quickjs-wasi``'s ``QuickJS`` class that code mode uses).

``quickjs.wasm`` is QuickJS-NG compiled to WASI by ``quickjs-wasi`` 3.6.2, the build upstream's sandbox pins, copied
unchanged next to this module. This module is the host side of that build's C API (``qjs_*`` exports, ``host_*``
imports) in Python: one :class:`QuickJS` is one wasm instance holding one QuickJS runtime and context.

Values are ``int`` pointers to heap-allocated ``JSValue``s, as in ``quickjs-wasi``; whoever gets one from a ``qjs_*``
function frees it with :meth:`QuickJS.free`, except the cached ``undefined``/``true``/``false``. A :class:`QuickJS` is
not thread-safe: create and use it on one thread. wasmtime releases the GIL while wasm runs, so a VM that computes on
its own thread does not hold up the host's event loop.

WASI is a shim of the six functions the build imports (clocks, randomness, ``fd_write`` and three stubs), so the VM
has no file system, environment or network; output to fd 1 and 2 is discarded. :class:`Clock` replaces the clocks and
randomness with virtual ones for deterministic runs.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

try:
    import wasmtime
    from wasmtime import FuncType, ValType
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "pi_codemode's sandbox runs scripts in QuickJS on wasmtime, which is not installed. "
        "Install it with `pip install 'pi-ai-python[codemode]'` (or `pip install wasmtime`)."
    ) from error

try:  # wasmtime-py's internals, for calls and reads without its per-call checks; the public API is the fallback.
    from wasmtime import _ffi as _wasmtime_ffi
    from wasmtime._func import maybe_raise_last_exn as _maybe_raise_last_exn
    from wasmtime._trap import Trap as _Trap
except ImportError:  # pragma: no cover - depends on the wasmtime version
    _wasmtime_ffi = None
    _maybe_raise_last_exn = None
    _Trap = None

__all__ = [
    "INTRINSICS_ALL",
    "INTRINSICS_WEAK_REF",
    "MAX_STACK_SIZE",
    "QUICKJS_WASI_VERSION",
    "WASM_PATH",
    "Clock",
    "JSException",
    "QuickJS",
    "compiled_module",
    "load_module",
]

#: The ``quickjs-wasi`` release ``quickjs.wasm`` comes from; part of the deterministic fingerprint.
QUICKJS_WASI_VERSION: Final = "3.6.2"
WASM_PATH: Final = Path(__file__).with_name("quickjs.wasm")
#: Largest QuickJS stack limit for this build (half of its 1 MiB linker-defined stack), as ``quickjs-wasi``'s.
MAX_STACK_SIZE: Final = 512 * 1024
#: Native stack wasmtime gives wasm code, the most it allows. Deep recursion that QuickJS checks (calls, callbacks,
#: ``toJSON``, proxies) throws a catchable ``RangeError`` at :data:`MAX_STACK_SIZE` well before this; recursion it
#: does not check (``JSON.parse`` of deeply nested input) traps, which ends the VM.
_MAX_WASM_STACK: Final = 2 * 1024 * 1024
#: ``Intrinsics`` of ``quickjs-wasi``: all of them, and ``WeakRef``/``FinalizationRegistry``.
INTRINSICS_ALL: Final = 0xFFFFFFFF
INTRINSICS_WEAK_REF: Final = 1 << 9

_ERRNO_SUCCESS: Final = 0
_ERRNO_BADF: Final = 8
_ERRNO_NOSYS: Final = 52
_CLOCK_REALTIME: Final = 0
_CLOCK_MONOTONIC: Final = 1

_lock = threading.Lock()
_engine: wasmtime.Engine | None = None
_modules: dict[Path, wasmtime.Module] = {}


def _engine_of() -> wasmtime.Engine:
    """The engine every module is compiled with and every VM runs on."""
    global _engine
    if _engine is None:
        config = wasmtime.Config()
        config.max_wasm_stack = _MAX_WASM_STACK
        _engine = wasmtime.Engine(config)
    return _engine


def compiled_module(path: str | os.PathLike[str] | None = None) -> wasmtime.Module | None:
    """The module :func:`load_module` compiled for ``path``, if it has."""
    return _modules.get(Path(path) if path is not None else WASM_PATH)


def load_module(path: str | os.PathLike[str] | None = None) -> wasmtime.Module:
    """Read and compile ``quickjs.wasm`` (default: the bundled one) once per path and process.

    Compiling takes a few hundred milliseconds; call this off the event loop. A failed load is retried on the next
    call.
    """
    resolved = Path(path) if path is not None else WASM_PATH
    with _lock:
        module = _modules.get(resolved)
        if module is None:
            module = wasmtime.Module(_engine_of(), resolved.read_bytes())
            _modules[resolved] = module
        return module


def _caller(store: wasmtime.Store, function: wasmtime.Func) -> Callable[..., Any]:
    """A fast way to call ``function``, which takes and returns only ``i32``/``f64`` values.

    ``Func.__call__`` looks the function's type up and converts every argument on each call, about 25 µs; QuickJS's
    C API is called several times per script output item. This reads the type once and calls ``wasmtime_func_call``
    directly, raising traps and host callback exceptions as wasmtime-py does. Without those internals it is
    ``Func.__call__``.
    """
    ffi, raise_last, trap_type = _wasmtime_ffi, _maybe_raise_last_exn, _Trap
    public = lambda *args: function(store, *args)  # noqa: E731
    if ffi is None or raise_last is None or trap_type is None:
        return public
    try:
        kind = function.type(store)
        i32, f64 = ValType.i32(), ValType.f64()
        if any(value not in (i32, f64) for value in (*kind.params, *kind.results)) or len(kind.results) > 1:
            return public
        params = [ffi.WASMTIME_I32 if param == i32 else ffi.WASMTIME_F64 for param in kind.params]
        returns = None if not kind.results else kind.results[0] == i32
        context = store._context()  # pyright: ignore[reportPrivateUsage]
        reference = ctypes.byref(function._func)  # pyright: ignore[reportPrivateUsage]
        params_type = ffi.wasmtime_val_t * len(params)
        results_type = ffi.wasmtime_val_t * (0 if returns is None else 1)
        trap_pointer = ctypes.POINTER(ffi.wasm_trap_t)
        call = ffi.wasmtime_func_call
        error_type = wasmtime.WasmtimeError
        i32_kind = ffi.WASMTIME_I32
    except Exception:  # pragma: no cover - depends on the wasmtime version
        return public
    count = len(params)
    results = 0 if returns is None else 1

    def fast(*args: Any) -> Any:
        raw = params_type()
        for index in range(count):
            raw[index].kind = params[index]
            if params[index] == i32_kind:
                raw[index].of.i32 = args[index]
            else:
                raw[index].of.f64 = args[index]
        out = results_type()
        trap = trap_pointer()
        error = call(context, reference, raw, count, out, results, ctypes.byref(trap))
        if trap:
            exception = trap_type._from_ptr(trap)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
            raise_last()
            raise exception
        if error:
            raise_last()
            raise error_type._from_ptr(error)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        if returns is None:
            return None
        return out[0].of.i32 if returns else out[0].of.f64

    return fast


def _signed(value: int) -> int:
    """A 32-bit value as the signed ``i32`` wasmtime passes."""
    value &= 0xFFFFFFFF
    return value - 0x1_0000_0000 if value >= 0x8000_0000 else value


def _mulberry32(seed: int) -> Callable[[], int]:
    """mulberry32, upstream's PRNG for ``random_get``: one 32-bit draw per call."""
    state = seed & 0xFFFFFFFF

    def imul(a: int, b: int) -> int:
        return (a * b) & 0xFFFFFFFF

    def draw() -> int:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        value = imul(state ^ (state >> 15), 1 | state)
        value = ((value + imul(value ^ (value >> 7), 61 | value)) & 0xFFFFFFFF) ^ value
        return (value ^ (value >> 14)) & 0xFFFFFFFF

    return draw


class Clock:
    """The virtual clock and randomness of a deterministic run, in ms since the epoch.

    The realtime clock reads :attr:`now`, the monotonic clock ``now - started_at``. While the context is created
    (:attr:`creating`), the realtime clock reads a value derived from the seed instead, because QuickJS seeds
    ``Math.random()`` from the first clock read, and which read comes first is the engine's business.
    ``random_get`` draws from mulberry32 over the seed.
    """

    __slots__ = ("_random", "creating", "now", "seed", "started_at")

    def __init__(self, started_at: float, seed: int) -> None:
        self.started_at = started_at
        self.now = started_at
        self.seed = seed & 0xFFFFFFFF
        self.creating = True
        self._random = _mulberry32(self.seed ^ 0x9E3779B9)

    def nanoseconds(self, clock_id: int) -> int:
        if self.creating:
            # QuickJS seeds Math.random() from this reading in microseconds and turns 0 into 1, so seed s reads s + 1.
            return (self.seed + 1) * 1000 if clock_id == _CLOCK_REALTIME else 0
        if clock_id == _CLOCK_REALTIME:
            return _round_half_up(self.now) * 1_000_000
        return _round_half_up(self.now - self.started_at) * 1_000_000

    def random_bytes(self, size: int) -> bytes:
        out = bytearray()
        while len(out) < size:
            out += self._random().to_bytes(4, "little")
        return bytes(out[:size])


def _round_half_up(value: float) -> int:
    """JavaScript's ``Math.round``."""
    return math.floor(value + 0.5)


def _host_offset_seconds(time_seconds: int) -> int:
    """The host's UTC offset at ``time_seconds``, east positive, as ``quickjs-wasi``'s ``'host'`` timezone."""
    try:
        return time.localtime(time_seconds).tm_gmtoff
    except (OverflowError, OSError, ValueError):
        return time.localtime().tm_gmtoff


class JSException(Exception):
    """The script threw: its ``name``, ``message`` and ``stack``, read when it was caught."""

    def __init__(self, name: str, message: str, stack: str | None) -> None:
        super().__init__(f"{name}: {message}" if message else name)
        self.name = name
        self.message = message
        self.stack = stack


#: ``host_call``'s callbacks: ``(this, *args)`` as borrowed value pointers, returning a value pointer the trampoline
#: duplicates.
HostFunction = Callable[..., int]


class QuickJS:
    """One QuickJS VM in its own wasm instance."""

    def __init__(
        self,
        module: wasmtime.Module,
        *,
        memory_limit: int | None = None,
        max_stack_size: int = MAX_STACK_SIZE,
        interrupt: Callable[[], bool] | None = None,
        clock: Clock | None = None,
        timezone_offset_minutes: int | None = None,
        intrinsics: int | None = None,
    ) -> None:
        """
        Args:
            memory_limit: QuickJS's allocation limit in bytes; allocations beyond it throw ``InternalError: out of
                memory`` in the script.
            max_stack_size: QuickJS's stack limit; deeper recursion throws a catchable ``RangeError``.
            interrupt: Polled while the VM runs; returning ``True`` throws an uncatchable error in the script.
            clock: Deterministic clocks and randomness. Default: the host's.
            timezone_offset_minutes: A fixed zone (``getTimezoneOffset`` convention: positive west of UTC).
                Default: the host's.
            intrinsics: ``quickjs-wasi``'s ``Intrinsics`` bitmask. Default: all.
        """
        if not 0 <= max_stack_size <= MAX_STACK_SIZE:
            raise ValueError(f"max_stack_size must be an integer between 0 and {MAX_STACK_SIZE}")
        engine = _engine_of()
        self._store = wasmtime.Store(engine)
        self._callbacks: dict[str, HostFunction] = {}
        self._interrupt = interrupt
        self._clock = clock
        self._timezone_offset_minutes = timezone_offset_minutes
        self._disposed = False
        self._memory: wasmtime.Memory | None = None
        linker = wasmtime.Linker(engine)
        self._define_imports(linker)
        # Exports are looked up by name through the linker: wasmtime-py's own export table wraps all 450 of them.
        linker.define_instance(self._store, "quickjs", linker.instantiate(self._store, module))
        self._linker = linker
        memory = linker.get(self._store, "quickjs", "memory")
        assert isinstance(memory, wasmtime.Memory)
        self._memory = memory
        self._fn: dict[str, Callable[..., Any]] = {}
        self._memory_data = self._memory_data_of(memory)
        self._call("_initialize")
        status = self._call("qjs_init") if intrinsics is None else self._call("qjs_init2", _signed(intrinsics))
        if status != 0:
            raise RuntimeError("Failed to initialize QuickJS runtime")
        if clock is not None:
            clock.creating = False
        if memory_limit is not None:
            self._call("qjs_set_memory_limit", _signed(min(memory_limit, 0xFFFFFFFF)))
        self._call("qjs_set_max_stack_size", max_stack_size)
        if interrupt is not None:
            self._call("qjs_set_interrupt_handler", 1)
        #: Where ``qjs_get_string_len`` writes a string's length; allocated once.
        self._length_pointer: int = self._call("wasm_malloc", 4)
        if self._length_pointer == 0:
            raise MemoryError("wasm_malloc failed")
        self.undefined: int = self._call("qjs_get_undefined")
        self.true: int = self._call("qjs_get_true")
        self.false: int = self._call("qjs_get_false")

    # ── imports ─────────────────────────────────────────────────────────────

    def _define_imports(self, linker: wasmtime.Linker) -> None:
        i32, i64 = ValType.i32(), ValType.i64()
        env, wasi = "env", "wasi_snapshot_preview1"
        linker.define_func(env, "host_call", FuncType([i32] * 5, [i32]), self._host_call)
        linker.define_func(env, "host_interrupt", FuncType([], [i32]), self._host_interrupt)
        linker.define_func(env, "host_promise_rejection", FuncType([i32] * 3, []), self._host_promise_rejection)
        linker.define_func(env, "host_module_normalize", FuncType([i32] * 2, [i32]), lambda _base, _name: 0)
        linker.define_func(env, "host_module_load", FuncType([i32] * 2, [i32]), lambda _name, _length: 0)
        linker.define_func(env, "host_get_timezone_offset", FuncType([i32] * 2, [i32]), self._host_timezone_offset)
        linker.define_func(wasi, "clock_time_get", FuncType([i32, i64, i32], [i32]), self._clock_time_get)
        linker.define_func(wasi, "fd_close", FuncType([i32], [i32]), lambda _fd: _ERRNO_NOSYS)
        linker.define_func(wasi, "fd_fdstat_get", FuncType([i32, i32], [i32]), self._fd_fdstat_get)
        linker.define_func(wasi, "fd_seek", FuncType([i32, i64, i32, i32], [i32]), lambda *_args: _ERRNO_NOSYS)
        linker.define_func(wasi, "fd_write", FuncType([i32] * 4, [i32]), self._fd_write)
        linker.define_func(wasi, "random_get", FuncType([i32, i32], [i32]), self._random_get)

    def _host_call(self, name_ptr: int, name_length: int, this: int, argc: int, argv: int) -> int:
        # The trampoline owns `this` and the arguments and frees them after the call.
        name = self.read(name_ptr, name_length).decode("utf-8", "surrogatepass")
        callback = self._callbacks.get(name)
        try:
            if callback is None:
                raise RuntimeError(f'Host callback "{name}" is not registered')
            args = struct.unpack(f"<{argc}I", self.read(argv, 4 * argc)) if argc > 0 and argv != 0 else ()
            result = callback(this, *args)
            return self._call("qjs_dup_value", result)
        except BaseException as error:
            # Thrown inside QuickJS; 0 tells the trampoline an exception is pending.
            self._throw(str(error) or type(error).__name__)
            return 0

    def _host_interrupt(self) -> int:
        try:
            return 1 if self._interrupt is not None and self._interrupt() else 0
        except BaseException:
            return 1

    def _host_promise_rejection(self, promise: int, reason: int, _handled: int) -> None:
        self._call("qjs_free_value", promise)
        self._call("qjs_free_value", reason)

    def _host_timezone_offset(self, high: int, low: int) -> int:
        """The UTC offset in seconds (east positive) at the time ``high << 32 | low`` in seconds."""
        if self._timezone_offset_minutes is not None:
            return -self._timezone_offset_minutes * 60
        seconds = ((high & 0xFFFFFFFF) << 32) | (low & 0xFFFFFFFF)
        if seconds >= 1 << 63:
            seconds -= 1 << 64
        return _host_offset_seconds(seconds)

    def _clock_time_get(self, clock_id: int, _precision: int, result: int) -> int:
        if self._clock is not None:
            nanoseconds = self._clock.nanoseconds(clock_id)
        elif clock_id in (_CLOCK_REALTIME, _CLOCK_MONOTONIC):
            nanoseconds = time.time_ns()
        else:
            return _ERRNO_NOSYS
        self.write(result, struct.pack("<Q", nanoseconds & 0xFFFF_FFFF_FFFF_FFFF))
        return _ERRNO_SUCCESS

    def _fd_fdstat_get(self, fd: int, stat: int) -> int:
        if fd not in (1, 2):
            return _ERRNO_BADF
        # A character device without rights.
        self.write(stat, struct.pack("<BxHxxxxQQ", 2, 0, 0, 0))
        return _ERRNO_SUCCESS

    def _fd_write(self, _fd: int, iovs: int, count: int, written: int) -> int:
        # Engine diagnostics belong to nobody here: discarded. Reporting every byte as written keeps libc from retrying.
        lengths = struct.unpack(f"<{2 * count}I", self.read(iovs, 8 * count)) if count > 0 else ()
        self.write(written, struct.pack("<I", sum(lengths[1::2]) & 0xFFFFFFFF))
        return _ERRNO_SUCCESS

    def _random_get(self, buffer: int, length: int) -> int:
        data = self._clock.random_bytes(length) if self._clock is not None else os.urandom(length)
        self.write(buffer, data)
        return _ERRNO_SUCCESS

    # ── memory ──────────────────────────────────────────────────────────────

    def _memory_data_of(self, memory: wasmtime.Memory) -> Callable[[], int] | None:
        """The current base address of ``memory``, without wasmtime-py's per-read checks, when its internals allow."""
        ffi = _wasmtime_ffi
        try:
            assert ffi is not None
            context = self._store._context()  # pyright: ignore[reportPrivateUsage]
            reference = ctypes.byref(memory._memory)  # pyright: ignore[reportPrivateUsage]
            data = ffi.wasmtime_memory_data
            address = ctypes.addressof

            def base() -> int:
                return address(data(context, reference).contents)

            base()
            return base
        except Exception:  # pragma: no cover - depends on the wasmtime version
            return None

    def _call(self, name: str, *args: Any) -> Any:
        function = self._fn.get(name)
        if function is None:
            found = self._linker.get(self._store, "quickjs", name)
            assert isinstance(found, wasmtime.Func)
            function = self._fn[name] = _caller(self._store, found)
        return function(*args)

    def read(self, pointer: int, length: int) -> bytes:
        assert self._memory is not None
        if self._memory_data is not None:
            # The base address changes when memory grows, so it is read every time; the copy is one memcpy.
            return ctypes.string_at(self._memory_data() + pointer, length) if length > 0 else b""
        return bytes(self._memory.read(self._store, pointer, pointer + length))

    def write(self, pointer: int, data: bytes) -> None:
        assert self._memory is not None
        self._memory.write(self._store, data, pointer)

    def _write_string(self, text: str) -> tuple[int, int]:
        """A NUL-terminated WTF-8 copy of ``text`` in wasm memory; the caller frees it."""
        data = text.encode("utf-8", "surrogatepass")
        pointer = self._call("wasm_malloc", len(data) + 1)
        if pointer == 0:
            raise MemoryError("wasm_malloc failed")
        self.write(pointer, data + b"\0")
        return pointer, len(data)

    def _read_cstring(self, pointer: int) -> str:
        assert self._memory is not None
        size = self._memory.data_len(self._store)
        data = bytearray()
        while pointer + len(data) < size:
            chunk = self.read(pointer + len(data), min(64, size - pointer - len(data)))
            end = chunk.find(b"\0")
            if end != -1:
                data += chunk[:end]
                break
            data += chunk
        return data.decode("utf-8", "replace")

    # ── values ──────────────────────────────────────────────────────────────

    @property
    def quickjs_version(self) -> str:
        return self._read_cstring(self._call("qjs_get_quickjs_version"))

    def free(self, value: int) -> None:
        if value not in (self.undefined, self.true, self.false):
            self._call("qjs_free_value", value)

    def new_string(self, text: str) -> int:
        pointer, length = self._write_string(text)
        try:
            return self._call("qjs_new_string", pointer, length)
        finally:
            self._call("wasm_free", pointer)

    def new_number(self, value: float) -> int:
        return self._call("qjs_new_number", float(value))

    def new_function(self, name: str, callback: HostFunction) -> int:
        """A function that calls ``callback(this, *args)`` with borrowed value pointers; it returns a value pointer,
        and raising throws an ``Error`` with the exception's text in the script."""
        if name in self._callbacks:
            raise ValueError(f'Host callback "{name}" is already registered')
        self._callbacks[name] = callback
        pointer, length = self._write_string(name)
        try:
            return self._call("qjs_new_host_function", pointer, length, 0)
        finally:
            self._call("wasm_free", pointer)

    def is_undefined(self, value: int) -> bool:
        return self._call("qjs_is_undefined", value) != 0

    def to_bool(self, value: int) -> bool:
        return self._call("qjs_get_bool", value) != 0

    def to_number(self, value: int) -> float:
        return self._call("qjs_get_float64", value)

    def to_string(self, value: int) -> str:
        """``String(value)``, losslessly (embedded NULs and lone surrogates survive)."""
        pointer = self._call("qjs_get_string_len", value, self._length_pointer)
        if pointer == 0:
            exception = self._call("qjs_get_exception")
            self.free(exception)
            raise RuntimeError("Could not convert a value to a string")
        (length,) = struct.unpack("<I", self.read(self._length_pointer, 4))
        text = self.read(pointer, length).decode("utf-8", "surrogatepass")
        self._call("qjs_free_cstring", pointer)
        return text

    def get_prop(self, value: int, name: str) -> int:
        pointer, _ = self._write_string(name)
        try:
            return self._call("qjs_get_prop_string", value, pointer)
        finally:
            self._call("wasm_free", pointer)

    def _throw(self, message: str) -> None:
        error = self._call("qjs_new_error")
        text = self.new_string(message)
        pointer, _ = self._write_string("message")
        try:
            self._call("qjs_set_prop_string", error, pointer, text)
        finally:
            self._call("wasm_free", pointer)
            self.free(text)
        self._call("qjs_throw", error)
        self.free(error)

    def _check(self, result: int) -> int:
        if self._call("qjs_is_exception", result) == 0:
            return result
        self.free(result)
        exception = self._call("qjs_get_exception")
        try:
            raise self._describe(exception)
        finally:
            self.free(exception)

    def _describe(self, exception: int) -> JSException:
        """``quickjs-wasi``'s ``JSException`` fields of a thrown value."""

        def prop(name: str) -> str | None:
            value = self.get_prop(exception, name)
            try:
                return None if self.is_undefined(value) else self.to_string(value)
            finally:
                self.free(value)

        name = prop("name") or "Error"
        message = prop("message")
        return JSException(name, self.to_string(exception) if message is None else message, prop("stack"))

    def eval(self, code: str, filename: str = "<eval>") -> int:
        """Evaluate a global script; raises :class:`JSException` when it throws."""
        code_pointer, code_length = self._write_string(code)
        filename_pointer, _ = self._write_string(filename)
        try:
            result = self._call("qjs_eval", code_pointer, code_length, filename_pointer, 0)
        finally:
            self._call("wasm_free", code_pointer)
            self._call("wasm_free", filename_pointer)
        return self._check(result)

    def call(self, function: int, this: int, *args: int) -> int:
        """Call ``function``; raises :class:`JSException` when it throws."""
        argv = 0
        if args:
            argv = self._call("wasm_malloc", 4 * len(args))
            if argv == 0:
                raise MemoryError("wasm_malloc failed")
            self.write(argv, struct.pack(f"<{len(args)}I", *args))
        try:
            result = self._call("qjs_call", function, this, len(args), argv)
        finally:
            if argv:
                self._call("wasm_free", argv)
        return self._check(result)

    def is_job_pending(self) -> bool:
        return self._call("qjs_is_job_pending") != 0

    def execute_pending_job(self) -> int:
        """Runs one promise job: ``1`` ran one, ``0`` none was pending, ``< 0`` it failed (an uncatchable error)."""
        return self._call("qjs_execute_pending_job")

    def dispose(self) -> None:
        """Drop the instance: its memory goes with the store, without running finalizers in the VM."""
        if self._disposed:
            return
        self._disposed = True
        self._callbacks.clear()
        self._fn = {}
        self._memory = None
