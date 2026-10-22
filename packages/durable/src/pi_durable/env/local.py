"""The local ``ExecutionEnv``: files and processes of the host OS (port of ``env/node.ts``).

:class:`LocalExecutionEnv` is upstream's ``NodeExecutionEnv`` on asyncio. Blocking file system calls run
on the event loop's default executor (one thread hop per operation, never blocking the loop); commands
run as asyncio subprocesses in their own process group, so a timeout, an abort or :meth:`cleanup` kills
the whole tree. Every operation honors its :class:`~pi_chord.context.Context` like upstream: a cancelled
context is reported as an ``aborted`` result before any side effect and after each step that may block,
and a running command is killed when its context is aborted.

Python-specific notes:

- Messages of :class:`~pi_durable.env.FileError` values are ``str()`` of the ``OSError`` (upstream: Node's
  error message); their codes map the same ``errno`` values.
- ``list_files`` (optional upstream, not implemented by ``NodeExecutionEnv``) returns ``not_supported``.
- Native watching uses Linux inotify (see :mod:`pi_durable.env.local_watch`); elsewhere watchers poll.
"""

from __future__ import annotations

import asyncio
import errno
import functools
import math
import os
import re
import shutil
import signal as signal_module
import stat as stat_module
import sys
import tempfile
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, TypedDict, TypeVar, cast
from urllib.parse import unquote, urlsplit

from pi_ai.abort import AbortSignal
from pi_chord.context import Context

from ..ids import MAX_SAFE_INTEGER
from .decode import StreamDecoder
from .line_scan import LineScanner
from .types import (
    BinaryReader,
    CreateDirOptions,
    CreateTempFileOptions,
    DirPage,
    DirReader,
    ExecutionError,
    FileError,
    FileInfo,
    FileKind,
    FileListing,
    FileWatcher,
    LineScan,
    ListFilesOptions,
    OpenBinaryReaderOptions,
    ReadTextLinesOptions,
    RemoveOptions,
    Result,
    ScanLinesOptions,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputInfo,
    TextLine,
    TextLineReader,
    WatchChange,
    WatchTarget,
    err,
    ok,
    to_error,
)

if TYPE_CHECKING:
    from .local_watch import LocalWatchOptions

__all__ = ["LocalExecutionEnv", "LocalWatchOptions", "resolve_path", "to_file_error"]

T = TypeVar("T")

MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000
EXIT_STDIO_GRACE_MS = 100
#: Spill bytes buffered before reading the command's output pauses until they are written.
SPILL_HIGH_WATER_MARK = 1024 * 1024
#: Largest single read of a ``BinaryReader``, so a huge ``length`` allocates only as much as the file yields.
BINARY_READ_CHUNK = 1024 * 1024
_LINE_CHUNK = 64 * 1024


def _is_windows() -> bool:
    return sys.platform == "win32"


def _js_number(value: float) -> str:
    """A number as JavaScript prints it (``5`` rather than ``5.0``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _resolve_timeout_ms(timeout: float | None) -> Result[float | None, ExecutionError]:
    if timeout is None:
        return ok(None)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not math.isfinite(timeout) or timeout <= 0:
        return err(ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"))
    timeout_ms = timeout * 1000
    if timeout_ms > MAX_TIMEOUT_MS:
        return err(ExecutionError("timeout", f"Invalid timeout: maximum is {_js_number(MAX_TIMEOUT_SECONDS)} seconds"))
    return ok(timeout_ms)


def _file_url_path(url: str) -> str | None:
    """``fileURLToPath(url)``, or ``None`` where Node would throw."""
    parts = urlsplit(url)
    if _is_windows():
        path = unquote(parts.path)
        if parts.netloc not in ("", "localhost"):
            return f"\\\\{parts.netloc}{path.replace('/', chr(92))}"
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path.replace("/", "\\")
    if parts.netloc not in ("", "localhost") or "%2f" in parts.path.lower():
        return None
    return unquote(parts.path)


def resolve_path(cwd: str, path: str) -> str:
    """Resolve ``path`` against ``cwd`` like upstream: ``~``, ``~/...`` and ``file://`` URLs are expanded."""
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/") or (_is_windows() and normalized.startswith("~\\")):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif normalized.startswith("file://"):
        # Keep malformed URLs as ordinary paths so filesystem methods preserve their non-throwing contract.
        converted = _file_url_path(normalized)
        if converted is not None:
            normalized = converted
    return os.path.abspath(os.path.join(cwd, normalized))


def _join_path(parts: Sequence[str]) -> str:
    """``path.join(...parts)``: joined with the separator and normalized, keeping a trailing separator."""
    joined = os.sep.join(part for part in parts if part)
    if joined == "":
        return "."
    normalized = os.path.normpath(joined)
    if normalized.startswith("//") and not joined.startswith("//"):
        normalized = normalized[1:]
    if joined.endswith(("/", os.sep)) and not normalized.endswith(os.sep):
        normalized += os.sep
    return normalized


def _file_kind(st: os.stat_result) -> FileKind | None:
    mode = st.st_mode
    if stat_module.S_ISREG(mode):
        return "file"
    if stat_module.S_ISDIR(mode):
        return "directory"
    if stat_module.S_ISLNK(mode):
        return "symlink"
    return None


def _file_info(path: str, st: os.stat_result) -> Result[FileInfo, FileError]:
    kind = _file_kind(st)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    info: FileInfo = {
        "name": os.path.basename(path),
        "path": path,
        "kind": kind,
        "size": st.st_size,
        "mtimeMs": st.st_mtime_ns / 1e6,
    }
    return ok(info)


_ERRNO_CODES: dict[int, Any] = {
    errno.ENOENT: "not_found",
    errno.EACCES: "permission_denied",
    errno.EPERM: "permission_denied",
    errno.ENOTDIR: "not_directory",
    errno.EISDIR: "is_directory",
    errno.EINVAL: "invalid",
}


def to_file_error(error: object, fallback_path: str | None = None) -> FileError:
    """Map a raised error to a :class:`~pi_durable.env.FileError` (``errno`` decides the code)."""
    if isinstance(error, FileError):
        return error
    cause = to_error(error)
    if isinstance(error, OSError):
        filename = error.filename
        path = filename if isinstance(filename, str) else fallback_path
        code = _ERRNO_CODES.get(error.errno) if error.errno is not None else None
        if code is not None:
            return FileError(code, str(error), path, cause)
        return FileError("unknown", str(error), path, cause)
    if isinstance(error, asyncio.CancelledError):
        return FileError("aborted", str(error) or "aborted", fallback_path, cause)
    return FileError("unknown", str(cause), fallback_path, cause)


def _aborted(signal: AbortSignal | None, path: str | None = None) -> Result[Any, FileError] | None:
    return err(FileError("aborted", "aborted", path)) if signal is not None and signal.aborted else None


def _closed_result(what: str, path: str) -> Result[Any, FileError]:
    return err(FileError("invalid", f"{what} is closed", path))


def _symlink_refused(path: str, cause: BaseException | None = None) -> FileError:
    return FileError("invalid", "Refusing to follow a symbolic link", path, cause)


def _is_safe_integer(value: object) -> bool:
    return type(value) is int and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER  # type: ignore[operator]


class _Aborted(Exception):
    """Raised inside a worker thread that saw its context aborted between steps."""


def _check(signal: AbortSignal | None) -> None:
    if signal is not None and signal.aborted:
        raise _Aborted


async def _blocking(function: Callable[..., T], *args: Any) -> T:
    """Run a blocking call on the loop's default executor."""
    return await asyncio.get_running_loop().run_in_executor(None, functools.partial(function, *args))


def _pread(fd: int, size: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)
    os.lseek(fd, offset, os.SEEK_SET)  # pragma: no cover - Windows
    return os.read(fd, size)  # pragma: no cover


def _fsync(fd: int) -> None:
    os.fsync(fd)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class _Handle:
    """An open file descriptor whose blocking calls run on threads; closing waits for calls in flight.

    A call keeps running when the task awaiting it is cancelled, so the descriptor is closed only once no
    thread uses it any more.
    """

    __slots__ = ("_inflight", "closed", "fd")

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.closed = False
        self._inflight: set[asyncio.Future[Any]] = set()

    async def call(self, function: Callable[..., T], *args: Any) -> T:
        future = asyncio.get_running_loop().run_in_executor(None, functools.partial(function, self.fd, *args))
        self._inflight.add(future)
        future.add_done_callback(self._inflight.discard)
        future.add_done_callback(_consume)
        return await asyncio.shield(future)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        try:
            os.close(self.fd)
        except OSError:
            pass  # Closing is best-effort, including after an earlier I/O failure.


class _LocalTextLineReader:
    """Strict LF reader that reports whether its final line was newline-terminated."""

    __slots__ = ("_buffered", "_decoder", "_ended", "_file", "_offset", "_path", "_position")

    def __init__(self, file: _Handle, path: str) -> None:
        self._file = file
        self._path = path
        self._decoder = StreamDecoder()
        self._offset = 0
        self._buffered = ""
        self._position = 0
        self._ended = False

    async def read_line(self, context: Context) -> Result[TextLine | None, FileError]:
        aborted = _aborted(context.abort_signal, self._path)
        if aborted is not None:
            return aborted
        if self._file.closed:
            return err(FileError("invalid", "Text line reader is closed", self._path))
        try:
            while True:
                newline = self._buffered.find("\n", self._position)
                if newline != -1:
                    text = self._buffered[self._position : newline]
                    self._position = newline + 1
                    line: TextLine = {"text": text, "terminated": True}
                    return ok(line)
                if self._ended:
                    if self._position >= len(self._buffered):
                        self._buffered = ""
                        self._position = 0
                        return ok(None)
                    text = self._buffered[self._position :]
                    self._buffered = ""
                    self._position = 0
                    last: TextLine = {"text": text, "terminated": False}
                    return ok(last)
                # Explicit positions allow an aborted read to be retried without skipping bytes.
                data = await self._file.call(_pread, _LINE_CHUNK, self._offset)
                after_read = _aborted(context.abort_signal, self._path)
                if after_read is not None:
                    return after_read
                self._offset += len(data)
                rest = self._buffered[self._position :]
                if len(data) == 0:
                    self._buffered = rest + self._decoder.decode()
                    self._ended = True
                else:
                    self._buffered = rest + self._decoder.decode(data)
                self._position = 0
        except Exception as error:
            return err(to_file_error(error, self._path))

    async def close(self, context: Context) -> None:
        if self._file.closed:
            return
        self._buffered = ""
        await self._file.close()


def _read_range(fd: int, offset: int, length: int, signal: AbortSignal | None) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < length:
        chunk = _pread(fd, min(length - total, BINARY_READ_CHUNK), offset + total)
        _check(signal)
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        total += len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _scan_lines(fd: int, scanner: LineScanner, signal: AbortSignal | None) -> LineScan:
    position = 0
    while True:
        chunk = _pread(fd, _LINE_CHUNK, position)
        _check(signal)
        if len(chunk) == 0:
            return scanner.finish()
        scanner.push(chunk)
        position += len(chunk)


class _LocalBinaryReader:
    __slots__ = ("_file", "_path")

    def __init__(self, file: _Handle, path: str) -> None:
        self._file = file
        self._path = path

    async def info(self, context: Context) -> Result[FileInfo, FileError]:
        aborted = _aborted(context.abort_signal, self._path)
        if aborted is not None:
            return aborted
        if self._file.closed:
            return _closed_result("Binary reader", self._path)
        try:
            return _file_info(self._path, await self._file.call(os.fstat))
        except Exception as error:
            return err(to_file_error(error, self._path))

    async def read(self, offset: int, length: int, context: Context) -> Result[bytes, FileError]:
        signal = context.abort_signal
        aborted = _aborted(signal, self._path)
        if aborted is not None:
            return aborted
        if self._file.closed:
            return _closed_result("Binary reader", self._path)
        if not _is_safe_integer(offset) or offset < 0 or not _is_safe_integer(length) or length < 0:
            return err(FileError("invalid", "Offset and length must be non-negative safe integers", self._path))
        try:
            return ok(await self._file.call(_read_range, offset, length, signal))
        except _Aborted:
            return err(FileError("aborted", "aborted", self._path))
        except Exception as error:
            return err(to_file_error(error, self._path))

    async def scan_lines(self, options: ScanLinesOptions, context: Context) -> Result[LineScan, FileError]:
        signal = context.abort_signal
        aborted = _aborted(signal, self._path)
        if aborted is not None:
            return aborted
        if self._file.closed:
            return _closed_result("Binary reader", self._path)
        try:
            scanner = LineScanner(options["startLine"], options.get("endLine"))
        except Exception:
            return err(FileError("invalid", "Invalid line range", self._path))
        try:
            return ok(await self._file.call(_scan_lines, scanner, signal))
        except _Aborted:
            return err(FileError("aborted", "aborted", self._path))
        except Exception as error:
            return err(to_file_error(error, self._path))

    async def close(self, context: Context) -> None:
        await self._file.close()


_DirIterator = Any  # os.scandir's iterator (its type is not public)


def _read_dir_page(
    iterator: _DirIterator, path: str, max_entries: int, signal: AbortSignal | None
) -> tuple[list[FileInfo], bool]:
    entries: list[FileInfo] = []
    while len(entries) < max_entries:
        entry = next(iterator, None)
        if entry is None:
            return entries, True
        _check(signal)
        entry_path = os.path.join(path, entry.name)
        try:
            st = os.lstat(entry_path)
        except FileNotFoundError:
            continue  # Removed between enumeration and lstat: not part of the listing any more.
        except OSError as error:
            raise _EntryError(error, entry_path) from error
        info = _file_info(entry_path, st)
        if info["ok"] is True:
            entries.append(info["value"])
    return entries, False


class _EntryError(Exception):
    def __init__(self, error: OSError, path: str) -> None:
        super().__init__(str(error))
        self.error = error
        self.path = path


class _LocalDirReader:
    __slots__ = ("_closed", "_done", "_inflight", "_iterator", "_path")

    def __init__(self, iterator: _DirIterator, path: str) -> None:
        self._iterator = iterator
        self._path = path
        self._done = False
        self._closed = False
        self._inflight: asyncio.Future[Any] | None = None

    async def next(self, max_entries: int, context: Context) -> Result[DirPage, FileError]:
        signal = context.abort_signal
        aborted = _aborted(signal, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return _closed_result("Directory reader", self._path)
        if not _is_safe_integer(max_entries) or max_entries <= 0:
            return err(FileError("invalid", "maxEntries must be a positive safe integer", self._path))
        if self._done:
            end: DirPage = {"entries": [], "done": True}
            return ok(end)
        future = asyncio.get_running_loop().run_in_executor(
            None, _read_dir_page, self._iterator, self._path, max_entries, signal
        )
        future.add_done_callback(_consume)
        self._inflight = future
        try:
            entries, done = await asyncio.shield(future)
        except _Aborted:
            return err(FileError("aborted", "aborted", self._path))
        except _EntryError as error:
            return err(to_file_error(error.error, error.path))
        except Exception as error:
            return err(to_file_error(error, self._path))
        self._done = done
        page: DirPage = {"entries": entries, "done": done}
        return ok(page)

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inflight is not None and not self._inflight.done():
            await asyncio.gather(self._inflight, return_exceptions=True)
        try:
            self._iterator.close()
        except Exception:
            pass


class _ShellConfig(TypedDict):
    shell: str
    args: list[str]
    commandTransport: Literal["argv", "stdin"]


def _path_exists(path: str) -> bool:
    return os.access(path, os.F_OK)


_LEGACY_WSL_BASH = re.compile(r"[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe")


def _is_legacy_wsl_bash_path(path: str) -> bool:
    return _LEGACY_WSL_BASH.fullmatch(path.replace("/", "\\").lower()) is not None


def _bash_shell_config(shell: str) -> _ShellConfig:
    if _is_legacy_wsl_bash_path(shell):
        return {"shell": shell, "args": ["-s"], "commandTransport": "stdin"}
    return {"shell": shell, "args": ["-c"], "commandTransport": "argv"}


def _find_bash_on_path() -> str | None:
    found = shutil.which("bash.exe" if _is_windows() else "bash")
    return found if found and _path_exists(found) else None


def _shell_config(custom_shell_path: str | None) -> Result[_ShellConfig, ExecutionError]:
    if custom_shell_path:
        if _path_exists(custom_shell_path):
            return ok(_bash_shell_config(custom_shell_path))
        return err(ExecutionError("shell_unavailable", f"Custom shell path not found: {custom_shell_path}"))
    if _is_windows():
        candidates: list[str] = []
        program_files = os.environ.get("ProgramFiles")  # noqa: SIM112 - upstream spelling
        if program_files:
            candidates.append(f"{program_files}\\Git\\bin\\bash.exe")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")  # noqa: SIM112
        if program_files_x86:
            candidates.append(f"{program_files_x86}\\Git\\bin\\bash.exe")
        for candidate in candidates:
            if _path_exists(candidate):
                return ok(_bash_shell_config(candidate))
        on_path = _find_bash_on_path()
        if on_path:
            return ok(_bash_shell_config(on_path))
        searched = "\n".join(f"  {path}" for path in candidates)
        return err(
            ExecutionError(
                "shell_unavailable",
                "No bash shell found. Options:\n"
                "  1. Install Git for Windows: https://git-scm.com/download/win\n"
                "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
                "  3. Configure an explicit shellPath\n\n"
                f"Searched Git Bash in:\n{searched}",
            )
        )
    if _path_exists("/bin/bash"):
        return ok(_bash_shell_config("/bin/bash"))
    on_path = _find_bash_on_path()
    if on_path:
        return ok(_bash_shell_config(on_path))
    fallback: _ShellConfig = {"shell": "sh", "args": ["-c"], "commandTransport": "argv"}
    return ok(fallback)


def _shell_env(
    base_env: Mapping[str, str] | None, extra_env: Mapping[str, str] | None, inherit_env: bool | None
) -> dict[str, str]:
    if inherit_env is False:
        return dict(extra_env or {})
    return {**os.environ, **(base_env or {}), **(extra_env or {})}


def _kill_process_tree(pid: int) -> None:
    if _is_windows():
        try:
            import subprocess

            system_root = os.environ.get("SystemRoot") or "C:\\Windows"  # noqa: SIM112
            subprocess.Popen(
                [os.path.join(system_root, "System32", "taskkill.exe"), "/F", "/T", "/PID", str(pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass  # Ignore errors, including a taskkill that cannot be spawned.
        return
    try:
        os.killpg(pid, signal_module.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal_module.SIGKILL)
        except Exception:
            pass  # Process already dead.


def _open_spill(path: str) -> BinaryIO:
    return open(path, "ab")


def _write_spill(file: BinaryIO, data: bytes) -> None:
    file.write(data)
    file.flush()


def _close_spill(file: BinaryIO) -> None:
    file.close()


class _Protocol(asyncio.SubprocessProtocol):
    __slots__ = ("_execution",)

    def __init__(self, execution: _Execution) -> None:
        self._execution = execution

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._execution.transport = cast(asyncio.SubprocessTransport, transport)

    def pipe_data_received(self, fd: int, data: bytes | bytearray) -> None:
        self._execution.on_data(fd, bytes(data))

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self._execution.on_pipe_closed(fd)

    def process_exited(self) -> None:
        self._execution.on_exit()

    def connection_lost(self, exc: Exception | None) -> None:
        self._execution.on_close()


class _Execution:
    """One running command: output delivery, spill, timeout, abort and settlement (upstream's ``exec`` closure)."""

    def __init__(
        self, env: LocalExecutionEnv, options: ShellExecOptions | None, context: Context, timeout_ms: float | None
    ) -> None:
        self.env = env
        self.options: ShellExecOptions = options or {}
        self.context = context
        self.signal = context.abort_signal
        self.timeout_ms = timeout_ms
        self.loop = asyncio.get_running_loop()
        self.transport: asyncio.SubprocessTransport | None = None
        self.pid: int | None = None
        self.settled = False
        self.timed_out = False
        self.callback_error: ExecutionError | None = None
        self.spill_error: ExecutionError | None = None
        self.timeout_handle: asyncio.TimerHandle | None = None
        self.remove_abort_listener: Callable[[], None] | None = None
        # One decoder per stream, so a character split across chunks of one stream survives interleaving.
        self.decoders = {1: StreamDecoder(), 2: StreamDecoder()}
        # Output seen before the spill starts: counted against the thresholds and kept for the spill's prefix.
        self.spill_prefix: list[bytes] = []
        self.seen_bytes = 0
        self.seen_newlines = 0
        self.spill_path: str | None = None
        self.spill_queue: list[bytes] = []
        self.spill_start: asyncio.Task[None] | None = None
        self.spill_file: BinaryIO | None = None
        self.spill_pending: deque[bytes] = deque()
        self.spill_pending_bytes = 0
        self.spill_writer: asyncio.Task[None] | None = None
        self.spill_backpressured = False
        self.output_paused = False
        # waitForChildProcess state.
        self.done: asyncio.Future[None] = self.loop.create_future()
        self.finalized = False
        self.exited = False
        self.stdout_ended = False
        self.stderr_ended = False
        self.idle_timer: asyncio.TimerHandle | None = None

    # ─── lifecycle ───────────────────────────────────────────────────────────

    async def run(
        self, program: str, args: Sequence[str], cwd: str, env: Mapping[str, str], stdin_data: str | bytes | None
    ) -> Result[ShellExecResult, ExecutionError]:
        import subprocess

        try:
            await self.loop.subprocess_exec(
                lambda: _Protocol(self),
                program,
                *args,
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL if stdin_data is None else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=not _is_windows(),
            )
        except Exception as error:
            if self.transport is not None:
                self.transport.close()
            return err(ExecutionError("spawn_error", str(error), error))
        transport = self.transport
        assert transport is not None
        self.pid = transport.get_pid()
        self.env._active_child_pids.add(self.pid)
        if stdin_data is not None:
            stdin = transport.get_pipe_transport(0)
            if stdin is not None:
                try:
                    data = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
                    cast(asyncio.WriteTransport, stdin).write(data)
                    stdin.close()
                except Exception:
                    pass

        if self.timeout_ms is not None:
            self.timeout_handle = self.loop.call_later(self.timeout_ms / 1000, self.on_timeout)
        if self.signal is not None:
            self.remove_abort_listener = self.signal.add_listener(self.on_abort)
        try:
            await asyncio.shield(self.done)
        except asyncio.CancelledError:
            # The awaiting task was cancelled: kill the command and release everything before re-raising.
            self.kill()
            self.release()
            raise
        await self.finish_spill()
        self.emit(self.decoders[1].decode(), "stdout")
        self.emit(self.decoders[2].decode(), "stderr")
        if self.callback_error is not None:
            return self.settle(err(self.callback_error))
        interrupted: ExecutionError | None = None
        if self.timed_out:
            interrupted = ExecutionError("timeout", f"timeout:{_js_number(self.options.get('timeout', 0))}")
        elif self.signal is not None and self.signal.aborted:
            interrupted = ExecutionError("aborted", "aborted")
        if interrupted is not None:
            if self.spill_path is not None:
                interrupted.spill_path = self.spill_path
            return self.settle(err(interrupted))
        if self.spill_error is not None:
            return self.settle(err(self.spill_error))
        # A process killed by a signal (e.g. OOM killer) has no exit code; map it to the conventional
        # 128 + signal number so callers do not mistake it for a successful exit.
        code = transport.get_returncode()
        exit_code = 1 if code is None else (128 - code if code < 0 else code)
        result: ShellExecResult = {"exitCode": exit_code}
        if self.spill_path is not None:
            result["spillPath"] = self.spill_path
        return self.settle(ok(result))

    def settle(self, result: Result[ShellExecResult, ExecutionError]) -> Result[ShellExecResult, ExecutionError]:
        self.release()
        return result

    def release(self) -> None:
        if self.settled:
            return
        self.settled = True
        if self.timeout_handle is not None:
            self.timeout_handle.cancel()
        if self.remove_abort_listener is not None:
            self.remove_abort_listener()
        if self.pid is not None:
            self.env._active_child_pids.discard(self.pid)
        self.finalize()

    def kill(self) -> None:
        if self.pid is not None and not self.settled:
            _kill_process_tree(self.pid)

    def on_abort(self) -> None:
        # Abort may come from any thread.
        try:
            self.loop.call_soon_threadsafe(self.kill)
        except RuntimeError:
            pass  # The loop is closed.

    def on_timeout(self) -> None:
        self.timed_out = True
        self.kill()

    # ─── output ──────────────────────────────────────────────────────────────

    def emit(self, text: str, stream: Literal["stdout", "stderr"]) -> None:
        # No output reaches the caller after exec() settled, for example from a descendant holding stdio open.
        on_output = self.options.get("onOutput")
        if self.settled or text == "" or on_output is None or self.callback_error is not None:
            return
        info: ShellOutputInfo = {"stream": stream}
        try:
            on_output(text, self.context, info)
        except Exception as error:
            self.fail_callback(error)

    def fail_callback(self, error: BaseException) -> None:
        if self.callback_error is not None:
            return
        cause = to_error(error)
        self.callback_error = ExecutionError("callback_error", str(cause), cause)
        self.kill()

    def on_data(self, fd: int, chunk: bytes) -> None:
        if fd not in (1, 2) or self.finalized:
            return
        self.emit(self.decoders[fd].decode(chunk), "stdout" if fd == 1 else "stderr")
        spill = self.options.get("spill")
        if spill is not None and chunk:
            if self.spill_start is not None:
                self.start_spill(chunk)
            else:
                self.seen_bytes += len(chunk)
                self.seen_newlines += chunk.count(b"\n")
                lines = self.seen_newlines + (0 if chunk.endswith(b"\n") else 1)
                if self.seen_bytes <= spill["afterBytes"] and lines <= spill["afterLines"]:
                    self.spill_prefix.append(chunk)
                else:
                    for prefix in self.spill_prefix:
                        self.start_spill(prefix)
                    self.spill_prefix.clear()
                    self.start_spill(chunk)
        if self.exited and not self.finalized:
            self.arm_idle_timer()

    def pause_output(self) -> None:
        self.output_paused = True
        for fd in (1, 2):
            pipe = None if self.transport is None else self.transport.get_pipe_transport(fd)
            if pipe is not None:
                cast(asyncio.ReadTransport, pipe).pause_reading()

    def resume_output(self) -> None:
        if (
            self.callback_error is not None
            or self.spill_error is not None
            or self.timed_out
            or (self.signal is not None and self.signal.aborted)
            or self.spill_backpressured
            or self.finalized
        ):
            return
        self.output_paused = False
        for fd in (1, 2):
            pipe = None if self.transport is None else self.transport.get_pipe_transport(fd)
            if pipe is not None:
                cast(asyncio.ReadTransport, pipe).resume_reading()
        # Output held back while paused arrives now: it gets a full post-exit grace period.
        if self.exited and not self.finalized:
            self.arm_idle_timer()

    # ─── spill ───────────────────────────────────────────────────────────────

    def fail_spill(self, error: BaseException) -> None:
        if self.spill_error is not None:
            return
        cause = to_error(error)
        self.spill_error = ExecutionError("unknown", f"Failed to preserve complete shell output: {cause}", cause)
        self.spill_backpressured = False
        self.kill()

    def write_spill(self, chunk: bytes) -> None:
        if self.spill_file is None or len(chunk) == 0 or self.spill_error is not None:
            return
        self.spill_pending.append(chunk)
        self.spill_pending_bytes += len(chunk)
        if self.spill_writer is None or self.spill_writer.done():
            self.spill_writer = self.loop.create_task(self.drain_spill())
        if self.spill_pending_bytes >= SPILL_HIGH_WATER_MARK and not self.spill_backpressured:
            self.spill_backpressured = True
            self.pause_output()

    async def drain_spill(self) -> None:
        file = self.spill_file
        assert file is not None
        try:
            while self.spill_pending and self.spill_error is None:
                data = b"".join(self.spill_pending)
                self.spill_pending.clear()
                await _blocking(_write_spill, file, data)
                self.spill_pending_bytes -= len(data)
        except Exception as error:
            self.fail_spill(error)
            return
        if self.spill_backpressured:
            self.spill_backpressured = False
            self.resume_output()

    def start_spill(self, chunk: bytes) -> None:
        if self.spill_file is not None:
            self.write_spill(chunk)
            return
        self.spill_queue.append(chunk)
        if self.spill_start is not None:
            return
        self.pause_output()
        self.spill_start = self.loop.create_task(self.open_spill())

    async def open_spill(self) -> None:
        try:
            created = await self.env.create_temp_file({"prefix": "pi-output-", "suffix": ".log"}, self.context)
            if created["ok"] is False:
                raise created["error"]
            self.spill_path = created["value"]
            self.spill_file = await _blocking(_open_spill, created["value"])
            for queued in self.spill_queue:
                self.write_spill(queued)
            self.spill_queue.clear()
        except Exception as error:
            self.fail_spill(error)
        finally:
            self.resume_output()

    def spill_is_draining(self) -> bool:
        return (
            self.spill_error is None
            and self.spill_start is not None
            and (self.spill_file is None or self.spill_backpressured)
        )

    async def finish_spill(self) -> None:
        if self.spill_start is not None:
            await self.spill_start
        file = self.spill_file
        if file is None:
            return
        while self.spill_writer is not None and not self.spill_writer.done():
            await self.spill_writer
        try:
            await _blocking(_close_spill, file)
        except Exception as error:
            if self.spill_error is None:
                self.fail_spill(error)

    # ─── waiting for the process (upstream ``waitForChildProcess``) ───────────

    def on_exit(self) -> None:
        self.exited = True
        self.maybe_finalize_after_exit()
        if not self.finalized:
            self.arm_idle_timer()

    def on_pipe_closed(self, fd: int) -> None:
        if fd == 1:
            self.stdout_ended = True
        elif fd == 2:
            self.stderr_ended = True
        self.maybe_finalize_after_exit()

    def on_close(self) -> None:
        self.finalize()
        if self.transport is not None:
            self.transport.close()

    def maybe_finalize_after_exit(self) -> None:
        if self.exited and self.stdout_ended and self.stderr_ended:
            self.finalize()

    def arm_idle_timer(self) -> None:
        if self.idle_timer is not None:
            self.idle_timer.cancel()
        self.idle_timer = self.loop.call_later(EXIT_STDIO_GRACE_MS / 1000, self.on_idle)

    def on_idle(self) -> None:
        self.idle_timer = None
        if self.spill_is_draining():
            self.arm_idle_timer()
        else:
            self.finalize()

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None
        transport = self.transport
        if transport is not None:
            for fd in (1, 2):
                pipe = transport.get_pipe_transport(fd)
                if pipe is not None:
                    pipe.close()
            if transport.get_returncode() is not None:
                transport.close()
        if not self.done.done():
            self.done.set_result(None)


class LocalExecutionEnv:
    """:class:`~pi_durable.env.ExecutionEnv` on the host OS (upstream ``NodeExecutionEnv``).

    ``cwd`` resolves relative paths; ``shell_path`` overrides the bash lookup; ``shell_env`` is merged over
    the process environment for every command; ``watch`` configures :meth:`watch`.
    """

    def __init__(
        self,
        *,
        cwd: str,
        shell_path: str | None = None,
        shell_env: Mapping[str, str] | None = None,
        watch: LocalWatchOptions | None = None,
    ) -> None:
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._watch_options: LocalWatchOptions = watch or {}
        self._active_child_pids: set[int] = set()

    @property
    def id(self) -> str:
        """Every local environment sees the same files."""
        return "local"

    async def watch(
        self, targets: Sequence[WatchTarget], on_change: Callable[[WatchChange], None], context: Context
    ) -> Result[FileWatcher, FileError]:
        from .local_watch import LocalFileWatcher

        aborted = _aborted(context.abort_signal)
        if aborted is not None:
            return aborted
        try:
            watcher = await LocalFileWatcher.open(
                targets, lambda path: resolve_path(self.cwd, path), on_change, self._watch_options
            )
            after_open = _aborted(context.abort_signal)
            if after_open is not None:
                await watcher.close(context)
                return after_open
            return ok(watcher)
        except Exception as error:
            return err(to_file_error(error))

    async def absolute_path(self, path: str, context: Context) -> Result[str, FileError]:
        return ok(resolve_path(self.cwd, path))

    async def join_path(self, parts: Sequence[str], context: Context) -> Result[str, FileError]:
        return ok(_join_path(parts))

    async def exec(
        self, command: str | Sequence[str], options: ShellExecOptions | None, context: Context
    ) -> Result[ShellExecResult, ExecutionError]:
        signal = context.abort_signal
        if signal is not None and signal.aborted:
            return err(ExecutionError("aborted", "aborted"))
        timeout = _resolve_timeout_ms(None if options is None else options.get("timeout"))
        if timeout["ok"] is False:
            return timeout
        cwd_option = None if options is None else options.get("cwd")
        cwd = resolve_path(self.cwd, cwd_option) if cwd_option else self.cwd
        # A string runs through the shell; an argv sequence runs its program directly, without shell parsing.
        stdin_input = None if options is None else options.get("stdin")
        stdin_data: str | bytes | None = stdin_input
        if isinstance(command, str):
            config = await _blocking(_shell_config, self._shell_path)
            if config["ok"] is False:
                return config
            shell = config["value"]
            program = shell["shell"]
            if shell["commandTransport"] == "stdin":
                if stdin_input is not None:
                    message = f"{program} reads the command from standard input, so it cannot take other input"
                    return err(ExecutionError("spawn_error", message))
                args: list[str] = list(shell["args"])
                stdin_data = command
            else:
                args = [*shell["args"], command]
        else:
            if len(command) == 0:
                return err(ExecutionError("spawn_error", "Empty argv: no program to run"))
            program, *args = command
        if not await _blocking(_path_exists, cwd):
            cause = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cwd)
            return err(
                ExecutionError(
                    "spawn_error", f"Working directory does not exist: {cwd}\nCannot execute bash commands.", cause
                )
            )
        env = _shell_env(
            self._shell_env,
            None if options is None else options.get("env"),
            None if options is None else options.get("inheritEnv"),
        )
        execution = _Execution(self, options, context, timeout["value"])
        return await execution.run(program, args, cwd, env, stdin_data)

    async def open_text_line_reader(self, path: str, context: Context) -> Result[TextLineReader, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            fd = await _blocking(os.open, resolved, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except Exception as error:
            return err(to_file_error(error, resolved))
        handle = _Handle(fd)
        after_open = _aborted(context.abort_signal, resolved)
        if after_open is not None:
            await handle.close()
            return after_open
        return ok(_LocalTextLineReader(handle, resolved))

    async def read_text_file(self, path: str, context: Context) -> Result[str, FileError]:
        result = await self.read_binary_file(path, context)
        if result["ok"] is False:
            return result
        return ok(result["value"].decode("utf-8", errors="replace"))

    async def read_text_lines(
        self, path: str, options: ReadTextLinesOptions | None, context: Context
    ) -> Result[list[str], FileError]:
        max_lines = None if options is None else options.get("maxLines")
        if max_lines is not None and max_lines <= 0:
            return ok([])
        opened = await self.open_text_line_reader(path, context)
        if opened["ok"] is False:
            return opened
        reader = opened["value"]
        lines: list[str] = []
        try:
            while max_lines is None or len(lines) < max_lines:
                line = await reader.read_line(context)
                if line["ok"] is False:
                    return line
                if line["value"] is None:
                    break
                lines.append(line["value"]["text"])
            return ok(lines)
        finally:
            await reader.close(context)

    async def read_binary_file(self, path: str, context: Context) -> Result[bytes, FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            data = await _blocking(_read_file, resolved)
        except Exception as error:
            return err(to_file_error(error, resolved))
        return _aborted(signal, resolved) or ok(data)

    async def open_binary_reader(
        self, path: str, options: OpenBinaryReaderOptions | None, context: Context
    ) -> Result[BinaryReader, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        no_follow = options is not None and options.get("noFollow") is True
        try:
            fd, st = await _blocking(_open_binary, resolved, no_follow)
        except _SymlinkRefused as error:
            return err(_symlink_refused(resolved, error.__cause__))
        except Exception as error:
            return err(to_file_error(error, resolved))
        handle = _Handle(fd)
        if not stat_module.S_ISREG(st.st_mode):
            await handle.close()
            if stat_module.S_ISDIR(st.st_mode):
                return err(FileError("is_directory", "EISDIR: illegal operation on a directory, read", resolved))
            return err(FileError("invalid", "Not a regular file", resolved))
        after_open = _aborted(context.abort_signal, resolved)
        if after_open is not None:
            await handle.close()
            return after_open
        return ok(_LocalBinaryReader(handle, resolved))

    async def write_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            await _blocking(_write_file, resolved, _bytes(content), "wb", signal)
        except _Aborted:
            return err(FileError("aborted", "aborted", resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))
        return ok(None)

    async def append_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            await _blocking(_write_file, resolved, _bytes(content), "ab", signal)
        except _Aborted:
            return err(FileError("aborted", "aborted", resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))
        return _aborted(signal, resolved) or ok(None)

    async def truncate_file(self, path: str, size: int, context: Context) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        if not _is_safe_integer(size) or size < 0:
            return err(FileError("invalid", "File size must be a non-negative safe integer", resolved))
        try:
            await _blocking(_truncate, resolved, size)
        except Exception as error:
            return err(to_file_error(error, resolved))
        return _aborted(signal, resolved) or ok(None)

    async def flush_file(self, path: str, context: Context) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            directory = await _blocking(_flush, resolved)
        except Exception as error:
            return err(to_file_error(error, resolved))
        if directory:
            return err(FileError("is_directory", "Is a directory", resolved))
        return _aborted(signal, resolved) or ok(None)

    async def rename_file(self, source_path: str, destination_path: str, context: Context) -> Result[None, FileError]:
        source = resolve_path(self.cwd, source_path)
        destination = resolve_path(self.cwd, destination_path)
        aborted = _aborted(context.abort_signal, destination)
        if aborted is not None:
            return aborted
        try:
            await _blocking(os.replace, source, destination)
        except Exception as error:
            return err(to_file_error(error, source))
        return ok(None)

    async def file_info(self, path: str, context: Context) -> Result[FileInfo, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return _file_info(resolved, await _blocking(os.lstat, resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def list_dir(self, path: str, context: Context) -> Result[list[FileInfo], FileError]:
        resolved = resolve_path(self.cwd, path)
        signal = context.abort_signal
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return ok(await _blocking(_list_dir, resolved, signal))
        except _Aborted:
            return err(FileError("aborted", "aborted", resolved))
        except _EntryError as error:
            return err(to_file_error(error.error, error.path))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def list_files(
        self, path: str, options: ListFilesOptions | None, context: Context
    ) -> Result[FileListing, FileError]:
        resolved = resolve_path(self.cwd, path)
        return err(FileError("not_supported", "LocalExecutionEnv does not list files", resolved))

    async def open_dir_reader(self, path: str, context: Context) -> Result[DirReader, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            iterator = await _blocking(os.scandir, resolved)
        except Exception as error:
            return err(to_file_error(error, resolved))
        after_open = _aborted(context.abort_signal, resolved)
        if after_open is not None:
            iterator.close()
            return after_open
        return ok(_LocalDirReader(iterator, resolved))

    async def canonical_path(self, path: str, context: Context) -> Result[str, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return ok(await _blocking(_realpath, resolved))
        except Exception as error:
            return err(to_file_error(error, resolved))

    async def exists(self, path: str, context: Context) -> Result[bool, FileError]:
        result = await self.file_info(path, context)
        if result["ok"] is True:
            return ok(True)
        if result["error"].code == "not_found":
            return ok(False)
        return err(result["error"])

    async def create_dir(
        self, path: str, options: CreateDirOptions | None, context: Context
    ) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        recursive = True if options is None else options.get("recursive", True)
        try:
            await _blocking(_make_dir, resolved, recursive)
        except Exception as error:
            return err(to_file_error(error, resolved))
        return ok(None)

    async def remove(self, path: str, options: RemoveOptions | None, context: Context) -> Result[None, FileError]:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted(context.abort_signal, resolved)
        if aborted is not None:
            return aborted
        recursive = False if options is None else options.get("recursive", False)
        force = False if options is None else options.get("force", False)
        try:
            await _blocking(_remove, resolved, recursive, force)
        except Exception as error:
            return err(to_file_error(error, resolved))
        return ok(None)

    async def create_temp_dir(self, prefix: str | None, context: Context) -> Result[str, FileError]:
        aborted = _aborted(context.abort_signal)
        if aborted is not None:
            return aborted
        try:
            return ok(await _blocking(tempfile.mkdtemp, "", "tmp-" if prefix is None else prefix))
        except Exception as error:
            return err(to_file_error(error))

    async def create_temp_file(self, options: CreateTempFileOptions | None, context: Context) -> Result[str, FileError]:
        directory = await self.create_temp_dir("tmp-", context)
        if directory["ok"] is False:
            return directory
        prefix = "" if options is None else options.get("prefix", "")
        suffix = "" if options is None else options.get("suffix", "")
        file_path = os.path.join(directory["value"], f"{prefix}{uuid.uuid4()}{suffix}")
        try:
            await _blocking(_write_file, file_path, b"", "wb", None)
        except Exception as error:
            return err(to_file_error(error, file_path))
        return ok(file_path)

    async def cleanup(self, context: Context) -> None:
        for pid in list(self._active_child_pids):
            _kill_process_tree(pid)
        self._active_child_pids.clear()


# ─── blocking helpers (run on executor threads) ─────────────────────────────


def _bytes(content: str | bytes) -> bytes:
    return content.encode("utf-8", errors="surrogatepass") if isinstance(content, str) else bytes(content)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()


def _write_file(path: str, data: bytes, mode: str, signal: AbortSignal | None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _check(signal)
    with open(path, mode) as file:
        file.write(data)


def _truncate(path: str, size: int) -> None:
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


def _flush(path: str) -> bool:
    """Sync a file's contents and the metadata needed to retrieve them; ``True`` when ``path`` is a directory."""
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        # POSIX refuses to open a directory for writing; Windows opens it, so check explicitly.
        if stat_module.S_ISDIR(os.fstat(fd).st_mode):
            return True
        _fsync(fd)
        return False
    finally:
        os.close(fd)


class _SymlinkRefused(Exception):
    pass


def _open_binary(path: str, no_follow: bool) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if _is_windows():
        # Windows has no O_NOFOLLOW; check the final component before opening (not race-free).
        if no_follow and stat_module.S_ISLNK(os.lstat(path).st_mode):
            raise _SymlinkRefused
    else:
        # Nonblocking, so opening a FIFO does not wait for a writer.
        flags |= os.O_NONBLOCK
        if no_follow:
            flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        # O_NOFOLLOW reports a final-component symlink as ELOOP (EMLINK on some BSDs).
        if no_follow and error.errno in (errno.ELOOP, errno.EMLINK):
            raise _SymlinkRefused from error
        raise
    try:
        return fd, os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise


def _list_dir(path: str, signal: AbortSignal | None) -> list[FileInfo]:
    infos: list[FileInfo] = []
    with os.scandir(path) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        _check(signal)
        entry_path = os.path.join(path, name)
        try:
            st = os.lstat(entry_path)
        except OSError as error:
            raise _EntryError(error, entry_path) from error
        info = _file_info(entry_path, st)
        if info["ok"] is True:
            infos.append(info["value"])
    return infos


def _realpath(path: str) -> str:
    return os.path.realpath(path, strict=True)


def _make_dir(path: str, recursive: bool) -> None:
    if recursive:
        os.makedirs(path, exist_ok=True)
    else:
        os.mkdir(path)


def _remove(path: str, recursive: bool, force: bool) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if force:
            return
        raise
    if stat_module.S_ISDIR(st.st_mode):
        if not recursive:
            raise IsADirectoryError(errno.EISDIR, "Path is a directory", path)
        shutil.rmtree(path)
    else:
        os.unlink(path)
