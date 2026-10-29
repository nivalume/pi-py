"""Execution environment capabilities (port of ``env/index.ts``; re-exported by :mod:`pi_durable.env`).

Operations return a :data:`Result` instead of raising for expected failures. ``Result`` is plain data,
``{"ok": True, "value": ...}`` or ``{"ok": False, "error": ...}``; narrow it with an identity test on
the literal, which type checkers understand (truthiness does not narrow):

.. code-block:: python

    result = await fs.read_text_file(path, context)
    if result["ok"] is False:
        return result
    text = result["value"]

Data shapes (``FileInfo``, ``LineScan``, option objects) are ``TypedDict`` s with upstream's keys;
the capabilities themselves are ``Protocol`` s with snake_case methods. Binary data is ``bytes``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, NotRequired, Protocol, TypeAlias, TypedDict, TypeVar

from pi_chord.context import Context

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

__all__ = [
    "Result",
    "Ok",
    "Err",
    "ok",
    "err",
    "get_or_throw",
    "get_or_none",
    "to_error",
    "FileKind",
    "FileErrorCode",
    "FileError",
    "ExecutionErrorCode",
    "ExecutionError",
    "FileInfo",
    "TextLine",
    "TextLineReader",
    "ScanLinesOptions",
    "BinaryReader",
    "WatchExclude",
    "WatchTarget",
    "WatchChange",
    "WatchChangePaths",
    "WatchChangeOverflow",
    "WatchChangeError",
    "FileWatcher",
    "LineScan",
    "DirPage",
    "DirReader",
    "ListFilesOptions",
    "FileListEntry",
    "FileListing",
    "ReadTextLinesOptions",
    "OpenBinaryReaderOptions",
    "CreateDirOptions",
    "RemoveOptions",
    "CreateTempFileOptions",
    "FileSystem",
    "ShellSpillOptions",
    "ShellExecResult",
    "ShellOutputWindow",
    "ShellOutputSkip",
    "ShellOutputInfo",
    "ShellExecOptions",
    "Shell",
    "ExecutionEnv",
]

T = TypeVar("T")
E = TypeVar("E")


class Ok(TypedDict, Generic[T]):
    ok: ReadOnly[Literal[True]]
    value: ReadOnly[T]


class Err(TypedDict, Generic[E]):
    ok: ReadOnly[Literal[False]]
    error: ReadOnly[E]


#: Result of a fallible operation. Expected failures are returned instead of raised.
Result: TypeAlias = "Ok[T] | Err[E]"


def ok(value: T) -> Ok[T]:
    return {"ok": True, "value": value}


def err(error: E) -> Err[E]:
    return {"ok": False, "error": error}


def get_or_throw(result: Result[T, Any]) -> T:
    """The value of a successful result; raises the error of a failed one (as :func:`to_error`)."""
    if result["ok"] is False:
        raise to_error(result["error"])
    return result["value"]


def get_or_none(result: Result[T, Any]) -> T | None:
    """Upstream ``getOrUndefined()``: the value of a successful result, ``None`` for a failed one."""
    return result["value"] if result["ok"] is True else None


def to_error(error: object) -> Exception:
    """Coerce a raised or returned value to an exception."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    try:
        return Exception(json.dumps(error))
    except Exception:
        return Exception(str(error))


FileKind: TypeAlias = Literal["file", "directory", "symlink"]

FileErrorCode: TypeAlias = Literal[
    "aborted", "not_found", "permission_denied", "not_directory", "is_directory", "invalid", "not_supported", "unknown"
]


class FileError(Exception):
    """An expected file system failure, returned in a ``Result``."""

    def __init__(
        self, code: FileErrorCode, message: str, path: str | None = None, cause: BaseException | None = None
    ) -> None:
        super().__init__(message)
        self.code: FileErrorCode = code
        self.path = path
        self.message = message
        if cause is not None:
            self.__cause__ = cause


ExecutionErrorCode: TypeAlias = Literal[
    "aborted", "timeout", "shell_unavailable", "spawn_error", "callback_error", "unknown"
]


class ExecutionError(Exception):
    """An expected command execution failure, returned in a ``Result``."""

    def __init__(self, code: ExecutionErrorCode, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.code: ExecutionErrorCode = code
        self.message = message
        #: Spill file of a command that timed out or was aborted after its output crossed the spill thresholds.
        self.spill_path: str | None = None
        if cause is not None:
            self.__cause__ = cause


class FileInfo(TypedDict):
    name: str
    path: str
    kind: FileKind
    size: int
    mtimeMs: float


class TextLine(TypedDict):
    text: str
    terminated: bool


class TextLineReader(Protocol):
    async def read_line(self, context: Context) -> Result[TextLine | None, FileError]: ...
    async def close(self, context: Context) -> None: ...


class ScanLinesOptions(TypedDict):
    startLine: int
    #: Absent: to the end.
    endLine: NotRequired[int]


class LineScan(TypedDict):
    """Where lines of a file are, as ``BinaryReader.scan_lines`` found them."""

    #: Newline bytes in the whole file; it has ``newlines + 1`` lines.
    newlines: int
    #: Byte range of the selected lines: from the start of the first to the end of the last, without the
    #: newline that ends it. A selection past the last line is empty at the end of the file.
    start: int
    end: int
    #: Where the first selected line ends: its newline, or the end of the file.
    firstLineEnd: int
    #: Where the last selected line starts.
    lastLineStart: int
    #: UTF-8 byte length of the decoded selection and of its first line.
    selectedBytes: int
    firstLineBytes: int


class BinaryReader(Protocol):
    """Positional reads from one opened regular file; all calls see the same file even if its path is renamed."""

    async def info(self, context: Context) -> Result[FileInfo, FileError]:
        """Metadata of the opened file, not of whatever its path names now."""
        ...

    async def read(self, offset: int, length: int, context: Context) -> Result[bytes, FileError]:
        """Up to ``length`` bytes at ``offset``; fewer only at end of file."""
        ...

    async def scan_lines(self, options: ScanLinesOptions, context: Context) -> Result[LineScan, FileError]:
        """One pass that locates lines ``[startLine, endLine)``, 0-based (see :class:`~pi_durable.env.LineScanner`)."""
        ...

    async def close(self, context: Context) -> None: ...


class WatchExclude(TypedDict, total=False):
    hidden: bool
    names: Sequence[str]


class WatchTarget(TypedDict):
    """A file or directory to watch. It may be missing; creating it is a change."""

    path: str
    #: Watch everything below a directory, not only its entries. Symbolic links below it are not followed.
    recursive: NotRequired[bool]
    #: Entries below ``path`` that are neither watched nor reported.
    exclude: NotRequired[WatchExclude]


class WatchChangePaths(TypedDict):
    """Something at or below each path may have changed (a directory path covers its subtree)."""

    paths: list[str]


class WatchChangeOverflow(TypedDict):
    """Coverage was uncertain for a while; rescan everything that is watched."""

    overflow: Literal[True]


class WatchChangeError(TypedDict):
    """The watcher stopped; no calls follow."""

    error: FileError


WatchChange: TypeAlias = WatchChangePaths | WatchChangeOverflow | WatchChangeError


class FileWatcher(Protocol):
    @property
    def mode(self) -> Literal["native", "polling"]:
        """``native``: changes reported within about two seconds; ``polling``: snapshots are compared."""
        ...

    async def close(self, context: Context) -> None:
        """Stop watching; no ``on_change`` call starts after this returns. Idempotent."""
        ...


class DirPage(TypedDict):
    entries: list[FileInfo]
    done: bool


class DirReader(Protocol):
    """Pages of one directory's entries."""

    async def next(self, max_entries: int, context: Context) -> Result[DirPage, FileError]:
        """Up to ``max_entries`` entries in file system order, continuing where the previous call stopped."""
        ...

    async def close(self, context: Context) -> None: ...


class ListFilesOptions(TypedDict, total=False):
    """What ``FileSystem.list_files`` lists (see upstream for the defaults)."""

    hidden: bool
    gitignore: bool
    exclude: Sequence[str]
    maxEntries: int
    maxBytes: int
    timeoutMs: float


class FileListEntry(TypedDict):
    #: Relative to the listed directory, with ``/`` separators on every system.
    path: str
    #: A symbolic link is listed as such and not followed.
    kind: FileKind


class FileListing(TypedDict):
    #: Depth first, parents before their entries, siblings in byte order of their names.
    entries: list[FileListEntry]
    #: A bound in ``ListFilesOptions`` ended the listing early.
    truncated: bool


class ReadTextLinesOptions(TypedDict, total=False):
    maxLines: int


class OpenBinaryReaderOptions(TypedDict, total=False):
    noFollow: bool


class CreateDirOptions(TypedDict, total=False):
    recursive: bool


class RemoveOptions(TypedDict, total=False):
    recursive: bool
    force: bool


class CreateTempFileOptions(TypedDict, total=False):
    prefix: str
    suffix: str


class FileSystem(Protocol):
    """Portable filesystem capability. Operations return failures rather than raising."""

    @property
    def id(self) -> str:
        """The file namespace: equal ids see the same files at the same paths, whatever their ``cwd``."""
        ...

    cwd: str

    async def absolute_path(self, path: str, context: Context) -> Result[str, FileError]: ...
    async def join_path(self, parts: Sequence[str], context: Context) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, context: Context) -> Result[str, FileError]: ...
    async def open_text_line_reader(self, path: str, context: Context) -> Result[TextLineReader, FileError]: ...
    async def read_text_lines(
        self, path: str, options: ReadTextLinesOptions | None, context: Context
    ) -> Result[list[str], FileError]: ...
    async def read_binary_file(self, path: str, context: Context) -> Result[bytes, FileError]: ...
    async def open_binary_reader(
        self, path: str, options: OpenBinaryReaderOptions | None, context: Context
    ) -> Result[BinaryReader, FileError]:
        """Open a regular file for bounded positional reads (``noFollow`` refuses a final symbolic link)."""
        ...

    async def write_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]: ...
    async def append_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]: ...
    async def truncate_file(self, path: str, size: int, context: Context) -> Result[None, FileError]:
        """Truncate or extend a file to exactly ``size`` bytes."""
        ...

    async def flush_file(self, path: str, context: Context) -> Result[None, FileError]:
        """Flush file contents and metadata needed to retrieve them."""
        ...

    async def rename_file(
        self, source_path: str, destination_path: str, context: Context
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, context: Context) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str, context: Context) -> Result[list[FileInfo], FileError]: ...
    async def list_files(
        self, path: str, options: ListFilesOptions | None, context: Context
    ) -> Result[FileListing, FileError]:
        """The files, directories and symbolic links below a directory, at any depth.

        Optional upstream; here every environment has the method and one that cannot list returns
        ``FileError("not_supported", ...)``, so callers fall back to their own walk.
        """
        ...

    async def open_dir_reader(self, path: str, context: Context) -> Result[DirReader, FileError]: ...
    async def watch(
        self, targets: Sequence[WatchTarget], on_change: Callable[[WatchChange], None], context: Context
    ) -> Result[FileWatcher, FileError]:
        """Report changes to files and directories; coverage is established when the watcher exists."""
        ...

    async def canonical_path(self, path: str, context: Context) -> Result[str, FileError]: ...
    async def exists(self, path: str, context: Context) -> Result[bool, FileError]: ...
    async def create_dir(
        self, path: str, options: CreateDirOptions | None, context: Context
    ) -> Result[None, FileError]: ...
    async def remove(self, path: str, options: RemoveOptions | None, context: Context) -> Result[None, FileError]: ...
    async def create_temp_dir(self, prefix: str | None, context: Context) -> Result[str, FileError]: ...
    async def create_temp_file(
        self, options: CreateTempFileOptions | None, context: Context
    ) -> Result[str, FileError]: ...
    async def cleanup(self, context: Context) -> None: ...


class ShellSpillOptions(TypedDict):
    """Spill the complete output to a temporary file once it exceeds either threshold."""

    afterBytes: int
    #: Complete or partial lines.
    afterLines: int


class ShellExecResult(TypedDict):
    exitCode: int
    #: Temporary file holding the complete raw output, when the spill thresholds were exceeded.
    spillPath: NotRequired[str]


class ShellOutputWindow(TypedDict):
    """The tail of the combined output a caller keeps, and how often it samples it."""

    #: UTF-8 bytes of decoded text kept at the end of the output.
    maxBytes: int
    #: Lines kept at the end of the output.
    maxLines: int
    #: Minimum pause between the caller's samples of the output.
    minIntervalMs: float
    #: Each sample also pauses the caller in proportion to its size at this rate.
    bytesPerSecond: float


class ShellOutputSkip(TypedDict):
    """Output an environment omitted (the shape of ``pi_chord.output.OutputSkip``)."""

    #: UTF-8 byte length of the omitted text.
    bytes: int
    #: Newlines (U+000A) in the omitted text.
    newlines: int
    #: Whether the omitted text ends with a newline.
    endsWithNewline: bool


class ShellOutputInfo(TypedDict):
    stream: Literal["stdout", "stderr"]
    #: Output omitted immediately before this chunk, only with ``window``.
    skipped: NotRequired[ShellOutputSkip]


class ShellExecOptions(TypedDict, total=False):
    cwd: str
    env: Mapping[str, str]
    inheritEnv: bool
    timeout: float
    #: Every decoded chunk of stdout and stderr as it arrives: ``onOutput(text, context, info)``.
    onOutput: Callable[[str, Context, ShellOutputInfo], None]
    spill: ShellSpillOptions
    #: The caller keeps only this tail of the output, so the environment may omit output outside it.
    window: ShellOutputWindow
    #: Written to the command's standard input, which is then closed; without it the command reads an empty input.
    stdin: str | bytes


class Shell(Protocol):
    async def exec(
        self, command: str | Sequence[str], options: ShellExecOptions | None, context: Context
    ) -> Result[ShellExecResult, ExecutionError]:
        """Run a command: a string through the environment's shell, a sequence directly without one."""
        ...

    async def cleanup(self, context: Context) -> None:
        """Kill every command this environment still runs; for its owner's shutdown."""
        ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """A file system and a shell sharing one ``cwd``."""

    async def cleanup(self, context: Context) -> None: ...
