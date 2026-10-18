"""File watching for :class:`~pi_durable.env.local.LocalExecutionEnv` (port of ``env/node-watch.ts``).

Watches by snapshots: native events only trigger a debounced rescan, and changes are the difference
between snapshots plus the event paths. A replaced file, a renamed or recreated ancestor, or a directory
created with its contents therefore never depends on which events an operating system sends. New
directories get their watchers before they are scanned again, so nothing written into them before the
watcher existed is missed.

``native`` mode uses Linux inotify through ``ctypes`` (one inotify instance per event loop, shared by
every watcher, as libuv shares one), with a watch on each existing ancestor of a target, on each target
directory and on each directory below a recursive target. Python's standard library has no portable
native watcher, so on other systems (and on file systems that do not report remote changes) watchers
poll; upstream also polls on Windows and uses FSEvents on macOS. Scans run on an executor thread.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import errno
import hashlib
import os
import stat as stat_module
import struct
import sys
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypedDict

from pi_chord.context import Context

from .types import FileError, WatchChange, WatchTarget

__all__ = ["LocalFileWatcher", "LocalWatchOptions"]


class LocalWatchOptions(TypedDict, total=False):
    """How :class:`~pi_durable.env.local.LocalExecutionEnv` watches (upstream ``NodeWatchOptions``)."""

    #: Force a mode. By default ``polling`` is chosen where no native watcher is available (anything but
    #: Linux) and for file systems that do not report remote changes.
    mode: Literal["native", "polling"]
    #: Interval between snapshots in ``polling`` mode; default 2000 ms.
    pollIntervalMs: float
    #: Most directories one watcher covers; default 10,000.
    maxDirectories: int


DEBOUNCE_MS = 50
DEFAULT_POLL_MS = 2000
DEFAULT_MAX_DIRECTORIES = 10_000
#: In polling mode, recently modified small files are also compared by content: a second write within the
#: file system's timestamp granularity can keep size and modification time.
HASH_MAX_BYTES = 256 * 1024
HASH_RECENT_MS = 5000

#: Linux ``statfs`` magic numbers of file systems that accept watches but do not report changes made elsewhere.
UNRELIABLE_FILE_SYSTEMS = frozenset(
    {
        0x6969,  # NFS
        0x517B,  # SMB
        0xFF534D42,  # CIFS
        0xFE534D42,  # SMB2
        0x65735546,  # FUSE (sshfs, Android shared storage)
        0x01021997,  # 9P (WSL2 Windows drives)
        0x0BD00BD0,  # Lustre
        0x47504653,  # GPFS
        0x00C36400,  # Ceph
        0x5346414F,  # OpenAFS
        0x6B414653,  # kAFS
        0x5DCA2DF5,  # sdcardfs
    }
)

_LINUX = sys.platform.startswith("linux")


class _BudgetExceeded(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    path: str
    recursive: bool
    hidden: bool
    names: frozenset[str]


class _Entry(NamedTuple):
    """What a snapshot remembers of one path. Directories and ancestors are compared by identity only."""

    kind: Literal["file", "directory", "symlink", "other"]
    dev: int
    ino: int
    size: int
    mtime_ms: float
    hash: str | None = None


_Snapshot = dict[str, _Entry]


class _Scan(NamedTuple):
    """A scan: the snapshot, and targets that are symbolic links to files, whose files need their own watchers."""

    snapshot: _Snapshot
    linked_files: frozenset[str]


def _is_denied(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno in (errno.EACCES, errno.EPERM)


def _entry_of(st: os.stat_result, hash: str | None = None) -> _Entry:
    mode = st.st_mode
    if stat_module.S_ISREG(mode):
        kind: Literal["file", "directory", "symlink", "other"] = "file"
    elif stat_module.S_ISDIR(mode):
        kind = "directory"
    elif stat_module.S_ISLNK(mode):
        kind = "symlink"
    else:
        kind = "other"
    if kind == "directory":
        return _Entry(kind, st.st_dev, st.st_ino, 0, 0, hash)
    return _Entry(kind, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns / 1e6, hash)


def _ancestors_of(path: str) -> list[str]:
    """Ancestors of ``path``, nearest first, up to the root."""
    result: list[str] = []
    current = os.path.dirname(path)
    while True:
        result.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            return result
        current = parent


def _is_within(path: str, ancestor: str) -> bool:
    """Whether ``path`` is ``ancestor`` or below it."""
    return path == ancestor or path.startswith(ancestor if ancestor.endswith(os.sep) else ancestor + os.sep)


def _excluded(target: _ResolvedTarget, name: str) -> bool:
    return (target.hidden and name.startswith(".")) or name in target.names


def _child(directory: str, name: str) -> str:
    return f"{directory if directory.endswith(os.sep) else directory + os.sep}{name}"


def _lstat(path: str) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError:
        return None


# ─── inotify ─────────────────────────────────────────────────────────────────

_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_NONBLOCK = 0o4000
_IN_CLOEXEC = 0o2000000
#: libuv's mask for ``fs.watch``.
_IN_MASK = (
    _IN_ATTRIB | _IN_CREATE | _IN_MODIFY | _IN_DELETE | _IN_DELETE_SELF | _IN_MOVE_SELF | _IN_MOVED_FROM | _IN_MOVED_TO
)
_EVENT = struct.Struct("iIII")

_libc: ctypes.CDLL | None = None


def _load_libc() -> ctypes.CDLL | None:
    global _libc
    if _libc is None and _LINUX:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            libc.inotify_init1.argtypes = [ctypes.c_int]
            libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
            libc.statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
            _libc = libc
        except (OSError, AttributeError):
            return None
    return _libc


def _statfs_type(path: str) -> int | None:
    libc = _load_libc()
    if libc is None:
        return None
    buffer = ctypes.create_string_buffer(512)
    if libc.statfs(os.fsencode(path), buffer) != 0:
        return None
    # `f_type` is the first field of `struct statfs`, a `long` (`__fsword_t`).
    return ctypes.c_long.from_buffer(buffer).value & 0xFFFFFFFF


def _any_unreliable(paths: Sequence[str]) -> bool:
    """Whether any path, or its nearest existing ancestor, is on a file system that does not report remote changes."""
    if not _LINUX:
        return False
    for path in paths:
        for candidate in [path, *_ancestors_of(path)]:
            kind = _statfs_type(candidate)
            if kind is None:
                continue
            if kind in UNRELIABLE_FILE_SYSTEMS:
                return True
            break
    return False


class _InotifyWatch:
    """One installed watch of a path; a hub may share a kernel watch descriptor between several."""

    __slots__ = ("_hub", "closed", "on_error", "on_event", "wd")

    def __init__(
        self, hub: _Inotify, wd: int, on_event: Callable[[str | None], None], on_error: Callable[[], None]
    ) -> None:
        self._hub = hub
        self.wd = wd
        self.on_event = on_event
        self.on_error = on_error
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._hub.remove(self)


class _Inotify:
    """The inotify instance of one event loop; watch descriptors are shared like inodes (the kernel dedupes them)."""

    __slots__ = ("_fd", "_libc", "_loop", "_watches")

    _hubs: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Inotify] = weakref.WeakKeyDictionary()

    def __init__(self, loop: asyncio.AbstractEventLoop, libc: ctypes.CDLL, fd: int) -> None:
        self._loop = loop
        self._libc = libc
        self._fd = fd
        self._watches: dict[int, list[_InotifyWatch]] = {}

    @classmethod
    def for_loop(cls) -> _Inotify:
        loop = asyncio.get_running_loop()
        hub = cls._hubs.get(loop)
        if hub is not None and hub._fd >= 0:
            return hub
        libc = _load_libc()
        if libc is None:
            raise OSError(errno.ENOSYS, "inotify is not available")
        fd = libc.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
        if fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        hub = cls(loop, libc, fd)
        loop.add_reader(fd, hub._read)
        cls._hubs[loop] = hub
        return hub

    def add(self, path: str, on_event: Callable[[str | None], None], on_error: Callable[[], None]) -> _InotifyWatch:
        wd = self._libc.inotify_add_watch(self._fd, os.fsencode(path), _IN_MASK)
        if wd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), path)
        watch = _InotifyWatch(self, wd, on_event, on_error)
        self._watches.setdefault(wd, []).append(watch)
        return watch

    def remove(self, watch: _InotifyWatch) -> None:
        watches = self._watches.get(watch.wd)
        if watches is None or watch not in watches:
            return
        watches.remove(watch)
        if not watches:
            del self._watches[watch.wd]
            self._libc.inotify_rm_watch(self._fd, watch.wd)
        if not self._watches:
            self._shutdown()

    def _shutdown(self) -> None:
        if self._fd < 0:
            return
        try:
            self._loop.remove_reader(self._fd)
        except Exception:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = -1
        if _Inotify._hubs.get(self._loop) is self:
            del _Inotify._hubs[self._loop]

    def _read(self) -> None:
        try:
            data = os.read(self._fd, 64 * 1024)
        except BlockingIOError:
            return
        except OSError:
            for watches in list(self._watches.values()):
                for watch in list(watches):
                    watch.on_error()
            return
        offset = 0
        while offset + _EVENT.size <= len(data):
            wd, mask, _cookie, length = _EVENT.unpack_from(data, offset)
            raw = data[offset + _EVENT.size : offset + _EVENT.size + length].rstrip(b"\0")
            offset += _EVENT.size + length
            if mask & _IN_Q_OVERFLOW:
                # Events were dropped: every watcher rescans.
                for watches in list(self._watches.values()):
                    for watch in list(watches):
                        watch.on_event(None)
                continue
            watches = self._watches.get(wd)
            if not watches:
                continue
            if mask & _IN_IGNORED:
                # The kernel removed the watch (its directory is gone): the watchers rescan and replace it.
                del self._watches[wd]
                for watch in list(watches):
                    watch.closed = True
                    watch.on_error()
                if not self._watches:
                    self._shutdown()
                continue
            name = os.fsdecode(raw) if raw else None
            for watch in list(watches):
                watch.on_event(name)


@dataclass(slots=True)
class _Installed:
    """A native watch and the identity of what it was installed on: a path replaced by another file needs a new one."""

    watch: _InotifyWatch
    dev: int
    ino: int


class LocalFileWatcher:
    """:class:`~pi_durable.env.FileWatcher` of :class:`~pi_durable.env.local.LocalExecutionEnv` (upstream ``NodeFileWatcher``)."""

    def __init__(
        self,
        targets: Sequence[_ResolvedTarget],
        on_change: Callable[[WatchChange], None],
        mode: Literal["native", "polling"],
        options: LocalWatchOptions,
    ) -> None:
        """Internal: use :meth:`open`, which establishes coverage first."""
        self._targets = tuple(targets)
        self._on_change = on_change
        self._mode: Literal["native", "polling"] = mode
        self._poll_interval_ms = options.get("pollIntervalMs", DEFAULT_POLL_MS)
        self._max_directories = options.get("maxDirectories", DEFAULT_MAX_DIRECTORIES)
        self._watchers: dict[str, _Installed] = {}
        self._events: set[str] = set()
        self._snapshot: _Snapshot = {}
        self._timer: asyncio.TimerHandle | None = None
        self._running: asyncio.Task[None] | None = None
        self._dirty = False
        self._closed = False
        self._loop = asyncio.get_running_loop()

    @classmethod
    async def open(
        cls,
        targets: Sequence[WatchTarget],
        resolve_path: Callable[[str], str],
        on_change: Callable[[WatchChange], None],
        options: LocalWatchOptions,
    ) -> LocalFileWatcher:
        """Establish coverage: watchers first, then the snapshot later changes are compared with."""
        resolved = [
            _ResolvedTarget(
                resolve_path(target["path"]),
                target.get("recursive") is True,
                target.get("exclude", {}).get("hidden") is True,
                frozenset(target.get("exclude", {}).get("names", ())),
            )
            for target in targets
        ]
        mode = options.get("mode")
        if mode is None:
            loop = asyncio.get_running_loop()
            unreliable = await loop.run_in_executor(None, _any_unreliable, [target.path for target in resolved])
            mode = "native" if _LINUX and _load_libc() is not None and not unreliable else "polling"
        watcher = cls(resolved, on_change, mode, options)
        try:
            await watcher._sync(False)
        except BaseException as error:
            watcher._stop()
            if isinstance(error, _BudgetExceeded):
                raise FileError("invalid", str(error)) from None
            raise
        watcher._schedule_poll()
        return watcher

    @property
    def mode(self) -> Literal["native", "polling"]:
        return self._mode

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._stop()
        running = self._running
        if running is not None:
            await asyncio.gather(running, return_exceptions=True)

    def _stop(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for installed in self._watchers.values():
            installed.watch.close()
        self._watchers.clear()

    def _deliver(self, change: WatchChange) -> None:
        if self._closed:
            return
        try:
            self._on_change(change)
        except Exception:
            pass  # A throwing callback must not stop watching.

    def _schedule_poll(self) -> None:
        if self._closed or self._mode != "polling":
            return

        def fire() -> None:
            self._timer = None
            self._flush().add_done_callback(lambda _: self._schedule_poll())

        self._timer = self._loop.call_later(self._poll_interval_ms / 1000, fire)

    def _schedule_flush(self) -> None:
        if self._closed or self._timer is not None:
            return

        def fire() -> None:
            self._timer = None
            self._flush()

        self._timer = self._loop.call_later(DEBOUNCE_MS / 1000, fire)

    def _flush(self) -> asyncio.Future[None]:
        if self._running is not None:
            self._dirty = True
            return self._running
        self._running = self._loop.create_task(self._run_flush())
        return self._running

    async def _run_flush(self) -> None:
        try:
            while True:
                self._dirty = False
                events = list(self._events)
                self._events.clear()
                changed = await self._sync(True)
                changed.update(events)
                if changed:
                    self._deliver({"paths": sorted(changed)})
                if not self._dirty or self._closed:
                    break
        except Exception as error:
            file_error = (
                error
                if isinstance(error, FileError)
                else FileError("permission_denied" if _is_denied(error) else "invalid", str(error))
            )
            self._deliver({"error": file_error})
            self._stop()
        finally:
            self._running = None

    async def _sync(self, report: bool) -> set[str]:
        """Rescan, report differences, and install watchers for new directories, rescanning until none are new."""
        changed: set[str] = set()
        for _round in range(10):
            if self._closed:
                break
            scan = await self._loop.run_in_executor(None, self._scan)
            # Closed during the scan: installing watchers now would leak them.
            if self._closed:
                break
            if report:
                changed.update(self._diff(self._snapshot, scan.snapshot))
            self._snapshot = scan.snapshot
            if self._mode == "polling" or not self._reconcile_watchers(scan):
                break
            # Something written into a new directory before its watcher existed shows up in the next round.
            report = True
        return changed

    def _diff(self, previous: _Snapshot, next: _Snapshot) -> list[str]:
        changed: list[str] = []
        for path, entry in next.items():
            before = previous.get(path)
            if before is None or before != entry:
                changed.append(self._reported(path))
        for path in previous:
            if path not in next:
                changed.append(self._reported(path))
        return changed

    def _reported(self, path: str) -> str:
        """An ancestor that changed identity moved every target below it; report those targets."""
        if any(_is_within(path, target.path) for target in self._targets):
            return path
        for target in self._targets:
            if _is_within(target.path, path):
                return target.path
        return path

    def _scan(self) -> _Scan:
        """One snapshot of every target (runs on an executor thread)."""
        snapshot: _Snapshot = {}
        linked_files: set[str] = set()
        # Directories are counted once, but traversed once per target: overlapping targets differ in recursion
        # and exclusions, and an entry recorded for one target must still be descended into for another.
        counted: set[str] = set()
        traversed: set[tuple[int, str]] = set()
        # Kinds of listed entries, not following links: a target that links to a directory is recorded by
        # `stat`, but symbolic links below a recursive target are not followed.
        listed: dict[str, str] = {}
        polling = self._mode == "polling"
        max_directories = self._max_directories

        def count_directory(path: str) -> None:
            counted.add(path)
            if len(counted) > max_directories:
                raise _BudgetExceeded(f"Watched paths exceed {max_directories} directories")

        def record(path: str, st: os.stat_result) -> None:
            hash: str | None = None
            if (
                polling
                and stat_module.S_ISREG(st.st_mode)
                and st.st_size <= HASH_MAX_BYTES
                and time.time() * 1000 - st.st_mtime_ns / 1e6 < HASH_RECENT_MS
            ):
                try:
                    with open(path, "rb") as file:
                        hash = hashlib.sha256(file.read()).hexdigest()
                except OSError:
                    hash = None
            snapshot[path] = _entry_of(st, hash)

        def scan_directory(index: int, target: _ResolvedTarget, directory: str) -> None:
            key = (index, directory)
            if key in traversed:
                return
            traversed.add(key)
            try:
                names = os.listdir(directory)
            except OSError as error:
                # The watched directory itself must be readable; below it, unreadable directories are skipped.
                if directory == target.path and _is_denied(error):
                    raise
                if error.errno in (errno.ENOENT, errno.EACCES, errno.EPERM, errno.ENOTDIR):
                    return
                raise
            for name in names:
                if _excluded(target, name):
                    continue
                path = _child(directory, name)
                kind = listed.get(path)
                if kind is None:
                    st = _lstat(path)
                    if st is None:
                        continue
                    kind = _entry_of(st).kind
                    listed[path] = kind
                    # A target's own entry (following links) wins over its listing by another target.
                    if path not in snapshot:
                        record(path, st)
                if target.recursive and kind == "directory":
                    count_directory(path)
                    scan_directory(index, target, path)

        for index, target in enumerate(self._targets):
            for ancestor in _ancestors_of(target.path):
                if ancestor in snapshot:
                    continue
                st = _lstat(ancestor)
                # Identity only: an ancestor's own timestamps change with every unrelated sibling.
                if st is not None:
                    snapshot[ancestor] = _entry_of(st)._replace(size=0, mtime_ms=0)
            # The target itself may be a symbolic link to what is watched; follow it. A missing target is watched
            # for its creation; one that cannot be reached for lack of permission fails.
            try:
                st = os.stat(target.path)
            except OSError as error:
                if _is_denied(error):
                    raise
                continue
            record(target.path, st)
            if stat_module.S_ISREG(st.st_mode):
                link = _lstat(target.path)
                if link is not None and stat_module.S_ISLNK(link.st_mode):
                    linked_files.add(target.path)
            if stat_module.S_ISDIR(st.st_mode):
                count_directory(target.path)
                scan_directory(index, target, target.path)
        return _Scan(snapshot, frozenset(linked_files))

    def _reconcile_watchers(self, scan: _Scan) -> bool:
        """Watch every existing ancestor of each target, each target directory, each target that is a symbolic link
        to a file (changes to that file are not events of the link's directory), and each directory below a recursive
        target. Returns whether a watcher was added."""
        snapshot = scan.snapshot
        wanted: dict[str, None] = dict.fromkeys(scan.linked_files)
        for target in self._targets:
            for ancestor in _ancestors_of(target.path):
                entry = snapshot.get(ancestor)
                if entry is not None and entry.kind == "directory":
                    wanted.setdefault(ancestor)
            entry = snapshot.get(target.path)
            if entry is None or entry.kind != "directory":
                continue
            wanted[target.path] = None
            if target.recursive:
                for path, child in snapshot.items():
                    if child.kind == "directory" and path != target.path and _is_within(path, target.path):
                        wanted[path] = None
        for path, installed in list(self._watchers.items()):
            entry = snapshot.get(path)
            # Gone, or replaced: a watcher follows the directory it was installed on, not the path.
            if path not in wanted or entry is None or entry.dev != installed.dev or entry.ino != installed.ino:
                installed.watch.close()
                del self._watchers[path]
        added = False
        hub: _Inotify | None = None
        for path in wanted:
            entry = snapshot.get(path)
            if path in self._watchers or entry is None:
                continue
            linked = path in scan.linked_files
            try:
                if hub is None:
                    hub = _Inotify.for_loop()
                watch = hub.add(
                    path,
                    (lambda name, path=path: self._on_linked_file_event(path))
                    if linked
                    else (lambda name, path=path: self._on_event(path, name)),
                    lambda path=path: self._on_watch_error(path),
                )
            except OSError as error:
                # Out of watches or unsupported: compare snapshots from now on, and say coverage was uncertain.
                if error.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                    self._switch_to_polling()
                    return False
                continue
            self._watchers[path] = _Installed(watch, entry.dev, entry.ino)
            added = True
        return added

    def _on_watch_error(self, path: str) -> None:
        """A watched directory that disappears or fails: rescan, which also replaces the watcher."""
        installed = self._watchers.get(path)
        if installed is not None and installed.watch.closed:
            del self._watchers[path]
        self._schedule_flush()

    def _switch_to_polling(self) -> None:
        if self._mode == "polling":
            return
        self._mode = "polling"
        for installed in self._watchers.values():
            installed.watch.close()
        self._watchers.clear()
        self._deliver({"overflow": True})
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._schedule_poll()

    def _on_event(self, directory: str, filename: str | None) -> None:
        if self._closed:
            return
        path = directory if filename is None else _child(directory, filename)
        # Events about unrelated siblings of an ancestor, or about excluded entries, are ignored.
        relevant = self._in_scope(path)
        if relevant:
            self._events.add(self._reported(path))
        if relevant or filename is None:
            self._schedule_flush()

    def _on_linked_file_event(self, target: str) -> None:
        """The file a target links to changed: report the target."""
        if self._closed:
            return
        self._events.add(target)
        self._schedule_flush()

    def _in_scope(self, path: str) -> bool:
        for target in self._targets:
            if _is_within(target.path, path):
                return True
            if not _is_within(path, target.path) or path == target.path:
                continue
            components = os.path.relpath(path, target.path).split(os.sep)
            if not target.recursive and len(components) > 1:
                continue
            if any(_excluded(target, name) for name in components):
                continue
            return True
        return False
