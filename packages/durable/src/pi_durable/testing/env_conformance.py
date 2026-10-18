"""Runner-independent conformance cases for ``ExecutionEnv`` implementations (port of ``testing/env-conformance.ts``)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pi_ai.abort import AbortController
from pi_chord.context import BACKGROUND_CONTEXT, Context, with_abort_signal

from ..env.types import (
    ExecutionEnv,
    FileInfo,
    ListFilesOptions,
    Result,
    ShellOutputInfo,
    WatchChange,
    WatchTarget,
    get_or_throw,
)
from .types import EnvConformanceCase, EnvConformanceOptions

__all__ = ["create_env_conformance"]

context = BACKGROUND_CONTEXT

_EnvTest = Callable[[ExecutionEnv], Awaitable[None]]


def _aborted_context() -> Context:
    controller = AbortController()
    controller.abort()
    return with_abort_signal(controller.signal, BACKGROUND_CONTEXT)


def _error_code(result: Result[Any, Any]) -> str | None:
    return None if result["ok"] is True else result["error"].code


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _whole_file_text(data: bytes) -> str:
    """``new TextDecoder().decode(bytes)``: replacement characters and a dropped leading byte-order mark."""
    text = _decode(data)
    return text[1:] if text.startswith("﻿") else text


def _utf8_length(text: str) -> int:
    return len(text.encode("utf-8", errors="surrogatepass"))


async def _read_all(env: ExecutionEnv, path: str, max_entries: int) -> tuple[list[list[FileInfo]], bool]:
    reader = get_or_throw(await env.open_dir_reader(path, context))
    pages: list[list[FileInfo]] = []
    try:
        for _page in range(1000):
            page = get_or_throw(await reader.next(max_entries, context))
            pages.append(page["entries"])
            if page["done"]:
                return pages, True
        return pages, False
    finally:
        await reader.close(context)


async def _listed(env: ExecutionEnv, path: str, options: ListFilesOptions | None = None) -> dict[str, Any]:
    """A listing as ``<kind letter> <path>`` lines (``f``, ``d``, ``l``), with its truncation."""
    listing = get_or_throw(await env.list_files(path, options, context))
    letter = {"file": "f", "directory": "d", "symlink": "l"}
    return {
        "entries": [f"{letter[entry['kind']]} {entry['path']}" for entry in listing["entries"]],
        "truncated": listing["truncated"],
    }


def _covers(change: WatchChange, path: str) -> bool:
    """Whether a change reports ``path``: an overflow, or a reported path at or above it."""
    if "overflow" in change:
        return True
    if "paths" not in change:
        return False
    return any(
        path == reported or path.startswith(f"{reported}/") or path.startswith(f"{reported}\\")
        for reported in change["paths"]  # type: ignore[typeddict-item]
    )


class _Watching:
    def __init__(self, env: ExecutionEnv, changes: list[WatchChange]) -> None:
        self._env = env
        self.changes = changes

    async def absolute(self, path: str) -> str:
        return get_or_throw(await self._env.absolute_path(path, context))

    async def expect_change(self, path: str, change: Callable[[], Awaitable[None]]) -> None:
        """Wait up to three seconds for a change, after the call, that reports ``path``."""
        target = await self.absolute(path)
        start = len(self.changes)
        await change()
        deadline = time.monotonic() + 3
        while not any(_covers(entry, target) for entry in self.changes[start:]):
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"No change reported {target}; got {json.dumps(self.changes[start:], default=str)}"
                )
            await asyncio.sleep(0.02)


async def _watching(
    env: ExecutionEnv, targets: Sequence[WatchTarget], run: Callable[[_Watching], Awaitable[None]]
) -> None:
    """Watch ``targets`` while ``run`` changes files."""
    changes: list[WatchChange] = []
    watcher = get_or_throw(await env.watch(targets, changes.append, context))
    try:
        await run(_Watching(env, changes))
    finally:
        await watcher.close(context)


def create_env_conformance(options: EnvConformanceOptions) -> Sequence[EnvConformanceCase]:
    """Create the runner-independent cases for an ``ExecutionEnv``.

    ``withEnv`` must call and await its callback exactly once per case with an environment whose ``cwd`` is a
    fresh, empty, writable directory.
    """
    a = options["assertions"]
    with_env = options["withEnv"]
    shell = list(options.get("shell", ["sh", "-c"]))
    symlinks = options.get("symlinks", True)
    cases: list[EnvConformanceCase] = []

    def case(name: str, timeout_ms: float | None = None) -> Callable[[_EnvTest], _EnvTest]:
        def register(test: _EnvTest) -> _EnvTest:
            cases.append(EnvConformanceCase(name, lambda: with_env(test), timeout_ms))
            return test

        return register

    # Watch cases wait up to three seconds per step, longer than test runners allow by default.
    def watch_case(name: str) -> Callable[[_EnvTest], _EnvTest]:
        return case(name, 30_000)

    async def exec_collect(env: ExecutionEnv, command: str | Sequence[str], cwd: str | None = None) -> dict[str, Any]:
        output = {"stdout": "", "stderr": ""}

        def on_output(text: str, _context: Context, info: ShellOutputInfo) -> None:
            output[info["stream"]] += text

        exec_options: Any = {"onOutput": on_output}
        if cwd is not None:
            exec_options["cwd"] = cwd
        result = await env.exec(command, exec_options, context)
        return {"result": result, **output}

    @case("binary reader reads byte ranges of the opened file")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("data.txt", "hello world", context))
        reader = get_or_throw(await env.open_binary_reader("data.txt", None, context))
        info = get_or_throw(await reader.info(context))
        a.partial_deep_equal(info, {"name": "data.txt", "kind": "file", "size": 11})
        a.strict_equal(_decode(get_or_throw(await reader.read(0, 5, context))), "hello")
        a.strict_equal(_decode(get_or_throw(await reader.read(6, 100, context))), "world")
        a.strict_equal(len(get_or_throw(await reader.read(11, 4, context))), 0)
        a.strict_equal(len(get_or_throw(await reader.read(50, 1, context))), 0)
        a.strict_equal(len(get_or_throw(await reader.read(3, 0, context))), 0)
        a.strict_equal(_error_code(await reader.read(-1, 1, context)), "invalid")
        a.strict_equal(_error_code(await reader.read(0, 1.5, context)), "invalid")  # type: ignore[arg-type]
        a.strict_equal(_error_code(await reader.read(0, 1, _aborted_context())), "aborted")
        await reader.close(context)
        await reader.close(context)
        a.strict_equal(_error_code(await reader.read(0, 1, context)), "invalid")
        a.strict_equal(_error_code(await reader.info(context)), "invalid")

    @case("binary reader scans lines like decoding the whole file")
    async def _(env: ExecutionEnv) -> None:
        # A byte-order mark, an invalid sequence before a newline, an empty line, a later U+FEFF, and no final newline.
        data = bytes([0xEF, 0xBB, 0xBF, 0x61, 0x0A, 0xE2, 0x82, 0x0A, 0x0A, 0xEF, 0xBB, 0xBF, 0x62, 0x0A, 0xC3, 0xA9])
        get_or_throw(await env.write_file("lines.txt", data, context))
        lines = _whole_file_text(data).split("\n")
        reader = get_or_throw(await env.open_binary_reader("lines.txt", None, context))
        try:
            for start_line, end_line in ((0, None), (0, 1), (1, 3), (2, 3), (3, None), (4, 9)):
                scan_options: Any = {"startLine": start_line}
                if end_line is not None:
                    scan_options["endLine"] = end_line
                scan = get_or_throw(await reader.scan_lines(scan_options, context))
                selected = lines[start_line:end_line]

                def text_range(start: int, end: int) -> str:
                    return _decode(data[start:end]) if start > 0 else _whole_file_text(data[start:end])

                a.strict_equal(scan["newlines"], len(lines) - 1)
                a.strict_equal(text_range(scan["start"], scan["end"]), "\n".join(selected))
                a.strict_equal(scan["selectedBytes"], _utf8_length("\n".join(selected)))
                a.strict_equal(text_range(scan["start"], scan["firstLineEnd"]), lines[start_line])
                a.strict_equal(scan["firstLineBytes"], _utf8_length(lines[start_line]))
            a.partial_deep_equal(
                get_or_throw(await reader.scan_lines({"startLine": 9}, context)),
                {"start": len(data), "end": len(data), "selectedBytes": 0},
            )
            a.strict_equal(_error_code(await reader.scan_lines({"startLine": 2, "endLine": 2}, context)), "invalid")
        finally:
            await reader.close(context)

    @case("binary reader keeps reading the file it opened after a rename")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("a.txt", "one", context))
        reader = get_or_throw(await env.open_binary_reader("a.txt", None, context))
        try:
            get_or_throw(await env.rename_file("a.txt", "b.txt", context))
            get_or_throw(await env.write_file("a.txt", "two", context))
            a.strict_equal(_decode(get_or_throw(await reader.read(0, 10, context))), "one")
        finally:
            await reader.close(context)

    @case("binary reader refuses directories, missing files and aborted opens")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("dir", None, context))
        get_or_throw(await env.write_file("file.txt", "x", context))
        a.strict_equal(_error_code(await env.open_binary_reader("dir", None, context)), "is_directory")
        a.strict_equal(_error_code(await env.open_binary_reader("missing.txt", None, context)), "not_found")
        a.strict_equal(_error_code(await env.open_binary_reader("file.txt", None, _aborted_context())), "aborted")

    @case("directory reader pages every entry exactly once")
    async def _(env: ExecutionEnv) -> None:
        names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
        for name in names:
            get_or_throw(await env.write_file(name, name, context))
        get_or_throw(await env.create_dir("sub", None, context))
        pages, done = await _read_all(env, ".", 2)
        a.ok(done, "directory reader reached the end")
        for page in pages:
            a.ok(len(page) <= 2, "page within maxEntries")
        entries = [entry for page in pages for entry in page]
        a.deep_equal(sorted(entry["name"] for entry in entries), sorted([*names, "sub"]))
        a.strict_equal(next((entry["kind"] for entry in entries if entry["name"] == "sub"), None), "directory")
        a.partial_deep_equal(
            next((entry for entry in entries if entry["name"] == "a.txt"), None), {"kind": "file", "size": 5}
        )

    @case("directory reader reports the end and refuses use after close")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("empty", None, context))
        reader = get_or_throw(await env.open_dir_reader("empty", context))
        a.deep_equal(get_or_throw(await reader.next(10, context)), {"entries": [], "done": True})
        a.deep_equal(get_or_throw(await reader.next(10, context)), {"entries": [], "done": True})
        a.strict_equal(_error_code(await reader.next(0, context)), "invalid")
        a.strict_equal(_error_code(await reader.next(1, _aborted_context())), "aborted")
        await reader.close(context)
        await reader.close(context)
        a.strict_equal(_error_code(await reader.next(1, context)), "invalid")

    @case("directory reader refuses missing paths and files")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("file.txt", "x", context))
        a.strict_equal(_error_code(await env.open_dir_reader("missing", context)), "not_found")
        a.strict_equal(_error_code(await env.open_dir_reader("file.txt", context)), "not_directory")
        a.strict_equal(_error_code(await env.open_dir_reader(".", _aborted_context())), "aborted")

    @case("directory reader skips entries removed during enumeration")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("dir", None, context))
        for name in ("x", "y", "z"):
            get_or_throw(await env.write_file(f"dir/{name}", name, context))
        reader = get_or_throw(await env.open_dir_reader("dir", context))
        try:
            for name in ("x", "y", "z"):
                get_or_throw(await env.remove(f"dir/{name}", None, context))
            entries: list[FileInfo] = []
            for _page in range(10):
                page = get_or_throw(await reader.next(10, context))
                entries.extend(page["entries"])
                if page["done"]:
                    break
            a.deep_equal(entries, [])
        finally:
            await reader.close(context)

    @watch_case("watch reports a missing file's creation, changes, replacement and removal")
    async def _(env: ExecutionEnv) -> None:
        async def run(w: _Watching) -> None:
            async def write(content: str) -> None:
                get_or_throw(await env.write_file("AGENTS.md", content, context))

            await w.expect_change("AGENTS.md", lambda: write("one"))
            await w.expect_change("AGENTS.md", lambda: write("two!"))

            # Editors replace a file by renaming a new one over it.
            async def replace() -> None:
                get_or_throw(await env.write_file("AGENTS.md.tmp", "three", context))
                get_or_throw(await env.rename_file("AGENTS.md.tmp", "AGENTS.md", context))

            await w.expect_change("AGENTS.md", replace)
            await w.expect_change("AGENTS.md", lambda: write("four"))

            async def remove() -> None:
                get_or_throw(await env.remove("AGENTS.md", None, context))

            await w.expect_change("AGENTS.md", remove)

        await _watching(env, [{"path": "AGENTS.md"}], run)

    @watch_case("watch reports a missing target whose ancestors are created")
    async def _(env: ExecutionEnv) -> None:
        async def run(w: _Watching) -> None:
            async def write() -> None:
                get_or_throw(await env.write_file("a/b/c/AGENTS.md", "x", context))

            await w.expect_change("a/b/c/AGENTS.md", write)

        await _watching(env, [{"path": "a/b/c/AGENTS.md"}], run)

    def writer(env: ExecutionEnv, path: str, content: str) -> Callable[[], Awaitable[None]]:
        async def write() -> None:
            get_or_throw(await env.write_file(path, content, context))

        return write

    @watch_case("watch follows directories created together with their contents")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("skills", None, context))

        async def run(w: _Watching) -> None:
            # Written before any watcher on the new directories can exist.
            await w.expect_change("skills/a/b/SKILL.md", writer(env, "skills/a/b/SKILL.md", "one"))
            await w.expect_change("skills/a/b/SKILL.md", writer(env, "skills/a/b/SKILL.md", "two!"))
            await w.expect_change("skills/a/b/c/SKILL.md", writer(env, "skills/a/b/c/SKILL.md", "deeper"))

        await _watching(env, [{"path": "skills", "recursive": True}], run)

    @watch_case("watch keeps watching a path whose parent is renamed and recreated")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("proj/.pi/skills/x.md", "x", context))

        async def run(w: _Watching) -> None:
            async def rename() -> None:
                get_or_throw(await env.rename_file("proj/.pi", "proj/old", context))

            await w.expect_change("proj/.pi/skills", rename)
            await w.expect_change("proj/.pi/skills/y.md", writer(env, "proj/.pi/skills/y.md", "y"))
            await w.expect_change("proj/.pi/skills/y.md", writer(env, "proj/.pi/skills/y.md", "yy"))

        await _watching(env, [{"path": "proj/.pi/skills", "recursive": True}], run)

    @watch_case("watch skips excluded entries and reports a rename out of them")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("skills", None, context))
        targets: list[WatchTarget] = [
            {"path": "skills", "recursive": True, "exclude": {"hidden": True, "names": ["node_modules"]}}
        ]

        async def run(w: _Watching) -> None:
            get_or_throw(await env.write_file("skills/node_modules/dep/SKILL.md", "dep", context))
            get_or_throw(await env.write_file("skills/.SKILL.md.tmp", "draft", context))

            async def rename() -> None:
                get_or_throw(await env.rename_file("skills/.SKILL.md.tmp", "skills/SKILL.md", context))

            await w.expect_change("skills/SKILL.md", rename)
            hidden = [await w.absolute("skills/node_modules"), await w.absolute("skills/.SKILL.md.tmp")]
            for change in w.changes:
                if "paths" not in change:
                    continue
                for path in change["paths"]:  # type: ignore[typeddict-item]
                    a.ok(
                        not any(path == excluded or path.startswith(excluded) for excluded in hidden),
                        f"excluded {path}",
                    )

        await _watching(env, targets, run)

    @watch_case("watch keeps recursive coverage where a non-recursive target overlaps")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("skills/a/one.md", "one", context))

        async def run(w: _Watching) -> None:
            await w.expect_change("skills/a/two.md", writer(env, "skills/a/two.md", "two"))

        await _watching(env, [{"path": "skills"}, {"path": "skills", "recursive": True}], run)

    @watch_case("watch follows a directory replaced at the same path")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.write_file("skills/a/x.md", "x", context))

        async def run(w: _Watching) -> None:
            async def replace() -> None:
                get_or_throw(await env.rename_file("skills/a", "skills-old", context))
                get_or_throw(await env.create_dir("skills/a", None, context))

            await w.expect_change("skills/a", replace)
            await w.expect_change("skills/a/y.md", writer(env, "skills/a/y.md", "y"))
            await w.expect_change("skills/a/y.md", writer(env, "skills/a/y.md", "yy"))

        await _watching(env, [{"path": "skills", "recursive": True}], run)

    @watch_case("watch stops reporting once closed")
    async def _(env: ExecutionEnv) -> None:
        changes: list[WatchChange] = []
        watcher = get_or_throw(await env.watch([{"path": "file.txt"}], changes.append, context))
        a.ok(watcher.mode in ("native", "polling"), "watcher reports its mode")
        await watcher.close(context)
        await watcher.close(context)
        get_or_throw(await env.write_file("file.txt", "x", context))
        await asyncio.sleep(0.3)
        a.deep_equal(changes, [])

    @case("argv exec passes arguments to the program without shell parsing")
    async def _(env: ExecutionEnv) -> None:
        hostile = "it's $(touch pwned) `touch pwned` *; touch pwned"
        collected = await exec_collect(env, [*shell, 'printf "%s|%s" "$1" "$2"', "argv0", hostile, "a b"])
        a.strict_equal(get_or_throw(collected["result"])["exitCode"], 0)
        a.strict_equal(collected["stdout"], f"{hostile}|a b")
        a.strict_equal(get_or_throw(await env.exists("pwned", context)), False)

    @case("exec reports the stream of every chunk in both forms")
    async def _(env: ExecutionEnv) -> None:
        script = "printf out; printf err >&2; printf more"
        argv = await exec_collect(env, [*shell, script])
        a.strict_equal(get_or_throw(argv["result"])["exitCode"], 0)
        a.strict_equal(argv["stdout"], "outmore")
        a.strict_equal(argv["stderr"], "err")
        string = await exec_collect(env, script)
        a.strict_equal(get_or_throw(string["result"])["exitCode"], 0)
        a.strict_equal(string["stdout"], "outmore")
        a.strict_equal(string["stderr"], "err")

    @case("argv exec honors cwd and exit codes")
    async def _(env: ExecutionEnv) -> None:
        get_or_throw(await env.create_dir("sub", None, context))
        made = await exec_collect(env, [*shell, "printf x > made.txt; exit 3"], "sub")
        a.strict_equal(get_or_throw(made["result"])["exitCode"], 3)
        a.strict_equal(get_or_throw(await env.read_text_file("sub/made.txt", context)), "x")

    @case("argv exec reports missing programs and empty argv as spawn errors")
    async def _(env: ExecutionEnv) -> None:
        a.strict_equal(
            _error_code(await env.exec(["pi-durable-conformance-missing-program"], None, context)), "spawn_error"
        )
        a.strict_equal(_error_code(await env.exec([], None, context)), "spawn_error")

    @case("windowed exec keeps the exact tail and counts what it skips")
    async def _(env: ExecutionEnv) -> None:
        lines = 2000
        window: Any = {"maxBytes": 200, "maxLines": 5, "minIntervalMs": 0, "bytesPerSecond": 1_000_000_000}
        counts = {"bytes": 0, "newlines": 0}
        tail = [""]

        def on_output(text: str, _context: Context, info: ShellOutputInfo) -> None:
            skipped = info.get("skipped")
            if skipped is not None:
                counts["bytes"] += skipped["bytes"]
                counts["newlines"] += skipped["newlines"]
                a.ok(
                    _utf8_length(text) > window["maxBytes"] or text.count("\n") > window["maxLines"],
                    "a skip is followed by more than the window",
                )
                tail[0] = ""
            counts["bytes"] += _utf8_length(text)
            counts["newlines"] += text.count("\n")
            tail[0] += text

        result = await env.exec(
            [*shell, f"i=0; while [ $i -lt {lines} ]; do echo line-$i; i=$((i+1)); done"],
            {"window": window, "onOutput": on_output},
            context,
        )
        a.strict_equal(get_or_throw(result)["exitCode"], 0)
        expected = [f"line-{index}\n" for index in range(lines)]
        a.strict_equal(counts["bytes"], len("".join(expected)))
        a.strict_equal(counts["newlines"], lines)
        a.ok(tail[0].endswith("".join(expected[-window["maxLines"] :])), "the delivered output ends with the tail")

    @case("argv exec distinguishes timeout from abort")
    async def _(env: ExecutionEnv) -> None:
        timed_out = await env.exec([*shell, "sleep 2"], {"timeout": 0.1}, context)
        a.strict_equal(_error_code(timed_out), "timeout")
        controller = AbortController()
        running = asyncio.ensure_future(
            env.exec([*shell, "sleep 2"], None, with_abort_signal(controller.signal, context))
        )
        asyncio.get_running_loop().call_later(0.1, controller.abort)
        a.strict_equal(_error_code(await running), "aborted")

    if options.get("listFiles", False):

        async def write_all(env: ExecutionEnv, files: Sequence[str]) -> None:
            for file in files:
                get_or_throw(await env.write_file(file, "", context))

        @case("listFiles lists files and directories depth first in name order")
        async def _(env: ExecutionEnv) -> None:
            await write_all(env, ["b.txt", "a/z.txt", "a/b/c.txt", ".hidden/x", "A.txt"])
            get_or_throw(await env.create_dir("empty", None, context))
            a.deep_equal(
                await _listed(env, "."),
                {
                    "entries": [
                        "d .hidden",
                        "f .hidden/x",
                        "f A.txt",
                        "d a",
                        "d a/b",
                        "f a/b/c.txt",
                        "f a/z.txt",
                        "f b.txt",
                        "d empty",
                    ],
                    "truncated": False,
                },
            )
            a.deep_equal(
                (await _listed(env, ".", {"hidden": False}))["entries"],
                ["f A.txt", "d a", "d a/b", "f a/b/c.txt", "f a/z.txt", "f b.txt", "d empty"],
            )
            expected = {"entries": ["d b", "f b/c.txt", "f z.txt"], "truncated": False}
            a.deep_equal(await _listed(env, "a"), expected)
            a.deep_equal(await _listed(env, get_or_throw(await env.absolute_path("a", context))), expected)
            a.deep_equal(await _listed(env, "empty"), {"entries": [], "truncated": False})

        @case("listFiles skips what git ignores in work trees, .git, and excluded names")
        async def _(env: ExecutionEnv) -> None:
            await write_all(env, ["keep.txt", "out/x.js", "sub/keep.txt", "sub/drop.log", "node_modules/m/i.js"])
            get_or_throw(await env.write_file(".gitignore", "out/\n*.log\n", context))
            get_or_throw(await env.write_file("sub/.gitignore", "!keep.txt\n", context))
            # Not a work tree yet: `.gitignore` files have no effect.
            a.ok("f out/x.js" in (await _listed(env, "."))["entries"], "no work tree yet")
            get_or_throw(await env.write_file(".git/HEAD", "ref: refs/heads/main\n", context))
            get_or_throw(await env.write_file(".git/info/exclude", "keep.txt\n", context))
            a.deep_equal(
                (await _listed(env, ".", {"exclude": ["node_modules"]}))["entries"],
                ["f .gitignore", "d sub", "f sub/.gitignore", "f sub/keep.txt"],
            )
            # A directory inside the work tree applies its parents' rules.
            a.deep_equal((await _listed(env, "sub"))["entries"], ["f .gitignore", "f keep.txt"])
            listed_all = (await _listed(env, ".", {"gitignore": False}))["entries"]
            a.ok(
                "f out/x.js" in listed_all and "f node_modules/m/i.js" in listed_all and "f keep.txt" in listed_all,
                "gitignore false lists everything",
            )
            a.ok(not any(".git/" in entry or entry == "d .git" for entry in listed_all), ".git is never listed")

        @case("listFiles stops at its bounds and reports truncated")
        async def _(env: ExecutionEnv) -> None:
            await write_all(env, ["a", "b", "c"])
            a.deep_equal(await _listed(env, ".", {"maxEntries": 2}), {"entries": ["f a", "f b"], "truncated": True})
            a.deep_equal(
                await _listed(env, ".", {"maxEntries": 3}), {"entries": ["f a", "f b", "f c"], "truncated": False}
            )
            # Each entry counts its path plus two bytes.
            a.deep_equal(await _listed(env, ".", {"maxBytes": 6}), {"entries": ["f a", "f b"], "truncated": True})
            a.deep_equal(await _listed(env, ".", {"maxEntries": 0}), {"entries": [], "truncated": True})
            a.strict_equal(_error_code(await env.list_files(".", {"maxEntries": -1}, context)), "invalid")
            a.strict_equal(_error_code(await env.list_files(".", {"timeoutMs": 1.5}, context)), "invalid")

        @case("listFiles fails on missing directories, files, and aborts")
        async def _(env: ExecutionEnv) -> None:
            get_or_throw(await env.write_file("file.txt", "", context))
            a.strict_equal(_error_code(await env.list_files("missing", None, context)), "not_found")
            a.strict_equal(_error_code(await env.list_files("file.txt", None, context)), "not_directory")
            a.strict_equal(_error_code(await env.list_files(".", None, _aborted_context())), "aborted")

    if symlinks and options.get("listFiles", False):

        @case("listFiles lists symbolic links without following them")
        async def _(env: ExecutionEnv) -> None:
            get_or_throw(await env.write_file("dir/inner.txt", "", context))
            linked = await env.exec(
                [*shell, "ln -s dir link && ln -s missing dangling && ln -s .. dir/loop"], None, context
            )
            a.strict_equal(get_or_throw(linked)["exitCode"], 0)
            a.deep_equal(
                (await _listed(env, "."))["entries"], ["l dangling", "d dir", "f dir/inner.txt", "l dir/loop", "l link"]
            )
            # The listed directory itself may be a link.
            a.deep_equal((await _listed(env, "link"))["entries"], ["f inner.txt", "l loop"])

    if symlinks:

        @case("binary reader follows symlinks unless noFollow refuses the final one")
        async def _(env: ExecutionEnv) -> None:
            get_or_throw(await env.write_file("target.txt", "target", context))
            get_or_throw(await env.create_dir("sub", None, context))
            get_or_throw(await env.write_file("sub/inner.txt", "inner", context))
            linked = await env.exec([*shell, "ln -s target.txt link.txt && ln -s sub dirlink"], None, context)
            a.strict_equal(get_or_throw(linked)["exitCode"], 0)

            followed = get_or_throw(await env.open_binary_reader("link.txt", None, context))
            a.strict_equal(_decode(get_or_throw(await followed.read(0, 10, context))), "target")
            await followed.close(context)

            a.strict_equal(
                _error_code(await env.open_binary_reader("link.txt", {"noFollow": True}, context)), "invalid"
            )

            # Only the final component is refused; earlier symlinked directories still resolve.
            inner = get_or_throw(await env.open_binary_reader("dirlink/inner.txt", {"noFollow": True}, context))
            a.strict_equal(_decode(get_or_throw(await inner.read(0, 10, context))), "inner")
            await inner.close(context)

        @watch_case("watch reports changes to the file a watched symbolic link points to")
        async def _(env: ExecutionEnv) -> None:
            get_or_throw(await env.write_file("data/real.md", "one", context))
            get_or_throw(await env.create_dir("config", None, context))
            linked = await env.exec([*shell, "ln -s ../data/real.md config/AGENTS.md"], None, context)
            a.strict_equal(get_or_throw(linked)["exitCode"], 0)

            async def run(w: _Watching) -> None:
                await w.expect_change("config/AGENTS.md", writer(env, "data/real.md", "two!"))

            await _watching(env, [{"path": "config/AGENTS.md"}], run)

    return cases
