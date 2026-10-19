"""Port of ``test/tools.test.ts``: the coding tools against a local environment and a minimal execution API.

``applyPatch`` (jsdiff) is :func:`apply_patch` below, a small unified-patch applier. ``NodeExecutionEnv`` is
``LocalExecutionEnv``; ``Object.create(env)`` overrides become subclasses or wrappers. Dropped: the
Windows-only "runs real PowerShell with UTF-8 output" case (skipped off Windows, as upstream).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pi_ai import TextContent
from pi_ai.abort import AbortController
from pi_chord.context import BACKGROUND_CONTEXT, Context, with_abort_signal
from pi_durable.env import (
    ExecutionError,
    FileError,
    Result,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputSkip,
    ShellOutputWindow,
    err,
    get_or_throw,
)
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.tools import (
    create_bash_tool,
    create_edit_tool,
    create_powershell_tool,
    create_read_tool,
    create_write_tool,
)
from pi_durable.tools.file_mutation_queue import with_file_mutation_queue
from pi_durable.tools.image import detect_supported_image_mime_type
from pi_durable.truncate import DEFAULT_MAX_LINES

context = BACKGROUND_CONTEXT


class FakeApi:
    """A minimal execution API: the environment, collected output and diagnostics, and nothing durable."""

    def __init__(self, env: Any) -> None:
        self.task_id = 1
        self.conversation_id = 1
        self.call_id = "call"
        self.env = env
        self.output_window: ShellOutputWindow | None = None
        self.streamed: list[str] = []
        self.reported: list[Any] = []

    def output(self, chunk: str | bytes, skipped: ShellOutputSkip | None = None) -> None:
        self.streamed.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace"))

    def retained_output(self) -> Any:
        return {"text": "".join(self.streamed), "truncated": False}

    def diagnostic(self, diagnostic: Any) -> None:
        self.reported.append(diagnostic)

    async def details(self, value: Any, call_context: Context) -> None:
        return None

    async def agent(self, call_context: Context) -> Any:
        # No model: `read` treats it as one that sees images.
        return {}


async def run(tool: Any, args: Any, env: Any, call_context: Context = context) -> dict[str, Any]:
    api = FakeApi(env)
    result = await tool.execute(args, api, call_context)
    return {**result, "streamed": api.streamed, "reported": api.reported}


async def run_failing(tool: Any, args: Any, env: Any) -> tuple[BaseException, list[str], list[Any]]:
    """Run a tool expected to raise; returns the error with what it streamed and reported first."""
    api = FakeApi(env)
    try:
        await tool.execute(args, api, context)
    except Exception as error:
        return error, api.streamed, api.reported
    raise AssertionError("Expected the tool to raise")


def append_text(path: str, text: str) -> None:
    with open(path, "a") as file:
        file.write(text)


def text_output(result: Any) -> str:
    return "\n".join(part.text for part in result.get("output") or [] if isinstance(part, TextContent))


def diagnostic_text(result: Any) -> str:
    return "\n".join(diagnostic["message"] for diagnostic in result.get("diagnostics") or [])


def deferred() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


def resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def aborting_context() -> tuple[AbortController, Context]:
    controller = AbortController()
    return controller, with_abort_signal(controller.signal, BACKGROUND_CONTEXT)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_patch(source: str, patch: str) -> str:
    """jsdiff ``applyPatch`` for the patches the edit tool writes (exact context, no fuzz)."""
    lines = source.split("\n")
    output: list[str] = []
    position = 0
    patch_lines = patch.split("\n")
    index = 0
    while index < len(patch_lines):
        match = _HUNK.match(patch_lines[index])
        index += 1
        if match is None:
            continue
        old_start = int(match.group(1))
        old_count = 1 if match.group(2) is None else int(match.group(2))
        start = old_start - 1 if old_count > 0 else old_start
        output.extend(lines[position:start])
        position = start
        while index < len(patch_lines) and patch_lines[index][:1] in (" ", "-", "+", "\\"):
            line = patch_lines[index]
            index += 1
            if line.startswith("\\"):
                continue
            if line[0] in " -":
                assert lines[position] == line[1:], f"context mismatch at line {position + 1}"
                position += 1
                if line[0] == " ":
                    output.append(line[1:])
            else:
                output.append(line[1:])
    output.extend(lines[position:])
    return "\n".join(output)


class SlowReadEnv(LocalExecutionEnv):
    async def read_text_file(self, path: str, context: Context) -> Result[str, FileError]:
        await asyncio.sleep(0.02)
        return await super().read_text_file(path, context)


class BlockingWriteEnv(LocalExecutionEnv):
    def __init__(self, *, cwd: str) -> None:
        super().__init__(cwd=cwd)
        self.first_write_started = deferred()
        self.finish_first_write = deferred()
        self.second_write_started = False

    async def write_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]:
        if content == "first\n":
            resolve(self.first_write_started)
            await self.finish_first_write
        elif content == "second\n":
            self.second_write_started = True
        return await super().write_file(path, content, context)


class BlockingEditEnv(LocalExecutionEnv):
    def __init__(self, *, cwd: str) -> None:
        super().__init__(cwd=cwd)
        self.first_edit_write_started = deferred()
        self.finish_first_edit_write = deferred()
        self.first_edit_write_settled = False
        self.second_edit_write_started = False

    async def write_file(self, path: str, content: str | bytes, context: Context) -> Result[None, FileError]:
        if content == "ALPHA\nbeta\n":
            resolve(self.first_edit_write_started)
            await self.finish_first_edit_write
            result = await super().write_file(path, content, BACKGROUND_CONTEXT)
            self.first_edit_write_settled = True
            return result
        if content in ("ALPHA\nBETA\n", "alpha\nBETA\n"):
            self.second_edit_write_started = True
        return await super().write_file(path, content, context)


TRUNCATED_OUTPUT_LINES = DEFAULT_MAX_LINES + 1


class TimeoutOutputEnv(LocalExecutionEnv):
    async def exec(
        self, command: str | Sequence[str], options: ShellExecOptions | None, context: Context
    ) -> Result[ShellExecResult, ExecutionError]:
        output = "".join(f"line-{index + 1}\n" for index in range(TRUNCATED_OUTPUT_LINES))
        spill_path = get_or_throw(await self.create_temp_file({"prefix": "timeout-", "suffix": ".log"}, context))
        get_or_throw(await self.write_file(spill_path, output, context))
        assert options is not None
        options["onOutput"](output, context, {"stream": "stdout"})
        error = ExecutionError("timeout", f"timeout:{options.get('timeout')}")
        error.spill_path = spill_path
        return err(error)


async def test_fail_with_an_ordinary_error_when_no_environment_is_configured() -> None:
    with pytest.raises(RuntimeError, match="No execution environment"):
        await run(create_read_tool(), {"path": "x"}, None)


# ─── read ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("signature", ["GIF87a", "GIF89a"])
def test_detects_the_complete_gif_signature(signature: str) -> None:
    assert detect_supported_image_mime_type(signature.encode("ascii")) == "image/gif"


async def test_reads_text_with_offsets_and_limits_and_reports_continuation_as_a_diagnostic(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("test.txt", "\n".join(f"Line {index + 1}" for index in range(100)), context))
    result = await run(create_read_tool(), {"path": "test.txt", "offset": 41, "limit": 20}, env)
    output = text_output(result)
    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "more lines" not in output
    assert diagnostic_text(result) == "40 more lines in file. Use offset=61 to continue."


async def test_truncates_large_text_by_line_count(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("large.txt", "\n".join(f"Line {index + 1}" for index in range(2500)), context))
    result = await run(create_read_tool(), {"path": "large.txt"}, env)
    assert diagnostic_text(result) == "Showing lines 1-2000 of 2500. Use offset=2001 to continue."
    assert result["diagnostics"][0]["code"] == "truncated"
    truncation = result["details"]["truncation"]
    assert truncation["truncated"] is True
    assert truncation["truncatedBy"] == "lines"
    assert truncation["totalLines"] == 2500
    assert truncation["outputLines"] == 2000


async def test_does_not_count_a_trailing_newline_as_an_extra_line_at_the_truncation_limit(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("exact.txt", "\n".join("x" for _ in range(2000)) + "\n", context))
    result = await run(create_read_tool(), {"path": "exact.txt"}, env)
    assert "details" not in result
    assert result["diagnostics"] == []


async def test_shows_the_start_of_a_line_longer_than_the_byte_limit(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("long.txt", "\u00e9" * 40_000 + "\nnext\n", context))
    result = await run(create_read_tool(), {"path": "long.txt"}, env)
    # Two-byte characters: the cut lands on a character boundary at or below the limit.
    assert text_output(result) == "\u00e9" * 25_600
    assert diagnostic_text(result) == (
        "Line 1 is 78.1KB, exceeds the 50.0KB limit; showing its first 50.0KB. Use bash: sed -n '1p' long.txt | "
        "tail -c +51201"
    )
    truncation = result["details"]["truncation"]
    assert truncation["truncated"] is True
    assert truncation["firstLineExceedsLimit"] is True
    assert truncation["outputBytes"] == 51_200
    assert truncation["outputLines"] == 1
    assert "content" not in truncation


async def test_rejects_offsets_beyond_the_file(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("short.txt", "one\ntwo\nthree", context))
    with pytest.raises(RuntimeError, match=re.escape("Offset 100 is beyond end of file (3 lines total)")):
        await run(create_read_tool(), {"path": "short.txt", "offset": 100}, env)


async def test_reads_a_log_that_grows_while_it_is_read(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("app.log", "one\ntwo\n", context))
    log_path = os.path.join(env.cwd, "app.log")

    # Every read appends, as a busy writer would; the scanned lines stay as they were.
    class GrowingReader:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        async def info(self, info_context: Context) -> Any:
            return await self.inner.info(info_context)

        async def scan_lines(self, options: Any, scan_context: Context) -> Any:
            return await self.inner.scan_lines(options, scan_context)

        async def close(self, close_context: Context) -> None:
            await self.inner.close(close_context)

        async def read(self, position: int, length: int, read_context: Context) -> Any:
            append_text(log_path, "more\n")
            return await self.inner.read(position, length, read_context)

    class GrowingEnv(LocalExecutionEnv):
        async def open_binary_reader(self, path: str, options: Any, context: Context) -> Any:
            opened = await super().open_binary_reader(path, options, context)
            if opened["ok"] is False:
                return opened
            return {"ok": True, "value": GrowingReader(opened["value"])}

    result = await run(create_read_tool(), {"path": "app.log", "limit": 2}, GrowingEnv(cwd=env.cwd))
    assert text_output(result) == "one\ntwo"


async def test_returns_images_by_content_as_one_image_block(tmp_path: Path) -> None:
    import base64

    from pi_ai import ImageContent

    env = LocalExecutionEnv(cwd=str(tmp_path))
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
    get_or_throw(await env.write_file("image.txt", base64.b64decode(data), context))
    result = await run(create_read_tool(), {"path": "image.txt"}, env)
    assert result["output"] == [ImageContent(data, "image/png")]
    assert "isError" not in result
    assert diagnostic_text(result) == "Read image file [image/png]."


# ─── write ───────────────────────────────────────────────────────────────────


async def test_writes_files_and_creates_parent_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = await run(create_write_tool(), {"path": "nested/dir/file.txt", "content": "hello"}, env)
    assert text_output(result) == "Successfully wrote to nested/dir/file.txt"
    assert get_or_throw(await env.read_text_file("nested/dir/file.txt", context)) == "hello"


async def test_keeps_the_mutation_queue_locked_until_an_aborted_write_settles(tmp_path: Path) -> None:
    env = BlockingWriteEnv(cwd=str(tmp_path))
    tool = create_write_tool()
    controller, abort_context = aborting_context()
    first_write = asyncio.ensure_future(run(tool, {"path": "file.txt", "content": "first\n"}, env, abort_context))
    await env.first_write_started
    controller.abort()
    second_write = asyncio.ensure_future(run(tool, {"path": "file.txt", "content": "second\n"}, env))
    await asyncio.sleep(0.02)
    assert env.second_write_started is False
    resolve(env.finish_first_write)
    with pytest.raises(Exception):  # noqa: B017 - upstream expects any rejection
        await first_write
    await second_write
    assert get_or_throw(await env.read_text_file("file.txt", context)) == "second\n"


# ─── edit ────────────────────────────────────────────────────────────────────


async def test_applies_disjoint_edits_and_returns_both_diff_formats(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    original = "alpha\nbeta\ngamma\ndelta\n"
    get_or_throw(await env.write_file("edit.txt", original, context))
    result = await run(
        create_edit_tool(),
        {
            "path": "edit.txt",
            "edits": [{"oldText": "alpha\n", "newText": "ALPHA\n"}, {"oldText": "gamma\n", "newText": "GAMMA\n"}],
        },
        env,
    )
    details = result["details"]
    assert text_output(result) == "Successfully replaced 2 block(s) in edit.txt."
    assert "ALPHA" in details["diff"]
    assert "GAMMA" in details["diff"]
    assert apply_patch(original, details["patch"]) == "ALPHA\nbeta\nGAMMA\ndelta\n"
    assert get_or_throw(await env.read_text_file("edit.txt", context)) == "ALPHA\nbeta\nGAMMA\ndelta\n"
    # The exact texts, as upstream (jsdiff) writes them.
    assert details["diff"] == "-1 alpha\n+1 ALPHA\n 2 beta\n-3 gamma\n+3 GAMMA\n 4 delta"
    assert details["patch"] == (
        "--- edit.txt\n+++ edit.txt\n@@ -1,4 +1,4 @@\n-alpha\n+ALPHA\n beta\n-gamma\n+GAMMA\n delta\n"
    )
    assert details["firstChangedLine"] == 1


def test_repairs_edits_sent_as_a_json_string_a_single_object_or_top_level_old_text_new_text() -> None:
    prepare = create_edit_tool().prepare_arguments
    assert prepare is not None
    edit = {"oldText": "a", "newText": "b"}
    as_string = {"path": "f", "edits": json.dumps([edit])}
    assert prepare(as_string) == {"path": "f", "edits": [edit]}
    assert as_string["edits"] == json.dumps([edit])
    assert prepare({"path": "f", "edits": json.dumps(edit)}) == {"path": "f", "edits": [edit]}
    assert prepare({"path": "f", "edits": edit}) == {"path": "f", "edits": [edit]}
    assert prepare({"path": "f", "edits": [edit], "oldText": "c", "newText": "d"}) == {
        "path": "f",
        "edits": [edit, {"oldText": "c", "newText": "d"}],
    }
    assert prepare({"path": "f", "edits": "not json"}) == {"path": "f", "edits": "not json"}


async def test_matches_all_edits_against_the_original_and_rejects_overlaps(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("edit.txt", "one\ntwo\nthree\n", context))
    with pytest.raises(RuntimeError, match="overlap"):
        await run(
            create_edit_tool(),
            {
                "path": "edit.txt",
                "edits": [
                    {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                    {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                ],
            },
            env,
        )
    assert get_or_throw(await env.read_text_file("edit.txt", context)) == "one\ntwo\nthree\n"


async def test_rejects_missing_and_duplicate_target_text(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("edit.txt", "foo foo foo", context))
    tool = create_edit_tool()
    with pytest.raises(RuntimeError, match="Could not find the exact text"):
        await run(tool, {"path": "edit.txt", "edits": [{"oldText": "bar", "newText": "baz"}]}, env)
    with pytest.raises(RuntimeError, match="Found 3 occurrences"):
        await run(tool, {"path": "edit.txt", "edits": [{"oldText": "foo", "newText": "bar"}]}, env)


async def test_keeps_the_mutation_queue_locked_until_an_aborted_edit_write_settles(tmp_path: Path) -> None:
    env = BlockingEditEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "alpha\nbeta\n", context))
    tool = create_edit_tool()
    controller, abort_context = aborting_context()
    first_edit = asyncio.ensure_future(
        run(tool, {"path": "file.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]}, env, abort_context)
    )
    await env.first_edit_write_started
    controller.abort()
    second_edit = asyncio.ensure_future(
        run(tool, {"path": "file.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]}, env)
    )
    await asyncio.sleep(0.02)
    assert env.second_edit_write_started is False
    resolve(env.finish_first_edit_write)
    with pytest.raises(RuntimeError, match="Operation aborted"):
        await first_edit
    await second_edit
    assert env.first_edit_write_settled is True
    assert get_or_throw(await env.read_text_file("file.txt", context)) == "ALPHA\nBETA\n"


async def test_serializes_concurrent_edits_through_canonical_and_symlink_paths(tmp_path: Path) -> None:
    env = SlowReadEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("target.txt", "alpha\nbeta\ngamma\n", context))
    os.symlink("target.txt", f"{env.cwd}/link.txt")
    tool = create_edit_tool()
    await asyncio.gather(
        run(tool, {"path": "target.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]}, env),
        run(tool, {"path": "link.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]}, env),
    )
    assert get_or_throw(await env.read_text_file("target.txt", context)) == "ALPHA\nBETA\ngamma\n"


async def test_serializes_edits_of_one_file_across_environment_objects_of_one_file_system(tmp_path: Path) -> None:
    first = SlowReadEnv(cwd=str(tmp_path))
    second = SlowReadEnv(cwd=str(tmp_path))
    get_or_throw(await first.write_file("file.txt", "alpha\nbeta\n", context))
    tool = create_edit_tool()
    await asyncio.gather(
        run(tool, {"path": "file.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]}, first),
        run(tool, {"path": "file.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]}, second),
    )
    assert get_or_throw(await first.read_text_file("file.txt", context)) == "ALPHA\nBETA\n"


async def test_serializes_a_new_file_created_through_a_symlinked_directory_with_its_canonical_path(
    tmp_path: Path,
) -> None:
    env = BlockingWriteEnv(cwd=str(tmp_path))
    os.mkdir(f"{env.cwd}/real")
    os.symlink(f"{env.cwd}/real", f"{env.cwd}/link")
    tool = create_write_tool()
    first = asyncio.ensure_future(run(tool, {"path": "link/new.txt", "content": "first\n"}, env))
    await env.first_write_started
    second = asyncio.ensure_future(run(tool, {"path": "real/new.txt", "content": "second\n"}, env))
    await asyncio.sleep(0.02)
    assert env.second_write_started is False
    resolve(env.finish_first_write)
    await asyncio.gather(first, second)
    assert get_or_throw(await env.read_text_file("real/new.txt", context)) == "second\n"


@pytest.mark.skipif(sys.platform == "win32", reason="a backslash is a separator on Windows")
async def test_keys_a_missing_file_whose_name_contains_a_backslash_like_the_created_file(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    created = deferred()
    release = deferred()

    async def first_mutation() -> None:
        get_or_throw(await env.write_file("a\\b.txt", "first\n", context))
        resolve(created)
        await release

    first = asyncio.ensure_future(with_file_mutation_queue(env, "a\\b.txt", first_mutation, context))
    await created
    entered = False

    async def second_mutation() -> None:
        nonlocal entered
        entered = True

    second = asyncio.ensure_future(with_file_mutation_queue(env, "a\\b.txt", second_mutation, context))
    await asyncio.sleep(0.02)
    assert entered is False
    resolve(release)
    await asyncio.gather(first, second)
    assert entered is True


async def test_does_not_serialize_the_same_path_on_different_file_systems(tmp_path: Path) -> None:
    class OtherFileSystem(BlockingWriteEnv):
        @property
        def id(self) -> str:
            return "other"

    local = BlockingWriteEnv(cwd=str(tmp_path))
    other = OtherFileSystem(cwd=str(tmp_path))
    tool = create_write_tool()
    blocked = asyncio.ensure_future(run(tool, {"path": "file.txt", "content": "first\n"}, local))
    await local.first_write_started
    await run(tool, {"path": "file.txt", "content": "second\n"}, other)
    assert other.second_write_started is True
    resolve(local.finish_first_write)
    await blocked


async def test_edits_regular_files_through_symlinks(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("target.txt", "before\n", context))
    os.symlink("target.txt", f"{env.cwd}/link.txt")
    await run(create_edit_tool(), {"path": "link.txt", "edits": [{"oldText": "before", "newText": "after"}]}, env)
    assert get_or_throw(await env.read_text_file("target.txt", context)) == "after\n"


async def test_preserves_bom_and_crlf_line_endings(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("edit.txt", "\ufeffone\r\ntwo\r\n", context))
    await run(create_edit_tool(), {"path": "edit.txt", "edits": [{"oldText": "two", "newText": "TWO"}]}, env)
    assert get_or_throw(await env.read_text_file("edit.txt", context)) == "\ufeffone\r\nTWO\r\n"


# ─── powershell ──────────────────────────────────────────────────────────────


def programs_env(tmp_path: Path, installed: Sequence[str], output: str, exit_code: int = 0) -> Any:
    """An environment where only the listed programs exist; it records each command and prints ``output``."""
    commands: list[Any] = []

    class ProgramsEnv(LocalExecutionEnv):
        async def exec(
            self, command: str | Sequence[str], options: ShellExecOptions | None, context: Context
        ) -> Result[ShellExecResult, ExecutionError]:
            commands.append(command)
            if isinstance(command, str) or command[0] not in installed:
                return err(ExecutionError("spawn_error", f"spawn {command[0]} ENOENT"))
            assert options is not None
            options["onOutput"](output, context, {"stream": "stdout"})
            return {"ok": True, "value": {"exitCode": exit_code}}

    return ProgramsEnv(cwd=str(tmp_path)), commands


async def test_runs_the_command_with_pwsh_as_one_argument_forcing_utf8_output(tmp_path: Path) -> None:
    env, commands = programs_env(tmp_path, ["pwsh"], "h\u00e9llo\n")
    result = await run(create_powershell_tool(), {"command": "Write-Output 'h\u00e9llo'"}, env)
    assert "".join(result["streamed"]) == "h\u00e9llo\n"
    assert commands == [
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\nWrite-Output 'h\u00e9llo'",
        ]
    ]


async def test_falls_back_to_windows_powershell_and_reports_the_last_start_failure(tmp_path: Path) -> None:
    windows_env, windows_commands = programs_env(tmp_path, ["powershell"], "ok")
    result = await run(create_powershell_tool(command_prefix="$x = 1"), {"command": "$x"}, windows_env)
    assert "".join(result["streamed"]) == "ok"
    assert [command[0] for command in windows_commands] == ["pwsh", "powershell"]
    assert re.search(r"\n\$x = 1\n\$x$", str(windows_commands[1][-1]))

    none_env, _ = programs_env(tmp_path, [], "")
    error, _, _ = await run_failing(create_powershell_tool(), {"command": "1"}, none_env)
    assert str(error) == "spawn powershell ENOENT"


async def test_answers_a_nonzero_exit_with_an_error_result_carrying_the_exit_code(tmp_path: Path) -> None:
    env, _ = programs_env(tmp_path, ["pwsh"], "partial", 3)
    result = await run(create_powershell_tool(), {"command": "exit 3"}, env)
    assert result["isError"] is True
    assert result["structuredOutput"] == {"output": "partial", "truncated": False, "exitCode": 3}
    assert result["diagnostics"] == [
        {"severity": "error", "code": "exit_code", "message": "Command exited with code 3"}
    ]
    assert "".join(result["streamed"]) == "partial"


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell")
async def test_runs_real_powershell_with_utf8_output(tmp_path: Path) -> None:
    result = await run(
        create_powershell_tool(),
        {"command": "Write-Output ('h' + [char]0xe9 + 'llo'); exit 0"},
        LocalExecutionEnv(cwd=str(tmp_path)),
    )
    assert "".join(result["streamed"]).strip() == "h\u00e9llo"


# ─── bash ────────────────────────────────────────────────────────────────────


async def test_passes_the_retained_window_to_the_environment_and_forwards_what_it_skipped(tmp_path: Path) -> None:
    window: ShellOutputWindow = {"maxBytes": 4, "maxLines": 1, "minIntervalMs": 100, "bytesPerSecond": 1024}
    skipped: ShellOutputSkip = {"bytes": 6, "newlines": 2, "endsWithNewline": True}
    received: list[Any] = []

    class SkippingEnv(LocalExecutionEnv):
        async def exec(
            self, command: str | Sequence[str], options: ShellExecOptions | None, context: Context
        ) -> Result[ShellExecResult, ExecutionError]:
            assert options is not None
            received.append(options.get("window"))
            options["onOutput"]("tail\n", context, {"stream": "stdout", "skipped": skipped})
            return {"ok": True, "value": {"exitCode": 0}}

    calls: list[tuple[str, Any]] = []

    class WindowApi(FakeApi):
        def output(self, chunk: str | bytes, skipped: ShellOutputSkip | None = None) -> None:
            calls.append((str(chunk), skipped))

    api = WindowApi(SkippingEnv(cwd=str(tmp_path)))
    api.output_window = window
    await create_bash_tool().execute({"command": "anything"}, api, context)  # type: ignore[arg-type]
    assert received == [window]
    assert calls == [("tail\n", skipped)]


async def test_streams_combined_stdout_and_stderr_returns_no_output_of_its_own_and_repeats_it_with_the_exit_code(
    tmp_path: Path,
) -> None:
    result = await run(
        create_bash_tool(), {"command": "printf out; printf err >&2"}, LocalExecutionEnv(cwd=str(tmp_path))
    )
    streamed = "".join(result["streamed"])
    assert "out" in streamed
    assert "err" in streamed
    assert "output" not in result
    assert result["structuredOutput"] == {"output": streamed, "truncated": False, "exitCode": 0}


async def test_answers_a_nonzero_exit_with_an_error_result_and_raises_on_a_timeout_after_streaming_the_output(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    tool = create_bash_tool()
    failed = await run(tool, {"command": "printf failed; exit 7"}, env)
    assert failed["isError"] is True
    assert failed["structuredOutput"] == {"output": "failed", "truncated": False, "exitCode": 7}
    assert [(each["code"], each["message"]) for each in failed["diagnostics"]] == [
        ("exit_code", "Command exited with code 7")
    ]
    assert "".join(failed["streamed"]) == "failed"
    error, _, _ = await run_failing(tool, {"command": "sleep 2", "timeout": 0.01}, env)
    assert str(error) == "Command timed out after 0.01 seconds"


async def test_reports_the_spill_of_a_command_that_times_out(tmp_path: Path) -> None:
    env = TimeoutOutputEnv(cwd=str(tmp_path))
    error, _, reported = await run_failing(
        create_bash_tool(), {"command": "emit-output-then-time-out", "timeout": 0.05}, env
    )
    assert str(error) == "Command timed out after 0.05 seconds"
    match = re.match(r"^Full output: (.+)$", reported[0]["message"])
    assert match is not None
    full_output = get_or_throw(await env.read_text_file(match.group(1), context))
    assert "line-1\nline-2" in full_output
    assert f"line-{DEFAULT_MAX_LINES}\nline-{TRUNCATED_OUTPUT_LINES}" in full_output


async def test_prepares_command_cwd_and_an_explicit_environment_with_the_calls_api(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path), shell_env={"PI_BASH_PREPARE_INHERITED": "inherited"})
    get_or_throw(await env.create_dir("workspace", None, context))
    workspace = f"{env.cwd}/workspace"
    controller, abort_context = aborting_context()
    received: dict[str, Any] = {}

    async def prepare(execution: Any, api: Any, call_context: Context) -> None:
        received["env"] = api.env
        received["signal"] = call_context.abort_signal
        execution["cwd"] = workspace
        execution["env"] = {"PI_BASH_PREPARE_EXPLICIT": "explicit"}
        execution["inheritEnv"] = False
        execution["command"] += (
            '\n: > prepared-cwd\nprintf \'%s:%s:%s\' "$prefix" "${PI_BASH_PREPARE_INHERITED-}" '
            '"$PI_BASH_PREPARE_EXPLICIT"'
        )
        # Git Bash on Windows reports $PWD as an MSYS path, so only POSIX compares it.
        if sys.platform != "win32":
            execution["command"] += "\nprintf ':%s' \"$PWD\""

    tool = create_bash_tool(command_prefix="prefix=ready", prepare=prepare)
    result = await run(tool, {"command": ":"}, env, abort_context)
    assert received["env"] is env
    assert received["signal"] is controller.signal
    pwd = "" if sys.platform == "win32" else f":{get_or_throw(await env.canonical_path(workspace, context))}"
    assert "".join(result["streamed"]) == f"ready::explicit{pwd}"
    assert get_or_throw(await env.exists(f"{workspace}/prepared-cwd", context)) is True


async def test_supports_command_prefixes(tmp_path: Path) -> None:
    result = await run(
        create_bash_tool(command_prefix="value=hello"),
        {"command": "printf $value"},
        LocalExecutionEnv(cwd=str(tmp_path)),
    )
    assert "".join(result["streamed"]) == "hello"


async def test_streams_every_byte_and_spills_complete_output_beyond_the_default_limits(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = await run(
        create_bash_tool(), {"command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done"}, env
    )
    expected = "".join(f"line-{index + 1}\n" for index in range(3000))
    assert "".join(result["streamed"]) == expected
    match = re.match(r"^Full output: (.+)$", result["reported"][0]["message"])
    assert match is not None
    assert get_or_throw(await env.read_text_file(match.group(1), context)) == expected


async def test_does_not_spill_output_within_the_limits(tmp_path: Path) -> None:
    result = await run(create_bash_tool(), {"command": "printf small"}, LocalExecutionEnv(cwd=str(tmp_path)))
    assert result["reported"] == []
