"""Port of ``test/env-node.test.ts`` (``NodeExecutionEnv`` is ``LocalExecutionEnv``).

Node scripts become Python scripts run with ``sys.executable``. Node ``FileHandle`` prototype spies become
patches of the module's ``_fsync`` and ``_pread`` helpers. The Windows-only case "settles after the shell
exits when a detached descendant retains inherited stdio" runs on every platform here, since the post-exit
grace period it checks is platform independent.
"""

from __future__ import annotations

import asyncio
import errno
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import pi_durable.env.local as local
from pi_ai.abort import AbortController
from pi_chord.context import BACKGROUND_CONTEXT, Context, with_abort_signal
from pi_durable.env import FileError, ShellExecOptions, ShellOutputInfo, get_or_throw
from pi_durable.env.local import LocalExecutionEnv

context = BACKGROUND_CONTEXT
windows = sys.platform == "win32"
python = json.dumps(sys.executable)


def aborted_context() -> Context:
    controller = AbortController()
    controller.abort()
    return with_abort_signal(controller.signal, BACKGROUND_CONTEXT)


async def wait_for(condition: Any, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not await condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"Condition not met within {timeout}s")
        await asyncio.sleep(0.01)


async def collect_shell_output(
    env: LocalExecutionEnv, command: str, options: ShellExecOptions | None, exec_context: Context
) -> tuple[Any, str]:
    output: list[str] = []

    def on_output(text: str, _context: Context, _info: ShellOutputInfo) -> None:
        output.append(text)

    result = await env.exec(command, {**(options or {}), "onOutput": on_output}, exec_context)
    return result, "".join(output)


def bash_quoted(value: str) -> str:
    return "'" + value.replace("\\", "/").replace("'", "'\"'\"'") + "'"


# ─── filesystem ──────────────────────────────────────────────────────────────


async def test_reads_writes_lists_and_removes_files_and_directories(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    assert get_or_throw(await env.absolute_path("nested/child", context)) == os.path.join(root, "nested", "child")
    assert get_or_throw(await env.join_path([root, "nested", "child"], context)) == os.path.join(
        root, "nested", "child"
    )
    get_or_throw(await env.create_dir("nested/child", None, context))
    get_or_throw(await env.write_file("nested/child/file.txt", "hel", context))
    get_or_throw(await env.append_file("nested/child/file.txt", "lo", context))
    assert get_or_throw(await env.read_text_file("nested/child/file.txt", context)) == "hello"
    assert get_or_throw(await env.read_text_lines("nested/child/file.txt", {"maxLines": 1}, context)) == ["hello"]
    assert get_or_throw(await env.read_binary_file("nested/child/file.txt", context)).decode() == "hello"

    entries = get_or_throw(await env.list_dir("nested/child", context))
    assert len(entries) == 1
    assert {key: entries[0][key] for key in ("name", "path", "kind", "size")} == {  # type: ignore[literal-required]
        "name": "file.txt",
        "path": os.path.join(root, "nested", "child", "file.txt"),
        "kind": "file",
        "size": 5,
    }
    assert isinstance(entries[0]["mtimeMs"], float)

    assert get_or_throw(await env.exists("nested/child/file.txt", context)) is True
    get_or_throw(await env.remove("nested/child/file.txt", None, context))
    assert get_or_throw(await env.exists("nested/child/file.txt", context)) is False


async def test_expands_home_relative_paths_and_file_urls(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    assert get_or_throw(await env.absolute_path("~/pi-node-env-test", context)) == os.path.join(
        os.path.expanduser("~"), "pi-node-env-test"
    )
    file_path = tmp_path / "file with spaces.txt"
    assert get_or_throw(await env.absolute_path(file_path.as_uri(), context)) == str(file_path)


async def test_returns_file_info_for_files_directories_and_symlinks_without_following_symlinks(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("dir", {"recursive": True}, context))
    get_or_throw(await env.write_file("dir/file.txt", "hello", context))
    os.symlink(os.path.join(root, "dir", "file.txt"), os.path.join(root, "file-link"))
    os.symlink(os.path.join(root, "dir"), os.path.join(root, "dir-link"))

    info = get_or_throw(await env.file_info("dir", context))
    assert (info["name"], info["path"], info["kind"]) == ("dir", os.path.join(root, "dir"), "directory")
    info = get_or_throw(await env.file_info("dir/file.txt", context))
    assert (info["name"], info["path"], info["kind"], info["size"]) == (
        "file.txt",
        os.path.join(root, "dir", "file.txt"),
        "file",
        5,
    )
    info = get_or_throw(await env.file_info("file-link", context))
    assert (info["name"], info["path"], info["kind"]) == ("file-link", os.path.join(root, "file-link"), "symlink")
    info = get_or_throw(await env.file_info("dir-link", context))
    assert (info["name"], info["path"], info["kind"]) == ("dir-link", os.path.join(root, "dir-link"), "symlink")
    assert get_or_throw(await env.canonical_path("file-link", context)) == os.path.realpath(
        os.path.join(root, "dir", "file.txt")
    )


async def test_lists_symlinks_as_symlinks(tmp_path: Path) -> None:
    root = str(tmp_path)
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("target.txt", "hello", context))
    os.symlink(os.path.join(root, "target.txt"), os.path.join(root, "link.txt"))
    entries = get_or_throw(await env.list_dir(".", context))
    assert sorted((entry["name"], entry["kind"]) for entry in entries) == [
        ("link.txt", "symlink"),
        ("target.txt", "file"),
    ]


async def test_stops_reading_text_lines_at_the_requested_limit(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "one\ntwo\nthree", context))
    assert get_or_throw(await env.read_text_lines("file.txt", {"maxLines": 1}, context)) == ["one"]


async def test_returns_file_error_for_missing_paths_and_keeps_exists_false_for_missing_paths(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    info = await env.file_info("missing.txt", context)
    assert info["ok"] is False
    assert isinstance(info["error"], FileError)
    assert type(info["error"]).__name__ == "FileError"
    assert info["error"].code == "not_found"
    assert info["error"].path == str(tmp_path / "missing.txt")
    assert get_or_throw(await env.exists("missing.txt", context)) is False


async def test_returns_file_error_for_listing_non_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "hello", context))
    result = await env.list_dir("file.txt", context)
    assert result["ok"] is False
    assert isinstance(result["error"], FileError)
    assert result["error"].code == "not_directory"


async def test_appends_to_new_files_and_creates_parent_directories(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.append_file("new/nested/file.txt", "a", context))
    get_or_throw(await env.append_file("new/nested/file.txt", "b", context))
    assert get_or_throw(await env.read_text_file("new/nested/file.txt", context)) == "ab"


async def test_atomically_renames_a_file_and_replaces_the_destination(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("source.txt", "new", context))
    get_or_throw(await env.write_file("destination.txt", "old", context))
    get_or_throw(await env.rename_file("source.txt", "destination.txt", context))
    assert get_or_throw(await env.exists("source.txt", context)) is False
    assert get_or_throw(await env.read_text_file("destination.txt", context)) == "new"


async def test_reports_the_source_path_when_rename_fails_because_the_source_is_missing(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("destination.txt", "unchanged", context))
    result = await env.rename_file("missing-source.txt", "destination.txt", context)
    assert result["ok"] is False
    assert result["error"].code == "not_found"
    assert result["error"].path == str(tmp_path / "missing-source.txt")
    assert get_or_throw(await env.read_text_file("destination.txt", context)) == "unchanged"


async def test_creates_temporary_directories_and_files(tmp_path: Path) -> None:
    import shutil

    env = LocalExecutionEnv(cwd=str(tmp_path))
    temp_dir = get_or_throw(await env.create_temp_dir("node-env-test-", context))
    try:
        assert os.path.isdir(temp_dir)
        assert os.path.basename(temp_dir).startswith("node-env-test-")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_file = get_or_throw(await env.create_temp_file({"prefix": "prefix-", "suffix": ".txt"}, context))
    try:
        assert os.path.isfile(temp_file)
        assert temp_file.endswith(".txt")
        assert os.path.basename(temp_file).startswith("prefix-")
        assert get_or_throw(await env.read_text_file(temp_file, context)) == ""
    finally:
        shutil.rmtree(os.path.dirname(temp_file), ignore_errors=True)


async def test_honors_create_dir_recursive_false_and_remove_recursive_force_options(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    created = await env.create_dir("missing/child", {"recursive": False}, context)
    assert created["ok"] is False and created["error"].code == "not_found"

    get_or_throw(await env.write_file("dir/child/file.txt", "hello", context))
    assert (await env.remove("dir", {"recursive": False}, context))["ok"] is False
    get_or_throw(await env.remove("dir", {"recursive": True}, context))
    assert get_or_throw(await env.exists("dir", context)) is False

    assert (await env.remove("missing", {"force": False}, context))["ok"] is False
    get_or_throw(await env.remove("missing", {"force": True}, context))


async def test_returns_aborted_results_without_side_effects_for_pre_aborted_file_operations(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "hello", context))
    aborted = aborted_context()
    results = await asyncio.gather(
        env.read_text_file("file.txt", aborted),
        env.read_text_lines("file.txt", None, aborted),
        env.read_binary_file("file.txt", aborted),
        env.open_text_line_reader("file.txt", aborted),
        env.write_file("other.txt", "hello", aborted),
        env.append_file("file.txt", " world", aborted),
        env.truncate_file("file.txt", 1, aborted),
        env.flush_file("file.txt", aborted),
        env.rename_file("file.txt", "renamed.txt", aborted),
        env.file_info("file.txt", aborted),
        env.list_dir(".", aborted),
        env.canonical_path("file.txt", aborted),
        env.exists("file.txt", aborted),
        env.create_dir("dir", None, aborted),
        env.remove("file.txt", None, aborted),
        env.create_temp_dir(None, aborted),
        env.create_temp_file(None, aborted),
    )
    for result in results:
        assert result["ok"] is False and result["error"].code == "aborted"
    assert get_or_throw(await env.read_text_file("file.txt", context)) == "hello"
    assert [entry["name"] for entry in get_or_throw(await env.list_dir(".", context))] == ["file.txt"]


async def test_truncates_and_extends_files_to_exact_byte_sizes(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.bin", bytes([0x61, 0xC3, 0xA9, 0x0A, 0x62, 0x0A]), context))
    # Truncation is byte-exact even when the boundary splits a UTF-8 sequence.
    get_or_throw(await env.truncate_file("file.bin", 2, context))
    assert list(get_or_throw(await env.read_binary_file("file.bin", context))) == [0x61, 0xC3]
    get_or_throw(await env.truncate_file("file.bin", 4, context))
    assert list(get_or_throw(await env.read_binary_file("file.bin", context))) == [0x61, 0xC3, 0, 0]
    get_or_throw(await env.truncate_file("file.bin", 0, context))
    assert get_or_throw(await env.file_info("file.bin", context))["size"] == 0


async def test_rejects_invalid_truncation_sizes_and_never_creates_missing_files(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "hello", context))
    for size in (-1, 1.5, math.nan, math.inf, 2**53):
        result = await env.truncate_file("file.txt", size, context)  # type: ignore[arg-type]
        assert result["ok"] is False
        assert (result["error"].code, result["error"].path) == ("invalid", str(tmp_path / "file.txt"))
    assert get_or_throw(await env.read_text_file("file.txt", context)) == "hello"
    result = await env.truncate_file("missing.txt", 0, context)
    assert result["ok"] is False
    assert (result["error"].code, result["error"].path) == ("not_found", str(tmp_path / "missing.txt"))
    assert get_or_throw(await env.exists("missing.txt", context)) is False


async def test_flushes_existing_files_without_changing_content_and_reports_missing_paths(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "durable", context))
    get_or_throw(await env.flush_file("file.txt", context))
    assert get_or_throw(await env.read_text_file("file.txt", context)) == "durable"
    missing = await env.flush_file("missing.txt", context)
    assert missing["ok"] is False
    assert (missing["error"].code, missing["error"].path) == ("not_found", str(tmp_path / "missing.txt"))
    assert get_or_throw(await env.exists("missing.txt", context)) is False
    get_or_throw(await env.create_dir("dir", None, context))
    directory = await env.flush_file("dir", context)
    assert directory["ok"] is False and directory["error"].code == "is_directory"


async def test_syncs_through_an_opened_handle_returns_sync_failures_and_always_closes_the_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("file.txt", "durable", context))
    synced: list[int] = []
    failure = OSError(errno.EIO, "i/o error, fsync")

    def sync(fd: int) -> None:
        synced.append(fd)
        if len(synced) == 2:
            raise failure
        os.fsync(fd)

    monkeypatch.setattr(local, "_fsync", sync)
    get_or_throw(await env.flush_file("file.txt", context))
    assert len(synced) == 1
    with pytest.raises(OSError):
        os.fstat(synced[0])

    result = await env.flush_file("file.txt", context)
    assert result["ok"] is False
    assert (result["error"].code, result["error"].path, result["error"].message) == (
        "unknown",
        str(tmp_path / "file.txt"),
        str(failure),
    )
    assert len(synced) == 2
    with pytest.raises(OSError):
        os.fstat(synced[1])


async def test_cleanup_is_best_effort(tmp_path: Path) -> None:
    assert await LocalExecutionEnv(cwd=str(tmp_path)).cleanup(context) is None


# ─── text line reader ────────────────────────────────────────────────────────


async def read_all_lines(env: LocalExecutionEnv, path: str) -> list[Any]:
    reader = get_or_throw(await env.open_text_line_reader(path, context))
    try:
        lines = []
        while (line := get_or_throw(await reader.read_line(context))) is not None:
            lines.append(line)
        assert get_or_throw(await reader.read_line(context)) is None
        return lines
    finally:
        await reader.close(context)


async def test_reports_whether_each_line_was_newline_terminated_and_preserves_carriage_returns(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("lines.txt", "one\r\n\ntwo\npartial", context))
    assert await read_all_lines(env, "lines.txt") == [
        {"text": "one\r", "terminated": True},
        {"text": "", "terminated": True},
        {"text": "two", "terminated": True},
        {"text": "partial", "terminated": False},
    ]


async def test_returns_no_lines_for_an_empty_file_and_one_terminated_empty_line_for_a_lone_newline(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("empty.txt", "", context))
    get_or_throw(await env.write_file("newline.txt", "\n", context))
    assert await read_all_lines(env, "empty.txt") == []
    assert await read_all_lines(env, "newline.txt") == [{"text": "", "terminated": True}]


async def test_decodes_multi_byte_characters_split_across_read_chunks_and_lines_longer_than_one_chunk(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    # The reader uses 64 KiB chunks; place a four-byte character across the first boundary.
    first = f"{'a' * (64 * 1024 - 2)}😀tail"
    second = "é" * 100_000
    get_or_throw(await env.write_file("large.txt", f"{first}\n{second}", context))
    assert await read_all_lines(env, "large.txt") == [
        {"text": first, "terminated": True},
        {"text": second, "terminated": False},
    ]


async def test_rejects_a_read_cancelled_while_pending_without_consuming_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("lines.txt", "one\ntwo\n", context))
    started = threading.Event()
    gate = threading.Event()
    original = local._pread
    calls = [0]

    def gated(fd: int, size: int, offset: int) -> bytes:
        calls[0] += 1
        if calls[0] == 1:
            started.set()
            gate.wait(5)
        return original(fd, size, offset)

    monkeypatch.setattr(local, "_pread", gated)
    reader = get_or_throw(await env.open_text_line_reader("lines.txt", context))
    try:
        controller = AbortController()
        pending = asyncio.ensure_future(reader.read_line(with_abort_signal(controller.signal, BACKGROUND_CONTEXT)))
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        controller.abort()
        gate.set()
        result = await pending
        assert result["ok"] is False and result["error"].code == "aborted"
        assert get_or_throw(await reader.read_line(context)) == {"text": "one", "terminated": True}
        assert get_or_throw(await reader.read_line(context)) == {"text": "two", "terminated": True}
        assert get_or_throw(await reader.read_line(context)) is None
    finally:
        gate.set()
        await reader.close(context)


async def test_rejects_reads_after_close_and_closes_idempotently(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    get_or_throw(await env.write_file("lines.txt", "one\n", context))
    reader = get_or_throw(await env.open_text_line_reader("lines.txt", context))
    await reader.close(context)
    assert await reader.close(context) is None
    result = await reader.read_line(context)
    assert result["ok"] is False
    assert (result["error"].code, result["error"].path) == ("invalid", str(tmp_path / "lines.txt"))


async def test_reports_missing_files_when_opening_a_reader(tmp_path: Path) -> None:
    result = await LocalExecutionEnv(cwd=str(tmp_path)).open_text_line_reader("missing.txt", context)
    assert result["ok"] is False
    assert (result["error"].code, result["error"].path) == ("not_found", str(tmp_path / "missing.txt"))


# ─── shell ───────────────────────────────────────────────────────────────────


async def test_executes_commands_in_cwd_with_env_overrides(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result, output = await collect_shell_output(
        env,
        'printf \'%s\' "$NODE_ENV_TEST" > cwd-marker.txt; printf \'%s:%s\' "$PWD" "$NODE_ENV_TEST"',
        {"env": {"NODE_ENV_TEST": "ok"}},
        context,
    )
    assert get_or_throw(result)["exitCode"] == 0
    assert (tmp_path / "cwd-marker.txt").read_text() == "ok"
    # Git Bash on Windows reports $PWD as an MSYS path (/tmp/...), not the Windows path.
    if not windows:
        assert output == f"{os.path.realpath(tmp_path)}:ok"


@pytest.mark.parametrize(
    ("overrides", "expected_session_file"),
    [
        (None, "x:/stale/parent.jsonl"),
        ({"PI_SESSION_FILE": ""}, "x:"),
        ({"PI_SESSION_FILE": "/sessions/current.jsonl"}, "x:/sessions/current.jsonl"),
    ],
    ids=[
        "a missing override preserves the base value",
        "an empty override shadows the base value",
        "a string override replaces the base value",
    ],
)
async def test_applies_string_shell_environment_overrides(
    tmp_path: Path, overrides: dict[str, str] | None, expected_session_file: str
) -> None:
    env = LocalExecutionEnv(
        cwd=str(tmp_path),
        shell_env={
            "PI_SESSION_FILE": "/stale/parent.jsonl",
            "PI_CODING_AGENT": "true",
            "PI_NODE_ENV_PRESERVED_TEST": "preserved",
        },
    )
    options: ShellExecOptions = {} if overrides is None else {"env": overrides}
    result, output = await collect_shell_output(
        env,
        'printf \'%s:%s|%s|%s\' "${PI_SESSION_FILE+x}" "${PI_SESSION_FILE-}" "$PI_CODING_AGENT" "$PI_NODE_ENV_PRESERVED_TEST"',
        options,
        context,
    )
    get_or_throw(result)
    assert output == f"{expected_session_file}|true|preserved"


async def test_can_replace_rather_than_inherit_the_default_shell_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited_key = "PI_NODE_ENV_INHERITED_TEST"
    configured_key = "PI_NODE_ENV_CONFIGURED_TEST"
    explicit_key = "PI_NODE_ENV_EXPLICIT_TEST"
    monkeypatch.setenv(inherited_key, "host")
    env = LocalExecutionEnv(cwd=str(tmp_path), shell_env={configured_key: "configured"})
    result, output = await collect_shell_output(
        env,
        f'printf \'%s:%s:%s\' "${{{inherited_key}-}}" "${{{configured_key}-}}" "${{{explicit_key}-}}"',
        {"inheritEnv": False, "env": {explicit_key: "explicit"}},
        context,
    )
    get_or_throw(result)
    assert output == "::explicit"


async def test_uses_stdin_command_transport_for_legacy_wsl_bash_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if windows:
        return
    root = str(tmp_path)
    shell_path = "C:\\Windows\\System32\\bash.exe"
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(
        await env.write_file(shell_path, '#!/bin/sh\nprintf \'args:%s\\n\' "$*" >&2\nexec /bin/bash "$@"\n', context)
    )
    os.chmod(os.path.join(root, shell_path), 0o755)
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", f"{root}{os.pathsep}{os.environ.get('PATH', '')}")
    wsl_env = LocalExecutionEnv(cwd=root, shell_path=shell_path)
    result, output = await collect_shell_output(wsl_env, "name='World'; echo \"Hello, ${name}!\"", None, context)
    assert "Hello, World!" in output
    assert "args:-s" in output
    assert get_or_throw(result)["exitCode"] == 0


async def test_settles_after_the_shell_exits_when_a_detached_descendant_retains_inherited_stdio(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
        "open(sys.argv[1], 'w').write(str(child.pid)); "
        "print('child-exiting', flush=True)"
    )
    env = LocalExecutionEnv(cwd=str(tmp_path))
    controller = AbortController()
    try:
        result, output = await asyncio.wait_for(
            collect_shell_output(
                env,
                f"{python} -c {bash_quoted(script)} {bash_quoted(str(pid_file))}",
                None,
                with_abort_signal(controller.signal, BACKGROUND_CONTEXT),
            ),
            3,
        )
        get_or_throw(result)
        assert "child-exiting" in output
    finally:
        controller.abort()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


async def test_cleanup_terminates_active_shell_processes(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    execution = asyncio.ensure_future(env.exec("touch started; sleep 60", None, context))
    try:

        async def started() -> bool:
            return get_or_throw(await env.exists("started", context))

        await wait_for(started, 10)
    finally:
        await env.cleanup(context)
    result = await asyncio.wait_for(execution, 3)
    assert result["ok"] is True


async def test_streams_combined_stdout_and_stderr(tmp_path: Path) -> None:
    result, output = await collect_shell_output(
        LocalExecutionEnv(cwd=str(tmp_path)), "printf out; printf err >&2", None, context
    )
    assert get_or_throw(result) == {"exitCode": 0}
    assert "out" in output
    assert "err" in output


async def test_decodes_utf8_split_across_raw_process_chunks(tmp_path: Path) -> None:
    script = (
        "import sys, time; b = '😀'.encode(); sys.stdout.buffer.write(b[:2]); sys.stdout.flush(); "
        "time.sleep(0.05); sys.stdout.buffer.write(b[2:]); sys.stdout.flush()"
    )
    _result, output = await collect_shell_output(
        LocalExecutionEnv(cwd=str(tmp_path)), f"{python} -c {bash_quoted(script)}", None, context
    )
    assert output == "😀"


async def test_reports_a_missing_working_directory_before_spawning(tmp_path: Path) -> None:
    result = await LocalExecutionEnv(cwd=str(tmp_path / "missing")).exec("printf ok", None, context)
    assert result["ok"] is False
    assert result["error"].code == "spawn_error"
    assert "Working directory does not exist" in result["error"].message


async def test_returns_non_zero_command_exit_codes_as_successful_execution_results(tmp_path: Path) -> None:
    assert get_or_throw(await LocalExecutionEnv(cwd=str(tmp_path)).exec("exit 7", None, context)) == {"exitCode": 7}


async def test_maps_signal_killed_processes_to_a_non_zero_exit_code(tmp_path: Path) -> None:
    # Regression test for https://github.com/earendil-works/pi/issues/8992
    if windows:
        return
    result = get_or_throw(await LocalExecutionEnv(cwd=str(tmp_path)).exec("kill -9 $$", None, context))
    assert result["exitCode"] == 128 + 9


async def test_returns_timeout_errors_for_commands_exceeding_the_timeout(tmp_path: Path) -> None:
    result = await LocalExecutionEnv(cwd=str(tmp_path)).exec("sleep 5", {"timeout": 0.01}, context)
    assert result["ok"] is False and result["error"].code == "timeout"


async def test_rejects_invalid_timeouts_before_spawning(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    for timeout in (0, -1, math.nan, math.inf, 2_147_484):
        result = await env.exec("touch spawned", {"timeout": timeout}, context)
        assert result["ok"] is False
        assert result["error"].code == "timeout"
        assert "Invalid timeout" in result["error"].message
    assert get_or_throw(await env.exists("spawned", context)) is False


async def test_returns_callback_errors_from_exec_stream_handlers(tmp_path: Path) -> None:
    def on_output(text: str, _context: Context, _info: ShellOutputInfo) -> None:
        raise RuntimeError("callback failed")

    result = await LocalExecutionEnv(cwd=str(tmp_path)).exec("printf out", {"onOutput": on_output}, context)
    assert result["ok"] is False
    assert (result["error"].code, result["error"].message) == ("callback_error", "callback failed")


async def test_returns_shell_unavailable_and_spawn_errors(tmp_path: Path) -> None:
    missing_shell = await LocalExecutionEnv(cwd=str(tmp_path), shell_path=str(tmp_path / "missing-shell")).exec(
        "printf ok", None, context
    )
    assert missing_shell["ok"] is False and missing_shell["error"].code == "shell_unavailable"

    shell_path = str(tmp_path / "not-executable-shell")
    get_or_throw(await LocalExecutionEnv(cwd=str(tmp_path)).write_file(shell_path, "not executable", context))
    spawn_error = await LocalExecutionEnv(cwd=str(tmp_path), shell_path=shell_path).exec("printf ok", None, context)
    assert spawn_error["ok"] is False and spawn_error["error"].code == "spawn_error"


async def test_returns_an_aborted_result_for_pre_aborted_and_aborted_commands(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = await env.exec("touch spawned", None, aborted_context())
    assert result["ok"] is False and result["error"].code == "aborted"
    assert get_or_throw(await env.exists("spawned", context)) is False

    controller = AbortController()
    pending = asyncio.ensure_future(env.exec("sleep 5", None, with_abort_signal(controller.signal, BACKGROUND_CONTEXT)))
    await asyncio.sleep(0)
    controller.abort()
    result = await pending
    assert result["ok"] is False and result["error"].code == "aborted"


async def test_ignores_taskkill_spawn_errors_during_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if windows:
        return
    pid_file = tmp_path / "shell.pid"
    controller = AbortController()
    env = LocalExecutionEnv(cwd=str(tmp_path), shell_path="/bin/bash")
    monkeypatch.setenv("SystemRoot", "/definitely/missing/windows")
    monkeypatch.setattr(local, "_is_windows", lambda: True)
    pid: int | None = None
    try:
        execution = asyncio.ensure_future(
            env.exec(
                f"echo $$ > {bash_quoted(str(pid_file))}; exec sleep 60",
                None,
                with_abort_signal(controller.signal, BACKGROUND_CONTEXT),
            )
        )
        for _attempt in range(100):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()

        controller.abort()
        await asyncio.sleep(0)
        pid = int(pid_file.read_text())
        os.kill(pid, signal.SIGKILL)

        result = await execution
        assert result["ok"] is False and result["error"].code == "aborted"
    finally:
        if pid is None and pid_file.exists():
            pid = int(pid_file.read_text() or 0) or None
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


async def test_does_not_create_a_spill_before_output_crosses_its_thresholds(tmp_path: Path) -> None:
    result = get_or_throw(
        await LocalExecutionEnv(cwd=str(tmp_path)).exec(
            "printf short", {"spill": {"afterBytes": 100, "afterLines": 10}}, context
        )
    )
    assert "spillPath" not in result


def remove_spill(path: str) -> None:
    import shutil

    shutil.rmtree(os.path.dirname(path), ignore_errors=True)


async def test_preserves_exact_raw_bytes_in_the_spill_while_streaming_decoded_text(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    expected = bytes([0x66, 0x80, 0x00, 0x6F])
    script = f"import sys; sys.stdout.buffer.write(bytes({list(expected)}))"
    result = get_or_throw(
        await env.exec(f"{python} -c {bash_quoted(script)}", {"spill": {"afterBytes": 1, "afterLines": 10}}, context)
    )
    spill_path = result.get("spillPath")
    assert spill_path is not None
    try:
        assert get_or_throw(await env.read_binary_file(spill_path, context)) == expected
    finally:
        remove_spill(spill_path)


async def test_reports_the_spill_of_a_command_that_times_out(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result = await env.exec(
        "printf 12345678901234567890; sleep 5", {"timeout": 0.3, "spill": {"afterBytes": 10, "afterLines": 10}}, context
    )
    assert result["ok"] is False and result["error"].code == "timeout"
    spill_path = result["error"].spill_path
    assert isinstance(spill_path, str)
    try:
        assert get_or_throw(await env.read_text_file(spill_path, context)) == "12345678901234567890"
    finally:
        remove_spill(spill_path)


class FailingSpillExecutionEnv(LocalExecutionEnv):
    async def create_temp_file(self, options: Any, create_context: Context) -> Any:
        if options is not None and options.get("prefix") == "pi-output-":
            return {"ok": True, "value": os.path.join(self.cwd, "missing", "spill.log")}
        return await super().create_temp_file(options, create_context)


async def test_fails_rather_than_silently_losing_a_requested_spill(tmp_path: Path) -> None:
    result = await FailingSpillExecutionEnv(cwd=str(tmp_path)).exec(
        "printf 12345678901234567890", {"spill": {"afterBytes": 10, "afterLines": 10}}, context
    )
    assert result["ok"] is False
    assert result["error"].code == "unknown"
    assert "Failed to preserve complete shell output" in result["error"].message


async def test_preserves_complete_large_output_in_the_spill(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    size = 500_000
    script = f"import sys; sys.stdout.write('x' * {size})"
    result = get_or_throw(
        await env.exec(f"{python} -c {bash_quoted(script)}", {"spill": {"afterBytes": 10, "afterLines": 10}}, context)
    )
    spill_path = result.get("spillPath")
    assert spill_path is not None
    try:
        assert len(get_or_throw(await env.read_text_file(spill_path, context))) == size
    finally:
        remove_spill(spill_path)


async def test_streams_every_line_and_spills_them_all_once_output_crosses_its_line_threshold(tmp_path: Path) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path))
    result, output = await collect_shell_output(
        env,
        "i=1; while [ $i -le 15000 ]; do echo line-$i; i=$((i+1)); done",
        {"spill": {"afterBytes": 1024 * 1024, "afterLines": 100}},
        context,
    )
    spill_path = get_or_throw(result).get("spillPath")
    assert output == "".join(f"line-{index + 1}\n" for index in range(15000))
    assert spill_path is not None
    try:
        spilled = get_or_throw(await env.read_text_lines(spill_path, None, context))
        assert len(spilled) == 15000
        assert spilled[-1] == "line-15000"
    finally:
        remove_spill(spill_path)
