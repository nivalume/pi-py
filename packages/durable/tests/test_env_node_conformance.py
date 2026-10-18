"""Port of ``test/env-node-conformance.test.ts``: the ``ExecutionEnv`` conformance cases on ``LocalExecutionEnv``."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable.env import WatchChange, get_or_throw
from pi_durable.env.local import LocalExecutionEnv
from pi_durable.testing import register_env_conformance

context = BACKGROUND_CONTEXT
windows = sys.platform == "win32"
program_files = os.environ.get("ProgramFiles", "C:\\Program Files")  # noqa: SIM112 - upstream spelling
git_bash = os.path.join(program_files, "Git", "bin", "bash.exe")


async def _with_env(use: Any) -> None:
    cwd = tempfile.mkdtemp(prefix="pi-durable-env-conformance-")
    try:
        await use(LocalExecutionEnv(cwd=cwd))
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


async def _with_polling_env(use: Any) -> None:
    cwd = tempfile.mkdtemp(prefix="pi-durable-env-conformance-")
    try:
        await use(LocalExecutionEnv(cwd=cwd, watch={"mode": "polling", "pollIntervalMs": 100}))
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


# Git Bash's `ln -s` copies instead of linking unless native symlinks are enabled.
_options: dict[str, Any] = {"shell": [git_bash, "-c"], "symlinks": False} if windows else {}
register_env_conformance(globals(), "LocalExecutionEnv conformance", _with_env, **_options)
register_env_conformance(globals(), "LocalExecutionEnv conformance with polling watches", _with_polling_env, **_options)


async def test_refuses_a_tree_over_the_directory_budget_and_stops_with_an_error_when_one_grows_past_it(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnv(cwd=str(tmp_path), watch={"maxDirectories": 3})
    get_or_throw(await env.create_dir("tree/a/b", None, context))
    get_or_throw(await env.create_dir("tree/c", None, context))
    refused = await env.watch([{"path": "tree", "recursive": True}], lambda change: None, context)
    assert refused["ok"] is False and refused["error"].code == "invalid"

    get_or_throw(await env.remove("tree/c", {"recursive": True}, context))
    changes: list[WatchChange] = []
    watcher = get_or_throw(await env.watch([{"path": "tree", "recursive": True}], changes.append, context))
    get_or_throw(await env.create_dir("tree/d", None, context))
    deadline = time.monotonic() + 3
    while not any("error" in change for change in changes) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert "error" in changes[-1] and changes[-1]["error"].code == "invalid"  # type: ignore[typeddict-item]
    await watcher.close(context)


async def test_reads_ranges_spanning_several_internal_chunks_exactly(tmp_path: Path) -> None:
    data = bytes((index * 31) % 251 for index in range(int(2.5 * 1024 * 1024)))
    (tmp_path / "big.bin").write_bytes(data)
    reader = get_or_throw(await LocalExecutionEnv(cwd=str(tmp_path)).open_binary_reader("big.bin", None, context))
    try:
        everything = get_or_throw(await reader.read(0, len(data) + 10, context))
        assert len(everything) == len(data)
        assert everything == data
        middle = get_or_throw(await reader.read(1024 * 1024 - 3, 7, context))
        assert middle == data[1024 * 1024 - 3 : 1024 * 1024 + 4]
    finally:
        await reader.close(context)


async def test_refuses_a_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    if windows:
        return
    os.mkfifo(tmp_path / "pipe")
    result = await LocalExecutionEnv(cwd=str(tmp_path)).open_binary_reader("pipe", None, context)
    assert result["ok"] is False and result["error"].code == "invalid"
