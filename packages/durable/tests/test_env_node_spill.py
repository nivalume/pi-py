"""Port of ``test/env-node-spill.test.ts``.

Upstream mocks ``fs.createWriteStream`` into a slow stream with a one-byte high-water mark; here the spill's
high-water mark and its blocking write are patched the same way.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import pytest

import pi_durable.env.local as local
from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_durable.env import ShellOutputInfo, get_or_throw
from pi_durable.env.local import LocalExecutionEnv

# Longer than the shell's post-exit stdio grace period plus the descendant's delayed write.
SPILL_WRITE_DELAY_S = 0.6


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell background jobs")
async def test_keeps_inherited_stdio_open_past_the_exit_grace_period_while_a_spill_write_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[int] = []
    pauses: list[int] = []
    original_write = local._write_spill
    original_pause = local._Execution.pause_output

    def slow_write(file: BinaryIO, data: bytes) -> None:
        writes.append(len(data))
        time.sleep(SPILL_WRITE_DELAY_S)
        original_write(file, data)

    def counting_pause(self: Any) -> None:
        pauses.append(1)
        original_pause(self)

    # Make the spill deterministically slow and immediately backpressured.
    monkeypatch.setattr(local, "SPILL_HIGH_WATER_MARK", 1)
    monkeypatch.setattr(local, "_write_spill", slow_write)
    monkeypatch.setattr(local._Execution, "pause_output", counting_pause)
    env = LocalExecutionEnv(cwd=str(tmp_path))
    # The shell exits after the first chunk crosses the spill threshold and backpressures the spill. A
    # background descendant retains stdout and writes after the post-exit grace period. Without the
    # pending-spill guard, settlement closes stdout before that descendant output is read.
    command = "printf '%020d' 0 | tr 0 a; (sleep 0.2; printf '%01000d' 0 | tr 0 b) &"
    output: list[str] = []

    def on_output(text: str, _context: Context, _info: ShellOutputInfo) -> None:
        output.append(text)

    result = get_or_throw(
        await env.exec(
            command, {"spill": {"afterBytes": 10, "afterLines": 10}, "onOutput": on_output}, BACKGROUND_CONTEXT
        )
    )
    assert len(writes) > 0
    assert len(pauses) > 0
    assert len("".join(output)) == 1020
    spill_path = result.get("spillPath")
    assert spill_path is not None
    try:
        assert get_or_throw(await env.read_text_file(spill_path, BACKGROUND_CONTEXT)) == f"{'a' * 20}{'b' * 1000}"
    finally:
        shutil.rmtree(os.path.dirname(spill_path), ignore_errors=True)
