"""``CodemodeSandbox``: runs scripts in Monty, Pydantic's sandbox for AI-written Python.

Importing this module needs the ``codemode`` extra (``pip install 'pi-ai-python[codemode]'``);
without it the import raises an ``ImportError`` that says so. The rest of pi_codemode (types,
source parsing, declarations, :class:`~pi_codemode.CodemodeSandboxBase`) works without Monty.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .runtime.execution import DEFAULT_TIMEOUT_MS, CodemodeSandboxBase
from .runtime.monty import DEFAULT_MAX_RECURSION_DEPTH, monty_launcher
from .types import CodemodeTool

__all__ = ["CodemodeSandbox"]


class CodemodeSandbox(CodemodeSandboxBase):
    """Runs each execution in its own session of a Monty worker pool.

    A session is a dedicated ``monty`` worker subprocess, so a runaway script cannot block the
    host's event loop, and a worker that crashes (``MontyCrashedError``) fails only its own
    execution: the pool replaces it. Executions run in parallel up to the pool's
    ``max_processes``; later ones wait for a free worker.

    By default the sandbox owns its pool: it starts an ``AsyncMonty`` on first use and closes it
    in :meth:`close`. Pass an entered ``AsyncMonty`` (or ``AsyncMontyWebsocket``) as ``pool`` to
    share one between sandboxes; the sandbox then never closes it.
    """

    def __init__(
        self,
        *,
        tools: Iterable[CodemodeTool] = (),
        globals: Iterable[CodemodeTool] = (),
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        memory_limit_bytes: int | None = None,
        pool: Any = None,
        max_processes: int | None = None,
        binary_path: str | Path | None = None,
        max_recursion_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
        timezone: str | Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            tools, globals, timeout_ms, memory_limit_bytes: See :class:`~pi_codemode.CodemodeSandboxBase`.
            pool: An entered Monty pool to check sessions out of. Default: a pool of this sandbox's own.
            max_processes: Workers of the sandbox's own pool (default: the CPU count).
            binary_path: The ``monty`` binary of the sandbox's own pool (default: the one
                ``pydantic-monty-runtime`` installs).
            max_recursion_depth: Deeper recursion raises a catchable ``RecursionError``. Default 1000.
            timezone: The zone ``execute()`` runs in: an IANA name or ``{"offset_seconds": ..., "name": ...}``.
                Default: the host's. Deterministic runs always use UTC.
        """
        self._shared_pool = pool
        self._max_processes = max_processes
        self._binary_path = binary_path
        self._own_pool: Any = None
        self._pool_lock = asyncio.Lock()
        self._drivers: set[asyncio.Task[None]] = set()
        super().__init__(
            monty_launcher(
                self._pool,
                max_recursion_depth=max_recursion_depth,
                timezone=timezone,
                on_driver=self._track_driver,
            ),
            tools=tools,
            globals=globals,
            timeout_ms=timeout_ms,
            memory_limit_bytes=memory_limit_bytes,
        )

    def _track_driver(self, task: asyncio.Task[None]) -> None:
        self._drivers.add(task)
        task.add_done_callback(self._drivers.discard)

    async def _pool(self) -> Any:
        if self._shared_pool is not None:
            return self._shared_pool
        async with self._pool_lock:
            if self.closed:
                raise RuntimeError("Sandbox is closed")
            if self._own_pool is None:
                from pydantic_monty import AsyncMonty

                pool = AsyncMonty(max_processes=self._max_processes, binary_path=self._binary_path)
                await pool.__aenter__()
                self._own_pool = pool
            return self._own_pool

    async def close(self) -> None:
        """Aborts in-flight executions, waits for their sessions to be released, and closes the sandbox's own pool."""
        await super().close()
        if self._drivers:
            await asyncio.gather(*list(self._drivers), return_exceptions=True)
        async with self._pool_lock:
            pool, self._own_pool = self._own_pool, None
        if pool is not None:
            await pool.__aexit__(None, None, None)
