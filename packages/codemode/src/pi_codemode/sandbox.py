"""``CodemodeSandbox``: runs JavaScript in QuickJS, compiled to WebAssembly and run by wasmtime.

This is upstream's sandbox: the same QuickJS build (``quickjs-wasi`` 3.6.2), the same prelude, the same results.
Importing this module needs the ``codemode`` extra (``pip install 'pi-ai-python[codemode]'``); without it the import
raises an ``ImportError`` that says so. The rest of pi_codemode (types, source parsing, declarations,
:class:`~pi_codemode.CodemodeSandboxBase`) works without wasmtime.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from .runtime.execution import DEFAULT_TIMEOUT_MS, CodemodeSandboxBase
from .runtime.host import quickjs_launcher
from .runtime.quickjs import MAX_STACK_SIZE
from .types import CodemodeTool

__all__ = ["CodemodeSandbox"]


class CodemodeSandbox(CodemodeSandboxBase):
    """Runs each execution, a JavaScript script, in a fresh QuickJS VM on a thread of its own.

    Each VM is its own wasm instance, so runs share nothing, and a VM that traps or runs out of memory fails only its
    own execution. wasmtime releases the GIL while a script computes, so the host's event loop keeps going; the
    deadline, an abort and :meth:`close` interrupt a computing script at once. The wasm module is compiled once per
    process, on first use (a few hundred milliseconds, off the event loop).
    """

    def __init__(
        self,
        *,
        tools: Iterable[CodemodeTool] = (),
        globals: Iterable[CodemodeTool] = (),
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        memory_limit_bytes: int | None = None,
        wasm_path: str | os.PathLike[str] | None = None,
        max_stack_size: int = MAX_STACK_SIZE,
        interrupt_budget: int | None = None,
    ) -> None:
        """
        Args:
            tools, globals, timeout_ms, memory_limit_bytes: See :class:`~pi_codemode.CodemodeSandboxBase`.
            wasm_path: ``quickjs.wasm`` to load. Default: the bundled ``quickjs-wasi`` 3.6.2 build.
            max_stack_size: QuickJS's stack limit in bytes, at most the default 512 KiB; deeper recursion throws a
                catchable ``RangeError``.
            interrupt_budget: Ends a script that computes for more than this many interrupt polls (QuickJS polls
                about every 10,000 opcodes) with ``kind="timeout"``. Default: none, only the deadline.
        """
        if not 0 <= max_stack_size <= MAX_STACK_SIZE:
            raise ValueError(f"max_stack_size must be an integer between 0 and {MAX_STACK_SIZE}")
        super().__init__(
            quickjs_launcher(wasm_path=wasm_path, max_stack_size=max_stack_size, interrupt_budget=interrupt_budget),
            language="javascript",
            tools=tools,
            globals=globals,
            timeout_ms=timeout_ms,
            memory_limit_bytes=memory_limit_bytes,
        )
