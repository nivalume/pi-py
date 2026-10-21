"""pi_codemode: run model-written scripts whose only capability is calling injected tools.

Python port of ``@earendil-works/pi-codemode``: nested tool calls never enter the model's context, only the script's
output and return value do. Scripts are JavaScript by default, run by QuickJS as upstream's are; Python is the
alternative, run by Monty (``pydantic-monty``). A sandbox, and so a code mode tool, runs one of the two.

- :class:`CodemodeSandbox` runs JavaScript in QuickJS on wasmtime (needs the ``codemode`` extra).
- :class:`MontyCodemodeSandbox` runs Python in a Monty worker pool (needs the ``codemode-python`` extra).
- :class:`CodemodeSandboxBase` is the portable host side: tool table, call records, output,
  deadlines, cancellation, and deterministic runs (:meth:`CodemodeSandboxBase.start`).
- :func:`parse_codemode_source` splits the options line (``// @options:``, ``# @options:``) off model input.
- :func:`render_declarations` renders TypeScript (or Python) declarations of the tools for the model.
- :func:`create_codemode_tool` wraps a sandbox as a pi_agent ``AgentTool``.

Everything but the two sandboxes imports without wasmtime or Monty. They load on first access (as does
:func:`create_codemode_tool`, which imports pi_agent) and raise an ``ImportError`` naming the extra when their
runtime is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .declarations import (
    DEFAULT_INPUT_SCHEMA_MAX_CHARS,
    render_declarations,
    render_tool_output_type,
    render_tool_sample,
    render_tool_signature,
    schema_to_type,
)
from .identifier import is_codemode_identifier, to_codemode_identifier
from .language import CODEMODE_LANGUAGES, DEFAULT_CODEMODE_LANGUAGE, CodemodeLanguage, check_codemode_language
from .runtime.execution import DEFAULT_TIMEOUT_MS, RESERVED_GLOBALS, CodemodeRun, CodemodeSandboxBase
from .runtime.prelude import MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, MAX_STORE_TOTAL_CHARS, MAX_STORE_VALUE_CHARS
from .source import (
    CODEMODE_OPTIONS_PREFIX,
    CODEMODE_SOURCE_GRAMMAR,
    PYTHON_CODEMODE_OPTIONS_PREFIX,
    PYTHON_CODEMODE_SOURCE_GRAMMAR,
    CodemodeSourceError,
    CodemodeSourceOptions,
    ParsedCodemodeSource,
    codemode_options_prefix,
    codemode_source_grammar,
    parse_codemode_source,
)
from .types import (
    CodemodeCall,
    CodemodeCallStatus,
    CodemodeDeclaredGlobal,
    CodemodeDeclaredTool,
    CodemodeDuring,
    CodemodeError,
    CodemodeErrorKind,
    CodemodeErrorOutcome,
    CodemodeFailure,
    CodemodeHostCall,
    CodemodeImageOutput,
    CodemodeJsonSchema,
    CodemodeOkOutcome,
    CodemodeOutcome,
    CodemodeOutputItem,
    CodemodeResult,
    CodemodeStoreWrites,
    CodemodeSuccess,
    CodemodeTextOutput,
    CodemodeTool,
    CodemodeToolContext,
    CodemodeToolDeclaration,
    CodemodeToolExecute,
    CodemodeToolLike,
)

if TYPE_CHECKING:
    from .agent import create_codemode_tool
    from .monty import MontyCodemodeSandbox
    from .sandbox import CodemodeSandbox

__all__ = [
    # languages
    "CODEMODE_LANGUAGES",
    "DEFAULT_CODEMODE_LANGUAGE",
    "CodemodeLanguage",
    "check_codemode_language",
    # sandboxes
    "CodemodeSandbox",
    "MontyCodemodeSandbox",
    "CodemodeSandboxBase",
    "CodemodeRun",
    "DEFAULT_TIMEOUT_MS",
    "RESERVED_GLOBALS",
    "MAX_OUTPUT_CHARS",
    "MAX_OUTPUT_ITEMS",
    "MAX_STORE_TOTAL_CHARS",
    "MAX_STORE_VALUE_CHARS",
    # pi_agent
    "create_codemode_tool",
    # source
    "CODEMODE_OPTIONS_PREFIX",
    "CODEMODE_SOURCE_GRAMMAR",
    "PYTHON_CODEMODE_OPTIONS_PREFIX",
    "PYTHON_CODEMODE_SOURCE_GRAMMAR",
    "CodemodeSourceError",
    "CodemodeSourceOptions",
    "ParsedCodemodeSource",
    "codemode_options_prefix",
    "codemode_source_grammar",
    "parse_codemode_source",
    # declarations
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "render_declarations",
    "render_tool_output_type",
    "render_tool_sample",
    "render_tool_signature",
    "schema_to_type",
    # identifiers
    "is_codemode_identifier",
    "to_codemode_identifier",
    # types
    "CodemodeCall",
    "CodemodeCallStatus",
    "CodemodeDeclaredGlobal",
    "CodemodeDeclaredTool",
    "CodemodeDuring",
    "CodemodeError",
    "CodemodeErrorKind",
    "CodemodeErrorOutcome",
    "CodemodeFailure",
    "CodemodeHostCall",
    "CodemodeImageOutput",
    "CodemodeJsonSchema",
    "CodemodeOkOutcome",
    "CodemodeOutcome",
    "CodemodeOutputItem",
    "CodemodeResult",
    "CodemodeStoreWrites",
    "CodemodeSuccess",
    "CodemodeTextOutput",
    "CodemodeTool",
    "CodemodeToolContext",
    "CodemodeToolDeclaration",
    "CodemodeToolExecute",
    "CodemodeToolLike",
]


def __getattr__(name: str) -> Any:
    # The sandboxes load on first access, so the portable parts import without wasmtime or Monty.
    if name == "CodemodeSandbox":
        from .sandbox import CodemodeSandbox

        return CodemodeSandbox
    if name == "MontyCodemodeSandbox":
        from .monty import MontyCodemodeSandbox

        return MontyCodemodeSandbox
    if name == "create_codemode_tool":
        from .agent import create_codemode_tool

        return create_codemode_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
