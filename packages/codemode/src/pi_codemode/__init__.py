"""pi_codemode: run model-written Python whose only capability is calling injected tools.

Python port of ``@earendil-works/pi-codemode``. Scripts are Python, executed by Monty
(``pydantic-monty``), Pydantic's sandbox for AI-written Python: nested tool calls never
enter the model's context, only the script's output and return value do.

- :class:`CodemodeSandbox` runs scripts in a Monty worker pool (needs the ``codemode`` extra).
- :class:`CodemodeSandboxBase` is the portable host side: tool table, call records, output,
  deadlines, cancellation, and deterministic runs (:meth:`CodemodeSandboxBase.start`).
- :func:`parse_codemode_source` splits the ``# @options:`` line off model input.
- :func:`render_declarations` renders Python declarations of the tools for the model.
- :func:`create_codemode_tool` wraps a sandbox as a pi_agent ``AgentTool``.

Everything but :class:`CodemodeSandbox` imports without Monty. It loads on first access
(as does :func:`create_codemode_tool`, which imports pi_agent) and raises an ``ImportError``
naming ``pi-ai-python[codemode]`` when Monty is missing.
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
from .runtime.execution import DEFAULT_TIMEOUT_MS, CodemodeRun, CodemodeSandboxBase
from .runtime.prelude import MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, MAX_STORE_TOTAL_CHARS, MAX_STORE_VALUE_CHARS
from .source import (
    CODEMODE_OPTIONS_PREFIX,
    CODEMODE_SOURCE_GRAMMAR,
    CodemodeSourceError,
    CodemodeSourceOptions,
    ParsedCodemodeSource,
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
    from .sandbox import CodemodeSandbox

__all__ = [
    # sandboxes
    "CodemodeSandbox",
    "CodemodeSandboxBase",
    "CodemodeRun",
    "DEFAULT_TIMEOUT_MS",
    "MAX_OUTPUT_CHARS",
    "MAX_OUTPUT_ITEMS",
    "MAX_STORE_TOTAL_CHARS",
    "MAX_STORE_VALUE_CHARS",
    # pi_agent
    "create_codemode_tool",
    # source
    "CODEMODE_OPTIONS_PREFIX",
    "CODEMODE_SOURCE_GRAMMAR",
    "CodemodeSourceError",
    "CodemodeSourceOptions",
    "ParsedCodemodeSource",
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
    # Monty-backed parts load on first access, so the portable parts import without the extra.
    if name == "CodemodeSandbox":
        from .sandbox import CodemodeSandbox

        return CodemodeSandbox
    if name == "create_codemode_tool":
        from .agent import create_codemode_tool

        return create_codemode_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
