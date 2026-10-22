"""Declarations of the script-visible API for the model, in the script's language (port of ``declarations.ts``).

JavaScript scripts read TypeScript declarations (:mod:`pi_codemode.typescript_declarations`, upstream's output),
Python scripts Python stubs (:mod:`pi_codemode.python_declarations`). Each function takes the ``language`` the
scripts are written in; the default is JavaScript.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import Final

from . import python_declarations, typescript_declarations
from .identifier import to_codemode_identifier
from .language import CodemodeLanguage, check_codemode_language
from .types import CodemodeJsonSchema, CodemodeToolLike

__all__ = [
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "render_declarations",
    "render_tool_output_type",
    "render_tool_sample",
    "render_tool_signature",
    "schema_to_type",
    "to_codemode_identifier",
]

#: Largest rendered input, in characters, before it falls back to ``unknown`` (TypeScript) or a catch-all
#: parameter list (Python).
DEFAULT_INPUT_SCHEMA_MAX_CHARS: Final = 16_000


def _renderer(language: CodemodeLanguage) -> ModuleType:
    """One of two modules with the same functions."""
    return typescript_declarations if check_codemode_language(language) == "javascript" else python_declarations


def render_declarations(
    *,
    tools: Iterable[CodemodeToolLike] = (),
    globals: Iterable[CodemodeToolLike] = (),
    language: CodemodeLanguage = "javascript",
) -> str:
    """Declarations of the script-visible API: in TypeScript, tools are members of ``declare const tools``, globals
    ``declare function`` statements and ``ns.member`` globals members of ``declare const ns``; in Python, methods of
    ``class tools``, ``async def`` functions and methods of ``class ns``. Descriptions become doc comments, schemas
    types."""
    return _renderer(language).render_declarations(tools=tools, globals=globals)


def render_tool_signature(
    tool: CodemodeToolLike, *, input_max_chars: int | None = None, language: CodemodeLanguage = "javascript"
) -> str:
    """One tool as a member of ``tools``, under the identifier scripts use: ``name(args: T): Promise<R>;`` or
    ``async def name(self, ...) -> R: ...``. Inputs longer than ``input_max_chars`` lose their constraint tags, then
    render as ``unknown`` (TypeScript) or ``args: Any = None, /, **kwargs: Any`` (Python)."""
    return _renderer(language).render_tool_signature(tool, input_max_chars=input_max_chars)


def render_tool_sample(
    tool: CodemodeToolLike, *, input_max_chars: int | None = None, language: CodemodeLanguage = "javascript"
) -> str:
    """A tool's sample: its description followed by its declaration, for tool listings."""
    return _renderer(language).render_tool_sample(tool, input_max_chars=input_max_chars)


def render_tool_output_type(schema: CodemodeJsonSchema | None, *, language: CodemodeLanguage = "javascript") -> str:
    """The type a tool call resolves to: the output schema's type, ``unknown``/``Any`` without a schema."""
    return _renderer(language).render_tool_output_type(schema)


def schema_to_type(
    schema: CodemodeJsonSchema, *, max_chars: int | None = None, language: CodemodeLanguage = "javascript"
) -> str:
    """A JSON Schema as a type expression of ``language``. A result longer than ``max_chars`` is rendered again
    without the constraint tags, and as ``unknown``/``Any`` if it is still too long."""
    return _renderer(language).schema_to_type(schema, max_chars=max_chars)
