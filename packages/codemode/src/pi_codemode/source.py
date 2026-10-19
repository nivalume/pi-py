"""Codemode source format: Python, optionally preceded by one options line (port of ``source.ts``).

::

    # @options: {"max_output_tokens": 2000, "timeout_ms": 30000}
    source = await tools.read(path="pyproject.toml")
    text(source.splitlines()[0])

The module has no dependencies, so hosts that only parse model input can import it alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = [
    "CODEMODE_OPTIONS_PREFIX",
    "CODEMODE_SOURCE_GRAMMAR",
    "CodemodeSourceError",
    "CodemodeSourceOptions",
    "ParsedCodemodeSource",
    "parse_codemode_source",
]

CODEMODE_OPTIONS_PREFIX: Final = "# @options:"

_SUPPORTED_FIELDS: Final = ("max_output_tokens", "timeout_ms")
_SUPPORTED_FIELDS_TEXT: Final = "`max_output_tokens` and `timeout_ms`"
#: Largest delay JavaScript's ``setTimeout`` supports, which bounds ``timeout_ms`` as upstream does.
_MAX_TIMEOUT_MS: Final = 2_147_483_647
_MAX_SAFE_INTEGER: Final = 2**53 - 1

#: Lark grammar for providers with grammar-constrained tool input. It only fixes the shape of the
#: options line; the options JSON and the code are checked by :func:`parse_codemode_source`.
CODEMODE_SOURCE_GRAMMAR: Final = r"""
start: options_source | plain_source
options_source: OPTIONS_LINE NEWLINE SOURCE
plain_source: SOURCE

OPTIONS_LINE: /[ \t]*# @options:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"""


@dataclass(slots=True, frozen=True)
class CodemodeSourceOptions:
    """The fields of an options line; ``None`` when the line does not set them."""

    #: Token budget for the script's output.
    max_output_tokens: int | None = None
    #: Hard deadline for the whole script in milliseconds, including tool calls.
    timeout_ms: int | None = None


@dataclass(slots=True, frozen=True)
class ParsedCodemodeSource:
    #: The script with the options line replaced by an empty line, so line numbers are unchanged.
    code: str
    options: CodemodeSourceOptions = field(default_factory=CodemodeSourceOptions)


class CodemodeSourceError(ValueError):
    """Invalid codemode input: empty, a malformed options line, or an options line in the wrong place."""


def _is_safe_integer(value: Any) -> bool:
    # JSON has one number type: `10.0` is the integer 10, as in JavaScript.
    if type(value) is float and value.is_integer():
        value = int(value)
    return type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER


def _parse_options(directive: str) -> CodemodeSourceOptions:
    if directive == "":
        raise CodemodeSourceError(f"@options must be a JSON object with supported fields {_SUPPORTED_FIELDS_TEXT}")
    try:
        value = json.loads(directive, parse_constant=_reject_constant)
    except ValueError as error:
        raise CodemodeSourceError(
            f"@options must be valid JSON with supported fields {_SUPPORTED_FIELDS_TEXT}: {error}"
        ) from None
    if not isinstance(value, dict):
        raise CodemodeSourceError(f"@options must be a JSON object with supported fields {_SUPPORTED_FIELDS_TEXT}")
    for key in value:
        if key not in _SUPPORTED_FIELDS:
            raise CodemodeSourceError(f"@options only supports {_SUPPORTED_FIELDS_TEXT}; got `{key}`")
    max_output_tokens = value.get("max_output_tokens")
    timeout_ms = value.get("timeout_ms")
    if "max_output_tokens" in value and not _is_safe_integer(max_output_tokens):
        raise CodemodeSourceError("@options field `max_output_tokens` must be a non-negative safe integer")
    if "timeout_ms" in value and (
        not _is_safe_integer(timeout_ms) or timeout_ms is None or timeout_ms == 0 or timeout_ms > _MAX_TIMEOUT_MS
    ):
        raise CodemodeSourceError(f"@options field `timeout_ms` must be a positive integer up to {_MAX_TIMEOUT_MS}")
    return CodemodeSourceOptions(
        max_output_tokens=None if max_output_tokens is None else int(max_output_tokens),
        timeout_ms=None if timeout_ms is None else int(timeout_ms),
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def parse_codemode_source(source: str) -> ParsedCodemodeSource:
    """Split an optional first-line ``# @options: {...}`` from the script.

    Raises :class:`CodemodeSourceError` for empty input, invalid options, an options line
    without code after it, and an options line that is not the first line, which would
    otherwise be ignored without notice.
    """
    if source.strip() == "":
        raise CodemodeSourceError(
            "Expected Python source text (non-empty). Provide Python only, optionally with a first line "
            '`# @options: {"max_output_tokens": 1000}`.'
        )
    lines = source.split("\n")
    for index, line in enumerate(lines):
        if index > 0 and line.lstrip().startswith(CODEMODE_OPTIONS_PREFIX):
            raise CodemodeSourceError(
                f"The {CODEMODE_OPTIONS_PREFIX} line must be the first line of the input; found one on line {index + 1}"
            )
    first_line = re.sub(r"\r$", "", lines[0])
    trimmed = first_line.lstrip()
    if not trimmed.startswith(CODEMODE_OPTIONS_PREFIX):
        return ParsedCodemodeSource(code=source)
    newline = source.find("\n")
    code = "" if newline == -1 else source[newline:]
    if code.strip() == "":
        raise CodemodeSourceError("The @options line must be followed by Python source on subsequent lines")
    return ParsedCodemodeSource(code=code, options=_parse_options(trimmed[len(CODEMODE_OPTIONS_PREFIX) :].strip()))
