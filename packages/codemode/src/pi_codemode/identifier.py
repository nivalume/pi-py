"""The identifier a script uses for a tool (port of ``identifier.ts``)."""

from __future__ import annotations

import keyword

from .language import CodemodeLanguage, check_codemode_language

__all__ = ["is_codemode_identifier", "to_codemode_identifier"]

_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_DIGITS = frozenset("0123456789")
_FIRST = {"javascript": _LETTERS | {"$"}, "python": _LETTERS}
_REST = {"javascript": _LETTERS | _DIGITS | {"$"}, "python": _LETTERS | _DIGITS}


def to_codemode_identifier(name: str, language: CodemodeLanguage = "javascript") -> str:
    """The identifier a script uses for a tool: ``tools.<identifier>``.

    Characters that are not valid in an ASCII identifier of ``language`` become ``_``, so ``mcp__docs__search``
    stays as is, ``my-tool`` becomes ``my_tool`` and ``2fa`` becomes ``_fa``. In Python a name that is a keyword also
    gets a trailing ``_`` (PEP 8's convention), so ``class`` becomes ``class_``; JavaScript reads ``tools.class``
    as is. An empty name becomes ``_``.
    """
    first, rest = _FIRST[check_codemode_language(language)], _REST[language]
    identifier = "".join(char if char in (first if index == 0 else rest) else "_" for index, char in enumerate(name))
    if identifier == "":
        return "_"
    return f"{identifier}_" if language == "python" and keyword.iskeyword(identifier) else identifier


def is_codemode_identifier(name: str, language: CodemodeLanguage = "javascript") -> bool:
    """Whether ``name`` is an ASCII identifier of ``language`` (in Python, also not a keyword)."""
    return name != "" and to_codemode_identifier(name, language) == name
