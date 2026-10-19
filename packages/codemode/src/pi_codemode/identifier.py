"""The identifier a script uses for a tool (port of ``identifier.ts``)."""

from __future__ import annotations

import keyword

__all__ = ["is_codemode_identifier", "to_codemode_identifier"]

_FIRST = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
_REST = _FIRST | frozenset("0123456789")


def to_codemode_identifier(name: str) -> str:
    """The identifier a script uses for a tool: ``tools.<identifier>``.

    Characters that are not valid in an ASCII Python identifier become ``_``, and a name
    that is a Python keyword gets a trailing ``_`` (PEP 8's convention), so ``mcp__docs__search``
    stays as is, ``my-tool`` becomes ``my_tool``, ``2fa`` becomes ``_fa`` and ``class``
    becomes ``class_``. An empty name becomes ``_``.
    """
    identifier = "".join(char if char in (_FIRST if index == 0 else _REST) else "_" for index, char in enumerate(name))
    if identifier == "":
        return "_"
    return f"{identifier}_" if keyword.iskeyword(identifier) else identifier


def is_codemode_identifier(name: str) -> bool:
    """Whether ``name`` is an ASCII Python identifier that is not a keyword."""
    return name != "" and to_codemode_identifier(name) == name
