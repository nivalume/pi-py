"""The language scripts are written in: JavaScript (the default, run by QuickJS) or Python (run by Monty).

A sandbox runs one language, and everything a model reads about it (declarations, the source format, the tool's
description) follows that sandbox's language, so a code mode tool offers exactly one of the two. The module has no
dependencies.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias, cast

__all__ = ["CODEMODE_LANGUAGES", "DEFAULT_CODEMODE_LANGUAGE", "CodemodeLanguage", "check_codemode_language"]

#: ``javascript``: scripts run in QuickJS (:class:`pi_codemode.CodemodeSandbox`), as upstream's.
#: ``python``: scripts run in Monty (:class:`pi_codemode.MontyCodemodeSandbox`).
CodemodeLanguage: TypeAlias = Literal["javascript", "python"]

CODEMODE_LANGUAGES: Final[tuple[CodemodeLanguage, ...]] = ("javascript", "python")
DEFAULT_CODEMODE_LANGUAGE: Final[CodemodeLanguage] = "javascript"


def check_codemode_language(language: object) -> CodemodeLanguage:
    """``language`` if it is a :data:`CodemodeLanguage`; raises ``ValueError`` otherwise."""
    if language in CODEMODE_LANGUAGES:
        return cast("CodemodeLanguage", language)
    raise ValueError(f'Unknown code mode language {language!r}: expected "javascript" or "python"')
