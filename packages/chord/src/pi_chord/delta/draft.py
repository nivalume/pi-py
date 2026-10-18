"""The ``Draft`` type (port of chord ``delta/draft.ts``).

``Draft[T]`` is a mutable, change-scoped view of a JSON value of type ``T``. Python has no
mapped types, so for type checkers ``Draft[T]`` *is* ``T`` (an ``Annotated`` alias):
annotate drafts with the document's ``TypedDict`` and mutate them like plain values.
At runtime a draft is a :class:`~pi_chord.delta.DraftDict` or
:class:`~pi_chord.delta.DraftList`, both ``dict``/``list`` subclasses.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias, TypeVar

__all__ = ["Draft"]

T = TypeVar("T")


class _DraftMarker:
    """``Annotated`` metadata naming a change-scoped draft view."""

    __slots__ = ()


#: A mutable view of a ``T`` that is valid only while its change is open.
Draft: TypeAlias = Annotated[T, _DraftMarker]
