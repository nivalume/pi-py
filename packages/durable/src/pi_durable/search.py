"""BM25 tool search (port of ``search.ts``).

Used by code mode's ``searchTools()`` and the ``tool_search`` tool: a copy of the coding agent's
tool search ranking.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing_extensions import ReadOnly

__all__ = ["SearchDocument", "rank_documents", "tokenize"]


class SearchDocument(TypedDict):
    name: ReadOnly[str]
    text: ReadOnly[str]


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

_PLURAL_ES = re.compile(r"(ches|shes|sses|xes|zes)$")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _stem(term: str) -> str:
    """Naive singular form, so ``issues`` matches ``issue`` and ``searches`` matches ``search``."""
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and _PLURAL_ES.search(term):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def tokenize(text: str) -> list[str]:
    """Lowercase terms, split at camelCase boundaries and non-alphanumerics, without stop words."""
    spaced = _ACRONYM.sub(r"\1 \2", _CAMEL.sub(r"\1 \2", text)).lower()
    return [_stem(term) for term in _SEPARATORS.split(spaced) if len(term) > 0 and term not in _STOP_WORDS]


def rank_documents(query: str, documents: Sequence[SearchDocument], limit: int) -> list[str]:
    """Okapi BM25 (k1 1.2, b 0.75): the names of the best matches, best first; ties keep document order."""
    k1 = 1.2
    b = 0.75
    query_terms = list(dict.fromkeys(tokenize(query)))
    if len(query_terms) == 0 or len(documents) == 0 or limit <= 0:
        return []
    term_counts: list[dict[str, int]] = []
    for document in documents:
        counts: dict[str, int] = {}
        for term in tokenize(document["text"]):
            counts[term] = counts.get(term, 0) + 1
        term_counts.append(counts)
    lengths = [sum(counts.values()) for counts in term_counts]
    average_length = sum(lengths) / len(documents) or 1
    idf: dict[str, float] = {}
    for term in query_terms:
        frequency = sum(1 for counts in term_counts if term in counts)
        idf[term] = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
    matches: list[tuple[str, float]] = []
    for index, document in enumerate(documents):
        score = 0.0
        for term in query_terms:
            count = term_counts[index].get(term)
            if not count:
                continue
            norm = k1 * (1 - b + (b * lengths[index]) / average_length)
            score += idf.get(term, 0) * ((count * (k1 + 1)) / (count + norm))
        if score > 0:
            matches.append((document["name"], score))
    matches.sort(key=lambda match: -match[1])
    return [name for name, _ in matches[:limit]]
