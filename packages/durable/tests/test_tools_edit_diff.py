"""The edit tool's diff text against upstream's: ``edit-diff.ts`` with jsdiff 8.0.4, run under Node.

``fixtures/edit_diff/upstream.json`` holds a seeded random corpus (line diffs with 0 to 4 context lines,
and edits with exact, fuzzy, missing, duplicate, empty and overlapping targets) with what upstream returned
for each: ``diffLines`` changes, the display diff, the unified patch, the applied edit, or the error text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pi_durable.tools import _jsdiff
from pi_durable.tools.edit_diff import (
    apply_edits_to_normalized_content,
    generate_diff_string,
    generate_unified_patch,
    normalize_for_fuzzy_match,
)

_CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "edit_diff" / "upstream.json").read_text("utf-8")
)["cases"]


def _actual(case: dict[str, Any]) -> dict[str, Any]:
    if case["kind"] == "diff":
        context = case.get("context", 4)
        diff = generate_diff_string(case["old"], case["new"], context)
        return {
            "diff": diff["diff"],
            "firstChangedLine": diff["firstChangedLine"],
            "patch": generate_unified_patch("f.txt", case["old"], case["new"], context),
            "parts": [
                [1 if part.added else -1 if part.removed else 0, part.count, part.value]
                for part in _jsdiff.diff_lines(case["old"], case["new"])
            ],
        }
    try:
        applied = apply_edits_to_normalized_content(case["content"], case["edits"], "f.txt")
    except Exception as error:
        return {"error": str(error)}
    return {
        **applied,
        "fuzzy": normalize_for_fuzzy_match(case["content"]),
        "diff": generate_diff_string(applied["baseContent"], applied["newContent"])["diff"],
        "patch": generate_unified_patch("f.txt", applied["baseContent"], applied["newContent"]),
    }


@pytest.mark.parametrize("index", range(len(_CASES)))
def test_matches_upstream(index: int) -> None:
    case = _CASES[index]
    assert _actual(case["input"]) == case["expected"]


def test_the_corpus_covers_every_outcome() -> None:
    outcomes = {
        "diff" if case["input"]["kind"] == "diff" else "error" if "error" in case["expected"] else "edit"
        for case in _CASES
    }
    assert outcomes == {"diff", "edit", "error"}
    errors = {
        case["expected"]["error"].split(" in ")[0].split(" of ")[0] for case in _CASES if "error" in case["expected"]
    }
    assert any(error.startswith("Found 2 occurrences") for error in errors)
    assert any("overlap" in error for error in errors)
    assert any("oldText must not be empty" in error for error in errors)
    assert any(error.startswith("Could not find") for error in errors)
    fuzzy = [
        case
        for case in _CASES
        if "newContent" in case["expected"] and case["expected"]["fuzzy"] != case["input"]["content"]
    ]
    assert fuzzy
