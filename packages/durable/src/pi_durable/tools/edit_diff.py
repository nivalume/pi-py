"""Shared diff computation utilities for the edit and similar tools (port of ``tools/edit-diff.ts``).

Diffs come from :mod:`pi_durable.tools._jsdiff`, a port of the jsdiff functions upstream calls, so the
diff and patch text are upstream's exactly. String functions that JavaScript defines differently from
Python (``trimEnd``, ``split("")``) are reproduced.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Sequence
from typing import Literal, TypedDict

from . import _jsdiff

__all__ = [
    "AppliedEditsResult",
    "BomSplit",
    "DiffStringResult",
    "Edit",
    "FuzzyMatchResult",
    "apply_edits_to_normalized_content",
    "apply_replacements_preserving_unchanged_lines",
    "detect_line_ending",
    "fuzzy_find_text",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_bom",
]

#: What JavaScript's ``String.prototype.trimEnd`` removes: WhiteSpace and LineTerminator code points.
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)

_SINGLE_QUOTES = re.compile("[\u2018\u2019\u201a\u201b]")
_DOUBLE_QUOTES = re.compile("[\u201c\u201d\u201e\u201f]")
# U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash, U+2013 en-dash, U+2014 em-dash,
# U+2015 horizontal bar, U+2212 minus.
_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
# U+00A0 NBSP, U+2002-U+200A various spaces, U+202F narrow NBSP, U+205F medium math space,
# U+3000 ideographic space.
_SPACES = re.compile("[\u00a0\u2002-\u200a\u202f\u205f\u3000]")
_LINES_WITH_ENDINGS = re.compile(r"[^\n]*\n|[^\n]+")
_BOM = "\ufeff"


def detect_line_ending(content: str) -> Literal["\r\n", "\n"]:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1:
        return "\n"
    if crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: Literal["\r\n", "\n"]) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching.

    NFKC, then trailing whitespace stripped from each line, smart quotes to ASCII quotes, Unicode dashes and
    hyphens to ``-``, and special Unicode spaces to a regular space.
    """
    normalized = "\n".join(line.rstrip(_JS_WHITESPACE) for line in unicodedata.normalize("NFKC", text).split("\n"))
    normalized = _SINGLE_QUOTES.sub("'", normalized)
    normalized = _DOUBLE_QUOTES.sub('"', normalized)
    normalized = _DASHES.sub("-", normalized)
    return _SPACES.sub(" ", normalized)


def _split_lines_with_endings(content: str) -> list[str]:
    return _LINES_WITH_ENDINGS.findall(content)


class _LineSpan(TypedDict):
    start: int
    end: int


class _TextReplacement(TypedDict):
    matchIndex: int
    matchLength: int
    newText: str


class _MatchedEdit(_TextReplacement):
    editIndex: int


def _get_line_spans(content: str) -> list[_LineSpan]:
    spans: list[_LineSpan] = []
    offset = 0
    for line in _split_lines_with_endings(content):
        spans.append({"start": offset, "end": offset + len(line)})
        offset += len(line)
    return spans


def _get_replacement_line_range(lines: list[_LineSpan], replacement: _TextReplacement) -> tuple[int, int]:
    replacement_start = replacement["matchIndex"]
    replacement_end = replacement["matchIndex"] + replacement["matchLength"]

    start_line = -1
    for index, line in enumerate(lines):
        if line["start"] <= replacement_start < line["end"]:
            start_line = index
            break
    if start_line == -1:
        raise RuntimeError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line]["end"] < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise RuntimeError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: Sequence[_TextReplacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement["matchIndex"] - offset
        result = result[:match_index] + replacement["newText"] + result[match_index + replacement["matchLength"] :]
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: Sequence[_TextReplacement]
) -> str:
    """Apply replacements matched against ``base_content`` to ``original_content``, keeping unchanged lines.

    ``base_content`` is a normalized view of the original. Each replacement is widened to the lines it
    touches, those lines are rewritten from the normalized base, and all other lines are copied back from
    ``original_content``. The replacement ranges drive preservation, so duplicate normalized lines cannot be
    aligned to the wrong occurrence.
    """
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise RuntimeError("Cannot preserve unchanged lines because the base content has a different line count.")

    # Each group: [start line, end line] and its replacements.
    groups: list[tuple[list[int], list[_TextReplacement]]] = []
    for replacement in sorted(replacements, key=lambda each: each["matchIndex"]):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1][0][1]:
            current = groups[-1]
            current[0][1] = max(current[0][1], end_line)
            current[1].append(replacement)
            continue
        groups.append(([start_line, end_line], [replacement]))

    original_line_index = 0
    result = ""
    for (start_line, end_line), group_replacements in groups:
        result += "".join(original_lines[original_line_index:start_line])
        group_start_offset = base_lines[start_line]["start"]
        group_end_offset = base_lines[end_line - 1]["end"]
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset], group_replacements, group_start_offset
        )
        original_line_index = end_line
    result += "".join(original_lines[original_line_index:])
    return result


class FuzzyMatchResult(TypedDict):
    #: Whether a match was found.
    found: bool
    #: The index where the match starts (in ``contentForReplacement``).
    index: int
    #: Length of the matched text.
    matchLength: int
    #: Whether fuzzy matching was used (``False`` = exact match).
    usedFuzzyMatch: bool
    #: The content to use for replacement: the original for an exact match, the normalized one for a fuzzy one.
    contentForReplacement: str


class Edit(TypedDict):
    oldText: str
    newText: str


class AppliedEditsResult(TypedDict):
    baseContent: str
    newContent: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find ``old_text`` in ``content``, exactly first, then in fuzzy-normalized space."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return {
            "found": True,
            "index": exact_index,
            "matchLength": len(old_text),
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return {
            "found": False,
            "index": -1,
            "matchLength": 0,
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    # Offsets in normalized space: callers compute replacements in the normalized content, then decide how
    # much of that output to write back.
    return {
        "found": True,
        "index": fuzzy_index,
        "matchLength": len(fuzzy_old_text),
        "usedFuzzyMatch": True,
        "contentForReplacement": fuzzy_content,
    }


class BomSplit(TypedDict):
    bom: str
    text: str


def strip_bom(content: str) -> BomSplit:
    """Strip a UTF-8 BOM if present; return the BOM (if any) and the text without it."""
    if content.startswith(_BOM):
        return {"bom": _BOM, "text": content[1:]}
    return {"bom": "", "text": content}


def _utf16_length(text: str) -> int:
    return len(text) + sum(1 for character in text if ord(character) > 0xFFFF)


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    if fuzzy_old_text == "":
        # JavaScript's `split("")` splits into UTF-16 code units, and an empty string into none.
        return _utf16_length(fuzzy_content) - 1
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"Could not find the exact text in {path}. The old text must match exactly including all whitespace "
            "and newlines."
        )
    return RuntimeError(
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace "
        "and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more "
            "context to make it unique."
        )
    return RuntimeError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please "
        "provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(f"oldText must not be empty in {path}.")
    return RuntimeError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"No changes made to {path}. The replacement produced identical content. This might indicate an issue "
            "with special characters or the text not existing as expected."
        )
    return RuntimeError(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(normalized_content: str, edits: Sequence[Edit], path: str) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content.

    All edits are matched against the same original content and applied in reverse order. If any edit
    needs fuzzy matching, the operation runs in fuzzy-normalized space and overlays the changed lines onto
    the original, so unchanged lines keep their original text.
    """
    normalized_edits: list[Edit] = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])} for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if len(edit["oldText"]) == 0:
            raise _empty_old_text_error(path, index, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit["oldText"]) for edit in normalized_edits]
    used_fuzzy_match = any(match["usedFuzzyMatch"] for match in initial_matches)
    replacement_base_content = normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content

    matched_edits: list[_MatchedEdit] = []
    for index, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit["oldText"])
        if not match_result["found"]:
            raise _not_found_error(path, index, len(normalized_edits))
        occurrences = _count_occurrences(replacement_base_content, edit["oldText"])
        if occurrences > 1:
            raise _duplicate_error(path, index, len(normalized_edits), occurrences)
        matched_edits.append(
            {
                "editIndex": index,
                "matchIndex": match_result["index"],
                "matchLength": match_result["matchLength"],
                "newText": edit["newText"],
            }
        )

    matched_edits.sort(key=lambda each: each["matchIndex"])
    for previous, current in itertools.pairwise(matched_edits):
        if previous["matchIndex"] + previous["matchLength"] > current["matchIndex"]:
            raise RuntimeError(
                f"edits[{previous['editIndex']}] and edits[{current['editIndex']}] overlap in {path}. Merge them "
                "into one edit or target disjoint regions."
            )

    base_content = normalized_content
    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base_content, matched_edits)
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, matched_edits)
    )
    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))
    return {"baseContent": base_content, "newContent": new_content}


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """A standard unified patch with file headers only."""
    return _jsdiff.create_two_files_patch(
        path, path, old_content, new_content, context=context_lines, header_options=_jsdiff.FILE_HEADERS_ONLY
    )


class DiffStringResult(TypedDict):
    diff: str
    #: The first changed line in the new file; ``None`` without changes.
    firstChangedLine: int | None


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> DiffStringResult:
    """A display-oriented diff with line numbers and context, and the first changed line of the new file."""
    parts = _jsdiff.diff_lines(old_content, new_content)
    output: list[str] = []

    max_line_num = max(len(old_content.split("\n")), len(new_content.split("\n")))
    width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None

    def context_line(line: str) -> None:
        nonlocal old_line_num, new_line_num
        output.append(f" {str(old_line_num).rjust(width)} {line}")
        old_line_num += 1
        new_line_num += 1

    def skip(lines: int) -> None:
        nonlocal old_line_num, new_line_num
        output.append(f" {''.rjust(width)} ...")
        old_line_num += lines
        new_line_num += lines

    for index, part in enumerate(parts):
        raw = part.value.split("\n")
        if raw[-1] == "":
            raw.pop()

        if part.added or part.removed:
            # Capture the first changed line (in the new file).
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in raw:
                if part.added:
                    output.append(f"+{str(new_line_num).rjust(width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(width)} {line}")
                    old_line_num += 1
            last_was_change = True
            continue

        # Context lines: only a few before and after changes.
        next_part_is_change = index < len(parts) - 1 and (parts[index + 1].added or parts[index + 1].removed)
        has_leading_change = last_was_change
        has_trailing_change = next_part_is_change

        if has_leading_change and has_trailing_change:
            if len(raw) <= context_lines * 2:
                for line in raw:
                    context_line(line)
            else:
                leading = raw[:context_lines]
                trailing = raw[len(raw) - context_lines :]
                skipped = len(raw) - len(leading) - len(trailing)
                for line in leading:
                    context_line(line)
                skip(skipped)
                for line in trailing:
                    context_line(line)
        elif has_leading_change:
            shown = raw[:context_lines]
            skipped = len(raw) - len(shown)
            for line in shown:
                context_line(line)
            if skipped > 0:
                skip(skipped)
        elif has_trailing_change:
            skipped = max(0, len(raw) - context_lines)
            if skipped > 0:
                skip(skipped)
            for line in raw[skipped:]:
                context_line(line)
        else:
            # Skip these context lines entirely.
            old_line_num += len(raw)
            new_line_num += len(raw)
        last_was_change = False

    return {"diff": "\n".join(output), "firstChangedLine": first_changed_line}
