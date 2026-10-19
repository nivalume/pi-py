"""The parts of jsdiff (the ``diff`` npm package, 8.0.4) that the edit tool uses, ported line for line.

``diff_lines`` (``diffLines``: the Myers diff over line tokens of ``diff/base.js`` and ``diff/line.js``),
``structured_patch``, ``format_patch`` and ``create_two_files_patch`` (``patch/create.js``), with the
header option sets. Only the synchronous forms and the options the edit tool passes (``context``,
``headerOptions``) are ported; the output is identical to jsdiff's, so the diff text a model sees is the
same in TypeScript and Python. Not :mod:`difflib`, whose matching differs.

Original code: https://github.com/kpdecker/jsdiff, under this license:

BSD 3-Clause License

Copyright (c) 2009-2015, Kevin Decker <kpdecker@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, TypedDict

__all__ = [
    "FILE_HEADERS_ONLY",
    "INCLUDE_HEADERS",
    "OMIT_HEADERS",
    "Change",
    "HeaderOptions",
    "Hunk",
    "StructuredPatch",
    "create_two_files_patch",
    "diff_lines",
    "format_patch",
    "structured_patch",
]


class HeaderOptions(TypedDict):
    includeIndex: bool
    includeUnderline: bool
    includeFileHeaders: bool


INCLUDE_HEADERS: Final[HeaderOptions] = {"includeIndex": True, "includeUnderline": True, "includeFileHeaders": True}
FILE_HEADERS_ONLY: Final[HeaderOptions] = {
    "includeIndex": False,
    "includeUnderline": False,
    "includeFileHeaders": True,
}
OMIT_HEADERS: Final[HeaderOptions] = {"includeIndex": False, "includeUnderline": False, "includeFileHeaders": False}


@dataclass(slots=True)
class Change:
    """One run of a diff (jsdiff's change object): ``count`` tokens, added, removed, or common."""

    count: int
    added: bool
    removed: bool
    value: str = ""
    lines: list[str] | None = None


@dataclass(slots=True)
class _Component:
    count: int
    added: bool
    removed: bool
    previous: _Component | None


@dataclass(slots=True)
class _Path:
    old_pos: int
    last: _Component | None


_LINE_SEPARATORS = re.compile(r"(\n|\r\n)")


def _tokenize(value: str) -> list[str]:
    """``diff/line.js`` ``tokenize`` without options: lines with their separators."""
    ret_lines: list[str] = []
    lines_and_newlines = _LINE_SEPARATORS.split(value)
    # Ignore the final empty token that occurs if the string ends with a new line.
    if not lines_and_newlines[-1]:
        lines_and_newlines.pop()
    # Merge the content and line separators into single tokens.
    for index, line in enumerate(lines_and_newlines):
        if index % 2:
            ret_lines[-1] += line
        else:
            ret_lines.append(line)
    return ret_lines


def _add_to_path(path: _Path, added: bool, removed: bool, old_pos_inc: int) -> _Path:
    last = path.last
    if last is not None and last.added == added and last.removed == removed:
        return _Path(path.old_pos + old_pos_inc, _Component(last.count + 1, added, removed, last.previous))
    return _Path(path.old_pos + old_pos_inc, _Component(1, added, removed, last))


def _extract_common(base_path: _Path, new_tokens: list[str], old_tokens: list[str], diagonal_path: int) -> int:
    new_len = len(new_tokens)
    old_len = len(old_tokens)
    old_pos = base_path.old_pos
    new_pos = old_pos - diagonal_path
    common_count = 0
    while new_pos + 1 < new_len and old_pos + 1 < old_len and old_tokens[old_pos + 1] == new_tokens[new_pos + 1]:
        new_pos += 1
        old_pos += 1
        common_count += 1
    if common_count:
        base_path.last = _Component(common_count, False, False, base_path.last)
    base_path.old_pos = old_pos
    return new_pos


def _build_values(last: _Component | None, new_tokens: list[str], old_tokens: list[str]) -> list[Change]:
    components: list[_Component] = []
    while last is not None:
        components.append(last)
        last = last.previous
    components.reverse()
    changes: list[Change] = []
    new_pos = 0
    old_pos = 0
    for component in components:
        change = Change(component.count, component.added, component.removed)
        if not component.removed:
            change.value = "".join(new_tokens[new_pos : new_pos + component.count])
            new_pos += component.count
            # Common case.
            if not component.added:
                old_pos += component.count
        else:
            change.value = "".join(old_tokens[old_pos : old_pos + component.count])
            old_pos += component.count
        changes.append(change)
    return changes


def diff_lines(old_str: str, new_str: str) -> list[Change]:
    """jsdiff ``diffLines(oldStr, newStr)``: the line changes turning ``old_str`` into ``new_str``."""
    old_tokens = [token for token in _tokenize(old_str) if token]
    new_tokens = [token for token in _tokenize(new_str) if token]
    new_len = len(new_tokens)
    old_len = len(old_tokens)
    edit_length = 1
    max_edit_length = new_len + old_len
    best_path: dict[int, _Path | None] = {0: _Path(-1, None)}
    first = best_path[0]
    assert first is not None
    # Seed edit_length = 0, i.e. the content starts with the same values.
    new_pos = _extract_common(first, new_tokens, old_tokens, 0)
    if first.old_pos + 1 >= old_len and new_pos + 1 >= new_len:
        # Identity per the equality and tokenizer.
        return _build_values(first.last, new_tokens, old_tokens)

    # Once the right edge of the edit graph is hit on some diagonal k, no move to diagonal k + 1 is worth
    # considering; once the bottom is hit, no move to lower diagonals (jsdiff's bound on Myers's algorithm).
    min_diagonal_to_consider = -max_edit_length - 1
    max_diagonal_to_consider = max_edit_length + 1
    while edit_length <= max_edit_length:
        diagonal_path = max(min_diagonal_to_consider, -edit_length)
        while diagonal_path <= min(max_diagonal_to_consider, edit_length):
            remove_path = best_path.get(diagonal_path - 1)
            add_path = best_path.get(diagonal_path + 1)
            if remove_path is not None:
                # No one else is going to attempt to use this value, clear it.
                best_path[diagonal_path - 1] = None
            can_add = False
            if add_path is not None:
                # What new_pos will be after an insertion.
                add_path_new_pos = add_path.old_pos - diagonal_path
                can_add = 0 <= add_path_new_pos < new_len
            can_remove = remove_path is not None and remove_path.old_pos + 1 < old_len
            if not can_add and not can_remove:
                # If this path is a terminal then prune.
                best_path[diagonal_path] = None
                diagonal_path += 2
                continue
            # Branch from the prior path whose position in the old string is farthest from the origin
            # without passing the bounds of the diff graph.
            if not can_remove or (can_add and remove_path.old_pos < add_path.old_pos):  # type: ignore[union-attr]
                assert add_path is not None
                base_path = _add_to_path(add_path, True, False, 0)
            else:
                assert remove_path is not None
                base_path = _add_to_path(remove_path, False, True, 1)
            new_pos = _extract_common(base_path, new_tokens, old_tokens, diagonal_path)
            if base_path.old_pos + 1 >= old_len and new_pos + 1 >= new_len:
                # Hit the end of both strings: done.
                return _build_values(base_path.last, new_tokens, old_tokens)
            best_path[diagonal_path] = base_path
            if base_path.old_pos + 1 >= old_len:
                max_diagonal_to_consider = min(max_diagonal_to_consider, diagonal_path - 1)
            if new_pos + 1 >= new_len:
                min_diagonal_to_consider = max(min_diagonal_to_consider, diagonal_path + 1)
            diagonal_path += 2
        edit_length += 1
    raise AssertionError("unreachable: an edit script of at most len(old) + len(new) edits exists")


@dataclass(slots=True)
class Hunk:
    oldStart: int
    oldLines: int
    newStart: int
    newLines: int
    lines: list[str]


@dataclass(slots=True)
class StructuredPatch:
    oldFileName: str
    newFileName: str
    oldHeader: str | None
    newHeader: str | None
    hunks: list[Hunk] = field(default_factory=lambda: [])


def _split_lines(text: str) -> list[str]:
    """``text`` as lines, each with its trailing newline where present."""
    has_trailing_nl = text.endswith("\n")
    result = [line + "\n" for line in text.split("\n")]
    if has_trailing_nl:
        result.pop()
    else:
        result.append(result.pop()[:-1])
    return result


def structured_patch(
    old_file_name: str,
    new_file_name: str,
    old_str: str,
    new_str: str,
    old_header: str | None = None,
    new_header: str | None = None,
    *,
    context: int = 4,
) -> StructuredPatch:
    """jsdiff ``structuredPatch``: the hunks of a unified diff with ``context`` lines of context."""
    diff = diff_lines(old_str, new_str)
    # Append an empty value to make cleanup easier.
    diff.append(Change(0, False, False, "", []))

    def context_lines(lines: list[str]) -> list[str]:
        return [" " + entry for entry in lines]

    hunks: list[Hunk] = []
    old_range_start = 0
    new_range_start = 0
    cur_range: list[str] = []
    old_line = 1
    new_line = 1
    for index, current in enumerate(diff):
        lines = current.lines if current.lines is not None else _split_lines(current.value)
        current.lines = lines
        if current.added or current.removed:
            # If there is previous context, start with that.
            if not old_range_start:
                prev = diff[index - 1] if index > 0 else None
                old_range_start = old_line
                new_range_start = new_line
                if prev is not None:
                    assert prev.lines is not None
                    cur_range = context_lines(prev.lines[-context:]) if context > 0 else []
                    old_range_start -= len(cur_range)
                    new_range_start -= len(cur_range)
            # Output the changes.
            for line in lines:
                cur_range.append(("+" if current.added else "-") + line)
            # Track the updated file position.
            if current.added:
                new_line += len(lines)
            else:
                old_line += len(lines)
        else:
            # Identical context lines. Track line changes.
            if old_range_start:
                # Close out any changes that have been output (or join overlapping).
                if len(lines) <= context * 2 and index < len(diff) - 2:
                    # Overlapping.
                    cur_range.extend(context_lines(lines))
                else:
                    # End the range and output.
                    context_size = min(len(lines), context)
                    cur_range.extend(context_lines(lines[:context_size]))
                    hunks.append(
                        Hunk(
                            old_range_start,
                            old_line - old_range_start + context_size,
                            new_range_start,
                            new_line - new_range_start + context_size,
                            cur_range,
                        )
                    )
                    old_range_start = 0
                    new_range_start = 0
                    cur_range = []
            old_line += len(lines)
            new_line += len(lines)

    # Eliminate the trailing newline of each line of each hunk and, where needed, add
    # "\ No newline at end of file".
    for hunk in hunks:
        index = 0
        while index < len(hunk.lines):
            if hunk.lines[index].endswith("\n"):
                hunk.lines[index] = hunk.lines[index][:-1]
            else:
                hunk.lines.insert(index + 1, "\\ No newline at end of file")
                index += 1  # Skip the line just added.
            index += 1
    return StructuredPatch(old_file_name, new_file_name, old_header, new_header, hunks)


def format_patch(patch: StructuredPatch, header_options: HeaderOptions | None = None) -> str:
    """jsdiff ``formatPatch`` of one structured patch."""
    options = INCLUDE_HEADERS if header_options is None else header_options
    ret: list[str] = []
    if options["includeIndex"] and patch.oldFileName == patch.newFileName:
        ret.append("Index: " + patch.oldFileName)
    if options["includeUnderline"]:
        ret.append("===================================================================")
    if options["includeFileHeaders"]:
        ret.append("--- " + patch.oldFileName + ("" if patch.oldHeader is None else "\t" + patch.oldHeader))
        ret.append("+++ " + patch.newFileName + ("" if patch.newHeader is None else "\t" + patch.newHeader))
    for hunk in patch.hunks:
        # Unified Diff Format quirk: if the chunk size is 0, the first number is one lower than one would expect.
        old_start = hunk.oldStart - 1 if hunk.oldLines == 0 else hunk.oldStart
        new_start = hunk.newStart - 1 if hunk.newLines == 0 else hunk.newStart
        ret.append(f"@@ -{old_start},{hunk.oldLines} +{new_start},{hunk.newLines} @@")
        ret.extend(hunk.lines)
    return "\n".join(ret) + "\n"


def create_two_files_patch(
    old_file_name: str,
    new_file_name: str,
    old_str: str,
    new_str: str,
    old_header: str | None = None,
    new_header: str | None = None,
    *,
    context: int = 4,
    header_options: HeaderOptions | None = None,
) -> str:
    """jsdiff ``createTwoFilesPatch``: a unified diff of two texts."""
    patch = structured_patch(old_file_name, new_file_name, old_str, new_str, old_header, new_header, context=context)
    return format_patch(patch, header_options)
