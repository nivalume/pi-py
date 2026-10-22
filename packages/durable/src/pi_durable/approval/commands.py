"""Shell commands for rule matching: splitting a command into sub-commands, stripping assignments and wrappers, and
matching a sub-command against a command pattern."""

from __future__ import annotations

import re
from typing import Final

__all__ = ["command_matches", "command_stem", "split_command"]

#: Wrappers that run the command after them; rules match what they wrap.
_WRAPPERS: Final = frozenset({"timeout", "time", "nice", "nohup", "stdbuf", "command", "builtin"})
_ASSIGNMENT: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def split_command(command: str) -> list[str] | None:
    """The sub-commands of ``command``: split at ``&&``, ``||``, ``;``, ``|``, ``|&``, ``&`` and newlines outside
    quotes, ``$(...)`` and backticks; trimmed, without leading ``NAME=value`` assignments and wrappers, empty ones
    dropped. ``None`` when quotes, ``$(`` or backticks are unbalanced."""
    parts: list[str] = []
    current: list[str] = []
    single = double = backtick = False
    depth = 0
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        following = command[index + 1] if index + 1 < length else None
        nested = single or double or backtick or depth > 0
        if char == "\\" and not single:
            current.append(char)
            if following is not None:
                current.append(following)
                index += 1
        elif char == "'" and not double:
            single = not single
            current.append(char)
        elif char == '"' and not single:
            double = not double
            current.append(char)
        elif char == "`" and not single:
            backtick = not backtick
            current.append(char)
        elif char == "$" and not single and following == "(":
            depth += 1
            current.append("$(")
            index += 1
        elif char == ")" and not single and depth > 0:
            depth -= 1
            current.append(char)
        elif char in ";\n" and not nested:
            parts.append("".join(current))
            current = []
        elif char == "|" and not nested:
            if following in ("|", "&"):
                index += 1
            parts.append("".join(current))
            current = []
        elif char == "&" and not nested and following == "&":
            index += 1
            parts.append("".join(current))
            current = []
        elif char == "&" and not nested and not _is_redirection(command, index):
            # `2>&1`, `&>file` and `>&2` redirect; they do not separate.
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if single or double or backtick or depth > 0:
        return None
    parts.append("".join(current))
    stripped = (_strip_prefixes(part.strip()) for part in parts)
    return [part for part in stripped if part]


def _is_redirection(text: str, index: int) -> bool:
    """Whether the ``&`` at ``index`` belongs to a redirection (``>&``, ``<&``, ``&>``)."""
    before = text[index - 1] if index > 0 else None
    return before in (">", "<") or (index + 1 < len(text) and text[index + 1] == ">")


def _word_end(text: str) -> int:
    """Where the first word of ``text`` ends (quotes respected)."""
    single = double = False
    for index, char in enumerate(text):
        if char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif char.isspace() and not single and not double:
            return index
    return len(text)


def _first_word(text: str) -> tuple[str, str]:
    """``text`` split into its first word and the rest, trimmed."""
    end = _word_end(text)
    return text[:end], text[end:].lstrip()


def _takes_value(wrapper: str, option: str) -> bool:
    """Whether ``option`` of ``wrapper`` is followed by its value as a separate word."""
    return (
        (wrapper == "nice" and option in ("-n", "--adjustment"))
        or (wrapper == "timeout" and option in ("-s", "-k", "--signal", "--kill-after"))
        or (wrapper == "stdbuf" and option in ("-i", "-o", "-e"))
    )


def _strip_prefixes(part: str) -> str:
    """``part`` without leading assignments and wrappers (with their options; ``timeout``'s duration and ``nice -n``'s
    value too)."""
    rest = part
    while True:
        word, after = _first_word(rest)
        if not word:
            return ""
        if _ASSIGNMENT.match(word):
            rest = after
            continue
        if word not in _WRAPPERS:
            return rest
        rest = after
        while True:
            option, after = _first_word(rest)
            if not option.startswith("-") or option == "-":
                break
            rest = after
            if _takes_value(word, option):
                rest = _first_word(rest)[1]
        if word == "timeout":
            rest = _first_word(rest)[1]


def command_matches(pattern: str, command: str) -> bool:
    """Whether the sub-command ``command`` matches ``pattern``: ``*`` matches any text; a trailing `` *`` or ``:*``
    also matches the bare command."""
    for suffix in (":*", " *"):
        if pattern.endswith(suffix):
            base = pattern[: -len(suffix)]
            return _wildcard(base, command) or _wildcard(f"{base} *", command)
    return _wildcard(pattern, command)


def _wildcard(pattern: str, text: str) -> bool:
    """``*`` matches any (possibly empty) text; everything else literally."""
    p = t = 0
    # The last `*` and the text position it was tried at, for backtracking.
    star: tuple[int, int] | None = None
    while t < len(text):
        if p < len(pattern) and pattern[p] == "*":
            star = (p, t)
            p += 1
        elif p < len(pattern) and pattern[p] == text[t]:
            p += 1
            t += 1
        elif star is not None:
            at, start = star
            p = at + 1
            t = start + 1
            star = (at, start + 1)
        else:
            return False
    return all(char == "*" for char in pattern[p:])


def command_stem(part: str) -> str:
    """The words of a sub-command for a proposed rule: its first word, and the second unless it is an option."""
    first, rest = _first_word(part)
    second, _ = _first_word(rest)
    if not second or second.startswith("-"):
        return first
    return f"{first} {second}"
