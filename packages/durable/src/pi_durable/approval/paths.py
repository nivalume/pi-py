"""Path patterns for rule matching: anchors and segment globs, matched lexically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

__all__ = ["Anchor", "Anchors", "PathPattern", "is_within_directory", "parent_rule_path", "segments"]

#: What a path pattern is relative to: ``//path`` absolute, ``~/path`` the home directory, ``/path`` the policy's
#: root, ``./path`` or ``path`` the call's directory.
Anchor: TypeAlias = Literal["absolute", "home", "root", "cwd"]


@dataclass(frozen=True, slots=True)
class Anchors:
    """The directories anchors stand for; ``None``: unknown, and a pattern anchored there matches nothing."""

    cwd: str | None = None
    home: str | None = None
    root: str | None = None


@dataclass(frozen=True, slots=True)
class PathPattern:
    """A path pattern: its anchor and the glob after it (without the anchor and leading slashes; may be empty, the
    anchor itself)."""

    anchor: Anchor
    glob: str

    @staticmethod
    def parse(text: str) -> PathPattern:
        if text.startswith("//"):
            anchor, glob = "absolute", text[2:]
        elif text == "~":
            anchor, glob = "home", ""
        elif text.startswith("~/"):
            anchor, glob = "home", text[2:]
        elif text.startswith("/"):
            anchor, glob = "root", text[1:]
        elif text == ".":
            anchor, glob = "cwd", ""
        else:
            anchor, glob = "cwd", text[2:] if text.startswith("./") else text
        return PathPattern(anchor, glob.lstrip("/"))

    def resolve(self, anchors: Anchors) -> str | None:
        """The absolute glob, ``None`` when the anchor's directory is unknown."""
        base = {"absolute": "/", "home": anchors.home, "root": anchors.root, "cwd": anchors.cwd}[self.anchor]
        if base is None:
            return None
        return f"{base.rstrip('/')}/{self.glob}"

    def matches(self, path: str, anchors: Anchors) -> bool:
        """Whether ``path`` (absolute) is the pattern's path or below it."""
        glob = self.resolve(anchors)
        if glob is None:
            return False
        return _match_segments(segments(glob), segments(path))


def is_within_directory(path: str, directory: str) -> bool:
    """Whether ``path`` (absolute) is ``directory`` or below it, lexically."""
    inside = segments(directory)
    parts = segments(path)
    return len(parts) >= len(inside) and parts[: len(inside)] == inside


def segments(path: str) -> list[str]:
    """The segments of an absolute path with ``.`` and ``..`` resolved lexically."""
    out: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if out:
                out.pop()
            continue
        out.append(segment)
    return out


def parent_rule_path(path: str) -> str:
    """The parent directory of an absolute path, as ``//dir`` for a rule."""
    parts = segments(path)
    if parts:
        parts.pop()
    return "//" + "/".join(parts)


def _match_segments(pattern: list[str], path: list[str]) -> bool:
    """Pattern segments against path segments; the path may continue past the pattern."""
    if not pattern:
        return True
    first, rest = pattern[0], pattern[1:]
    if first == "**":
        return any(_match_segments(rest, path[skip:]) for skip in range(len(path) + 1))
    if not path:
        return False
    return _segment_matches(first, path[0]) and _match_segments(rest, path[1:])


def _segment_matches(pattern: str, segment: str) -> bool:
    """One segment: ``*`` any characters, ``?`` one character."""
    p = t = 0
    # The last `*` and the segment position it was tried at, for backtracking.
    star: tuple[int, int] | None = None
    while t < len(segment):
        if p < len(pattern) and pattern[p] == "*":
            star = (p, t)
            p += 1
        elif p < len(pattern) and pattern[p] in ("?", segment[t]):
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
