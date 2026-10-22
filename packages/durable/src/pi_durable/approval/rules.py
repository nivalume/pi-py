"""Permission rules: Claude Code's ``Tool(specifier)`` grammar, and how a rule matches a call.

``Bash(<command pattern>)`` matches command subjects, ``Read(<path pattern>)`` read subjects, ``Edit(...)`` and
``Write(...)`` write subjects (the names are case-insensitive); any other name, without a specifier, matches the tool
of that exact name whatever its subject. ``Bash``, ``Bash(*)`` and the like match every subject of their kind.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from .commands import command_matches, command_stem, split_command
from .paths import Anchors, PathPattern, is_within_directory, parent_rule_path

__all__ = [
    "CommandSubject",
    "PathAccess",
    "PathSubject",
    "PermissionSubject",
    "Rule",
    "RuleCall",
    "RuleError",
    "RuleErrorCode",
    "allowed_by",
    "is_within",
    "propose_rule",
]

PathAccess: TypeAlias = Literal["read", "write"]


class CommandSubject(TypedDict):
    """A shell command the call runs."""

    type: Literal["command"]
    command: str


class PathSubject(TypedDict):
    """A path the call reads or writes, absolute."""

    type: Literal["path"]
    path: str
    access: PathAccess


#: What permission rules match a call on (``ToolRegistration.permission_subject``).
PermissionSubject: TypeAlias = CommandSubject | PathSubject

RuleErrorCode: TypeAlias = Literal["empty", "unbalanced", "empty_specifier", "unsupported_specifier", "no_root"]


class RuleError(ValueError):
    """A rule that does not parse, or a ``/path`` rule where no root is configured."""

    def __init__(self, code: RuleErrorCode, rule: str = "") -> None:
        messages = {
            "empty": "Rule is empty",
            "unbalanced": f"Rule {rule!r} has unbalanced parentheses",
            "empty_specifier": f"Rule {rule!r} has an empty specifier",
            "unsupported_specifier": f"Rule {rule!r}: only Bash, Read, Edit and Write take a specifier",
            "no_root": f"Rule {rule!r} is relative to the root, and no root is configured",
        }
        super().__init__(messages[code])
        self.code: RuleErrorCode = code
        self.rule = rule


@dataclass(frozen=True, slots=True)
class _Command:
    """Command subjects; ``None``: every command."""

    pattern: str | None


@dataclass(frozen=True, slots=True)
class _Path:
    """Path subjects of ``access``; ``None``: every path."""

    access: PathAccess
    pattern: PathPattern | None


@dataclass(frozen=True, slots=True)
class _Tool:
    """The tool of this name, whatever its subject."""

    name: str


@dataclass(frozen=True, slots=True)
class RuleCall:
    """The call a rule is matched against."""

    tool: str
    subject: PermissionSubject | None
    anchors: Anchors


@dataclass(frozen=True, slots=True)
class Rule:
    """One rule, as written (``Bash(cargo test:*)``, ``Read(./.env)``, ``my_tool``)."""

    text: str
    _matcher: _Command | _Path | _Tool

    @staticmethod
    def parse(text: str) -> Rule:
        """Parse a rule; raises :class:`RuleError`."""
        text = text.strip()
        if not text:
            raise RuleError("empty")
        open_at = text.find("(")
        specifier: str | None
        if open_at < 0:
            if ")" in text:
                raise RuleError("unbalanced", text)
            name, specifier = text, None
        else:
            if not text.endswith(")"):
                raise RuleError("unbalanced", text)
            inner = text[open_at + 1 : -1].strip()
            if not inner:
                raise RuleError("empty_specifier", text)
            name, specifier = text[:open_at].strip(), None if inner == "*" else inner
        if not name:
            raise RuleError("empty")
        kind = name.lower()
        matcher: _Command | _Path | _Tool
        if kind == "bash":
            matcher = _Command(specifier)
        elif kind in ("read", "edit", "write"):
            access: PathAccess = "read" if kind == "read" else "write"
            matcher = _Path(access, None if specifier is None else PathPattern.parse(specifier))
        elif specifier is not None:
            raise RuleError("unsupported_specifier", text)
        else:
            matcher = _Tool(name)
        return Rule(text, matcher)

    def __str__(self) -> str:
        return self.text

    @property
    def needs_root(self) -> bool:
        """Whether the rule depends on the policy's root (``/path``)."""
        matcher = self._matcher
        return isinstance(matcher, _Path) and matcher.pattern is not None and matcher.pattern.anchor == "root"

    def restricts(self, call: RuleCall) -> bool:
        """Whether the rule, as a deny or ask rule, applies to ``call``: a command rule when any sub-command matches
        (the whole text when it does not split), a ``Read`` rule also for writes."""
        matcher = self._matcher
        subject = call.subject
        if isinstance(matcher, _Tool):
            return matcher.name == call.tool
        if isinstance(matcher, _Command):
            if subject is None or subject["type"] != "command":
                return False
            if matcher.pattern is None:
                return True
            parts = split_command(subject["command"])
            if parts is None:
                return command_matches(matcher.pattern, subject["command"].strip())
            pattern = matcher.pattern
            return any(command_matches(pattern, part) for part in parts)
        if subject is None or subject["type"] != "path":
            return False
        reaches = matcher.access == subject["access"] or matcher.access == "read"
        return reaches and (matcher.pattern is None or matcher.pattern.matches(subject["path"], call.anchors))

    def _allows_command(self, part: str) -> bool:
        """Whether the rule, as an allow rule, matches the sub-command ``part``."""
        matcher = self._matcher
        if not isinstance(matcher, _Command):
            return False
        return matcher.pattern is None or command_matches(matcher.pattern, part)

    def _allows(self, call: RuleCall) -> bool:
        """Whether the rule, as an allow rule, matches a call that is not a command."""
        matcher = self._matcher
        if isinstance(matcher, _Tool):
            return matcher.name == call.tool
        subject = call.subject
        if not isinstance(matcher, _Path) or subject is None or subject["type"] != "path":
            return False
        return matcher.access == subject["access"] and (
            matcher.pattern is None or matcher.pattern.matches(subject["path"], call.anchors)
        )

    def _names(self, tool: str) -> bool:
        return isinstance(self._matcher, _Tool) and self._matcher.name == tool


def allowed_by(rules: Iterable[Rule], call: RuleCall) -> bool:
    """Whether ``rules``, as allow rules, allow ``call``: a command when every sub-command matches one of them (never a
    command that does not split), anything else when one matches. A rule naming the tool allows every call of it."""
    rules = list(rules)
    if any(rule._names(call.tool) for rule in rules):
        return True
    subject = call.subject
    if subject is not None and subject["type"] == "command":
        parts = split_command(subject["command"])
        return parts is not None and all(any(rule._allows_command(part) for rule in rules) for part in parts)
    return any(rule._allows(call) for rule in rules)


def propose_rule(tool: str, subject: PermissionSubject | None) -> str | None:
    """The rule a UI offers to always allow a call like this one: ``Bash(<stem>:*)`` for a single command (none for a
    compound one), ``Edit(//<dir>/**)`` or ``Read(//<dir>/**)`` for a path, the tool's name without a subject."""
    if subject is None:
        return tool
    if subject["type"] == "command":
        parts = split_command(subject["command"])
        if parts is None or len(parts) != 1:
            return None
        return f"Bash({command_stem(parts[0])}:*)"
    kind = "Read" if subject["access"] == "read" else "Edit"
    parent = parent_rule_path(subject["path"])
    glob = "//**" if parent == "//" else f"{parent}/**"
    return f"{kind}({glob})"


def is_within(path: str, directory: str, cwd: str | None) -> bool:
    """Whether ``path`` is inside ``directory`` (absolute, or relative to ``cwd``)."""
    if directory.startswith("/"):
        return is_within_directory(path, directory)
    return cwd is not None and is_within_directory(path, f"{cwd.rstrip('/')}/{directory}")
