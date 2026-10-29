"""The ``hooks`` object of Claude Code's settings: events, matcher groups and handlers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NotRequired, TypeAlias, TypedDict

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "SUPPORTED_EVENTS",
    "CommandHook",
    "ConfigError",
    "HookCommand",
    "HooksConfig",
    "HttpHook",
    "Matcher",
    "MatcherGroup",
]

#: Events this bridge runs; the others are listed by :meth:`HooksConfig.unsupported`.
SUPPORTED_EVENTS: Final = (
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "PreCompact",
)

#: A handler's default timeout; ``UserPromptSubmit`` handlers default to 30 seconds.
DEFAULT_TIMEOUT_SECONDS: Final = 600.0
_PROMPT_TIMEOUT_SECONDS: Final = 30.0
_NAMES: Final = re.compile(r"[A-Za-z0-9_|]+")


class ConfigError(ValueError):
    """A configuration that does not load: a malformed shape or an invalid matcher."""


class CommandHook(TypedDict):
    """A command run through the conversation's environment, the event's JSON on standard input."""

    type: Literal["command"]
    command: str
    #: Seconds.
    timeout: NotRequired[float]


class HttpHook(TypedDict):
    """A POST of the event's JSON."""

    type: Literal["http"]
    url: str
    #: Values may name ``$VAR`` or ``${VAR}`` from ``allowedEnvVars``.
    headers: NotRequired[dict[str, str]]
    allowedEnvVars: NotRequired[list[str]]
    #: Seconds.
    timeout: NotRequired[float]


#: One handler.
HookCommand: TypeAlias = CommandHook | HttpHook


def handler_timeout(handler: HookCommand, event: str) -> float:
    """The handler's timeout for ``event``, in seconds."""
    seconds = handler.get("timeout")
    if isinstance(seconds, int | float) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0:
        return float(seconds)
    return _PROMPT_TIMEOUT_SECONDS if event == "UserPromptSubmit" else DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class Matcher:
    """What a group's ``matcher`` matches: every value (``kind="all"``), exact names, or a regular expression."""

    kind: Literal["all", "names", "pattern"]
    names: tuple[str, ...] = ()
    pattern: re.Pattern[str] | None = None

    @staticmethod
    def parse(matcher: str | None) -> Matcher:
        """``*``, empty or absent: all; letters, digits, ``_`` and ``|``: exact names; else an unanchored regular
        expression."""
        text = (matcher or "").strip()
        if text in ("", "*"):
            return Matcher("all")
        if _NAMES.fullmatch(text):
            return Matcher("names", tuple(text.split("|")))
        try:
            return Matcher("pattern", pattern=re.compile(text))
        except re.error as error:
            raise ConfigError(f"Invalid matcher {text!r}: {error}") from error

    def matches(self, values: Sequence[str]) -> bool:
        """Whether any of ``values`` matches."""
        if self.kind == "all":
            return True
        if self.kind == "names":
            return any(value in self.names for value in values)
        assert self.pattern is not None
        return any(self.pattern.search(value) is not None for value in values)


@dataclass(frozen=True, slots=True)
class MatcherGroup:
    """A matcher and its handlers."""

    matcher: Matcher
    hooks: tuple[HookCommand, ...]


def _handler(value: Any) -> HookCommand:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Invalid hooks configuration: a handler must be an object, got {value!r}")
    kind = value.get("type")
    timeout = value.get("timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int | float)):
        raise ConfigError(f"Invalid hooks configuration: timeout must be a number, got {timeout!r}")
    if kind == "command":
        command = value.get("command")
        if not isinstance(command, str):
            raise ConfigError("Invalid hooks configuration: a command handler needs a command")
        handler: Any = {"type": "command", "command": command}
    elif kind == "http":
        url = value.get("url")
        headers = value.get("headers", {})
        allowed = value.get("allowedEnvVars", [])
        if not isinstance(url, str):
            raise ConfigError("Invalid hooks configuration: an http handler needs a url")
        if not isinstance(headers, Mapping) or not all(isinstance(item, str) for item in headers.values()):
            raise ConfigError("Invalid hooks configuration: headers must map names to strings")
        if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
            raise ConfigError("Invalid hooks configuration: allowedEnvVars must be a list of names")
        handler = {"type": "http", "url": url, "headers": dict(headers), "allowedEnvVars": list(allowed)}
    else:
        raise ConfigError(f"Invalid hooks configuration: unknown handler type {kind!r}")
    if timeout is not None:
        handler["timeout"] = float(timeout)
    return handler


@dataclass(frozen=True, slots=True)
class HooksConfig:
    """The ``hooks`` object of a settings file, by event name."""

    events: Mapping[str, tuple[MatcherGroup, ...]] = field(default_factory=dict)

    @staticmethod
    def from_value(value: Any) -> HooksConfig:
        """Parse the ``hooks`` object (``{"PreToolUse": [{"matcher": "Bash", "hooks": [...]}]}``)."""
        if not isinstance(value, Mapping):
            raise ConfigError("Invalid hooks configuration: expected an object of events")
        events: dict[str, tuple[MatcherGroup, ...]] = {}
        for event, groups in value.items():
            if not isinstance(groups, list):
                raise ConfigError(f"Invalid hooks configuration: {event} must be a list of matcher groups")
            parsed: list[MatcherGroup] = []
            for group in groups:
                if not isinstance(group, Mapping):
                    raise ConfigError(f"Invalid hooks configuration: a matcher group of {event} must be an object")
                matcher = group.get("matcher")
                if matcher is not None and not isinstance(matcher, str):
                    raise ConfigError(f"Invalid hooks configuration: a matcher of {event} must be a string")
                hooks = group.get("hooks", [])
                if not isinstance(hooks, list):
                    raise ConfigError(f"Invalid hooks configuration: hooks of {event} must be a list")
                parsed.append(MatcherGroup(Matcher.parse(matcher), tuple(_handler(item) for item in hooks)))
            events[str(event)] = tuple(parsed)
        return HooksConfig(events)

    @staticmethod
    def from_settings(settings: Mapping[str, Any]) -> HooksConfig:
        """Parse a whole settings file and take its ``hooks`` object; none is an empty configuration."""
        hooks = settings.get("hooks")
        return HooksConfig() if hooks is None else HooksConfig.from_value(hooks)

    def handlers(self, event: str, values: Sequence[str]) -> list[HookCommand]:
        """The handlers of ``event`` whose matcher matches one of ``values``; an event without a matcher target passes
        no values and runs every group."""
        return [
            handler
            for group in self.events.get(event, ())
            if not values or group.matcher.matches(values)
            for handler in group.hooks
        ]

    def unsupported(self) -> list[str]:
        """Configured events this bridge does not run (``SessionStart``, ``Notification``, ...)."""
        return [event for event in self.events if event not in SUPPORTED_EVENTS]
