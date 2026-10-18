"""The application-owned registry of extensions (port of ``harness/registry.ts``)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Final

from ..types import AnyTask
from .agent import INSTRUCTIONS_KEY
from .compaction import CompactionTask
from .generation import GenerationTask
from .tool import ToolTask
from .types import Extension, Registry, RegistrySection, RegistrySnapshot, RegistryTool

__all__ = ["BUILTIN_TASKS", "create_registry"]

_SECTION_KEY: Final = re.compile(r"[a-z][a-z0-9_-]*")

#: Built-in task definitions every registry holds; they are not an extension and cannot be removed or replaced.
BUILTIN_TASKS: Final[Sequence[AnyTask]] = (GenerationTask, ToolTask, CompactionTask)


class _RegistryState:
    """Immutable published registry state (a :class:`RegistrySnapshot`)."""

    __slots__ = ("_by_name", "_extensions", "_tasks")

    def __init__(self, extensions: Sequence[Extension]) -> None:
        self._extensions = tuple(extensions)
        self._by_name = {extension.name: extension for extension in self._extensions}
        tasks: dict[str, AnyTask] = {task.definition.name: task for task in BUILTIN_TASKS}
        for extension in self._extensions:
            for task in extension.tasks:
                name = task.definition.name
                if name in tasks:
                    raise RuntimeError(f"Task {name} of extension {extension.name} is already installed")
                tasks[name] = task
        self._tasks = tasks

    def installed(self) -> Sequence[Extension]:
        return self._extensions

    def extension(self, name: str) -> Extension | None:
        return self._by_name.get(name)

    def tools(self) -> Sequence[RegistryTool]:
        return [{"extension": extension, "tool": tool} for extension in self._extensions for tool in extension.tools]

    def sections(self) -> Sequence[RegistrySection]:
        return [
            {"extension": extension, "section": section}
            for extension in self._extensions
            for section in extension.sections
        ]

    def tasks(self) -> Sequence[AnyTask]:
        return list(self._tasks.values())

    def task(self, name: str) -> AnyTask | None:
        return self._tasks.get(name)


class _RegistryImpl:
    """Application-owned registry; install and uninstall publish a new snapshot synchronously."""

    __slots__ = ("_current", "_listeners")

    def __init__(self) -> None:
        self._current = _RegistryState(())
        self._listeners: dict[Callable[[], None], None] = {}

    def snapshot(self) -> RegistrySnapshot:
        return self._current

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        token = _Listener(listener)
        self._listeners[token] = None
        return lambda: self._listeners.pop(token, None) and None

    def install(self, extension: Extension) -> None:
        _validate_extension(extension)
        current = self._current.installed()
        index = next((at for at, installed in enumerate(current) if installed.name == extension.name), -1)
        following = (
            [*current, extension]
            if index < 0
            else [extension if at == index else installed for at, installed in enumerate(current)]
        )
        self._publish(following)

    def uninstall(self, extension: Extension) -> None:
        current = self._current.installed()
        if not any(installed.name == extension.name for installed in current):
            return
        self._publish([installed for installed in current if installed.name != extension.name])

    def _publish(self, extensions: Sequence[Extension]) -> None:
        """Build and validate the next state, which raises on a task name collision, then publish it synchronously."""
        self._current = _RegistryState(extensions)
        for listener in list(self._listeners):
            listener()


class _Listener:
    """A subscription: the same function subscribed twice is two subscriptions, as upstream's ``Set`` of closures."""

    __slots__ = ("_listener",)

    def __init__(self, listener: Callable[[], None]) -> None:
        self._listener = listener

    def __call__(self) -> None:
        self._listener()


def _validate_extension(extension: Extension) -> None:
    """Unique tool names and section keys within one extension; valid, unreserved section keys."""
    tools: set[str] = set()
    for tool in extension.tools:
        if tool.name in tools:
            raise RuntimeError(f"Extension {extension.name} has two tools named {tool.name}")
        tools.add(tool.name)
    sections: set[str] = set()
    for section in extension.sections:
        key = section.key
        if not isinstance(key, str) or not _SECTION_KEY.fullmatch(key):
            raise TypeError(f"Section key {json.dumps(key)} must match /^[a-z][a-z0-9_-]*$/")
        if key == INSTRUCTIONS_KEY:
            raise RuntimeError(f"Section key {key} is reserved for the agent's instructions")
        if key in sections:
            raise RuntimeError(f"Extension {extension.name} has two sections with key {key}")
        sections.add(key)


def create_registry() -> Registry:
    """Create an application-owned registry holding only the built-in tasks."""
    return _RegistryImpl()
