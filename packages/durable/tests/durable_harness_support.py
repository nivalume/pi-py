"""Shared helpers of the Harness tests (port of ``test/harness-support.ts``, ``task-support.ts`` and ``chat-support.ts``).

- Messages: :func:`user`, :func:`assistant`, :func:`tool_result`, :func:`system`, :func:`describe_message`.
- Registries: :func:`tool`, :func:`add_tool`, :func:`add_task`, :func:`add_hooks`, :func:`add_section` (each
  returns an :class:`Installed` whose ``dispose()`` uninstalls it), :func:`counting_reader`.
- Harnesses: :func:`open_harness` (a registry with named no-op tools), :func:`open_tasks` (a registry with
  tasks, collected reports and infos, optional clock/settings/onWake/setTimer), :func:`chat_setup` and
  :func:`open_chat` (the faux provider, models, a registry and live settings that survive a reopen).
- Waiting: :func:`flush`, :func:`eventually`, :func:`settled`, :func:`wait_for` (real time), :func:`aborted`,
  :func:`unanswered` (a faux step that blocks until the generation is cancelled).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from durable_session_support import context, eventually, flush

from pi_ai import AbortSignal, Type, create_models, faux_provider
from pi_ai.types import AssistantMessage, SystemMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from pi_chord.context import Context
from pi_durable import (
    Harness,
    create_registry,
    define_extension,
    define_tool,
    hook,
    section,
)
from pi_durable.types import ReportInfo, Storage

__all__ = [
    "ChatSetup",
    "Installed",
    "aborted",
    "aborted_with",
    "add_hooks",
    "add_section",
    "add_task",
    "add_tool",
    "all_entries",
    "assistant",
    "chat_setup",
    "completed",
    "context",
    "counting_reader",
    "describe_message",
    "eventually",
    "flush",
    "open_chat",
    "open_harness",
    "open_tasks",
    "settled",
    "system",
    "text_of",
    "tool",
    "tool_result",
    "tools_named",
    "unanswered",
    "user",
    "wait_for",
]


async def _no_output(_args: Any, _api: Any, _context: Context) -> Any:
    return {"output": []}


def tool(name: str, description: str | None = None) -> Any:
    """A tool with an empty object schema that returns no output."""
    return define_tool(
        name=name,
        description=f"{name} tool" if description is None else description,
        parameters=Type.Object({}),
        execute=_no_output,
    )


@dataclass(slots=True)
class OpenedHarness:
    harness: Harness
    registry: Any


async def open_harness(
    storage: Storage,
    tool_names: Sequence[str] = (),
    *,
    registry: Any = None,
    on_report: Callable[[BaseException, ReportInfo], None] | None = None,
) -> OpenedHarness:
    """Open a Harness with a fresh registry holding the named tools."""
    registry = registry if registry is not None else create_registry()
    if len(tool_names) > 0:
        registry.install(define_extension("tools", tools=[tool(name) for name in tool_names]))
    options: Any = {"models": create_models(), "registry": registry}
    if on_report is not None:
        options["onReport"] = on_report
    harness = await Harness.open(storage, options, context)
    return OpenedHarness(harness, registry)


def user(text: str) -> UserMessage:
    return UserMessage(text, timestamp=1)


def assistant(text: str, *, calls: Sequence[str] = (), stop_reason: str | None = None) -> AssistantMessage:
    tool_calls = [ToolCall(id, f"tool-{id}", {}) for id in calls]
    return AssistantMessage(
        [TextContent(text), *tool_calls],
        api="faux",
        provider="faux",
        model="faux",
        usage=Usage(),
        stop_reason=stop_reason if stop_reason is not None else ("toolUse" if tool_calls else "stop"),  # type: ignore[arg-type]
        timestamp=2,
    )


def tool_result(id: str, text: str | None = None) -> ToolResultMessage:
    return ToolResultMessage(
        id, f"tool-{id}", [TextContent(f"result {id}" if text is None else text)], False, timestamp=3
    )


def system(sections: dict[str, str | None]) -> SystemMessage:
    return SystemMessage("", sections=sections, timestamp=4)


def describe_message(message: Any) -> str:
    """Compact message rendering for assertions."""
    role = message.role
    if role == "user":
        return f"user:{message.content}"
    if role == "assistant":
        text = next((content for content in message.content if isinstance(content, TextContent)), None)
        return f"assistant:{'' if text is None else text.text}"
    if role == "toolResult":
        text = next((content for content in message.content if isinstance(content, TextContent)), None)
        shown = "error" if message.is_error else ("" if text is None else text.text)
        return f"result:{message.tool_call_id}:{shown}"
    return f"system:{','.join((message.sections or {}).keys())}"


class Installed:
    """Uninstalls what one of the helpers below installed."""

    def __init__(self, registry: Any, extension: Any) -> None:
        self._registry = registry
        self._extension = extension

    def dispose(self) -> None:
        self._registry.uninstall(self._extension)


def _install_one(registry: Any, extension: Any) -> Installed:
    registry.install(extension)
    return Installed(registry, extension)


def add_tool(registry: Any, registration: Any, name: str | None = None) -> Installed:
    """Install a one-tool extension named after the tool."""
    return _install_one(registry, define_extension(name or f"tool:{registration.name}", tools=[registration]))


def add_task(registry: Any, task: Any, name: str | None = None) -> Installed:
    """Install a one-task extension named after the task."""
    return _install_one(registry, define_extension(name or f"task:{task.definition.name}", tasks=[task]))


_hook_extensions = [0]


def add_hooks(registry: Any, task: Any, handlers: dict[str, Any], name: str | None = None) -> Installed:
    """Install an extension with one hook registration for ``task``."""
    if name is None:
        _hook_extensions[0] += 1
        name = f"hooks:{_hook_extensions[0]}"
    return _install_one(registry, define_extension(name, hooks=[hook(task, handlers)]))


def add_section(registry: Any, key: str, render: Any, *, tag: bool | None = None, name: str | None = None) -> Installed:
    """Install a one-section extension named after the section."""
    return _install_one(registry, define_extension(name or f"section:{key}", sections=[section(key, render, tag=tag)]))


# ─── task-support ───────────────────────────────────────────────────────────


async def aborted(signal: AbortSignal) -> Any:
    """Raise the signal's reason once it aborts; for handlers that block until cancelled."""
    reason = await signal.wait()
    raise reason if isinstance(reason, BaseException) else RuntimeError(str(reason))


async def settled(awaitable: Awaitable[Any]) -> bool:
    """Whether ``awaitable`` has settled after pending work flushes."""
    future = asyncio.ensure_future(awaitable)
    await flush()
    if future.done():
        if not future.cancelled():
            future.exception()
        return True
    return False


@dataclass(slots=True)
class OpenedTasks:
    harness: Harness
    registry: Any
    reports: list[BaseException]
    infos: list[ReportInfo]


async def open_tasks(
    storage: Storage,
    tasks: Sequence[Any],
    *,
    registry: Any = None,
    now: Callable[[], float] | None = None,
    settings: Any = None,
    on_wake: Any = None,
    set_timer: Any = None,
) -> OpenedTasks:
    """Open a Harness whose registry holds ``tasks``; reports are collected, their infos in ``infos``."""
    registry = registry if registry is not None else create_registry()
    if len(tasks) > 0:
        registry.install(define_extension("tasks", tasks=list(tasks)))
    reports: list[BaseException] = []
    infos: list[ReportInfo] = []

    def on_report(error: BaseException, info: ReportInfo) -> None:
        reports.append(error)
        infos.append(info)

    options: Any = {"models": create_models(), "registry": registry, "onReport": on_report}
    if now is not None:
        options["now"] = now
    if settings is not None:
        options["settings"] = settings
    if on_wake is not None:
        options["onWake"] = on_wake
    if set_timer is not None:
        options["setTimer"] = set_timer
    harness = await Harness.open(storage, options, context)
    return OpenedTasks(harness, registry, reports, infos)


def completed(result: Any) -> Any:
    """Next state that completes a task with ``result``."""
    return {"status": "terminal", "outcome": {"status": "completed", "result": result}}


def aborted_with(reason: str) -> Any:
    """Next state that aborts a task."""
    return {"status": "terminal", "outcome": {"status": "aborted", "reason": reason}}


class CountingReader:
    """Registry reader that counts live subscriptions."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._count = 0

    def snapshot(self) -> Any:
        return self._registry.snapshot()

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._count += 1
        unsubscribe = self._registry.subscribe(listener)
        active = [True]

        def stop() -> None:
            if not active[0]:
                return
            active[0] = False
            self._count -= 1
            unsubscribe()

        return stop

    def subscriptions(self) -> int:
        return self._count


def counting_reader(registry: Any) -> CountingReader:
    return CountingReader(registry)


# ─── chat-support ───────────────────────────────────────────────────────────


@dataclass(slots=True, eq=False)
class ChatSetup:
    """Models and registry that survive a close/reopen, like a host process's own objects."""

    faux: Any
    models: Any
    registry: Any
    reports: list[BaseException] = field(default_factory=list)
    #: Live Harness settings; tests assign keys to change them between decisions.
    settings: dict[str, Any] = field(default_factory=dict)
    now: Callable[[], float] = field(default=lambda: int(time.time() * 1000))


def chat_setup(**options: Any) -> ChatSetup:
    faux = faux_provider(**options)
    models = create_models()
    models.set_provider(faux.provider)
    return ChatSetup(faux, models, create_registry())


@dataclass(slots=True)
class OpenedChat:
    harness: Harness
    root: Any


async def open_chat(storage: Storage, setup: ChatSetup, *, env: Any = None) -> OpenedChat:
    """Open a Harness over ``storage`` and return its root, configured with the faux model on first creation."""
    options: Any = {
        "models": setup.models,
        "registry": setup.registry,
        "settings": setup.settings,
        "now": lambda: setup.now(),
        "onReport": lambda error, _info: setup.reports.append(error),
    }
    if env is not None:
        options["env"] = env if callable(env) else (lambda _target, _context: env)
    harness = await Harness.open(storage, options, context)
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    return OpenedChat(harness, root)


async def all_entries(conversation: Any, call_context: Context = context) -> list[Any]:
    """Raw entries of a conversation, oldest first."""
    page = await conversation.entries({}, 1000, None, call_context)
    return list(reversed(page["items"]))


def text_of(message: Any) -> str | None:
    """Text of the first text content of a message."""
    if message is None or message.role == "system":
        return None
    if isinstance(message.content, str):
        return message.content
    text = next((content for content in message.content if isinstance(content, TextContent)), None)
    return None if text is None else text.text


async def wait_for(check: Callable[[], bool | Awaitable[bool]], timeout_ms: float = 5000) -> None:
    """Poll ``check`` in real time until it holds; for waits that span throttle windows and timers."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        result = check()
        if not isinstance(result, bool):
            result = await result
        if result:
            return
        if time.monotonic() > deadline:
            raise AssertionError("Condition was not reached")
        await asyncio.sleep(0.005)


@dataclass(slots=True)
class Unanswered:
    step: Any
    reached: asyncio.Future[None]


def unanswered() -> Unanswered:
    """Faux response that never answers; the run stays busy until its generation is cancelled.

    ``reached`` resolves once the request was sent, after the generation's preparation and request commits.
    """
    reached: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def step(_context: Any, options: Any, _state: Any, _model: Any) -> Any:
        if not reached.done():
            reached.set_result(None)
        signal: AbortSignal = options["signal"]
        reason = await signal.wait()
        raise reason if isinstance(reason, BaseException) else RuntimeError(str(reason))

    return Unanswered(step, reached)


def tools_named(setup: ChatSetup, *names: str) -> list[Any]:
    """The installed tools with these names, as ``configure()`` takes them."""
    installed = setup.registry.snapshot().tools()
    found: list[Any] = []
    for name in names:
        match = next((entry for entry in installed if entry["tool"].name == name), None)
        if match is None:
            raise RuntimeError(f"Tool {name} is not installed")
        found.append(match["tool"])
    return found
