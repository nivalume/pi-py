"""The agent document, settings resolution, and agent resolution (port of ``harness/agent.ts``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeVar

from pi_chord.json import copy_json

from ..documents import define_doc
from ..session import document_retention
from ..types import ConversationId, ConversationRecord, RewindableConversationDocToken, Tx
from .types import (
    Agent,
    AgentChange,
    AgentState,
    CompactionPolicy,
    ConversationRetryPolicy,
    Extension,
    HarnessSettings,
    ProgressPolicy,
    PromptSection,
    RegistrySnapshot,
    SectionWrap,
    Settings,
    ToolRegistration,
    ToolWrap,
)

__all__ = [
    "DEFAULT_APPROVAL_TIMEOUT_MS",
    "DEFAULT_COMPACTION_POLICY",
    "DEFAULT_PROGRESS_POLICY",
    "DEFAULT_RESTART_LIMIT",
    "DEFAULT_RETRY_POLICY",
    "INSTRUCTIONS_KEY",
    "AgentDoc",
    "add_tools",
    "agent_hooks",
    "configure",
    "create_agent",
    "resolve_agent",
    "resolve_settings",
    "section_extension",
    "tool_extension",
]

T = TypeVar("T")

DEFAULT_RETRY_POLICY: Final[ConversationRetryPolicy] = {
    "enabled": True,
    "maxRetries": 3,
    "baseDelayMs": 2000,
    "maxAgentDelayMs": 60000,
}

DEFAULT_COMPACTION_POLICY: Final[CompactionPolicy] = {
    "enabled": True,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
    "backgroundTokens": 32768,
}

DEFAULT_PROGRESS_POLICY: Final[ProgressPolicy] = {"partialIntervalMs": 100, "outputIntervalMs": 100}

#: The reserved section key of the agent's ``instructions``.
INSTRUCTIONS_KEY: Final = "instructions"

#: Built-in agent document; rewindable so forks start from the agent at their fork entry.
AgentDoc: RewindableConversationDocToken[AgentState] = define_doc(
    kind="pi.agent",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: {},
    checkpoint_when=lambda *_args: True,
)

#: Default ``settings.contextRetentionMs``: ten minutes.
_DEFAULT_CONTEXT_RETENTION_MS: Final = 600_000

#: Default ``settings.restartLimit``.
DEFAULT_RESTART_LIMIT: Final = 5

#: Default ``settings.approvalTimeoutMs``: fifteen minutes.
DEFAULT_APPROVAL_TIMEOUT_MS: Final = 15 * 60 * 1000


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _restart_limit(limit: object) -> float:
    """A restart limit as given when it is a number of restarts, ``1`` or more, rounded down (``math.inf`` never
    parks); else the default."""
    if _is_number(limit) and limit >= 1:  # type: ignore[operator]
        return limit if limit == math.inf else math.floor(limit)  # type: ignore[arg-type,return-value]
    return DEFAULT_RESTART_LIMIT


def _given(settings: Any, key: str) -> Any:
    return None if settings is None else settings.get(key)


def resolve_settings(settings: HarnessSettings | None) -> Settings:
    """Resolve the host settings: every field over its built-in default, object fields merged."""
    extensions = _given(settings, "extensions")
    progress = _given(settings, "progress") or {}
    partial = progress.get("partialIntervalMs")
    output = progress.get("outputIntervalMs")
    resolved: dict[str, Any] = {} if extensions is None else {"extensions": extensions}
    resolved.update(
        {
            "stream": {**(_given(settings, "stream") or {})},
            "retry": {**DEFAULT_RETRY_POLICY, **(_given(settings, "retry") or {})},
            "compaction": {**DEFAULT_COMPACTION_POLICY, **(_given(settings, "compaction") or {})},
            # Field by field, so an explicitly absent interval keeps its default.
            "progress": {
                "partialIntervalMs": DEFAULT_PROGRESS_POLICY["partialIntervalMs"] if partial is None else partial,
                "outputIntervalMs": DEFAULT_PROGRESS_POLICY["outputIntervalMs"] if output is None else output,
            },
            "toolExecution": _given(settings, "toolExecution") or "parallel",
            "steeringMode": _given(settings, "steeringMode") or "one-at-a-time",
            "followUpMode": _given(settings, "followUpMode") or "one-at-a-time",
            "contextRetentionMs": _first(_given(settings, "contextRetentionMs"), _DEFAULT_CONTEXT_RETENTION_MS),
            "documentRetentionMs": document_retention(_given(settings, "documentRetentionMs")),
            "afterFailedRun": _given(settings, "afterFailedRun") or "keep",
            "restartLimit": _restart_limit(_given(settings, "restartLimit")),
            "approvalTimeoutMs": _first(_given(settings, "approvalTimeoutMs"), DEFAULT_APPROVAL_TIMEOUT_MS),
        }
    )
    return resolved  # type: ignore[return-value]


def _first(value: Any, default: Any) -> Any:
    return default if value is None else value


async def configure(tx: Tx, conversation_id: ConversationId, change: AgentChange) -> None:
    """Apply one change to ``pi.agent``: a given key replaces the stored field, ``None`` clears it, absent changes nothing."""
    state = await tx.doc(AgentDoc, conversation_id)
    _apply_change(state, change)


async def add_tools(tx: Tx, conversation_id: ConversationId, added: Sequence[str]) -> None:
    """``addTools`` of a tool round, applied to ``tools`` and ``modelTools``, so the next request offers the tools.

    A list gets each name it lacks appended, and ``{"remove"}`` loses the names. An unset ``tools`` already
    enables every tool, so nothing is written for it. An unset or object ``modelTools`` also gets each name
    appended to its ``add``, which loads deferred tools and has no effect for others. By name, without the
    registry: a tool that is not registered now is offered once it is again.
    """
    state: Any = await tx.doc(AgentDoc, conversation_id)
    tools = state.get("tools")
    if isinstance(tools, list):
        for name in added:
            if name not in tools:
                tools.append(name)
    elif tools is not None and any(name in added for name in tools["remove"]):
        state["tools"] = {"remove": [name for name in tools["remove"] if name not in added]}
    model_tools = state.get("modelTools")
    if isinstance(model_tools, list):
        for name in added:
            if name not in model_tools:
                model_tools.append(name)
        return
    # Assigned whole; a document write of an equal value records no change.
    remove = [name for name in (model_tools.get("remove", []) if model_tools is not None else []) if name not in added]
    add = list(model_tools.get("add", []) if model_tools is not None else [])
    for name in added:
        if name not in add:
            add.append(name)
    state["modelTools"] = {**({} if len(remove) == 0 else {"remove": remove}), "add": add}


def _names(items: Sequence[Any]) -> list[str]:
    return [item.name for item in items]


def _apply_change(state: Any, change: AgentChange) -> None:
    def set_field(key: str, value: Any) -> None:
        if key not in change:
            return
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value

    model = change.get("model")
    set_field("model", None if model is None else {"provider": model["provider"], "modelId": model["modelId"]})
    set_field("thinkingLevel", change.get("thinkingLevel"))
    extensions: Any = change.get("extensions")
    set_field(
        "extensions",
        None
        if extensions is None
        else _names(extensions)
        if not isinstance(extensions, dict)
        else {
            **({} if extensions.get("add") is None else {"add": _names(extensions["add"])}),
            **({} if extensions.get("remove") is None else {"remove": _names(extensions["remove"])}),
        },
    )
    tools: Any = change.get("tools")
    set_field(
        "tools",
        None
        if tools is None
        else _names(tools)
        if not isinstance(tools, dict)
        else {"remove": _names(tools["remove"])},
    )
    model_tools: Any = change.get("modelTools")
    set_field(
        "modelTools",
        None
        if model_tools is None
        else _names(model_tools)
        if not isinstance(model_tools, dict)
        else {
            **({} if model_tools.get("remove") is None else {"remove": _names(model_tools["remove"])}),
            **({} if model_tools.get("add") is None else {"add": _names(model_tools["add"])}),
        },
    )
    set_field("instructions", change.get("instructions"))
    set_field("cwd", change.get("cwd"))


async def create_agent(tx: Tx, conversation: ConversationRecord) -> None:
    """Built-in part of every Harness commit that creates or forks a conversation, for ``pi.agent``.

    A fork keeps its ``asOf`` copy; a new task-owned conversation copies the stored agent of its owner
    task's conversation; a new ownerless one starts empty.
    """
    if "parent" in conversation:
        return
    agent = await tx.doc(AgentDoc, conversation["id"])
    owner_edge = conversation.get("owner")
    if owner_edge is None:
        return
    owner = await tx.doc(AgentDoc, owner_edge["conversationId"])
    agent.update(copy_json(owner))  # type: ignore[arg-type]


def agent_hooks(agent: Agent, task_name: str) -> list[tuple[str, Any]]:
    """Handlers of the selected extensions' hooks for a task name, in extension order: ``(extension, handlers)``."""
    handlers: list[tuple[str, Any]] = []
    for extension in agent["extensions"]:
        for registration in extension.hooks:
            if registration.task == task_name:
                handlers.append((extension.name, registration.handlers))
    return handlers


def section_extension(agent: Agent, key: str) -> str | None:
    """The extension whose code a section of ``agent`` runs: the last selected one that provides or wraps it."""
    found: str | None = None
    for extension in agent["extensions"]:
        provides = any(section.key == key for section in extension.sections)
        wraps = any(isinstance(wrap, SectionWrap) and wrap.section == key for wrap in extension.wraps)
        if provides or wraps:
            found = extension.name
    return found


def tool_extension(agent: Agent, name: str) -> str | None:
    """The extension whose code a tool of ``agent`` runs: the last selected one that provides or wraps it."""
    found: str | None = None
    for extension in agent["extensions"]:
        provides = any(tool.name == name for tool in extension.tools)
        wraps = any(isinstance(wrap, ToolWrap) and wrap.tool == name for wrap in extension.wraps)
        if provides or wraps:
            found = extension.name
    return found


def resolve_agent(
    state: AgentState | None,
    snapshot: RegistrySnapshot,
    settings: Settings,
    report: Callable[[BaseException, str], None],
) -> Agent:
    """Resolve an agent from its stored state (``None``: every field unset), a registry snapshot, and settings.

    A wrapper that raises or renames drops its target and is reported (``report(error, extension)``); a
    wrapper without a target does nothing.
    """
    stored: Any = state or {}
    extensions = _select_extensions(stored.get("extensions"), snapshot, settings)

    composed: dict[str, ToolRegistration] = {}
    for extension in extensions:
        for tool in extension.tools:
            composed[tool.name] = tool
    sections: dict[str, PromptSection] = {}
    for extension in extensions:
        for section in extension.sections:
            sections[section.key] = section
    for extension in extensions:
        for wrap in extension.wraps:

            def wrap_report(error: BaseException, name: str = extension.name) -> None:
                report(error, name)

            if isinstance(wrap, ToolWrap):
                _apply_wrap(composed, wrap.tool, wrap.wrap, lambda tool: tool.name, wrap_report)
            else:
                _apply_wrap(sections, wrap.section, wrap.wrap, lambda section: section.key, wrap_report)

    enabled = _filter_tools(list(composed.values()), stored.get("tools"))

    def callable_by(caller: str) -> Callable[[ToolRegistration], bool]:
        return lambda tool: tool.callers is None or caller in tool.callers

    model_callable = [tool for tool in enabled if callable_by("model")(tool)]
    tools = _offered_tools(model_callable, stored.get("modelTools"))
    offered = {id(tool) for tool in tools}
    deferred = [tool for tool in model_callable if tool.deferred is True and id(tool) not in offered]
    callable_tools = [tool for tool in enabled if callable_by("tools")(tool)]

    instructions = stored.get("instructions")
    agent_sections = list(sections.values())
    if instructions is not None:
        text: str = instructions
        agent_sections.append(PromptSection(key=INSTRUCTIONS_KEY, render=lambda _input, _context: text))

    agent: dict[str, Any] = {}
    if stored.get("model") is not None:
        agent["model"] = stored["model"]
    agent["thinkingLevel"] = stored.get("thinkingLevel") or "off"
    agent["extensions"] = extensions
    agent["tools"] = tools
    agent["callable"] = callable_tools
    agent["deferred"] = deferred
    agent["sections"] = agent_sections
    if instructions is not None:
        agent["instructions"] = instructions
    if stored.get("cwd") is not None:
        agent["cwd"] = stored["cwd"]
    return agent  # type: ignore[return-value]


def _offered_tools(tools: list[ToolRegistration], selection: Any) -> list[ToolRegistration]:
    """The tools offered to the model, as ``AgentState["modelTools"]`` describes."""
    if isinstance(selection, list):
        return _filter_tools(tools, selection)
    removed = set((selection or {}).get("remove", []))
    kept = [tool for tool in tools if tool.deferred is not True and tool.name not in removed]
    # Loaded deferred tools follow in load order, so loading one appends to the declared tools.
    by_name = {tool.name: tool for tool in tools}
    loaded: list[ToolRegistration] = []
    for name in dict.fromkeys((selection or {}).get("add", [])):
        tool = by_name.get(name)
        if tool is not None and tool.deferred is True and name not in removed:
            loaded.append(tool)
    return [*kept, *loaded]


def _filter_tools(tools: list[ToolRegistration], selection: Any) -> list[ToolRegistration]:
    """``tools`` through a stored filter: a list selects exactly its names, in its order; ``{"remove"}`` drops names."""
    if selection is None:
        return tools
    if isinstance(selection, list):
        by_name = {tool.name: tool for tool in tools}
        return [by_name[name] for name in dict.fromkeys(selection) if name in by_name]
    removed = set(selection["remove"])
    return [tool for tool in tools if tool.name not in removed]


def _select_extensions(stored: Any, snapshot: RegistrySnapshot, settings: Settings) -> list[Extension]:
    """Selected installed extensions: the stored list, or the default selection edited by ``{"add", "remove"}``."""
    if isinstance(stored, list):
        selected: list[str] = list(stored)
    else:
        defaults = settings.get("extensions")
        base = [extension.name for extension in (defaults if defaults is not None else snapshot.installed())]
        edit = stored or {}
        removed = set(edit.get("remove", []))
        selected = [name for name in [*base, *edit.get("add", [])] if name not in removed]
    extensions: list[Extension] = []
    for name in dict.fromkeys(selected):
        extension = snapshot.extension(name)
        if extension is not None:
            extensions.append(extension)
    return extensions


def _apply_wrap(
    items: dict[str, T],
    target: str,
    wrap: Callable[[T], T],
    name_of: Callable[[T], str],
    report: Callable[[BaseException], None],
) -> None:
    item = items.get(target)
    if item is None:
        return
    try:
        wrapped = wrap(item)
        if name_of(wrapped) != target:
            raise RuntimeError(f"Wrapper renamed {target} to {name_of(wrapped)}")
        items[target] = wrapped
    except Exception as error:
        del items[target]
        report(error)
