"""Definition helpers for extensions, tools, prompt sections, hooks and wraps (port of ``harness/define.ts``).

Upstream's identity functions take object literals; here they take the same fields as keyword
arguments (snake_case) and build the definition dataclasses of :mod:`pi_durable.harness.types`::

    count = define_tool(
        name="count",
        description="Count from 1 to n",
        parameters=Type.Object({"n": Type.Number()}),
        execute=count_execute,
    )
    Coding = define_extension(
        name="coding",
        tools=[count],
        sections=[section("preamble", lambda input, context: "You are a concise coding assistant.", tag=False)],
        hooks=[hook(ToolTask, {"before_tool": guard})],
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar

from pi_ai.types import ConstrainedSamplingConfig
from pi_chord.context import Context

from ..types import AnyTask, Task
from .types import (
    Extension,
    HookRegistration,
    PromptInput,
    PromptSection,
    SectionWrap,
    ToolExecutionApi,
    ToolExecutionMode,
    ToolExecutionResult,
    ToolOutputLimits,
    ToolRegistration,
    ToolWrap,
    Wrap,
)

__all__ = ["define_extension", "define_tool", "hook", "section", "wrap_section", "wrap_tool"]

H = TypeVar("H")

_DEFAULT_PARAMETERS: Mapping[str, Any] = {"type": "object", "properties": {}}


def define_extension(
    name: str,
    *,
    tools: Sequence[ToolRegistration] = (),
    sections: Sequence[PromptSection] = (),
    hooks: Sequence[HookRegistration] = (),
    wraps: Sequence[Wrap] = (),
    tasks: Sequence[AnyTask] = (),
) -> Extension:
    """Build an extension: a named bundle of tools, sections, hooks, wraps, and tasks."""
    return Extension(name=name, tools=tools, sections=sections, hooks=hooks, wraps=wraps, tasks=tasks)


def define_tool(
    *,
    name: str,
    description: str = "",
    parameters: Any = None,
    execute: Callable[[Any, ToolExecutionApi, Context], Awaitable[ToolExecutionResult]],
    replay: Literal["safe", "unsafe"] | None = None,
    execution_mode: ToolExecutionMode | None = None,
    prepare_arguments: Callable[[Any], Any] | None = None,
    structured_output_schema: Mapping[str, Any] | None = None,
    callers: Sequence[Literal["model", "tools"]] | None = None,
    deferred: Literal[True] | None = None,
    output_limits: ToolOutputLimits | None = None,
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = None,
) -> ToolRegistration:
    """Build a tool registration; ``execute(args, api, context)`` receives arguments validated against ``parameters``.

    ``parameters`` is a JSON Schema (:class:`pi_ai.Type`) or a Pydantic model; default an empty object schema.
    """
    return ToolRegistration(
        name,
        description,
        dict(_DEFAULT_PARAMETERS) if parameters is None else parameters,
        constrained_sampling=constrained_sampling,
        execute=execute,
        replay=replay,
        execution_mode=execution_mode,
        prepare_arguments=prepare_arguments,
        structured_output_schema=structured_output_schema,
        callers=callers,
        deferred=deferred,
        output_limits=output_limits,
    )


def section(
    key: str,
    render: Callable[[PromptInput, Context], str | Awaitable[str | None] | None],
    *,
    tag: bool | None = None,
) -> PromptSection:
    """A prompt section; tagged (``<key>...</key>``) unless ``tag`` is ``False``."""
    return PromptSection(key=key, render=render, tag=tag)


def hook(task: Task[Any, Any, Any, H], handlers: H) -> HookRegistration:
    """Hook handlers for tasks with ``task``'s name; ``handlers`` is the task's hook table, keyed by snake_case name."""
    return HookRegistration(task=task.definition.name, handlers=dict(handlers))  # type: ignore[call-overload]


def wrap_tool(tool: ToolRegistration, wrapper: Callable[[ToolRegistration], ToolRegistration]) -> Wrap:
    """Wrap the tool named like ``tool`` wherever the wrapping extension is selected.

    A wrapper usually returns ``dataclasses.replace(tool, execute=...)``, upstream's ``{ ...tool, execute }``.
    """
    return ToolWrap(tool=tool.name, wrap=wrapper)


def wrap_section(key: str, wrapper: Callable[[PromptSection], PromptSection]) -> Wrap:
    """Wrap the section ``key`` wherever the wrapping extension is selected."""
    return SectionWrap(section=key, wrap=wrapper)
