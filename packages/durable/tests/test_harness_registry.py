"""Port of ``test/harness-registry.test.ts``."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest
from durable_session_support import context

from pi_ai import Type
from pi_durable import (
    CompactionTask,
    GenerationTask,
    ToolRegistration,
    ToolTask,
    create_registry,
    define_extension,
    define_task,
    hook,
    section,
    wrap_section,
    wrap_tool,
)
from pi_durable.harness.agent import agent_hooks, resolve_agent, resolve_settings
from pi_durable.harness.types import PromptSection


@dataclass(slots=True, eq=False)
class AppTool(ToolRegistration):
    snippet: str | None = None


async def _no_output(_args: Any, _api: Any, _context: Any) -> Any:
    return {"output": []}


def tool(name: str, **extra: Any) -> AppTool:
    fields: dict[str, Any] = {"description": f"{name} tool", **extra}
    return AppTool(name, fields.pop("description"), Type.Object({}), execute=_no_output, **fields)


async def _nothing(*_args: Any) -> None:
    return None


def task(name: str, version: int = 1) -> Any:
    return define_task(
        name=name, version=version, initial=lambda _input: {"phase": "run"}, phases={"run": _nothing}, abort=_nothing
    )


def names(items: Any) -> list[str]:
    return [item.name for item in items]


def resolve(state: Any, snapshot: Any, settings: Any = None, reports: list[Any] | None = None) -> Any:
    collected = reports if reports is not None else []
    return resolve_agent(
        state, snapshot, resolve_settings(settings or {}), lambda error, _extension: collected.append(error)
    )


async def rendered(agent: Any) -> list[tuple[str, str | None]]:
    async def nothing(*_args: Any) -> None:
        return None

    class Read:
        snapshot = staticmethod(nothing)
        snapshot_as_of = staticmethod(nothing)

    prompt_input: Any = {"conversationId": 1, "agent": agent, "env": None, "shown": {}, "read": Read()}
    result: list[tuple[str, str | None]] = []
    for item in agent["sections"]:
        text = item.render(prompt_input, context)
        if hasattr(text, "__await__"):
            text = await text
        result.append((item.key, text))
    return result


def test_installs_replaces_in_place_and_uninstalls_extensions_by_name() -> None:
    registry = create_registry()
    listener: list[str] = []
    registry.subscribe(lambda: listener.append(",".join(names(registry.snapshot().installed()))))
    a = define_extension("a", tools=[tool("read", snippet="Read files")])
    b = define_extension("b", tools=[tool("read"), tool("bash")])
    registry.install(a)
    registry.install(b)
    before = registry.snapshot()
    # A new object with an installed name replaces it at its position.
    a2 = define_extension("a", tools=[tool("grep")])
    registry.install(a2)
    assert names(registry.snapshot().installed()) == ["a", "b"]
    assert registry.snapshot().extension("a") is a2
    assert [f"{entry['tool'].name}@{entry['extension'].name}" for entry in registry.snapshot().tools()] == [
        "grep@a",
        "read@b",
        "bash@b",
    ]
    # Old snapshots stay as they were.
    assert before.extension("a") is a
    assert before.tools()[0]["tool"].snippet == "Read files"  # type: ignore[attr-defined]
    # Uninstall matches the name, whichever object; a later install appends.
    registry.uninstall(a)
    registry.uninstall(a)
    assert names(registry.snapshot().installed()) == ["b"]
    registry.install(a)
    assert names(registry.snapshot().installed()) == ["b", "a"]
    assert listener == ["a", "a,b", "a,b", "b", "b,a"]


def test_validates_the_registry_as_it_would_be_after_an_install_and_publishes_nothing_when_invalid() -> None:
    registry = create_registry()
    registry.install(define_extension("tasks", tasks=[task("app.index")]))
    published: list[int] = []
    registry.subscribe(lambda: published.append(1))
    before = registry.snapshot()
    invalid = [
        (define_extension("x", tools=[tool("read"), tool("read")]), "two tools named read"),
        (define_extension("x", sections=[section("a", lambda *_: "1"), section("a", lambda *_: "2")]), "two sections"),
        (define_extension("x", sections=[section("Bad Key", lambda *_: "")]), "must match"),
        (define_extension("x", sections=[section("instructions", lambda *_: "")]), "reserved"),
        (define_extension("x", tasks=[task("pi.generation")]), "already installed"),
        (define_extension("x", tasks=[task("app.index")]), "already installed"),
    ]
    for extension, message in invalid:
        with pytest.raises(Exception, match=message):
            registry.install(extension)
    assert registry.snapshot() is before
    assert published == []
    # Replacing the extension that holds a task name is valid: the check runs on the state after replacement.
    registry.install(define_extension("tasks", tasks=[task("app.index", 2)]))
    found = registry.snapshot().task("app.index")
    assert found is not None and found.definition.version == 2


def test_always_holds_the_built_in_tasks_which_are_not_an_extension() -> None:
    registry = create_registry()
    assert list(registry.snapshot().installed()) == []
    assert list(registry.snapshot().tasks()) == [GenerationTask, ToolTask, CompactionTask]
    custom = task("app.custom")
    registry.install(define_extension("custom", tasks=[custom]))
    assert list(registry.snapshot().tasks()) == [GenerationTask, ToolTask, CompactionTask, custom]
    assert registry.snapshot().task("app.custom") is custom
    registry.uninstall(define_extension("custom"))
    assert registry.snapshot().task("app.custom") is None


def test_keeps_the_default_of_a_progress_interval_given_as_absent() -> None:
    assert resolve_settings({"progress": {"outputIntervalMs": 250}})["progress"] == {
        "partialIntervalMs": 100,
        "outputIntervalMs": 250,
    }
    assert resolve_settings({"progress": {"partialIntervalMs": None, "outputIntervalMs": 250}})["progress"] == {  # type: ignore[typeddict-item]
        "partialIntervalMs": 100,
        "outputIntervalMs": 250,
    }


read = tool("read")
bash = tool("bash")
edit = tool("edit")
Coding = define_extension(
    "coding",
    tools=[read, bash, edit],
    sections=[section("preamble", lambda *_: "You code.", tag=False), section("cwd", lambda *_: "/repo")],
)
Skills = define_extension("skills", sections=[section("skills", lambda *_: "S")])
Reviewer = define_extension("reviewer", sections=[section("role", lambda *_: "Review.")])


def _snapshot() -> Any:
    registry = create_registry()
    for extension in (Coding, Skills, Reviewer):
        registry.install(extension)
    return registry.snapshot()


def test_selects_the_default_a_list_or_the_default_edited_by_add_and_remove() -> None:
    snapshot = _snapshot()
    assert names(resolve(None, snapshot)["extensions"]) == ["coding", "skills", "reviewer"]
    settings = {"extensions": [Coding, Skills]}
    assert names(resolve({}, snapshot, settings)["extensions"]) == ["coding", "skills"]
    assert names(resolve({"extensions": ["reviewer", "coding"]}, snapshot, settings)["extensions"]) == [
        "reviewer",
        "coding",
    ]
    # Add appends, remove drops, duplicates keep their first position, uninstalled names are skipped.
    edited = {"extensions": {"add": ["reviewer", "coding", "gone"], "remove": ["skills"]}}
    assert names(resolve(edited, snapshot, settings)["extensions"]) == ["coding", "reviewer"]
    # An old object stands for its name: the installed extension is selected.
    stale = {"extensions": [define_extension("skills"), Skills]}
    assert list(resolve({}, snapshot, stale)["extensions"]) == [Skills]


def test_skips_uninstalled_names_and_resolves_them_again_once_they_are_installed() -> None:
    local = create_registry()
    local.install(Coding)
    state: Any = {"extensions": ["coding", "skills"]}
    assert names(resolve(state, local.snapshot())["extensions"]) == ["coding"]
    local.install(Skills)
    assert names(resolve(state, local.snapshot())["extensions"]) == ["coding", "skills"]


def test_replaces_same_name_tools_in_place_wraps_the_winner_then_applies_the_filter() -> None:
    local = create_registry()
    venv_bash = tool("bash", description="venv bash")
    calls: list[str] = []

    def unused(_inner: Any) -> Any:
        calls.append("grep")
        return tool("grep")

    Venv = define_extension("venv", tools=[venv_bash])
    Timing = define_extension(
        "timing",
        wraps=[
            wrap_tool(bash, lambda inner: dataclasses.replace(inner, description=f"{inner.description} (timed)")),
            wrap_tool(bash, lambda inner: dataclasses.replace(inner, description=f"{inner.description} [2]")),
            # No ``grep`` is selected: the wrapper does nothing and reports nothing.
            wrap_tool(tool("grep"), unused),
        ],
    )
    for extension in (Coding, Venv, Timing):
        local.install(extension)
    reports: list[Any] = []
    agent = resolve(None, local.snapshot(), {}, reports)
    assert [(each.name, each.description) for each in agent["tools"]] == [
        ("read", "read tool"),
        ("bash", "venv bash (timed) [2]"),
        ("edit", "edit tool"),
    ]
    assert reports == []
    assert calls == []

    # A list keeps exactly these names in its order, a repeated name at its first position.
    filtered = resolve({"tools": ["edit", "missing", "read", "edit"]}, local.snapshot())
    assert names(filtered["tools"]) == ["edit", "read"]
    assert names(resolve({"tools": {"remove": ["bash"]}}, local.snapshot())["tools"]) == ["read", "edit"]


def test_drops_a_tool_or_section_whose_wrapper_raises_or_renames_it_and_reports_the_failure() -> None:
    local = create_registry()

    def fail(_inner: Any) -> Any:
        raise RuntimeError("wrapper failed")

    def fail_section(_inner: Any) -> Any:
        raise RuntimeError("section wrapper failed")

    Broken = define_extension(
        "broken",
        wraps=[
            wrap_tool(read, fail),
            wrap_tool(edit, lambda inner: dataclasses.replace(inner, name="renamed")),
            wrap_section("cwd", fail_section),
        ],
    )
    local.install(Coding)
    local.install(Broken)
    reports: list[Any] = []
    agent = resolve(None, local.snapshot(), {}, reports)
    assert names(agent["tools"]) == ["bash"]
    assert [each.key for each in agent["sections"]] == ["preamble"]
    assert [str(error) for error in reports] == [
        "wrapper failed",
        "Wrapper renamed edit to renamed",
        "section wrapper failed",
    ]


async def test_orders_sections_by_extension_replaces_same_keys_in_place_and_renders_instructions_last() -> None:
    local = create_registry()

    def wrap_cwd(inner: PromptSection) -> PromptSection:
        async def render(prompt_input: Any, ctx: Any) -> str:
            return f"{inner.render(prompt_input, ctx)}!"

        return dataclasses.replace(inner, render=render)

    def never(_inner: Any) -> Any:
        raise RuntimeError("never")

    Override = define_extension(
        "override",
        sections=[section("preamble", lambda *_: "You review.", tag=False)],
        # Instructions are not wrapped.
        wraps=[wrap_section("cwd", wrap_cwd), wrap_section("instructions", never)],
    )
    for extension in (Coding, Skills, Override):
        local.install(extension)
    reports: list[Any] = []
    agent = resolve({"instructions": "Be terse."}, local.snapshot(), {}, reports)
    assert await rendered(agent) == [
        ("preamble", "You review."),
        ("cwd", "/repo!"),
        ("skills", "S"),
        ("instructions", "Be terse."),
    ]
    assert agent["sections"][-1].tag is None
    assert reports == []


def test_collects_hooks_of_the_selected_extensions_in_extension_order_and_applies_field_defaults() -> None:
    local = create_registry()
    first = {"before_tool": lambda *_: None}
    second = {"before_tool": lambda *_: None}
    on_yield = {"on_yield": lambda *_: None}
    local.install(define_extension("a", hooks=[hook(ToolTask, first), hook(GenerationTask, on_yield)]))  # type: ignore[arg-type]
    local.install(define_extension("b", hooks=[hook(ToolTask, second)]))  # type: ignore[arg-type]
    a = ("a", first)
    b = ("b", second)
    assert agent_hooks(resolve(None, local.snapshot()), "pi.tool") == [a, b]
    assert agent_hooks(resolve({"extensions": ["b", "a"]}, local.snapshot()), "pi.tool") == [b, a]
    assert agent_hooks(resolve({"extensions": ["b"]}, local.snapshot()), "pi.generation") == []

    defaults = resolve(None, local.snapshot())
    assert "model" not in defaults
    assert defaults["thinkingLevel"] == "off"
    assert "cwd" not in defaults
    configured = resolve(
        {"model": {"provider": "p", "modelId": "m"}, "thinkingLevel": "high", "cwd": "/w"}, local.snapshot()
    )
    assert configured["model"] == {"provider": "p", "modelId": "m"}
    assert configured["thinkingLevel"] == "high"
    assert configured["cwd"] == "/w"
