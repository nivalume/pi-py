"""Port of ``test/harness-reports.test.ts``."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Sequence
from typing import Any, NamedTuple

from durable_harness_support import chat_setup, completed, eventually
from durable_session_support import STORAGE_BACKENDS, ControlledStorage, context, deferred, resolve

from pi_ai import Type, create_models, faux_assistant_message
from pi_chord.context import Context
from pi_durable import (
    GenerationTask,
    Harness,
    MemoryStorage,
    create_registry,
    define_extension,
    define_task,
    define_tool,
    hook,
    section,
    wrap_tool,
)


class Report(NamedTuple):
    message: str
    info: Any


class Opened(NamedTuple):
    harness: Harness
    registry: Any
    reports: list[Report]


async def _nothing(*_args: Any) -> None:
    return None


async def open(tasks: Sequence[Any], *, storage: Any = None, on_wake: Any = None, set_timer: Any = None) -> Opened:
    """Open a Harness with ``tasks`` installed that collects every report's message and info."""
    registry = create_registry()
    registry.install(define_extension("test", tasks=list(tasks)))
    reports: list[Report] = []
    options: Any = {
        "models": create_models(),
        "registry": registry,
        "onReport": lambda error, info: reports.append(Report(str(error), info)),
    }
    if on_wake is not None:
        options["onWake"] = on_wake
    if set_timer is not None:
        options["setTimer"] = set_timer
    harness = await Harness.open(storage if storage is not None else MemoryStorage(), options, context)
    return Opened(harness, registry, reports)


def info_of(reports: list[Report], message: str) -> Any:
    return next((report.info for report in reports if report.message == message), None)


async def test_say_which_task_reported_a_failure_or_faulted() -> None:
    async def run(_task: Any, runtime: Any, _ctx: Context) -> None:
        runtime.report(RuntimeError("noted"))
        raise RuntimeError("broken")

    Faulty = define_task(
        name="test.faulty", version=1, initial=lambda _input: {"phase": "run"}, phases={"run": run}, abort=_nothing
    )
    opened = await open([Faulty])
    harness, reports = opened.harness, opened.reports
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Faulty, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    assert (await harness.wait_for_task(id, context))["state"]["outcome"]["status"] == "faulted"
    await eventually(lambda: len(reports) == 2)
    # The task's own failures name the extension that installed it.
    task = {"task": {"id": id, "kind": "test.faulty", "conversationId": root.id}, "extension": "test"}
    assert reports == [
        Report("noted", {"type": "task_error", **task}),
        Report("broken", {"type": "task_faulted", **task, "reason": "threw"}),
    ]
    await harness.close(context)


async def test_name_the_extension_whose_definition_ran_also_after_another_one_replaced_it() -> None:
    gate = deferred()
    started = deferred()

    def definition(name: str) -> Any:
        async def run(*_args: Any) -> None:
            resolve(started)
            await gate
            raise RuntimeError(f"broken in {name}")

        return define_task(
            name="test.replaced",
            version=1,
            initial=lambda _input: {"phase": "run"},
            phases={"run": run},
            abort=_nothing,
        )

    Old = definition("old")
    opened = await open([])
    harness, registry, reports = opened.harness, opened.registry, opened.reports
    old = define_extension("old-extension", tasks=[Old])
    registry.install(old)
    root = await harness.root(context)
    id = await root.commit(lambda tx: tx.create_task(Old, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await started
    registry.uninstall(old)
    registry.install(define_extension("new-extension", tasks=[definition("new")]))
    resolve(gate)
    await harness.wait_for_task(id, context)

    def faulted(report: Report) -> bool:
        return report.info["type"] == "task_faulted"

    await eventually(lambda: any(faulted(report) for report in reports))
    report = next(report for report in reports if faulted(report))
    assert report.message == "broken in old"
    assert report.info["extension"] == "old-extension"
    await harness.close(context)


async def test_name_the_extension_of_a_failing_section_or_hook_and_tell_them_from_a_failing_environment_builder() -> (
    None
):
    setup = chat_setup()
    setup.faux.set_responses([faux_assistant_message("answer")])

    def broken_section(*_args: Any) -> Any:
        raise RuntimeError("section")

    def broken_hook(*_args: Any) -> Any:
        raise RuntimeError("hook")

    def broken_env(*_args: Any) -> Any:
        raise RuntimeError("env")

    setup.registry.install(define_extension("sections", sections=[section("broken", broken_section)]))
    setup.registry.install(define_extension("hooks", hooks=[hook(GenerationTask, {"after_response": broken_hook})]))
    reports: list[Report] = []
    harness = await Harness.open(
        MemoryStorage(),
        {
            "models": setup.models,
            "registry": setup.registry,
            "env": broken_env,
            "onReport": lambda error, info: reports.append(Report(str(error), info)),
        },
        context,
    )
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    await (await root.submit({"type": "input", "content": "hi"}, context)).wait(context)
    generation = {"kind": "pi.generation", "conversationId": root.id}

    def matches(info: Any, expected: dict[str, Any]) -> bool:
        if info is None:
            return False
        for key, value in expected.items():
            if key == "task":
                if any(info["task"].get(name) != part for name, part in value.items()):
                    return False
            elif info.get(key) != value:
                return False
        return True

    assert matches(info_of(reports, "env"), {"type": "host_error", "callback": "env", "task": generation})
    assert matches(
        info_of(reports, "section"), {"type": "extension_error", "extension": "sections", "task": generation}
    )
    assert matches(info_of(reports, "hook"), {"type": "extension_error", "extension": "hooks", "task": generation})
    await harness.close(context)


async def test_tell_extension_listener_wake_and_timer_failures_from_a_failed_session() -> None:
    async def run(_task: Any, runtime: Any, ctx: Context) -> None:
        # The host's set_timer raises: reported, and armed with the loop's timer instead.
        runtime.set_timer(lambda: None, 10)
        await runtime.sleep_until(runtime.now() + 60_000, {"phase": "done"}, ctx)

    async def done(_task: Any, runtime: Any, ctx: Context) -> None:
        await runtime.commit(lambda _tx, _current: completed(None), ctx)

    Sleeper = define_task(
        name="test.sleeper",
        version=1,
        initial=lambda _input: {"phase": "run"},
        phases={"run": run, "done": done},
        abort=_nothing,
    )
    storage = ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])

    def alarm(*_args: Any) -> None:
        raise RuntimeError("alarm")

    def no_timers(*_args: Any) -> Any:
        raise RuntimeError("no timers")

    opened = await open([Sleeper], storage=storage, on_wake=alarm, set_timer=no_timers)
    harness, registry, reports = opened.harness, opened.registry, opened.reports

    async def echo_execute(*_args: Any) -> Any:
        return {}

    echo = define_tool(name="echo", description="echo", parameters=Type.Object({}), execute=echo_execute)
    registry.install(define_extension("tools", tools=[echo]))
    registry.install(
        define_extension("wraps", wraps=[wrap_tool(echo, lambda tool: dataclasses.replace(tool, name="renamed"))])
    )

    def listener(*_args: Any) -> None:
        raise RuntimeError("listener")

    harness.subscribe_commits(listener)
    root = await harness.root(context)
    await root.agent(context)
    await root.commit(lambda tx: tx.create_task(Sleeper, None, {"ownership": {"kind": "conversation"}}), context)
    harness.resume()
    await eventually(
        lambda: all(any(report.message == message for report in reports) for message in ["alarm", "no timers"])
    )
    assert info_of(reports, "Wrapper renamed echo to renamed") == {
        "type": "extension_error",
        "conversationId": root.id,
        "extension": "wraps",
    }
    assert info_of(reports, "listener") == {"type": "listener_error"}
    assert info_of(reports, "alarm") == {"type": "host_error", "callback": "onWake"}
    assert info_of(reports, "no timers") == {"type": "host_error", "callback": "setTimer"}

    failure = RuntimeError("disk gone")
    storage.fail_next_commit(failure)
    with contextlib.suppress(BaseException):
        await root.commit(lambda tx: tx.create_conversation({"ownership": {"kind": "ownerless"}}), context)
    await harness.closed
    assert info_of(reports, "disk gone") == {"type": "session_failed"}
