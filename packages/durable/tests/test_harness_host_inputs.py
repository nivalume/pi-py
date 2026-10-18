"""Port of ``test/harness-host-inputs.test.ts``."""

from __future__ import annotations

from typing import Any, NamedTuple

from durable_harness_support import all_entries, text_of
from durable_session_support import context

from pi_ai import Type, create_models, faux_assistant_message, faux_provider, faux_text, faux_tool_call
from pi_ai.types import AssistantMessage
from pi_durable import Harness, MemoryStorage, create_registry, define_extension, define_tool


def answering(text: str) -> Any:
    """A ``Models`` whose only provider is a faux one answering every request with ``text``."""
    faux = faux_provider()
    faux.set_responses([faux_assistant_message([faux_text(text)]) for _ in range(10)])
    models = create_models()
    models.set_provider(faux.provider)
    return models


class Opened(NamedTuple):
    harness: Harness
    root: Any


async def open_with(options: dict[str, Any], registry: Any = None) -> Opened:
    registry = registry if registry is not None else create_registry()
    harness = await Harness.open(MemoryStorage(), {**options, "registry": registry}, context)  # type: ignore[typeddict-item]
    root = await harness.root(context, agent={"model": {"provider": "faux", "modelId": "faux-1"}})
    return Opened(harness, root)


async def turn(root: Any, content: str) -> str | None:
    settled = await (await root.submit({"type": "input", "content": content}, context)).wait(context)
    assert settled["status"] == "done"
    entries = await all_entries(root)
    model = entries[-1].get("model") if entries else None
    return text_of(model[0] if model else None)


async def test_generation_uses_the_current_return_value_of_a_models_function_at_each_request() -> None:
    current = [answering("first")]
    harness, root = await open_with({"models": lambda: current[0]})
    assert await turn(root, "one") == "first"
    current[0] = answering("second")
    assert await turn(root, "two") == "second"
    await harness.close(context)


async def test_still_accepts_a_plain_models_object() -> None:
    harness, root = await open_with({"models": answering("plain")})
    assert await turn(root, "one") == "plain"
    await harness.close(context)


async def test_tools_read_the_current_models_at_each_access_of_api_models() -> None:
    first = create_models()
    second = create_models()
    faux = faux_provider()
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("probe", {}, id="call-1")], stop_reason="toolUse"),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    first.set_provider(faux.provider)
    second.set_provider(faux.provider)
    current = [first]
    seen: list[Any] = []

    async def execute(_args: Any, api: Any, _context: Any) -> Any:
        seen.append(api.models)
        current[0] = second
        seen.append(api.models)
        return {"output": []}

    registry = create_registry()
    registry.install(
        define_extension(
            "probe",
            tools=[
                define_tool(
                    name="probe",
                    description="Records api.models before and after the host swaps it",
                    parameters=Type.Object({}),
                    execute=execute,
                )
            ],
        )
    )
    harness, root = await open_with({"models": lambda: current[0]}, registry)
    assert await turn(root, "go") == "done"
    assert seen == [first, second]
    assert seen[0] is first
    assert seen[1] is second
    await harness.close(context)


async def test_generation_reads_the_current_return_value_of_a_settings_function() -> None:
    timeouts: list[Any] = []
    faux = faux_provider()

    def reply(_request: Any, options: Any, *_rest: Any) -> AssistantMessage:
        timeouts.append(None if options is None else options.get("timeout_ms"))
        return faux_assistant_message([faux_text("ok")])

    faux.set_responses([reply, reply])
    models = create_models()
    models.set_provider(faux.provider)
    settings: list[Any] = [{"stream": {"timeoutMs": 1000}}]
    harness, root = await open_with({"models": models, "settings": lambda: settings[0]})
    await turn(root, "one")
    settings[0] = {"stream": {"timeoutMs": 2000}}
    await turn(root, "two")
    assert timeouts == [1000, 2000]
    await harness.close(context)
