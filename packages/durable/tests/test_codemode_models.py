"""Port of ``test/codemode-models.test.ts``: ``models.*`` in code mode scripts.

Adaptations: scripts are Python with the snake_case globals (``models.get_model_of_type``, ``models.generate_images``,
``search_tools``, ``describe_tool``); results are pi-ai's JSON, read as dicts (``result["stopReason"]``). Upstream's
``Object.keys(tools).some(...)`` becomes membership checks of the internal tool's name and identifier, since the
``tools`` namespace of a Monty script is not iterable. Error texts name Python shapes (``context["state"] must be a
dict``).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("pydantic_monty")

from durable_harness_support import ChatSetup, all_entries, chat_setup, open_chat
from durable_session_support import context, deferred, resolve

from pi_ai import (
    AssistantImages,
    AuthResult,
    ClassifierModel,
    ClassifierResult,
    ImageContent,
    ImageModel,
    ModelAuth,
    ModelCost,
    ProviderAuth,
    TextContent,
    Usage,
    UsageCost,
    create_provider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from pi_ai.auth.types import ApiKeyAuth
from pi_codemode import MontyCodemodeSandbox
from pi_durable import MemoryStorage, ToolResultEntry, UsageDoc
from pi_durable.codemode import create_codemode_extension
from pi_durable.env.local import LocalExecutionEnv

USAGE = Usage(10, 2, 0, 0, 12, UsageCost(0.001, 0.0002, 0, 0, 0.0012))
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
HEADER = re.compile(r"^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n")


@dataclass
class Calls:
    classify: int = 0
    images: int = 0
    in_flight: int = 0
    max_in_flight: int = 0


@dataclass
class Gate:
    """``at_four``: hold classifier calls until four run, then a little longer, so a fifth would show up."""

    open: asyncio.Future[Any] = field(default_factory=deferred)
    at_four: bool = False


@dataclass
class ModelsChat:
    setup: ChatSetup
    calls: Calls
    gate: Gate
    sandbox: MontyCodemodeSandbox


_sandboxes: list[MontyCodemodeSandbox] = []


@pytest.fixture(autouse=True)
async def _close_sandboxes() -> Any:
    yield
    sandboxes = list(_sandboxes)
    _sandboxes.clear()
    await asyncio.gather(*(sandbox.close() for sandbox in sandboxes), return_exceptions=True)


def models_chat() -> ModelsChat:
    """A chat whose models also have a classifier and an image model of a test provider, and the code mode extension."""
    setup = chat_setup()
    calls = Calls()
    gate = Gate()
    resolve(gate.open)

    class Classifier:
        async def classify(self, model: Any, classifier_context: Any, _options: Any = None) -> ClassifierResult:
            calls.classify += 1
            if classifier_context.state.get("fail") is True:
                return ClassifierResult(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    stop_reason="error",
                    error_message="quota exceeded",
                    timestamp=0,
                )
            calls.in_flight += 1
            calls.max_in_flight = max(calls.max_in_flight, calls.in_flight)
            if gate.at_four and calls.in_flight == 4:
                opening = gate.open
                asyncio.get_running_loop().call_later(0.05, lambda: resolve(opening))
            await gate.open
            calls.in_flight -= 1
            return ClassifierResult(
                api=model.api,
                provider=model.provider,
                model=model.id,
                answers={"good": {"value": True}},  # pyright: ignore[reportArgumentType]
                usage=USAGE,
                stop_reason="stop",
                timestamp=0,
            )

    class Images:
        async def generate_images(self, model: Any, _context: Any, _options: Any = None) -> AssistantImages:
            calls.images += 1
            return AssistantImages(
                api=model.api,
                provider=model.provider,
                model=model.id,
                output=[ImageContent(PNG, "image/png")],
                stop_reason="stop",
                timestamp=0,
            )

    async def resolve_auth(**_: Any) -> AuthResult:
        return AuthResult(ModelAuth())

    setup.models.set_provider(
        create_provider(
            id="test",
            auth=ProviderAuth(api_key=ApiKeyAuth("Test", resolve_auth)),
            models=[
                ClassifierModel(
                    id="judge",
                    name="Judge",
                    api="test-classifier",
                    provider="test",
                    base_url="https://example.invalid",
                    input=["text"],
                    cost=ModelCost(),
                    headers={"authorization": "secret"},
                    context_window=1000,
                ),
                ImageModel(
                    id="painter",
                    name="Painter",
                    api="test-images",
                    provider="test",
                    base_url="https://example.invalid",
                    input=["text"],
                    cost=ModelCost(),
                    output=["image"],
                ),
            ],
            classifiers={"test-classifier": Classifier()},
            images={"test-images": Images()},
        )
    )
    sandbox = MontyCodemodeSandbox()
    _sandboxes.append(sandbox)
    setup.registry.install(create_codemode_extension(sandbox=sandbox))
    return ModelsChat(setup, calls, gate, sandbox)


@dataclass
class Run:
    text: str
    message: Any
    usage: Any


async def run(chat: ModelsChat, code: str, env: Any = None) -> Run:
    chat.setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("codemode", {"code": textwrap.dedent(code).strip("\n")}, id="s1")],
                stop_reason="toolUse",
            ),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), chat.setup, env=env)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entry = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    text = "".join(item.text if isinstance(item, TextContent) else f"<{item.type}>" for item in message.content)
    usage = await opened.harness.snapshot(UsageDoc, opened.root.id, context)
    await opened.harness.close(context)
    return Run(HEADER.sub("", text, count=1), message, usage)


CONTEXT = '{"state": {}, "questions": {"good": {"type": "bool", "instructions": "?", "criteria": {"true": "y", "false": "n"}}}}'


async def test_reads_the_model_catalog_without_credentials_in_headers() -> None:
    chat = models_chat()
    result = await run(
        chat,
        """
        classifiers = await models.get_models_of_type("classifier", "test")
        one = await models.get_model_of_type("classifier", "test", "judge")
        available = await models.get_available_of_type("image")
        return [[m["id"] for m in classifiers], one["id"], "headers" in one, [m["id"] for m in available]]
        """,
    )
    assert json.loads(result.text) == [["judge"], "judge", False, ["painter"]]


async def test_classifies_through_an_internal_tool_a_nested_call_with_its_usage_hidden_from_tools_and_listings() -> (
    None
):
    chat = models_chat()
    result = await run(
        chat,
        """
        model = await models.get_model_of_type("classifier", "test", "judge")
        result = await models.classify(model, {
            "state": {"text": "hi"},
            "questions": {"good": {"type": "bool", "instructions": "Is it good?", "criteria": {"true": "yes", "false": "no"}}},
        })
        found = len(await search_tools("classify generate images models"))
        described = (await describe_tool("pi.codemode.models.classify")) is not None
        hidden = "pi.codemode.models.classify" in tools or "pi_codemode_models_classify" in tools
        return [result["stopReason"], result["answers"]["good"]["value"], hidden, any("models" in tool["name"] for tool in ALL_TOOLS), found, described]
        """,
    )
    assert json.loads(result.text) == ["stop", True, False, False, 0, False]
    assert chat.calls.classify == 1
    assert [{"name": call["name"], "status": call["status"]} for call in result.message.details["calls"]] == [
        {"name": "models.classify", "status": "ok"}
    ]
    # Counted once, under the internal tool, by the Harness.
    assert result.usage["tools"]["pi.codemode.models.classify"]["cost"]["total"] == pytest.approx(0.0012)


async def test_resolves_a_model_call_that_failed_with_its_result_as_the_coding_agent_does() -> None:
    chat = models_chat()
    result = await run(
        chat,
        """
        model = await models.get_model_of_type("classifier", "test", "judge")
        result = await models.classify(model, {
            "state": {"fail": True},
            "questions": {"good": {"type": "bool", "instructions": "?", "criteria": {"true": "y", "false": "n"}}},
        })
        return [result["stopReason"], result["errorMessage"]]
        """,
    )
    assert json.loads(result.text) == ["error", "quota exceeded"]
    # The row still shows the failure.
    assert [{"name": call["name"], "status": call["status"]} for call in result.message.details["calls"]] == [
        {"name": "models.classify", "status": "error"}
    ]


async def test_rejects_a_bad_classifier_context_with_the_expected_shape_before_any_request() -> None:
    chat = models_chat()
    result = await run(
        chat,
        """
        model = await models.get_model_of_type("classifier", "test", "judge")
        try:
            await models.classify(model, {"prompt": "x"})
        except Exception as error:
            return str(error)
        """,
    )
    assert 'context["state"] must be a dict' in result.text
    assert "Expected context:" in result.text
    assert chat.calls.classify == 0


async def test_runs_at_most_four_model_calls_at_once() -> None:
    chat = models_chat()
    chat.gate.open = deferred()
    chat.gate.at_four = True
    result = await run(
        chat,
        f"""
        import asyncio
        model = await models.get_model_of_type("classifier", "test", "judge")
        context = {CONTEXT}
        return len(await asyncio.gather(*[models.classify(model, context) for _ in range(8)]))
        """,
    )
    assert result.text == "8"
    assert chat.calls.classify == 8
    assert chat.calls.max_in_flight == 4


async def test_does_not_start_a_model_call_that_waited_for_a_slot_after_the_script_ended() -> None:
    chat = models_chat()
    chat.gate.open = deferred()
    directory = tempfile.mkdtemp(prefix="pi-codemode-models-")

    # The result spills to a file after the script ended and before the tool returns. Then the running calls finish
    # and free their slots, which the two waiting calls would take.
    class SlowTempFiles(LocalExecutionEnv):
        async def create_temp_dir(self, prefix: str | None, call_context: Any) -> Any:
            resolve(chat.gate.open)
            await asyncio.sleep(0.1)
            created = await super().create_temp_dir(prefix, call_context)
            if created["ok"] is True:
                spills.append(created["value"])
            return created

    spills: list[str] = []
    try:
        result = await run(
            chat,
            f"""
            model = await models.get_model_of_type("classifier", "test", "judge")
            context = {CONTEXT}
            for i in range(6):
                models.classify(model, context)
            # A journaled call after them: by its result, the first four have been launched.
            await models.get_models_of_type("image")
            return "x" * 60000
            """,
            SlowTempFiles(cwd=directory),
        )
        assert "[Full output: " in result.text
        assert chat.calls.classify == 4
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        for spill in spills:
            shutil.rmtree(spill, ignore_errors=True)


async def test_notes_generated_images_the_script_did_not_show() -> None:
    chat = models_chat()
    hidden = await run(
        chat,
        """
        model = await models.get_model_of_type("image", "test", "painter")
        result = await models.generate_images(model, {"input": [{"type": "text", "text": "a cat"}]})
        return len(result["output"])
        """,
    )
    assert "Note: models.generate_images() returned 1 image that the script did not show" in hidden.text
    shown = await run(
        models_chat(),
        """
        model = await models.get_model_of_type("image", "test", "painter")
        result = await models.generate_images(model, {"input": [{"type": "text", "text": "a cat"}]})
        image(result["output"][0])
        """,
    )
    assert "Note:" not in shown.text
    assert any(isinstance(item, ImageContent) for item in shown.message.content)
