"""Port of ``test/codemode-models.test.ts`` for JavaScript scripts: ``models.*`` on QuickJS (``CodemodeSandbox``).

Upstream's scripts and globals (``models.getModelOfType``, ``models.generateImages``, ``searchTools``,
``describeTool``) and upstream's error texts (``context.state must be an object``). ``test_codemode_models.py``
covers the Python names on Monty.
"""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import pytest
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
from pi_codemode import CodemodeSandbox
from pi_durable import MemoryStorage, ToolResultEntry, UsageDoc
from pi_durable.codemode import create_codemode_extension

USAGE = Usage(10, 2, 0, 0, 12, UsageCost(0.001, 0.0002, 0, 0, 0.0012))
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
HEADER = re.compile(r"^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n")
CONTEXT = '{ state: {}, questions: { good: { type: "bool", instructions: "?", criteria: { true: "y", false: "n" } } } }'


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


_sandboxes: list[CodemodeSandbox] = []


@pytest.fixture(autouse=True)
async def _close_sandboxes() -> Any:
    yield
    sandboxes = list(_sandboxes)
    _sandboxes.clear()
    await asyncio.gather(*(sandbox.close() for sandbox in sandboxes), return_exceptions=True)


def models_chat() -> ModelsChat:
    """A chat whose models also have a classifier and an image model of a test provider, and the code mode extension
    on a QuickJS sandbox."""
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
    sandbox = CodemodeSandbox()
    _sandboxes.append(sandbox)
    setup.registry.install(create_codemode_extension(sandbox=sandbox))
    return ModelsChat(setup, calls, gate)


@dataclass
class Run:
    text: str
    message: Any
    usage: Any


async def run(chat: ModelsChat, code: str) -> Run:
    chat.setup.faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("codemode", {"code": textwrap.dedent(code).strip("\n")}, id="s1")],
                stop_reason="toolUse",
            ),
            faux_assistant_message([faux_text("done")]),
        ]
    )
    opened = await open_chat(MemoryStorage(), chat.setup)
    await (await opened.root.submit({"type": "input", "content": "go"}, context)).wait(context)
    entry: Any = next(candidate for candidate in await all_entries(opened.root) if ToolResultEntry.is_(candidate))
    message = entry["model"][0]
    text = "".join(item.text if isinstance(item, TextContent) else f"<{item.type}>" for item in message.content)
    usage = await opened.harness.snapshot(UsageDoc, opened.root.id, context)
    await opened.harness.close(context)
    return Run(HEADER.sub("", text, count=1), message, usage)


async def test_reads_the_model_catalog_without_credentials_in_headers() -> None:
    result = await run(
        models_chat(),
        """
        const classifiers = await models.getModelsOfType("classifier", "test");
        const one = await models.getModelOfType("classifier", "test", "judge");
        const available = await models.getAvailableOfType("image");
        const unknown = await models.getModelOfType("classifier", "test", "nope");
        return [classifiers.map((m) => m.id), one.id, "headers" in one, available.map((m) => m.id), unknown === undefined];
        """,
    )
    assert json.loads(result.text) == [["judge"], "judge", False, ["painter"], True]


async def test_classifies_through_an_internal_tool_hidden_from_tools_and_listings() -> None:
    chat = models_chat()
    result = await run(
        chat,
        """
        const model = await models.getModelOfType("classifier", "test", "judge");
        const result = await models.classify(model, {
          state: { text: "hi" },
          questions: { good: { type: "bool", instructions: "Is it good?", criteria: { true: "yes", false: "no" } } },
        });
        const found = (await searchTools("classify generate images models")).length;
        const described = (await describeTool("pi.codemode.models.classify")) !== undefined;
        const hidden = Object.keys(tools).some((name) => name.includes("models"));
        return [result.stopReason, result.answers.good.value, hidden, ALL_TOOLS.some((tool) => tool.name.includes("models")), found, described];
        """,
    )
    assert json.loads(result.text) == ["stop", True, False, False, 0, False]
    assert chat.calls.classify == 1
    assert [{"name": call["name"], "status": call["status"]} for call in result.message.details["calls"]] == [
        {"name": "models.classify", "status": "ok"}
    ]
    # Counted once, under the internal tool, by the Harness.
    assert result.usage["tools"]["pi.codemode.models.classify"]["cost"]["total"] == pytest.approx(0.0012)


async def test_resolves_a_model_call_that_failed_with_its_result() -> None:
    chat = models_chat()
    result = await run(
        chat,
        f"""
        const model = await models.getModelOfType("classifier", "test", "judge");
        const context = {CONTEXT};
        context.state.fail = true;
        const result = await models.classify(model, context);
        return [result.stopReason, result.errorMessage];
        """,
    )
    assert json.loads(result.text) == ["error", "quota exceeded"]
    assert [{"name": call["name"], "status": call["status"]} for call in result.message.details["calls"]] == [
        {"name": "models.classify", "status": "error"}
    ]


async def test_rejects_bad_arguments_with_upstreams_messages_before_any_request() -> None:
    chat = models_chat()
    result = await run(
        chat,
        """
        const model = await models.getModelOfType("classifier", "test", "judge");
        const painter = await models.getModelOfType("image", "test", "painter");
        const errors = [];
        for (const call of [
          () => models.classify(model, { prompt: "x" }),
          () => models.classify(undefined, {}),
          () => models.classify(painter, {}),
          () => models.generateImages(painter, { prompt: "a cat" }),
          () => models.getModelOfType("classifier", "test/judge"),
          () => models.getModelsOfType("video"),
        ]) {
          try { await call(); errors.push("no error"); } catch (error) { errors.push(error.message); }
        }
        return errors;
        """,
    )
    errors = json.loads(result.text)
    assert errors[0].startswith("models.classify() context.state must be an object, got ")
    assert errors[0].endswith(
        'Expected context: { state: { ... }, images?: [{ type: "image", data: <base64>, mimeType }], questions: { '
        '<id>: { type: "choice", instructions, criteria: { <label>: <meaning> } } | { type: "score", instructions, '
        'criteria: [<lowest level>, ..., <highest level>] } | { type: "bool", instructions, criteria: { true: '
        "<meaning>, false: <meaning> } } } }."
    )
    assert errors[1] == (
        "models.classify() expects a classifier model as its first argument, got null. "
        "models.getModelOfType() returns undefined for an unknown provider or id. "
        'List the classifier models you can use with models.getAvailableOfType("classifier").'
    )
    assert errors[2] == (
        '"test/painter" is an image model, not a classifier model. '
        'List the classifier models you can use with models.getAvailableOfType("classifier").'
    )
    assert errors[3].startswith("models.generateImages() context.input must be a non-empty array of blocks, got ")
    assert errors[3].endswith(
        'Expected context: { input: [{ type: "text", text: <prompt> }, ...optional { type: "image", data: <base64>, '
        "mimeType } references] }."
    )
    assert errors[4] == (
        "models.getModelOfType(type, provider, id) expects three strings, got (a string, a string). The provider and "
        'the id are separate arguments, for example models.getModelOfType("classifier", "typesafe", "jev-latest").'
    )
    assert errors[5] == 'Unknown model type "video". Use "chat", "image", or "classifier".'
    assert chat.calls.classify == 0
    assert chat.calls.images == 0


async def test_describes_a_missing_context_field_as_undefined_as_upstream_does() -> None:
    result = await run(
        models_chat(),
        """
        const model = await models.getModelOfType("classifier", "test", "judge");
        const painter = await models.getModelOfType("image", "test", "painter");
        const errors = [];
        try { await models.classify(model, { prompt: "x" }); } catch (error) { errors.push(error.message); }
        try { await models.generateImages(painter, { prompt: "x" }); } catch (error) { errors.push(error.message); }
        return errors.map((message) => message.split(". Expected")[0]);
        """,
    )
    assert json.loads(result.text) == [
        "models.classify() context.state must be an object, got undefined",
        "models.generateImages() context.input must be a non-empty array of blocks, got undefined",
    ]


async def test_runs_at_most_four_model_calls_at_once() -> None:
    chat = models_chat()
    chat.gate.open = deferred()
    chat.gate.at_four = True
    result = await run(
        chat,
        f"""
        const model = await models.getModelOfType("classifier", "test", "judge");
        const context = {CONTEXT};
        return (await Promise.all(Array.from({{ length: 8 }}, () => models.classify(model, context)))).length;
        """,
    )
    assert result.text == "8"
    assert chat.calls.classify == 8
    assert chat.calls.max_in_flight == 4


async def test_notes_generated_images_the_script_did_not_show() -> None:
    hidden = await run(
        models_chat(),
        """
        const model = await models.getModelOfType("image", "test", "painter");
        const result = await models.generateImages(model, { input: [{ type: "text", text: "a cat" }] });
        return result.output.length;
        """,
    )
    assert (
        "Note: models.generateImages() returned 1 image that the script did not show. "
        "Show each image block of result.output with image(block)."
    ) in hidden.text
    shown = await run(
        models_chat(),
        """
        const model = await models.getModelOfType("image", "test", "painter");
        const result = await models.generateImages(model, { input: [{ type: "text", text: "a cat" }] });
        image(result.output[0]);
        """,
    )
    assert "Note:" not in shown.text
    assert any(isinstance(item, ImageContent) for item in shown.message.content)
