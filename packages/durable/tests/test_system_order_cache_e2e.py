"""Port of ``test/system-order-cache-e2e.test.ts`` (regression coverage for #10542).

Upstream runs it live against Anthropic with credentials from ``~/.pi/agent/auth.json``. The live run is skipped
here; the offline port drives the same two turns through pi-ai's real Anthropic Messages client over a mocked HTTP
endpoint (``httpx.MockTransport``) that simulates prompt caching as the service does: a request reads from cache
the longest prefix it shares with the previous one, in the order the service builds the prompt (tools, system,
messages; ``cache_control`` markers aside). It checks what the live test checks of Durable's request order: with
the system message leading, a tool added on the second turn arrives in the conversation and the cached prefix
survives; with the input ahead of it (the order before the fix, restored by a ``before_request`` hook), the tool
list at the top of the request changes and the cache is lost from the first tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from durable_session_support import context

from pi_ai import Type, normalize_context, uuidv7
from pi_ai.api import anthropic_messages
from pi_ai.providers.all import builtin_models
from pi_ai.types import AssistantMessage, Message
from pi_durable import (
    AssistantEntry,
    GenerationTask,
    Harness,
    MemoryStorage,
    create_registry,
    define_extension,
    define_tool,
    hook,
)

#: Sets both mid-conversation compat flags, so tool changes use Anthropic's native additions when allowed.
MODEL = {"provider": "anthropic", "modelId": "claude-sonnet-5-5"}


@pytest.fixture(autouse=True)
def _clean_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "PI_CACHE_RETENTION"):
        monkeypatch.delenv(name, raising=False)


def instructions(nonce: str) -> str:
    lines = [
        "This is an automated prompt-cache test. The repeated context below is intentional.",
        "Never call tools. Reply with exactly the text the user asks for.",
    ]
    for index in range(250):
        lines.append(
            f"{nonce} cache record {str(index).rjust(3, '0')}: alpha beta gamma delta epsilon zeta eta theta iota"
            " kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
        )
    return "\n".join(lines)


Parameters = Type.Object({"text": Type.String()})


async def _no_output(*_args: Any) -> Any:
    return {"output": []}


first = define_tool(name="probe_first", description="Unused probe tool.", parameters=Parameters, execute=_no_output)
second = define_tool(
    name="probe_second",
    description="Unused probe tool added on the second turn.",
    parameters=Parameters,
    execute=_no_output,
)


def committed_order(messages: Sequence[Message]) -> list[Message]:
    """Puts the leading system message back after the user messages before it: the request order before the fix."""
    if not messages or messages[0].role != "system":
        return list(messages)
    system, *rest = messages
    at = next((index for index, message in enumerate(rest) if message.role != "user"), len(rest))
    return [*rest[:at], system, *rest[at:]]


def tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _without_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_cache_control(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_control(item) for item in value]
    return value


def segments(body: dict[str, Any]) -> list[str]:
    """What the service caches, in prompt order: each tool, the system blocks, then each message."""
    body = _without_cache_control(body)
    return [
        *(json.dumps(tool, sort_keys=True) for tool in body.get("tools", [])),
        json.dumps(body.get("system"), sort_keys=True),
        *(json.dumps(message, sort_keys=True) for message in body.get("messages", [])),
    ]


class CachingAnthropic:
    """A mocked Messages endpoint that answers every request with the text asked for and simulates its cache."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._previous: list[str] = []

    def _usage(self, body: dict[str, Any]) -> tuple[int, int]:
        current = segments(body)
        total = sum(tokens(segment) for segment in current)
        shared = 0
        for before, now in zip(self._previous, current, strict=False):
            if before != now:
                break
            shared += tokens(now)
        self._previous = current
        return total, shared

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        body = json.loads(request.content)
        self.bodies.append(body)
        total, cached = self._usage(body)
        asked = body["messages"][-1]["content"]
        asked_text = (
            asked
            if isinstance(asked, str)
            else next((block["text"] for block in reversed(asked) if block.get("type") == "text"), "")
        )
        text = asked_text.removeprefix("Reply exactly: ")
        usage = {"input_tokens": total - cached, "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        events = [
            ("message_start", {"type": "message_start", "message": {"id": f"msg_{len(self.bodies)}", "usage": usage}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        content = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


class Forwarding:
    """``models`` whose ``stream_simple`` calls ``stream`` (upstream's Proxy over ``builtinModels()``)."""

    def __init__(self, models: Any, stream: Callable[[Any, Any, Any], Any]) -> None:
        self._models = models
        self._stream = stream

    def stream_simple(self, model: Any, request: Any, options: Any = None) -> Any:
        return self._stream(model, request, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


async def cache_read_after_tool_change(legacy_order: bool) -> float:
    """Share of the prompt the second turn read from cache after adding a tool."""
    server = CachingAnthropic()

    def stream(model: Any, request: Any, options: Any) -> Any:
        return anthropic_messages.stream_simple(
            model,
            normalize_context(request),
            {**(options or {}), "api_key": "fake-key", "http_client": server.client()},
        )

    models = Forwarding(builtin_models(), stream)
    registry = create_registry()
    probe = define_extension("probe", tools=[first, second])

    def before_request(request: Any, *_rest: Any) -> Any:
        return {"messages": committed_order(request["messages"])}

    legacy = define_extension("legacy-order", hooks=[hook(GenerationTask, {"before_request": before_request})])
    registry.install(probe)
    registry.install(legacy)
    harness = await Harness.open(
        MemoryStorage(),
        {"models": models, "registry": registry, "settings": {"compaction": {"enabled": False}}},  # type: ignore[typeddict-item]
        context,
    )
    try:
        root = await harness.root(
            context,
            agent={
                "model": MODEL,  # type: ignore[typeddict-item]
                "thinkingLevel": "low",
                "instructions": instructions(uuidv7()),
                "extensions": [probe, legacy] if legacy_order else [probe],
                "tools": [first],
            },
        )
        harness.resume()

        async def ask(text: str) -> AssistantMessage:
            submission = await root.submit({"type": "input", "content": f"Reply exactly: {text}"}, context)
            settled = await submission.wait(context)
            assert settled["status"] == "done" and settled["type"] == "input", f"Unexpected {settled['status']}"
            record = await harness.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
            message = None if record is None else record["model"][0]
            assert isinstance(message, AssistantMessage)
            assert message.stop_reason == "stop", message.error_message
            return message

        before = await ask("READY")
        assert before.usage.input + before.usage.cache_read + before.usage.cache_write >= 4_000
        await root.configure({"tools": [first, second]}, context)
        after = (await ask("READY AGAIN")).usage
        return after.cache_read / (after.input + after.cache_read + after.cache_write)
    finally:
        await harness.close(context)


async def test_keeps_the_prompt_cache_across_a_tool_change_only_when_the_system_message_leads_offline() -> None:
    # Before the fix: the input precedes the system message, so the tool list is sent at the top of the request,
    # and adding a tool changes the cached prefix from its first token.
    assert await cache_read_after_tool_change(True) < 0.2
    # The fix: the system message leads, the initial tools stay fixed, and the new tool arrives in the conversation.
    assert await cache_read_after_tool_change(False) > 0.8


@pytest.mark.skip(
    reason="live network: needs Anthropic credentials (upstream resolveApiKey from ~/.pi/agent/auth.json); "
    "covered offline by test_keeps_the_prompt_cache_across_a_tool_change_only_when_the_system_message_leads_offline"
)
async def test_keeps_the_prompt_cache_across_a_tool_change_only_when_the_system_message_leads() -> None:
    pass
