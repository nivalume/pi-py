"""Port of ``test/provider-session-cache-e2e.test.ts`` (regression coverage for #10424).

Upstream runs it live against OpenAI Codex with credentials from ``~/.pi/agent/auth.json``. The live run is
skipped here; the offline port drives the same Durable turns through pi-ai's real Codex Responses client over a
mocked HTTP endpoint (``httpx.MockTransport``) that simulates prompt caching per ``prompt_cache_key``, as the
service does: a request reads from cache the longest prefix of input it shares with the previous request of the
same key. It checks what the live test checks of Durable: every request carries the conversation's provider
session ID, and later turns only append, so the cache is hit from the second turn on.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from durable_session_support import context

from pi_ai import normalize_context, uuidv7
from pi_ai.api import openai_codex_responses as codex
from pi_ai.providers.all import builtin_models
from pi_ai.types import AssistantMessage, Usage
from pi_durable import AssistantEntry, Harness, MemoryStorage, ProviderDoc, create_registry

#: The service caches only prompts of at least this many tokens, in steps of this many (OpenAI: 1024, then 128).
MIN_CACHED_TOKENS = 1024
CACHE_STEP = 128


def mock_token(account_id: str = "acc_test") -> str:
    payload = base64.b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    ).decode()
    return f"aaa.{payload}.bbb"


def cache_probe(nonce: str) -> str:
    lines = [
        "This is a real automated prompt-cache test. The repeated context below is intentional.",
        "Read it silently, then reply with exactly: CACHE PROBE READY",
    ]
    for index in range(180):
        lines.append(
            f"{nonce} immutable cache record {str(index).rjust(3, '0')}: alpha beta gamma delta epsilon zeta eta theta"
            " iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
        )
    return "\n".join(lines)


def tokens(text: str) -> int:
    return (len(text) + 3) // 4


def segments(body: dict[str, Any]) -> list[str]:
    """What the service caches, in prompt order: instructions, tools, then each input item."""
    return [
        json.dumps(body.get("instructions"), sort_keys=True),
        json.dumps(body.get("tools"), sort_keys=True),
        *(json.dumps(item, sort_keys=True) for item in body.get("input", [])),
    ]


class CachingCodex:
    """A mocked Codex Responses endpoint that answers every request and simulates its prompt cache."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._previous: dict[str, list[str]] = {}

    def _usage(self, body: dict[str, Any]) -> tuple[int, int]:
        current = segments(body)
        total = sum(tokens(segment) for segment in current)
        key = body.get("prompt_cache_key")
        previous = self._previous.get(key, []) if key is not None else []
        shared = 0
        for before, now in zip(previous, current, strict=False):
            if before != now:
                break
            shared += tokens(now)
        if key is not None:
            self._previous[key] = current
        cached = 0 if shared < MIN_CACHED_TOKENS else shared - shared % CACHE_STEP
        return total, cached

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        body = json.loads(request.content)
        self.bodies.append(body)
        total, cached = self._usage(body)
        text = "CACHE PROBE READY" if len(self.bodies) == 1 else f"CACHE PROBE FOLLOW-UP {len(self.bodies) - 1}"
        item = {
            "type": "message",
            "id": f"msg_{len(self.bodies)}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
        events = [
            {"type": "response.output_item.added", "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.output_item.done", "item": item},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": total,
                        "output_tokens": 5,
                        "total_tokens": total + 5,
                        "input_tokens_details": {"cached_tokens": cached},
                    },
                },
            },
        ]
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
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


async def assistant(harness: Harness, conversation_id: Any, answer: Any) -> AssistantMessage | None:
    record = await harness.commit(lambda tx: tx.entry(AssistantEntry, answer), context)
    if record is None or record["conversationId"] != conversation_id:
        raise RuntimeError("Answer belongs to another conversation")
    model = record.get("model")
    return model[0] if model else None  # type: ignore[return-value]


def cache_hit_rate(usage: Usage) -> float:
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    return 0 if prompt_tokens == 0 else usage.cache_read / prompt_tokens


async def run_turns(models: Any) -> tuple[Harness, Any, list[Usage], int | None]:
    harness = await Harness.open(
        MemoryStorage(),
        {
            "models": models,
            "registry": create_registry(),
            "settings": {"stream": {"transport": "sse"}, "compaction": {"enabled": False}},  # type: ignore[typeddict-item]
        },
        context,
    )
    root = await harness.root(context, agent={"model": {"provider": "openai-codex", "modelId": "gpt-6.1-sol"}})
    harness.resume()
    usages: list[Usage] = []
    first_hit: int | None = None
    for turn in range(8):
        content = cache_probe(uuidv7()) if turn == 0 else f"Reply exactly: CACHE PROBE FOLLOW-UP {turn}"
        settled = await (await root.submit({"type": "input", "content": content}, context)).wait(context)
        if settled["status"] != "done" or settled["type"] != "input":
            raise RuntimeError(f"Unexpected submission status: {settled['status']}")
        message = await assistant(harness, root.id, settled["answer"])
        assert message is not None
        assert message.stop_reason != "error", message.error_message
        usages.append(message.usage)
        if first_hit is None and message.usage.cache_read > 0:
            first_hit = len(usages) - 1
        if first_hit is not None and len(usages) - first_hit >= 3:
            break
    return harness, root, usages, first_hit


async def test_reuses_the_codex_prompt_cache_across_durable_turns_offline() -> None:
    server = CachingCodex()
    forwarded: list[str] = []

    def stream(model: Any, request: Any, options: Any) -> Any:
        options = dict(options or {})
        forwarded.append(options.get("session_id") or "")
        return codex.stream_simple(
            model,
            normalize_context(request),
            {**options, "api_key": mock_token(), "transport": "sse", "http_client": server.client()},
        )

    harness, root, usages, first_hit = await run_turns(Forwarding(builtin_models(), stream))
    try:
        provider = await harness.snapshot(ProviderDoc, root.id, context)
        assert provider is not None
        session_id = provider["sessionId"]
        assert forwarded == [session_id] * len(usages)
        # The session ID reaches the service as the prompt cache key of every request.
        assert [body.get("prompt_cache_key") for body in server.bodies] == [session_id] * len(usages)
        first_prompt = usages[0].input + usages[0].cache_read + usages[0].cache_write
        assert first_prompt >= 4_000
        assert usages[0].cache_read == 0
        assert first_hit == 1
        cached = usages[first_hit:]
        assert len(cached) == 3
        for usage in cached:
            assert cache_hit_rate(usage) >= 0.9
    finally:
        await harness.close(context)


@pytest.mark.skip(
    reason="live network: needs OpenAI Codex credentials (upstream resolveApiKey from ~/.pi/agent/auth.json); "
    "covered offline by test_reuses_the_codex_prompt_cache_across_durable_turns_offline"
)
async def test_reuses_the_codex_prompt_cache_across_durable_turns() -> None:
    pass
