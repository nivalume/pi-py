"""End-to-end streaming throughput: SSE bytes in, pi events out.

Serves a canned response through ``httpx.MockTransport`` (no network) and
measures how fast the openai-completions and anthropic-messages adapters turn it
into events: 20,000 text deltas followed by a tool call whose arguments arrive
in 2,000 deltas.

Run: python benchmarks/bench_stream.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from pi_ai import Context, Type, UserMessage, normalize_context
from pi_ai.api import anthropic_messages, openai_completions
from pi_ai.providers.catalog import get_builtin_model
from pi_ai.types import Tool

TEXT_DELTAS = 20_000
ARG_DELTAS = 2_000


def sse(events: list[Any]) -> bytes:
    lines = []
    for event in events:
        if isinstance(event, tuple):
            name, data = event
            lines.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
        else:
            lines.append(f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n")
    return "".join(lines).encode()


def argument_chunks() -> list[str]:
    content = "x = 1\\n" * (ARG_DELTAS * 2)
    text = json.dumps({"path": "a.py", "content": content})
    size = max(1, len(text) // ARG_DELTAS)
    return [text[i : i + size] for i in range(0, len(text), size)]


def openai_body() -> bytes:
    events: list[Any] = [{"id": "c1", "choices": [{"delta": {"role": "assistant", "content": ""}}]}]
    events += [{"choices": [{"delta": {"content": "token "}}]} for _ in range(TEXT_DELTAS)]
    chunks = argument_chunks()
    events.append(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "write", "arguments": ""}}]
                    }
                }
            ]
        }
    )
    events += [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": c}}]}}]} for c in chunks]
    events.append(
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": TEXT_DELTAS},
        }
    )
    events.append("[DONE]")
    return sse(events)


def anthropic_body() -> bytes:
    usage = {"input_tokens": 10, "output_tokens": 1}
    events: list[Any] = [
        ("message_start", {"type": "message_start", "message": {"id": "m1", "usage": usage}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "token "}}
    events += [("content_block_delta", delta)] * TEXT_DELTAS
    events.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "write", "input": {}},
            },
        )
    )
    for chunk in argument_chunks():
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": chunk},
                },
            )
        )
    events.append(("content_block_stop", {"type": "content_block_stop", "index": 1}))
    events.append(
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": TEXT_DELTAS}},
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return sse(events)


def client(body: bytes) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def measure(label: str, module: Any, model: Any, body: bytes, rounds: int = 3) -> None:
    tool = Tool("write", "Write a file", Type.Object({"path": Type.String(), "content": Type.String()}))
    context = normalize_context(Context(messages=[UserMessage("hi")], tools=[tool]))
    best = float("inf")
    count = 0
    for _ in range(rounds):
        http = client(body)
        started = time.perf_counter()
        stream = module.stream(model, context, {"api_key": "k", "http_client": http, "cache_retention": "none"})
        count = 0
        async for _event in stream:
            count += 1
        message = await stream.result()
        best = min(best, time.perf_counter() - started)
        await http.aclose()
        assert message.stop_reason == "toolUse", message.error_message
    print(
        f"{label:<22} {count:>7} events  {best * 1000:8.1f} ms  {count / best:>10,.0f} events/s  ({len(body) / 1e6:.1f} MB)"
    )


async def main() -> None:
    openai = get_builtin_model("openai", "gpt-4o-mini")
    anthropic = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert openai is not None and anthropic is not None
    import dataclasses

    openai = dataclasses.replace(openai, api="openai-completions")
    await measure("openai-completions", openai_completions, openai, openai_body())
    await measure("anthropic-messages", anthropic_messages, anthropic, anthropic_body())


if __name__ == "__main__":
    asyncio.run(main())
