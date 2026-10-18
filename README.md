# pi-py

Python port of [pi](https://github.com/earendil-works/pi)'s core libraries:

| Package | Import | Upstream | What it does |
| --- | --- | --- | --- |
| [`pi-ai`](packages/ai/README.md) | `pi_ai` | `@earendil-works/pi-ai` | Unified LLM API: provider collections, auth resolution (API keys, OAuth), streaming events, token and cost tracking, tool calling, cross-provider context hand-off, image generation, classification |
| [`pi-agent-core`](packages/agent/README.md) | `pi_agent` | `@earendil-works/pi-agent-core` | Stateful agent: tool execution (parallel or sequential), steering and follow-up queues, lifecycle events, hooks |

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

```python
import asyncio

from pi_ai import Context, UserMessage
from pi_ai.providers.all import builtin_models


async def main() -> None:
    models = builtin_models()  # every built-in provider; auth comes from env vars such as ANTHROPIC_API_KEY
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    stream = models.stream(model, Context(system_prompt="Be brief.", messages=[UserMessage("Hello!")]))
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    message = await stream.result()
    print(f"\n{message.usage.input} in / {message.usage.output} out, ${message.usage.cost.total:.4f}")


asyncio.run(main())
```

```python
from pi_agent import Agent, AgentInitialState, agent_tool


@agent_tool
async def get_time(timezone: str = "UTC") -> str:
    """Get the current time.

    Args:
        timezone: IANA timezone name.
    """
    ...


agent = Agent(
    initial_state=AgentInitialState(system_prompt="You are helpful.", model=model, tools=[get_time]),
    stream_fn=models.stream_simple,
)
async for event in agent.events("What time is it in Tokyo?"):
    print(event.type)
```

## Same API, Python conventions

The port keeps pi's architecture and semantics: the same `Models`/`Provider`
runtime, the same event protocol (`start` → `text_delta`… → `done`/`error`), the
same transcript model (system prompt and tools carried by system messages), the
same auth precedence rules, the same model catalog, and the same agent loop.
Names follow Python conventions:

| TypeScript | Python |
| --- | --- |
| `createModels()`, `models.streamSimple(...)` | `create_models()`, `models.stream_simple(...)` |
| `{ apiKey, sessionId, maxTokens }` options | `api_key=..., session_id=..., max_tokens=...` keyword arguments (or a dict) |
| `message.stopReason`, `usage.cacheRead` | `message.stop_reason`, `usage.cache_read` |
| `{ role: "user", content: "hi", timestamp }` | `UserMessage("hi")` (timestamp defaults to now) |
| `Type.Object({...})` (TypeBox) | `Type.Object({...})` (built in) or a Pydantic model |
| `AbortController` / `signal` | `pi_ai.AbortController` / `signal=` |
| `for await (const e of stream)` / `await stream.result()` | `async for e in stream` / `await stream.result()` |
| `agent.continue()` | `agent.continue_()` (alias `agent.resume()`) |

String values that appear on the wire or in persisted transcripts are unchanged
(`"toolUse"`, `"toolResult"`, `"text_delta"`, API ids, provider ids), and
`pi_ai.serialization` reads and writes pi's camelCase JSON, so sessions, model
definitions and `auth.json` credential files are interchangeable with TypeScript pi.

## Design notes

* **Official vendor SDKs.** As in pi, the Anthropic, OpenAI (Chat Completions,
  Responses, Azure, OpenRouter images), Google (Gemini, Vertex) and Bedrock adapters use
  the vendors' official SDKs: `anthropic`, `openai`, `google-genai` and `boto3`
  (`pi-ai[bedrock]`). The SDK clients share one pooled `httpx.AsyncClient` per event loop.
  APIs that pi calls without an SDK (Mistral, OpenAI Codex, pi-messages, classifiers)
  use `httpx` directly.
* **Fast streaming path.** Slotted data classes, an allocation-light event queue,
  zero-copy forwarding between wrapped streams, an incremental partial-JSON parser
  for tool-call arguments (about 60x faster than re-parsing each delta on large
  arguments), linear-time accumulation of streamed text, optional `orjson` decoding,
  and compiled, cached tool-argument validators. On one core the anthropic-messages
  adapter processes about 110,000 stream events per second and openai-completions about
  70,000, where the openai SDK's SSE decoder is the limit (see `benchmarks/`).
* **Lazy everything.** Model catalogs load per provider on first use; API
  implementations and OAuth flows import on first request.
* **Async-native cancellation.** `AbortSignal` aborts interrupt pending network reads
  immediately (task cancellation scoped by `abort_scope`), and aborted requests end
  their stream with `stop_reason == "aborted"` instead of raising.
* **Python-friendly additions.** Pydantic models as tool parameter schemas,
  `@agent_tool` to build tools from typed functions, `Agent.events()` as an async
  iterator, `FileCredentialStore` (pi's `auth.json` format with cross-process locking),
  keyword-argument options, streams that can be created outside an event loop and
  closed with `async with`.

## Development

```bash
uv sync                      # create .venv with both packages (editable) and dev tools
uv run pytest                # run all tests (offline; HTTP is mocked)
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run python benchmarks/bench_partial_json.py
uv run python benchmarks/bench_stream.py
```

Model catalog data (`packages/ai/src/pi_ai/providers/data/*.json`) is the generated
catalog shipped with `@earendil-works/pi-ai` 1.3.0.

## License

MIT. Based on pi by Mario Zechner and contributors.
