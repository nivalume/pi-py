# pi-py

Python port of [pi](https://github.com/earendil-works/pi)'s core libraries, published on
PyPI as one package, `pi-ai-python`, with these import packages:

| Import | Upstream | What it does |
| --- | --- | --- |
| [`pi_ai`](https://github.com/nivalume/pi-py/blob/main/packages/ai/README.md) | `@earendil-works/pi-ai` | Unified LLM API: provider collections, auth resolution (API keys, OAuth), streaming events, token and cost tracking, tool calling, cross-provider context hand-off, image generation, classification |
| [`pi_agent`](https://github.com/nivalume/pi-py/blob/main/packages/agent/README.md) | `@earendil-works/pi-agent-core` | Stateful agent: tool execution (parallel or sequential), steering and follow-up queues, lifecycle events, hooks |
| [`pi_durable`](https://github.com/nivalume/pi-py/blob/main/packages/durable/README.md) | `@earendil-works/pi-durable` | Durable agent harness: conversations, model turns, tool calls and your own state are committed to storage (memory, SQLite, JSONL) before anything is shown, and work resumes after a crash or restart |
| [`pi_chord`](https://github.com/nivalume/pi-py/blob/main/packages/chord/README.md) | `@earendil-works/chord` (subset) | What `pi_durable` builds on: explicit cancellation contexts, strict JSON, the transactional JSON tracker and op batches, replicated state, bounded output, an async SQLite layer |
| [`pi_codemode`](https://github.com/nivalume/pi-py/blob/main/packages/codemode/README.md) | `@earendil-works/pi-codemode` | Code mode: model-written JavaScript run in QuickJS (upstream's build, on wasmtime) or, as an alternative, Python run in Monty, whose only capability is calling injected tools; TypeScript (or Python) declarations for the model, deterministic runs for durable replay |
| [`pi_telemetry`](https://github.com/nivalume/pi-py/blob/main/packages/telemetry/README.md) | `@earendil-works/pi-telemetry` | Vendor-neutral telemetry: explicit callback-based spans, typed schemas, an in-memory reference adapter and conformance suite, an OpenTelemetry adapter with GenAI conventions and metrics; the other packages record `pi.ai.*`, `pi.agent.*`, `pi.harness.*`, `pi.session.*` and `pi.codemode.*` spans when given a context |

```bash
pip install pi-ai-python               # httpx + official SDKs: anthropic, openai, google-genai
pip install "pi-ai-python[bedrock]"    # + boto3 for Amazon Bedrock
pip install "pi-ai-python[images]"     # + Pillow, for pi_durable's image processor
pip install "pi-ai-python[codemode]"   # + wasmtime, pi_codemode's JavaScript sandbox (QuickJS)
pip install "pi-ai-python[codemode-python]"  # + pydantic-monty, pi_codemode's Python sandbox (Monty)
pip install "pi-ai-python[otel]"       # + opentelemetry-api, for pi_telemetry's OpenTelemetry adapter
pip install "pi-ai-python[all]"        # + orjson, websockets, pydantic, boto3, pillow, wasmtime, pydantic-monty, opentelemetry-api
```

中文说明见 [README.zh-CN.md](https://github.com/nivalume/pi-py/blob/main/README.zh-CN.md)。

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

A durable conversation keeps its transcript, running turns and tool calls in storage. Run
the same script again after a crash and `resume()` picks the work up where it stopped:

```python
import asyncio

from pi_ai import create_models
from pi_ai.providers.openai import openai_provider
from pi_chord import BACKGROUND_CONTEXT
from pi_durable import AssistantEntry, Harness, create_registry
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage


async def main() -> None:
    context = BACKGROUND_CONTEXT
    models = create_models()
    models.set_provider(openai_provider())  # reads OPENAI_API_KEY

    storage = await open_sqlite_storage("./session.sqlite")
    harness = await Harness.open(storage, {"models": models, "registry": create_registry()}, context)
    root = await harness.root(context, agent={"model": {"provider": "openai", "modelId": "gpt-6-sol"}})
    harness.resume()  # continue any run an earlier process left unfinished

    submission = await root.submit({"type": "input", "content": "What is the capital of France?"}, context)
    settled = await submission.wait(context)
    if settled["status"] == "done" and settled["type"] == "input":
        answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
        print(answer["model"][0].content[0].text if answer else None)
    await harness.close(context)


asyncio.run(main())
```

`pi_durable` stores records in the same format as TypeScript pi-durable, so a SQLite or JSONL
storage written by one opens in the other. The coding tools (`pi_durable.tools`: read, write,
edit, bash), tool search, an image processor on Pillow, and the code mode extension
(`pi_durable.codemode`, whose scripts are JavaScript run by QuickJS, or Python run by Monty, and survive a restart)
are included;
not ported yet: the Cloudflare adapters.

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
  (`pi-ai-python[bedrock]`). The SDK clients share one pooled `httpx.AsyncClient` per event loop.
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
* **Explicit telemetry.** `pi_telemetry` ports pi's callback-based span contract (no ambient current span). Pass a
  context (`telemetry_context=`, or `HarnessOptions["telemetryContext"]`) and the packages record `pi.ai.request`,
  `pi.agent.run`, `pi.harness.*`, `pi.session.commit` and `pi.codemode.run` spans, with no content by default;
  `pi_telemetry.otel` exports them to OpenTelemetry with the GenAI semantic conventions and metrics. Without a
  context nothing is recorded and the streaming path is unchanged. Upstream 1.3.0 only propagates the context.
* **Python-friendly additions.** Pydantic models as tool parameter schemas,
  `@agent_tool` to build tools from typed functions, `Agent.events()` as an async
  iterator, `FileCredentialStore` (pi's `auth.json` format with cross-process locking),
  keyword-argument options, streams that can be created outside an event loop and
  closed with `async with`.

## Development

```bash
uv sync                      # create .venv with the package (editable) and dev tools
uv run pytest                # run all tests (offline; HTTP is mocked)
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run python benchmarks/bench_partial_json.py
uv run python benchmarks/bench_stream.py
uv run python benchmarks/bench_chord_delta.py
uv run python benchmarks/bench_chord_sqlite.py
uv run python benchmarks/bench_durable_storage.py --quick
uv run python benchmarks/bench_codemode.py --quick
```

The port tracks pi 1.3.0 (`@earendil-works/pi-ai`, `pi-agent-core`, `pi-durable`, `pi-codemode`, `pi-telemetry`, `chord`).

Model catalog data (`packages/ai/src/pi_ai/providers/data/*.json`) is the generated
catalog shipped with `@earendil-works/pi-ai` 1.3.0.

## License

MIT. Based on pi by Mario Zechner and contributors.
