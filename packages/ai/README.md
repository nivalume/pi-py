# pi-ai (Python)

Unified LLM API with provider collections, automatic auth resolution, token and cost
tracking, and simple context persistence and hand-off to other models mid-session.
Python port of `@earendil-works/pi-ai`: same architecture, same event protocol, same
model catalog, Pythonic names.

**Note**: the chat catalog only includes models that support tool calling, as this is
essential for agentic workflows. Image and classifier catalogs use their own capabilities.

## Table of contents

- [Supported providers](#supported-providers)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Providers and models](#providers-and-models)
- [Auth](#auth)
- [Tools](#tools)
- [Image input](#image-input)
- [Image generation](#image-generation)
- [Classification](#classification)
- [Thinking / reasoning](#thinking--reasoning)
- [Stop reasons](#stop-reasons)
- [Error handling and cancellation](#error-handling-and-cancellation)
- [Custom providers](#custom-providers)
- [Faux provider for tests](#faux-provider-for-tests)
- [Cross-provider hand-offs](#cross-provider-handoffs)
- [System messages](#system-messages)
- [Context serialization](#context-serialization)
- [OAuth providers](#oauth-providers)
- [Legacy global API](#legacy-global-api)
- [Performance notes](#performance-notes)
- [Differences from the TypeScript package](#differences-from-the-typescript-package)

## Supported providers

OpenAI (incl. the Decisions classifier API), Ant Ling, Azure OpenAI (Responses), OpenAI
Codex (ChatGPT Plus/Pro, OAuth), Radius (API key or OAuth, dynamic catalog), TypeSafe
(System One classifiers), DeepSeek, NVIDIA NIM, Anthropic, Google, Vertex AI, Mistral,
Groq, Cerebras, Cloudflare AI Gateway, Cloudflare Workers AI, xAI, OpenRouter, Vercel AI
Gateway, ZAI Coding Plan (global and China), MiniMax (global and China), Together AI,
Baseten, Hugging Face, Moonshot AI (global and China), GitHub Copilot (OAuth), Amazon
Bedrock, OpenCode Zen, OpenCode Go, Fireworks, Kimi For Coding, Meta, Qwen Token Plan
(Individual, existing and China), Xiaomi MiMo (API billing and Token Plans), and any
OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp, ...).

## Installation

```bash
pip install pi-ai               # httpx + official SDKs: anthropic, openai, google-genai
pip install "pi-ai[bedrock]"    # + boto3 for Amazon Bedrock
pip install "pi-ai[fast]"       # + orjson for faster stream decoding
pip install "pi-ai[websocket]"  # + websockets for the OpenAI Codex WebSocket transport
pip install "pi-ai[pydantic]"   # + pydantic, to use models as tool parameter schemas
pip install "pi-ai[all]"
```

Python 3.11+. Everything is `asyncio`-based.

## Quick start

```python
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from pi_ai import Context, TextContent, Tool, ToolResultMessage, Type, UserMessage
from pi_ai.providers.all import builtin_models


async def main() -> None:
    # A Models collection with every built-in provider registered
    models = builtin_models()
    model = models.get_model("openai", "gpt-4o-mini")

    tools = [
        Tool(
            "get_time",
            "Get the current time",
            Type.Object({"timezone": Type.Optional(Type.String(description="IANA timezone, e.g. America/New_York"))}),
        )
    ]
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage("What time is it?")],
        tools=tools,
    )

    # Option 1: streaming with all event types. Auth resolves through the provider
    # (OPENAI_API_KEY from the environment here).
    stream = models.stream(model, context)
    async for event in stream:
        match event.type:
            case "start":
                print(f"Starting with {event.partial.model}")
            case "text_delta" | "thinking_delta":
                print(event.delta, end="", flush=True)
            case "toolcall_end":
                print(f"\nTool called: {event.tool_call.name}({event.tool_call.arguments})")
            case "done":
                print(f"\nFinished: {event.reason}")
            case "error":
                print(f"Error: {event.error.error_message}")

    final = await stream.result()
    context.messages.append(final)

    # Handle tool calls
    for call in (block for block in final.content if block.type == "toolCall"):
        zone = ZoneInfo(call.arguments.get("timezone") or "UTC")
        context.messages.append(
            ToolResultMessage(call.id, call.name, [TextContent(datetime.now(zone).isoformat())])
        )

    # Option 2: get the complete response without streaming
    if final.stop_reason == "toolUse":
        continuation = await models.complete(model, context)
        context.messages.append(continuation)
        print(continuation.content)

    print(f"Tokens: {final.usage.input} in, {final.usage.output} out; cost ${final.usage.cost.total:.4f}")


asyncio.run(main())
```

Options can be passed as keyword arguments or as a dict (both styles merge):

```python
await models.complete(model, context, max_tokens=512, temperature=0.2, session_id="s1")
await models.complete(model, context, {"max_tokens": 512})
```

## Providers and models

A **provider** is the runtime unit: it owns its model catalog, its auth (API key
resolution, OAuth flows), and its stream behavior. A `Models` collection holds
providers and routes every request to the provider that owns the model. Providers
share **API implementations** (wire protocols): Anthropic models use
`anthropic-messages`, OpenAI uses `openai-responses`, most others share
`openai-completions`. Mixed-API providers (GitHub Copilot, OpenCode Zen) dispatch per model.

### Provider factories

One factory per built-in provider; each loads only its own catalog:

```python
from pi_ai import create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.openrouter import openrouter_provider

models = create_models()
models.set_provider(anthropic_provider())
models.set_provider(openrouter_provider())
```

`pi_ai.providers.all.builtin_models()` registers every built-in provider and accepts
the same options as `create_models()` (`credentials`, `models_store`, `auth_context`);
`builtin_providers()` returns the provider list.

### Querying models

Reads are synchronous and return the last-known lists:

```python
models.get_providers()
models.get_provider("anthropic")
models.get_models()                      # every chat model
models.get_models("anthropic")
model = models.get_model("anthropic", "claude-sonnet-4-5")
print(model.id, model.api, model.context_window, "image" in model.input, model.reasoning)

models.get_models_of_type("image", "openrouter")
models.get_model_of_type("classifier", "typesafe", "jev-latest")
await models.get_available_of_type("image")   # only providers with configured auth
models.get_all_models()                       # every type
```

A model's `type` (`"chat"`, `"image"`, `"classifier"`) decides which operation accepts
it. Use `is_model_type(model, "image")` / `get_model_type(model)` on mixed lists and
`has_api(model, "anthropic-messages")` before passing API-specific options.

### Static catalog reads

```python
from pi_ai.providers.all import (
    get_builtin_model, get_builtin_models, get_builtin_image_model,
    get_builtin_classifier_model, get_builtin_providers, get_all_builtin_models,
)

get_builtin_model("openai", "gpt-4o-mini")
get_builtin_image_model("openrouter", "black-forest-labs/flux.2-pro")
```

### Dynamic providers

```python
await models.refresh(providers=["llamacpp"])   # one provider
result = await models.refresh()                # all, concurrently, best effort
if result.aborted:
    ...
for provider_id, error in result.errors.items():
    print(provider_id, error)
```

`refresh(allow_network=False)` restores persisted catalogs offline; `force=True`
bypasses freshness checks. Static providers are no-ops. Radius ships a static catalog
and overlays its `/v1/config` gateway catalog when refreshed with configured auth.

## Auth

### How auth resolves

`models.stream()` resolves auth through the owning provider and merges it into the
request. Explicit per-request values always win:

```python
await models.complete(model, context)                      # env var, stored credential, or OAuth token
await models.complete(model, context, api_key="sk-explicit")

auth = await models.get_auth(model)          # or models.get_auth("anthropic")
if auth:
    print(auth.source, auth.auth.headers)    # e.g. "ANTHROPIC_API_KEY"
```

`get_auth()` returns `None` for unconfigured providers and raises `ModelsError` when
something is broken (`code == "oauth"`: token refresh failed and the credential is kept
for re-login; `code == "auth"`: key resolution or credential store failure). Request
paths surface the same failures as stream errors.

### Transforming request headers

```python
import uuid

await models.complete_simple(
    model,
    context,
    headers={"X-Client": "my-app"},
    transform_headers=lambda headers: {**headers, "X-Request-ID": str(uuid.uuid4())},
)
```

Order: provider auth headers → `model.headers` → explicit `headers` → `transform_headers`
→ provider. Header names merge case-insensitively; a `None` value suppresses a default.

### Credential store

Stored credentials (keys entered interactively, OAuth tokens) live in a
`CredentialStore`, one credential per provider. The default is in memory;
`FileCredentialStore` uses pi's `auth.json` format (atomic writes, `0600`, a
cross-process file lock on POSIX):

```python
from pi_ai import FileCredentialStore, create_models

models = create_models(credentials=FileCredentialStore("~/.config/my-app/auth.json"))
```

Custom stores implement async `read`, `list`, `modify(provider_id, fn)` (the only write
path, a serialized read-modify-write; OAuth refresh runs inside it) and `delete`.
Credentials are plain dicts:

```python
{"type": "api_key", "key": "...", "env": {"CLOUDFLARE_ACCOUNT_ID": "...", "CLOUDFLARE_GATEWAY_ID": "..."}}
{"type": "oauth", "access": "...", "refresh": "...", "expires": 1767225600000}
```

A stored credential owns its provider: environment variables are consulted only when
nothing is stored, and a failed refresh never falls back to an env key.

### Environment variables

| Provider | Variables |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN` or `ANTHROPIC_AUTH_TOKEN` (bearer) |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_API_KEY`, or `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` + ADC |
| Amazon Bedrock | `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, ECS/IRSA credentials |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` or `AZURE_OPENAI_RESOURCE_NAME` |
| Mistral / Groq / Cerebras / xAI | `MISTRAL_API_KEY` / `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `XAI_API_KEY` |
| OpenRouter / Vercel AI Gateway | `OPENROUTER_API_KEY` / `AI_GATEWAY_API_KEY` |
| Cloudflare | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` (+ `CLOUDFLARE_GATEWAY_ID` for AI Gateway) |
| DeepSeek / NVIDIA / Together / Baseten / Fireworks | `DEEPSEEK_API_KEY` / `NVIDIA_API_KEY` / `TOGETHER_API_KEY` / `BASETEN_API_KEY` / `FIREWORKS_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| ZAI / MiniMax / Moonshot / Kimi | `ZAI_API_KEY`, `ZAI_CODING_CN_API_KEY` / `MINIMAX_API_KEY`, `MINIMAX_CN_API_KEY` / `MOONSHOT_API_KEY` / `KIMI_API_KEY` |
| OpenCode | `OPENCODE_API_KEY` |
| Meta, Radius, TypeSafe, Ant Ling | `META_API_KEY`, `RADIUS_API_KEY`, `TYPESAFE_API_KEY`, `ANT_LING_API_KEY` |
| Qwen Token Plan | `QWEN_TOKEN_PLAN_API_KEY`, `QWEN_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo | `XIAOMI_API_KEY`, `XIAOMI_TOKEN_PLAN_{CN,AMS,SGP}_API_KEY` |
| GitHub Copilot | `COPILOT_GITHUB_TOKEN` |

### Provider-scoped environment overrides

Pass `env` to scope provider configuration to a single request (account ids, Azure
settings, Vertex project/location, `PI_CACHE_RETENTION`, `HTTP(S)_PROXY`). Values take
precedence over `os.environ`:

```python
await models.complete(model, context, env={
    "CLOUDFLARE_API_KEY": "...", "CLOUDFLARE_ACCOUNT_ID": "account-id", "CLOUDFLARE_GATEWAY_ID": "gateway-id",
})
```

## Tools

### Defining tools

Tool parameters are JSON Schema. Build them with the TypeBox-style `Type` helpers, write
the dict by hand, or pass a Pydantic model class:

```python
from pydantic import BaseModel, Field
from pi_ai import StringEnum, Tool, Type

weather = Tool(
    "get_weather",
    "Get current weather for a location",
    Type.Object({
        "location": Type.String(description="City name or coordinates"),
        "units": StringEnum(["celsius", "fahrenheit"], default="celsius"),
    }),
)

book_meeting = Tool(
    "book_meeting",
    "Schedule a meeting",
    Type.Object({
        "title": Type.String(min_length=1),
        "start_time": Type.String(format="date-time"),
        "attendees": Type.Array(Type.String(format="email"), min_items=1),
    }),
)


class Search(BaseModel):
    query: str = Field(description="Search terms")
    limit: int = 10


search = Tool("search", "Search the docs", Search)   # validated arguments are Search instances
```

Use `StringEnum` instead of `Type.Enum` for Google compatibility (Google rejects
`anyOf`/`const` enums). `Type.Optional(...)` marks an object property as not required.

### Constrained sampling

```python
strict_tool = Tool(
    "edit_file", "Edit a file",
    Type.Object({"path": Type.String(), "content": Type.String()}, additional_properties=False),
    constrained_sampling={"type": "json_schema", "strict": "prefer"},   # or "require"
)

patch_tool = Tool(
    "apply_patch", "Apply a patch",
    Type.Object({"input": Type.String()}, additional_properties=False),
    constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/s"}},
)
```

`"prefer"` uses provider-side strict schemas when supported and otherwise falls back;
`"require"` fails the request when the provider/model cannot honor it. Strict JSON
schemas are supported by OpenAI, Anthropic, capable Bedrock models, Mistral and Gemini 3;
grammar tools by GPT-5+ on endpoints that pass OpenAI custom tools through.

### Handling tool calls

```python
response = await models.complete(model, context)
for block in response.content:
    if block.type == "toolCall":
        result = await execute_weather_api(block.arguments)
        context.messages.append(
            ToolResultMessage(block.id, block.name, [TextContent(json.dumps(result))])
        )

# Tool results can include images for vision models
context.messages.append(ToolResultMessage(
    "tool_xyz", "generate_chart",
    [TextContent("Generated chart"), ImageContent(base64.b64encode(png).decode(), "image/png")],
))
```

### Streaming tool calls with partial JSON

During `toolcall_delta` events, `event.partial.content[event.content_index].arguments`
holds the best-effort parse of the arguments so far (at least `{}`); fields may be
missing or truncated. `toolcall_end` carries the complete (not yet validated) call. When
the final arguments text was not a valid JSON object, `tool_call.arguments` is `{}` and
`tool_call.arguments_error` describes the problem.

### Validating tool arguments

```python
from pi_ai import ToolValidationError, validate_tool_call

try:
    args = validate_tool_call(tools, tool_call)        # coerces "42" -> 42 etc. like pi
except ToolValidationError as error:
    context.messages.append(ToolResultMessage(tool_call.id, tool_call.name, [TextContent(str(error))], True))
```

Validation compiles each schema once into a cached validator and reports every problem
with its path (`  - attendees: must NOT have fewer than 1 items`).

### Event reference

Successful generation is `start → updates* → done`; failures after start are
`start → updates* → error`; setup failures emit only `error`.

| Event | Description | Key fields |
| --- | --- | --- |
| `start` | Stream begins | `partial` |
| `text_start` / `text_delta` / `text_end` | Text block | `content_index`, `delta`, `content` |
| `thinking_start` / `thinking_delta` / `thinking_end` | Thinking block | `content_index`, `delta`, `content` |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | Tool call | `content_index`, `delta`, `tool_call` |
| `done` | Complete | `reason` (`"stop"`, `"length"`, `"toolUse"`, `"deferred"`), `message` |
| `error` | Failed or aborted | `reason` (`"error"`, `"aborted"`), `error` (message with partial content) |

`partial` is the shared live message, not a snapshot: inspect it when handling an event.
Events for different blocks can interleave; use `content_index` to associate them. All
events are slotted dataclasses (`TextDeltaEvent`, `ToolCallEndEvent`, ...), so
`match event: case TextDeltaEvent(delta=d): ...` works too.

## Image input

```python
if "image" in model.input:
    response = await models.complete(model, Context(messages=[UserMessage([
        TextContent("What is in this image?"),
        ImageContent(base64.b64encode(Path("image.png").read_bytes()).decode(), "image/png"),
    ])]))
```

Images sent to non-vision models are replaced by a placeholder.

## Image generation

```python
from pi_ai import ImagesContext, TextContent

model = models.get_model_of_type("image", "openrouter", "google/gemini-2.5-flash-image")
result = await models.generate_images(model, ImagesContext([TextContent("A red circle on white.")]))
for block in result.output:
    print(block.type, getattr(block, "mime_type", None))
```

`generate_images()` never raises; failures return `stop_reason == "error"`. Custom
providers declare image support with `create_provider(images={"<api>": impl})` where
`impl` has an async `generate_images(model, context, options)`.

## Classification

```python
from pi_ai import ClassifierBoolQuestion, ClassifierChoiceQuestion, ClassifierContext, ClassifierScoreQuestion

model = models.get_model_of_type("classifier", "typesafe", "jev-latest")
result = await models.classify(model, ClassifierContext(
    state={"message": "The change works perfectly, thanks."},
    questions={
        "category": ClassifierChoiceQuestion("Classify the message.", {
            "approval": "The user approves of the result",
            "correction": "The user requests a correction",
        }),
        "satisfaction": ClassifierScoreQuestion("Score user satisfaction.", ["dissatisfied", "neutral", "satisfied"]),
        "approved": ClassifierBoolQuestion("Does the user approve?", {"true": "Approval", "false": "No approval"}),
    },
))
print(result.answers)
```

Classifier APIs: TypeSafe/System One (`typesafe-system-one`, also via OpenRouter, Vercel,
OpenCode), Cloudflare Workers AI (`cloudflare-workers-ai-system-one`), OpenAI Decisions
(`openai-decisions`, GPT-6 Luna), and `llama-cpp-classify`, which turns a chat model served
by llama.cpp into a classifier using next-token log-probabilities. `classify()` never
raises; `options={"temperature": 2}` softens overconfident distributions where supported.

## Thinking / reasoning

### Unified interface

```python
response = await models.complete_simple(model, context, reasoning="medium")
for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
```

Levels: `"minimal"`, `"low"`, `"medium"`, `"high"`, and the model-specific opt-in levels
`"xhigh"`/`"max"`. `get_supported_thinking_levels(model)` lists what a model exposes and
`clamp_thinking_level(model, level)` picks the nearest supported one. `thinking_budgets`
overrides token budgets for token-based providers.

### Provider-specific options

`stream()`/`complete()` accept the owning API's full option set:

```python
if has_api(model, "openai-responses"):
    await models.complete(model, context, reasoning_effort="medium", reasoning_summary="detailed")

if has_api(model, "anthropic-messages"):
    await models.complete(model, context, thinking_enabled=True, thinking_budget_tokens=8192)

if has_api(model, "google-generative-ai"):
    await models.complete(model, context, thinking={"enabled": True, "budget_tokens": 8192})
```

## Stop reasons

`"stop"`, `"length"`, `"toolUse"`, `"error"`, `"aborted"`, `"deferred"`, and `"pending"`
(partial messages only). `message.response_id` holds the upstream response id when the
API exposes one; `message.raw_stop_reason` the provider's own value.

## Error handling and cancellation

Request failures never raise once a stream is returned: the stream ends with an `error`
event and the final message carries `error_message`, the partial content and the partial
usage. Direct API `stream_simple()` calls raise synchronously when required auth is missing.

```python
from pi_ai import AbortController

controller = AbortController()
asyncio.get_running_loop().call_later(2, controller.abort)

stream = models.stream(model, Context(messages=[UserMessage("Write a long story")]), signal=controller.signal)
async for event in stream:
    if event.type == "text_delta":
        print(event.delta, end="")
response = await stream.result()
if response.stop_reason == "aborted":
    context.messages.append(response)                      # aborted messages can be continued
    context.messages.append(UserMessage("Please continue"))
```

`abort()` may be called from any thread. Aborting interrupts pending network reads
immediately. You can also stop consuming and cancel the request with
`await stream.aclose()` (or `async with models.stream(...) as stream:`).

`is_context_overflow(message, context_window)` detects context overflow across providers
(error messages, silent overflow, truncated length stops); `is_retryable_assistant_error`
and `retry_assistant_call(produce, RetryPolicy(...), signal)` implement pi's retry policy.
Requests also accept `max_retries`, `max_retry_delay_ms` and `timeout_ms` (HTTP retries
honoring `retry-after`; Bedrock maps `max_retries` onto botocore's retry config).

### Debugging payloads and raw events

```python
await models.complete(model, context, on_payload=lambda payload, model: print(json.dumps(payload, indent=2)))
await models.complete(model, context, on_response=lambda response, model: print(response.status, response.headers))
await models.complete(model, context, on_provider_stream_event=lambda chunk, model: print(chunk))
```

`on_payload` may return a replacement payload. Callbacks may be sync or async and are
awaited in stream order.

## Custom providers

### `create_provider()`

```python
from pi_ai import Model, ModelCost, create_models, create_provider, env_api_key_auth, keyless_auth, ProviderAuth
from pi_ai.api.registry import openai_completions_api

ollama_model = Model(
    id="llama-3.1-8b",
    name="Llama 3.1 8B (Ollama)",
    api="openai-completions",
    provider="ollama",
    base_url="http://localhost:11434/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(0, 0, 0, 0),
    context_window=128_000,
    max_tokens=32_000,
    compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False},
)

ollama = create_provider(
    id="ollama",
    name="Ollama",
    base_url="http://localhost:11434/v1",
    auth=ProviderAuth(api_key=keyless_auth("Ollama")),   # keyless local server
    models=[ollama_model],
    api=openai_completions_api(),
)

models = create_models()
models.set_provider(ollama)
await models.complete(models.get_model("ollama", "llama-3.1-8b"), context)
```

For providers with keys use `env_api_key_auth("My proxy API key", ["MY_PROXY_API_KEY"])`
(stored credential wins, then the first set env var). Mixed-API providers pass a dict:
`api={"anthropic-messages": anthropic_messages_api(), "openai-responses": openai_responses_api()}`.
Dynamic catalogs use `fetch_models=async (context) -> [models]`; `create_provider` persists
them through the `ModelsStore` and restores them on `refresh(allow_network=False)`.
`models=` may also be a zero-argument callable, evaluated on first access.

`compat` is a plain dict with pi's `models.json` keys (camelCase), so existing pi model
configuration can be pasted as is. Unset fields are auto-detected from the base URL.
`thinking_level_map` maps pi levels to provider values (`None` marks a level unsupported).

### Calling API implementations directly

```python
from pi_ai import normalize_context
from pi_ai.api.anthropic_messages import stream

s = stream(claude_model, normalize_context(context), {"api_key": os.environ["ANTHROPIC_API_KEY"], "thinking_enabled": True})
```

| API id | Module | Options type |
| --- | --- | --- |
| `anthropic-messages` | `pi_ai.api.anthropic_messages` | `AnthropicOptions` |
| `openai-completions` | `pi_ai.api.openai_completions` | `OpenAICompletionsOptions` |
| `openai-responses` | `pi_ai.api.openai_responses` | `OpenAIResponsesOptions` |
| `openai-codex-responses` | `pi_ai.api.openai_codex_responses` | `OpenAICodexResponsesOptions` |
| `azure-openai-responses` | `pi_ai.api.azure_openai_responses` | `AzureOpenAIResponsesOptions` |
| `google-generative-ai` | `pi_ai.api.google_generative_ai` | `GoogleOptions` |
| `google-vertex` | `pi_ai.api.google_vertex` | `GoogleVertexOptions` |
| `mistral-conversations` | `pi_ai.api.mistral_conversations` | `MistralOptions` |
| `bedrock-converse-stream` | `pi_ai.api.bedrock_converse_stream` | `BedrockOptions` |
| `pi-messages` | `pi_ai.api.pi_messages` | `PiMessagesOptions` |

`pi_ai.api.registry` has lazy wrappers (`anthropic_messages_api()`, ...) that import a
module on first request.

### Using a custom HTTP client

Every request accepts `http_client=httpx.AsyncClient(...)` (the analog of pi's `fetch`
option): useful for custom TLS, transports, or `httpx.MockTransport` in tests. It is handed
to the vendor SDK (`http_client=` for `anthropic`/`openai`, `HttpOptions(httpx_async_client=...)`
for `google-genai`). Without it, requests share a pooled client per event loop;
`await pi_ai.aclose_http_clients()` closes it. Bedrock goes through boto3 and does not use
`http_client`.

## Faux provider for tests

```python
from pi_ai import (
    Context, TextContent, ToolResultMessage, UserMessage, create_models,
    faux_assistant_message, faux_provider, faux_text, faux_thinking, faux_tool_call,
)

faux = faux_provider(tokens_per_second=50)   # pacing is optional
models = create_models()
models.set_provider(faux.provider)
model = faux.get_model()

faux.set_responses([
    faux_assistant_message([faux_thinking("Need to inspect first."), faux_tool_call("echo", {"text": "package.json"})], stop_reason="toolUse"),
    faux_assistant_message([faux_text("Here is the summary.")]),
])
context = Context(messages=[UserMessage("Summarize package.json")])
first = await models.complete(model, context, session_id="session-1", cache_retention="short")
```

Responses are consumed in request order; when the queue is empty the request fails with
`"No more faux responses queued"`. Responses may be factories
`(context, options, state, model) -> AssistantMessage` (sync or async). Usage is estimated
at about 4 characters per token, prompt caching is simulated per `session_id`, tool call
arguments stream as `toolcall_delta` chunks, and deferred responses
(`complete_simple(..., deferred=True)` + `fetch_deferred`) are supported.

## Cross-provider hand-offs

Contexts move freely between providers. Assistant messages from the same provider/API/
model are replayed as is (including thinking and signatures); messages from other models
keep text and tool calls, drop thinking, and get their tool call ids normalized for the
target API. Errored and aborted turns are skipped, and orphaned tool calls get synthetic
error results.

```python
context = Context(messages=[UserMessage("What is 25 * 18?")])
context.messages.append(await models.complete_simple(models.get_model("anthropic", "claude-sonnet-4-5"), context, reasoning="medium"))
context.messages.append(UserMessage("Is that correct?"))
context.messages.append(await models.complete(models.get_model("openai", "gpt-5-mini"), context))
```

## System messages

`Context.system_prompt` and `Context.tools` are shorthand for a leading system message.
`normalize_context()` folds them in once; providers receive a `TranscriptContext` whose
system messages carry the prompt and tools. Later system messages change them without
rewriting history:

```python
from pi_ai import SystemMessage, ToolReference, get_current_system_prompt, get_current_tools

messages = [
    SystemMessage("You are helpful.", sections={"rules": "<rules>Be brief.</rules>"}, tools_added=[read_tool], timestamp=1),
    UserMessage("hi", timestamp=2),
    SystemMessage("", sections={"rules": "<rules>Be thorough.</rules>"}, tools_removed=[ToolReference("read")], timestamp=3),
]
get_current_system_prompt(messages)   # "You are helpful.\n\n<rules>Be thorough.</rules>"
get_current_tools(messages)           # []
```

Models with `supportsMidConvoSystemMessages` receive later system messages in place, so
the cached prompt prefix survives; other models get the replayed prompt as the leading
system message.

## Context serialization

Contexts, messages and models are data classes; `pi_ai.serialization` converts them to
and from pi's camelCase JSON (the format TypeScript pi persists):

```python
from pi_ai import Context
from pi_ai.serialization import dumps, loads, to_dict, from_dict

text = dumps(context)              # pi-compatible JSON
restored = loads(text, Context)
restored.messages.append(UserMessage("Tell me more"))
await models.complete(models.get_model("anthropic", "claude-3-5-haiku-20241022"), restored)

model_json = to_dict(model)        # {"id": ..., "baseUrl": ..., "contextWindow": ..., "type": "chat"}
```

`normalize_context()` and `Agent.prompt()` also accept messages as dicts in pi's format.

## OAuth providers

Anthropic (Claude Pro/Max), OpenAI (Sign in with ChatGPT), OpenAI Codex (legacy),
GitHub Copilot, OpenRouter (OAuth PKCE minting a key), Kimi For Coding, xAI, Meta and
Radius carry an `OAuthAuth` on `provider.auth.oauth`. `Models` drives login and persists
the credential; refresh is automatic and locked through the credential store.

```python
from pi_ai import AuthEvent, AuthPrompt, FileCredentialStore, create_models
from pi_ai.providers.anthropic import anthropic_provider


class ConsoleInteraction:
    signal = None

    async def prompt(self, prompt: AuthPrompt) -> str:
        return await asyncio.to_thread(input, prompt.message + " ")

    def notify(self, event: AuthEvent) -> None:
        if event.type == "auth_url":
            print("Open:", event.url)
        elif event.type == "device_code":
            print(f"Code {event.user_code} at {event.verification_uri}")
        elif event.message:
            print(event.message)


models = create_models(credentials=FileCredentialStore("auth.json"))
models.set_provider(anthropic_provider())
await models.login("anthropic", "oauth", ConsoleInteraction())
await models.complete(models.get_model("anthropic", "claude-sonnet-4-5"), context)
await models.logout("anthropic")
```

### CLI login

```bash
pi-ai login              # interactive provider selection
pi-ai login anthropic
pi-ai list
```

Credentials are saved to `auth.json` in the current directory.

### Vertex AI

API key (`GOOGLE_CLOUD_API_KEY` or `api_key=`), or Application Default Credentials:
`gcloud auth application-default login` locally or `GOOGLE_APPLICATION_CREDENTIALS` with
a service account file, plus `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. The
`google-genai` SDK resolves ADC through `google-auth`.

### Amazon Bedrock

Requires `pip install "pi-ai[bedrock]"` (boto3). Requests go through the boto3
`bedrock-runtime` client, so credentials, region and endpoints resolve through botocore's
standard chain (env keys, `~/.aws` profiles, SSO, ECS/EC2 roles, web identity), plus
`AWS_BEARER_TOKEN_BEDROCK` or the `credentials=` option. boto3 is synchronous; the stream
is read on a worker thread and abort closes it.

## Legacy global API

`pi_ai.compat` keeps pi's old global API (`get_model()`, `stream()`/`complete()`
dispatching on `model.api` with env-key injection, `register_api_provider()`,
`get_env_api_key()`, `register_faux_provider()`, `get_image_model()`/`generate_images()`):

```python
from pi_ai.compat import complete, get_model
```

New code should use `create_models()` and provider factories.

## Performance notes

- Streaming objects are slotted dataclasses; the event queue is a deque with direct
  hand-off to waiting consumers; wrapped streams (auth setup, lazy loading) forward events
  without an extra queue hop.
- Tool-call argument streaming uses an incremental partial-JSON lexer
  (`StreamingJsonParser`), about 60x faster than re-parsing the buffer on every delta for
  large arguments (`benchmarks/bench_partial_json.py`).
- Streamed text and thinking accumulate in amortized linear time (`append_text`): plain
  `block.text += delta` copies the whole string on every delta in CPython.
- End to end on one core (`benchmarks/bench_stream.py`): anthropic-messages handles about
  110,000 events per second, openai-completions about 70,000 (the openai SDK's SSE decoder
  dominates). The google-genai SDK validates every streamed chunk with pydantic, roughly
  100 µs per chunk; real Gemini chunks carry many tokens, so this is a few milliseconds per
  response.
- With `pi-ai[fast]`, SSE chunks decode with orjson.
- Tool-argument validators compile once per schema and are cached.
- Catalogs load lazily per provider; API implementations and OAuth flows import on first use.
- HTTP connections are pooled per event loop.

## Differences from the TypeScript package

- Names are snake_case; options are keyword arguments/dicts with snake_case keys
  (`thinking_enabled`, `reasoning_effort`, `cache_retention`). `model.compat` keeps pi's
  camelCase keys because it is configuration data shared with `models.json`.
- Tool schemas are JSON Schema dicts (`Type` builder included) or Pydantic models; TypeBox
  is not needed.
- `fetch` is replaced by `http_client` (an `httpx.AsyncClient`).
- The adapters use the Python editions of the SDKs pi uses (`anthropic`, `openai`,
  `google-genai`, and `boto3` in place of `@aws-sdk/client-bedrock-runtime`). Error messages
  keep the JS SDK wording that pi's overflow and retry detection rely on.
- Cloudflare Workers AI *bindings* (Workers runtime only) are not available.
- Telemetry options (`telemetry_context`) are accepted and ignored.

## License

MIT
