# pi_telemetry

Vendor-neutral telemetry contracts and typed schema utilities for pi packages. Python port of
[`@earendil-works/pi-telemetry`](https://github.com/earendil-works/pi/tree/main/packages/telemetry), plus an
OpenTelemetry adapter and the instrumentation of `pi_ai`, `pi_agent`, `pi_durable` and `pi_codemode`.

This package provides:

- an explicit, callback-based `TelemetryContext` / `TelemetrySpan` contract;
- a shared `NOOP_TELEMETRY_CONTEXT`;
- a reference `InMemoryTelemetryContext` implementation;
- serializable schema definitions, with runtime checks of span names and attributes;
- an explicit opt-in for recording content (`with_content_capture`);
- an adapter conformance suite (`pi_telemetry.testing`);
- an OpenTelemetry adapter (`pi_telemetry.otel`, the `otel` extra);
- no exporter, global current-span state, `contextvars`, or dependency on a telemetry backend in the core.

Applications can use the in-memory reference, the OpenTelemetry adapter, or their own adapter for Sentry, logs, or
another backend. Pi packages pass telemetry contexts explicitly and define their domain schemas separately.

## Table of Contents

- [Installation](#installation)
- [Telemetry Concepts](#telemetry-concepts)
- [Core Context API](#core-context-api)
- [Adapter Contract](#adapter-contract)
- [No-op Context](#no-op-context)
- [In-Memory Reference Adapter](#in-memory-reference-adapter)
- [Adapter Conformance](#adapter-conformance)
- [Typed Schemas](#typed-schemas)
  - [Start and Completion Attributes](#start-and-completion-attributes)
  - [Runtime Checks](#runtime-checks)
- [Schema Metadata](#schema-metadata)
- [OpenTelemetry Adapter](#opentelemetry-adapter)
  - [End-to-end Example](#end-to-end-example)
  - [GenAI Semantic Conventions](#genai-semantic-conventions)
  - [Metrics](#metrics)
- [Pi Package Integration](#pi-package-integration)
- [Span Catalog](#span-catalog)
- [Data Policy](#data-policy)
- [API Reference](#api-reference)
- [Differences from the TypeScript Package](#differences-from-the-typescript-package)
- [License](#license)

## Installation

The core ships with `pi-ai-python` and has no dependencies. The OpenTelemetry adapter needs `opentelemetry-api`; add
the SDK (or a distribution) to export:

```bash
pip install pi-ai-python
pip install "pi-ai-python[otel]" opentelemetry-sdk
```

## Telemetry Concepts

Telemetry describes what a program did while it was running. This package models that work using spans,
attributes, events, statuses, and explicit context:

| Concept | Plain-language meaning |
|---|---|
| **Span** | A timed record of one operation, such as loading an account or making an AI request. It begins before the work and ends when the work finishes. |
| **Parent and child spans** | Operations can contain smaller operations. A request span might contain a cache lookup and a database query. Together they form a tree showing where time was spent. |
| **Attribute** | A named fact attached to a span, such as `provider: "openai"`, `cache.hit: True`, or `item_count: 12`. Attributes describe the operation and its result. |
| **Event** | A named occurrence at a point during a span, such as `retry.scheduled` or `cache.lookup`. Events have no duration and may carry their own attributes. |
| **Status** | The operation's outcome: `ok` or `error`. An error status may include an error name and message. |
| **Context** | A handle identifying where new work belongs in the span tree. Starting a span from a context makes it a child of that context. |

For example, loading an account could produce this telemetry:

```text
example.account.load                         span
├─ attributes: account.id=123, found=true   facts about the span
├─ event: example.cache.lookup              occurrence during the span
│  └─ attribute: cache.hit=false            fact about the event
└─ status: ok                               final outcome
```

A span is diagnostic data, not business state. Recording it must not change whether the account load runs,
succeeds, fails, or is persisted. An adapter translates these generic concepts into the corresponding concepts used
by OpenTelemetry, Sentry, logs, or another backend.

## Core Context API

A `TelemetryContext` starts a span around a callback. The callback receives a `TelemetrySpan`, which is also the
explicit parent context for child spans. `start_span()` returns an awaitable of the callback's result:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, TelemetryContext, TelemetrySpan

ACCOUNTS = {"123": {"name": "Ada"}}


async def read_account(account_id: str) -> dict | None:
    return ACCOUNTS.get(account_id)


async def load_account(account_id: str, telemetry_context: TelemetryContext = NOOP_TELEMETRY_CONTEXT) -> dict | None:
    async def load(span: TelemetrySpan) -> dict | None:
        account = await read_account(account_id)
        span.set_attributes({"example.account.found": account is not None})
        return account

    return await telemetry_context.start_span(
        {"name": "example.account.load", "attributes": {"example.account.id": account_id}}, load
    )


print(await load_account("123"))
```

Pass the callback span to lower-level work to create explicit nesting:

```python
async def perform_work() -> str:
    return "done"


async def parent(parent_span: TelemetrySpan) -> str:
    async def child(child_span: TelemetrySpan) -> str:
        child_span.add_event("example.cache.lookup", {"example.cache.hit": True})
        return await perform_work()

    return await parent_span.start_span({"name": "example.child"}, child)


print(await NOOP_TELEMETRY_CONTEXT.start_span({"name": "example.parent"}, parent))
```

There is no public `end()` method. `start_span()` owns settlement and keeps the span open until the callback's value
or awaitable settles. For an expected failure represented by a normal return value, set the status explicitly:

```python
async def save() -> dict:
    return {"ok": False, "reason": "conflict"}


async def saving(span: TelemetrySpan) -> dict:
    result = await save()
    if not result["ok"]:
        span.set_status({"status": "error", "error": {"name": "SaveError", "message": result["reason"]}})
    return result


print(await NOOP_TELEMETRY_CONTEXT.start_span({"name": "example.save"}, saving))
```

The callback runs synchronously at the call, exactly once. A plain function's span settles at once; an `async def`
callback returns a coroutine, which runs, and keeps its span open, when the result of `start_span()` is awaited. As
with any coroutine, await it (or wrap it in a task).

## Adapter Contract

An adapter implements `TelemetryContext` and bridges the generic API to its backend. It must:

- create a child span and invoke the callback synchronously, exactly once;
- preserve the callback's returned value and raised exception (the same object), returning an awaitable that raises
  it again after a synchronous raise;
- keep the native span open until a returned awaitable settles;
- treat normal completion as `ok` and raises as errors unless an explicit status was set;
- make repeated `set_status()` calls last-write-wins;
- merge `set_attributes()` calls, with later non-`None` values replacing earlier values and `None` ignored;
- make recording methods synchronous, passive, and non-raising;
- ignore calls made after settlement;
- ignore a failed recording call atomically (an unreadable payload or a value that is not a primitive or array),
  suppress backend failures, and still execute the business callback exactly once.

Adapters may activate backend-native ambient context internally for automatic instrumentation (the OpenTelemetry
adapter does), but pi code always propagates the parent through `TelemetryContext` arguments. Exporter buffering,
flushing, sampling, backend IDs, and backend-specific context objects belong to the adapter. Use the
[adapter conformance suite](#adapter-conformance) to check these observable semantics.

## No-op Context

Use `NOOP_TELEMETRY_CONTEXT` when telemetry is optional:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, is_telemetry_enabled


async def run_operation(_span: TelemetrySpan) -> int:
    return 7


result = await NOOP_TELEMETRY_CONTEXT.start_span({"name": "example.operation"}, run_operation)
print(result, is_telemetry_enabled(NOOP_TELEMETRY_CONTEXT), is_telemetry_enabled(None))
```

The no-op context:

- invokes callbacks synchronously;
- preserves returned values and raised exceptions, and turns a synchronous raise into an awaitable that raises it;
- uses one shared frozen inert span, including for nested spans;
- does not inspect or retain names, attributes, events, or statuses.

`is_telemetry_enabled(context)` is `False` for `None` and the no-op context. Pi's instrumentation checks it first, so
the disabled path builds no attributes and starts no spans.

## In-Memory Reference Adapter

`InMemoryTelemetryContext` is the backend-neutral reference implementation. It is useful for tests, local
diagnostics, and applications that intentionally want process-local capture without an exporter:

```python
from pi_telemetry import InMemoryTelemetryContext

telemetry = InMemoryTelemetryContext()


async def operation(span: TelemetrySpan) -> None:
    span.add_event("example.started")
    span.set_attributes({"output_count": 3})
    await span.start_span({"name": "example.child"}, lambda child: None)


await telemetry.start_span({"name": "example.operation", "attributes": {"input": "demo"}}, operation)
for span in telemetry.get_spans():
    print(span.id, span.parent_id, span.name, span.attributes, [event.name for event in span.events], span.end_sequence)
```

`get_spans()` returns detached `RecordedTelemetrySpan` snapshots in span-start order. Each contains a deterministic
numeric `id`, `parent_id`, merged `attributes`, ordered `events`, final `status`, `settled` flag, and deterministic
`end_sequence`. It records no timestamps. Arrays are recorded as lists.

The adapter is safe to use as an ordinary `TelemetryContext`, but storage is unbounded and process-local. Create a
fresh instance (or call `clear()`) to isolate tests or recording scopes, and do not capture sensitive attributes
unless the caller's data policy allows them.

## Adapter Conformance

`pi_telemetry.testing` exports a runner-independent conformance suite modeled as grouped cases. A fixture supplies a
fresh context and converts its backend's finished spans into normalized `RecordedTelemetrySpan` snapshots. Register
the cases with pytest, or run them as plain coroutines:

```python
import pytest

from pi_telemetry.testing import create_telemetry_adapter_conformance


class Fixture:
    def __init__(self) -> None:
        self.adapter = InMemoryTelemetryContext()  # your adapter here
        self.context = self.adapter

    async def get_spans(self):
        return self.adapter.get_spans()  # flush and normalize your backend's spans here

    async def aclose(self) -> None:
        pass


async def make_fixture() -> Fixture:
    return Fixture()


CASES = create_telemetry_adapter_conformance(make_fixture)


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.group}: {case.name}")
async def test_conformance(case) -> None:
    await case.run()


for case in CASES:
    await case.run()
print(len(CASES), "cases passed:", sorted({case.group for case in CASES}))
```

The suite checks synchronous single admission, result and exception identity, automatic and explicit status,
attribute merging, event ordering, inert post-settlement calls, nested and concurrent parentage, and suppression of
unreadable telemetry payload failures. `get_spans()` may flush an asynchronous exporter before returning. The cases
use plain `assert` statements and import no test framework. `packages/telemetry/tests/test_otel.py` runs the suite
against the OpenTelemetry adapter with the SDK's span processors.

## Typed Schemas

The low-level span API intentionally accepts open names and attribute bags so adapters remain generic. Domain
packages define closed, serializable schemas. A schema is plain JSON data with upstream's keys, so the same document
describes the vocabulary in TypeScript and Python; the `TypedDict`s (`TelemetrySchemaDefinition`,
`TelemetrySpanDefinition`, ...) let a type checker verify a schema literal's structure.

```python
from pi_telemetry import SchemaTelemetrySpan, TypedSpanStarter, create_typed_span_starter, define_telemetry_schema

EXAMPLE_TELEMETRY_SCHEMA = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            "example.read": {
                "description": "Read one resource",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "example.resource": {
                        "type": "string",
                        "required": True,
                        "values": ["account", "project"],
                        "description": "Resource kind",
                    },
                },
                "endAttributes": {
                    "example.item_count": {"type": "number", "description": "Number of returned items"},
                },
                "events": {
                    "example.cache": {
                        "description": "Cache lookup result",
                        "attributes": {
                            "example.cache.hit": {
                                "type": "boolean",
                                "required": True,
                                "description": "Whether the cache contained the resource",
                            },
                        },
                    },
                },
                "status": {"default": "ok", "errorWhen": "The read raises or returns an error result"},
            },
        },
    }
)

telemetry = InMemoryTelemetryContext()
start_span = create_typed_span_starter(telemetry, [EXAMPLE_TELEMETRY_SCHEMA])
```

The starter is called with a span name, its start attributes, and a callback. The callback receives the span, viewed
through its schema (`SchemaTelemetrySpan`), and a child starter over the same schemas, already bound to that span:

```python
async def read_accounts() -> list[str]:
    return ["ada", "grace"]


async def read_projects() -> list[str]:
    return ["pi"]


async def read(span: SchemaTelemetrySpan, start_child_span: TypedSpanStarter) -> list[str]:
    span.add_event("example.cache", {"example.cache.hit": True})
    accounts = await read_accounts()
    span.set_attributes({"example.item_count": len(accounts)})

    async def read_child(child_span: SchemaTelemetrySpan, _start: TypedSpanStarter) -> None:
        projects = await read_projects()
        child_span.set_attributes({"example.item_count": len(projects)})

    await start_child_span("example.read", {"example.resource": "project"}, read_child)
    return accounts


print(await start_span("example.read", {"example.resource": "account"}, read))
print([(span.name, span.parent_id, span.attributes) for span in telemetry.get_spans()])
```

### Start and Completion Attributes

`startAttributes` and `endAttributes` describe when an attribute is normally known, not separate runtime storage:

| Schema field | How values are recorded | Requiredness |
|---|---|---|
| `startAttributes` | Passed in the typed starter's `attributes` argument when the span is created | Each definition explicitly sets `required: True` or `False` |
| `endAttributes` | Added later through the schema-scoped span's `set_attributes()` method | Always optional |

Both sets become ordinary attributes on the same backend span. There is no separate end-attribute payload or end
callback. In the preceding example, `example.resource` is known when `example.read` starts, while
`example.item_count` is known only after `read_accounts()` returns.

"End" means completion enrichment: an end attribute may be set at any point while the callback is active, and it
may be omitted when unavailable. Calling `set_attributes()` zero times is valid. This matters for early failures,
cancellation, and provider-specific data that may not exist on every path.

Repeated `set_attributes()` calls merge into the same attribute bag. A later value replaces an earlier value for the
same key, while `None` is ignored. The schema-scoped method accepts only the current span's declared end attributes.

Attributes do not end the span. Returning, raising, or settling the callback's awaitable controls settlement;
`start_span()` performs the actual end operation. Adapter calls made after settlement are inert.

A starter can compose multiple independently versioned schemas; duplicate span names across them are rejected when
the starter is created:

```python
from pi_agent.telemetry import AGENT_TELEMETRY_SCHEMAS
from pi_ai.telemetry import AI_TELEMETRY_SCHEMAS

start_agent_span = create_typed_span_starter(telemetry, [*AGENT_TELEMETRY_SCHEMAS, *AI_TELEMETRY_SCHEMAS])
print(start_agent_span.span_names)
```

`define_telemetry_schema()` is a typed identity function. It returns ordinary JSON-serializable data and performs no
validation or parent-rule enforcement.

### Runtime Checks

TypeScript rejects unknown names, missing required attributes, unknown keys, invalid closed-set values, undeclared
events, and attributes on empty schemas at compile time. Python cannot, so typed starters check them when they are
called. `on_invalid` picks what a violation does:

| `on_invalid` | A violation |
|---|---|
| `"raise"` (default) | raises `TelemetrySchemaError` (a `TypeError`) before anything is recorded: the analog of a compile error |
| `"warn"` | drops the offending attributes or event, emits a `TelemetrySchemaWarning`, and keeps going |
| `"ignore"` | drops them silently |

```python
from pi_telemetry import TelemetrySchemaError

try:
    start_span("example.read", {"example.resource": "invoice"}, read)
except TelemetrySchemaError as error:
    print(error)
```

Pi's own instrumentation uses `"warn"`, so a schema mistake can never change the behavior of the code it observes;
pi's test suite turns `TelemetrySchemaWarning` into an error (`filterwarnings`). To check recorded spans against
schemas in your own tests, including parent rules, use `pi_telemetry.testing.assert_spans_match_schemas(spans,
schemas)`. For static typing of a vocabulary, owning packages wrap the starter in functions with typed parameters.

## Schema Metadata

Supported attribute types are:

- `string`, `number`, and `boolean` (`str`, `int` or `float`, `bool`; a `bool` is not a number);
- `string[]`, `number[]`, and `boolean[]` (lists or tuples, recorded as lists).

Attribute definitions support:

- `values`: a closed set for scalar values;
- `elementValues`: a closed set for array elements;
- `examples`: documentation examples;
- `sensitive`: marks data requiring special handling;
- `cardinality`: records expected `low` or `high` cardinality.

Start and event attributes declare `required`. End attributes do not; see
[Start and Completion Attributes](#start-and-completion-attributes).

Parent metadata is descriptive schema data:

- `{"kind": "any"}`: root or any caller span;
- `{"kind": "root_or_external"}`: root or a caller-owned span outside the schema;
- `{"kind": "spans", "spans": [...]}`: only the listed schema spans.

Adapters do not need to understand schema objects. Instrumentation helpers and tests use them to keep emitted names
and attributes consistent.

## OpenTelemetry Adapter

`pi_telemetry.otel.OpenTelemetryContext` is a root `TelemetryContext` backed by the OpenTelemetry API:

```python
from opentelemetry.sdk.trace import TracerProvider

from pi_telemetry.otel import OpenTelemetryContext

telemetry = OpenTelemetryContext(tracer_provider=TracerProvider())
```

- A pi span becomes an OTel span whose parent is the pi parent span. A root pi span's parent is `parent` (an OTel
  `Context`) or, by default, the OTel context current when the span starts, so pi spans join an application trace.
- Attributes keep their `pi.*` names; events map to span events.
- An explicit `error` status maps to `StatusCode.ERROR` with the error name as `error.type` and the message (if any)
  as the description; an explicit `ok` to `StatusCode.OK`; an automatic `ok` leaves the status unset.
- A raise sets `StatusCode.ERROR` and `error.type` (the exception class) without a description: exception messages
  are free-form error details. `record_exception=True` also records them as OTel exception events.
- While a callback runs, its OTel span is the current OTel span (`activate=True`), so automatic instrumentation, for
  example of `httpx`, nests under it.
- `pi.ai.request` spans are `CLIENT` spans with [GenAI semantic-convention attributes](#genai-semantic-conventions)
  and feed the [metrics](#metrics) (`record_metrics=True`; `meter_provider` defaults to the global one).

Importing `pi_telemetry` never imports OpenTelemetry; only `pi_telemetry.otel` does.

### End-to-end Example

An agent run exported to the OpenTelemetry SDK's in-memory span exporter and metric reader. The faux provider
stands in for a model; with a real provider nothing changes but the models collection.

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pi_agent import Agent, AgentInitialState, agent_tool
from pi_ai import create_models, faux_assistant_message, faux_provider, faux_tool_call

exporter = InMemorySpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
# For a quick look at the raw spans: tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter())).
metric_reader = InMemoryMetricReader()
telemetry = OpenTelemetryContext(
    tracer_provider=tracer_provider, meter_provider=MeterProvider(metric_readers=[metric_reader])
)

faux = faux_provider()
models = create_models()
models.set_provider(faux.provider)
faux.set_responses(
    [
        faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3})], stop_reason="toolUse"),
        faux_assistant_message("2 + 3 = 5"),
    ]
)


@agent_tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


agent = Agent(
    initial_state=AgentInitialState(model=faux.get_model(), tools=[add]),
    stream_fn=models,
    telemetry_context=telemetry,
)
await agent.prompt("What is 2 + 3?")

spans = sorted(exporter.get_finished_spans(), key=lambda span: span.start_time)
names = {span.context.span_id: span.name for span in spans}
for span in spans:
    print(span.name, "<-", names.get(span.parent.span_id) if span.parent else None)
request = spans[1]
print({key: value for key, value in request.attributes.items() if key.startswith("gen_ai.") and "time" not in key})
run = spans[0]
print(run.attributes["pi.agent.turns"], run.attributes["pi.agent.tool_calls"], run.attributes["pi.agent.end_reason"])
data = metric_reader.get_metrics_data()
print(sorted(metric.name for resource in data.resource_metrics for scope in resource.scope_metrics for metric in scope.metrics))
```

The same spans and metrics go to any OpenTelemetry exporter (OTLP, console, a vendor's) configured on the providers.

### GenAI Semantic Conventions

At settlement a `pi.ai.request` span also gets these OpenTelemetry GenAI attributes (the `pi.ai.*` attributes stay):

| GenAI attribute | From |
|---|---|
| `gen_ai.operation.name` | `"chat"` |
| `gen_ai.provider.name`, `gen_ai.system` | `pi.ai.provider` (`gen_ai.system` is the older convention many backends still read) |
| `gen_ai.request.model` | `pi.ai.model` |
| `gen_ai.response.model` | `pi.ai.response.model` |
| `gen_ai.conversation.id` | `pi.ai.session_id` |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature` | `pi.ai.request.max_tokens`, `pi.ai.request.temperature` |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | `pi.ai.usage.input_tokens`, `pi.ai.usage.output_tokens` |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens` | `pi.ai.usage.cache_read_tokens`, `pi.ai.usage.cache_write_tokens` |
| `gen_ai.usage.reasoning.output_tokens` | `pi.ai.usage.reasoning_tokens` |
| `gen_ai.response.finish_reasons` | `[pi.ai.stop_reason]`: `stop`, `length`, `tool_call` (toolUse), `error`, `cancelled` (aborted), `deferred` |
| `gen_ai.response.time_to_first_chunk` | `pi.ai.time_to_first_token_ms` / 1000 |
| `error.type` | the status's error name (`pi.ai.error_kind`) |

The span keeps pi's name (`pi.ai.request`) rather than GenAI's `"{operation} {model}"`.

### Metrics

Recorded once per `pi.ai.request` span. Labels are low-cardinality only: `gen_ai.operation.name`,
`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `pi.ai.api`, plus `error.type` on failed
requests and the token type where noted. Session IDs, task IDs and other per-request values are never labels.

| Instrument | Kind, unit | Value |
|---|---|---|
| `gen_ai.client.operation.duration` | histogram, s | request duration (`error.type` on failures) |
| `gen_ai.client.operation.time_to_first_chunk` | histogram, s | time to the first streamed delta |
| `gen_ai.client.token.usage` | histogram, {token} | input and output tokens (`gen_ai.token.type`) |
| `pi.ai.requests` | counter, {request} | requests (`error.type` on failures) |
| `pi.ai.errors` | counter, {request} | requests that ended with an error or abort, by `error.type` |
| `pi.ai.tokens` | counter, {token} | tokens by `pi.ai.token.type`: `input`, `output`, `cache_read`, `cache_write` |
| `pi.ai.cost` | counter, USD | estimated cost |

## Pi Package Integration

Package ownership is split as upstream intends: `pi_telemetry` owns the contract; each emitting package owns its
schema.

| Package | Schema (`<module>.<NAME>`) | Spans | Where the context comes from |
|---|---|---|---|
| `pi_ai` | `pi_ai.telemetry.AI_TELEMETRY_SCHEMA` | `pi.ai.request` | the `telemetry_context` request option of `Models.stream`/`stream_simple`/`complete`/`complete_simple`/`stream_deferred`/`fetch_deferred` (and `pi_ai.compat`) |
| `pi_agent` | `pi_agent.telemetry.AGENT_TELEMETRY_SCHEMA` | `pi.agent.run` | `Agent(telemetry_context=...)` or `AgentLoopConfig.telemetry_context` |
| `pi_durable` | `pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`, `SESSION_TELEMETRY_SCHEMA` | `pi.harness.generation`, `pi.harness.tool`, `pi.harness.task`, `pi.session.commit` | `HarnessOptions["telemetryContext"]`, `create_session(..., telemetry_context=...)` |
| `pi_codemode` | `pi_codemode.telemetry.CODEMODE_TELEMETRY_SCHEMA` | `pi.codemode.run` | `CodemodeSandboxBase.execute(..., telemetry_context=...)`, `create_codemode_tool(..., telemetry_context=...)`; in pi_durable, the code mode tool's `pi.harness.tool` span |

How the context flows, always as an explicit argument:

```text
app context
├─ pi.agent.run                         Agent(telemetry_context=app)
│  └─ pi.ai.request                     loop passes the run span as stream option telemetry_context
│     └─ (provider spans, pi.ai.retry)  Models passes the request span to the provider and the HTTP retry path
└─ Harness (HarnessOptions["telemetryContext"] = app)
   ├─ pi.session.commit                 every Session commit
   ├─ pi.harness.generation             each generation phase; runtime.telemetry_context is the phase span
   │  └─ pi.ai.request                  request/poll phases pass it as telemetry_context
   ├─ pi.harness.task                   other tasks' phases (compaction: with its pi.ai.request)
   └─ pi.harness.tool                   each tool call; api.telemetry_context is the tool span
      ├─ pi.codemode.run                the code mode tool's script run
      └─ pi.harness.tool (nested)       execute_tool() calls, while the caller's span is open
```

`pi_ai` passes `telemetry_context` through `build_base_options()` to providers exactly like upstream. Under `Models`,
providers receive the `pi.ai.request` span itself (an `AiRequestSpan`), so provider-level spans nest under the request
and `retry_provider_request()` records `pi.ai.retry` events on it. Without a `telemetry_context`, or with
`NOOP_TELEMETRY_CONTEXT`, nothing is built: `Models` returns the same stream it always did, and providers see the
options unchanged.

Telemetry contexts are process-local. pi_durable keeps the Harness's context in memory only: it is never written to
records, documents, checkpoints, or journals, and a task resumed after a restart runs under the context of the
process that reopened the Harness.

## Span Catalog

Every span pi emits, generated from the schemas. "start" attributes are set when the span starts, "end" attributes
when known (always optional). Every attribute is a primitive or an array; none carries content.

<!-- span-catalog:start -->
#### `pi.ai.request`

One model request through pi_ai.Models, from dispatch to the terminal message. Owner: `pi_ai` (`pi_ai.telemetry.AI_TELEMETRY_SCHEMA`, version 1).
Parent: any (root or a caller's span). Error status: the terminal message's stop reason is error or aborted.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.ai.provider` | start, required | string | Provider ID |
| `pi.ai.api` | start, required | string | API implementation |
| `pi.ai.model` | start, required | string | Requested model ID |
| `pi.ai.operation` | start, required | string: `stream` \| `stream_simple` \| `fetch_deferred` | Models entry point: full options, provider-neutral options, or a deferred fetch |
| `pi.ai.streaming` | start, required | boolean | Whether the caller consumes events (stream*) or only the result (complete*, fetch_deferred) |
| `pi.ai.reasoning` | start | string: `off` \| `minimal` \| `low` \| `medium` \| `high` \| `xhigh` \| `max` | Requested thinking level (stream_simple) |
| `pi.ai.tool_count` | start, required | number | Tools declared to the model |
| `pi.ai.message_count` | start, required | number | Transcript messages sent |
| `pi.ai.session_id` | start | string | The request's session_id option, when set |
| `pi.ai.request.max_tokens` | start | number | Requested output token limit |
| `pi.ai.request.temperature` | start | number | Requested temperature |
| `pi.ai.usage.input_tokens` | end | number | Input tokens |
| `pi.ai.usage.output_tokens` | end | number | Output tokens |
| `pi.ai.usage.cache_read_tokens` | end | number | Input tokens read from cache |
| `pi.ai.usage.cache_write_tokens` | end | number | Input tokens written to cache |
| `pi.ai.usage.reasoning_tokens` | end | number | Output tokens spent on reasoning, when the provider reports them |
| `pi.ai.cost` | end | number | Estimated total cost in USD |
| `pi.ai.stop_reason` | end | string: `stop` \| `length` \| `toolUse` \| `error` \| `aborted` \| `deferred` \| `pending` | Stop reason of the terminal message |
| `pi.ai.error_kind` | end | string: `aborted` \| `cancelled` \| `context_overflow` \| `rate_limited` \| `http_error` \| `provider_error` | Why the request failed; never the provider's error text |
| `pi.ai.time_to_first_token_ms` | end | number | Milliseconds from the start of the request to its first text, thinking or tool-call delta |
| `pi.ai.duration_ms` | end | number | Milliseconds until the terminal message |
| `pi.ai.http.status` | end | number | HTTP status of the last provider response, when the provider reports one |
| `pi.ai.response.model` | end | string | Model ID the provider reports in its response |
| `pi.ai.retry_count` | end | number | HTTP retries within the request |

| Event | Attributes | Description |
| --- | --- | --- |
| `pi.ai.retry` | `pi.ai.retry.attempt`, `pi.ai.retry.reason`, `pi.ai.retry.delay_ms`, `pi.ai.retry.http_status` (optional) | A retryable HTTP failure; the request is sent again after the delay |
| `pi.ai.context_overflow` | `pi.ai.context_window` (optional) | The request exceeded the model's context window |
| `pi.ai.abort` | - | The request was aborted through its signal |

Declares `pi.content` (recorded only under content capture).

#### `pi.agent.run`

One run of the agent loop, from its prompt (or continuation) to agent_end. Owner: `pi_agent` (`pi_agent.telemetry.AGENT_TELEMETRY_SCHEMA`, version 1).
Parent: any (root or a caller's span). Error status: the run ends with an error or abort, or the loop raises.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.agent.provider` | start, required | string | Provider of the run's initial model |
| `pi.agent.model` | start, required | string | The run's initial model ID |
| `pi.agent.tool_count` | start, required | number | Tools the agent can execute |
| `pi.agent.mode` | start, required | string: `prompt` \| `continue` | New prompt messages, or a continuation of the existing context |
| `pi.agent.reasoning` | start | string | Requested thinking level |
| `pi.agent.turns` | end | number | Model responses in the run |
| `pi.agent.tool_calls` | end | number | Tool results the run produced |
| `pi.agent.usage.input_tokens` | end | number | Input tokens summed over the run's responses |
| `pi.agent.usage.output_tokens` | end | number | Output tokens summed over the run's responses |
| `pi.agent.cost` | end | number | Estimated cost of the run's responses in USD |
| `pi.agent.end_reason` | end | string: `completed` \| `ended` \| `error` \| `aborted` \| `failed` | Why the run ended |

| Event | Attributes | Description |
| --- | --- | --- |
| `pi.agent.steering` | `pi.agent.message_count` | Steering messages were injected into the run |
| `pi.agent.follow_up` | `pi.agent.message_count` | Follow-up messages continued the run after it would have stopped |

#### `pi.harness.generation`

One phase handler run of a generation task (pi.generation). Owner: `pi_durable` (`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`, version 1).
Parent: root or a caller-owned span. Error status: the phase handler raises.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | Task ID |
| `pi.harness.conversation_id` | start, required | number | Conversation ID of the task |
| `pi.harness.phase` | start, required | string: `prepare` \| `request` \| `retry` \| `poll` \| `tools` \| `abort` | Phase handler (abort: the abort handler) |
| `pi.harness.restarts` | start | number | Consecutive restarts that cut the task off while it ran without changing its state |
| `pi.harness.attempt` | start | number | Generation attempt of the phase |
| `pi.harness.model.provider` | start | string | Provider of the phase's model, once chosen |
| `pi.harness.model.id` | start | string | Model ID of the phase's model, once chosen |
| `pi.harness.phase_outcome` | end | string: `returned` \| `raised` \| `aborted` | Whether the handler returned (durable progress), raised, or ended by an abort |
| `pi.harness.duration_ms` | end | number | Milliseconds the handler ran |
| `pi.harness.generation.stop_reason` | end | string: `stop` \| `length` \| `toolUse` \| `error` \| `aborted` \| `deferred` \| `pending` | Stop reason of the response a request or poll phase received |
| `pi.harness.generation.tool_calls` | end | number | Tool calls in that response |

#### `pi.harness.tool`

One execution of a tool call: the call phase, a recovery, or the abort handler. Owner: `pi_durable` (`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`, version 1).
Parent: any (root or a caller's span). Error status: the call does not succeed.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | Task ID |
| `pi.harness.conversation_id` | start, required | number | Conversation ID of the task |
| `pi.harness.phase` | start, required | string: `call` \| `execute` \| `abort` | call: first run; execute: recovery after a restart; abort: the abort handler |
| `pi.harness.tool.name` | start, required | string | Called tool |
| `pi.harness.tool.extension` | start | string | Extension that provides the tool |
| `pi.harness.tool.replay` | start | string: `safe` \| `unsafe` | The tool's replay policy, when it resolved |
| `pi.harness.tool.nested` | start, required | boolean | Whether a running tool made the call (execute_tool()) instead of the model |
| `pi.harness.restarts` | start | number | Consecutive restarts that cut the task off while it ran without changing its state |
| `pi.harness.tool.outcome` | end | string: `success` \| `error` \| `blocked` \| `interrupted` | How the call ended |
| `pi.harness.tool.error_kind` | end | string: `tool_unavailable` \| `invalid_arguments` \| `blocked` \| `tool_error` \| `invalid_structured_output` \| `interrupted` \| `aborted` \| `abandoned` | Why the call failed; never the error text |
| `pi.harness.tool.duration_ms` | end | number | Milliseconds the tool's execute() ran |
| `pi.harness.tool.output_bytes` | end | number | UTF-8 bytes of the settled text output |
| `pi.harness.tool.truncated` | end | boolean | Whether the output was cut to the tool's output limits |

| Event | Attributes | Description |
| --- | --- | --- |
| `pi.harness.tool.blocked` | - | A before_tool hook blocked the call (the reason is not recorded) |
| `pi.harness.tool.validation_failed` | `pi.harness.tool.validation_stage` | The call's arguments failed validation |

Declares `pi.content` (recorded only under content capture).

#### `pi.harness.task`

One phase handler run of a task other than generations and tool calls. Owner: `pi_durable` (`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`, version 1).
Parent: root or a caller-owned span. Error status: the phase handler raises.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | Task ID |
| `pi.harness.conversation_id` | start, required | number | Conversation ID of the task |
| `pi.harness.phase` | start, required | string | Phase handler (abort: the abort handler) |
| `pi.harness.restarts` | start | number | Consecutive restarts that cut the task off while it ran without changing its state |
| `pi.harness.task.kind` | start, required | string | Task definition name |
| `pi.harness.task.extension` | start | string | Extension that installed the definition; absent for a built-in |
| `pi.harness.phase_outcome` | end | string: `returned` \| `raised` \| `aborted` | Whether the handler returned (durable progress), raised, or ended by an abort |
| `pi.harness.duration_ms` | end | number | Milliseconds the handler ran |

#### `pi.session.commit`

One Session commit: the change callback, settlement, and the Storage commit. Owner: `pi_durable` (`pi_durable.telemetry.SESSION_TELEMETRY_SCHEMA`, version 1).
Parent: any (root or a caller's span). Error status: the change callback, settlement, or storage raises.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.session.storage` | start, required | string | Storage backend |
| `pi.session.write_count` | end | number | Storage writes the commit made |
| `pi.session.duration_ms` | end | number | Milliseconds on the mutation line |

#### `pi.codemode.run`

One execution of a model-written script. Owner: `pi_codemode` (`pi_codemode.telemetry.CODEMODE_TELEMETRY_SCHEMA`, version 1).
Parent: any (root or a caller's span). Error status: the script fails, times out, is aborted, or diverges.

| Attribute | When | Type | Description |
| --- | --- | --- | --- |
| `pi.codemode.line_count` | start, required | number | Lines of the script |
| `pi.codemode.timeout_ms` | start | number | The run's deadline; absent when it has none |
| `pi.codemode.mode` | start, required | string: `execute` \| `deterministic` | execute(): the sandbox calls tools; deterministic: start(), the host answers |
| `pi.codemode.tool_count` | start, required | number | Tools the script can call |
| `pi.codemode.outcome` | end | string: `success` \| `error` | Whether the script succeeded |
| `pi.codemode.error_kind` | end | string: `script` \| `timeout` \| `aborted` \| `sandbox` \| `different_path` \| `version_mismatch` \| `deadline_passed` \| `host` | Why the run failed; never the error text |
| `pi.codemode.nested_call_count` | end | number | Tool calls the script made |
| `pi.codemode.duration_ms` | end | number | Milliseconds the run took |
| `pi.codemode.replayed` | end | boolean | Whether the run replayed a journal after a restart |
| `pi.codemode.replayed_call_count` | end | number | Journaled results the replay delivered |

| Event | Attributes | Description |
| --- | --- | --- |
| `pi.codemode.different_path` | - | A replay took a different path than the run it resumes |
| `pi.codemode.timeout` | `pi.codemode.timeout_ms` (optional) | The run's deadline expired |

Declares `pi.content` (recorded only under content capture).
<!-- span-catalog:end -->

## Data Policy

Telemetry is process-local diagnostics, not durable application state. Do not persist a `TelemetryContext`,
`TelemetrySpan`, or backend-native trace object in records, messages, snapshots, or deferred handles.

Attribute values are limited to primitive scalars and arrays. By default pi's instrumentation records no prompts,
completions, thinking, tool arguments or output, file contents, provider payloads, headers, credentials, or free-form
error details: only names, counts, sizes, durations, token usage, cost, closed-set kinds, and IDs. Error statuses
carry a kind (`rate_limited`, `tool_error`, an exception class name), never a message. Pi's test suites check this
with sentinel strings.

Content is recorded only under an explicit opt-in. `with_content_capture(context, max_length=..., redact=...)` wraps
a context; instrumentation running under it, and under every span started from it, records content as `pi.content`
events: the kind (`system`, `user`, `assistant`, `thinking`, `tool_call`, `tool_arguments`, `tool_output`, `code`,
`output`, `error`, ...), the text after `redact(kind, text)` (return `None` to drop it), its length, and whether it
was cut at `max_length` characters (default 4096; `-1` keeps everything):

```python
from pi_ai import Context, UserMessage
from pi_telemetry import with_content_capture


def redact(kind: str, text: str) -> str | None:
    return None if kind == "system" else text.replace("sk-secret", "[key]")


captured = InMemoryTelemetryContext()
faux.set_responses([faux_assistant_message("The key is sk-secret.")])
await models.complete(
    faux.get_model(),
    Context(messages=[UserMessage("What is the key?")]),
    telemetry_context=with_content_capture(captured, max_length=12, redact=redact),
)
[request] = captured.get_spans()
for event in request.events:
    print(event.attributes["pi.content.kind"], repr(event.attributes["pi.content.text"]), event.attributes["pi.content.truncated"])
```

The opt-in travels with the context, explicitly, like any parent; nothing global changes. Pi code records content
with `record_content(span, kind, text)`, a no-op without the opt-in.

The package does not use `contextvars` or another ambient context API. Backend adapters remain responsible for their
own runtime behavior.

## API Reference

### Core types and values (`pi_telemetry`)

| Export | Purpose |
|---|---|
| `TelemetryContext` | Starts callback-managed child spans (`start_span(options, callback)`) |
| `TelemetrySpan` | Records attributes, events, and status; also acts as a child context |
| `SpanOptions` | Span `name` and optional start `attributes` |
| `SpanAttributes` / `AttributeValue` / `AttributeScalar` | Open adapter-level attribute bag and supported values |
| `SpanStatus` (`OkSpanStatus`, `ErrorSpanStatus`, `SpanErrorInfo`) | Explicit `ok` or `error` status |
| `NOOP_TELEMETRY_CONTEXT` | Shared passive context for disabled telemetry |
| `is_telemetry_enabled()` | `False` for `None` and the no-op context |
| `InMemoryTelemetryContext` | Reference adapter with deterministic process-local recording (`get_spans()`, `clear()`) |
| `RecordedTelemetrySpan` | Normalized captured span snapshot |
| `RecordedTelemetryEvent` | Normalized captured event snapshot |

### Schema definitions and checks (`pi_telemetry`)

| Export | Purpose |
|---|---|
| `define_telemetry_schema()` | Typed identity helper for serializable schema data |
| `create_typed_span_starter()` | Binds a parent context to one or more schema vocabularies (`on_invalid`) |
| `bind_typed_span_starter()` | Rebinds a starter to another parent without re-indexing its schemas |
| `TypedSpanStarter` | The starter: `starter(name, attributes, callback)`; `span_names`, `context` |
| `SchemaTelemetrySpan` | Span view restricted to one schema span (`name`, `definition`, `span`) |
| `TelemetrySchemaError` / `TelemetrySchemaWarning` | A schema violation (raised, or warned with `on_invalid="warn"`) |
| `TelemetrySchemaDefinition` | Top-level schema shape |
| `TelemetrySpanDefinition` | Span metadata, parents, attributes, events, and status rule |
| `TelemetryAttributeType` | Supported scalar and array type names |
| `TelemetryAttributeDefinition` | Attribute type, allowed values, examples, and metadata |
| `TelemetryStartAttributeDefinition` / `TelemetryEventAttributeDefinition` | Definitions with requiredness |
| `TelemetryEventDefinition` | Event description and attribute definitions |
| `TelemetryParentDefinition` (`TelemetryParentAny`, `TelemetryParentRootOrExternal`, `TelemetryParentSpans`) | Parent rules |
| `TelemetrySpanStatusRule` | `{"default": "ok", "errorWhen": ...}` |
| `schema_span_names()`, `check_attribute_value()`, `check_start_attributes()`, `check_end_attributes()`, `check_event()` | The runtime checks, for tools and tests |

### Content capture (`pi_telemetry`)

| Export | Purpose |
|---|---|
| `with_content_capture()` | Wrap a context so content is recorded as `pi.content` events |
| `content_capture_of()` | The `ContentCapture` policy a context carries, or `None` |
| `record_content()` | Record one `pi.content` event if the span's context opted in |
| `ContentCapture` / `ContentRedactor` | The policy (`max_length`, `redact`) and the redactor type |
| `CONTENT_EVENT` / `CONTENT_EVENT_NAME` | The event definition schemas declare, and its name |

### Testing (`pi_telemetry.testing`)

| Export | Purpose |
|---|---|
| `create_telemetry_adapter_conformance()` | Creates runner-independent adapter conformance cases |
| `TelemetryAdapterFixture` | Fresh `context`, normalized `get_spans()`, and `aclose()` for one case |
| `TelemetryAdapterFixtureFactory` | Creates isolated fixtures |
| `TelemetryAdapterConformanceCase` | Grouped case (`group`, `name`, `run()`) that test runners execute |
| `assert_spans_match_schemas()` / `check_spans_against_schemas()` | Check recorded spans against schemas, parent rules included |
| `unreadable_mapping()`, `unreadable_sequence()`, `UnreadableError` | Payloads whose every read raises, for adapter tests |

### OpenTelemetry (`pi_telemetry.otel`)

| Export | Purpose |
|---|---|
| `OpenTelemetryContext` | The adapter (`tracer_provider`, `meter_provider`, `parent`, `record_metrics`, `record_exception`, `activate`) |
| `GEN_AI_FINISH_REASONS` | pi stop reasons as `gen_ai.response.finish_reasons` values |
| `AI_REQUEST_SPAN` | The span name that gets the GenAI mapping and metrics |

## Differences from the TypeScript Package

- Python naming: `start_span`, `add_event`, `set_attributes`, `set_status`, `get_spans`; `RecordedTelemetrySpan` is a
  frozen dataclass with `parent_id` and `end_sequence` (`None` while open). Schema data keeps upstream's keys.
- `start_span()` returns an awaitable. An `async def` callback's body runs when that awaitable is awaited (upstream's
  promises run eagerly); the callback itself is still invoked synchronously, exactly once.
- `undefined` attribute values are `None`; rejection values are exceptions (the same object is raised again).
- Names and attributes are checked at runtime instead of compile time ([Runtime Checks](#runtime-checks)); starters
  therefore read their schemas at creation and reject duplicate span names there.
- Adapters reject attribute values that are not primitives or homogeneous arrays (the call is ignored atomically).
- Additions: `is_telemetry_enabled()`, `bind_typed_span_starter()`, `InMemoryTelemetryContext.clear()`, content
  capture, schema checks for tests, the OpenTelemetry adapter, and the pi instrumentation: upstream 1.3.0's pi-ai only
  propagates `telemetryContext`, and its agent-core and pi-durable emit no spans yet.

## License

MIT
