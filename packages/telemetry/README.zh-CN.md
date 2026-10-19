# pi_telemetry

面向 pi 包的与后端无关的遥测契约与类型化 schema 工具。这是 [`@earendil-works/pi-telemetry`](https://github.com/earendil-works/pi/tree/main/packages/telemetry) 的 Python 移植版，外加一个 OpenTelemetry 适配器，以及 `pi_ai`、`pi_agent`、`pi_durable` 和 `pi_codemode` 的插桩。

本包提供：

- 一个显式的、基于回调的 `TelemetryContext` / `TelemetrySpan` 契约；
- 一个共享的 `NOOP_TELEMETRY_CONTEXT`；
- 一个参考实现 `InMemoryTelemetryContext`；
- 可序列化的 schema 定义，带 span 名和属性的运行时检查；
- 记录内容（`with_content_capture`）的显式选择加入；
- 一个适配器一致性套件（`pi_telemetry.testing`）；
- 一个 OpenTelemetry 适配器（`pi_telemetry.otel`，`otel` extra）；
- 核心中没有任何 exporter、全局当前 span 状态、`contextvars` 或对遥测后端的依赖。

应用可以使用内存参考实现、OpenTelemetry 适配器，或自己的适配器对接 Sentry、日志或其他后端。Pi 包显式传递遥测上下文，并单独定义各自的领域 schema。

## 目录

- [安装](#安装)
- [遥测概念](#遥测概念)
- [核心 Context API](#核心-context-api)
- [适配器契约](#适配器契约)
- [No-op Context](#no-op-context)
- [内存参考适配器](#内存参考适配器)
- [适配器一致性](#适配器一致性)
- [类型化 Schema](#类型化-schema)
  - [起始与完成属性](#起始与完成属性)
  - [运行时检查](#运行时检查)
- [Schema 元数据](#schema-元数据)
- [OpenTelemetry 适配器](#opentelemetry-适配器)
  - [端到端示例](#端到端示例)
  - [GenAI 语义约定](#genai-语义约定)
  - [指标](#指标)
- [Pi 包集成](#pi-包集成)
- [Span 目录](#span-目录)
- [数据策略](#数据策略)
- [API 参考](#api-参考)
- [与 TypeScript 包的差异](#与-typescript-包的差异)
- [许可证](#许可证)

## 安装

核心随 `pi-ai-python` 一起分发，且没有依赖。OpenTelemetry 适配器需要 `opentelemetry-api`；要导出再加 SDK（或发行版）：

```bash
pip install pi-ai-python
pip install "pi-ai-python[otel]" opentelemetry-sdk
```

## 遥测概念

遥测描述程序在运行期间做了什么。本包用 spans、attributes、events、statuses 和显式 context 来建模这项工作：

| 概念 | 通俗含义 |
|---|---|
| **Span** | 一次操作的计时记录，例如加载账号或发起 AI 请求。它在工作开始前开始，在工作结束时结束。 |
| **父与子 span** | 操作可以包含更小的操作。一个请求 span 可能包含一次缓存查找和一次数据库查询。它们共同构成一棵显示时间花在哪里的树。 |
| **Attribute** | 附加到 span 的命名事实，例如 `provider: "openai"`、`cache.hit: True` 或 `item_count: 12`。Attributes 描述操作及其结果。 |
| **Event** | span 过程中某个时点的命名发生事件，例如 `retry.scheduled` 或 `cache.lookup`。事件没有持续时间，可以携带自己的 attributes。 |
| **Status** | 操作的结果：`ok` 或 `error`。错误状态可以包含错误名和消息。 |
| **Context** | 一个标识新工作在 span 树中属于哪里的句柄。从某个 context 启动 span 会让它成为该 context 的子 span。 |

例如，加载一个账号可能产生这样的遥测：

```text
example.account.load                         span
├─ attributes: account.id=123, found=true   关于该 span 的事实
├─ event: example.cache.lookup              该 span 过程中的发生事件
│  └─ attribute: cache.hit=false            关于该事件的事实
└─ status: ok                               最终结果
```

span 是诊断数据，不是业务状态。记录它绝不能改变账号加载是否运行、成功、失败或是否被持久化。适配器把这些通用概念翻译成 OpenTelemetry、Sentry、日志或其他后端对应的概念。

## 核心 Context API

一个 `TelemetryContext` 围绕一个回调启动 span。回调收到一个 `TelemetrySpan`，它也是子 span 的显式父 context。`start_span()` 返回回调结果的可 await 对象：

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

把回调 span 传给更低层的工作以创建显式嵌套：

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

没有公开的 `end()` 方法。`start_span()` 拥有结算权，并在回调的值或可 await 对象落定前保持 span 打开。对由普通返回值表示的预期失败，显式设置状态：

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

回调在调用时同步运行，恰好一次。普通函数的 span 立即结算；`async def` 回调返回一个协程，它在 `start_span()` 的结果被 await 时运行，并保持其 span 打开。像任何协程一样，await 它（或把它包进任务）。

## 适配器契约

适配器实现 `TelemetryContext`，并把通用 API 桥接到它的后端。它必须：

- 创建一个子 span 并同步调用回调，恰好一次；
- 保留回调的返回值与抛出的异常（同一个对象），返回一个在同步抛出后会再次抛出它的可 await 对象；
- 保持原生 span 打开，直到返回的可 await 对象落定；
- 把正常完成视为 `ok`，抛出视为错误，除非已显式设置状态；
- 让重复的 `set_status()` 调用后写覆盖先写；
- 合并 `set_attributes()` 调用，后面的非 `None` 值替换更早的值，`None` 被忽略；
- 让记录方法同步、被动、不抛出；
- 忽略结算后的调用；
- 原子地忽略一次失败的记录调用（不可读的 payload，或不是基本类型或数组的值）、抑制后端失败，并且仍然恰好执行一次业务回调。

适配器可以在内部为自动插桩激活后端原生的环境上下文（OpenTelemetry 适配器就这样做），但 pi 代码总是通过 `TelemetryContext` 参数传播父 span。Exporter 缓冲、刷新、采样、后端 ID 和后端专属的上下文对象都属于适配器。用[适配器一致性套件](#适配器一致性)检查这些可观察语义。

## No-op Context

在遥测可选时使用 `NOOP_TELEMETRY_CONTEXT`：

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, is_telemetry_enabled


async def run_operation(_span: TelemetrySpan) -> int:
    return 7


result = await NOOP_TELEMETRY_CONTEXT.start_span({"name": "example.operation"}, run_operation)
print(result, is_telemetry_enabled(NOOP_TELEMETRY_CONTEXT), is_telemetry_enabled(None))
```

no-op 上下文：

- 同步调用回调；
- 保留返回值和抛出的异常，并把同步抛出变成会抛出它的可 await 对象；
- 使用一个共享的冻结惰性 span，包括嵌套 span 也用同一个；
- 不检查也不保留 names、attributes、events 或 statuses。

`is_telemetry_enabled(context)` 对 `None` 和 no-op 上下文为 `False`。Pi 的插桩先检查它，因此禁用路径不构建 attributes、不启动 span。

## 内存参考适配器

`InMemoryTelemetryContext` 是与后端无关的参考实现。它对测试、本地诊断以及故意想要无需 exporter 的进程内捕获的应用有用：

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

`get_spans()` 按 span 启动顺序返回分离的 `RecordedTelemetrySpan` 快照。每个都包含确定性的数值 `id`、`parent_id`、合并后的 `attributes`、有序的 `events`、最终 `status`、`settled` 标志和确定性的 `end_sequence`。它不记录时间戳。数组记录为列表。

该适配器可以安全地作为普通 `TelemetryContext` 使用，但存储无界且为进程本地。创建新实例（或调用 `clear()`）以隔离测试或记录范围，并且除非调用者的数据策略允许，否则不要捕获敏感属性。

## 适配器一致性

`pi_telemetry.testing` 导出一个与运行器无关的一致性套件，建模为分组用例。一个 fixture 提供全新的上下文，并把其后端的已完成 span 转成规范化的 `RecordedTelemetrySpan` 快照。用 pytest 注册这些用例，或把它们作为普通协程运行：

```python
import pytest

from pi_telemetry.testing import create_telemetry_adapter_conformance


class Fixture:
    def __init__(self) -> None:
        self.adapter = InMemoryTelemetryContext()  # 你的适配器在这里
        self.context = self.adapter

    async def get_spans(self):
        return self.adapter.get_spans()  # 在这里刷新并规范化你后端的 span

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

该套件检查同步的单次收录、结果与异常的同一性、自动与显式状态、attribute 合并、事件顺序、结算后调用的惰性、嵌套与并发父子关系，以及不可读遥测 payload 失败的抑制。`get_spans()` 在返回前可以刷新异步 exporter。用例使用普通 `assert` 语句，不导入任何测试框架。`packages/telemetry/tests/test_otel.py` 用 SDK 的 span processors 对 OpenTelemetry 适配器运行该套件。

## 类型化 Schema

底层 span API 有意接受开放的名字和 attribute 包，以保持适配器通用。领域包定义封闭、可序列化的 schema。schema 是带上游键的普通 JSON 数据，因此同一份文档在 TypeScript 和 Python 中描述同一套词汇；`TypedDict`（`TelemetrySchemaDefinition`、`TelemetrySpanDefinition` 等）让类型检查器验证 schema 字面量的结构。

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

starter 用 span 名、它的 start attributes 和一个回调来调用。回调收到该 span（通过它的 schema 查看，即 `SchemaTelemetrySpan`）和一个基于相同 schemas、已绑定到该 span 的子 starter：

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

### 起始与完成属性

`startAttributes` 和 `endAttributes` 描述一个属性通常何时已知，而不是分开的运行时存储：

| Schema 字段 | 值如何记录 | 必填性 |
|---|---|---|
| `startAttributes` | 在创建 span 时通过类型化 starter 的 `attributes` 参数传入 | 每个定义显式设置 `required: True` 或 `False` |
| `endAttributes` | 稍后通过 schema 作用域的 span 的 `set_attributes()` 方法加入 | 总是可选 |

两组都会成为同一后端 span 上的普通 attributes。没有单独的 end-attribute payload 或 end 回调。在前面的示例中，`example.read` 开始时已知 `example.resource`，而 `example.item_count` 只在 `read_accounts()` 返回后才已知。

“End”意味着完成时的富化：end attribute 可以在回调活跃期间的任何时点设置，不可用时可以省略。调用零次 `set_attributes()` 是合法的。这对早期失败、取消以及可能不会出现在每条路径上的提供商专属数据很重要。

重复的 `set_attributes()` 调用合并进同一个 attribute 包。同一键的较晚值替换较早值，`None` 被忽略。schema 作用域的方法只接受当前 span 声明的 end attributes。

Attributes 不结束 span。返回、抛出或让回调的可 await 对象落定控制结算；`start_span()` 执行实际的结束操作。结算后对适配器的调用是惰性的。

一个 starter 可以组合多个独立版本化的 schema；跨它们重复的 span 名在 starter 创建时被拒绝：

```python
from pi_agent.telemetry import AGENT_TELEMETRY_SCHEMAS
from pi_ai.telemetry import AI_TELEMETRY_SCHEMAS

start_agent_span = create_typed_span_starter(telemetry, [*AGENT_TELEMETRY_SCHEMAS, *AI_TELEMETRY_SCHEMAS])
print(start_agent_span.span_names)
```

`define_telemetry_schema()` 是一个类型化同一性函数。它返回普通 JSON 可序列化数据，不执行任何验证或父规则执行。

### 运行时检查

TypeScript 在编译时拒绝未知名、缺失的必填属性、未知键、无效的封闭集值、未声明的事件以及空 schema 上的属性。Python 做不到，因此类型化 starter 在它们被调用时检查这些。`on_invalid` 决定违规做什么：

| `on_invalid` | 一次违规 |
|---|---|
| `"raise"`（默认） | 在记录任何东西之前抛 `TelemetrySchemaError`（一个 `TypeError`）：编译错误的类比 |
| `"warn"` | 丢弃违规的 attributes 或 event，发出 `TelemetrySchemaWarning`，然后继续 |
| `"ignore"` | 静默丢弃它们 |

```python
from pi_telemetry import TelemetrySchemaError

try:
    start_span("example.read", {"example.resource": "invoice"}, read)
except TelemetrySchemaError as error:
    print(error)
```

Pi 自己的插桩使用 `"warn"`，因此 schema 错误绝不能改变它观察的代码的行为；pi 的测试套件把 `TelemetrySchemaWarning` 变成错误（`filterwarnings`）。要在你自己的测试中对照 schema（包括父规则）检查记录的 span，用 `pi_telemetry.testing.assert_spans_match_schemas(spans, schemas)`。要对一套词汇做静态类型化，拥有方包把 starter 包在带类型参数的函数中。

## Schema 元数据

受支持的 attribute 类型：

- `string`、`number` 和 `boolean`（`str`、`int` 或 `float`、`bool`；`bool` 不是 number）；
- `string[]`、`number[]` 和 `boolean[]`（列表或元组，记录为列表）。

Attribute 定义支持：

- `values`：标量值的封闭集；
- `elementValues`：数组元素的封闭集；
- `examples`：文档示例；
- `sensitive`：标记需要特殊处理的数据；
- `cardinality`：记录期望的 `low` 或 `high` 基数。

Start 和 event attributes 声明 `required`。End attributes 不声明；见[起始与完成属性](#起始与完成属性)。

父元数据是描述性 schema 数据：

- `{"kind": "any"}`：根或任何调用者 span；
- `{"kind": "root_or_external"}`：根或 schema 之外的调用者拥有的 span；
- `{"kind": "spans", "spans": [...]}`：仅列出的 schema span。

适配器不需要理解 schema 对象。插桩辅助和测试用它们来保持发出的名字和 attributes 一致。

## OpenTelemetry 适配器

`pi_telemetry.otel.OpenTelemetryContext` 是一个由 OpenTelemetry API 支撑的根 `TelemetryContext`：

```python
from opentelemetry.sdk.trace import TracerProvider

from pi_telemetry.otel import OpenTelemetryContext

telemetry = OpenTelemetryContext(tracer_provider=TracerProvider())
```

- pi span 变成父 span 为 pi 父 span 的 OTel span。根 pi span 的父 span 是 `parent`（一个 OTel `Context`），或默认是 span 启动时当前的 OTel 上下文，因此 pi span 会加入应用的 trace。
- Attributes 保留它们的 `pi.*` 名字；events 映射到 span events。
- 显式的 `error` 状态映射到带错误名作为 `error.type`、消息（如有）作为 description 的 `StatusCode.ERROR`；显式 `ok` 映射到 `StatusCode.OK`；自动 `ok` 让状态保持未设置。
- 抛出会设置 `StatusCode.ERROR` 和 `error.type`（异常类），不带 description：异常消息是自由形式的错误细节。`record_exception=True` 也把它们记录为 OTel 异常事件。
- 回调运行时，它的 OTel span 是当前 OTel span（`activate=True`），因此自动插桩（例如对 `httpx` 的）会嵌套在它之下。
- `pi.ai.request` span 是带 [GenAI 语义约定属性](#genai-语义约定) 的 `CLIENT` span，并为[指标](#指标)供数（`record_metrics=True`；`meter_provider` 默认是全局的）。

导入 `pi_telemetry` 绝不导入 OpenTelemetry；只有 `pi_telemetry.otel` 会。

### 端到端示例

一次导出到 OpenTelemetry SDK 内存 span exporter 和 metric reader 的 agent 运行。faux provider 代替模型；用真实提供商时除了 models 集合之外什么都不变。

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
# 想快速查看原始 span：tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))。
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

同样的 span 和指标会去往配置在 providers 上的任何 OpenTelemetry exporter（OTLP、console、厂商的）。

### GenAI 语义约定

结算时，`pi.ai.request` span 还会获得这些 OpenTelemetry GenAI 属性（`pi.ai.*` 属性保留）：

| GenAI 属性 | 来源 |
|---|---|
| `gen_ai.operation.name` | `"chat"` |
| `gen_ai.provider.name`、`gen_ai.system` | `pi.ai.provider`（`gen_ai.system` 是许多后端仍在读取的较老约定） |
| `gen_ai.request.model` | `pi.ai.model` |
| `gen_ai.response.model` | `pi.ai.response.model` |
| `gen_ai.conversation.id` | `pi.ai.session_id` |
| `gen_ai.request.max_tokens`、`gen_ai.request.temperature` | `pi.ai.request.max_tokens`、`pi.ai.request.temperature` |
| `gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens` | `pi.ai.usage.input_tokens`、`pi.ai.usage.output_tokens` |
| `gen_ai.usage.cache_read.input_tokens`、`gen_ai.usage.cache_creation.input_tokens` | `pi.ai.usage.cache_read_tokens`、`pi.ai.usage.cache_write_tokens` |
| `gen_ai.usage.reasoning.output_tokens` | `pi.ai.usage.reasoning_tokens` |
| `gen_ai.response.finish_reasons` | `[pi.ai.stop_reason]`：`stop`、`length`、`tool_call`（toolUse）、`error`、`cancelled`（aborted）、`deferred` |
| `gen_ai.response.time_to_first_chunk` | `pi.ai.time_to_first_token_ms` / 1000 |
| `error.type` | 状态的错误名（`pi.ai.error_kind`） |

span 保留 pi 的名字（`pi.ai.request`）而不是 GenAI 的 `"{operation} {model}"`。

### 指标

每个 `pi.ai.request` span 记录一次。标签只有低基数：`gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`pi.ai.api`，失败请求上还有 `error.type`，以及注明的 token 类型。会话 ID、任务 ID 和其他每个请求的值绝不作为标签。

| Instrument | 种类、单位 | 值 |
|---|---|---|
| `gen_ai.client.operation.duration` | histogram, s | 请求持续时间（失败时带 `error.type`） |
| `gen_ai.client.operation.time_to_first_chunk` | histogram, s | 到第一个流式 delta 的时间 |
| `gen_ai.client.token.usage` | histogram, {token} | 输入和输出 token（`gen_ai.token.type`） |
| `pi.ai.requests` | counter, {request} | 请求（失败时带 `error.type`） |
| `pi.ai.errors` | counter, {request} | 以错误或中止结束的请求，按 `error.type` |
| `pi.ai.tokens` | counter, {token} | 按 `pi.ai.token.type` 的 token：`input`、`output`、`cache_read`、`cache_write` |
| `pi.ai.cost` | counter, USD | 估算成本 |

## Pi 包集成

包所有权按上游的意图划分：`pi_telemetry` 拥有契约；每个发 span 的包拥有自己的 schema。

| 包 | Schema（`<module>.<NAME>`） | Spans | 上下文来自哪里 |
|---|---|---|---|
| `pi_ai` | `pi_ai.telemetry.AI_TELEMETRY_SCHEMA` | `pi.ai.request` | `Models.stream`/`stream_simple`/`complete`/`complete_simple`/`stream_deferred`/`fetch_deferred`（以及 `pi_ai.compat`）的 `telemetry_context` 请求选项 |
| `pi_agent` | `pi_agent.telemetry.AGENT_TELEMETRY_SCHEMA` | `pi.agent.run` | `Agent(telemetry_context=...)` 或 `AgentLoopConfig.telemetry_context` |
| `pi_durable` | `pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`、`SESSION_TELEMETRY_SCHEMA` | `pi.harness.generation`、`pi.harness.tool`、`pi.harness.task`、`pi.session.commit` | `HarnessOptions["telemetryContext"]`、`create_session(..., telemetry_context=...)` |
| `pi_codemode` | `pi_codemode.telemetry.CODEMODE_TELEMETRY_SCHEMA` | `pi.codemode.run` | `CodemodeSandboxBase.execute(..., telemetry_context=...)`、`create_codemode_tool(..., telemetry_context=...)`；在 pi_durable 中则是代码模式工具的 `pi.harness.tool` span |

上下文如何流动，始终作为显式参数：

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

`pi_ai` 像上游一样通过 `build_base_options()` 把 `telemetry_context` 传给提供商。在 `Models` 之下，提供商收到 `pi.ai.request` span 本身（一个 `AiRequestSpan`），因此提供商级 span 嵌套在请求之下，而 `retry_provider_request()` 在它上面记录 `pi.ai.retry` 事件。没有 `telemetry_context`，或传 `NOOP_TELEMETRY_CONTEXT` 时，什么都不构建：`Models` 返回它一直返回的同一个流，提供商看到的选项不变。

遥测上下文是进程本地的。pi_durable 只在内存中保留 Harness 的上下文：它绝不写进记录、文档、检查点或日志，重启后恢复的任务在重新打开 Harness 的进程的上下文下运行。

## Span 目录

pi 发出的每个 span，由 schemas 生成。“start” attributes 在 span 启动时设置，“end” attributes 在已知时设置（总是可选）。每个 attribute 都是基本类型或数组；没有一个携带内容。

<!-- span-catalog:start -->
#### `pi.ai.request`

pi_ai.Models 的一次模型请求，从分发到终端消息。拥有者：`pi_ai`（`pi_ai.telemetry.AI_TELEMETRY_SCHEMA`，版本 1）。父：任意（根或调用者的 span）。错误状态：终端消息的停止原因是 error 或 aborted。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.ai.provider` | start, required | string | 提供商 ID |
| `pi.ai.api` | start, required | string | API 实现 |
| `pi.ai.model` | start, required | string | 请求的模型 ID |
| `pi.ai.operation` | start, required | string: `stream` \| `stream_simple` \| `fetch_deferred` | Models 入口：完整选项、提供商无关选项，或延迟获取 |
| `pi.ai.streaming` | start, required | boolean | 调用者是否消费事件（stream*）或只消费结果（complete*、fetch_deferred） |
| `pi.ai.reasoning` | start | string: `off` \| `minimal` \| `low` \| `medium` \| `high` \| `xhigh` \| `max` | 请求的思考级别（stream_simple） |
| `pi.ai.tool_count` | start, required | number | 提供给模型的工具数 |
| `pi.ai.message_count` | start, required | number | 发送的 transcript 消息数 |
| `pi.ai.session_id` | start | string | 请求的 session_id 选项（设置时） |
| `pi.ai.request.max_tokens` | start | number | 请求的输出 token 上限 |
| `pi.ai.request.temperature` | start | number | 请求的 temperature |
| `pi.ai.usage.input_tokens` | end | number | 输入 token |
| `pi.ai.usage.output_tokens` | end | number | 输出 token |
| `pi.ai.usage.cache_read_tokens` | end | number | 从缓存读取的输入 token |
| `pi.ai.usage.cache_write_tokens` | end | number | 写入缓存的输入 token |
| `pi.ai.usage.reasoning_tokens` | end | number | 提供商报告时，花在思考上的输出 token |
| `pi.ai.cost` | end | number | 以 USD 计的估算总成本 |
| `pi.ai.stop_reason` | end | string: `stop` \| `length` \| `toolUse` \| `error` \| `aborted` \| `deferred` \| `pending` | 终端消息的停止原因 |
| `pi.ai.error_kind` | end | string: `aborted` \| `cancelled` \| `context_overflow` \| `rate_limited` \| `http_error` \| `provider_error` | 请求失败的原因；绝不是提供商的错误文本 |
| `pi.ai.time_to_first_token_ms` | end | number | 从请求开始到第一个文本、思考或工具调用 delta 的毫秒数 |
| `pi.ai.duration_ms` | end | number | 到终端消息的毫秒数 |
| `pi.ai.http.status` | end | number | 提供商报告时，最后一次提供商响应的 HTTP 状态 |
| `pi.ai.response.model` | end | string | 提供商在其响应中报告的模型 ID |
| `pi.ai.retry_count` | end | number | 请求内的 HTTP 重试次数 |

| Event | Attributes | 说明 |
| --- | --- | --- |
| `pi.ai.retry` | `pi.ai.retry.attempt`、`pi.ai.retry.reason`、`pi.ai.retry.delay_ms`、`pi.ai.retry.http_status`（可选） | 一次可重试的 HTTP 失败；延迟后再次发送请求 |
| `pi.ai.context_overflow` | `pi.ai.context_window`（可选） | 请求超出模型的上下文窗口 |
| `pi.ai.abort` | - | 请求通过它的 signal 被中止 |

声明 `pi.content`（只在内容捕获下记录）。

#### `pi.agent.run`

agent 循环的一次运行，从它的 prompt（或续接）到 agent_end。拥有者：`pi_agent`（`pi_agent.telemetry.AGENT_TELEMETRY_SCHEMA`，版本 1）。父：任意（根或调用者的 span）。错误状态：run 以错误或中止结束，或循环抛出。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.agent.provider` | start, required | string | run 初始模型的提供商 |
| `pi.agent.model` | start, required | string | run 的初始模型 ID |
| `pi.agent.tool_count` | start, required | number | agent 可以执行的工具数 |
| `pi.agent.mode` | start, required | string: `prompt` \| `continue` | 新的 prompt 消息，或对现有上下文的续接 |
| `pi.agent.reasoning` | start | string | 请求的思考级别 |
| `pi.agent.turns` | end | number | run 中的模型响应数 |
| `pi.agent.tool_calls` | end | number | run 产生的工具结果数 |
| `pi.agent.usage.input_tokens` | end | number | 对 run 各响应求和的输入 token |
| `pi.agent.usage.output_tokens` | end | number | 对 run 各响应求和的输出 token |
| `pi.agent.cost` | end | number | run 各响应以 USD 计的估算成本 |
| `pi.agent.end_reason` | end | string: `completed` \| `ended` \| `error` \| `aborted` \| `failed` | run 结束的原因 |

| Event | Attributes | 说明 |
| --- | --- | --- |
| `pi.agent.steering` | `pi.agent.message_count` | 引导消息被注入 run |
| `pi.agent.follow_up` | `pi.agent.message_count` | 跟进消息在本会停止的 run 之后继续它 |

#### `pi.harness.generation`

生成任务（pi.generation）的一次阶段处理器运行。拥有者：`pi_durable`（`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`，版本 1）。父：根或调用者拥有的 span。错误状态：阶段处理器抛出。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | 任务 ID |
| `pi.harness.conversation_id` | start, required | number | 任务的会话 ID |
| `pi.harness.phase` | start, required | string: `prepare` \| `request` \| `retry` \| `poll` \| `tools` \| `abort` | 阶段处理器（abort：abort 处理器） |
| `pi.harness.restarts` | start | number | 在运行期间未改变其状态就切断任务的连续重启次数 |
| `pi.harness.attempt` | start | number | 该阶段的生成尝试次数 |
| `pi.harness.model.provider` | start | string | 选定后，该阶段模型的提供商 |
| `pi.harness.model.id` | start | string | 选定后，该阶段模型的模型 ID |
| `pi.harness.phase_outcome` | end | string: `returned` \| `raised` \| `aborted` | 处理器是返回了（持久化进度）、抛出了，还是因中止结束 |
| `pi.harness.duration_ms` | end | number | 处理器运行的毫秒数 |
| `pi.harness.generation.stop_reason` | end | string: `stop` \| `length` \| `toolUse` \| `error` \| `aborted` \| `deferred` \| `pending` | 请求或轮询阶段收到的响应的停止原因 |
| `pi.harness.generation.tool_calls` | end | number | 该响应中的工具调用数 |

#### `pi.harness.tool`

一次工具调用的执行：调用阶段、恢复，或 abort 处理器。拥有者：`pi_durable`（`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`，版本 1）。父：任意（根或调用者的 span）。错误状态：调用未成功。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | 任务 ID |
| `pi.harness.conversation_id` | start, required | number | 任务的会话 ID |
| `pi.harness.phase` | start, required | string: `call` \| `execute` \| `abort` | call：首次运行；execute：重启后的恢复；abort：abort 处理器 |
| `pi.harness.tool.name` | start, required | string | 被调用的工具 |
| `pi.harness.tool.extension` | start | string | 提供该工具的扩展 |
| `pi.harness.tool.replay` | start | string: `safe` \| `unsafe` | 工具的重放策略（已解析时） |
| `pi.harness.tool.nested` | start, required | boolean | 调用是由运行中的工具（execute_tool()）而非模型发起的 |
| `pi.harness.restarts` | start | number | 在运行期间未改变其状态就切断任务的连续重启次数 |
| `pi.harness.tool.outcome` | end | string: `success` \| `error` \| `blocked` \| `interrupted` | 调用如何结束 |
| `pi.harness.tool.error_kind` | end | string: `tool_unavailable` \| `invalid_arguments` \| `blocked` \| `tool_error` \| `invalid_structured_output` \| `interrupted` \| `aborted` \| `abandoned` | 调用失败的原因；绝不是错误文本 |
| `pi.harness.tool.duration_ms` | end | number | 工具 execute() 运行的毫秒数 |
| `pi.harness.tool.output_bytes` | end | number | 结算后文本输出的 UTF-8 字节数 |
| `pi.harness.tool.truncated` | end | boolean | 输出是否被截到工具的输出限制 |

| Event | Attributes | 说明 |
| --- | --- | --- |
| `pi.harness.tool.blocked` | - | 一个 before_tool hook 阻塞了调用（原因不记录） |
| `pi.harness.tool.validation_failed` | `pi.harness.tool.validation_stage` | 调用的参数未通过校验 |

声明 `pi.content`（只在内容捕获下记录）。

#### `pi.harness.task`

生成和工具调用之外的任务的一次阶段处理器运行。拥有者：`pi_durable`（`pi_durable.telemetry.HARNESS_TELEMETRY_SCHEMA`，版本 1）。父：根或调用者拥有的 span。错误状态：阶段处理器抛出。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.harness.task_id` | start, required | number | 任务 ID |
| `pi.harness.conversation_id` | start, required | number | 任务的会话 ID |
| `pi.harness.phase` | start, required | string | 阶段处理器（abort：abort 处理器） |
| `pi.harness.restarts` | start | number | 在运行期间未改变其状态就切断任务的连续重启次数 |
| `pi.harness.task.kind` | start, required | string | 任务定义名 |
| `pi.harness.task.extension` | start | string | 安装该定义的扩展；内置任务缺省 |
| `pi.harness.phase_outcome` | end | string: `returned` \| `raised` \| `aborted` | 处理器是返回了（持久化进度）、抛出了，还是因中止结束 |
| `pi.harness.duration_ms` | end | number | 处理器运行的毫秒数 |

#### `pi.session.commit`

一次 Session 提交：change 回调、结算和 Storage 提交。拥有者：`pi_durable`（`pi_durable.telemetry.SESSION_TELEMETRY_SCHEMA`，版本 1）。父：任意（根或调用者的 span）。错误状态：change 回调、结算或存储抛出。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.session.storage` | start, required | string | 存储后端 |
| `pi.session.write_count` | end | number | 提交所做的存储写入数 |
| `pi.session.duration_ms` | end | number | 在变更行上的毫秒数 |

#### `pi.codemode.run`

一次模型编写脚本的执行。拥有者：`pi_codemode`（`pi_codemode.telemetry.CODEMODE_TELEMETRY_SCHEMA`，版本 1）。父：任意（根或调用者的 span）。错误状态：脚本失败、超时、被中止或发散。

| Attribute | 何时 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pi.codemode.line_count` | start, required | number | 脚本行数 |
| `pi.codemode.timeout_ms` | start | number | run 的截止时间；没有时缺省 |
| `pi.codemode.mode` | start, required | string: `execute` \| `deterministic` | execute()：沙箱调用工具；deterministic：start()，宿主应答 |
| `pi.codemode.tool_count` | start, required | number | 脚本可以调用的工具数 |
| `pi.codemode.outcome` | end | string: `success` \| `error` | 脚本是否成功 |
| `pi.codemode.error_kind` | end | string: `script` \| `timeout` \| `aborted` \| `sandbox` \| `different_path` \| `version_mismatch` \| `deadline_passed` \| `host` | run 失败的原因；绝不是错误文本 |
| `pi.codemode.nested_call_count` | end | number | 脚本发出的工具调用数 |
| `pi.codemode.duration_ms` | end | number | run 花费的毫秒数 |
| `pi.codemode.replayed` | end | boolean | run 是否在重启后重放了日志 |
| `pi.codemode.replayed_call_count` | end | number | 重放投递的日志化结果数 |

| Event | Attributes | 说明 |
| --- | --- | --- |
| `pi.codemode.different_path` | - | 重放走了与它所恢复的 run 不同的路径 |
| `pi.codemode.timeout` | `pi.codemode.timeout_ms`（可选） | run 的截止时间已过 |

声明 `pi.content`（只在内容捕获下记录）。
<!-- span-catalog:end -->

## 数据策略

遥测是进程本地的诊断，不是持久的应用状态。不要把 `TelemetryContext`、`TelemetrySpan` 或后端原生的 trace 对象持久化进 records、messages、snapshots 或 deferred 句柄。

Attribute 值限于基本标量和数组。默认情况下，pi 的插桩不记录提示、完成、思考、工具参数或输出、文件内容、提供商 payload、头、凭据或自由形式的错误细节：只记录名字、计数、大小、持续时间、token 用量、成本、封闭集种类和 ID。错误状态携带一种类（`rate_limited`、`tool_error`、异常类名），绝不携带消息。Pi 的测试套件用哨兵字符串检查这一点。

内容只在显式选择加入后记录。`with_content_capture(context, max_length=..., redact=...)` 包装一个上下文；在它之下、以及从它启动的每个 span 之下运行的插桩，把内容记录为 `pi.content` 事件：种类（`system`、`user`、`assistant`、`thinking`、`tool_call`、`tool_arguments`、`tool_output`、`code`、`output`、`error` 等）、`redact(kind, text)` 之后的文本（返回 `None` 以丢弃它）、它的长度，以及它是否在 `max_length` 个字符处被截断（默认 4096；`-1` 保留一切）：

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

选择加入随上下文一起显式传播，像任何父 span 一样；没有全局的东西改变。Pi 代码用 `record_content(span, kind, text)` 记录内容，没有选择加入时它是空操作。

本包不使用 `contextvars` 或其他环境上下文 API。后端适配器仍然对自己运行时的行为负责。

## API 参考

### 核心类型与值（`pi_telemetry`）

| 导出 | 用途 |
|---|---|
| `TelemetryContext` | 启动由回调管理的子 span（`start_span(options, callback)`） |
| `TelemetrySpan` | 记录 attributes、events 和 status；也充当子上下文 |
| `SpanOptions` | span 的 `name` 和可选 start `attributes` |
| `SpanAttributes` / `AttributeValue` / `AttributeScalar` | 开放适配器级的 attribute 包与受支持的值 |
| `SpanStatus`（`OkSpanStatus`、`ErrorSpanStatus`、`SpanErrorInfo`） | 显式的 `ok` 或 `error` 状态 |
| `NOOP_TELEMETRY_CONTEXT` | 禁用遥测的共享被动上下文 |
| `is_telemetry_enabled()` | 对 `None` 和 no-op 上下文为 `False` |
| `InMemoryTelemetryContext` | 带确定性进程内记录的参考适配器（`get_spans()`、`clear()`） |
| `RecordedTelemetrySpan` | 规范化的捕获 span 快照 |
| `RecordedTelemetryEvent` | 规范化的捕获事件快照 |

### Schema 定义与检查（`pi_telemetry`）

| 导出 | 用途 |
|---|---|
| `define_telemetry_schema()` | 可序列化 schema 数据的类型化同一性辅助 |
| `create_typed_span_starter()` | 把一个父上下文绑定到一个或多个 schema 词汇（`on_invalid`） |
| `bind_typed_span_starter()` | 把 starter 重新绑定到另一个父上下文，而不重新索引其 schemas |
| `TypedSpanStarter` | 该 starter：`starter(name, attributes, callback)`；`span_names`、`context` |
| `SchemaTelemetrySpan` | 限定到一个 schema span 的 span 视图（`name`、`definition`、`span`） |
| `TelemetrySchemaError` / `TelemetrySchemaWarning` | 一次 schema 违规（抛出，或用 `on_invalid="warn"` 警告） |
| `TelemetrySchemaDefinition` | 顶层 schema 形状 |
| `TelemetrySpanDefinition` | span 元数据、父、attributes、events 和状态规则 |
| `TelemetryAttributeType` | 受支持的标量与数组类型名 |
| `TelemetryAttributeDefinition` | attribute 类型、允许值、示例和元数据 |
| `TelemetryStartAttributeDefinition` / `TelemetryEventAttributeDefinition` | 带必填性的定义 |
| `TelemetryEventDefinition` | 事件描述与 attribute 定义 |
| `TelemetryParentDefinition`（`TelemetryParentAny`、`TelemetryParentRootOrExternal`、`TelemetryParentSpans`） | 父规则 |
| `TelemetrySpanStatusRule` | `{"default": "ok", "errorWhen": ...}` |
| `schema_span_names()`、`check_attribute_value()`、`check_start_attributes()`、`check_end_attributes()`、`check_event()` | 运行时检查，供工具和测试使用 |

### 内容捕获（`pi_telemetry`）

| 导出 | 用途 |
|---|---|
| `with_content_capture()` | 包装一个上下文，使内容被记录为 `pi.content` 事件 |
| `content_capture_of()` | 上下文携带的 `ContentCapture` 策略，或 `None` |
| `record_content()` | 如果 span 的上下文选择加入，记录一个 `pi.content` 事件 |
| `ContentCapture` / `ContentRedactor` | 策略（`max_length`、`redact`）与脱敏器类型 |
| `CONTENT_EVENT` / `CONTENT_EVENT_NAME` | schemas 声明的事件定义，及其名字 |

### 测试（`pi_telemetry.testing`）

| 导出 | 用途 |
|---|---|
| `create_telemetry_adapter_conformance()` | 创建与运行器无关的适配器一致性用例 |
| `TelemetryAdapterFixture` | 一个用例的全新 `context`、规范化 `get_spans()` 和 `aclose()` |
| `TelemetryAdapterFixtureFactory` | 创建隔离的 fixtures |
| `TelemetryAdapterConformanceCase` | 测试运行器执行的分组用例（`group`、`name`、`run()`） |
| `assert_spans_match_schemas()` / `check_spans_against_schemas()` | 对照 schemas（含父规则）检查记录的 span |
| `unreadable_mapping()`、`unreadable_sequence()`、`UnreadableError` | 每次读取都抛出的 payload，供适配器测试使用 |

### OpenTelemetry（`pi_telemetry.otel`）

| 导出 | 用途 |
|---|---|
| `OpenTelemetryContext` | 该适配器（`tracer_provider`、`meter_provider`、`parent`、`record_metrics`、`record_exception`、`activate`） |
| `GEN_AI_FINISH_REASONS` | 作为 `gen_ai.response.finish_reasons` 值的 pi 停止原因 |
| `AI_REQUEST_SPAN` | 获得 GenAI 映射和指标的 span 名 |

## 与 TypeScript 包的差异

- Python 命名：`start_span`、`add_event`、`set_attributes`、`set_status`、`get_spans`；`RecordedTelemetrySpan` 是带 `parent_id` 和 `end_sequence`（打开时为 `None`）的冻结 dataclass。Schema 数据保留上游的键。
- `start_span()` 返回一个可 await 对象。`async def` 回调的函数体在该可 await 对象被 await 时运行（上游的 promises 急切运行）；回调本身仍被同步调用，恰好一次。
- `undefined` 的 attribute 值是 `None`；拒绝值是异常（同一个对象被再次抛出）。
- 名字和 attributes 在运行时而非编译时检查（[运行时检查](#运行时检查)）；因此 starters 在创建时读取它们的 schemas 并在那里拒绝重复的 span 名。
- 适配器拒绝不是基本类型或同质数组的 attribute 值（该调用被原子忽略）。
- 新增：`is_telemetry_enabled()`、`bind_typed_span_starter()`、`InMemoryTelemetryContext.clear()`、内容捕获、测试用的 schema 检查、OpenTelemetry 适配器，以及 pi 插桩：上游 1.3.0 的 pi-ai 只传播 `telemetryContext`，而它的 agent-core 和 pi-durable 还不发 span。

## 许可证

MIT
