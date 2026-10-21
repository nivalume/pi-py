# pi-py

[pi](https://github.com/earendil-works/pi) 核心库的 Python 实现，包含以下模块：

| Import 名 | 对应上游 | 作用 |
| --- | --- | --- |
| [`pi_ai`](packages/ai/README.md) | `@earendil-works/pi-ai` | 统一的 LLM API：provider 集合、鉴权解析（API key / OAuth）、流式事件、token 与费用统计、工具调用、跨 provider 上下文切换、图像生成、分类 |
| [`pi_agent`](packages/agent/README.md) | `@earendil-works/pi-agent-core` | 有状态 Agent：工具执行（并行/串行）、steering 与 follow-up 队列、生命周期事件、钩子 |
| [`pi_durable`](packages/durable/README.md) | `@earendil-works/pi-durable` | 持久化 Agent 运行时：会话、模型调用、工具调用和你自己的状态都先写入存储（内存、SQLite、JSONL）再对外展示，崩溃或重启后从断点继续 |
| [`pi_chord`](packages/chord/README.md) | `@earendil-works/chord`（子集） | `pi_durable` 的底层：显式取消上下文、严格 JSON、事务式 JSON 跟踪与修改操作、可订阅状态、有界输出、异步 SQLite |
| [`pi_codemode`](packages/codemode/README.md) | `@earendil-works/pi-codemode` | 代码模式：模型编写的 JavaScript 在 QuickJS（与上游相同的构建，运行在 wasmtime 上）中运行，也可改用在 Monty 中运行的 Python，唯一的能力是调用注入的工具；为模型生成 TypeScript（或 Python）声明，并支持可重放的确定性运行 |
| [`pi_telemetry`](packages/telemetry/README.md) | `@earendil-works/pi-telemetry` | 与后端无关的遥测：显式的回调式 span、带类型的 schema、内存参考适配器与一致性测试套件、带 GenAI 语义约定和指标的 OpenTelemetry 适配器；传入上下文后，其他模块会记录 `pi.ai.*`、`pi.agent.*`、`pi.harness.*`、`pi.session.*` 和 `pi.codemode.*` span |

这些模块在 PyPI 上作为同一个包 `pi-ai-python` 发布。

## 安装

```bash
pip install pi-ai-python                  # 依赖 httpx 和官方 SDK：anthropic、openai、google-genai
pip install "pi-ai-python[bedrock]"       # 可选：boto3，用于 Amazon Bedrock
pip install "pi-ai-python[fast]"          # 可选：orjson 加速 JSON 解码
pip install "pi-ai-python[images]"        # 可选：Pillow，用于 pi_durable 的图片处理
pip install "pi-ai-python[codemode]"      # 可选：wasmtime，pi_codemode 的 JavaScript 沙箱（QuickJS）
pip install "pi-ai-python[codemode-python]"  # 可选：pydantic-monty，pi_codemode 的 Python 沙箱（Monty）
pip install "pi-ai-python[otel]"          # 可选：opentelemetry-api，pi_telemetry 的 OpenTelemetry 适配器
pip install "pi-ai-python[all]"           # 可选：orjson + websockets + pydantic + boto3 + pillow + wasmtime + pydantic-monty + opentelemetry-api
```

开发环境：

```bash
uv sync
uv run pytest
```

## 快速开始

```python
import asyncio

from pi_ai import Context, UserMessage
from pi_ai.providers.all import builtin_models


async def main() -> None:
    models = builtin_models()                     # 注册全部内置 provider，鉴权读取 ANTHROPIC_API_KEY 等环境变量
    model = models.get_model("anthropic", "claude-sonnet-4-5")
    stream = models.stream(model, Context(system_prompt="回答要简洁。", messages=[UserMessage("你好！")]))
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
    message = await stream.result()
    print(f"\n输入 {message.usage.input} / 输出 {message.usage.output} tokens，费用 ${message.usage.cost.total:.4f}")


asyncio.run(main())
```

Agent：

```python
from pi_agent import Agent, AgentInitialState, agent_tool


@agent_tool
async def get_weather(city: str) -> str:
    """查询天气。

    Args:
        city: 城市名。
    """
    return f"{city}：晴，25°C"


agent = Agent(
    initial_state=AgentInitialState(system_prompt="你是一个助手。", model=model, tools=[get_weather]),
    stream_fn=models.stream_simple,
)
async for event in agent.events("北京天气怎么样？"):
    if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
        print(event.assistant_message_event.delta, end="")
```

持久化会话把对话记录、进行中的模型调用和工具调用都存进存储。进程崩溃后重新运行同一个脚本，
`resume()` 会从断点继续：

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
    models.set_provider(openai_provider())  # 读取 OPENAI_API_KEY

    storage = await open_sqlite_storage("./session.sqlite")
    harness = await Harness.open(storage, {"models": models, "registry": create_registry()}, context)
    root = await harness.root(context, agent={"model": {"provider": "openai", "modelId": "gpt-6-sol"}})
    harness.resume()  # 继续上一个进程没跑完的工作

    submission = await root.submit({"type": "input", "content": "What is the capital of France?"}, context)
    settled = await submission.wait(context)
    if settled["status"] == "done" and settled["type"] == "input":
        answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
        print(answer["model"][0].content[0].text if answer else None)
    await harness.close(context)


asyncio.run(main())
```

`pi_durable` 的存储格式和 TypeScript 版 pi-durable 一致，一方写的 SQLite 或 JSONL 存储另一方可以直接打开。
已包含编码工具（`pi_durable.tools`：read/write/edit/bash）、工具搜索、基于 Pillow 的图片处理，以及代码模式扩展（`pi_durable.codemode`：脚本默认是由 QuickJS 运行的 JavaScript，也可以是由 Monty 运行的 Python，进程重启后可继续执行）；尚未移植：Cloudflare 适配器。

## 与 pi 保持一致的 API

架构与语义与 TypeScript 版一致：同样的 `Models`/`Provider` 运行时、同样的事件协议
（`start` → `text_delta`… → `done`/`error`）、同样的 transcript 模型（系统提示词和工具由
system message 承载）、同样的鉴权优先级、同样的模型目录和 Agent 循环。命名遵循 Python 习惯：

| TypeScript | Python |
| --- | --- |
| `createModels()`、`models.streamSimple(...)` | `create_models()`、`models.stream_simple(...)` |
| `{ apiKey, sessionId, maxTokens }` | 关键字参数 `api_key=..., session_id=..., max_tokens=...`（也可传 dict） |
| `message.stopReason`、`usage.cacheRead` | `message.stop_reason`、`usage.cache_read` |
| `{ role: "user", content: "hi", timestamp }` | `UserMessage("hi")`（timestamp 默认为当前时间） |
| TypeBox `Type.Object({...})` | 内置 `Type.Object({...})` 或 Pydantic 模型 |
| `AbortController` / `signal` | `pi_ai.AbortController` / `signal=` |
| `agent.continue()` | `agent.continue_()`（别名 `agent.resume()`） |

线上协议和持久化中出现的字符串值保持不变（`"toolUse"`、`"toolResult"`、`"text_delta"`、API id、
provider id）。`pi_ai.serialization` 读写 pi 的 camelCase JSON，会话记录、模型定义和
`auth.json` 凭据文件都可以与 TypeScript 版互通。

## 设计改进

* **使用厂商官方 SDK**：与 pi 一致，Anthropic、OpenAI（Chat Completions、Responses、Azure、OpenRouter 图片）、
  Google（Gemini、Vertex）和 Bedrock 适配器分别使用官方 SDK `anthropic`、`openai`、`google-genai`、`boto3`
  （`pi-ai-python[bedrock]`），各 SDK 客户端共用每个事件循环一个的 `httpx.AsyncClient` 连接池。
  pi 本身不经 SDK 调用的 API（Mistral、OpenAI Codex、pi-messages、分类器）直接使用 `httpx`。
* **性能**：slots 数据类、低分配的事件队列、包装流之间零拷贝转发、增量式部分 JSON 解析
  （大参数工具调用比逐 delta 全量重解析快约 60 倍）、流式文本线性时间累加、可选 orjson、编译并缓存的参数校验器。
  单核每秒处理流事件：anthropic-messages 约 11 万，openai-completions 约 7 万（瓶颈在 openai SDK 的 SSE 解码，见 `benchmarks/`）。
* **全面懒加载**：模型目录按 provider 首次访问时加载；API 实现和 OAuth 流程在首次请求时导入。
* **原生异步取消**：`AbortSignal` 可立即中断挂起的网络读取，被中止的请求以 `stop_reason == "aborted"` 结束而不是抛异常。
* **显式遥测**：`pi_telemetry` 移植了 pi 的回调式 span 约定（没有隐式的“当前 span”）。传入上下文
  （`telemetry_context=` 或 `HarnessOptions["telemetryContext"]`）后，各模块记录 `pi.ai.request`、`pi.agent.run`、
  `pi.harness.*`、`pi.session.commit` 和 `pi.codemode.run` span，默认不记录任何内容（提示词、输出、工具参数等）；
  `pi_telemetry.otel` 按 GenAI 语义约定导出到 OpenTelemetry 并记录指标。不传上下文时什么也不记录，流式路径不变。
  上游 1.3.0 只负责传递上下文。
* **Python 友好的扩展**：Pydantic 模型作为工具参数、`@agent_tool` 从类型注解生成工具、`Agent.events()` 异步迭代器、
  `FileCredentialStore`（pi 的 `auth.json` 格式，带跨进程锁）、事件循环外创建的流、`async with` 关闭流。

详细文档：[pi_ai](packages/ai/README.md)、[pi_agent](packages/agent/README.md)、[pi_telemetry](packages/telemetry/README.md)。

## 许可证

MIT，基于 Mario Zechner 等人的 pi。
