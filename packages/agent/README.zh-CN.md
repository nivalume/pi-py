# pi_agent

基于 [`pi_ai`](../ai/README.md) 构建的有状态 agent：带工具执行与事件流。通过 `pip install pi-ai-python` 安装。这是 `@earendil-works/pi-agent-core` 的 Python 移植版；行为与事件顺序和 TypeScript 包一致。

## 安装

```bash
pip install pi-ai-python             # 同时包含 pi_agent 和 pi_ai
```

## 快速开始

```python
import asyncio

from pi_agent import Agent, AgentInitialState
from pi_ai import create_models
from pi_ai.providers.anthropic import anthropic_provider


async def main() -> None:
    models = create_models()
    models.set_provider(anthropic_provider())
    model = models.get_model("anthropic", "claude-sonnet-4-6")

    agent = Agent(
        initial_state=AgentInitialState(system_prompt="You are a helpful assistant.", model=model),
        stream_fn=models.stream_simple,
    )

    def on_event(event, signal):
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)

    agent.subscribe(on_event)
    await agent.prompt("Hello!")


asyncio.run(main())
```

监听器可以是普通函数或协程；异步监听器按订阅顺序被 await。`Agent.events()` 以异步迭代器的形式暴露同一个流：

```python
async for event in agent.events("Summarize README.md"):
    if event.type == "tool_execution_start":
        print("running", event.tool_name, event.args)
```

## 核心概念

### Agent 消息与 LLM 消息

transcript 保存的是 *agent 消息*：来自 `pi_ai` 的 LLM 消息（`SystemMessage`、`UserMessage`、`AssistantMessage`、`ToolResultMessage`），再加上任何带 `role` 属性的应用特定消息对象（例如一个 dataclass）。LLM 只理解四种标准角色；`convert_to_llm` 在每次请求前弥合这个差距：

```
agent messages → transform_context() → agent messages → convert_to_llm() → LLM messages → provider
                    (optional)                             (default: keep the 4 LLM roles)
```

```python
from dataclasses import dataclass, field
from typing import ClassVar

from pi_ai import UserMessage, now_ms


@dataclass
class Notification:
    role: ClassVar[str] = "notification"
    text: str
    timestamp: int = field(default_factory=now_ms)


def convert_to_llm(messages):
    converted = []
    for message in messages:
        if message.role == "notification":
            continue  # UI only
        converted.append(message)
    return converted


agent = Agent(stream_fn=models.stream_simple, convert_to_llm=convert_to_llm)
```

## 事件流

`prompt("Hello")`：

```
agent_start
turn_start
message_start   (user message)
message_end     (user message)
message_start   (assistant message, partial)
message_update  (streaming chunks, assistant only)
message_end     (assistant message, final)
turn_end        (message, tool_results=[])
agent_end       (messages=[...])
```

带工具调用时，循环以另一轮（turn）继续：

```
... message_end (assistant with tool calls)
tool_execution_start  (tool_call_id, tool_name, args)
tool_execution_update (partial_result)          # if the tool streams progress
tool_execution_end    (result, is_error, duration_ms)
message_start/end     (tool result message)
turn_end              (message, tool_results=[...])
turn_start                                      # next turn: the model sees the results
...
agent_end
```

| 事件 | 字段 | 说明 |
| --- | --- | --- |
| `agent_start` | | 运行开始 |
| `agent_end` | `messages` | 运行的最后一个事件；被 await 的监听器仍计入结算 |
| `turn_start` | | 一次提供商请求及其工具执行 |
| `turn_end` | `message`、`tool_results` | 一轮完成 |
| `message_start` / `message_end` | `message` | 任意消息（system、user、assistant、tool result） |
| `message_update` | `message`、`assistant_message_event` | 仅 assistant 流式输出 |
| `tool_execution_start` | `tool_call_id`、`tool_name`、`args` | 工具开始 |
| `tool_execution_update` | `partial_result` | 工具进度 |
| `tool_execution_end` | `result`、`is_error`、`duration_ms` | 工具完成 |

事件类（`MessageUpdateEvent`、`ToolExecutionEndEvent` 等）是槽位化 dataclass，因此也可以使用结构化模式匹配：

```python
from pi_agent import MessageUpdateEvent, ToolExecutionEndEvent
from pi_ai import TextDeltaEvent

match event:
    case MessageUpdateEvent(assistant_message_event=TextDeltaEvent(delta=delta)):
        print(delta, end="")
    case ToolExecutionEndEvent(tool_name=name, is_error=True):
        print(f"{name} failed")
```

### 工具执行模式

- `"parallel"`（默认）：工具调用按顺序预检，允许的工具并发运行，每个工具完成时发出 `tool_execution_end`，工具结果消息以及 `turn_end.tool_results` 按 assistant 源顺序排列。
- `"sequential"`：逐个执行。

用 `Agent(tool_execution="sequential")` 设置，或按工具用 `execution_mode="sequential"` 设置（批次中任一顺序工具都会使整个批次变为顺序执行）。

### Hooks

```python
from pi_agent import AfterToolCallResult, BeforeToolCallResult


async def before_tool_call(ctx, signal):
    if ctx.tool_call.name == "bash":
        return BeforeToolCallResult(block=True, reason="bash is disabled", terminate=True)


async def after_tool_call(ctx, signal):
    if ctx.tool_call.name == "notify_done" and not ctx.is_error:
        return AfterToolCallResult(terminate=True)
    if not ctx.is_error:
        return AfterToolCallResult(details={**(ctx.result.details or {}), "audited": True})


agent = Agent(stream_fn=models.stream_simple, before_tool_call=before_tool_call, after_tool_call=after_tool_call)
```

- `before_tool_call` 在 `tool_execution_start` 与参数校验之后运行，可以阻止执行。
- `after_tool_call` 在执行之后、`tool_execution_end` 之前运行；省略的字段保持原值。
- `terminate=True`（来自工具结果、被阻止的调用或覆盖值）只在批次中*每一个*已定稿的结果都终止时，才跳过自动的后续请求。
- 使用 `Agent` 类时，assistant 的 `message_end` 处理是工具预检前的屏障，因此 `before_tool_call` 看到的 state 已经包含 assistant 消息。

### 请求准备与一轮收尾

```python
from pi_agent import AgentContext, AgentRequestUpdate, AgentTurnDecision


async def prepare_request(request, signal):
    # Install canonical persisted context right before every provider request.
    return AgentRequestUpdate(context=AgentContext(messages=await session.load(), tools=request.context.tools))


async def finish_turn(turn, signal):
    if turn.message.stop_reason in ("error", "aborted"):
        return None
    if should_end(turn.message):
        return AgentTurnDecision("end")
    return AgentTurnDecision("continue") if needs_more(turn.message) else None


agent.prepare_request = prepare_request
agent.finish_turn = finish_turn
```

`finish_turn` 也可以返回字符串 `"continue"`/`"end"` 或 `{"action": ...}`。`"end"` 在 `turn_end` 之后停止，不轮询队列。`"continue"` 保证再多发一次请求；已经排程请求的工具结果、引导或跟进可以满足它。出错和中止的响应总是结束运行。每个提供商轮的生命周期：

```
selected input events → prepare_request → provider response → tool results → finish_turn → turn_end → next turn or agent_end
```

### `continue_()` 与排队输入

`continue_()`（别名 `resume()`；`continue` 是 Python 关键字）从现有 transcript 继续。空 transcript 和只有系统消息的 transcript 会抛异常。结尾不是 assistant 时会重试；结尾是 assistant 时回退到一个排队的引导批次，再一个排队的跟进批次。

## Agent 选项

```python
agent = Agent(
    # systemPrompt and tools become the leading system message unless messages already start with one
    initial_state=AgentInitialState(
        system_prompt="...",
        model=model,
        thinking_level="medium",          # "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
        tools=[read_file],
        messages=[],
    ),
    stream_fn=models.stream_simple,       # required (or set_default_stream_fn); a Models instance also works
    convert_to_llm=convert_to_llm,        # agent messages → LLM messages
    transform_context=prune,              # async (messages, signal) -> messages
    steering_mode="one-at-a-time",        # or "all"
    follow_up_mode="one-at-a-time",
    session_id="session-123",             # provider prompt caching
    get_api_key=refresh_token,            # (provider) -> key, for expiring OAuth tokens
    tool_execution="parallel",
    before_tool_call=before_tool_call,
    after_tool_call=after_tool_call,
    prepare_request=prepare_request,
    finish_turn=finish_turn,
    thinking_budgets={"minimal": 128, "low": 512, "medium": 1024, "high": 2048},
    stream_options={"temperature": 0.2, "max_tokens": 4096},   # any other SimpleStreamOptions
    telemetry_context=telemetry,          # pi_telemetry context: one pi.agent.run span per run (see Telemetry)
)
```

## Agent 状态

```python
agent.state.model            # active model for future turns (assignable)
agent.state.thinking_level   # assignable
agent.state.tools            # assigning copies the list
agent.state.messages         # assigning copies the list; appending mutates state
agent.state.system_prompt    # read-only, replayed from the transcript's system messages
agent.state.is_streaming     # True until awaited agent_end listeners settle
agent.state.streaming_message
agent.state.pending_tool_calls
agent.state.error_message
```

transcript 拥有系统提示与工具声明。每次请求前，循环会把 `state.tools` 与 transcript 声明的工具做差异比较，并在一条系统消息中公布这一变化。要在对话中途修改提示，添加一条系统消息：

```python
from pi_ai import SystemMessage, UserMessage

await agent.prompt([
    SystemMessage("", sections={"skills": "<skills>...</skills>"}),
    UserMessage("Continue"),
])
```

## 方法

```python
await agent.prompt("Hello")
await agent.prompt("What's in this image?", [ImageContent(data, "image/jpeg")])
await agent.prompt(UserMessage("Hello"))
await agent.continue_()

agent.steer(UserMessage("Stop! Do this instead."))   # injected after the current turn's tools finish
agent.follow_up(UserMessage("Also summarize."))       # runs when the agent would otherwise stop
agent.peek_queued_messages()
agent.clear_steering_queue(); agent.clear_follow_up_queue(); agent.clear_all_queues()

agent.abort()
await agent.wait_for_idle()
agent.reset()
unsubscribe = agent.subscribe(listener)
```

## 工具

```python
from pathlib import Path

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent, Type


async def read_file(tool_call_id, params, signal=None, on_update=None):
    if on_update:
        on_update(AgentToolResult([TextContent("Reading...")]))
    text = Path(params["path"]).read_text()
    return AgentToolResult([TextContent(text)], {"path": params["path"], "size": len(text)})


read_file_tool = AgentTool(
    "read_file",
    "Read a file's contents",
    Type.Object({"path": Type.String(description="File path")}),
    execute=read_file,
    label="Read File",
    execution_mode="sequential",
)
agent.state.tools = [read_file_tool]
```

工具失败时抛出异常（或返回 `AgentToolResult(..., is_error=True)`）；错误文本作为错误工具结果报告给模型。

### 从函数生成工具

`@agent_tool` 从类型提示与 docstring 构建 schema：

```python
from enum import Enum
from typing import Literal

from pi_agent import agent_tool


@agent_tool(execution_mode="sequential")
async def search(query: str, limit: int = 10, sort: Literal["relevance", "date"] = "relevance", signal=None) -> list[dict]:
    """Search the knowledge base.

    Args:
        query: Search terms.
        limit: Maximum number of results.
    """
    return await kb.search(query, limit=limit, sort=sort)
```

支持的注解：`str`、`int`、`float`、`bool`、`list[...]`、`dict[...]`、`Literal`、`Enum`、`X | None`、`Annotated[..., "description"]`、dataclass 和 Pydantic 模型（单个 Pydantic 参数会成为整个 schema，并以校验后的形式传入）。`tool_call_id`、`signal` 和 `on_update` 参数会被注入。`str` 结果变成文本内容，其他值以 JSON 发送。同步函数在工作线程中运行。

## 代理（Proxy）用法

对于把 LLM 调用路由到 pi 代理服务器的应用：

```python
from pi_agent import Agent, stream_proxy

agent = Agent(
    stream_fn=lambda model, context, options: stream_proxy(
        model, context, {**(options or {}), "auth_token": token, "proxy_url": "https://your-server.com"}
    ),
)
```

## 底层 API

```python
from pi_agent import AgentContext, AgentLoopConfig, agent_loop, agent_loop_continue
from pi_ai import SystemMessage, UserMessage

context = AgentContext(messages=[SystemMessage("You are helpful.")], tools=[])
config = AgentLoopConfig(
    model=model,
    convert_to_llm=lambda messages: [m for m in messages if m.role in ("system", "user", "assistant", "toolResult")],
    tool_execution="parallel",
)

async for event in agent_loop([UserMessage("Hello")], context, config, None, models.stream_simple):
    print(event.type)

async for event in agent_loop_continue(context, config, None, models.stream_simple):
    print(event.type)
```

这些底层流是观察性的：它们保留事件顺序，但不会等你处理完事件再继续后面的循环阶段。当消息处理必须是工具预检前的屏障时，请使用 `Agent`。`run_tool_call()` 用相同的准备、校验和 hooks 运行一次工具调用，但不发出事件。

## 遥测

`Agent(telemetry_context=...)`（或 `AgentLoopConfig.telemetry_context`）接受一个 [`pi_telemetry`](../telemetry/README.md) 上下文；这是新增能力，上游 1.3.0 的 agent-core 不记录 span。此后每次运行记录一个 `pi.agent.run` span：开始时记录模型、工具数量与模式；结束时记录轮次、工具调用、token、成本与结束原因（`completed`、`ended`、`error`、`aborted`、`failed`）；排队的消息加入运行时触发 `pi.agent.steering` 与 `pi.agent.follow_up` 事件。循环把运行 span 作为 `telemetry_context` 传给流函数，因此使用 `models.stream_simple` 时每个模型请求都是 `pi.ai.request` 子 span。

```python
from pi_agent import Agent, AgentInitialState
from pi_ai import create_models, faux_assistant_message, faux_provider
from pi_telemetry import InMemoryTelemetryContext

faux = faux_provider()
models = create_models()
models.set_provider(faux.provider)
faux.set_responses([faux_assistant_message("Hello!")])

telemetry = InMemoryTelemetryContext()
agent = Agent(initial_state=AgentInitialState(model=faux.get_model()), stream_fn=models, telemetry_context=telemetry)
await agent.prompt("Hi")
spans = telemetry.get_spans()
names = {span.id: span.name for span in spans}
print([(span.name, names.get(span.parent_id)) for span in spans])
print(spans[0].attributes["pi.agent.end_reason"], spans[0].attributes["pi.agent.turns"])
```

不传上下文时不记录任何内容，请求也拿不到 `telemetry_context`。span 只记录计数与用量，从不记录消息；schema 是 `pi_agent.telemetry.AGENT_TELEMETRY_SCHEMA`。

## 许可证

MIT
