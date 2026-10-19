# pi_ai

统一的 LLM API：内置提供商集合、自动鉴权解析、token 与成本追踪，以及简单的上下文持久化与会话中途切换到其他模型的无缝交接。这是 `@earendil-works/pi-ai` 的 Python 移植版：相同的架构、相同的事件协议、相同的模型目录，采用 Python 风格命名。

**注意**：聊天目录只包含支持工具调用的模型，因为这对 agentic 工作流至关重要。图像与分类器目录则使用各自的能力。

## 目录

- [支持的提供商](#支持的提供商)
- [安装](#安装)
- [快速开始](#快速开始)
- [提供商与模型](#提供商与模型)
- [鉴权](#鉴权)
- [工具](#工具)
- [图像输入](#图像输入)
- [图像生成](#图像生成)
- [分类](#分类)
- [思考 / 推理](#思考--推理)
- [停止原因](#停止原因)
- [错误处理与取消](#错误处理与取消)
- [自定义提供商](#自定义提供商)
- [用于测试的 Faux 提供商](#用于测试的-faux-提供商)
- [跨提供商交接](#跨提供商交接)
- [系统消息](#系统消息)
- [上下文序列化](#上下文序列化)
- [OAuth 提供商](#oauth-提供商)
- [旧版全局 API](#旧版全局-api)
- [遥测](#遥测)
- [性能说明](#性能说明)
- [与 TypeScript 包的差异](#与-typescript-包的差异)

## 支持的提供商

OpenAI（含 Decisions 分类器 API）、Ant Ling、Azure OpenAI（Responses）、OpenAI Codex（ChatGPT Plus/Pro，OAuth）、Radius（API key 或 OAuth，动态目录）、TypeSafe（System One 分类器）、DeepSeek、NVIDIA NIM、Anthropic、Google、Vertex AI、Mistral、Groq、Cerebras、Cloudflare AI Gateway、Cloudflare Workers AI、xAI、OpenRouter、Vercel AI Gateway、ZAI Coding Plan（全球与中国区）、MiniMax（全球与中国区）、Together AI、Baseten、Hugging Face、Moonshot AI（全球与中国区）、GitHub Copilot（OAuth）、Amazon Bedrock、OpenCode Zen、OpenCode Go、Fireworks、Kimi For Coding、Meta、Qwen Token Plan（个人版、现有用户与中国区）、Xiaomi MiMo（API 计费与 Token Plan），以及任何 OpenAI 兼容 API（Ollama、vLLM、LM Studio、llama.cpp 等）。

## 安装

```bash
pip install pi-ai-python               # pi_ai + pi_agent；httpx + 官方 SDK：anthropic、openai、google-genai
pip install "pi-ai-python[bedrock]"    # 增加 boto3（用于 Amazon Bedrock）
pip install "pi-ai-python[fast]"       # 增加 orjson，加快流式解码
pip install "pi-ai-python[websocket]"  # 增加 websockets（用于 OpenAI Codex WebSocket 传输）
pip install "pi-ai-python[pydantic]"   # 增加 pydantic，可用模型类作为工具参数 schema
pip install "pi-ai-python[all]"
```

要求 Python 3.11+。所有内容都基于 `asyncio`。

## 快速开始

```python
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from pi_ai import Context, TextContent, Tool, ToolResultMessage, Type, UserMessage
from pi_ai.providers.all import builtin_models


async def main() -> None:
    # 一个注册了全部内置提供商的 Models 集合
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

    # 方式一：带全部事件类型的流式输出。鉴权通过提供商解析
    # （此处从环境变量读取 OPENAI_API_KEY）。
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

    # 处理工具调用
    for call in (block for block in final.content if block.type == "toolCall"):
        zone = ZoneInfo(call.arguments.get("timezone") or "UTC")
        context.messages.append(
            ToolResultMessage(call.id, call.name, [TextContent(datetime.now(zone).isoformat())])
        )

    # 方式二：不做流式，直接获取完整响应
    if final.stop_reason == "toolUse":
        continuation = await models.complete(model, context)
        context.messages.append(continuation)
        print(continuation.content)

    print(f"Tokens: {final.usage.input} in, {final.usage.output} out; cost ${final.usage.cost.total:.4f}")


asyncio.run(main())
```

选项既可以关键字参数的形式传入，也可以 dict 的形式传入（两种风格会合并）：

```python
await models.complete(model, context, max_tokens=512, temperature=0.2, session_id="s1")
await models.complete(model, context, {"max_tokens": 512})
```

## 提供商与模型

**提供商**（provider）是运行时单元：它拥有自己的模型目录、鉴权（API key 解析、OAuth 流程）和流式行为。一个 `Models` 集合持有若干提供商，并把每个请求路由到拥有该模型的提供商。提供商之间共享 **API 实现**（线上协议）：Anthropic 模型使用 `anthropic-messages`，OpenAI 使用 `openai-responses`，其余大多数共享 `openai-completions`。混合 API 的提供商（GitHub Copilot、OpenCode Zen）按模型分发。

### 提供商工厂

每个内置提供商一个工厂；每个工厂只加载自己的目录：

```python
from pi_ai import create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.openrouter import openrouter_provider

models = create_models()
models.set_provider(anthropic_provider())
models.set_provider(openrouter_provider())
```

`pi_ai.providers.all.builtin_models()` 会注册全部内置提供商，并接受与 `create_models()` 相同的选项（`credentials`、`models_store`、`auth_context`）；`builtin_providers()` 返回提供商列表。

### 查询模型

读取是同步的，返回最近一次已知的列表：

```python
models.get_providers()
models.get_provider("anthropic")
models.get_models()                      # 全部聊天模型
models.get_models("anthropic")
model = models.get_model("anthropic", "claude-sonnet-4-5")
print(model.id, model.api, model.context_window, "image" in model.input, model.reasoning)

models.get_models_of_type("image", "openrouter")
models.get_model_of_type("classifier", "typesafe", "jev-latest")
await models.get_available_of_type("image")   # 只返回已配置鉴权的提供商
models.get_all_models()                       # 全部类型
```

模型的 `type`（`"chat"`、`"image"`、`"classifier"`）决定哪个操作接受它。在混合列表上使用 `is_model_type(model, "image")` / `get_model_type(model)`，并在传入 API 专属选项之前使用 `has_api(model, "anthropic-messages")`。

### 静态目录读取

```python
from pi_ai.providers.all import (
    get_builtin_model, get_builtin_models, get_builtin_image_model,
    get_builtin_classifier_model, get_builtin_providers, get_all_builtin_models,
)

get_builtin_model("openai", "gpt-4o-mini")
get_builtin_image_model("openrouter", "black-forest-labs/flux.2-pro")
```

### 动态提供商

```python
await models.refresh(providers=["llamacpp"])   # 单个提供商
result = await models.refresh()                # 全部，并发执行，尽力而为
if result.aborted:
    ...
for provider_id, error in result.errors.items():
    print(provider_id, error)
```

`refresh(allow_network=False)` 可在离线时恢复已持久化的目录；`force=True` 跳过新鲜度检查。静态提供商是空操作。Radius 附带一份静态目录，并在配置好鉴权后刷新时叠加其 `/v1/config` 网关目录。

## 鉴权

### 鉴权如何解析

`models.stream()` 通过所属提供商解析鉴权，并将其合并进请求。请求中显式给出的值总是优先：

```python
await models.complete(model, context)                      # 环境变量、已存储的凭据或 OAuth token
await models.complete(model, context, api_key="sk-explicit")

auth = await models.get_auth(model)          # 或 models.get_auth("anthropic")
if auth:
    print(auth.source, auth.auth.headers)    # 例如 "ANTHROPIC_API_KEY"
```

`get_auth()` 对未配置的提供商返回 `None`，在出现问题（`code == "oauth"`：token 刷新失败且凭据被保留以便重新登录；`code == "auth"`：key 解析或凭据存储失败）时抛出 `ModelsError`。请求路径会把同样的失败表现为流错误。

### 转换请求头

```python
import uuid

await models.complete_simple(
    model,
    context,
    headers={"X-Client": "my-app"},
    transform_headers=lambda headers: {**headers, "X-Request-ID": str(uuid.uuid4())},
)
```

顺序：提供商鉴权头 → `model.headers` → 显式 `headers` → `transform_headers` → 提供商。头名称按大小写不敏感方式合并；值为 `None` 时抑制对应默认头。

### 凭据存储

存储的凭据（交互式输入的 key、OAuth token）存放在 `CredentialStore` 中，每个提供商一个凭据。默认在内存中；`FileCredentialStore` 使用 pi 的 `auth.json` 格式（原子写入、权限 `0600`、POSIX 上的跨进程文件锁）：

```python
from pi_ai import FileCredentialStore, create_models

models = create_models(credentials=FileCredentialStore("~/.config/my-app/auth.json"))
```

自定义存储实现异步的 `read`、`list`、`modify(provider_id, fn)`（唯一的写路径，串行化的读-改-写；OAuth 刷新在它内部执行）和 `delete`。凭据是普通 dict：

```python
{"type": "api_key", "key": "...", "env": {"CLOUDFLARE_ACCOUNT_ID": "...", "CLOUDFLARE_GATEWAY_ID": "..."}}
{"type": "oauth", "access": "...", "refresh": "...", "expires": 1767225600000}
```

已存储的凭据拥有对应提供商：只有当没有存储任何内容时才查询环境变量，刷新失败也绝不会回退到环境变量里的 key。

### 环境变量

| 提供商 | 变量 |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY`、`ANTHROPIC_OAUTH_TOKEN` 或 `ANTHROPIC_AUTH_TOKEN`（bearer） |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_API_KEY`，或 `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` + ADC |
| Amazon Bedrock | `AWS_PROFILE`、`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`AWS_BEARER_TOKEN_BEDROCK`、ECS/IRSA 凭据 |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` 或 `AZURE_OPENAI_RESOURCE_NAME` |
| Mistral / Groq / Cerebras / xAI | `MISTRAL_API_KEY` / `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `XAI_API_KEY` |
| OpenRouter / Vercel AI Gateway | `OPENROUTER_API_KEY` / `AI_GATEWAY_API_KEY` |
| Cloudflare | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`（AI Gateway 还需要 `CLOUDFLARE_GATEWAY_ID`） |
| DeepSeek / NVIDIA / Together / Baseten / Fireworks | `DEEPSEEK_API_KEY` / `NVIDIA_API_KEY` / `TOGETHER_API_KEY` / `BASETEN_API_KEY` / `FIREWORKS_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| ZAI / MiniMax / Moonshot / Kimi | `ZAI_API_KEY`、`ZAI_CODING_CN_API_KEY` / `MINIMAX_API_KEY`、`MINIMAX_CN_API_KEY` / `MOONSHOT_API_KEY` / `KIMI_API_KEY` |
| OpenCode | `OPENCODE_API_KEY` |
| Meta、Radius、TypeSafe、Ant Ling | `META_API_KEY`、`RADIUS_API_KEY`、`TYPESAFE_API_KEY`、`ANT_LING_API_KEY` |
| Qwen Token Plan | `QWEN_TOKEN_PLAN_API_KEY`、`QWEN_TOKEN_PLAN_CN_API_KEY` |
| Xiaomi MiMo | `XIAOMI_API_KEY`、`XIAOMI_TOKEN_PLAN_{CN,AMS,SGP}_API_KEY` |
| GitHub Copilot | `COPILOT_GITHUB_TOKEN` |

### 提供商作用域的环境变量覆盖

传入 `env` 可以把提供商配置限定到单个请求（账号 id、Azure 设置、Vertex project/location、`PI_CACHE_RETENTION`、`HTTP(S)_PROXY`）。这些值优先于 `os.environ`：

```python
await models.complete(model, context, env={
    "CLOUDFLARE_API_KEY": "...", "CLOUDFLARE_ACCOUNT_ID": "account-id", "CLOUDFLARE_GATEWAY_ID": "gateway-id",
})
```

## 工具

### 定义工具

工具参数是 JSON Schema。可以用 TypeBox 风格的 `Type` 辅助函数来构建，也可以手写 dict，或直接传入一个 Pydantic 模型类：

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


search = Tool("search", "Search the docs", Search)   # 校验后的参数是 Search 实例
```

为兼容 Google（Google 拒绝 `anyOf`/`const` 枚举）请使用 `StringEnum` 而不是 `Type.Enum`。`Type.Optional(...)` 把对象属性标记为非必填。

### 受限采样（constrained sampling）

```python
strict_tool = Tool(
    "edit_file", "Edit a file",
    Type.Object({"path": Type.String(), "content": Type.String()}, additional_properties=False),
    constrained_sampling={"type": "json_schema", "strict": "prefer"},   # 或 "require"
)

patch_tool = Tool(
    "apply_patch", "Apply a patch",
    Type.Object({"input": Type.String()}, additional_properties=False),
    constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/s"}},
)
```

`"prefer"` 在提供商支持时使用服务端严格 schema，否则回退；`"require"` 在提供商/模型无法满足时直接让请求失败。OpenAI、Anthropic、具备能力的 Bedrock 模型、Mistral 和 Gemini 3 支持严格 JSON schema；在能把 OpenAI 自定义工具透传的端点上，GPT-5+ 支持 grammar 工具。

### 处理工具调用

```python
response = await models.complete(model, context)
for block in response.content:
    if block.type == "toolCall":
        result = await execute_weather_api(block.arguments)
        context.messages.append(
            ToolResultMessage(block.id, block.name, [TextContent(json.dumps(result))])
        )

# 工具结果可以为视觉模型包含图片
context.messages.append(ToolResultMessage(
    "tool_xyz", "generate_chart",
    [TextContent("Generated chart"), ImageContent(base64.b64encode(png).decode(), "image/png")],
))
```

### 带部分 JSON 的流式工具调用

在 `toolcall_delta` 事件期间，`event.partial.content[event.content_index].arguments` 保存对到目前为止的参数所做的尽力解析（至少是 `{}`）；字段可能缺失或被截断。`toolcall_end` 携带完整的（尚未校验的）调用。当最终参数文本不是合法的 JSON 对象时，`tool_call.arguments` 为 `{}`，`tool_call.arguments_error` 描述具体问题。

### 校验工具参数

```python
from pi_ai import ToolValidationError, validate_tool_call

try:
    args = validate_tool_call(tools, tool_call)        # 与 pi 一致，会把 "42" 强转为 42
except ToolValidationError as error:
    context.messages.append(ToolResultMessage(tool_call.id, tool_call.name, [TextContent(str(error))], True))
```

校验会把每个 schema 各编译一次并缓存校验器，且会报告每个问题及其路径（如 `  - attendees: must NOT have fewer than 1 items`）。

### 事件参考

成功的生成为 `start → updates* → done`；start 之后的失败为 `start → updates* → error`；设置阶段的失败只发出 `error`。

| 事件 | 说明 | 关键字段 |
| --- | --- | --- |
| `start` | 流开始 | `partial` |
| `text_start` / `text_delta` / `text_end` | 文本块 | `content_index`、`delta`、`content` |
| `thinking_start` / `thinking_delta` / `thinking_end` | 思考块 | `content_index`、`delta`、`content` |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | 工具调用 | `content_index`、`delta`、`tool_call` |
| `done` | 完成 | `reason`（`"stop"`、`"length"`、`"toolUse"`、`"deferred"`）、`message` |
| `error` | 失败或中止 | `reason`（`"error"`、`"aborted"`）、`error`（带部分内容的消息） |

`partial` 是共享的活动消息而非快照：处理事件时请检查它。不同块的事件可以交错出现；用 `content_index` 关联它们。所有事件都是槽位化 dataclass（`TextDeltaEvent`、`ToolCallEndEvent` 等），所以 `match event: case TextDeltaEvent(delta=d): ...` 也能用。

## 图像输入

```python
if "image" in model.input:
    response = await models.complete(model, Context(messages=[UserMessage([
        TextContent("What is in this image?"),
        ImageContent(base64.b64encode(Path("image.png").read_bytes()).decode(), "image/png"),
    ])]))
```

发给非视觉模型的图片会被替换为占位符。

## 图像生成

```python
from pi_ai import ImagesContext, TextContent

model = models.get_model_of_type("image", "openrouter", "google/gemini-2.5-flash-image")
result = await models.generate_images(model, ImagesContext([TextContent("A red circle on white.")]))
for block in result.output:
    print(block.type, getattr(block, "mime_type", None))
```

`generate_images()` 从不抛异常；失败返回 `stop_reason == "error"`。自定义提供商通过 `create_provider(images={"<api>": impl})` 声明图像支持，其中 `impl` 有异步方法 `generate_images(model, context, options)`。

## 分类

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

分类器 API：TypeSafe/System One（`typesafe-system-one`，也可经由 OpenRouter、Vercel、OpenCode）、Cloudflare Workers AI（`cloudflare-workers-ai-system-one`）、OpenAI Decisions（`openai-decisions`，GPT-6 Luna），以及 `llama-cpp-classify`——它利用 next-token 对数概率，把由 llama.cpp 服务的聊天模型变成分类器。`classify()` 从不抛异常；在支持的地方，`options={"temperature": 2}` 可以软化过度自信的分布。

## 思考 / 推理

### 统一接口

```python
response = await models.complete_simple(model, context, reasoning="medium")
for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
```

级别：`"minimal"`、`"low"`、`"medium"`、`"high"`，以及模型特定的可选级别 `"xhigh"`/`"max"`。`get_supported_thinking_levels(model)` 列出模型暴露的级别，`clamp_thinking_level(model, level)` 选取最近的一个受支持级别。`thinking_budgets` 为基于 token 的提供商覆盖 token 预算。

### 提供商专属选项

`stream()`/`complete()` 接受所属 API 的完整选项集：

```python
if has_api(model, "openai-responses"):
    await models.complete(model, context, reasoning_effort="medium", reasoning_summary="detailed")

if has_api(model, "anthropic-messages"):
    await models.complete(model, context, thinking_enabled=True, thinking_budget_tokens=8192)

if has_api(model, "google-generative-ai"):
    await models.complete(model, context, thinking={"enabled": True, "budget_tokens": 8192})
```

## 停止原因

`"stop"`、`"length"`、`"toolUse"`、`"error"`、`"aborted"`、`"deferred"` 和 `"pending"`（仅部分消息）。API 暴露时，`message.response_id` 保存上游响应 id；`message.raw_stop_reason` 保存提供商自己的值。

## 错误处理与取消

一旦返回流，请求失败就不会再抛异常：流以 `error` 事件结束，最终消息携带 `error_message`、部分内容和部分用量。缺少必需鉴权时，直接调用 API 的 `stream_simple()` 会同步抛异常。

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
    context.messages.append(response)                      # 被中止的消息可以继续
    context.messages.append(UserMessage("Please continue"))
```

`abort()` 可从任意线程调用。中止会立即中断挂起的网络读取。你也可以停止消费并用 `await stream.aclose()` 取消请求（或 `async with models.stream(...) as stream:`）。

`is_context_overflow(message, context_window)` 跨提供商检测上下文溢出（错误消息、静默溢出、被截断的 length 停止）；`is_retryable_assistant_error` 与 `retry_assistant_call(produce, RetryPolicy(...), signal)` 实现 pi 的重试策略。请求也接受 `max_retries`、`max_retry_delay_ms` 和 `timeout_ms`（HTTP 重试遵循 `retry-after`；Bedrock 把 `max_retries` 映射到 botocore 的重试配置）。

### 调试 payload 与原始事件

```python
await models.complete(model, context, on_payload=lambda payload, model: print(json.dumps(payload, indent=2)))
await models.complete(model, context, on_response=lambda response, model: print(response.status, response.headers))
await models.complete(model, context, on_provider_stream_event=lambda chunk, model: print(chunk))
```

`on_payload` 可以返回一个替换用的 payload。回调可以是同步或异步的，并按流顺序被 await。

## 自定义提供商

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
    auth=ProviderAuth(api_key=keyless_auth("Ollama")),   # 无 key 的本地服务
    models=[ollama_model],
    api=openai_completions_api(),
)

models = create_models()
models.set_provider(ollama)
await models.complete(models.get_model("ollama", "llama-3.1-8b"), context)
```

需要 key 的提供商使用 `env_api_key_auth("My proxy API key", ["MY_PROXY_API_KEY"])`（已存储凭据优先，然后是第一个已设置的环境变量）。混合 API 提供商传入一个 dict：`api={"anthropic-messages": anthropic_messages_api(), "openai-responses": openai_responses_api()}`。动态目录使用 `fetch_models=async (context) -> [models]`；`create_provider` 会通过 `ModelsStore` 持久化它们，并在 `refresh(allow_network=False)` 时恢复。`models=` 也可以是一个零参 callable，首次访问时求值。

`compat` 是保存 pi 的 `models.json` 键（camelCase）的普通 dict，因此现有 pi 模型配置可以直接粘贴使用。未设置的字段会根据 base URL 自动检测。`thinking_level_map` 把 pi 级别映射到提供商值（`None` 表示该级别不受支持）。

### 直接调用 API 实现

```python
from pi_ai import normalize_context
from pi_ai.api.anthropic_messages import stream

s = stream(claude_model, normalize_context(context), {"api_key": os.environ["ANTHROPIC_API_KEY"], "thinking_enabled": True})
```

| API id | 模块 | 选项类型 |
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

`pi_ai.api.registry` 提供懒加载包装器（`anthropic_messages_api()` 等），在首次请求时才导入模块。

### 使用自定义 HTTP 客户端

每个请求都接受 `http_client=httpx.AsyncClient(...)`（对应 pi 的 `fetch` 选项）：适用于自定义 TLS、传输，或测试中的 `httpx.MockTransport`。它会交给厂商 SDK（`anthropic`/`openai` 用 `http_client=`，`google-genai` 用 `HttpOptions(httpx_async_client=...)`）。不传时，请求按事件循环共享一个连接池客户端；`await pi_ai.aclose_http_clients()` 会关闭它。Bedrock 走 boto3，不使用 `http_client`。

## 用于测试的 Faux 提供商

```python
from pi_ai import (
    Context, TextContent, ToolResultMessage, UserMessage, create_models,
    faux_assistant_message, faux_provider, faux_text, faux_thinking, faux_tool_call,
)

faux = faux_provider(tokens_per_second=50)   # 限速是可选的
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

响应按请求顺序消费；队列为空时请求失败并报 `"No more faux responses queued"`。响应可以是工厂函数 `(context, options, state, model) -> AssistantMessage`（同步或异步）。用量按大约 4 个字符/token 估算，提示缓存按 `session_id` 模拟，工具调用参数以 `toolcall_delta` 分块流式输出，也支持延迟响应（`complete_simple(..., deferred=True)` + `fetch_deferred`）。

## 跨提供商交接

上下文可以在提供商之间自由移动。来自同一提供商/API/模型的助手消息按原样重放（包括思考与签名）；来自其他模型的消息保留文本和工具调用、丢弃思考，并把工具调用 id 规范化为目标 API 的格式。出错和被中止的轮次会被跳过，孤立的工具调用得到合成错误结果。

```python
context = Context(messages=[UserMessage("What is 25 * 18?")])
context.messages.append(await models.complete_simple(models.get_model("anthropic", "claude-sonnet-4-5"), context, reasoning="medium"))
context.messages.append(UserMessage("Is that correct?"))
context.messages.append(await models.complete(models.get_model("openai", "gpt-5-mini"), context))
```

## 系统消息

`Context.system_prompt` 和 `Context.tools` 是一条前置系统消息的简写。`normalize_context()` 只把它们折叠一次；提供商收到的是 `TranscriptContext`，其系统消息携带提示和工具。后续的系统消息会修改它们而不重写历史：

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

支持 `supportsMidConvoSystemMessages` 的模型会原地接收后续系统消息，因此缓存过的提示前缀得以保留；其他模型则把重放后的提示作为前置系统消息。

## 上下文序列化

上下文、消息和模型都是数据类；`pi_ai.serialization` 在它们与 pi 的 camelCase JSON（TypeScript 版 pi 持久化所用的格式）之间互转：

```python
from pi_ai import Context
from pi_ai.serialization import dumps, loads, to_dict, from_dict

text = dumps(context)              # 与 pi 兼容的 JSON
restored = loads(text, Context)
restored.messages.append(UserMessage("Tell me more"))
await models.complete(models.get_model("anthropic", "claude-3-5-haiku-20241022"), restored)

model_json = to_dict(model)        # {"id": ..., "baseUrl": ..., "contextWindow": ..., "type": "chat"}
```

`normalize_context()` 和 `Agent.prompt()` 也接受 pi 格式的 dict 消息。

## OAuth 提供商

Anthropic（Claude Pro/Max）、OpenAI（Sign in with ChatGPT）、OpenAI Codex（旧版）、GitHub Copilot、OpenRouter（OAuth PKCE 铸 key）、Kimi For Coding、xAI、Meta 和 Radius 在 `provider.auth.oauth` 上携带 `OAuthAuth`。`Models` 驱动登录并持久化凭据；刷新自动进行，并通过凭据存储加锁。

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

### CLI 登录

```bash
pi-ai login              # 交互式选择提供商
pi-ai login anthropic
pi-ai list
```

凭据保存在当前目录的 `auth.json` 中。

### Vertex AI

API key（`GOOGLE_CLOUD_API_KEY` 或 `api_key=`），或使用 Application Default Credentials：本地 `gcloud auth application-default login`，或通过 `GOOGLE_APPLICATION_CREDENTIALS` 指定服务账号文件，再加上 `GOOGLE_CLOUD_PROJECT` 和 `GOOGLE_CLOUD_LOCATION`。`google-genai` SDK 通过 `google-auth` 解析 ADC。

### Amazon Bedrock

需要 `pip install "pi-ai-python[bedrock]"`（boto3）。请求经由 boto3 的 `bedrock-runtime` 客户端，因此凭据、区域和端点通过 botocore 的标准链解析（环境变量 key、`~/.aws` 配置、SSO、ECS/EC2 角色、web identity），外加 `AWS_BEARER_TOKEN_BEDROCK` 或 `credentials=` 选项。boto3 是同步的；流在工作线程上读取，abort 会关闭它。

## 旧版全局 API

`pi_ai.compat` 保留 pi 的旧版全局 API（`get_model()`、按 `model.api` 分发并注入环境 key 的 `stream()`/`complete()`、`register_api_provider()`、`get_env_api_key()`、`register_faux_provider()`、`get_image_model()`/`generate_images()`）：

```python
from pi_ai.compat import complete, get_model
```

新代码应使用 `create_models()` 和提供商工厂。

## 遥测

每个请求选项面都接受 `telemetry_context`，即一个 [`pi_telemetry`](../telemetry/README.md) 上下文：请求所产生遥测的显式父节点。与上游一致，`build_base_options()` 会把它传递下去，因此每个提供商都会在 options 中收到它。

在上游之外，`Models`（以及 `pi_ai.compat`）为每个聊天请求记录一个 `pi.ai.request` span：`stream`、`stream_simple`、`complete`、`complete_simple`、`stream_deferred` 和 `fetch_deferred`，无论由哪个提供商服务。它携带提供商、API、模型、操作、推理级别、工具与消息数量、设置的 session ID，并在结束时记录 token 用量、成本、停止原因、首 token 时间、持续时间、HTTP 状态、重试次数和响应模型；重试（`pi.ai.retry`，来自 HTTP 重试路径）、上下文溢出和中止是事件，失败则只按其种类设置错误状态。提供商收到作为 `telemetry_context` 的请求 span，因此它自己的 span 嵌套在其下。

```python
from pi_ai import Context, UserMessage, create_models, faux_assistant_message, faux_provider
from pi_telemetry import InMemoryTelemetryContext

faux = faux_provider()
models = create_models()
models.set_provider(faux.provider)
faux.set_responses([faux_assistant_message("Hi there")])

telemetry = InMemoryTelemetryContext()
await models.complete(faux.get_model(), Context(messages=[UserMessage("Hello")]), telemetry_context=telemetry)
[span] = telemetry.get_spans()
print(span.name, {key: span.attributes[key] for key in ("pi.ai.provider", "pi.ai.stop_reason", "pi.ai.usage.output_tokens")})
```

不传 `telemetry_context`（或传 `NOOP_TELEMETRY_CONTEXT`）时不构建任何东西：请求与之前完全一样运行，没有可测成本。默认情况下 span 不保存提示、完成、思考、工具参数、头或错误文本；`pi_telemetry.with_content_capture()` 可选择开启。`pi.ai` schema 是 `pi_ai.telemetry.AI_TELEMETRY_SCHEMA`；`pi_telemetry.otel` 按 GenAI 语义约定和指标把它导出到 OpenTelemetry。图像生成和分类会收到 `telemetry_context`，但不记录 span。

## 性能说明

- 流式对象是槽位化 dataclass；事件队列是 deque，直接交接给等待中的消费者；被包装的流（鉴权设置、懒加载）转发事件而不多一次队列跳跃。
- 工具调用参数流式输出使用增量部分 JSON 词法分析器（`StreamingJsonParser`），对于大参数，比每个 delta 都重新解析缓冲区快约 60 倍（`benchmarks/bench_partial_json.py`）。
- 流式文本和思考以摊还线性时间累积（`append_text`）：在 CPython 中，直接 `block.text += delta` 会在每个 delta 拷贝整个字符串。
- 单核端到端（`benchmarks/bench_stream.py`）：anthropic-messages 每秒处理约 110,000 个事件，openai-completions 约 70,000（openai SDK 的 SSE 解码器占主导）。google-genai SDK 用 pydantic 校验每个流式块，每个块约 100 µs；真实的 Gemini 块携带大量 token，因此每个响应约几毫秒。
- 安装 `pi-ai-python[fast]` 后，SSE 块用 orjson 解码。
- 工具参数校验器按 schema 编译一次并缓存。
- 目录按提供商懒加载；API 实现和 OAuth 流程在首次使用时才导入。
- HTTP 连接按事件循环池化。

## 与 TypeScript 包的差异

- 命名是 snake_case；选项是带 snake_case 键的关键字参数/dict（`thinking_enabled`、`reasoning_effort`、`cache_retention`）。`model.compat` 保留 pi 的 camelCase 键，因为它是与 `models.json` 共享的配置数据。
- 工具 schema 是 JSON Schema dict（含 `Type` 构建器）或 Pydantic 模型；不需要 TypeBox。
- `fetch` 被 `http_client`（一个 `httpx.AsyncClient`）取代。
- 适配器使用 pi 所用 SDK 的 Python 版本（`anthropic`、`openai`、`google-genai`，以及替代 `@aws-sdk/client-bedrock-runtime` 的 `boto3`）。错误消息保留 pi 的溢出与重试检测所依赖的 JS SDK 措辞。
- Cloudflare Workers AI 的 *bindings*（仅 Workers 运行时）不可用。
- `telemetry_context` 与上游一样传播；`Models` 另外记录 `pi.ai.request` span（见[遥测](#遥测)）。

## 许可证

MIT
