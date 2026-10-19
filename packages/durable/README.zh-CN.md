# pi_durable

> **实验性。** 本 API 在版本之间会无预告地变化。

一个持久化的 agent 框架。对话、模型回合、工具调用以及你自己的状态，都会在展示任何内容之前先提交到存储。如果进程在回合中途死掉，重新打开存储就会从停止的地方继续工作。

这是 [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable) 1.3.0 的 Python 移植版。模型访问基于 [`pi_ai`](../ai/README.md)，文档状态基于 [`pi_chord`](../chord/README.md)。通过 `pip install pi-ai-python` 安装。

## 目录

- [安装](#安装)
- [本移植包含的内容](#本移植包含的内容)
- [快速开始](#快速开始)
- [两条 Python 规则](#两条-python-规则)
- [概念](#概念)
- [持久化与恢复](#持久化与恢复)
- [扩展](#扩展)
- [工具](#工具)
- [代码模式](#代码模式)
- [系统提示](#系统提示)
- [每会话 Agent](#每会话-agent)
- [设置](#设置)
- [环境](#环境)
- [重新加载](#重新加载)
- [观察会话](#观察会话)
- [忙碌的会话](#忙碌的会话)
- [重置与交接](#重置与交接)
- [压缩](#压缩)
- [Agent 事件（实验性）](#agent-事件实验性)
- [Hooks](#hooks)
- [更多会话与分叉](#更多会话与分叉)
- [中止与子 Agent](#中止与子-agent)
- [子任务](#子任务)
- [任务图](#任务图)
- [你自己的状态](#你自己的状态)
- [用量与成本](#用量与成本)
- [存储](#存储)
- [错误](#错误)
- [遥测](#遥测)
- [与 TypeScript 包的差异](#与-typescript-包的差异)
- [设计说明](#设计说明)

## 安装

```bash
pip install pi-ai-python    # 包含 pi_durable、pi_chord、pi_ai 和 pi_agent
pip install "pi-ai-python[images]"    # 增加 Pillow，用于 read 工具的图像处理器
pip install "pi-ai-python[codemode]"  # 增加 pydantic-monty，即代码模式的沙箱
```

要求 Python 3.11 或更高。一切都在 `asyncio` 上运行。

## 本移植包含的内容

pi-durable 的核心：Session 及其存储（memory、SQLite、JSONL）、带内置生成/工具/压缩任务的 Harness、registry、扩展、你定义的工具、hooks、文档、子任务、任务图、视图、watch、agent 事件、本地执行环境，以及存储和环境一致性测试套件。还有 `pi_durable.tools` 的编码工具（`read`、`write`、`edit`、`bash`、`powershell` 和 `CodingTools` 扩展）、`pi_durable.tool_search` 的 `tool_search` 工具、位于 `pi_durable.images` 的基于 Pillow 的 `read` 图像处理器（`images` extra），以及 `pi_durable.codemode` 的代码模式扩展，其脚本是由 Monty 运行的 Python（`codemode` extra）。参见[工具](#工具)与[代码模式](#代码模式)。

尚未包含：Cloudflare 适配器（Durable Object SQLite 存储与 Workers 图像处理器）。

## 快速开始

```python
import asyncio

from pi_ai import create_models
from pi_ai.providers.openai import openai_provider
from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import AssistantEntry, Harness, MemoryStorage, create_registry

context = BACKGROUND_CONTEXT


async def main() -> None:
    models = create_models()
    models.set_provider(openai_provider())  # 读取 OPENAI_API_KEY

    harness = await Harness.open(MemoryStorage(), {"models": models, "registry": create_registry()}, context)
    root = await harness.root(context, agent={"model": {"provider": "openai", "modelId": "gpt-6-sol"}})

    submission = await root.submit({"type": "input", "content": "What is the capital of France?"}, context)
    settled = await submission.wait(context)
    if settled["status"] == "done" and settled["type"] == "input":
        answer = await root.commit(lambda tx: tx.entry(AssistantEntry, settled["answer"]), context)
        message = answer["model"][0]
        print("".join(part.text for part in message.content if part.type == "text"))
    await harness.close(context)


asyncio.run(main())
```

发生了什么：

- `Harness.open()` 在一个存储后端上打开一个 Session。`MemoryStorage` 把所有内容保存在内存中。
- `root()` 返回根会话，首次使用时按给定的 agent 选择创建它。会话是条不可变 entry 组成的 transcript。
- `submit()` 持久化地收录你的输入并返回一个 `Submission`。内置的生成任务调用模型并把答案追加进去。
- `wait()` 在输入被回答（`done`，答案的 entry ID 在 `answer` 中）或失败（`unanswered`，带 `reason`）后返回。

每个异步调用都接受一个 Chord `Context`（`pi_chord.context`），它携带取消。`BACKGROUND_CONTEXT` 永不取消。取消一次 wait 只取消这次 wait，绝不取消工作本身：传入 `with_cancel(context).context` 并调用它的 `cancel()` 来停止等待。

本 README 中的其他示例都是片段。它们使用顶层 `await`（`python -m asyncio` 允许这样；在脚本中请放进 `async def`），并复用快速开始的 `harness`、`root`、`models` 和 `context`，外加一个你自己保留的 `registry`：`registry = create_registry()`，以 `"registry": registry` 传入。

## 两条 Python 规则

大部分 API 是上游的 snake_case 版本。有两条规则是本移植特有的。

**记录（records）是以 camelCase 键写的 TypedDict。** 所有数据类内容（records、submissions、任务状态与结果、documents、settings、选项对象、工具结果、事件）都是类型为 `TypedDict` 的普通 `dict`，其键与 TypeScript 的属性名完全一致：`settled["answer"]`、`record["conversationId"]`、`{"type": "input", "content": "Hi", "whenBusy": "steer"}`、`{"ownership": {"kind": "task", "taskId": id}}`。一条 record 就是它持久化后的 JSON，因此 TypeScript 版 pi-durable 写入的 SQLite 文件或 JSONL 目录可以在这里打开，反过来也一样；测试套件对照一份由 TypeScript 包写出的数据库和 JSONL 目录做了检查。方法名、函数名和关键字参数是 snake_case（`root.view_state()`、`define_tool(structured_output_schema=...)`）；dict 键则永远不会。唯一的例外是 hook 表——它们是方法表而非数据：`hook(ToolTask, {"before_tool": ...})`。消息与内容块是 `pi_ai` 的 dataclass（`UserMessage`、`TextContent` 等）。

**许多方法返回一个已经开始运行的 future。** 在 JavaScript 中，`async` 函数一旦被调用就会同步执行到第一个 `await`，因此 `root.submit()` 在调用时就收录了提交。Python 协程在未被 await 之前什么都不做。在这个差异重要的地方，移植版的方法是普通函数——它们在调用时完成工作并返回一个已启动的 `asyncio` future：`Conversation.submit`、`configure`、`commit`、`reset`、`compact`、`fork`、`abort`、`watch`、`view_state` 等；`Harness.root`、`conversation`、`create_conversation`、`wait_for_task`、`close` 等；以及每个读写数据的 `Tx` 方法。你通常直接 `await` 它们。要启动一个并继续做别的，保留这个 future：

```python
pending = root.submit({"type": "write", "entry": {"kind": "app.note", "data": "saved"}}, context)
# 提交已被收录：之后发生的 close 无法超越它。
submission = await pending
print(submission.id)
```

像任何 future 一样使用它（`asyncio.gather()`、`asyncio.ensure_future()`、`add_done_callback()`）。对它调用 `asyncio.create_task()` 会抛 `TypeError`，因为它不是协程，而调用时工作已经收录了。你交给 Harness 的回调（commit changes、工具 `execute`、任务阶段、hooks、sections、监听器）可以是普通函数或协程函数；Harness 会 await 一切可 await 的东西。

## 概念

- **Harness**：一个已打开的存储，加上在其上运行 agent 的机制。所有变更都经过同一行原子提交，任何内容在它的提交被存储之前都不会被展示。
- **Conversation**：一个 transcript。`root()` 在首次使用时创建根会话；你可以创建更多会话并分叉它们。一个 `Conversation` 句柄不保存任何状态；用 `id` 比较句柄。
- **Entry**：一条不可变的 transcript 记录，例如一条用户消息（`pi.user`）、一个模型响应（`pi.assistant`）、一个工具结果（`pi.tool-result`）、一次系统提示变更（`pi.system`）、一次重置（`pi.reset`），或你自己的类型。模型只看到最近一次重置之后的 entry。
- **Commit**：一次原子写入。`conversation.commit(change, context)` 调用 `change(tx)`，它可以在一次提交中同时追加 entries、编辑 documents 和创建 tasks；要么全部被存储，要么全部不被存储。
- **Document**：存储在 transcript 旁边的带类型 JSON 状态，在 commit 中变更。内置的文档保存每个会话的 agent 选择（`pi.agent`）、面向提供商的会话身份（`pi.provider`）、进行中的生成与工具（`pi.live`）、排队的提交（`pi.inbox`）和花费（`pi.usage`）。
- **Task**：一个持久化状态机，每一步都保存检查点，因此重启后的进程会从最后一个检查点继续。每个任务都有一个拥有者：它的会话，或另一个任务。Harness 把回答作为内置任务运行：`pi.generation` 调用模型并拥有其工具调用的 `pi.tool` 任务、等待它们，然后把运行交给下一个 generation。
- **Submission**：你交给会话的东西，要么是用户输入，要么是要写入的 entry，你可以等待它。
- **Turn 与 run**：一个 turn 是一次模型响应及其工具调用；一个 run 是从一次输入到其最终答案之间的所有 turn。有 run 在进行时会话就是忙碌的。
- **Extension**：一组命名的工具、系统提示段、hooks、包装器和任务。
- **Registry**：本进程安装的扩展。它可以在 Harness 运行期间变化；新工作使用新状态。
- **Agent**：会话运行所用的内容：模型、思考级别、选中的扩展、工具、指令和工作目录。以名字存储在每会话的 `pi.agent` 中，在每次使用时对照 registry 解析。

一次被回答的输入，以 entries 和 tasks 表示：

```text
submit(input) → pi.user
  pi.generation → pi.system (仅当提示或工具变化时), pi.assistant (tool calls)
    pi.tool × n → pi.tool-result × n   (归该 generation 所有，它会等待它们)
  pi.generation → pi.assistant (answer) → submission done
```

## 持久化与恢复

使用 SQLite 或 JSONL 存储来跨重启保留会话：

```python
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

harness = await Harness.open(await open_sqlite_storage("./session.sqlite"), {"models": models, "registry": registry}, context)
root = await harness.root(context)  # 与上次相同的根会话
harness.resume()  # 继续上一次进程未完成的任何 run
```

因崩溃或关闭而中断的工作保持 pending 状态。`resume()` 启动任务调度器；提交或等待也会启动它。每个会话在 `pi.provider` 中持久化自己独有的 UUIDv7，作为 `session_id` 转发给 pi_ai，用于提供商的提示缓存和会话亲和性。它会在重新打开、重试、重置、压缩和模型更换后存活；子会话或分叉会收到新的身份。没有该身份的会话会在其第一次生成或压缩请求前收到并持久化一个。

带相同 `requestId` 的重试提交会返回已有的 submission，而不是重复提交：

```python
submission = await root.submit({"type": "input", "content": "Hello", "requestId": "greeting-1"}, context)
# 重启后：相同的请求 ID 找到同一个 submission。
again = await root.submit({"type": "input", "content": "Hello", "requestId": "greeting-1"}, context)
assert again.id == submission.id
```

`harness.submission(id, context)` 按 ID 重新取得一个 submission，例如在重启后等待它。

### 休眠与唤醒

任务通过以休眠状态结束某个阶段来等待时间。休眠是被存储的，因此任务在此期间不占用任何调用，能跨重启存活，并在 Harness 时钟到达 `until` 时从检查点恢复：

```python
from typing import Any, TypedDict

from pi_durable import Task, define_extension, define_task


class ReminderInput(TypedDict):
    text: str
    at: float  # 纪元毫秒


async def wait(task, runtime, context):
    # 一旦休眠被存储就返回；该阶段在那里结束。
    await runtime.sleep_until(task["input"]["at"], {"phase": "deliver"}, context)


async def deliver(task, runtime, context):
    await notify(task["input"]["text"])  # 你的代码
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "completed", "result": None}}, context)


async def abort(task, runtime, context):
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, context)


# 标注 token：结果类型（此处的 None）不会出现在任何参数中。
Reminder: Task[ReminderInput, Any, None, Any] = define_task(
    name="app.reminder",
    version=1,
    initial=lambda input: {"phase": "wait"},
    phases={"wait": wait, "deliver": deliver},
    abort=abort,
)
registry.install(define_extension("reminders", tasks=[Reminder]))

reminder = await root.commit(
    lambda tx: tx.create_task(Reminder, {"text": "Stand up", "at": harness.now() + 200}, {"ownership": {"kind": "conversation"}}),
    context,
)
settled = await harness.wait_for_task(reminder, context)
print(settled["state"]["outcome"])  # {'status': 'completed', 'result': None}
```

阶段处理器是 `handler(task, runtime, context)`；`task` 是任务记录，`runtime.commit(change, context)` 调用 `change(tx, current)`，其返回的状态替换任务的状态。阶段中间没有休眠：阶段在重启后会从顶部重新运行，因此等待之后的工作要放进下一个阶段，而它需要的东西要放进检查点。从 `runtime.commit()` 的 change 中返回的 `{"status": "sleeping", "until": ..., "checkpoint": ...}` 是同一种休眠，与其他写入一起提交。内置任务的重试退避和延迟响应轮询也用同样的方式休眠。

Session、调度器及其任务的定时器都经过 `setTimer` 选项（默认：当前事件循环上的 `loop.call_later`），它们的时间经过 `now`（默认：`time.time()` 的毫秒数）；两者都传入就可以在假时钟上运行 Harness。`setTimer(callback, delay_ms, options)` 返回一个解除定时器的函数；对 `options["background"]` 为真的定时器可以返回 `None` 拒绝（那些只释放内存）。本地环境和文件监视器保留自己的定时器。休眠截止时间存储在 `now` 上，因此打开同一存储的每个进程必须使用同一个时钟。

asyncio 定时器不会像 Node.js 定时器那样让进程保持存活：`asyncio.run(main())` 在 `main()` 返回时结束，无论有没有休眠任务。在期待有工作期间保持事件循环运行，例如 await `harness.wait_for_task(id, context)` 或 `harness.closed`。

宿主可以在事件间隙随时停止进程，因此它自己唤醒自己。使用 `onWake` 时，Harness 不会为休眠布下定时器；它会在 `harness.wake()` 返回的内容变化时告知宿主：当某些任务现在可以运行或正在运行时是 `active`，以及最早的休眠时间 `at`。宿主保持自己的闹钟布防，并在其触发时调用 `resume()`：

```python
import time

alarm = {"at": None}  # 宿主闹钟的纪元毫秒，或 None


def on_wake(wake):
    # 有工作在运行时是心跳；否则是最早的休眠时间（如果有）。
    alarm["at"] = time.time() * 1000 + 60_000 if wake["active"] else wake.get("at")


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "onWake": on_wake}, context)
on_wake(harness.wake())  # 在可能开始工作之前布防
harness.resume()  # 处理丢失进程留下的工作，以及到期的休眠

# 稍后，当宿主的闹钟触发时：
harness.resume()
```

在任何一个可以开始工作的调用（`submit`、`compact`、`abort_task`、`abort`、registry 变更）之前布防闹钟：`onWake` 在提交之后运行。过早的闹钟无害；缺失的闹钟会让工作搁浅。

### 失败与恢复

**崩溃循环。** 在到达下一个检查点之前就确定性崩溃进程的工作（耗尽内存或 CPU），否则会被恢复并永远崩溃下去。每次崩溃、驱逐或重置中断一个任务都会计入 Open 的计数；close 不计入，任务提交的新状态会让计数归零。一旦计数达到 `settings["restartLimit"]`（默认 5；`math.inf` 永不停车），任务就被停车（parked）：`inspect()` 和任务图把它显示为 reason 为 `restart_limit` 的 blocked，`wake()` 不计它为 active，`onReport` 作为 `task_parked` 收到它一次。进程死亡时每个正在运行的任务都会被计数，所以请显示被停车的任务并提供 `harness.retry_task(id, context)`，它会把任务再运行一次。`abort_task()` 仍会运行它的 abort 处理器；被中断这么多次的 abort 处理器会让任务以 `faulted` 结束。

**报告。** `onReport(error, info)` 在 `info["type"]` 中说明发生了什么：`session_failed` 是让 Harness 失败的错误（存储错误）；`task_faulted`、`task_parked`、`migration_failed`、`task_outdated` 或 `task_error` 是关于任务的，带 `info["task"]`（其 `id`、`kind` 和 `conversationId`）以及在 `info["extension"]` 中安装它的扩展；`extension_error` 是关于 hooks、sections、包装器和工具的，带扩展名；`host_error` 是关于宿主回调的，在 `info["callback"]` 中指名；以及 `listener_error`。宿主可以把某种 `task_faulted` 报告喂给自己的熔断器。

```python
def on_report(error, info):
    print(info["type"], info.get("task", {}).get("kind"), error)


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "onReport": on_report}, context)
```

**失败 run 后排队的输入。** 因模型错误而失败的 run 会把排在其后的输入留在队列里，直到下一条输入到达。`settings["afterFailedRun"]` 改变这一行为：`continue` 像回答之后那样用它们启动下一个 run，`withdraw` 则把它们以 reason `run_failed` 结算为 `unanswered`。

**被阻塞的任务。** 阻塞在缺失或无法迁移的定义上的任务也会在任务图中显示其 reason，而解除阻塞的 registry 变更会让 `onWake` 报告 Session 为 active，因此宿主无需轮询它们。

## 扩展

除内置任务外，Harness 运行的代码以命名扩展的形式出现在你进程拥有的 registry 中：

```python
from pi_durable import ToolTask, create_registry, define_extension, hook, section


def needs_approval(call, api, context):
    return {"block": "Needs approval"} if call.name == "deploy" else None


Coding = define_extension(
    "coding",
    sections=[section("preamble", lambda input, context: "You are a concise coding assistant.", tag=False)],
    hooks=[hook(ToolTask, {"before_tool": needs_approval})],
)

registry = create_registry()
registry.install(Coding)
```

一个扩展可以携带 `tools`、`sections`、`hooks`、`wraps`（按名字装饰工具或段的装饰器）和 `tasks`。默认情况下每个会话都按安装顺序选择每个已安装的扩展。registry 中什么都不存储；会话存储扩展名。

## 工具

`pi_durable.tools` 提供 `read`、`write`、`edit` 和 `bash`，以及包含全部四个的 `CodingTools` 扩展（`create_powershell_tool()` 会加上 `powershell`）。它们只通过调用的环境（见[环境](#环境)）接触文件和进程，而且没有东西会自动安装它们：

```python
from pi_durable.tools import CodingTools

registry.install(CodingTools)
```

`read` 把文本连同续读提示作为 diagnostics 返回（如 `Use offset=2001 to continue.`），图片文件则作为一个图像块返回，调用它的程序得到 `ImageContent`。图像限制是模型的 `input_limits.images.resize`，默认 2000x2000 像素和 4.5 MiB 的 base64。没有图像处理器时，它按原样发送范围内的 PNG、JPEG、GIF 和 WebP 文件（尺寸从文件头读取），不做解码，并拒绝更大的文件和其他格式。有处理器时，它解码每张图像、按 EXIF 方向摆正、把 BMP 转为 PNG，并缩小到适合为止。基于 Pillow 的处理器需要 `images` extra，并且只在你导入它时才被加载：

```python
from pi_durable.images import create_pillow_images
from pi_durable.tools import create_coding_tools

images = create_pillow_images()
registry.install(create_coding_tools(images=images))
```

没有图像输入的模型会看到一个占位符而不是图像，`read` 会在 diagnostic 中说明。处理器在自己的工作线程中逐张准备图像，因此一张大照片（大约一秒的工作量）永远不会阻塞事件循环：Pillow 在解码、缩放和编码期间释放 GIL。`create_pillow_images(worker=False)` 改在调用线程上工作，`executor=` 把你的 executor（一个线程池可以同时准备多张图像，或一个进程池）用于该工作。`images.close()` 停止其工作线程；空闲的线程不持有任何图像内存。它写出的图像不是 Photon 的字节，但尺寸、格式和方向遵循同样的规则；在只有字节限制决定图像缩小到多小的地方，Pillow 较小的 JPEG 可以在更大的尺寸停下。

`edit` 替换精确、唯一、不重叠的文本（对尾部空格、智能引号、破折号和特殊空格有模糊回退），保留文件的 BOM 和 CRLF 换行，并在其 `details` 中返回显示用的 diff 和统一补丁——与上游 jsdiff 写出的文本相同。`edit` 和 `write` 在进程内串行化对单个文件的修改。`bash` 流式输出并保留其尾部；超过 2000 行或 50KB 的输出会溢写到 diagnostic 点名的文件中。`create_bash_tool(command_prefix=..., prepare=...)` 在每条命令前运行若干行，或修改它的命令、工作目录和环境。

用 `define_tool()` 定义你自己的工具，它接受关键字参数。`parameters` 是用 `pi_ai.Type`（或普通 dict，或 Pydantic 模型）构建的 JSON Schema，Harness 在 `execute(args, api, context)` 运行前校验它。`api.output()` 流式输出运行中的输出，当 `execute()` 没有返回 `output` 时它就变成结果：

```python
from pi_ai import Type
from pi_durable import define_extension, define_tool


async def count_execute(args, api, context):
    for i in range(1, int(args["n"]) + 1):
        api.output(f"{i}\n")
    return {}


count = define_tool(
    name="count",
    description="Count from 1 to n",
    parameters=Type.Object({"n": Type.Number()}),
    execute=count_execute,
)
registry.install(define_extension("count", tools=[count]))
```

每次调用都作为自己的持久化任务运行。它的 intent 在 `execute()` 运行前提交。如果进程在调用中途死亡，只有声明为 `replay="safe"` 的工具才会在重新打开时重跑；否则模型会收到带已提交输出和 `interrupted` 错误的结果。从 `execute()` 抛出异常会给模型一个错误结果。结果也可以返回 `usage`（一个 `pi_ai.Usage`），它会加进会话的[用量](#用量与成本)。它还可以返回 `"control": {"terminate": True}`：当该轮每个结果都这样请求时，run 结束，不再发模型请求。

结果是一个带三个通道的 dict：`output` 给模型（`pi_ai` `TextContent` 和 `ImageContent` 的列表），`structuredOutput` 给调用工具的程序（见[工具调用工具](#工具调用工具)），`details` 给 UI。在 `output` 中使用 `pi_ai` 内容类，不要用 dict：输出限制（`output_limits`）只约束 `TextContent`。声明了 `structured_output_schema` 的工具返回匹配的 `structuredOutput`，Harness 会校验它。调用未声明该 schema 的工具的程序会得到其受约束的输出：一个文本项作为字符串，一个图像作为其 `ImageContent`，什么都没有则作为 `""`，其他情况作为内容列表——错误时也是如此。`structuredOutput` 始终是工具自己的值：schema 工具的出错结果保留它返回的值，而没有 schema 的工具其文本在 `tool_error` diagnostic 中，正如抛出的错误消息那样。transcript 只存储 `output` 和 `details`：

```python
from pi_ai import TextContent


async def lines_execute(args, api, context):
    read = await api.env.read_text_file(args["path"], context)
    if read["ok"] is False:
        raise read["error"]
    count = len(read["value"].split("\n"))
    return {"output": [TextContent(f"{count} lines")], "structuredOutput": {"lines": count}}


lines = define_tool(
    name="lines",
    description="Count the lines of a file",
    parameters=Type.Object({"path": Type.String()}),
    structured_output_schema=Type.Object({"lines": Type.Number()}),
    execute=lines_execute,
)
```

缺失、未声明或无效的 `structuredOutput` 会把嵌套调用的结果变成带 `invalid_structured_output` diagnostic 的错误结果；对模型发起的调用（它永远不会看到这个值），则丢弃并通过 `onReport` 报告。错误结果可以省略它，程序随后会在 `tool_error` diagnostic 中找到它的文本；Harness 自己写出的错误结果（blocked、invalid、interrupted、aborted）不带任何值，它们的 `diagnostics` 说明出了什么问题。对 schema 工具的 `output` 做脱敏的 `after_tool` hook 也必须脱敏它的 `structuredOutput`。内置的 `bash` 返回 `{"output", "truncated", "fullOutputPath"?, "exitCode"}`，其中 `output` 是模型看到的保留尾部，用 `api.retained_output()` 读取，非零退出时它用仍然携带输出的错误结果回答。

后一个扩展的同名工具在两个都被选中时替换前一个，`wrap_tool()` 装饰胜出的那个。`ToolRegistration` 是 dataclass；包装器用 `dataclasses.replace()` 返回修改后的副本：

```python
import dataclasses
import time

from pi_durable import wrap_tool
from pi_durable.tools import create_bash_tool


def timed(tool):
    async def execute(args, api, context):
        start = time.monotonic()
        try:
            return await tool.execute(args, api, context)
        finally:
            api.diagnostic({"severity": "info", "message": f"took {time.monotonic() - start:.3f}s"})

    return dataclasses.replace(tool, execute=execute)


Timing = define_extension("timing", wraps=[wrap_tool(count, timed)])
registry.install(Timing)

# Python 虚拟环境内的 bash：两者都被选中时，它替换 CodingTools 的 bash。
Venv = define_extension("venv", tools=[create_bash_tool(command_prefix="source .venv/bin/activate")])
```

### 工具调用工具

工具用 `api.execute_tool(name, args, context, *, key=None, progress=None)` 调用另一个工具，正如代码模式脚本或 MCP 桥那样。嵌套调用是它自己的工具任务，归调用者所有：它被校验、运行 `ToolTask` hooks、保持自己的输出限制和重放策略，并把结果返回给调用者而不是 transcript：

```python
async def word_count_execute(args, api, context):
    words = len(args["text"].split())
    return {"output": [TextContent(f"{words} words")], "structuredOutput": {"words": words}}


word_count = define_tool(
    name="word_count",
    description="Count the words of a text",
    parameters=Type.Object({"text": Type.String()}),
    structured_output_schema=Type.Object({"words": Type.Number()}),
    execute=word_count_execute,
)


async def long_text_execute(args, api, context):
    run = await api.execute_tool("word_count", {"text": args["text"]}, context, progress=False)
    # 嵌套调用未完成时缺省。
    words = (run.get("structuredOutput") or {}).get("words")
    return {"output": [TextContent("long" if words is not None and words > 100 else "short")]}


long_text = define_tool(
    name="long_text",
    description="Say whether a text is long",
    parameters=Type.Object({"text": Type.String()}),
    execute=long_text_execute,
)
registry.install(define_extension("text", tools=[word_count, long_text]))
```

- 调用者拿到嵌套调用的 `taskId`、`structuredOutput`、`details`、`diagnostics`、`usage`、`isError` 和 `durationMs`，而不是给模型看的 `output`。被阻塞、无效、失败或中止的嵌套调用返回 `isError` 结果。`execute_tool()` 本身只在 `execute()` 已返回、调用者被中止或 Harness 关闭时抛出；取消它的 `context` 只停止等待。
- Hooks 能看到嵌套调用的 `call.parent`，因此守卫可以对来自工具的不同调用区别对待。
- 调用者运行时，它的嵌套调用显示在 `docs["pi.live"]["nestedTools"]` 中，带它们运行的参数，完成后还有 `summary`（错误、时长、用量），并以带 `parentToolCallId` 和 `parentTaskId` 的工具事件出现。`progress=False` 让嵌套调用的运行输出不进入 `pi.live`。调用者落定前，它遗留的在运行的嵌套调用会被中止，包括更早尝试留下的。调用者的结果消息不列出其嵌套调用；它们的任务记录保留它们。
- 每个嵌套调用有一个 key，即它在调用者嵌套调用中的位置，除非调用者传入 `key=`。可安全重放的调用者在崩溃后重跑并以相同顺序调用（或传相同的 key）时，会拿到它已经做过的嵌套调用——无论已完成还是仍在运行——而不是重新启动它们。崩溃在结果提交前中断的嵌套调用遵循它自己的 `replay` 策略，因此带外部副作用且可安全重放的工具仍然需要自己的幂等。不可安全重放的调用者永远不会重跑，因此重启后它未完成的嵌套调用（连同它们拥有的一切）在再次运行前被废弃。
- 嵌套调用的 `usage` 计入它自己的工具名下；在调用者的结果中把它传下去会重复计数。
- 嵌套结果不会被保留：它们只活到调用者落定为止，位于调用者的任务文档中（`NestedCallDoc`，每个嵌套调用一个，按 key 索引，持有 `{"taskId": ..., "result": ...}`）。嵌套调用的任务记录只保留一张小收据。
- 没有东西限制嵌套的深度；带默认 `callers` 的工具甚至可以调用自己。在重要的地方限制 `callers`（见下）。

### 谁可以调用工具

工具上的 `callers` 说明谁可以调用它：模型、通过 `execute_tool()` 的其他工具，或两者，即默认。会话的 `modelTools` 收窄模型被提供哪些工具，而不把它们从工具那里夺走：

```python
lookup = define_tool(name="lookup", callers=["tools"], execute=word_count_execute)  # 永不提供给模型
registry.install(define_extension("lookup", tools=[lookup]))

# 模型只被提供 long_text；long_text 仍可调用 word_count。
await root.configure({"modelTools": [long_text]}, context)
# 模型被提供除 word_count 外的一切，而工具仍可调用 word_count。
await root.configure({"modelTools": {"remove": [word_count]}}, context)

agent = await root.agent(context)
print([tool.name for tool in agent["tools"]], [tool.name for tool in agent["callable"]])
```

`conversation.agent()` 解析两个列表：`tools`（提供给模型）和 `callable`（嵌套调用在其间解析）。`tools` 仍决定什么根本被启用；`modelTools` 不能向它增加内容。

`deferred=True` 工具在被加载前不提供给模型，但其他工具可以随时调用它。来自 `pi_durable.tool_search` 的 `tool_search` 工具（`create_tool_search_extension()`）让模型搜索被延迟的工具（对它们的名字、描述和参数做 BM25），并用 `"control": {"addTools": [...]}` 加载匹配项，这会把它们记录在 `modelTools` 中；加载的工具在该分支上保持被提供，压缩之后也如此，而且 `codemode` 段把延迟工具排除在外。任何工具结果都可以那样加载延迟工具。用于大型工具集，例如一个 MCP 服务器的：

```python
from pi_durable.tool_search import create_tool_search_extension


async def get_issue_execute(args, api, context):
    return {"output": [TextContent(f"Issue {args['number']}: flaky login test")]}


get_issue = define_tool(
    name="github_get_issue",
    description="Get a GitHub issue by number",
    parameters=Type.Object({"number": Type.Number()}),
    deferred=True,
    execute=get_issue_execute,
)
registry.install(define_extension("github", tools=[get_issue]))
registry.install(create_tool_search_extension())
# 模型被提供 tool_search；用 {"query": "github issue"} 调用它会加载 github_get_issue。
```

### 重启时被废弃的任务

创建者只在内存中 await、且创建者重启后永不恢复的任务，可以在其任务选项中设 `"abandonOnRestart": True`。当后面的 Harness 开始调度并发现它仍存活时，它会在再次运行之前连同它拥有的一切以 `abortReason` `"restart"` 被中止。不可安全重放的工具的嵌套调用会自动获得这一点，这样的工具用 `api.create_task()` 创建的子任务也一样。以这种方式被废弃且定义缺失的任务会等待其扩展被安装，以便它的 abort 处理器仍然清理。它的调用者及其会话与它一起等待；`harness.inspect()` 把它显示为 blocked。随后中止它会让它成为孤儿，就像任何中止一样。

## 代码模式

`pi_durable.codemode` 增加一个 `codemode` 工具：模型编写把会话的工具作为 `await tools.<name>(...)` 调用的 Python，运行在 Monty 沙箱（[`pi_codemode`](../codemode/README.md)）中。每次调用都是嵌套工具调用，只有脚本的输出到达模型。脚本在重启后存活：它从停下的地方恢复，而不会重新运行已完成的调用。这就是上游代码模式的 Python 脚本版而非 JavaScript；沙箱需要 `codemode` extra（`pip install "pi-ai-python[codemode]"`）。

```python
from pi_codemode import CodemodeSandbox
from pi_durable.codemode import create_codemode_extension

sandbox = CodemodeSandbox(memory_limit_bytes=256 * 1024 * 1024)
registry.install(create_codemode_extension(sandbox=sandbox))
# 一次带 {"code": 'return await tools.word_count(text="a b c")'} 的 codemode 调用回答 '{"words":3}'。
# 进程关闭时：await harness.close(context)，然后 await sandbox.close()。
```

脚本是异步函数体，由模型作为工具的原始输入写出：

```text
# @options: {"max_output_tokens": 4000, "timeout_ms": 60000}
import asyncio
contents = await asyncio.gather(*[tools.read(path=path) for path in ["a.txt", "b.txt"]])
print("read", len(contents), "files")
found = await search_tools("github issues", limit=3)
return [tool["name"] for tool in found]
```

该扩展还增加一个 `codemode` 提示段，以 Python 声明（`async def read(self, *, path: str) -> str`）的形式在 token 预算内（`inline_budget=`，默认 3000）列出脚本可以调用的工具；脚本用 `search_tools()`、`describe_tool()` 和 `describe_namespace()` 找到其他工具。用 `namespaces=[{"name": ..., "description": ..., "prefix": ...}]` 按名字前缀分组工具，例如一个 MCP 服务器的 `mcp__github__`。结果读起来像编码 agent 的：一个头部、脚本的输出，失败时还有已经做过的调用；超过 `max_output_tokens` 的文本被截断，完整文本在一个临时文件中。脚本还获得 `models.*`：模型目录、`models.classify()` 和 `models.generate_images()`，它们作为内部工具的嵌套调用运行（`models=False` 关闭它们）。脚本运行期间关闭沙箱（例如关闭时）不是一种结果：脚本用下一个进程的沙箱恢复。失败的嵌套调用会在脚本中抛出带错误文本的 `Exception`，除非工具声明了 schema 并随错误返回了自己的值：`tools.bash` 对非零退出也返回 `{"output": ..., "exitCode": ..., ...}`，因此脚本检查 `exitCode`。`store(key, value)` 和 `load(key)` 按会话保留 JSON 值；分叉在进入时看到它们当时的值。`pi_durable.codemode` 不导入沙箱：任何 `pi_codemode.CodemodeSandboxBase` 都可以。

脚本全局变量是 snake_case，而上游是 camelCase；工具名不变（`tools.mcp__github__list_issues`，以及内部工具 `pi.codemode.models.classify` 和 `pi.codemode.models.generateImages`）：

| 上游（JavaScript） | 这里（Python） |
| --- | --- |
| `// @options: {...}` | `# @options: {...}` |
| `await tools.read({ path })` | `await tools.read(path=...)` 或 `await tools.read({"path": ...})` |
| `Promise.all([...])` | `asyncio.gather(...)` |
| `console.log(...)` | `print(...)` |
| `searchTools(query, { limit, namespace })` | `search_tools(query, limit=8, namespace=None)`，或选项作为一个 dict |
| `describeTool(name)`、`describeNamespace(name)` | `describe_tool(name)`、`describe_namespace(name)` |
| `models.getModelsOfType(type, provider?)` | `models.get_models_of_type(type, provider=None)` |
| `models.getAvailableOfType(type, provider?)` | `models.get_available_of_type(type, provider=None)` |
| `models.getModelOfType(type, provider, id)` | `models.get_model_of_type(type, provider, id)` |
| `models.classify(model, context)` | `models.classify(model, context)` |
| `models.generateImages(model, context)` | `models.generate_images(model, context)` |
| `undefined`（缺失值） | `None` |

目录项、分类器和图像上下文以及模型结果都是 pi-ai 的 JSON，键为 camelCase（`result["stopReason"]`、`{"type": "image", "data": ..., "mimeType": ...}`）。Monty 中的异常只携带消息，因此上游附加到失败调用错误上的 `structuredOutput` 对脚本不可见。返回 `None` 的脚本不会给输出增加任何值，因为 Python 无法区分 `return None` 和没有 `return`。

它如何在重启后存活：沙箱确定性运行脚本（虚拟时钟、带种子的随机数、UTC），工具在把每个结果交给脚本之前把它写入日志（journal）。重启后，工具再次运行脚本，并按相同顺序把日志化的结果交给它，因此它用相同参数做相同调用；这些按 key 重新挂到它们的嵌套调用上，然后脚本继续。当脚本走了与之前不同的路径、沙箱指纹或代码模式版本变化、或它的 `timeout_ms` 期间已过时，重跑会以错误结束调用。重启时正在运行的嵌套调用遵循它自己的重放策略。run 和 journal 文档保持上游的格式，但一个 run 不能在语言间移动：在这里恢复的 TypeScript run（或反之）以“the sandbox changed since it started”结束，因为脚本、全局变量名和沙箱指纹都不同。

## 系统提示

系统提示由所选扩展的段构建，在每次请求前按顺序渲染。段的 `render(input, context)` 看到解析后的 agent、为请求构建的环境以及已提交的文档（`input["agent"]`、`input["env"]`、`input["read"]`），并且可以是异步的：

```python
cwd = section("cwd", lambda input, context: input["agent"].get("cwd"))  # 渲染为 <cwd>\n...\n</cwd>；None 省略它
registry.install(define_extension("cwd", sections=[cwd]))
```

会话的 `instructions` 最后渲染，作为 `instructions` 段。段和工具变更作为位置性系统 entry 存储在 transcript 中。只有变化的内容会被再次发送，这能保持提供商的提示缓存温热。每次都返回不同内容的段（例如当前时间）会破坏这一点。

## 每会话 Agent

每个会话把运行所用的内容存储在自己的 `pi.agent` 文档中。`configure()` 在一次提交中变更它；未设置的字段跟随宿主：

```python
await root.configure(
    {
        "model": {"provider": "openai", "modelId": "gpt-6-sol"},
        "thinkingLevel": "high",
        "extensions": {"remove": [Coding]},  # 编辑宿主默认值；列表则精确按序选择这些
        "tools": [lines, count],  # 列表精确启用这些；{"remove": [...]} 移除一些
        "modelTools": [lines],  # 其中，只把 lines 提供给模型；工具仍可调用 count
        "instructions": "Only read; never edit files.",
        "cwd": "/work/repo",
    },
    context,
)
await root.configure({"tools": None}, context)  # None 把字段清回宿主默认值
agent = await root.agent(context)  # 解析结果：model、extensions、tools、sections、cwd
```

扩展和工具以对象传入、按名字存储，因此已存储的名字比它的代码活得更久：扩展被卸载后，选择它的会话只是不再得到它，直到它再次被安装。`create_conversation()` 和 `fork()` 在选项中以 `"agent"` 接受同样的变更，`root()` 以 `agent=` 接受。任务拥有的会话（例如子 agent 的）以它的拥有者会话的 agent 的副本开始。分叉以它父会话在分叉 entry 时的 agent 开始。请求的模型、提示和提供的工具在它被准备时固定；变更从下一个请求生效。工具调用和 hooks 使用任务阶段解析它们的 agent，环境在每次使用时从当前 `cwd` 构建，因此 `cwd` 或扩展的变更可以到达模型已经做过的调用。

在提交内部，`pi_durable` 的 `configure(tx, conversation_id, change)` 应用同样的变更。

## 设置

每个会话共享的运行策略作为 `settings` 传入。它每次使用都被读取，从不存储。传一个 dict，或一个返回 dict 的函数以保持设置实时更新，例如由设置文件支持：

```python
user_settings = {"followUpMode": "one-at-a-time"}


def settings():
    return {
        "extensions": [Coding],  # 默认选择；缺省：每个已安装的扩展
        "stream": {"timeoutMs": 120_000},
        "retry": {"maxRetries": 3},
        "compaction": {"reserveTokens": 16384},
        "progress": {"partialIntervalMs": 100, "outputIntervalMs": 100},
        "toolExecution": "parallel",
        "contextRetentionMs": 600_000,  # 空闲会话的上下文在内存中保留这么久
        "documentRetentionMs": 600_000,  # 无人观察的已加载文档在最后一次使用后于内存中保留这么久
        "followUpMode": user_settings["followUpMode"],
    }


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "settings": settings}, context)
```

你保留并在原处修改的 dict 也是实时的，因为每次使用都会读取它。settings 函数必须是同步且快速的。

## 环境

`env` 为每次工具调用、段渲染和 `runtime.env()` 构建执行环境。它以 `env(target, context)` 被调用，参数是会话 ID、其 agent 的 `cwd` 和已提交的读取（`target["conversationId"]`、`target.get("cwd")`、`target["read"]`），因此一个函数可以为每个会话提供一个目录或每个会话一个容器：

```python
import os

from pi_durable.env.local import LocalExecutionEnv

harness = await Harness.open(
    MemoryStorage(),
    {
        "models": models,
        "registry": registry,
        "env": lambda target, context: LocalExecutionEnv(cwd=target.get("cwd") or os.getcwd()),
    },
    context,
)
```

从 `env` 抛出异常会成为该调用的错误结果。没有环境时，`api.env` 是 `None`，内置工具以错误结果失败。每次调用一个全新的环境对象没问题：`edit` 和 `write` 按环境的 `id` 和规范路径串行化对单个文件的修改。自定义 `ExecutionEnv` 设置 `id`，使相等的 id 在相同路径看到相同文件，例如每个容器一个 id；`LocalExecutionEnv` 的 id 是 `"local"`。

文件和 shell 操作返回结果 dict 而不是抛出预期中的失败：`{"ok": True, "value": ...}` 或 `{"ok": False, "error": FileError(...)}`（`pi_durable.env` 的 `get_or_throw()` 和 `get_or_none()` 解开其一）。宿主也可以直接使用环境，例如显示项目的文件。`open_binary_reader()` 读取一个已打开文件的字节范围，`open_dir_reader()` 对目录分页，带参数列表的 `exec()` 不用 shell 运行程序，并报告每个输出块来自哪个流：

```python
env = LocalExecutionEnv(cwd=os.getcwd())
status = {"stdout": "", "stderr": ""}


def on_output(text, context, info):
    status[info["stream"]] += text


result = await env.exec(["ls", "-a"], {"onOutput": on_output}, context)
print(result["ok"] and result["value"]["exitCode"])  # 0
```

取消 context 以停止一次调用；`cleanup()` 用于关闭环境。自定义环境可以用 `pi_durable.testing` 的 `register_env_conformance()` 自我检查，就像下面的存储一样。

环境也有 `list_files()`：一个目录下任意深度的文件和目录，如 `fd` 所列，跳过 git 忽略的内容，并在边界处以 `truncated` 停止。无法列目录的环境返回 code 为 `not_supported` 的错误，`LocalExecutionEnv` 就是这样；调用者随后回退到自己的遍历。

## 重新加载

安装一个已安装名字的扩展会在一步之内原地替换它：

```python
registry.install(define_extension("coding", sections=[section("preamble", lambda input, context: "You are a careful coding assistant.", tag=False)]))
```

`registry.uninstall(extension)` 移除已安装的该名字扩展，无论它是哪个对象。

已经开始的工作保留它取到的代码：运行中的工具调用在旧实现下完成，每个任务阶段只从阶段开始时的 registry 解析一次 hooks 和 agent。下一个阶段、请求或调用使用新代码。重启后，再次安装同样的扩展；扩展 `tasks` 的挂起任务一旦安装就恢复。

## 观察会话

UI 需要的每个东西都是已提交的状态。`view_state()` 把会话的结构视图作为只读 Chord 状态返回，在每次触碰它的提交之后更新。订阅者被调用时带该值、提交的 context 和一个投递对象（`delivery.kind` 先是 `"hydrate"`，然后 `"update"`），并且必须接受全部三个参数：

```python
view = await root.view_state(context)


def render(value, context, delivery):
    # value["entries"]：活动 transcript
    # value["docs"]["pi.live"]：进行中的生成（流式 partial、retry、deferred）和工具调用（output、details）
    # value["docs"]["pi.inbox"]、["pi.usage"]、["pi.agent"]、["pi.provider"]
    print(delivery.kind, len(value["entries"]), "entries")


unsubscribe = view.subscribe(render)
# 稍后：view.dispose()
```

`watch()` 以每次提交的精确 Chord 操作投递同一个视图，一次一个回调：

```python
watch = await root.watch(context)
print(sorted(watch.value))  # 附加时的状态：['conversation', 'docs', 'entries']


async def send(value, ops, context):
    print(ops)  # 例如，发送给应用它们的远程客户端


watch.start(send)
# 稍后：await watch.stop()
```

慢速 watch 最多保留 100 帧未投递。超过后，挂起帧被一个持有整个最新视图的帧替换。迟到加入或重连的客户端从当前视图开始；没有重放。

部分答案和工具输出默认最多每 100 ms 提交一次，因此崩溃最多丢失这个窗口。[`settings`](#设置) 的 `progress` 改变间隔；存储为远程的宿主可以提交得更少，例如 `{"partialIntervalMs": 500, "outputIntervalMs": 500}`。

## 忙碌的会话

有 run 在处理输入时会话就是忙碌的。提交给忙碌的会话会把提交排队到会话的收件箱 `docs["pi.inbox"]`（视图里）：

```python
from pi_durable import ConversationBusy

await root.submit({"type": "input", "content": "Also run the tests"}, context)  # follow-up（默认）
await root.submit({"type": "input", "content": "Use pnpm, not npm", "whenBusy": "steer"}, context)
await root.submit({"type": "write", "entry": {"kind": "app.note", "data": "user opened a file"}}, context)
try:
    await root.submit({"type": "input", "content": "Only if idle", "whenBusy": "reject"}, context)
except ConversationBusy:
    print("busy")
```

- **引导（Steers）** 放在当前工具轮之后，加入进行中的工作。
- **跟进（Follow-ups）** 放在 run 回答时，并启动下一个 run。
- **写入（Writes）** 不询问模型就追加一条 entry。只含 `kind` 和 `data` 的写入不会改变模型看到的任何东西，因此它立即被追加——即使在 run 中途——也从不排队：聊天消息一发送就显示在 transcript 中。因此它可能出现在仍排队的更早提交项之前，除非排在一个排队的 reset 之后，在那里它会等待，以免 reset 把它藏起来。带 `model`、`head` 或 `edits` 的写入像 steer 一样等待下一个边界。
- `await submission.abort(context)` 撤回一个排队的提交。
- [设置](#设置) 的 `"steeringMode": "all"` 和 `"followUpMode": "all"` 一次放入所有排队的项，而不是每轮一个。

如果 run 失败，排队的项会留在收件箱，直到下一条输入或排队的写入把它们放入，最旧的优先。

输入以 `pi.user` entry 放置。要把它显示为其他内容，给它一个带你的 kind 和 data 的 `entry`；模型仍把 `content` 视为用户消息：

```python
await root.submit(
    {
        "type": "input",
        "content": "ann: can you check the build?",
        "entry": {"kind": "chat.mention", "data": {"author": "ann", "text": "can you check the build?"}},
    },
    context,
)
# 放置为 {"kind": "chat.mention", "data": {"author": "ann", ...}, "model": [UserMessage("ann: can you check the build?")]}
```

它表现得像任何其他输入：它启动 run、引导、跟进或拒绝，并被回答。排队期间，它的收件箱项携带该 entry。kind 不能以 `pi.` 开头。

## 重置与交接

`reset()` 启动一个新的上下文。模型不再看到更旧的 entry，但它们仍留在存储中：

```python
await root.reset(None, context)  # 从零开始
await root.reset("We were fixing the flaky login test. Continue.", context)  # 从交接说明开始
```

忙碌时，reset 像 write 一样排队。当它在工具轮期间被放置时，当前 run 结束。工具可以用 `"control": {"handoff": "..."}` 请求同样的事情。

## 压缩

压缩缩小模型看到的内容：它总结较旧的 entry，并追加一个持有摘要、同时成为它保留的第一条 entry 之头的 `pi.compaction` entry。更旧的 entry 留在存储中。

```python
task_id = await root.compact("Keep the failing test names", context)  # 手动，带可选指令
outcome = (await harness.wait_for_task(task_id, context))["state"]["outcome"]
if outcome["status"] == "completed" and outcome["result"].get("submissionId") is not None:
    placed = await (await harness.submission(outcome["result"]["submissionId"], context)).wait(context)
    print(placed["status"])  # "done"，或 reason 为 "stale" 的 "unanswered"
```

制作摘要期间会话继续工作。会话空闲时摘要立即放置，否则在下一个轮边界放置。`abort()` 取消手动压缩。找不到可总结内容的压缩会以空结果完成。

生成也会自行压缩，由[设置](#设置)控制：

```python
settings = {
    "compaction": {
        "enabled": True,  # 自动压缩；手动 compact() 始终可用
        "reserveTokens": 16384,  # 超过 contextWindow - reserveTokens 时，下一个请求等待压缩
        "keepRecentTokens": 20000,  # 大致多少近期上下文保持原样
        "backgroundTokens": 32768,  # 低于它这么远时压缩在后台开始；0 关闭它
        "modelOverrides": {
            # 精确的 "provider/modelId"；此处设置的每个字段为该模型替换上面的对应值
            "local/small-32k": {"reserveTokens": 4096, "keepRecentTokens": 8000, "backgroundTokens": 4096},
        },
    },
}
```

固定的 token 计数很少能同时适合 32k 和 1M 的上下文窗口；`modelOverrides` 让一个 harness 服务两者。会话当前模型在阈值检查时、上下文溢出后、以及压缩选择要总结什么时选择该条目。

当提供商因上下文过长而拒绝请求时，生成会压缩并重试一次。会在当前上下文起点之前截断的摘要，在放置时以 `stale` 结算，因此当多个摘要同时在飞时，最远的截断保持生效。摘要花费计入 `pi.usage`。`CompactionTask` 上的 `before_compact` hook 可以拒绝（`{"decline": True}`）或提供自己的摘要（`{"summary": "..."}`）。

当生成在阻塞阈值等待的压缩失败时，生成也失败，不发请求：它的输入以 `model_error` 结算为 `unanswered`，detail 为 `Compaction failed: <error>`。找不到可压缩内容或被拒绝的压缩会让请求继续。

进行中的压缩列在 `docs["pi.live"]["compactions"]` 中，带 reason、尝试次数和重试退避。最新一次失败保留在 `docs["pi.live"]["compactionFailure"]`（task、reason、error）中，直到压缩产生摘要。agent 事件加入 `compaction_start` 和 `compaction_end`（压缩失败时带 `error`），以及快照中的 `compactions` 和 `compactionFailure` 字段。

## Agent 事件（实验性）

给想要编码 agent 风格事件（`message_start`、`message_update`、`tool_execution_start` 等）而非结构状态的消费者：

```python
from pi_durable import watch_events

stream = await watch_events(harness, root.id, context)
print(sorted(stream.snapshot))  # entries, tools, nestedTools, compactions, inbox, agent, usage, ...


def on_events(events, context):
    for event in events:
        print(event["type"])


stream.start(on_events)
# 稍后：await stream.stop()
```

事件是带上游键的 dict，由提交派生而来，每提交一批，并应用在快照之上。事件内的消息是 `pi_ai` dataclass；在 `json.dumps()` 之前使用 `pi_ai.serialization.to_json_value()`。消息和工具更新携带增量：文本和思考的追加、追加的工具调用参数文本，以及输出的修剪和追加。当消费者落后超过 100 批时，它收到一个全新的 `snapshot` 事件。

## Hooks

Hooks 让扩展观察或调整内置任务，作用于选择它们的会话。hook 表按 snake_case 的 hook 名作为键，每个处理器接受上游的参数加 `api` 和 `context`：

```python
from pi_durable import GenerationTask, ToolTask, define_extension, hook


def guard(call, api, context):
    return {"block": "bash is disabled here"} if call.name == "bash" else None


def keep_going(answer, api, context):
    text = "".join(part.text for part in answer.content if part.type == "text")
    return {"continue": "Keep going."} if "TODO" in text else None


Guard = define_extension(
    "guard",
    hooks=[hook(ToolTask, {"before_tool": guard}), hook(GenerationTask, {"on_yield": keep_going})],
)
registry.install(Guard)
```

- **Generation：** `before_request(request, api, context)`（替换一次请求的消息）、`after_response(message, api, context)`、`on_yield(answer, api, context)`（用另一条用户消息继续 run）和 `after_tools(assistant_entry_id, result_entry_ids, api, context)`（一轮的工具做完后运行一次）。
- **Tools：** `before_tool(call, api, context)`（阻塞，或重写 `arguments`）和 `after_tool(call, result, api, context)`（替换结果）。
- **Compaction：** `before_compact(compaction, api, context)`（拒绝，或提供摘要）。

要把 hook 限制到某些会话，只在那里选择它的扩展，例如 `configure({"extensions": {"add": [Guard]}})`。

## 更多会话与分叉

```python
other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
fork = await root.fork(entry_id, {"ownership": {"kind": "ownerless"}}, context)
```

分叉看到父会话直到 `entry_id` 的 entries，然后独立继续。它保留父会话在该 entry 时的 agent，但收到新的提供商会话身份。两者都在选项中以 `"agent"` 和 `"init"` 接受内容，在创建提交中应用；`root()` 以 `agent=` 和 `init=` 接受。

## 中止与子 Agent

`await root.abort(context)` 停止一个会话：排队的输入被撤回（排队的写入保留），其当前工作的每个任务都被中止，调用在会话空闲后返回。

一个会话可以由一个任务**拥有**。子 agent 工具在 `api.commit()` 内用 `{"ownership": {"kind": "task", "taskId": api.task_id}}` 创建它的子会话，然后通过 `api.conversation(id, context)` 驱动它：

```python
from pi_durable import configure


async def subagent_execute(args, api, context):
    async def find_or_create(tx):
        # 所有权索引记住子会话，因此重跑会复用它。
        existing = (await tx.scan_conversations({"ownerTaskId": api.task_id}, 1))["items"]
        if existing:
            return existing[0]["id"]
        # 以本会话 agent 的副本开始：model、extensions、tools、cwd。
        created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
        # 更便宜的模型，并且没有它自己的子 agent。
        await configure(tx, created["id"], {"model": {"provider": "openai", "modelId": "gpt-6-luna"}, "extensions": {"remove": [Subagent]}})
        return created["id"]

    child = await api.commit(find_or_create, context)
    await api.details({"conversationId": child}, context)  # 让 UI 可以挂到子会话
    conversation = await api.conversation(child, context)
    request = {"type": "input", "content": args["task"], "requestId": f"subagent:{api.task_id}"}
    settled = await (await conversation.submit(request, context)).wait(context)
    return {"output": [TextContent(settled["status"])]}


Subagent = define_extension(
    "subagent",
    tools=[
        define_tool(
            name="subagent",
            description="Delegate a self-contained task to a subagent and get its answer back.",
            parameters=Type.Object({"task": Type.String()}),
            replay="safe",  # 崩溃后的重跑找到同一个子会话和提交
            execute=subagent_execute,
        )
    ],
)
registry.install(Subagent)
```

被拥有的工作属于它的拥有者：

- 中止调用会中止子会话。调用失败也一样：`execute()` 抛异常，或崩溃中断了不可安全重放的调用。
- 只有子会话空闲后父会话才空闲。
- 以 `"background": True` 创建的任务是一个边界：它拥有的工作在父会话中止后存活，并且不会让父会话保持忙碌。`root.abort(context, background=True)` 也中止它。

## 子任务

任务可以拥有子任务，用 `{"ownership": {"kind": "task", "taskId": ...}}` 创建，并通过提交一个 `waiting` 状态等待它们：

```python
def completed(result):
    return {"status": "terminal", "outcome": {"status": "completed", "result": result}}


async def charge(task, runtime, context):
    if task["input"]["card"] == "declined":
        failed = {"status": "terminal", "outcome": {"status": "failed", "error": {"message": "card declined"}}}
        await runtime.commit(lambda tx, current: failed, context)
    else:
        await runtime.commit(lambda tx, current: completed({"card": task["input"]["card"]}), context)


async def refund(task, runtime, context):
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, context)


Payment: Task[Any, Any, Any, Any] = define_task(
    name="app.payment", version=1, initial=lambda input: {"phase": "charge"}, phases={"charge": charge}, abort=refund
)


async def pay(task, runtime, context):
    async def create_payments(tx, current):
        payments = []
        for card in task["input"]["cards"]:
            payments.append(await tx.create_task(Payment, {"card": card}, {"ownership": {"kind": "task", "taskId": task["id"]}}))
        # 每个支付都做完后在 `decide` 恢复；第一次失败中止其余。
        return {"status": "waiting", "checkpoint": {"phase": "decide", "payments": payments}, "on": payments, "policy": "failFast"}

    await runtime.commit(create_payments, context)


async def decide(task, runtime, context):
    outcomes = await runtime.outcomes(task["state"]["checkpoint"]["payments"], context)
    paid = all(outcome["status"] == "completed" for outcome in outcomes)
    await runtime.commit(lambda tx, current: completed({"paid": paid}), context)


Checkout: Task[Any, Any, Any, Any] = define_task(
    name="app.checkout", version=1, initial=lambda input: {"phase": "pay"}, phases={"pay": pay, "decide": decide}, abort=refund
)
registry.install(define_extension("shop", tasks=[Checkout, Payment]))

checkout = await root.commit(lambda tx: tx.create_task(Checkout, {"cards": ["visa", "amex"]}, {"ownership": {"kind": "conversation"}}), context)
print((await harness.wait_for_task(checkout, context))["state"]["outcome"])  # {'status': 'completed', 'result': {'paid': True}}
```

- **等待：** 等待期间任务不运行任何代码。用 `allSettled` 时，`on` 中的每个任务都做完后它恢复；用 `failFast` 时，第一个失败的子任务也会中止其他任务。`on` 也可以用 `allSettled` 指名其他任务。
- **收尾：** 在它拥有的工作仍在运行时完成的任务处于 `completing`：它的结果已定，但只有等工作做完后它才成为 terminal，`wait_for_task()` 才返回。失败或中止的结果会先中止那部分工作。
- **中止：** 中止自底向上运行。中止一个任务会先中止它拥有的工作，它自己的 abort 处理器只在那个工作做完后启动，因此每个任务撤销自己的影响。

## 任务图

`harness.task_graph(context)` 把 Session 的每个存活任务显示为一个 Chord 状态，供任务面板或调试使用。每个节点都有它的拥有者边（`owner` 任务，或其会话拥有的任务为 none）、它的状态、它是否 `background` 或被标记中止、它无法运行时的 `blocked` 原因，以及它拥有的会话。`harness.watch_task_graph(context)` 像会话的 `watch()` 一样以 watch 投递同一个值。

```python
graph = await harness.task_graph(context)


def show(value, context, delivery):
    for node in value["tasks"].values():
        state = node["state"]
        status = f"waiting on {state['on']}" if state["status"] == "waiting" else state["status"]
        print(node["id"], node["kind"], status, node.get("owner", f"conversation {node['conversationId']}"))


graph.subscribe(show)
# 稍后：graph.dispose()
```

任务随创建它的提交出现，随让它 terminal 的提交离开。状态是已提交的：`pending`、`running`、`waiting`（带 `on` 和 `policy`）、`sleeping`（带 `until`）和 `completing`（带持有的结果的 status）；每个节点还携带它的检查点 `phase`。重启后，`running` 的任务在再次运行前显示为 `pending`。图只列出存活任务：一旦子 agent 的拥有者任务 terminal，它的会话中较晚的任务就是顶层节点，会话的 `ConversationRecord["owner"]`（也在其视图的 `conversation` 中）把它连到父任务。`harness.inspect(context)` 报告同样的存活任务以及调度器会对每个任务做什么。

## 你自己的状态

文档是与 entries 一起提交的带类型 JSON 对象。定义一个，并在提交中编辑它；`tx.doc()` 返回一个你在原处修改的草稿：

```python
from typing import TypedDict

from pi_durable import ConversationDocToken, define_doc


class Todos(TypedDict):
    items: list[str]


TodosDoc: ConversationDocToken[Todos] = define_doc(
    kind="app.todos",
    version=1,
    scope="conversation",
    history="latest",  # 或 "rewindable" 以用 snapshot_as_of() 读取旧值
    fork="initial",  # 分叉从什么开始："initial"、"current" 或 "asOf"
    initial=lambda: {"items": []},
)


async def add_todo(tx):
    (await tx.doc(TodosDoc, root.id))["items"].append("write docs")


await root.commit(add_todo, context)
print(await harness.snapshot(TodosDoc, root.id, context))  # {'items': ['write docs']}
```

`harness.watch_doc()` 和 `harness.document_state()` 像上面的视图一样观察一个文档。`conversationCreated(tx, conversation)` 选项在每次创建或分叉会话的提交中运行，包括工具的原始 `tx.create_conversation()`，因此每个会话都得到你的文档；`create_conversation()` 和 `fork()` 选项中的 `"init"`，以及 `root()` 的 `init=`，在同一提交中写入每次调用的数据。扩展的工具、段和 hooks 通过 `api`（`api.snapshot(...)`）或 `input["read"]` 读取它们自己的文档，并把缺失的当作其默认值。`define_doc_family()` 定义按键索引的文档，每个 key 一个。

## 用量与成本

每个会话在 `docs["pi.usage"]` 中保留 token 和成本总计：模型响应按 `provider/model`，报告用量的工具结果按工具名。失败和中止的尝试也计数。对整个 Session：

```python
usage = await harness.usage(context)  # {"models": {"openai/gpt-6-sol": {...}}, "tools": {...}}
print(usage["models"]["openai/gpt-6-sol"]["totalTokens"])
```

总计是作为 camelCase JSON 的 `pi_ai` `Usage`（`input`、`output`、`cacheRead`、`cacheWrite`、`totalTokens`、`cost`）。

## 存储

| 后端 | 导入 | 说明 |
|---|---|---|
| Memory | `pi_durable` 的 `MemoryStorage` | 什么都不持久化。 |
| SQLite | `pi_durable.storage.sqlite.stdlib` 的 `await open_sqlite_storage(path)` | 标准 `sqlite3` 模块上的单个数据库文件，在它自己的工作线程上运行。WAL 模式带 `synchronous = NORMAL`：提交在进程崩溃后存活；最新的提交可能在断电或宿主故障时丢失。 |
| JSONL | `pi_durable.storage.jsonl.local` 的 `await open_local_jsonl_storage(directory, context)` | 一个目录中的追加文件。传 `fsync=True` 在每次提交标记前刷新。它还在内存中保留整个历史，因此内存随其增长；长期运行的 agent 请用 SQLite。 |

两种文件格式都是 TypeScript 包的：任一方实现都能打开另一方写出的东西。

SQLite 存储在共享数据库时可以给表和索引加命名空间：向 `SqliteStorage.open(database, prefix=...)` 或 `open_sqlite_storage(path, prefix=...)` 传 `prefix="pi_"`。存储把带前缀的名字渲染进它的 SQL，因此适配器无需支持。前缀必须是 SQLite 标识符，且不能以 `sqlite_` 开头。更改前缀不会迁移已有数据。要共享数据库与宿主表，自己打开数据库门面，把它同时用于 Pi 和所有宿主查询：

```python
from pi_durable.storage.sqlite import SqliteStorage
from pi_durable.storage.sqlite.stdlib import open_sqlite_database

database = await open_sqlite_database("./app.sqlite")
pi_storage = await SqliteStorage.open(database, prefix="pi_")
await database.exec("CREATE TABLE IF NOT EXISTS host_notes (text TEXT)")
await database.run("INSERT INTO host_notes (text) VALUES (?)", "a note")
```

关闭 Harness 会关闭它的存储，连同数据库。

一个存储一次只能由一个进程拥有；没有跨进程锁。可移植核心（`pi_durable.storage.sqlite.SqliteStorage`、`pi_durable.storage.jsonl.JsonlStorage`）运行在任何异步 `SqliteDatabase` 门面（`pi_chord.sqlite`）或任何 `FileSystem`（`pi_durable.env`）之上：`await JsonlStorage.open(directory, fs, context)`。

SQLite 门面实现异步的 `exec`、`run`、`get`、`all`、`transaction` 和 `close`。`run`、`get` 和 `all` 接受 SQL 文本加位置绑定；行是按键列名的 dict。事务回调收到一个事务句柄；事务中的所有工作都必须使用它，句柄在回调返回时过期。门面会把无关操作和其他事务排队到事务结束，因此在回调内部调用 `database` 本身永远不会返回：

```python
async def create_example(transaction):
    await transaction.exec("CREATE TABLE example (value TEXT)")
    await transaction.run("INSERT INTO example (value) VALUES (?)", "stored atomically")


await database.transaction(create_example)
print(await database.all("SELECT value FROM example"))  # [{'value': 'stored atomically'}]
```

自定义后端可以在 pytest 下运行共享一致性套件。`register_storage_conformance()` 在调用它的模块中定义一个参数化测试：

```python
# test_my_storage.py
from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable.testing import register_storage_conformance


async def with_storage(use):
    storage = await open_my_storage()
    try:
        await use(storage)
    finally:
        await storage.close(BACKGROUND_CONTEXT)


register_storage_conformance(globals(), "My Storage", with_storage)
```

自定义 `Storage` 实现 `pi_durable.types` 中的协议（`commit`、`mint_id`、记录读取与扫描、`close`）；`packages/durable/DESIGN.md` 描述它。

## 错误

三条规则决定错误做什么：

1. **存储错误结束 Harness。** 碰到它的调用抛出该错误。其后的每个东西和每个挂起的 wait 都抛出 `SessionFailed`（其 `cause` 和 `__cause__` 都是那个错误）；watch 和事件流以 `session_failed` 结束；运行中的任务收到它们的中止信号；`onReport` 收到该错误一次。然后 Harness 自行关闭。什么都不重试；自定义存储重试自己的瞬时失败。重新打开以恢复：提交是原子的，因此存储是一致的，用相同 `requestId` 重新提交会找到一个已被收录的提交。
2. **无效读取只让那次调用失败：** 存储读取对坏游标、未知会话或文档不保留的历史抛出 `StorageRequestError`。你取消的读取也只让它自己失败。豁免只覆盖读取：`commit()` 或 `mint_id()` 的任何错误（包括 `StorageRequestError`）都像规则 1 那样结束 Harness。
3. **你的代码只让它自己的单元失败：** 抛出的任务阶段让该任务 fault，抛出的工具得到错误结果，抛出的 hook 被报告并跳过（`before_tool` 阻塞该调用），抛出的提交回调回滚那次提交。抛出异常或返回失败 awaitable 的监听器会被报告，其他监听器仍会运行。例外是宿主 `RegistryReader` 的 `snapshot()` 抛出：没有它 Harness 无法调度，因此它像存储错误一样让 Harness 失败。

`harness.closed` 是一个在 Harness 关闭（通过 `close()` 或在存储错误后）后落定的 future，因此宿主这样在失败时重启：

```python
async def supervise(harness):
    end = await harness.closed
    if end["reason"] == "failed":
        await reopen(end["error"])  # 你的代码
```

失败会让 Harness 自行关闭；一旦关闭完成 `closed` 就落定，即使存储无法干净关闭。关闭会等待运行中的任务代码返回，因此忽略其中止信号的代码会推迟 `closed`。

## 遥测

传一个 [`pi_telemetry`](../telemetry/README.md) 上下文作为 `telemetryContext`（这是新增：上游 1.3.0 的 pi-durable 不记录 span）。Harness 随后为一次生成的每个阶段记录一个 `pi.harness.generation` span（其模型请求是子级 `pi.ai.request` span），为每次工具调用记录一个 `pi.harness.tool` span（嵌套调用是其调用者 span 的子级；代码模式工具的脚本运行是子级 `pi.codemode.run`），为任何其他任务的每个阶段记录一个 `pi.harness.task` span，并为每次提交记录一个 `pi.session.commit` span。`create_session(storage, telemetry_context=...)` 记录普通 Session 的提交 span。

```python
from pi_telemetry import InMemoryTelemetryContext

telemetry = InMemoryTelemetryContext()
traced = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "telemetryContext": telemetry}, context)
traced_root = await traced.root(context, agent={"model": {"provider": "openai", "modelId": "gpt-6-sol"}})
await (await traced_root.submit({"type": "input", "content": "Hi"}, context)).wait(context)
await traced.close(context)
print(sorted({span.name for span in telemetry.get_spans()}))
```

任务代码把运行中阶段的 span 作为 `runtime.telemetry_context` 拿到，工具作为 `api.telemetry_context` 拿到：把它传给你想让其嵌套在下的工作，例如作为模型请求的 `telemetry_context`。上下文只在内存中：它永远不会写进记录、文档、检查点或日志，重启后恢复的任务在重新打开 Harness 的进程的上下文下运行。默认没有 span 记录提示、工具参数或输出、或错误文本；参见[span 目录与数据策略](../telemetry/README.md#span-catalog)。

## 与 TypeScript 包的差异

- **名字。** 函数、方法、属性和关键字参数是 snake_case（`createConversation` 是 `create_conversation`，`structuredOutputSchema=` 是 `structured_output_schema=`）；dict 键保留上游的 camelCase，包括 `"onReport"` 和 `"setTimer"` 这样的选项键。hook 表以 snake_case 的 hook 名作为键。
- **定义**（`define_tool`、`define_task`、`define_doc`、`section`、`hook`）以上游的字段作为关键字参数，并返回 dataclass 或 token 对象，而非普通对象。工具 schema 是 JSON Schema dict（`pi_ai.Type`）或 Pydantic 模型，而不是 TypeBox；`args` 不由 schema 类型化。
- **可选尾部选项**成为仅关键字参数：`root(context, agent=..., init=...)`、`abort(context, background=True)`、`execute_tool(name, args, context, key=..., progress=...)`、`context(context, at=...)`、`section(key, render, tag=False)`。
- **Future。** 返回 Promise 的方法是 `async def`，或在上游的工作在调用时开始的场合返回已启动 future 的普通函数（见[两条 Python 规则](#两条-python-规则)）。`closed` 属性是 future。
- **回调**按位置拿到上游的参数：状态订阅者是 `listener(value, context, delivery)`，watch 监听器是 `listener(value, ops, context)`，事件监听器是 `listener(events, context)`，运行时提交 change 是 `change(tx, current)`，段是 `render(input, context)`。状态订阅者和 watch 监听器可以声明更少的参数，如 JavaScript 那样（`lambda value: ...` 只拿值）；其他回调必须接受全部参数。每个都可以是同步或异步的。
- **定时器**在 asyncio 循环上运行，本身从不保持进程存活；没有最大延迟，也不像 Workers 那样拒绝任何东西。
- **错误**保留上游的类和消息。裸 JavaScript `Error` 是 `RuntimeError`，`RangeError` 是 `ValueError`。
- **存储和环境模块**：`openNodeSqliteStorage` 是 `open_sqlite_storage`（在 `sqlite3` 上），`openNodeJsonlStorage` 是 `open_local_jsonl_storage`，`NodeExecutionEnv` 是 `LocalExecutionEnv`（id `"local"`），`Storage.entry(conversationId, id)` 是 `visible_entry()`，`FileSystem.list_files()` 是可能返回 `not_supported` 的必需方法。一致性套件用 pytest 注册。
- **二进制数据**是 `bytes`；ID 是普通 `int`（`TaskId`、`EntryId` 等是类型检查器品牌）。
- **工具与图像**：`createCodingTools({ images })` 是 `create_coding_tools(images=...)`，`createBashTool({ commandPrefix, prepare })` 是 `create_bash_tool(command_prefix=..., prepare=...)`（`prepare` 改变 `BashExecution` dict 的 `command`、`cwd`、`env` 和 `inheritEnv`），`createNodePhotonImages()` 是 `create_pillow_images()`，用 Pillow 而非 Photon，在无空闲超时的工作线程中。`read` 对 `offset` 和 `limit` 保留 JavaScript 的数字语义，因此分数、负数和巨大的值会选择并打印相同的行。
- **代码模式**在 Monty 中运行 Python 脚本而非 QuickJS 中的 JavaScript：`createCodemodeExtension({ sandbox })` 是 `create_codemode_extension(sandbox=...)`，脚本全局变量是 snake_case（`search_tools()`、`models.generate_images()`），一种语言日志化的 run 不会在另一种中恢复（见[代码模式](#代码模式)）。
- **尚未移植**：Cloudflare 适配器（见[本移植包含的内容](#本移植包含的内容)）。

## 设计说明

[`DESIGN.md`](DESIGN.md) 是移植契约：每个 TypeScript 类型如何映射到 Python、持久化 JSON 规则、Storage 协议，以及 Session 与 Harness 内部。上游的[规范](https://github.com/earendil-works/pi/blob/main/packages/durable/docs/spec.md)是行为的规范性描述。

## 许可证

MIT
