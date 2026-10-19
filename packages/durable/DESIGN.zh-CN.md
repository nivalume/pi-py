# pi_durable 设计说明

把 `@earendil-works/pi-durable` 1.3.0 移植到 Python 的契约。阶段 1（这个基础）确定了下面的规则；后续阶段遵循它们，并在加入新规则时扩展本文件。

## 1. 模块布局

上游的 `src/x/y-z.ts` 变成 `pi_durable/x/y_z.py`。阶段 1 移植了：

| 上游 | Python |
|---|---|
| `types.ts` | `pi_durable/types.py`：records、tokens、`Tx`、`Session`、`TaskRuntime`、`Storage` |
| `ids.ts`、`errors.ts`、`timers.ts`、`tasks.ts`、`entries.ts`、`documents.ts`、`search.ts`、`truncate.ts` | 同名 |
| `harness/types.ts`、`harness/define.ts`、`harness/json.ts` | `pi_durable/harness/{types,define,json}.py` |
| `env/index.ts` | `pi_durable/env/types.py`，由 `pi_durable/env/__init__.py` 再导出 |
| `env/decode.ts`、`env/line-scan.ts` | `pi_durable/env/{decode,line_scan}.py` |
| `storage/memory.ts`、`storage/scan.ts` | `pi_durable/storage/{memory,scan}.py` |
| （新增） | `pi_durable/storage/codec.py`：用于序列化后端的记录 JSON 编码 |

`pi_durable/__init__.py` 对应上游 `index.ts` 中已存在的内容；后续阶段增加它们的导出。测试在 `packages/durable/tests/test_<上游名>.py`。

## 2. TypeScript 类型如何映射到 Python

**每种类型一条规则。**

| TypeScript | Python | 示例 |
|---|---|---|
| 数据的对象类型（records、writes、queries、pages、outcomes、settings、选项对象、`Result`） | `TypedDict`，其键**恰好是上游的属性名**，含 camelCase | `{"id": 2, "conversationId": 1, "kind": "pi.user"}` |
| `readonly` 属性 | `ReadOnly[...]`（来自 `typing_extensions`，在 `TYPE_CHECKING` 下导入） | `id: ReadOnly[EntryId]` |
| `x?: T` | `NotRequired[T]`：缺失键 = JS 的 `undefined`；`None` = JSON 的 `null` | `head: ReadOnly[NotRequired[EntryId]]` |
| `T \| undefined` 返回 | `T \| None` | `async def task(...) -> AnyTaskRecord \| None` |
| 判别联合 | 共享一个 `Literal` 键的 `TypedDict` 联合；匿名成员命名为 `<Union><Variant>` | `TaskStateRunning`、`SubmissionRecordInputDone`、`StorageWriteEntry` |
| 带方法的接口（`Storage`、`Session`、`Tx`、`TaskRuntime`、`WatchHandle`、`FileSystem`、`Conversation`、`Harness`） | 带 snake_case 方法和属性的 `Protocol` | `storage.scan_entries(query, limit, cursor, context)` |
| 应用定义的内容（doc/entry/task tokens、`Extension`、`ToolRegistration`、`PromptSection`、`Wrap`、`HookRegistration`） | 带 snake_case 属性的槽位类 / 冻结 dataclass，由接受上游字段作为关键字参数的 `define_*` 辅助函数构建 | `define_doc(kind=..., checkpoint_when=...)` |
| Hook 表（`GenerationHooks`、`ToolHooks`、任务的 `H`） | 以 **snake_case** hook 名为键的处理器的 `TypedDict(total=False)`（方法表，不是数据） | `hook(ToolTask, {"before_tool": guard})` |
| `Id<Kind>` 品牌 | `NewType` over `int`：`ConversationId`、`EntryId`、`SubmissionId`、`DocumentId`、`Seq`；`TaskId[R]` 是供类型检查器用的泛型 `int` 子类，运行时是普通 `int` | `EntryId(5)`、`TaskId(4)`、`id_from_number(4)`（返回 `Any`） |
| records 中的 `JsonValue` payload（entry `data`、任务 input/checkpoint/result、memos、`detail`） | `Any`（运行时必须是严格 JSON）；上游指存储内部 JSON 的地方用 `pi_chord.json` 的 `JsonValue`/`JsonObject`（文档值、游标） | |
| records 内的 `Message`（`EntryRecord["model"]`、`ContextEdit["messages"]`、`ContextView`） | `pi_ai` 消息 dataclass | `"model": [UserMessage("hi")]` |
| `Op` | `pi_chord.delta.Op` 元组（`("s", ("count",), 2)`） | |
| `Promise<T>` 方法 | `async def`；**急切**的（§5）是返回已启动 future 的 `def ... -> Awaitable[T]` | `storage.commit(...)`；`session.commit(...)` |
| `Promise<T>` 属性（`closed`） | 返回可 await 对象的属性（`asyncio.Future`，可多次 await） | `await session.closed` |
| `T \| Promise<T>` 回调结果 | `T \| Awaitable[T]`；实现用 `inspect.isawaitable(result)` 决定是否 `await` | `commit(change, context)` |
| `JsonRepresentation<T>` | 没有计算类型：值类型就是 `TypedDict` 本身，且值必须是严格 JSON（`pi_chord.json.copy_json` 接受它们） | |

泛型记录：`TaskRecord[I, S, R]`、`RunningTask[I, S, R]`、`TaskState[S, R]`、`TaskOutcome[R]`、`SettledTask[R]`、`TypedEntry[D]`、`Page[T]`（上游 `Page<T, Cursor>`；游标类型始终是 `Cursor`）。`AnyTaskRecord` 是 `TaskRecord[Any, Any, Any]`（上游 `TaskRecord<JsonValue, ...>`）。

处理记录：

```python
running: TaskStateRunning[Checkpoint] = {"status": "running", "checkpoint": {"phase": "run"}}
updated: AnyTaskRecord = {**task, "state": running}          # 上游 {...task, state}
if write["type"] == "entry":                                 # 在下标本身收窄：
    write["value"]["conversationId"]                         # `kind = write["type"]; if kind == ...` 不会收窄
if result["ok"] is False:                                    # env Result：对字面量的同一性收窄有效，
    return result                                            # 真值（`if not result["ok"]`）无效
```

上游在类型中编码、Python 却没有的不变量：`memos` 只在存活任务记录上；`latest` 历史永不与 fork `asOf` 配对（`define_doc` 在运行时检查）；阶段映射是穷尽的；无数据的 kind（`UserEntry`）仍以 `TypedEntry["data"]` 类型化（不要读取它）。

## 3. 持久化 JSON 兼容性

一条 record **就是**它持久化的 JSON：字段名和值都是上游的（`conversationId`、`abortRequested`、`createdAt`、`deltasSinceBase`、`"document.create"`、`"asOf"` 等）。绝不要在线上重命名键。例外——全部由 `pi_durable.storage.codec` 处理：

- `EntryRecord["model"]` 和 `ContextEdit["messages"]` 中的**消息**用 `pi_ai.serialization` 编码（camelCase、`role` 判别符），正是 TypeScript pi 写出的。
- **Ops** 在内存中是元组，在磁盘上是 JSON 数组（`decode_ops` 用元组路径恢复元组）。
- 后端持久化 `encode_storage_write(write)` / `encode_entry(entry)`，并返回 `decode_entry(row)`；record 中的其他一切都是严格的 JSON。

在 JSON payload 内部（任务 checkpoint/result、文档值、entry `data`、工具 `details`），Python 中任何 dataclass 的东西都必须以 JSON 存储：消息和 `Usage` 用 `pi_ai.serialization.to_json_value`，工具 schema 用 `pi_ai.utils.schema.parameters_schema`（`pi_ai.Type` 返回 `copy_json` 拒绝的 `dict` 子类）。`ToolRegistration` 永不进入 transcript；写入 `registration.to_tool()`。

与上游不同的 Python 面向名字：snake_case 的函数、方法、属性和关键字参数（`mint_id`、`find_latest_head_marker`、`checkpoint_when`、`execution_mode`）；dict 键永远不会。时间是以毫秒计的纪元时间（注解为 `float`；像 `Date.now()` 那样写 `int`）。

## 4. 声明 entries、documents、tasks、tools 和 extensions

```python
from typing import Any, TypedDict
from pi_ai import Type
from pi_durable import (ConversationDocToken, Entry, Task, define_doc, define_entry, define_extension,
                        define_task, define_tool, hook, section)

class NoteData(TypedDict):
    text: str

Notes: Entry[NoteData] = define_entry("app.note")              # 标注以类型化 `data`

class Todos(TypedDict):
    items: list[str]

TodosDoc: ConversationDocToken[Todos] = define_doc(
    kind="app.todos", version=1, scope="conversation",
    history="latest",                                         # 或 "rewindable" 以用 snapshot_as_of()
    fork="initial",                                           # "initial"、"current" 或 "asOf"
    initial=lambda: {"items": []},
)
# await root.commit(lambda tx: ..., context)；内部：(await tx.doc(TodosDoc, root.id))["items"].append("x")

class ReminderInput(TypedDict):
    text: str
    at: float

async def wait(task, runtime, context):                      # handler(task, runtime, context)
    await runtime.sleep_until(task["input"]["at"], {"phase": "deliver"}, context)

async def deliver(task, runtime, context):
    await notify(task["input"]["text"])
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "completed", "result": None}}, context)

async def abort(task, runtime, context):
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, context)

Reminder: Task[ReminderInput, Any, None, Any] = define_task(  # 标注：R 不出现在任何参数中
    name="app.reminder", version=1, initial=lambda input: {"phase": "wait"},
    phases={"wait": wait, "deliver": deliver}, abort=abort,
)

async def count_execute(args, api, context):
    for i in range(1, args["n"] + 1):
        api.output(f"{i}\n")
    return {}                                                 # ToolExecutionResult：上游键

count = define_tool(name="count", description="Count from 1 to n",
                    parameters=Type.Object({"n": Type.Number()}), execute=count_execute)
Coding = define_extension(
    "coding", tools=[count],
    sections=[section("preamble", lambda input, context: "You are a concise coding assistant.", tag=False)],
    hooks=[hook(ToolTask, {"before_tool": lambda call, api, context: {"block": "Needs approval"} if dangerous(call) else None})],
)
```

Schema 是用 `pi_ai.Type` 构建的 JSON Schema dict（或手写，或在 pi-ai 接受 Pydantic 模型作为工具参数的地方用 Pydantic 模型）；`Static<T>` 没有 Python 对应物，因此 `args` 是 `Any`。`wrap_tool(tool, lambda t: dataclasses.replace(t, execute=...))` 是上游的 `{...tool, execute}`。

## 5. 上下文、取消与异步边界

- 上游每个接受 `context` 的操作都在**相同位置**接受一个 `pi_chord.context.Context`（通常最后一个位置参数）。用 `context.throw_if_aborted()` 检查它；用 `await_with_context(awaitable, context)` 等待；用 `with_abort_signal`/`without_abort_signal` 派生（必须清理）；根处用 `BACKGROUND_CONTEXT`。run 的信号（`TaskRuntime.signal`）是一个 `pi_ai.abort.AbortSignal`。
- 上游 `async`/返回 Promise 的函数是 asyncio 上的 `async def`；同步的保持同步（`Tx.settle_submission`、`Tx.place_submission`、`subscribe_commits`、`WatchHandle.start`、`TaskRuntime.now/set_timer/report`、`Registry.install`）。commit/close 监听器是同步的，不能阻塞。为排序而保持未 await 的 TS Promise（例如 `closed`）变成在当前循环上创建的 `asyncio.Future`。绝不要阻塞循环。
- **急切操作**（阶段 2a）。JavaScript `async` 函数同步运行到第一个 `await`，返回 Promise 的方法在返回前完成它的工作；Python 协程在 await 之前什么都不做。在上游语义依赖那个同步部分的地方（进入 Session 行、关闭时的密封、读后写规则、事务内的暂存顺序），协议声明 `def ... -> Awaitable[T]`，实现运行同步部分并在调用时返回一个已启动的 `asyncio` future（同步部分错误变成失败的 future，如被拒绝的 Promise）。这覆盖上游以 Promise 返回的每个 `Tx` 方法、`Session.commit`、`close`、`document_state`、`DocumentReader.snapshot`/`snapshot_as_of`、`DocumentObserver.watch_doc` 和 `WatchHandle.stop`。调用者只需 `await`；要启动一个而不 await，保留返回的 future（用 `asyncio.ensure_future` 而非 `create_task` 包装它）。当没有东西依赖急切性时，实现仍可以是 `async def`（协程就是 `Awaitable`），例如 `TaskRuntime` 和扩展 `DocumentReader` 的 harness API 可以。`Storage` 保持 `async def`。
- 类型为 `T | Promise<T>` 的回调接受同步或异步函数；调用者 await 可 await 对象。
- 定时器（`SetTimer`、`host_timers`、`native_timer`）在循环上运行（`loop.call_later`）；定时器回调是同步的。
- 可选尾部选项对象成为仅关键字参数（`section(key, render, *, tag=None)`、`truncate_head(content, *, max_lines=None)`、`TaskRuntime.context(cid, context, *, at=None)`、`Conversation.abort(context, *, background=False)`、`execute_tool(name, args, context, *, key=None, progress=None)`、`Harness.root(context, *, agent=None, init=None)`）。必需或中间位置的选项对象保持 `TypedDict`（`TaskOptions`、`ConversationCreateOptions`、`Tx.create_conversation` 的 `{"ownership": ...}`、`FileSystem.read_text_lines` 的 `ReadTextLinesOptions | None`）。
- 重载保留上游的参数列表，在协议中是仅位置参数，因此实现可以像上游的 `resolve_address(definition, args)` 那样分发：`async def doc(self, token, *args)`；`memo(name, context)` 与 `memo(name, candidate, context)` 用 `isinstance(last, Context)` 区分。
- 错误保留上游的类和消息（`pi_durable.errors`、`FileError`、`ExecutionError`）。裸 JS `Error` 是 `RuntimeError`，`TypeError` 是 `TypeError`，`RangeError` 是 `ValueError`；`throw new Error(msg, { cause })` 是 `raise RuntimeError(msg) from cause`。`SessionFailed.cause` 也是 `__cause__`。

## 6. Storage 协议（`pi_durable.types.Storage`）

```python
class Storage(Protocol):
    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq
    async def mint_id(self) -> int                                  # 打品牌：EntryId(await storage.mint_id())
    async def conversation(self, id, context) -> ConversationRecord | None
    async def scan_conversations(self, query, limit, cursor, context) -> Page[ConversationRecord]
    async def entry(self, id, context) -> StoredEntry | None        # {"entry": ..., "commitSeq": ...}
    async def visible_entry(self, conversation_id, id, context) -> StoredEntry | None   # 上游 entry(cid, id)
    async def find_latest_head_marker(self, conversation_id, at_or_before_entry_id, context) -> HeadMarker | None
    async def scan_entries(self, query, limit, cursor, context) -> Page[EntryRecord]
    async def task(self, id, context) -> AnyTaskRecord | None
    async def scan_tasks(self, query, limit, cursor, context) -> Page[AnyTaskRecord]
    async def submission(self, id, context) -> SubmissionRecord | None
    async def scan_submissions(self, query, limit, cursor, context) -> Page[SubmissionRecord]
    async def submission_by_request(self, conversation_id, request_id, context) -> SubmissionRecord | None
    async def find_document(self, address, at, context) -> DocumentRecord | None
    async def document(self, id, at, context) -> StoredDocument | None
    async def scan_documents(self, query, limit, cursor, context) -> Page[DocumentRecord]
    async def close(self, context) -> None
```

- 一次 commit 是一个 `StorageWrite` dict 列表（`{"type": "entry", "value": record}`、`{"type": "document.change", "id": id, "content": {"kind": "delta", "version": 1, "ops": ops}}` 等），原子应用；它返回 commit 的 `Seq`。`DocumentPoint` 是 `Seq` 或 `"current"`。
- 唯一的改名：上游重载的 `entry(conversationId, id, context)` 是 `visible_entry`。
- 内置后端的游标是 `{"after": id, "order": order}`（`storage.scan.next_cursor`；`scan_start` 校验它们并抛 `StorageRequestError`）。
- 分离：后端复制它保留的和返回的内容（内存中是 `storage.codec.clone_value`，其他后端序列化）。`MemoryStorage.prepare_commit(writes, seq=None)` 返回带 `.seq`、`.writes`（分离且保留：绝不修改）和 `.apply()` 的 `PreparedMemoryCommit`，供 JSONL 后端使用，如上游。
- 错误：`StorageRequestError` 用于无效读取（无持久影响）；其他任何错误对 Session 都是致命的。ID 耗尽超过 `ids.MAX_SAFE_INTEGER` 抛 `RuntimeError("ID space is exhausted")`。
- `MemoryStorage` 静态满足协议（`tests/typing_cases.py`），是参考实现。

## 7. 后续阶段的说明

- （阶段 3 完成。）`harness/types.py` 在 `TYPE_CHECKING` 下从 `harness/view.py`、`task_graph.py`、`usage.py` 导入 `ConversationView`、`TaskGraph`、`TaskGraphWatch`、`UsageState`，并通过模块 `__getattr__` 懒导出它们（那些模块导入 `harness/types.py`）。
- `GenerationContinue` 是 `{"continue": input}`（函数式 `TypedDict`，`continue` 是关键字）。
- `ToolHookCall` 以仅关键字的 `parent` 子类化 `pi_ai.ToolCall`。
- `ToolRegistration` 子类化 `pi_ai.Tool`；它按同一性哈希和比较。
- `FileSystem.list_files` 是必需方法（上游可选）：无法列目录的环境返回 `err(FileError("not_supported", ...))`。
- `get_or_undefined` 是 `get_or_none`；二进制数据是 `bytes`。
- 静态检查：在 `tests/typing_cases.py` 中加 `# pyright: ignore` 行（预期错误）和 `assert_type` 调用；`tests/test_types.py` 在其上运行 pyright。

## 8. Session（阶段 2a）

`pi_durable/session/` 移植 `src/session/`：`create_session(storage, *, now=None, document_retention_ms=None, set_timer=None) -> Session`、具体类 `SessionImpl` 和 `Transaction`，以及观察者 `CommittedStateSource`（一个 `pi_chord.state.ReplicatedStateSource`）和 `CommittedWatch`（一个 `WatchHandle`）。文档标记为 *internal*（上游 "Internal:"）的方法服务于 Harness 和测试；标记为 *protected*（上游 `protected`）的方法意在被子类覆盖，正如 Harness 子类化 `SessionImpl`；两者都保留上游的名字，snake_case，不带前导下划线：

- `SessionImpl(storage, now=None, document_retention_ms=None, set_timer=None)`：像上游一样位置参数；`document_retention_ms` 是每次使用都读取的 callable。
- Internal：`commit_with(change, context, scope=None)`（scope 是 `TransactionScope` TypedDict `{"conversationId", "taskId"}`）、`read_on_line(job)`、`conversation_document_on_line(token, cid, context)`（返回 `(record, version, value)` 或 `None`）、`observe_commits`、`fail(error)`、`failure`（带 `.error` 的 `SessionFailure`）、`failed`（以 `SessionFailed` 失败的 future）、`assert_usable()`、`report(error, info)`、`unload_documents()`、`document_cache_sizes`（`DocumentCacheSizes` 命名元组）。
- Protected：`storage`（受守卫的 Storage）、`conversation_created(tx, record)`（异步）、`before_close()`（异步）、`write_at_close(writes)`。
- `Transaction` 内部：`create_root_conversation()`、`submission(id)`、`set_task(record)`（同步）、`staged_tasks()`、`staged_conversations()`、`settle_failure/settle_success/discard/adopt`。
- 模块辅助：`contained(callback, on_error)`、`document_retention(ms)`、`set_default_document_retention(ms)`（测试；`tests/conftest.py` 中的 `PI_DURABLE_DOCUMENT_RETENTION_MS`）、`date_now()`（`Date.now()` 作为 `int`）。

规则：

- 投递（状态帧、watch 帧）用 `loop.call_soon` 调度（上游 `queueMicrotask`）；watch 监听器和状态订阅者可以同步或异步。
- Session 为发布创建的 Op 列表是列表（`RETIREMENT_OPERATIONS == [("r", None)]`，创建/退役 `ops == []`），按契约不可变。
- `Tx.append_entry` 用 `copy_json` 复制 entry JSON，并通过 `storage.codec.clone_value` 分离 pi-ai 消息（`model`、edit `messages`）；表记录中的其他一切都是严格 JSON。
- 点名 API 的错误文本使用 Python 名字（`Tx.scan_conversations() cannot read tables after the first table write`、`Tx.create_task() requires options.conversationId`、`Session.snapshot_as_of() requires an entry ID`）。
- `math.inf` 的保留时间不布下定时器。

## 9. 存储后端、本地环境与测试套件（阶段 2b）

| 上游 | Python |
|---|---|
| `storage/sqlite/{schema,migrations,storage,index}.ts` | `pi_durable/storage/sqlite/{schema,migrations,storage,__init__}.py`（可移植：不加载驱动） |
| `storage/sqlite/node.ts` | `pi_durable/storage/sqlite/stdlib.py`：`open_sqlite_storage(path, *, prefix=None, wal_auto_checkpoint_pages=None, busy_timeout_ms=None)`，再导出 `pi_chord.sqlite.stdlib` |
| `storage/jsonl/{storage,index}.ts` | `pi_durable/storage/jsonl/{storage,__init__}.py` |
| `storage/jsonl/node.ts` | `pi_durable/storage/jsonl/local.py`：`open_local_jsonl_storage(directory, context, *, fsync=False)` |
| `env/node.ts`、`env/node-watch.ts` | `pi_durable/env/local.py`（`LocalExecutionEnv`）、`pi_durable/env/local_watch.py`（`LocalFileWatcher`、`LocalWatchOptions`） |
| `testing/*.ts` | `pi_durable/testing/*.py`；`register_storage_conformance(globals(), name, provider)` / `register_env_conformance(...)` 定义一次参数化 pytest 测试 |

规则：

- **线上文本。** 序列化后端写入 `storage.codec.json_stringify(value)`，即上游的 `JSON.stringify`：紧凑、非 ASCII 原样、孤立的 UTF-16 代理对转义（`"\ud800"`），因此每个字符串都能在 UTF-8 和 SQLite 中存活，且被索引的字符串（任务种类、请求 ID、文档种类和键，像上游那样存储为 JSON 编码）跨实现比较相等。浮点数使用 Python 的最短 repr（`0.0`、`1e-07`），它解析为相同的数字；键顺序是记录的。`json_parse` 是 `JSON.parse`（无 `NaN`）。Entry 经过 `encode_entry`/`decode_entry`；其他记录原样存储。
- **字节兼容的 schema。** 迁移语句是上游的文本，含空白（SQLite 把它保存在 `sqlite_schema` 中）；`tests/test_storage_interop.py` 检查 TypeScript 写出的数据库和 JSONL 目录（`tests/fixtures/storage_interop`）在这里读取一致，而且 Python 写出的读取方式与它们相同。
- `SqliteStorage.open(db, *, prefix=None)` 和 `JsonlStorage.open(directory, fs, context, *, fsync=False)`：尾部选项对象仅关键字；`SqliteStorageOptions`/`JsonlStorageOptions` 是它们的 `TypedDict`（可作 `**options` 使用）。在事务内 `SqliteStorage` 把独立的语句一起发出并作为一个批次 await（按句柄上的调用顺序），因此线程适配器每个批次只付一跳；失败在批次落定后、COMMIT 之前抛出。
- `SqliteStorage.close` 是 `async def`；重复调用 await 同一个 close，它等待已收录的多查询读取（上游返回同一个 Promise）。
- `LocalExecutionEnv(*, cwd, shell_path=None, shell_env=None, watch=None)`：构造函数接受关键字参数（上游的必需选项对象）；`watch` 是 `LocalWatchOptions` `TypedDict`（`mode`、`pollIntervalMs`、`maxDirectories`）。它的 `id` 是 `"local"`（上游 `"node:local"`）。阻塞的文件调用在循环的默认 executor 上运行；等待它的任务被取消时已运行的调用继续运行，读者只在其完成后关闭描述符。命令通过 `loop.subprocess_exec` 在新会话（进程组）中运行，在超时、中止、等待任务取消或 `cleanup()` 时作为树被杀。`FileError` 消息是 `str(OSError)`。
- `LocalExecutionEnv.list_files` 返回 `not_supported`（上游的 Node 环境没有 `listFiles`）。
- 原生监视是 Linux 上的 inotify（经 `ctypes`，每个事件循环一个实例，监视描述符共享）；其他平台和不可靠的文件系统轮询。扫描在 executor 线程上运行。
- 一致性用例是 `StorageConformanceCase(name, run, timeout_ms=None)`；`create_assertions()` 抛 `AssertionError`（上游适配一个 Vitest `expect`）。仅 JavaScript 的原型用例保留为“键 `__proto__`/`constructor` 是普通分离数据”。
- 基准：`StorageReadBenchmark(name, run, expected)` 和 `StorageWriteBenchmark(name, expected, run)` dataclass；`benchmarks/bench_durable_storage.py` 对 memory、SQLite 和 JSONL 运行它们。
- 包根不导入 SQLite/JSONL 后端、本地环境或 `testing`，如上游的 `index.ts`（`tests/test_storage_runtime_boundary.py`）。

## 10. Harness（阶段 3）

`pi_durable/harness/` 逐文件移植 `src/harness/`（`task-graph.ts` 是 `task_graph.py`）：`harness, scheduler, generation, tool, compaction, context, agent, events, inbox, live, prompt, provider, registry, submissions, task_graph, usage, util, view`，外加 `stream.py`（无上游模块：它把存储的 camelCase `ConversationStreamOptions` 转成 pi-ai 的 snake_case 关键字选项，并把解析后的 `ConversationRetryPolicy` 转成 `pi_ai.utils.retry.RetryPolicy`）。`pi_durable/__init__.py` 现在完全对应 `index.ts`。

API 形状：

- `Harness` 是具体类（`class Harness(SessionImpl)`），用 `await Harness.open(storage, options, context)` 打开；`options` 是带上游键的 `HarnessOptions` dict（`models`、`registry`、`settings`、`env`、`now`、`setTimer`、`onWake`、`onReport`、`conversationCreated`）。`settings` 可以是 dict 或返回 dict 的函数；每次使用都读取（上游的 getters）。`Harness` 协议留在 `pi_durable.harness.types`；类满足它（`tests/typing_cases.py` 检查）。
- **急切方法。** 与 Session 的（第 5 节）一样，上游函数体在第一个 `await` 前把工作收录进 Session 行的 Harness 和会话方法是返回已启动 `asyncio.Future` 的普通 `def`：`Conversation.configure/submit/reset/compact/commit/context/entries/fork/abort/wait_for_idle/agent/view_state/watch`（视图附加在调用时收录，因此后面的 `close()` 无法超越它）、`Harness.root/conversation/create_conversation/get_task/inspect/submission/abort_submission/abort_task/retry_task/wait_for_task/wait_for_idle/task_graph/watch_task_graph`、`Submission.status/wait/abort`、`ToolExecutionApi.details`（它的检查和更新在调用时运行）和 `ToolExecutionApi.execute_tool`（它的检查、key 和收录在调用时开始，因此紧接着返回的调用者仍然让该调用被收录并由清理中止）。`harness/types.py` 中的协议把这些都是声明为 `def ... -> Awaitable[T]`。`harness/util.py` 中的辅助：`eager(awaitable, then)` 把已开始的操作保持为 future；`chain(future, then, context=None)` 通过完成回调实现 `promise.then(then)`（无额外任务跳；被取消的 `context` 只拒绝等待）。
- 调度器的 `queueMicrotask` 是 `loop.call_soon`；它的 drain 和 reconcile 任务直接生成（一跳循环，像微任务）。
- Hook 表是以 snake_case 名为键的 dict（第 2 节）：`GenerationTask` 的 `before_request`、`after_response`、`on_yield`、`after_tools`；`ToolTask` 的 `before_tool`、`after_tool`；`CompactionTask` 的 `before_compact`。处理器接受上游的参数，`context` 最后，可以同步或异步。内部 `agent_hooks()` 返回 `(extension_name, handlers)` 元组。
- 消息作为 **JSON** 存在文档和任务检查点中（`pi_ai.serialization.to_json_value`，用 `message_from_dict` 读回）：`pi.live` 的 `generation.message`、deferred 句柄、嵌套 `ToolCall`、`pi.usage`（`Usage` 作为 camelCase JSON：`usage_json(usage)`）、收件箱写入项（`entry` 持有带序列化 `model` 消息的草稿）。Entry 保留 pi-ai dataclass（第 8 节）。
- 运行时提交回调是 `change(tx, current)`；`Conversation.commit`/`Session.commit` 接受 `change(tx)`。`subscribe_commits` 监听器是 `listener(publication, context)`。
- `watch_events(harness, conversation_id, context) -> AgentEventStream`；`stream.start(listener)` 调用 `listener(events, context)`（同步或异步）；`stop()` 返回一个 future。
- `Conversation.abort(context, *, background=False)` 替换上游的 `ConversationAbortOptions`。
- 工具 `execute(args, api, context)` 收到对照 `parameters`（一个 JSON Schema dict，或 Pydantic 模型，它随后收到模型实例）校验过的参数。`api.output(chunk, skipped=None)` 接受 `OutputSkip` 或 `ShellOutputSkip` dict。
- 错误文本是上游的，除非 Python 的性质本来就不同（非 JSON 值报告 Python 的 `json` 消息，"... is not JSON serializable"）。

测试辅助（`tests/durable_harness_support.py`，`harness-support.ts`、`task-support.ts`、`chat-support.ts` 的移植；用 `from durable_harness_support import ...` 导入）：

- Registries：`tool(name)`（空对象 schema，无输出）、`add_tool/add_task/add_hooks/add_section`（每个安装一个单项扩展并返回带 `dispose()` 的 `Installed`）、`counting_reader`。
- `open_harness(storage, tool_names, *, registry=None, on_report=None) -> OpenedHarness(harness, registry)`。
- `open_tasks(storage, tasks, *, registry=None, now=None, settings=None, on_wake=None, set_timer=None) -> OpenedTasks(harness, registry, reports, infos)`；`completed(result)`、`aborted_with(reason)`。
- Chat：`chat_setup(**faux_options) -> ChatSetup(faux, models, registry, reports, settings, now)`（faux 选项是 snake_case，`tokens_per_second=200, token_size={"min": 1, "max": 1}` 用于慢速流式）；`open_chat(storage, setup, *, env=None) -> OpenedChat(harness, root)` 在相同 models、registry 和 **live** `setup.settings` dict 上打开（在决策之间赋键，如 `setup.settings["followUpMode"] = "all"`），并用 `faux/faux-1` 配置根会话。用 `setup.faux.set_responses([...])` 编排模型：消息（`faux_assistant_message`、`faux_text`、`faux_tool_call`）或步骤函数 `step(request, options, state, model)`（`request.messages` 持有请求的带 `sections`、`tools_added`、`tools_removed` 的 `SystemMessage`；`options["signal"]` 是中止信号）。`setup.faux.state.call_count` 统计请求。
- 假时钟：`setup.now = lambda: 7`（通过 lambda 读取，因此打开后可替换）；`open_tasks(..., now=..., set_timer=...)` 用于调度器定时器（记录回调的 `set_timer(callback, delay_ms, options)` 让测试可以手动触发它们）。
- 等待：`flush()`（25 次循环轮转）、`eventually(check)`（用 `flush` 轮询）、`wait_for(check, timeout_ms=5000)`（真实时间，用于节流窗口和定时器）、`settled(awaitable)`、`aborted(signal)`（中止时抛出 reason）、`unanswered()`（一个阻塞到取消的 faux 步骤；`reached` 在请求发出后落定）。`durable_session_support.deferred()`/`resolve()` 是门。
- `all_entries(conversation)`、`text_of(message)`、`describe_message`、`tools_named(setup, *names)`。
- `match_object(actual, expected)` / `assert_match(actual, expected)`：Vitest 的 `toMatchObject`（递归 dict 键；dataclass 匹配其属性名 dict）；`gated(message)`：一个保持到 `release()` 或取消的 faux 步骤，带 `reached`。
- `race_aborted(awaitable, signal)`：`Promise.race([awaitable, aborted(signal)])`（绝不取消共享门）；`StorageSpy(inner)`：委托每个存储方法，子类化它以观察或保持某些方法（上游重新绑定存储实例的方法；Python 后端是槽位类）。
- SQLite 存储在 executor 线程上运行，因此对它的等待以真实时间轮询（`wait_for`，而非 `eventually`，其固定轮转数对上游同步的 `node:sqlite` 足够）。
- 确定性驱动：Harness 在 `harness.resume()`（或 `submit`、`compact`、`abort`、`wait_for_task` 或 `wait_for_idle`，它们会 resume）之前不调度任何东西；`ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])` 给出 `hold_commits()`（返回带 `entered`/`release()` 以占据 Session 行的门）、`fail_next_commit(error)`、`crash()` 和 `reopen()`。优先用门（`deferred()`）而非睡眠，放在工具 `execute` 函数、faux 步骤和任务阶段内。
- 竞态：Python 提交比 JavaScript 在更少的循环轮转中落定（对已完成 awaitable 的 `await` 不会让出），因此依赖 JS 微任务交错的测试可能需要显式门。`test_harness_tasks.py` 的双中止用例出于该原因把 abort 处理器保持在一个门上。

## 11. 工具、工具搜索与图像（阶段 2）

| 上游 | Python |
|---|---|
| `tools/index.ts` | `pi_durable/tools/__init__.py`：`create_coding_tools(*, images=None)`、`CodingTools`、`create_*_tool` 工厂及其类型 |
| `tools/{read,write,edit,bash,image,env}.ts` | `pi_durable/tools/{read,write,edit,bash,image,env}.py` |
| `tools/{edit-diff,image-processor,path-utils,file-mutation-queue}.ts` | `pi_durable/tools/{edit_diff,image_processor,path_utils,file_mutation_queue}.py` |
| `diff`（jsdiff 8.0.4，npm） | `pi_durable/tools/_jsdiff.py`（私有；其头部有 BSD-3-Clause 声明） |
| `tool-search/{index,tool-search}.ts` | `pi_durable/tool_search/{__init__,tool_search}.py` |
| `images/index.ts`、`images/node.ts`（+ `node-worker.ts`） | `pi_durable/images/__init__.py`（Pillow）：`prepare_image`、`PillowImages`、`create_pillow_images` |
| `images/exif.ts` | `pi_durable/images/exif.py` |

未移植：`images/cloudflare.ts`、`images/photon-wasm.d.ts`（Photon 专属）。

API 形状：

- 工厂把上游的选项对象作为关键字参数：`create_read_tool(*, images=None)`、`create_bash_tool(*, command_prefix=None, prepare=None)`、`create_powershell_tool(*, command_prefix=None, prepare=None, programs=None)`。`BashExecution` 是带上游键的 `TypedDict`（`command`、`cwd`、`env`、`inheritEnv`），`prepare(execution, api, context)` 在原处修改它（同步或异步）。
- 数据形状保留上游的键：`ImageLimits`（`maxWidth`、`maxHeight`、`maxBytes`）、`PreparedImage`（`data`、`mimeType`、`resized: {"from", "to"}`、`convertedFrom`）、`EditToolDetails`（`diff`、`patch`、`firstChangedLine`）、`ReadToolDetails`（`truncation`）、bash 的 `structuredOutput`（`output`、`truncated`、`fullOutputPath`、`exitCode`）、`ToolSearchToolDetails`（`loaded`）。`ImageProcessor` 是带 `async def prepare(data, mime_type, limits)` 的 `Protocol`。`ByteSource`（图像头嗅探）是冻结 dataclass `(size, read)`。
- 工具名、描述和参数 schema 等于上游的（在 Node 下对照上游的 `createCodingTools()` 和 `createPowerShellTool()` 比较），因此 transcript 和提示缓存看到相同的工具。

规则：

- **精确文本。** 模型看到的是上游的文本：`edit` 的 diff 和补丁来自 jsdiff 移植（Myers `diffLines`、`structuredPatch`、`formatPatch`、`createTwoFilesPatch`），绝不用 `difflib`。`tests/test_tools_edit_diff.py` 重放一份在上游 `edit-diff.ts`（jsdiff 8.0.4，Node 下）跑过的语料（`tests/fixtures/edit_diff/upstream.json`）。有差异的 JavaScript 字符串函数被复现：`trimEnd` 的空白集、按 UTF-16 码元计数的 `split("")`、无 `NaN` 的 `JSON.parse`。
- **`read` 中的 JavaScript 数字。** `offset` 和 `limit` 按 `slice`、`Math.trunc`、NaN 传播的 `Math.min` 和 `Number.isSafeInteger` 语义作为浮点数使用（`tools/_js.py`），并按 `String(number)` 打印（`2.5`、`61`、`100000000000000000000`），因此每个参数的选择和报告与上游一致（`tests/test_tools_read_differential.py`，其参考读取使用与上游相同的种子）。
- **变更队列。** 按事件循环、`env.id` 和规范路径为键；等待轮次的调用通过 `asyncio.shield` 等待，因此取消它不会取消排在它前面的调用，它的槽位在 `finally` 中释放。
- **图像。** `pi_durable.images` 在模块加载时导入 Pillow，没有它则抛点名 `pi-ai-python[images]` 的 `ImportError`；`pi_durable` 和 `pi_durable.tools` 永不导入它（`tests/test_storage_runtime_boundary.py`）。`prepare_image` 遵循 Photon 的规则：解码后正立、内联、在限制内的图像直通；来自 `exif.py` 的 EXIF 方向（Pillow 按方向自带的转置）；用 `Math.round` 缩放适应；PNG 然后 JPEG 80/70/55/40（JPEG 源先用 JPEG），然后按 3/4 步降到 1x1；BMP 和其他非内联格式变成 PNG；解码为 RGB，或带透明度的 RGBA，JPEG 丢弃 alpha；LANCZOS 重采样。输出字节与 Photon 的不同。在只有字节限制决定图像缩到多小的地方，Pillow 的 JPEG（同质量更小）可能比 Photon 的停在更大的尺寸。
- **离开循环。** `PillowImages` 在它自己的一个工作线程上运行 `prepare_image`（在第一张图像时启动，`close()` 停止它），如上游的单一工作线程；`worker=False` 在调用线程上运行；`executor=` 接受任何 `concurrent.futures.Executor`。是线程不是进程：Pillow 在解码、转换、缩放和编码时释放 GIL，因此循环保持运行（测试测量最长的循环停顿），而且没有东西被 pickle；`ProcessPoolExecutor` 也可以，因为 `prepare_image` 及其参数可 pickle。没有空闲超时：空闲线程不持有图像内存，不像 Photon 的 WebAssembly 内存。
- `tool_search` 用 `pi_ai.utils.schema.parameters_schema(tool.parameters)` 构建工具的搜索文本，因此 Pydantic 参数像 JSON Schema 参数一样被搜索。

## 12. 代码模式（阶段 3）

| 上游 | Python |
|---|---|
| `codemode/index.ts` | `pi_durable/codemode/__init__.py`：`create_codemode_extension`、`create_codemode_tool`、`CodemodeStoreDoc`、`CodemodeNamespace`、`CodemodeToolDetails`、`CodemodeNestedCall`、`CodemodeToolInput`、`CodemodeGlobal`、`CODEMODE_MODULE_VERSION`、`CODEMODE_TOOL_NAME`、`DEFAULT_CODEMODE_INLINE_BUDGET` |
| `codemode/codemode.ts` | `pi_durable/codemode/codemode.py`（还有 `CodemodeRunDoc`、`CodemodeJournalDoc`；调用是 `_CodemodeCall`） |
| `codemode/describe.ts`、`models.ts`、`output.ts` | `pi_durable/codemode/{describe,models,output}.py` |

脚本是 Python，由 `pi_codemode` 沙箱（Monty）通过 `CodemodeSandboxBase.start()`/`CodemodeRun.deliver()` 运行；其他一切都是上游的。

API 形状：

- `create_codemode_extension(*, sandbox, namespaces=(), inline_budget=None, models=True, globals=())` 和 `create_codemode_tool(...)` 把上游的 `CodemodeToolOptions` 作为关键字参数。`CodemodeNamespace` 是 `TypedDict`（`name`、`prefix`、可选 `description`、`instructions`）；`CodemodeGlobal(name, execute, spread=False)` 是带 `execute(args, api, context)` 的冻结 dataclass（内部，如上游的 `globals`）。
- 模型读到的内容为 Python 重写：工具描述（`await tools.<name>(key=value, ...)`、`asyncio.gather`、`print`、`# @options:`）、`codemode` 段（pi_codemode 的 Python 声明、`### \`id\`` 标题和预算规则，如上游）、参数描述（`Raw Python source.`）、`models.*` 描述以及显示脚本语法的错误文本（`context["state"] must be a dict`、`models.get_model_of_type(...)`）。
- 脚本全局变量是 snake_case：`search_tools`、`describe_tool`、`describe_namespace`、`models.get_models_of_type`、`models.get_available_of_type`、`models.get_model_of_type`、`models.classify`、`models.generate_images`。工具名保持上游的，包括内部 `pi.codemode.models.classify` 和 `pi.codemode.models.generateImages`（details 行像上游一样显示 `models.generateImages`）。spread 全局变量把位置参数作为列表、关键字参数作为尾部 dict 收到；`models.bind_arguments(args, names)` 把其键全部指名未填充参数的尾部 dict 绑定为关键字，因此 `f(a, b)`、`f(a, b=b)` 和 `f(a=a, b=b)` 一致。`search_tools(query, limit=..., namespace=...)` 以关键字或上游的一个选项 dict 接受其选项。
- 模型调用：上下文像上游那样检查，并用 `pi_ai.serialization.from_dict` 解码为 `ClassifierContext`/`ImagesContext`；目录项和结果是 `to_json_value`（camelCase），`headers` 被丢弃。
- `Start.tools[].jsName` 映射到 `CodemodeDeclaredTool.js_name`；隐藏（内部）工具被排除在 run 的工具之外。

规则：

- **持久化 JSON 是上游的**：`pi.codemode.run`（带 `version`、`startedAt`、`seed`、`fingerprint`、`deadline`、`tools[{name, jsName, description, structured?, hidden?}]`、`namespaces`、`globals`、`maxOutputTokens`、`store` 的 `start`；`outputDir`；`result`）、`pi.codemode.journal`（`entries[{id, at, issuedAfter?, global?: {name, args?, outcome}}]`、`firstIssued`）、`pi.codemode.store`（`values`，可回退，`fork="asOf"`）。存储的 `result` 是带其 `output` 作为 pi-ai JSON 的工具结果。`CODEMODE_MODULE_VERSION` 保持 2。Run 不在语言间移动：指纹（`pydantic-monty ...` 对比 QuickJS 的）不同，因此恢复的外语 run 以“the sandbox changed”结束。
- **重放**逐行遵循上游：指纹门、重放结果的按日志顺序投递、每结果的确认计数、在启动前对照更早尝试的 `NestedCallDoc` 键（`c<VM call id>`）检查保持的调用、只在调用的 issued 计数提交后启动、在 run 文档中一次性决定发散与不匹配、对结束于实时 run 继续处的重放检查 `during`。上游的 `void promise` 工作作为 `_spawn` 保持的任务运行（意外错误使调用失败）；门上的 `then` 是完成回调。调用结束时门打开、模型槽位等待者唤醒，因此没有东西永远等待；两者都检查 `stopped` 且不收录任何东西。
- 两个刻意的差异，结果中都不可见：释放的模型槽位直接传给第一个等待者（上游释放它并让等待者在一个微任务后取走，新调用可能抢先），以及 journal 提交后下一个提交只为排队的（结果）或已赶上的 VM 的计数开始；VM 仍在处理的（结果）的计数随下一个提交走（上游的声明规则；其代码立即提交，而 Monty 0.3 ms 的往返会让每几个结果花一次提交）。
- 脚本值：`None` 是上游的 `undefined`（全局变量的 `None` 不带 `value` 记入 journal；返回 `None` 的脚本不给输出加值）。失败调用的结果保留上游的 `properties.structuredOutput`，Monty 脚本看不到它（异常只持消息）。不是由调用造成的 "aborted" 结果（`Sandbox closed`）不存储任何东西，并在调用的 context 中止时抛出，如上游。
- 输出（`output.py`）：上游的布局、预算、溢出和图像文件；长度按码点计数，`value_text` 是紧凑 `json.dumps`（`separators=(",", ":")`，非 ASCII 原样），墙钟时间 `f"{ms / 1000:.1f}"`。
- `pi_durable.codemode` 只导入 pi_codemode 的可移植模块（`source`、`identifier`、`declarations`、`types`，以及宿主运行时的类型）；沙箱来自调用者。`pi_durable` 不导入 `pi_durable.codemode`，两者都不加载 `pi_codemode.sandbox` 或 `pydantic_monty`（`tests/test_storage_runtime_boundary.py`）。
- 测试（`tests/test_codemode*.py`）需要 Monty，没有则跳过，描述/段测试除外；每个重启用例都在 MemoryStorage 和 SQLite 上运行。
