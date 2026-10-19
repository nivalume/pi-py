# pi_durable design notes

The contract for porting `@earendil-works/pi-durable` 1.3.0 to Python. Stage 1 (this foundation) fixes
the rules below; later stages follow them and extend this file when they add a rule.

## 1. Module layout

Upstream `src/x/y-z.ts` becomes `pi_durable/x/y_z.py`. Stage 1 ported:

| Upstream | Python |
|---|---|
| `types.ts` | `pi_durable/types.py`: records, tokens, `Tx`, `Session`, `TaskRuntime`, `Storage` |
| `ids.ts`, `errors.ts`, `timers.ts`, `tasks.ts`, `entries.ts`, `documents.ts`, `search.ts`, `truncate.ts` | same names |
| `harness/types.ts`, `harness/define.ts`, `harness/json.ts` | `pi_durable/harness/{types,define,json}.py` |
| `env/index.ts` | `pi_durable/env/types.py`, re-exported by `pi_durable/env/__init__.py` |
| `env/decode.ts`, `env/line-scan.ts` | `pi_durable/env/{decode,line_scan}.py` |
| `storage/memory.ts`, `storage/scan.ts` | `pi_durable/storage/{memory,scan}.py` |
| (new) | `pi_durable/storage/codec.py`: JSON encoding of records for serializing backends |

`pi_durable/__init__.py` mirrors upstream `index.ts` for what exists; later stages add their exports.
Tests are `packages/durable/tests/test_<upstream-name>.py`.

## 2. How TypeScript types map to Python

**One rule per kind of type.**

| TypeScript | Python | Example |
|---|---|---|
| Object type of data (records, writes, queries, pages, outcomes, settings, option objects, `Result`) | `TypedDict` whose keys are **exactly upstream's property names**, camelCase included | `{"id": 2, "conversationId": 1, "kind": "pi.user"}` |
| `readonly` property | `ReadOnly[...]` (from `typing_extensions`, imported under `TYPE_CHECKING`) | `id: ReadOnly[EntryId]` |
| `x?: T` | `NotRequired[T]`: absent key = JS `undefined`; `None` = JSON `null` | `head: ReadOnly[NotRequired[EntryId]]` |
| `T \| undefined` return | `T \| None` | `async def task(...) -> AnyTaskRecord \| None` |
| Discriminated union | union of `TypedDict`s sharing a `Literal` key; anonymous members are named `<Union><Variant>` | `TaskStateRunning`, `SubmissionRecordInputDone`, `StorageWriteEntry` |
| Interface with methods (`Storage`, `Session`, `Tx`, `TaskRuntime`, `WatchHandle`, `FileSystem`, `Conversation`, `Harness`) | `Protocol` with snake_case methods and properties | `storage.scan_entries(query, limit, cursor, context)` |
| What an application defines (doc/entry/task tokens, `Extension`, `ToolRegistration`, `PromptSection`, `Wrap`, `HookRegistration`) | slotted classes / frozen dataclasses with snake_case attributes, built by `define_*` helpers taking upstream's fields as keyword arguments | `define_doc(kind=..., checkpoint_when=...)` |
| Hook tables (`GenerationHooks`, `ToolHooks`, a task's `H`) | `TypedDict(total=False)` of handlers keyed by **snake_case** hook name (method tables, not data) | `hook(ToolTask, {"before_tool": guard})` |
| `Id<Kind>` brands | `NewType` over `int`: `ConversationId`, `EntryId`, `SubmissionId`, `DocumentId`, `Seq`; `TaskId[R]` is a generic `int` subclass for type checkers and plain `int` at runtime | `EntryId(5)`, `TaskId(4)`, `id_from_number(4)` (returns `Any`) |
| `JsonValue` payloads in records (entry `data`, task input/checkpoint/result, memos, `detail`) | `Any` (must be strict JSON at runtime); `JsonValue`/`JsonObject` from `pi_chord.json` where upstream means storage-internal JSON (document values, cursors) | |
| `Message` inside records (`EntryRecord["model"]`, `ContextEdit["messages"]`, `ContextView`) | `pi_ai` message dataclasses | `"model": [UserMessage("hi")]` |
| `Op` | `pi_chord.delta.Op` tuples (`("s", ("count",), 2)`) | |
| `Promise<T>` method | `async def`; **eager** ones (§5) are `def ... -> Awaitable[T]` returning a started future | `storage.commit(...)`; `session.commit(...)` |
| `Promise<T>` property (`closed`) | property returning an awaitable (`asyncio.Future`, awaitable many times) | `await session.closed` |
| `T \| Promise<T>` callback results | `T \| Awaitable[T]`; implementations `await` when `inspect.isawaitable(result)` | `commit(change, context)` |
| `JsonRepresentation<T>` | no computed type: the value types are the `TypedDict`s themselves, and values must be strict JSON (`pi_chord.json.copy_json` accepts them) | |

Generic records: `TaskRecord[I, S, R]`, `RunningTask[I, S, R]`, `TaskState[S, R]`, `TaskOutcome[R]`,
`SettledTask[R]`, `TypedEntry[D]`, `Page[T]` (upstream `Page<T, Cursor>`; the cursor type is always
`Cursor`). `AnyTaskRecord` is `TaskRecord[Any, Any, Any]` (upstream `TaskRecord<JsonValue, ...>`).

Working with records:

```python
running: TaskStateRunning[Checkpoint] = {"status": "running", "checkpoint": {"phase": "run"}}
updated: AnyTaskRecord = {**task, "state": running}          # upstream {...task, state}
if write["type"] == "entry":                                 # narrow on the subscript itself:
    write["value"]["conversationId"]                         # `kind = write["type"]; if kind == ...` does not narrow
if result["ok"] is False:                                    # env Result: identity on the literal narrows,
    return result                                            # truthiness (`if not result["ok"]`) does not
```

Invariants upstream encodes in types but Python does not: `memos` only on live task records; `latest`
history never pairs with fork `asOf` (`define_doc` checks it at runtime); a phase map is exhaustive;
kinds without data (`UserEntry`) still type `TypedEntry["data"]` (do not read it).

## 3. Persisted JSON compatibility

A record **is** its persisted JSON: field names and values are upstream's (`conversationId`,
`abortRequested`, `createdAt`, `deltasSinceBase`, `"document.create"`, `"asOf"`, ...). Never rename a key
on the wire. Exceptions, all handled by `pi_durable.storage.codec`:

- **Messages** in `EntryRecord["model"]` and `ContextEdit["messages"]` are encoded with
  `pi_ai.serialization` (camelCase, `role` discriminator), exactly what TypeScript pi writes.
- **Ops** are tuples in memory and JSON arrays on disk (`decode_ops` restores tuples with tuple paths).
- Backends persist `encode_storage_write(write)` / `encode_entry(entry)` and return
  `decode_entry(row)`; everything else in a record is already strict JSON.

Inside JSON payloads (task checkpoints/results, document values, entry `data`, tool `details`),
anything that is a dataclass in Python must be stored as its JSON: messages and `Usage` via
`pi_ai.serialization.to_json_value`, tool schemas via `pi_ai.utils.schema.parameters_schema`
(`pi_ai.Type` returns a `dict` subclass, which `copy_json` rejects). A `ToolRegistration` never enters
the transcript; write `registration.to_tool()`.

Python-facing names that differ from upstream: snake_case functions, methods, attributes, and keyword
arguments (`mint_id`, `find_latest_head_marker`, `checkpoint_when`, `execution_mode`); dict keys never.
Times are epoch milliseconds (`float` in annotations; write `int`s like `Date.now()` does).

## 4. Declaring entries, documents, tasks, tools and extensions

```python
from typing import Any, TypedDict
from pi_ai import Type
from pi_durable import (ConversationDocToken, Entry, Task, define_doc, define_entry, define_extension,
                        define_task, define_tool, hook, section)

class NoteData(TypedDict):
    text: str

Notes: Entry[NoteData] = define_entry("app.note")              # annotate to type `data`

class Todos(TypedDict):
    items: list[str]

TodosDoc: ConversationDocToken[Todos] = define_doc(
    kind="app.todos", version=1, scope="conversation",
    history="latest",                                         # or "rewindable" for snapshot_as_of()
    fork="initial",                                           # "initial", "current", or "asOf"
    initial=lambda: {"items": []},
)
# await root.commit(lambda tx: ..., context); inside: (await tx.doc(TodosDoc, root.id))["items"].append("x")

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

Reminder: Task[ReminderInput, Any, None, Any] = define_task(  # annotate: R appears in no argument
    name="app.reminder", version=1, initial=lambda input: {"phase": "wait"},
    phases={"wait": wait, "deliver": deliver}, abort=abort,
)

async def count_execute(args, api, context):
    for i in range(1, args["n"] + 1):
        api.output(f"{i}\n")
    return {}                                                 # ToolExecutionResult: upstream keys

count = define_tool(name="count", description="Count from 1 to n",
                    parameters=Type.Object({"n": Type.Number()}), execute=count_execute)
Coding = define_extension(
    "coding", tools=[count],
    sections=[section("preamble", lambda input, context: "You are a concise coding assistant.", tag=False)],
    hooks=[hook(ToolTask, {"before_tool": lambda call, api, context: {"block": "Needs approval"} if dangerous(call) else None})],
)
```

Schemas are JSON Schema dicts built with `pi_ai.Type` (or hand-written, or a Pydantic model where
pi-ai accepts one for tool parameters); `Static<T>` has no Python counterpart, so `args` is `Any`.
`wrap_tool(tool, lambda t: dataclasses.replace(t, execute=...))` is upstream's `{...tool, execute}`.

## 5. Context, cancellation, and async boundaries

- Every operation that takes `context` upstream takes a `pi_chord.context.Context` in the **same
  position** (usually last positional). Check it with `context.throw_if_aborted()`; wait with
  `await_with_context(awaitable, context)`; derive with `with_abort_signal`/`without_abort_signal`
  (mandatory cleanup); `BACKGROUND_CONTEXT` at the roots. A run's signal (`TaskRuntime.signal`) is a
  `pi_ai.abort.AbortSignal`.
- Upstream `async`/Promise-returning functions are `async def` on asyncio; synchronous ones stay
  synchronous (`Tx.settle_submission`, `Tx.place_submission`, `subscribe_commits`, `WatchHandle.start`,
  `TaskRuntime.now/set_timer/report`, `Registry.install`). Commit/close listeners are synchronous
  and must not block. A TS Promise kept unawaited for ordering (e.g. `closed`) becomes an
  `asyncio.Future` created on the running loop. Never block the loop.
- **Eager operations** (stage 2a). A JavaScript `async` function runs synchronously until its first
  `await`, and a Promise-returning method does its work before it returns; a Python coroutine does
  nothing until awaited. Where upstream semantics depend on that synchronous part (admission to the
  Session line, sealing on close, the read-after-write rule, staging order inside a transaction), the
  protocol declares `def ... -> Awaitable[T]` and implementations run the synchronous part at the call
  and return an already started `asyncio` future (a sync-part error becomes a failed future, as a
  rejected Promise). This covers every `Tx` method that upstream returns as a Promise, `Session.commit`,
  `close`, `document_state`, `DocumentReader.snapshot`/`snapshot_as_of`, `DocumentObserver.watch_doc`,
  and `WatchHandle.stop`. Callers simply `await`; to start one without awaiting, keep the returned future
  (use `asyncio.ensure_future`, not `create_task`, to wrap it). An implementation may still be
  `async def` (a coroutine is an `Awaitable`) when nothing depends on eagerness, as `TaskRuntime` and the
  harness APIs that extend `DocumentReader` may. `Storage` stays `async def`.
- Callbacks typed `T | Promise<T>` accept sync or async functions; the caller awaits awaitables.
- Timers (`SetTimer`, `host_timers`, `native_timer`) run on the loop (`loop.call_later`); a timer
  callback is synchronous.
- Optional trailing option objects become keyword-only parameters (`section(key, render, *, tag=None)`,
  `truncate_head(content, *, max_lines=None)`, `TaskRuntime.context(cid, context, *, at=None)`,
  `Conversation.abort(context, *, background=False)`, `execute_tool(name, args, context, *, key=None,
  progress=None)`, `Harness.root(context, *, agent=None, init=None)`). Required or mid-position option
  objects stay `TypedDict`s (`TaskOptions`, `ConversationCreateOptions`, `{"ownership": ...}` for
  `Tx.create_conversation`, `ReadTextLinesOptions | None` in `FileSystem.read_text_lines`).
- Overloads keep upstream's argument lists and are positional-only in the protocols, so an
  implementation can dispatch like upstream's `resolve_address(definition, args)`:
  `async def doc(self, token, *args)`; `memo(name, context)` vs `memo(name, candidate, context)` is
  told apart by `isinstance(last, Context)`.
- Errors keep upstream's classes and messages (`pi_durable.errors`, `FileError`, `ExecutionError`).
  A bare JS `Error` is `RuntimeError`, `TypeError` is `TypeError`, `RangeError` is `ValueError`;
  `throw new Error(msg, { cause })` is `raise RuntimeError(msg) from cause`. `SessionFailed.cause` is
  also `__cause__`.

## 6. The Storage protocol (`pi_durable.types.Storage`)

```python
class Storage(Protocol):
    async def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq
    async def mint_id(self) -> int                                  # brand it: EntryId(await storage.mint_id())
    async def conversation(self, id, context) -> ConversationRecord | None
    async def scan_conversations(self, query, limit, cursor, context) -> Page[ConversationRecord]
    async def entry(self, id, context) -> StoredEntry | None        # {"entry": ..., "commitSeq": ...}
    async def visible_entry(self, conversation_id, id, context) -> StoredEntry | None   # upstream entry(cid, id)
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

- A commit is one list of `StorageWrite` dicts (`{"type": "entry", "value": record}`,
  `{"type": "document.change", "id": id, "content": {"kind": "delta", "version": 1, "ops": ops}}`, ...)
  applied atomically; it returns the commit's `Seq`. `DocumentPoint` is a `Seq` or `"current"`.
- The only rename: upstream's overloaded `entry(conversationId, id, context)` is `visible_entry`.
- Cursors of the built-in backends are `{"after": id, "order": order}` (`storage.scan.next_cursor`,
  `scan_start` validates them and raises `StorageRequestError`).
- Detachment: a backend copies what it retains and what it returns (`storage.codec.clone_value` in
  memory, serialization elsewhere). `MemoryStorage.prepare_commit(writes, seq=None)` returns a
  `PreparedMemoryCommit` with `.seq`, `.writes` (detached and retained: never mutate) and `.apply()`,
  for the JSONL backend, as upstream.
- Errors: `StorageRequestError` for invalid reads (no durable effect); anything else is fatal to the
  Session. Exhausted IDs raise `RuntimeError("ID space is exhausted")` past `ids.MAX_SAFE_INTEGER`.
- `MemoryStorage` satisfies the protocol statically (`tests/typing_cases.py`) and is the reference.

## 7. Notes for later stages

- (Done in stage 3.) `harness/types.py` imports `ConversationView`, `TaskGraph`, `TaskGraphWatch`,
  `UsageState` from `harness/view.py`, `task_graph.py`, `usage.py` under `TYPE_CHECKING` and exports them
  lazily through a module `__getattr__` (those modules import `harness/types.py`).
- `GenerationContinue` is `{"continue": input}` (functional `TypedDict`, `continue` is a keyword).
- `ToolHookCall` subclasses `pi_ai.ToolCall` with a keyword-only `parent`.
- `ToolRegistration` subclasses `pi_ai.Tool`; it is hashed and compared by identity.
- `FileSystem.list_files` is a required method (upstream optional): an environment that cannot list
  returns `err(FileError("not_supported", ...))`.
- `get_or_undefined` is `get_or_none`; binary data is `bytes`.
- Static checks: add `# pyright: ignore` lines (expected errors) and `assert_type` calls to
  `tests/typing_cases.py`; `tests/test_types.py` runs pyright on it.

## 8. The Session (stage 2a)

`pi_durable/session/` ports `src/session/`: `create_session(storage, *, now=None,
document_retention_ms=None, set_timer=None) -> Session`, the concrete `SessionImpl` and `Transaction`,
and the observers `CommittedStateSource` (a `pi_chord.state.ReplicatedStateSource`) and `CommittedWatch`
(a `WatchHandle`). Methods documented *internal* (upstream "Internal:") serve the Harness and tests; those
documented *protected* (upstream `protected`) are meant to be overridden by a subclass, as the Harness
subclasses `SessionImpl`; both keep upstream's names, snake_cased, without a leading underscore:

- `SessionImpl(storage, now=None, document_retention_ms=None, set_timer=None)`: positional like upstream;
  `document_retention_ms` is a callable read at each use.
- Internal: `commit_with(change, context, scope=None)` (scope is the `TransactionScope` TypedDict
  `{"conversationId", "taskId"}`), `read_on_line(job)`, `conversation_document_on_line(token, cid, context)`
  (returns `(record, version, value)` or `None`), `observe_commits`, `fail(error)`, `failure`
  (`SessionFailure` with `.error`), `failed` (a future failing with `SessionFailed`), `assert_usable()`,
  `report(error, info)`, `unload_documents()`, `document_cache_sizes` (`DocumentCacheSizes` named tuple).
- Protected: `storage` (the guarded Storage), `conversation_created(tx, record)` (async), `before_close()`
  (async), `write_at_close(writes)`.
- `Transaction` internals: `create_root_conversation()`, `submission(id)`, `set_task(record)` (sync),
  `staged_tasks()`, `staged_conversations()`, `settle_failure/settle_success/discard/adopt`.
- Module helpers: `contained(callback, on_error)`, `document_retention(ms)`,
  `set_default_document_retention(ms)` (tests; `PI_DURABLE_DOCUMENT_RETENTION_MS` in `tests/conftest.py`),
  `date_now()` (`Date.now()` as an `int`).

Rules:

- Deliveries (state frames, watch frames) are scheduled with `loop.call_soon` (upstream `queueMicrotask`);
  watch listeners and state subscribers may be sync or async.
- Op lists the Session creates for publications are lists (`RETIREMENT_OPERATIONS == [("r", None)]`,
  creation/retirement `ops == []`), immutable by contract.
- `Tx.append_entry` copies entry JSON with `copy_json` and detaches pi-ai messages (`model`, edit
  `messages`) through `storage.codec.clone_value`; everything else in table records must be strict JSON.
- Error texts naming an API use the Python names (`Tx.scan_conversations() cannot read tables after the
  first table write`, `Tx.create_task() requires options.conversationId`,
  `Session.snapshot_as_of() requires an entry ID`).
- A retention of `math.inf` arms no timer.

## 9. Storage backends, the local environment, and the testing suites (stage 2b)

| Upstream | Python |
|---|---|
| `storage/sqlite/{schema,migrations,storage,index}.ts` | `pi_durable/storage/sqlite/{schema,migrations,storage,__init__}.py` (portable: no driver loaded) |
| `storage/sqlite/node.ts` | `pi_durable/storage/sqlite/stdlib.py`: `open_sqlite_storage(path, *, prefix=None, wal_auto_checkpoint_pages=None, busy_timeout_ms=None)`, re-exports `pi_chord.sqlite.stdlib` |
| `storage/jsonl/{storage,index}.ts` | `pi_durable/storage/jsonl/{storage,__init__}.py` |
| `storage/jsonl/node.ts` | `pi_durable/storage/jsonl/local.py`: `open_local_jsonl_storage(directory, context, *, fsync=False)` |
| `env/node.ts`, `env/node-watch.ts` | `pi_durable/env/local.py` (`LocalExecutionEnv`), `pi_durable/env/local_watch.py` (`LocalFileWatcher`, `LocalWatchOptions`) |
| `testing/*.ts` | `pi_durable/testing/*.py`; `register_storage_conformance(globals(), name, provider)` / `register_env_conformance(...)` define one parametrized pytest test |

Rules:

- **Wire text.** Serializing backends write `storage.codec.json_stringify(value)`, upstream's `JSON.stringify`:
  compact, non-ASCII raw, lone UTF-16 surrogates escaped (`"\ud800"`), so every string survives UTF-8 and
  SQLite and indexed strings (task kinds, request IDs, document kinds and keys, stored JSON-encoded as
  upstream does) compare equal across implementations. Floats use Python's shortest repr (`0.0`, `1e-07`),
  which parses to the same number; key order is the record's. `json_parse` is `JSON.parse` (no `NaN`).
  Entries go through `encode_entry`/`decode_entry`; other records are stored as they are.
- **Byte-compatible schema.** Migration statements are upstream's text, whitespace included (SQLite keeps it
  in `sqlite_schema`); `tests/test_storage_interop.py` checks a TypeScript-written database and JSONL directory
  (`tests/fixtures/storage_interop`) read identically here, and that Python-written ones read like them.
- `SqliteStorage.open(db, *, prefix=None)` and `JsonlStorage.open(directory, fs, context, *, fsync=False)`:
  trailing option objects are keyword-only; `SqliteStorageOptions`/`JsonlStorageOptions` are their
  `TypedDict`s (usable as `**options`). Inside a transaction `SqliteStorage` issues independent statements
  together and awaits them as one batch (in call order on the handle), so a threaded adapter pays one hop per
  batch; a failure is raised after the batch settles, before COMMIT.
- `SqliteStorage.close` is `async def`; repeated calls await the same close, which waits for admitted
  multi-query reads (upstream returns the same Promise).
- `LocalExecutionEnv(*, cwd, shell_path=None, shell_env=None, watch=None)`: a constructor takes keyword
  arguments (upstream's required options object); `watch` is the `LocalWatchOptions` `TypedDict`
  (`mode`, `pollIntervalMs`, `maxDirectories`). Its `id` is `"local"` (upstream `"node:local"`).
  Blocking file calls run on the loop's default executor; a call already running keeps running when its
  awaiting task is cancelled, and readers close their descriptor only after it finished. Commands run through
  `loop.subprocess_exec` in a new session (process group), killed as a tree on timeout, abort, cancellation of
  the awaiting task, or `cleanup()`. `FileError` messages are `str(OSError)`.
- `LocalExecutionEnv.list_files` returns `not_supported` (upstream's Node environment has no `listFiles`).
- Native watching is Linux inotify via `ctypes` (one instance per event loop, watch descriptors shared);
  other platforms and unreliable file systems poll. Scans run on an executor thread.
- Conformance cases are `StorageConformanceCase(name, run, timeout_ms=None)`; `create_assertions()` raises
  `AssertionError` (upstream adapts a Vitest `expect`). The JavaScript-only prototype case is kept as "the
  keys `__proto__`/`constructor` are ordinary detached data".
- Benchmarks: `StorageReadBenchmark(name, run, expected)` and `StorageWriteBenchmark(name, expected, run)`
  dataclasses; `benchmarks/bench_durable_storage.py` runs them for memory, SQLite and JSONL.
- The package root does not import the SQLite/JSONL backends, the local environment or `testing`, as
  upstream's `index.ts` (`tests/test_storage_runtime_boundary.py`).

## 10. The Harness (stage 3)

`pi_durable/harness/` ports `src/harness/` file for file (`task-graph.ts` is `task_graph.py`):
`harness, scheduler, generation, tool, compaction, context, agent, events, inbox, live, prompt, provider,
registry, submissions, task_graph, usage, util, view`, plus `stream.py` (no upstream module: it converts the
stored camelCase `ConversationStreamOptions` to pi-ai's snake_case keyword options, and a resolved
`ConversationRetryPolicy` to `pi_ai.utils.retry.RetryPolicy`). `pi_durable/__init__.py` now mirrors
`index.ts` completely.

API shape:

- `Harness` is the concrete class (`class Harness(SessionImpl)`), opened with
  `await Harness.open(storage, options, context)`; `options` is the `HarnessOptions` dict with upstream's
  keys (`models`, `registry`, `settings`, `env`, `now`, `setTimer`, `onWake`, `onReport`,
  `conversationCreated`). `settings` may be a dict or a function returning one; it is read at every use
  (upstream's getters). The `Harness` protocol stays in `pi_durable.harness.types`; the class satisfies it
  (`tests/typing_cases.py` checks).
- **Eager methods.** As the Session's (section 5), the Harness and conversation methods whose upstream body
  admits work to the Session line before its first `await` are plain `def`s returning an already started
  `asyncio.Future`: `Conversation.configure/submit/reset/compact/commit/context/entries/fork/abort/
  wait_for_idle/agent/view_state/watch` (the view attachment is admitted at the call, so a `close()` that
  follows cannot overtake it), `Harness.root/conversation/create_conversation/get_task/inspect/submission/
  abort_submission/abort_task/retry_task/wait_for_task/wait_for_idle/task_graph/watch_task_graph`,
  `Submission.status/wait/abort`, `ToolExecutionApi.details` (its checks and the update run at the
  call) and `ToolExecutionApi.execute_tool` (its checks, key and admission start at the call, so a caller
  that returns right after still has the call admitted and aborted by cleanup). The protocols in `harness/types.py` declare these `def ... -> Awaitable[T]`. Helpers in
  `harness/util.py`: `eager(awaitable, then)` keeps a started operation as a future; `chain(future, then,
  context=None)` is `promise.then(then)` through done callbacks (no extra task hop; a cancelled `context`
  rejects only the wait).
- The scheduler's `queueMicrotask` is `loop.call_soon`; its drain and reconcile tasks are spawned directly
  (one loop hop, like a microtask).
- Hook tables are dicts keyed by snake_case names (section 2): `GenerationTask` `before_request`,
  `after_response`, `on_yield`, `after_tools`; `ToolTask` `before_tool`, `after_tool`; `CompactionTask`
  `before_compact`. Handlers take upstream's arguments, `context` last, and may be sync or async.
  Internally `agent_hooks()` returns `(extension_name, handlers)` tuples.
- Messages live in documents and task checkpoints as **JSON** (`pi_ai.serialization.to_json_value`,
  read back with `message_from_dict`): `pi.live` `generation.message`, the deferred handle, nested
  `ToolCall`s, `pi.usage` (`Usage` as camelCase JSON: `usage_json(usage)`), inbox write items (`entry`
  holds the draft with its `model` messages serialized). Entries keep pi-ai dataclasses (section 8).
- Runtime commit callbacks are `change(tx, current)`; `Conversation.commit`/`Session.commit` take
  `change(tx)`. `subscribe_commits` listeners are `listener(publication, context)`.
- `watch_events(harness, conversation_id, context) -> AgentEventStream`; `stream.start(listener)` calls
  `listener(events, context)` (sync or async); `stop()` returns a future.
- `Conversation.abort(context, *, background=False)` replaces upstream's `ConversationAbortOptions`.
- Tool `execute(args, api, context)` receives arguments validated against `parameters` (a JSON Schema
  dict, or a Pydantic model, which then receives the model instance). `api.output(chunk, skipped=None)`
  takes an `OutputSkip` or the `ShellOutputSkip` dict.
- Error texts are upstream's except where Python's differ by nature (a non-JSON value reports Python's
  `json` message, "... is not JSON serializable").

Test helpers (`tests/durable_harness_support.py`, ports of `harness-support.ts`, `task-support.ts`,
`chat-support.ts`; import with `from durable_harness_support import ...`):

- Registries: `tool(name)` (empty-object schema, no output), `add_tool/add_task/add_hooks/add_section`
  (each installs a one-item extension and returns an `Installed` with `dispose()`), `counting_reader`.
- `open_harness(storage, tool_names, *, registry=None, on_report=None) -> OpenedHarness(harness, registry)`.
- `open_tasks(storage, tasks, *, registry=None, now=None, settings=None, on_wake=None, set_timer=None) ->
  OpenedTasks(harness, registry, reports, infos)`; `completed(result)`, `aborted_with(reason)`.
- Chat: `chat_setup(**faux_options) -> ChatSetup(faux, models, registry, reports, settings, now)` (faux
  options are snake_case, `tokens_per_second=200, token_size={"min": 1, "max": 1}` to stream slowly);
  `open_chat(storage, setup, *, env=None) -> OpenedChat(harness, root)` opens over the same models,
  registry and **live** `setup.settings` dict (assign keys such as `setup.settings["followUpMode"] = "all"`
  between decisions) and configures the root with `faux/faux-1`. Script the model with
  `setup.faux.set_responses([...])`: messages (`faux_assistant_message`, `faux_text`, `faux_tool_call`) or
  step functions `step(request, options, state, model)` (`request.messages` holds the request's
  `SystemMessage`s with `sections`, `tools_added`, `tools_removed`; `options["signal"]` is the abort
  signal). `setup.faux.state.call_count` counts requests.
- Fake clocks: `setup.now = lambda: 7` (read through a lambda, so it can be replaced after opening);
  `open_tasks(..., now=..., set_timer=...)` for scheduler timers (a `set_timer(callback, delay_ms,
  options)` that records callbacks lets a test fire them by hand).
- Waiting: `flush()` (25 loop turns), `eventually(check)` (polls with `flush`), `wait_for(check,
  timeout_ms=5000)` (real time, for throttle windows and timers), `settled(awaitable)`, `aborted(signal)`
  (raises the reason when it aborts), `unanswered()` (a faux step that blocks until cancelled; `reached`
  resolves once the request was sent). `durable_session_support.deferred()`/`resolve()` are gates.
- `all_entries(conversation)`, `text_of(message)`, `describe_message`, `tools_named(setup, *names)`.
- `match_object(actual, expected)` / `assert_match(actual, expected)`: Vitest's `toMatchObject` (dict keys
  recursively; a dataclass matches a dict of its attribute names); `gated(message)`: a faux step held until
  `release()` or cancellation, with `reached`.
- `race_aborted(awaitable, signal)`: `Promise.race([awaitable, aborted(signal)])` (never cancels a shared gate);
  `StorageSpy(inner)`: delegates every storage method, subclass it to observe or hold some (upstream rebinds
  methods of a storage instance; the Python backends are slotted).
- SQLite storage runs on executor threads, so waits on it poll in real time (`wait_for`, not `eventually`,
  whose fixed number of turns suffices for upstream's synchronous `node:sqlite`).
- Deterministic driving: a Harness schedules nothing until `harness.resume()` (or a `submit`, `compact`,
  `abort`, `wait_for_task` or `wait_for_idle`, which resume); `ControlledStorage(MemoryStorage(), STORAGE_BACKENDS["memory"])` gives
  `hold_commits()` (returns a gate with `entered`/`release()` to occupy the Session line),
  `fail_next_commit(error)`, `crash()` and `reopen()`. Prefer gates (`deferred()`) inside tool `execute`
  functions, faux steps and task phases to sleeps.
- Races: Python commits settle in fewer loop turns than JavaScript's (an `await` of a completed
  awaitable does not yield), so a test that relies on a JS microtask interleaving may need an explicit
  gate. `test_harness_tasks.py`'s double-abort case holds the abort handler on a gate for that reason.

## 11. Tools, tool search, and images (phase 2)

| Upstream | Python |
|---|---|
| `tools/index.ts` | `pi_durable/tools/__init__.py`: `create_coding_tools(*, images=None)`, `CodingTools`, the `create_*_tool` factories and their types |
| `tools/{read,write,edit,bash,image,env}.ts` | `pi_durable/tools/{read,write,edit,bash,image,env}.py` |
| `tools/{edit-diff,image-processor,path-utils,file-mutation-queue}.ts` | `pi_durable/tools/{edit_diff,image_processor,path_utils,file_mutation_queue}.py` |
| `diff` (jsdiff 8.0.4, npm) | `pi_durable/tools/_jsdiff.py` (private; BSD-3-Clause notice in its header) |
| `tool-search/{index,tool-search}.ts` | `pi_durable/tool_search/{__init__,tool_search}.py` |
| `images/index.ts`, `images/node.ts` (+ `node-worker.ts`) | `pi_durable/images/__init__.py` (Pillow): `prepare_image`, `PillowImages`, `create_pillow_images` |
| `images/exif.ts` | `pi_durable/images/exif.py` |

Not ported: `images/cloudflare.ts`, `images/photon-wasm.d.ts` (Photon-specific).

API shape:

- Factories take upstream's options object as keyword arguments: `create_read_tool(*, images=None)`,
  `create_bash_tool(*, command_prefix=None, prepare=None)`, `create_powershell_tool(*, command_prefix=None,
  prepare=None, programs=None)`. `BashExecution` is a `TypedDict` with upstream's keys (`command`, `cwd`, `env`,
  `inheritEnv`) that `prepare(execution, api, context)` changes in place (sync or async).
- Data shapes keep upstream's keys: `ImageLimits` (`maxWidth`, `maxHeight`, `maxBytes`), `PreparedImage` (`data`,
  `mimeType`, `resized: {"from", "to"}`, `convertedFrom`), `EditToolDetails` (`diff`, `patch`, `firstChangedLine`),
  `ReadToolDetails` (`truncation`), the bash `structuredOutput` (`output`, `truncated`, `fullOutputPath`, `exitCode`),
  `ToolSearchToolDetails` (`loaded`). `ImageProcessor` is a `Protocol` with `async def prepare(data, mime_type,
  limits)`. `ByteSource` (image header sniffing) is a frozen dataclass `(size, read)`.
- Tool names, descriptions, and parameter schemas equal upstream's (compared against upstream's `createCodingTools()`
  and `createPowerShellTool()` under Node), so the transcript and prompt caches see the same tools.

Rules:

- **Exact text.** What the model sees is upstream's text: `edit`'s diff and patch come from the jsdiff port (Myers
  `diffLines`, `structuredPatch`, `formatPatch`, `createTwoFilesPatch`), never `difflib`. `tests/test_tools_edit_diff.py`
  replays a corpus run through upstream's `edit-diff.ts` with jsdiff 8.0.4 under Node
  (`tests/fixtures/edit_diff/upstream.json`). JavaScript string functions that differ are reproduced: `trimEnd`'s
  whitespace set, `split("")` counting UTF-16 code units, `JSON.parse` without `NaN`.
- **JavaScript numbers in `read`.** `offset` and `limit` are used as floats with `slice`, `Math.trunc`, NaN-propagating
  `Math.min`, and `Number.isSafeInteger` semantics (`tools/_js.py`), and printed like `String(number)` (`2.5`, `61`,
  `100000000000000000000`), so every argument selects and reports what upstream does
  (`tests/test_tools_read_differential.py`, whose reference read uses the same seeds as upstream's).
- **Mutation queue.** Per event loop, keyed by `env.id` and canonical path; a call waiting for its turn waits through
  `asyncio.shield`, so cancelling it does not cancel the call ahead of it, and its slot is released in `finally`.
- **Images.** `pi_durable.images` imports Pillow at module load and raises `ImportError` naming
  `pi-ai-python[images]` without it; `pi_durable` and `pi_durable.tools` never import it
  (`tests/test_storage_runtime_boundary.py`). `prepare_image` follows Photon's rules: pass-through of an upright,
  inline, within-limits image after it decodes; EXIF orientation from `exif.py` (Pillow's own transpose per
  orientation); scale to fit with `Math.round`; PNG then JPEG 80/70/55/40 (JPEG first for JPEG sources), then 3/4
  steps down to 1x1; BMP and other non-inline formats become PNG; decoded as RGB, or RGBA with transparency, alpha
  dropped for JPEG; LANCZOS resampling. Output bytes differ from Photon's. Where only the byte limit decides how far
  an image shrinks, Pillow's JPEGs (smaller at the same quality) may stop at a larger size than Photon's.
- **Off the loop.** `PillowImages` runs `prepare_image` on one worker thread of its own (started at the first image,
  `close()` stops it), as upstream's single worker thread; `worker=False` runs it on the calling thread; `executor=`
  takes any `concurrent.futures.Executor`. A thread, not a process: Pillow releases the GIL while it decodes,
  converts, resizes, and encodes, so the loop keeps running (the test measures the longest loop stall), and nothing is
  pickled; a `ProcessPoolExecutor` works too, since `prepare_image` and its arguments pickle. There is no idle
  timeout: an idle thread holds no image memory, unlike Photon's WebAssembly memory.
- `tool_search` builds a tool's search text from `pi_ai.utils.schema.parameters_schema(tool.parameters)`, so Pydantic
  parameters are searched like JSON Schema ones.
