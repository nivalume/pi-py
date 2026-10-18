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
| `Promise<T>` method | `async def` | |
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

- `harness/types.py` aliases `ConversationView`, `TaskGraph`, `TaskGraphWatch`, `UsageState` to `Any`
  until `harness/view.py`, `task_graph.py`, `usage.py` exist; replace the aliases with imports.
- `GenerationContinue` is `{"continue": input}` (functional `TypedDict`, `continue` is a keyword).
- `ToolHookCall` subclasses `pi_ai.ToolCall` with a keyword-only `parent`.
- `ToolRegistration` subclasses `pi_ai.Tool`; it is hashed and compared by identity.
- `FileSystem.list_files` is a required method (upstream optional): an environment that cannot list
  returns `err(FileError("not_supported", ...))`.
- `get_or_undefined` is `get_or_none`; binary data is `bytes`.
- Static checks: add `# pyright: ignore` lines (expected errors) and `assert_type` calls to
  `tests/typing_cases.py`; `tests/test_types.py` runs pyright on it.
