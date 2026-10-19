# pi_chord

The parts of [`@earendil-works/chord`](https://github.com/earendil-works/pi/tree/main/packages/chord) that [`pi_durable`](../durable/README.md) uses, ported to Python: explicit cancellation contexts, strict JSON, the transactional JSON tracker with drafts and operation batches, replicated state, bounded output, and an async SQLite facade with an adapter for the standard `sqlite3` module. Installed with `pip install pi-ai-python`.

Chord's objects, contracts, hosts, links, and hot reload are not part of this port, nor is the Cloudflare Durable Object SQLite adapter. Operation batches and SQLite schemas stay compatible with the TypeScript package.

| Module | Contents |
|---|---|
| `pi_chord.context` | `Context`, `BACKGROUND_CONTEXT`, `with_cancel`, `with_abort_signal`, `without_abort_signal`, `with_context_value`, `create_context_key`, `await_with_context` |
| `pi_chord.json` | `copy_json`, `is_json_value`, `UNDEFINED`, `JsonValue`, `JsonObject` |
| `pi_chord.delta` | `track`, `Tracker`, `Draft`, `apply`, `apply_immutable`, `diff_revisions`, `encoder`, `decoder`, the `Op` tuples |
| `pi_chord.state` | `replicated_state`, `MutableReplicatedState`, `AttachedReplicatedState`, `ReplicatedStateSource` |
| `pi_chord.output` | `OutputBuffer`, `OutputLimits`, `bound_output`, `sanitize_output`, `utf8_byte_length`, `Progress` |
| `pi_chord.sqlite` | `SqliteDatabase`, `SqliteExecutor`, `SqliteMigration`, `apply_sqlite_migrations`, `validate_sqlite_schema_prefix` |
| `pi_chord.sqlite.stdlib` | `open_sqlite_database`, `SqliteDatabaseOptions`, `StdlibSqliteDatabase` |

The package root re-exports the context, JSON, `Draft`, and replicated state names. The examples below use top-level `await` (as `python -m asyncio` allows; in a script, put them in an `async def`).

## Context

A `Context` is an immutable value passed explicitly to every operation. It carries cancellation (a `pi_ai.AbortSignal`) and scoped values; deriving a context never changes its parent.

```python
import asyncio

from pi_chord.context import (
    BACKGROUND_CONTEXT,
    await_with_context,
    create_context_key,
    with_cancel,
    with_context_value,
    without_abort_signal,
)

USER = create_context_key("user")
context = with_context_value(USER, "ann", BACKGROUND_CONTEXT)
print(context.value(USER))  # ann

child = with_cancel(context)  # a context you can cancel, and its cancel function
child.cancel()
try:
    await await_with_context(asyncio.sleep(10), child.context)
except Exception as error:
    print(type(error).__name__)  # AbortError

cleanup = without_abort_signal(child.context)  # same values, no cancellation: for mandatory cleanup
print(cleanup.abort_signal)  # None
```

`BACKGROUND_CONTEXT` is never cancelled. `context.throw_if_aborted()` raises the cancellation reason, and `context.abort_signal` is the signal itself, which `pi_ai` requests accept directly. `await_with_context()` cancels only the wait; the awaited work keeps running in its own task.

## JSON

Strict JSON is `None`, `bool`, `int`, finite `float`, `str`, and exact `list` and `dict` (string keys) without cycles. Tuples, subclasses, and other objects are rejected rather than converted:

```python
from pi_chord.json import UNDEFINED, copy_json, is_json_value

print(copy_json({"a": [1, 2.5, None], "b": UNDEFINED}, omit_undefined_properties=True))  # {'a': [1, 2.5, None]}
print(is_json_value({"x": float("nan")}))  # False
try:
    copy_json({"pair": (1, 2)})
except TypeError as error:
    print(error)  # Value contains a non-JSON tuple; expected strict JSON
```

`UNDEFINED` stands for JavaScript's `undefined`: a key that is present in code and absent from JSON.

## Delta and Drafts

`track(initial)` starts a tracker over an immutable JSON revision. A change hands out a mutable draft; preparing it yields the exact operations that turn the old revision into the new one, and adopting it makes the new revision current:

```python
from pi_chord.delta import apply_immutable, decoder, diff_revisions, encoder, track

tracker = track({"output": "", "count": 0})  # the tracker owns this value now
change = tracker.begin_change()
change.state["output"] += "done\n"  # the draft is mutable only while the change is open
change.state["count"] += 1
prepared = change.prepare()
tracker.adopt(prepared)

print(prepared.ops)  # [('a', ('output',), 'done\n'), ('s', ('count',), 1)]
print(tracker.value)  # {'output': 'done\n', 'count': 1}
replica = apply_immutable(prepared.base, prepared.ops)  # shares containers: never mutate it
print(replica == tracker.value)  # True

print(diff_revisions({"items": ["a"]}, {"items": ["a", "b"]}))  # [('p', ('items',), 1, 0, ['b'])]

wire = encoder().encode(prepared.ops)  # interns repeated paths for one ordered stream
print(decoder().decode(wire) == prepared.ops)  # True
```

Operations are tuples (JSON arrays on the wire): `("r", value)` replaces the whole value, `("s", path, value)` sets, `("d", path)` deletes, `("a", path, text)` appends to a string, `("t", path, count)` removes a string's first `count` UTF-16 code units, `("p", path, index, remove, items)` splices an array, and `("m", path, permutation)` reorders one. Paths are tuples of keys and indices. Counting UTF-16 code units keeps batches interchangeable with JavaScript replicas.

Immutability is an ownership contract, not enforced: nothing is frozen or copied defensively. Never mutate a value you passed to `track()`, `replicated_state()`, or `replace()`, or one you received as `tracker.value`, `prepared.ops`, or a published state value. A draft and the handles read from it work only while their change is open; afterwards every use raises `TypeError`. Values assigned into a draft are copied. Drafts follow Python semantics where JavaScript's differ: `del draft[key]` raises `KeyError` for a missing key (assign `UNDEFINED` to delete quietly), and an index write must address an existing element (use `append()`).

## Replicated State

`replicated_state(initial)` is an observable JSON value. Each `change()` publishes one atomic revision; subscribers are called in order, one delivery at a time per subscriber:

```python
from pi_chord import BACKGROUND_CONTEXT, replicated_state

status = replicated_state({"output": "", "count": 0})


def show(value, context, delivery):
    print(delivery.kind, delivery.sequence, value)


unsubscribe = status.subscribe(show)  # hydrate 0 {'output': '', 'count': 0}


def bump(draft):
    draft["output"] += "done\n"
    draft["count"] += 1


status.change(BACKGROUND_CONTEXT, bump)  # update 1 {'output': 'done\n', 'count': 1}
status.replace(BACKGROUND_CONTEXT, {"output": "", "count": 5})  # update 2 {'output': '', 'count': 5}
unsubscribe()
```

A subscriber is called as `listener(value, context, delivery)` and must accept all three arguments; it may be a coroutine function, and the next delivery waits for it. Read the `value` it receives: `status.value` may already be a later revision. At most 100 deliveries wait behind a running callback; on overflow only the newest is kept, so update sequences may skip. A failing subscriber is reported (to the event loop's exception handler for a mutable state, to `on_error` for an attached one) and delivery continues; a subscriber that takes fewer arguments fails this way on every delivery. A `change()` callback must be synchronous; if it raises, nothing changes.

`replicated_state(source, ReplicatedStateSourceOptions(on_error=...))` instead attaches a read-only `AttachedReplicatedState` to a `ReplicatedStateSource`, which publishes every committed revision in cursor order. pi_durable's documents, conversation views, and task graph are such states; call `dispose()` when done.

## Output

`OutputBuffer` keeps a head or tail window of a stream at a cost proportional to each chunk, and counts what it dropped. Sizes are UTF-8 bytes:

```python
from pi_chord.output import OutputBuffer, OutputLimits, bound_output, sanitize_output

buffer = OutputBuffer(OutputLimits(max_bytes=10, max_lines=2, retain="tail"))
buffer.push("line 1\nline 2\nline 3\n")  # str or bytes; bytes are decoded as a stream
buffer.end()
print(buffer.snapshot())  # BoundedOutput(text='line 3\n', dropped_bytes=14, dropped_lines=2)

print(bound_output("line 1\nline 2\n", OutputLimits(max_bytes=100, max_lines=1, retain="head")).text)  # line 1
print(repr(sanitize_output("a\x07b\tc\n")))  # 'ab\tc\n': control characters go, tabs and newlines stay
```

`Progress(write, on_error, min_interval_ms)` paces commits of a changing value. `mark()` schedules a commit; the first change after an idle period commits at once, and each commit delays the next by at least `min_interval_ms` and by its size at 100 KiB/s. Changes made meanwhile coalesce:

```python
import asyncio

from pi_chord.output import Progress

text = {"value": ""}
commits = []


async def write():
    commits.append(text["value"])
    return len(text["value"].encode())  # bytes written


progress = Progress(write, on_error=print, min_interval_ms=50)
for i in range(5):
    text["value"] += f"{i}\n"
    progress.mark()
    await asyncio.sleep(0.01)
await progress.mark_and_wait()  # settles with the commit that includes this change
await progress.stop()
print(len(commits) < 5, commits[-1] == text["value"])  # True True
```

## SQLite

`pi_chord.sqlite` defines the async facade every SQLite adapter implements: `exec(sql)`, `run(sql, *params)`, `get(sql, *params)`, `all(sql, *params)`, `transaction(callback)`, and `close()`. Rows are dicts keyed by column name. `apply_sqlite_migrations()` applies versioned migrations atomically, tracking the version in a metadata table:

```python
from pi_chord.sqlite import SqliteMigration, apply_sqlite_migrations
from pi_chord.sqlite.stdlib import SqliteDatabaseOptions, open_sqlite_database

database = await open_sqlite_database("./notes.sqlite", SqliteDatabaseOptions(busy_timeout_ms=1000))
migrations = [
    SqliteMigration(1, ["CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL)"]),
    SqliteMigration(2, ["ALTER TABLE notes ADD COLUMN created INTEGER"]),
]
await apply_sqlite_migrations(database, migrations, table="app_schema")  # versions start at 1 and are contiguous


async def add_note(transaction):
    await transaction.run("INSERT INTO notes (text, created) VALUES (?, ?)", "hello", 1)


await database.transaction(add_note)
print(await database.all("SELECT id, text, created FROM notes"))  # [{'id': 1, 'text': 'hello', 'created': 1}]
print(await database.get("SELECT version FROM app_schema"))  # {'version': 2}
await database.close()
```

The standard library adapter (`pi_chord.sqlite.stdlib`, upstream's `node:sqlite` adapter) runs every `sqlite3` call of a database on one worker thread that owns the connection, so the event loop never blocks on SQLite. It opens a file in WAL mode with `synchronous = NORMAL`, a 5 s busy timeout, and an auto-checkpoint every 1,000 pages (`SqliteDatabaseOptions`); `":memory:"` opens a private in-memory database. Operations run in call order. A transaction holds the connection from `BEGIN IMMEDIATE` to `COMMIT`: its callback must use the handle it receives, and everything else waits until it ends, so calling the database itself inside the callback never returns. A raising callback rolls back. Values bind as `None`, `int`, `float`, `str`, or bytes; `bool` and other types are rejected. The methods take no `Context`; cancelling the awaiting task skips a statement the worker has not reached and rolls back a transaction.

## Differences from the TypeScript Package

- Only the modules above are ported; see the table at the top.
- Names are snake_case (`beginChange` is `begin_change`, `withCancel` is `with_cancel`). Options are keyword arguments or small dataclasses (`OutputLimits`, `SqliteDatabaseOptions`, `ReplicatedStateSourceOptions`).
- Operations and paths are tuples in memory; decoders and appliers also accept the JSON arrays read back from storage or the wire.
- Listeners receive upstream's arguments positionally; like JavaScript, a listener may declare fewer (`lambda value: ...`).
- `await_with_context()` raises the signal's reason when it is an exception, otherwise `pi_ai.AbortError`.
- The SQLite adapter wraps `sqlite3` instead of `node:sqlite`; integer results are exact Python ints, without JavaScript's safe-integer limit.

## License

MIT
