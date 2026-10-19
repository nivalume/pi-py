# pi_durable

> **Experimental.** The API changes without notice between releases.

A durable agent harness. Conversations, model turns, tool calls, and your own state are committed to storage before anything is shown. If the process dies mid-turn, reopening the storage picks the work up where it stopped.

Python port of [`@earendil-works/pi-durable`](https://github.com/earendil-works/pi/tree/main/packages/durable) 1.3.0. Built on [`pi_ai`](../ai/README.md) for model access and [`pi_chord`](../chord/README.md) for document state. Installed with `pip install pi-ai-python`.

## Table of Contents

- [Installation](#installation)
- [What Phase 1 Includes](#what-phase-1-includes)
- [Quick Start](#quick-start)
- [Two Python Rules](#two-python-rules)
- [Concepts](#concepts)
- [Persist and Resume](#persist-and-resume)
- [Extensions](#extensions)
- [Tools](#tools)
- [System Prompt](#system-prompt)
- [Per-Conversation Agent](#per-conversation-agent)
- [Settings](#settings)
- [Environment](#environment)
- [Reload](#reload)
- [Watching a Conversation](#watching-a-conversation)
- [Busy Conversations](#busy-conversations)
- [Reset and Handoff](#reset-and-handoff)
- [Compaction](#compaction)
- [Agent Events (Experimental)](#agent-events-experimental)
- [Hooks](#hooks)
- [More Conversations and Forks](#more-conversations-and-forks)
- [Abort and Subagents](#abort-and-subagents)
- [Child Tasks](#child-tasks)
- [Task Graph](#task-graph)
- [Your Own State](#your-own-state)
- [Usage and Cost](#usage-and-cost)
- [Storage](#storage)
- [Errors](#errors)
- [Differences from the TypeScript Package](#differences-from-the-typescript-package)
- [Design Notes](#design-notes)

## Installation

```bash
pip install pi-ai-python    # contains pi_durable, pi_chord, pi_ai and pi_agent
```

Python 3.11 or newer. Everything runs on `asyncio`.

## What Phase 1 Includes

This release ports the core of pi-durable: the Session and its storages (memory, SQLite, JSONL), the Harness with its built-in generation, tool, and compaction tasks, the registry, extensions, tools you define, hooks, documents, child tasks, the task graph, views, watches, agent events, the local execution environment, and the storage and environment conformance suites.

Not included yet:

- The built-in coding tools of `@earendil-works/pi-durable/tools` (`read`, `write`, `edit`, `bash`, and image reading) and the `CodingTools` extension. Define your own tools with `define_tool()` (see [Tools](#tools)); they get the [environment](#environment) through `api.env`.
- Tool search (`@earendil-works/pi-durable/tool-search`). Deferred tools (`deferred=True`) and `control.addTools` work; there is no `tool_search` tool that loads them.
- Image processing (`@earendil-works/pi-durable/images`).
- The code mode extension (`@earendil-works/pi-durable/codemode`).
- The Cloudflare adapters: Durable Object SQLite storage and the Workers image processor.

## Quick Start

```python
import asyncio

from pi_ai import create_models
from pi_ai.providers.openai import openai_provider
from pi_chord.context import BACKGROUND_CONTEXT
from pi_durable import AssistantEntry, Harness, MemoryStorage, create_registry

context = BACKGROUND_CONTEXT


async def main() -> None:
    models = create_models()
    models.set_provider(openai_provider())  # reads OPENAI_API_KEY

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

What happened:

- `Harness.open()` opens a Session over a storage backend. `MemoryStorage` keeps everything in memory.
- `root()` returns the root conversation, creating it on first use with the given agent choices. A conversation is a transcript of immutable entries.
- `submit()` durably admits your input and returns a `Submission`. A built-in generation task calls the model and appends the answer.
- `wait()` returns once the input is answered (`done`, with the answer's entry ID in `answer`) or has failed (`unanswered`, with a `reason`).

Every async call takes a Chord `Context` (`pi_chord.context`), which carries cancellation. `BACKGROUND_CONTEXT` never cancels. Cancelling a wait only cancels that wait, never the work: pass `with_cancel(context).context` and call its `cancel()` to stop waiting.

The other examples in this README are fragments. They use top-level `await` (as `python -m asyncio` allows; in a script, put them in an `async def`) and the `harness`, `root`, `models`, and `context` of the quick start, plus a `registry` you keep: `registry = create_registry()`, passed as `"registry": registry`.

## Two Python Rules

Most of the API is upstream's with snake_case names. Two rules are specific to the port.

**Records are TypedDicts with upstream's camelCase keys.** Everything that is data (records, submissions, task states and outcomes, documents, settings, option objects, tool results, events) is a plain `dict` typed as a `TypedDict`, and its keys are exactly the TypeScript property names: `settled["answer"]`, `record["conversationId"]`, `{"type": "input", "content": "Hi", "whenBusy": "steer"}`, `{"ownership": {"kind": "task", "taskId": id}}`. A record is its persisted JSON, so a SQLite file or JSONL directory written by TypeScript pi-durable opens here and the other way round; the test suite checks this against a database and a JSONL directory written by the TypeScript package. Method names, function names, and keyword arguments are snake_case (`root.view_state()`, `define_tool(structured_output_schema=...)`); dict keys never are. The one exception is hook tables, which are method tables rather than data: `hook(ToolTask, {"before_tool": ...})`. Messages and content blocks are `pi_ai` dataclasses (`UserMessage`, `TextContent`, ...).

**Many methods return an already started future.** In JavaScript, an `async` function runs up to its first `await` as soon as it is called, so `root.submit()` admits the submission at the call. A Python coroutine does nothing until it is awaited. Where that difference matters, the port's methods are plain functions that do their work at the call and return a started `asyncio` future: `Conversation.submit`, `configure`, `commit`, `reset`, `compact`, `fork`, `abort`, `watch`, `view_state`, and the like; `Harness.root`, `conversation`, `create_conversation`, `wait_for_task`, `close`, and the like; and every `Tx` method that reads or writes. You normally just `await` them. To start one and keep going, keep the future:

```python
pending = root.submit({"type": "write", "entry": {"kind": "app.note", "data": "saved"}}, context)
# The submission is already admitted: a close that follows cannot overtake it.
submission = await pending
print(submission.id)
```

Use it like any future (`asyncio.gather()`, `asyncio.ensure_future()`, `add_done_callback()`). `asyncio.create_task()` raises `TypeError` for it, because it is not a coroutine, after the call has already admitted the work. Callbacks you hand to the Harness (commit changes, tool `execute`, task phases, hooks, sections, listeners) may be plain functions or coroutine functions; the Harness awaits whatever is awaitable.

## Concepts

- **Harness**: one open storage plus the machinery that runs agents on it. All changes go through one line of atomic commits, and nothing is shown before its commit is stored.
- **Conversation**: a transcript. `root()` creates the root conversation on first use; you can create more and fork them. A `Conversation` handle holds no state; compare handles by `id`.
- **Entry**: one immutable transcript record, such as a user message (`pi.user`), a model response (`pi.assistant`), a tool result (`pi.tool-result`), a system prompt change (`pi.system`), a reset (`pi.reset`), or your own kind. The model sees the entries from the newest reset onward.
- **Commit**: an atomic write. `conversation.commit(change, context)` calls `change(tx)`, which can append entries, edit documents, and create tasks together; either all of it is stored or none of it.
- **Document**: typed JSON state stored next to the transcript and changed in commits. Built-in ones hold each conversation's agent choices (`pi.agent`), provider-facing session identity (`pi.provider`), running generation and tools (`pi.live`), queued submissions (`pi.inbox`), and spend (`pi.usage`).
- **Task**: a durable state machine that saves a checkpoint at every step, so a restarted process continues from the last one. Every task has an owner: its conversation, or another task. The Harness runs answers as built-in tasks: `pi.generation` calls the model and owns the `pi.tool` tasks of its tool calls, waits for them, and hands the run to the next generation.
- **Submission**: something you hand to a conversation, either user input or an entry to write, which you can wait for.
- **Turn and run**: a turn is one model response and its tool calls; a run is the turns from an input to its final answer. A conversation is busy while a run is going.
- **Extension**: a named bundle of tools, system prompt sections, hooks, wrappers, and tasks.
- **Registry**: the extensions this process installed. It can change while the Harness runs; new work uses the new state.
- **Agent**: what a conversation runs with: model, thinking level, selected extensions, tools, instructions, and working directory. Stored per conversation as names in `pi.agent`, resolved against the registry at each use.

One answered input, as entries and tasks:

```text
submit(input) → pi.user
  pi.generation → pi.system (only if the prompt or tools changed), pi.assistant (tool calls)
    pi.tool × n → pi.tool-result × n   (owned by the generation, which waits for them)
  pi.generation → pi.assistant (answer) → submission done
```

## Persist and Resume

Use SQLite or JSONL storage to keep conversations across restarts:

```python
from pi_durable.storage.sqlite.stdlib import open_sqlite_storage

harness = await Harness.open(await open_sqlite_storage("./session.sqlite"), {"models": models, "registry": registry}, context)
root = await harness.root(context)  # the same root as last time
harness.resume()  # continue any run the last process left unfinished
```

Work interrupted by a crash or close stays pending. `resume()` starts the task scheduler; submitting or waiting starts it too. Each conversation has its own persisted UUIDv7 in `pi.provider`, forwarded to pi_ai as `session_id` for provider prompt caches and session affinity. It survives reopen, retries, reset, compaction, and model changes; a child or fork receives a fresh identity. A conversation stored without one receives and persists one before its first generation or compaction request.

A retried submission with the same `requestId` returns the existing submission instead of submitting twice:

```python
submission = await root.submit({"type": "input", "content": "Hello", "requestId": "greeting-1"}, context)
# After a restart: the same request ID finds the same submission.
again = await root.submit({"type": "input", "content": "Hello", "requestId": "greeting-1"}, context)
assert again.id == submission.id
```

`harness.submission(id, context)` reacquires a submission by ID, for example to wait for it after a restart.

### Sleeping and Waking Up

A task waits for time by ending its phase asleep. The sleep is stored, so the task holds no invocation meanwhile, survives restarts, and resumes at the checkpoint once the Harness clock reaches `until`:

```python
from typing import Any, TypedDict

from pi_durable import Task, define_extension, define_task


class ReminderInput(TypedDict):
    text: str
    at: float  # epoch milliseconds


async def wait(task, runtime, context):
    # Returns once the sleep is stored; the phase ends there.
    await runtime.sleep_until(task["input"]["at"], {"phase": "deliver"}, context)


async def deliver(task, runtime, context):
    await notify(task["input"]["text"])  # your code
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "completed", "result": None}}, context)


async def abort(task, runtime, context):
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, context)


# Annotate the token: the result type (None here) appears in no argument.
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

A phase handler is `handler(task, runtime, context)`; `task` is the task record, and `runtime.commit(change, context)` calls `change(tx, current)`, whose returned state replaces the task's. There is no sleep in the middle of a phase: a phase reruns from its top after a restart, so work after the wait goes in the next phase, and what it needs goes in the checkpoint. `{"status": "sleeping", "until": ..., "checkpoint": ...}` returned from a `runtime.commit()` change is the same sleep, committed together with other writes. Retry backoffs and deferred-response polls of the built-in tasks sleep the same way.

The timers of the Session, the scheduler, and its tasks go through the `setTimer` option (default: `loop.call_later` on the running event loop), and their times through `now` (default: `time.time()` in milliseconds); pass both to run a Harness on a fake clock. `setTimer(callback, delay_ms, options)` returns a function that disarms the timer, or `None` to decline a timer whose `options["background"]` is true (those only free memory). The local environment and file watcher keep their own timers. Sleep deadlines are stored on `now`, so every process that opens one storage must use the same clock.

An asyncio timer does not keep a process alive the way a Node.js timer does: `asyncio.run(main())` ends when `main()` returns, sleeping tasks or not. Keep the loop running while you expect work, for example by awaiting `harness.wait_for_task(id, context)` or `harness.closed`.

A host whose process can be stopped between events wakes it itself. With `onWake`, the Harness arms no timer for sleeps; it tells the host what `harness.wake()` says whenever that changes: `active` while some task can run now or is running, and `at`, the earliest sleep. The host keeps its own alarm armed and calls `resume()` when it fires:

```python
import time

alarm = {"at": None}  # epoch milliseconds of the host's alarm, or None


def on_wake(wake):
    # A heartbeat while work runs; otherwise the earliest sleep, if any.
    alarm["at"] = time.time() * 1000 + 60_000 if wake["active"] else wake.get("at")


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "onWake": on_wake}, context)
on_wake(harness.wake())  # arm before anything can start work
harness.resume()  # work a lost process left, and due sleeps

# Later, when the host's alarm fires:
harness.resume()
```

Arm the alarm before any call that can start work (`submit`, `compact`, `abort_task`, `abort`, a registry change): `onWake` runs after the commit. An early alarm is harmless; a missing one strands the work.

### Failures and Recovery

**Crash loops.** Work that deterministically crashes its process, by exhausting memory or CPU before its next checkpoint, would otherwise be resumed and crash again forever. Open counts each task a crash, eviction, or reset cut off; a close is not counted, and a new state the task commits starts the count over. Once the count reaches `settings["restartLimit"]` (default 5; `math.inf` never parks), the task is parked: `inspect()` and the task graph show it as blocked with reason `restart_limit`, `wake()` does not count it as active, and `onReport` gets it once as `task_parked`. Every task running when a process dies is counted, so show parked tasks and offer `harness.retry_task(id, context)`, which runs one again. `abort_task()` still runs its abort handler; an abort handler cut off that many times ends the task `faulted`.

**Reports.** `onReport(error, info)` says what happened in `info["type"]`: `session_failed` for the error that failed the Harness (a storage error); `task_faulted`, `task_parked`, `migration_failed`, `task_outdated`, or `task_error` for a task, with `info["task"]` (its `id`, `kind`, and `conversationId`) and the extension that installed it in `info["extension"]`; `extension_error` for hooks, sections, wrappers, and tools, with the extension's name; `host_error` for a host callback, named in `info["callback"]`; and `listener_error`. A host can feed `task_faulted` reports of one kind to its own breaker.

```python
def on_report(error, info):
    print(info["type"], info.get("task", {}).get("kind"), error)


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "onReport": on_report}, context)
```

**Queued inputs after a failed run.** A run that fails with a model error leaves the inputs queued behind it waiting until the next input arrives. `settings["afterFailedRun"]` changes that: `continue` starts the next run with them, as after an answer, and `withdraw` settles them `unanswered` with reason `run_failed`.

**Blocked tasks.** Tasks blocked on a missing or unmigratable definition show their reason in the task graph too, and a registry change that unblocks them makes `onWake` report the Session active, so a host need not poll them.

## Extensions

Code the Harness runs, other than its built-in tasks, comes in named extensions installed in a registry your process owns:

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

An extension may bring `tools`, `sections`, `hooks`, `wraps` (decorators of a tool or section by name), and `tasks`. By default every conversation selects every installed extension, in install order. Nothing in the registry is stored; conversations store extension names.

## Tools

Phase 1 has no built-in tools (see [What Phase 1 Includes](#what-phase-1-includes)); define your own. `define_tool()` takes keyword arguments. `parameters` is a JSON Schema built with `pi_ai.Type` (or a plain dict, or a Pydantic model), which the Harness validates before `execute(args, api, context)` runs. `api.output()` streams running output, which becomes the result when `execute()` returns no `output`:

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

Each call runs as its own durable task. Its intent is committed before `execute()` runs. If the process dies mid-call, the tool reruns on reopen only when it is declared `replay="safe"`; otherwise the model gets an `interrupted` error result with the output committed so far. Raising from `execute()` gives the model an error result. A result can also return `usage` (a `pi_ai.Usage`), which is added to the conversation's [usage](#usage-and-cost). It can also return `"control": {"terminate": True}`: when every result of the round asks for it, the run ends without another model request.

A result is a dict with three channels: `output` for the model (a list of `pi_ai` `TextContent` and `ImageContent`), `structuredOutput` for programs that call the tool (see [Calling tools from tools](#calling-tools-from-tools)), and `details` for UIs. Use the `pi_ai` content classes in `output`, not dicts: output limits (`output_limits`) only bound `TextContent`. A tool that declares `structured_output_schema` returns a matching `structuredOutput`, which the Harness validates. Programs calling a tool without one get its bounded output: one text item as a string, one image as its `ImageContent`, nothing as `""`, and anything else as the content list, for errors too. `structuredOutput` is always the tool's own value: a schema tool's error result keeps the value it returned, and the text of one without is in a `tool_error` diagnostic, as a raised error's message is. The transcript stores only `output` and `details`:

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

A missing, undeclared, or invalid `structuredOutput` turns a nested call's result into an error result with an `invalid_structured_output` diagnostic; for a call the model made, which never sees it, it is dropped and reported through `onReport`. An error result may omit it, and programs then find its text in a `tool_error` diagnostic; error results the Harness writes itself (blocked, invalid, interrupted, aborted) carry none, and their `diagnostics` say what went wrong. An `after_tool` hook that redacts a schema tool's `output` must redact its `structuredOutput` too.

A later extension's tool with the same name replaces an earlier one where both are selected, and `wrap_tool()` decorates whichever tool won. A `ToolRegistration` is a dataclass; a wrapper returns a changed copy with `dataclasses.replace()`:

```python
import dataclasses
import time

from pi_durable import wrap_tool


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
```

### Calling tools from tools

A tool calls another tool with `api.execute_tool(name, args, context, *, key=None, progress=None)`, as an MCP bridge does. The nested call is its own tool task, owned by the calling one: it is validated, runs the `ToolTask` hooks, keeps its own output limits and replay policy, and returns its result to the caller instead of the transcript:

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
    # Absent when the nested call did not finish.
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

- The caller gets the nested call's `taskId`, `structuredOutput`, `details`, `diagnostics`, `usage`, `isError`, and `durationMs`, not the `output` meant for the model. A blocked, invalid, failed, or aborted nested call returns an `isError` result. `execute_tool()` itself raises once `execute()` has returned, when the caller is aborted, or when the Harness closes; cancelling its `context` stops only the wait.
- Hooks see a nested call's `call.parent`, so a guard can treat calls from tools differently.
- While the caller runs, its nested calls show in `docs["pi.live"]["nestedTools"]`, with the arguments they run with and a `summary` (error, duration, usage) once done, and as tool events with `parentToolCallId` and `parentTaskId`. `progress=False` keeps a nested call's running output out of `pi.live`. Before the caller settles, nested calls it left running are aborted, also those an earlier attempt made. The caller's result message does not list its nested calls; their task records keep them.
- Each nested call has a key, its position among the caller's nested calls unless the caller passes `key=`. A replay-safe caller that reruns after a crash and calls in the same order, or passes the same keys, gets the nested calls it already made back, finished or still running, instead of starting them again. A nested call the crash interrupted before its result was committed follows its own `replay` policy, so a replay-safe tool with external effects still needs its own idempotency. A caller that is not replay-safe never reruns, so after a restart its unfinished nested calls are abandoned, with everything they own, before they run again.
- A nested call's `usage` is counted under its own tool; passing it on in the caller's result counts it twice.
- Nested results are not kept: they live in the caller's task documents (`NestedCallDoc`, one per nested call, keyed by its key, holding `{"taskId": ..., "result": ...}`) only until the caller settles. The nested call's task record keeps only a small receipt.
- Nothing bounds how deeply calls nest; a tool with the default `callers` may even call itself. Restrict `callers` where that matters (below).

### Who may call a tool

`callers` on a tool says who may call it: the model, other tools through `execute_tool()`, or both, the default. A conversation's `modelTools` narrows which of its tools the model is offered, without taking them away from tools:

```python
lookup = define_tool(name="lookup", callers=["tools"], execute=word_count_execute)  # never offered to the model
registry.install(define_extension("lookup", tools=[lookup]))

# The model is offered only long_text; long_text may still call word_count.
await root.configure({"modelTools": [long_text]}, context)
# The model is offered everything but word_count, which tools may still call.
await root.configure({"modelTools": {"remove": [word_count]}}, context)

agent = await root.agent(context)
print([tool.name for tool in agent["tools"]], [tool.name for tool in agent["callable"]])
```

`conversation.agent()` resolves both lists: `tools`, offered to the model, and `callable`, which nested calls resolve among. `tools` still decides what is enabled at all; `modelTools` cannot add to it.

A `deferred=True` tool is not offered to the model until it is loaded, though other tools can call it any time. A tool result loads deferred tools with `"control": {"addTools": [...]}`, which records them in `modelTools`; loaded tools stay offered on that branch, also after a compaction. Upstream's `tool_search` tool, which lets the model search and load deferred tools, is not ported yet.

### Tasks a restart abandons

A task that its creator awaits only in memory, and that the creator never resumes after a restart, can be created with `"abandonOnRestart": True` in its task options. When a later Harness starts scheduling and finds it still live, it is aborted with `abortReason` `"restart"`, together with everything it owns, before any of it runs again. Nested calls of tools that are not replay-safe get this automatically, and so do the child tasks such a tool creates with `api.create_task()`. A task abandoned this way whose definition is missing waits until its extension is installed, so its abort handler still cleans up. Its caller, and the conversation, wait with it; `harness.inspect()` shows it as blocked. Aborting it then orphans it, as any abort does.

## System Prompt

The system prompt is built from the selected extensions' sections, rendered in order before each request. A section's `render(input, context)` sees the resolved agent, the environment built for the request, and committed documents (`input["agent"]`, `input["env"]`, `input["read"]`), and may be async:

```python
cwd = section("cwd", lambda input, context: input["agent"].get("cwd"))  # rendered as <cwd>\n...\n</cwd>; None omits it
registry.install(define_extension("cwd", sections=[cwd]))
```

A conversation's `instructions` render last, as the section `instructions`. Sections and tool changes are stored as positional system entries in the transcript. Only what changed is sent again, which keeps provider prompt caches warm. A section that returns something different every time, such as the current time, defeats that.

## Per-Conversation Agent

Each conversation stores what it runs with in its `pi.agent` document. `configure()` changes it in one commit; unset fields follow the host:

```python
await root.configure(
    {
        "model": {"provider": "openai", "modelId": "gpt-6-sol"},
        "thinkingLevel": "high",
        "extensions": {"remove": [Coding]},  # edits the host default; a list selects exactly these, in order
        "tools": [lines, count],  # a list enables exactly these; {"remove": [...]} drops some
        "modelTools": [lines],  # of those, offer the model only lines; tools may still call count
        "instructions": "Only read; never edit files.",
        "cwd": "/work/repo",
    },
    context,
)
await root.configure({"tools": None}, context)  # None clears a field back to the host default
agent = await root.agent(context)  # resolved: model, extensions, tools, sections, cwd
```

Extensions and tools are passed as objects and stored by name, so a stored name outlives its code: after an extension is uninstalled, conversations that select it just stop getting it until it is installed again. `create_conversation()` and `fork()` take the same change as `"agent"` in their options, and `root()` as `agent=`. A task-owned conversation, such as a subagent's, starts as a copy of its owner's conversation's agent. A fork starts with the agent its parent had at the fork entry. The model, prompt, and offered tools of a request are fixed when it is prepared; a change applies from the next request. Tool calls and hooks use the agent as their task phase resolves it, and the environment is built from the current `cwd` at each use, so a `cwd` or extension change can reach calls the model already made.

Inside a commit, `configure(tx, conversation_id, change)` from `pi_durable` applies the same change.

## Settings

Run policy shared by every conversation is passed as `settings`. It is read at every use and never stored. Pass a dict, or a function returning one to keep settings live, for example backed by a settings file:

```python
user_settings = {"followUpMode": "one-at-a-time"}


def settings():
    return {
        "extensions": [Coding],  # default selection; absent: every installed extension
        "stream": {"timeoutMs": 120_000},
        "retry": {"maxRetries": 3},
        "compaction": {"reserveTokens": 16384},
        "progress": {"partialIntervalMs": 100, "outputIntervalMs": 100},
        "toolExecution": "parallel",
        "contextRetentionMs": 600_000,  # idle conversations keep their context in memory this long
        "documentRetentionMs": 600_000,  # loaded documents nothing observes stay in memory this long after their last use
        "followUpMode": user_settings["followUpMode"],
    }


harness = await Harness.open(MemoryStorage(), {"models": models, "registry": registry, "settings": settings}, context)
```

A dict you keep and change in place is live too, since it is read at every use. The settings function must be synchronous and fast.

## Environment

`env` builds the execution environment for each tool call, section rendering, and `runtime.env()`. It is called as `env(target, context)` with the conversation's ID, its agent `cwd`, and committed reads (`target["conversationId"]`, `target.get("cwd")`, `target["read"]`), so one function serves a directory per conversation or a container per conversation:

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

A raise from `env` becomes the call's error result. Without an environment, `api.env` is `None`. A fresh environment object per call is fine. A custom `ExecutionEnv` sets `id` so that equal ids see the same files at the same paths, for example one id per container; `LocalExecutionEnv`'s id is `"local"`.

File and shell operations return a result dict instead of raising expected failures: `{"ok": True, "value": ...}` or `{"ok": False, "error": FileError(...)}` (`get_or_throw()` and `get_or_none()` in `pi_durable.env` unwrap one). Hosts can use the environment directly too, for example to show a project's files. `open_binary_reader()` reads byte ranges of one opened file, `open_dir_reader()` pages a directory, and `exec()` with an argument list runs a program without a shell, reporting which stream each output chunk came from:

```python
env = LocalExecutionEnv(cwd=os.getcwd())
status = {"stdout": "", "stderr": ""}


def on_output(text, context, info):
    status[info["stream"]] += text


result = await env.exec(["ls", "-a"], {"onOutput": on_output}, context)
print(result["ok"] and result["value"]["exitCode"])  # 0
```

Cancel the context to stop one call; `cleanup()` is for shutting the environment down. A custom environment can check itself with `register_env_conformance()` from `pi_durable.testing`, like storage below.

Environments also have `list_files()`: the files and directories below a directory at any depth, as `fd` lists them, skipping what git ignores and stopping at bounds with `truncated`. An environment that cannot list returns an error with code `not_supported`, as `LocalExecutionEnv` does; callers then fall back to their own walk.

## Reload

Installing an extension with an installed name replaces it in place, in one step:

```python
registry.install(define_extension("coding", sections=[section("preamble", lambda input, context: "You are a careful coding assistant.", tag=False)]))
```

`registry.uninstall(extension)` removes the installed extension with that name, whichever object it is.

Work that already started keeps the code it took: a running tool call finishes under its old implementation, and each task phase resolves hooks and the agent once, from the registry at the phase's start. The next phase, request, or call uses the new code. After a restart, install the same extensions again; pending tasks of an extension's `tasks` resume once it is installed.

## Watching a Conversation

Everything a UI needs is committed state. `view_state()` returns the conversation's structural view as a read-only Chord state, updated after every commit that touches it. A subscriber is called with the value, the commit's context, and a delivery (`delivery.kind` is `"hydrate"`, then `"update"`), and must accept all three:

```python
view = await root.view_state(context)


def render(value, context, delivery):
    # value["entries"]: the active transcript
    # value["docs"]["pi.live"]: the running generation (streamed partial, retry, deferred) and tool calls (output, details)
    # value["docs"]["pi.inbox"], ["pi.usage"], ["pi.agent"], ["pi.provider"]
    print(delivery.kind, len(value["entries"]), "entries")


unsubscribe = view.subscribe(render)
# later: view.dispose()
```

`watch()` delivers the same view with the exact Chord operations of each commit, one callback at a time:

```python
watch = await root.watch(context)
print(sorted(watch.value))  # the state at attachment: ['conversation', 'docs', 'entries']


async def send(value, ops, context):
    print(ops)  # for example, send them to a remote client that applies them


watch.start(send)
# later: await watch.stop()
```

A slow watch keeps at most 100 undelivered frames. After that, the pending frames are replaced by one frame holding the whole newest view. A client that joins late or reconnects starts from the current view; nothing is replayed.

Partial answers and tool output are committed at most every 100 ms by default, so a crash loses at most that window. `settings["progress"]` changes the intervals; a host whose storage is remote can commit less often, for example `{"partialIntervalMs": 500, "outputIntervalMs": 500}`.

## Busy Conversations

A conversation is busy while a run is working on an input. Submitting to a busy conversation queues the submission in the conversation's inbox, `docs["pi.inbox"]` in the view:

```python
from pi_durable import ConversationBusy

await root.submit({"type": "input", "content": "Also run the tests"}, context)  # follow-up (default)
await root.submit({"type": "input", "content": "Use pnpm, not npm", "whenBusy": "steer"}, context)
await root.submit({"type": "write", "entry": {"kind": "app.note", "data": "user opened a file"}}, context)
try:
    await root.submit({"type": "input", "content": "Only if idle", "whenBusy": "reject"}, context)
except ConversationBusy:
    print("busy")
```

- **Steers** are placed after the current tool round and join the running work.
- **Follow-ups** are placed when the run answers, and start the next run.
- **Writes** append an entry without asking the model anything. A write of only `kind` and `data` changes nothing the model sees, so it is appended at once, even mid-run, and is never queued: a chat message shows in the transcript as soon as it is sent. It can therefore appear before items submitted earlier that are still queued, except behind a queued reset, where it waits so the reset does not hide it. A write with `model`, `head`, or `edits` waits for the next boundary like a steer.
- `await submission.abort(context)` withdraws a queued submission.
- The [settings](#settings) `"steeringMode": "all"` and `"followUpMode": "all"` place every queued item at once instead of one per turn.

If a run fails, queued items stay in the inbox until the next input or queued write places them, oldest first.

An input is placed as a `pi.user` entry. To show it as something else, give it an `entry` with your kind and data; the model still sees `content` as the user message:

```python
await root.submit(
    {
        "type": "input",
        "content": "ann: can you check the build?",
        "entry": {"kind": "chat.mention", "data": {"author": "ann", "text": "can you check the build?"}},
    },
    context,
)
# placed as {"kind": "chat.mention", "data": {"author": "ann", ...}, "model": [UserMessage("ann: can you check the build?")]}
```

It behaves like any other input: it starts a run, steers, follows up, or rejects, and is answered. While queued, its inbox item carries the entry. The kind must not start with `pi.`.

## Reset and Handoff

`reset()` starts a new context. The model no longer sees older entries, but they stay in storage:

```python
await root.reset(None, context)  # start from nothing
await root.reset("We were fixing the flaky login test. Continue.", context)  # start from a handoff note
```

While busy, the reset is queued like a write. When it is placed during a tool round, the current run ends. A tool can request the same with `"control": {"handoff": "..."}`.

## Compaction

Compaction shrinks what the model sees: it summarizes older entries and appends a `pi.compaction` entry that holds the summary and heads the first entry it keeps. Older entries stay in storage.

```python
task_id = await root.compact("Keep the failing test names", context)  # manual, with optional instructions
outcome = (await harness.wait_for_task(task_id, context))["state"]["outcome"]
if outcome["status"] == "completed" and outcome["result"].get("submissionId") is not None:
    placed = await (await harness.submission(outcome["result"]["submissionId"], context)).wait(context)
    print(placed["status"])  # "done", or "unanswered" with reason "stale"
```

The conversation keeps working while the summary is made. The summary is placed at once when the conversation is idle, otherwise at the next turn boundary. `abort()` cancels a manual compaction. A compaction that finds nothing to summarize completes with an empty result.

Generation also compacts on its own, controlled by the [settings](#settings):

```python
settings = {
    "compaction": {
        "enabled": True,  # automatic compaction; manual compact() always works
        "reserveTokens": 16384,  # above contextWindow - reserveTokens, the next request waits for a compaction
        "keepRecentTokens": 20000,  # roughly how much recent context stays verbatim
        "backgroundTokens": 32768,  # this far below that, a compaction starts in the background; 0 disables it
        "modelOverrides": {
            # exact "provider/modelId"; each field set here replaces the one above for that model
            "local/small-32k": {"reserveTokens": 4096, "keepRecentTokens": 8000, "backgroundTokens": 4096},
        },
    },
}
```

The fixed token counts rarely suit a 32k and a 1M context window alike; `modelOverrides` lets one harness serve both. The conversation's current model picks the entry, at the threshold check, after a context overflow, and when a compaction selects what to summarize.

When a provider rejects a request because the context is too long, generation compacts and retries once. A summary that would cut before the start of the current context settles as `stale` when it is placed, so when several are in flight, the furthest cut stays in effect. Summarization spend counts in `pi.usage`. A `before_compact` hook on `CompactionTask` can decline (`{"decline": True}`) or supply its own summary (`{"summary": "..."}`).

When a compaction that a generation waits for at the blocking threshold fails, the generation fails too, without sending a request: its inputs settle `unanswered` with `model_error` and detail `Compaction failed: <error>`. One that finds nothing to compact or is declined lets the request go ahead.

Running compactions are listed in `docs["pi.live"]["compactions"]` with their reason, attempt, and retry backoff. The newest failure stays in `docs["pi.live"]["compactionFailure"]` (task, reason, error) until a compaction produces a summary. The agent events add `compaction_start` and `compaction_end` (with `error` when the compaction failed), and `compactions` and `compactionFailure` fields in the snapshot.

## Agent Events (Experimental)

For consumers that want coding-agent style events (`message_start`, `message_update`, `tool_execution_start`, ...) instead of structural state:

```python
from pi_durable import watch_events

stream = await watch_events(harness, root.id, context)
print(sorted(stream.snapshot))  # entries, tools, nestedTools, compactions, inbox, agent, usage, ...


def on_events(events, context):
    for event in events:
        print(event["type"])


stream.start(on_events)
# later: await stream.stop()
```

Events are dicts with upstream's keys, derived from commits, one batch per commit, and apply on top of the snapshot. Messages inside events are `pi_ai` dataclasses; use `pi_ai.serialization.to_json_value()` before `json.dumps()`. Message and tool updates carry deltas: text and thinking appends, appended tool-call argument text, and output trims and appends. When a consumer falls more than 100 batches behind, it receives a fresh `snapshot` event instead.

## Hooks

Hooks let extensions observe or adjust the built-in tasks, in the conversations that select them. A hook table is keyed by the snake_case hook name, and every handler takes upstream's arguments plus `api` and `context`:

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

- **Generation:** `before_request(request, api, context)` (replace the messages of one request), `after_response(message, api, context)`, `on_yield(answer, api, context)` (continue the run with another user message), and `after_tools(assistant_entry_id, result_entry_ids, api, context)` (runs once a round's tools are done).
- **Tools:** `before_tool(call, api, context)` (block, or rewrite `arguments`) and `after_tool(call, result, api, context)` (replace the result).
- **Compaction:** `before_compact(compaction, api, context)` (decline, or supply a summary).

To limit a hook to some conversations, select its extension only there, for example with `configure({"extensions": {"add": [Guard]}})`.

## More Conversations and Forks

```python
other = await harness.create_conversation({"ownership": {"kind": "ownerless"}}, context)
fork = await root.fork(entry_id, {"ownership": {"kind": "ownerless"}}, context)
```

A fork sees its parent's entries up to `entry_id` and continues independently. It keeps the parent's agent as of that entry but receives a fresh provider session identity. Both take `"agent"` and `"init"` in their options, applied in the creating commit; `root()` takes them as `agent=` and `init=`.

## Abort and Subagents

`await root.abort(context)` stops a conversation: queued inputs are withdrawn (queued writes stay), every task of its current work is aborted, and the call returns once the conversation is idle.

A conversation can be **owned** by a task. A subagent tool creates its child inside `api.commit()` with `{"ownership": {"kind": "task", "taskId": api.task_id}}`, then drives it through `api.conversation(id, context)`:

```python
from pi_durable import configure


async def subagent_execute(args, api, context):
    async def find_or_create(tx):
        # The ownership index remembers the child, so a rerun reuses it.
        existing = (await tx.scan_conversations({"ownerTaskId": api.task_id}, 1))["items"]
        if existing:
            return existing[0]["id"]
        # Starts as a copy of this conversation's agent: model, extensions, tools, cwd.
        created = await tx.create_conversation({"ownership": {"kind": "task", "taskId": api.task_id}})
        # A cheaper model, and no subagents of its own.
        await configure(tx, created["id"], {"model": {"provider": "openai", "modelId": "gpt-6-luna"}, "extensions": {"remove": [Subagent]}})
        return created["id"]

    child = await api.commit(find_or_create, context)
    await api.details({"conversationId": child}, context)  # lets a UI attach to the child
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
            replay="safe",  # a rerun after a crash finds the same child and submission
            execute=subagent_execute,
        )
    ],
)
registry.install(Subagent)
```

Owned work belongs to its owner:

- Aborting the call aborts the child. So does the call failing: `execute()` raising, or a crash that interrupts a call that is not replay-safe.
- The parent is idle only once the child is.
- A task created with `"background": True` is a boundary: work it owns survives the parent's abort and does not keep the parent busy. `root.abort(context, background=True)` aborts it too.

## Child Tasks

A task can own child tasks, created with `{"ownership": {"kind": "task", "taskId": ...}}`, and wait for them by committing a `waiting` state:

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
        # Resume in `decide` once every payment is done; the first failure aborts the rest.
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

- **Waiting:** the task runs no code while it waits. With `allSettled` it resumes once every task in `on` is done; with `failFast` the first failed child also aborts the others. `on` may name other tasks too, with `allSettled`.
- **Finishing:** a task that finishes while work it owns is still running is `completing`: its outcome is decided, but it becomes terminal, and `wait_for_task()` returns, only once that work is done. A failed or aborted outcome aborts that work first.
- **Aborting:** abort runs bottom-up. Aborting a task aborts the work it owns first, and its own abort handler starts only once that work is done, so each task undoes its own effects.

## Task Graph

`harness.task_graph(context)` shows every live task of the Session as one Chord state, for a task panel or debugging. Each node has its owner edge (`owner` task, or none for a task its conversation owns), its status, whether it is `background` or abort-marked, why it is `blocked` when it cannot run, and the conversations it owns. `harness.watch_task_graph(context)` delivers the same value as a watch, like a conversation's `watch()`.

```python
graph = await harness.task_graph(context)


def show(value, context, delivery):
    for node in value["tasks"].values():
        state = node["state"]
        status = f"waiting on {state['on']}" if state["status"] == "waiting" else state["status"]
        print(node["id"], node["kind"], status, node.get("owner", f"conversation {node['conversationId']}"))


graph.subscribe(show)
# later: graph.dispose()
```

A task appears with the commit that creates it and leaves with the commit that makes it terminal. Statuses are the committed ones: `pending`, `running`, `waiting` (with `on` and `policy`), `sleeping` (with `until`), and `completing` (with the held outcome's status); each node also carries its checkpoint `phase`. After a restart, tasks that were `running` show as `pending` until they run again. The graph lists live tasks only: once a subagent's owner task is terminal, a later task in its conversation is a top-level node, and the conversation's `ConversationRecord["owner"]` (also in its view's `conversation`) links it to its parent. `harness.inspect(context)` reports the same live tasks with what the scheduler would do with each.

## Your Own State

Documents are typed JSON objects committed together with entries. Define one, and edit it in a commit; `tx.doc()` returns a draft you change in place:

```python
from typing import TypedDict

from pi_durable import ConversationDocToken, define_doc


class Todos(TypedDict):
    items: list[str]


TodosDoc: ConversationDocToken[Todos] = define_doc(
    kind="app.todos",
    version=1,
    scope="conversation",
    history="latest",  # or "rewindable" to read old values with snapshot_as_of()
    fork="initial",  # what a fork starts with: "initial", "current", or "asOf"
    initial=lambda: {"items": []},
)


async def add_todo(tx):
    (await tx.doc(TodosDoc, root.id))["items"].append("write docs")


await root.commit(add_todo, context)
print(await harness.snapshot(TodosDoc, root.id, context))  # {'items': ['write docs']}
```

`harness.watch_doc()` and `harness.document_state()` observe one document like the view above. The `conversationCreated(tx, conversation)` option runs in every commit that creates or forks a conversation, including a tool's raw `tx.create_conversation()`, so every conversation gets your documents; `"init"` in `create_conversation()` and `fork()` options, and `init=` in `root()`, writes per-call data in the same commit. An extension's tools, sections, and hooks read their own documents through `api` (`api.snapshot(...)`) or `input["read"]`, and treat an absent one as its default. `define_doc_family()` defines keyed documents, one per key.

## Usage and Cost

Each conversation keeps token and cost totals in `docs["pi.usage"]`: per `provider/model` for model responses, and per tool name for tool results that report usage. Failed and aborted attempts count too. For the whole Session:

```python
usage = await harness.usage(context)  # {"models": {"openai/gpt-6-sol": {...}}, "tools": {...}}
print(usage["models"]["openai/gpt-6-sol"]["totalTokens"])
```

Totals are `pi_ai` `Usage` as camelCase JSON (`input`, `output`, `cacheRead`, `cacheWrite`, `totalTokens`, `cost`).

## Storage

| Backend | Import | Notes |
|---|---|---|
| Memory | `MemoryStorage` from `pi_durable` | Nothing is persisted. |
| SQLite | `await open_sqlite_storage(path)` from `pi_durable.storage.sqlite.stdlib` | One database file on the standard `sqlite3` module, which runs on its own worker thread. WAL mode with `synchronous = NORMAL`: commits survive process crashes; the newest may be lost on power or host failure. |
| JSONL | `await open_local_jsonl_storage(directory, context)` from `pi_durable.storage.jsonl.local` | Append-only files in one directory. Pass `fsync=True` to flush before each commit marker. Keeps the whole history in memory too, so memory grows with it; use SQLite for long-running agents. |

Both file formats are the TypeScript package's: either implementation can open what the other wrote.

SQLite storage can namespace its tables and indexes when sharing a database: pass `prefix="pi_"` to `SqliteStorage.open(database, prefix=...)` or `open_sqlite_storage(path, prefix=...)`. Storage renders the prefixed names into its SQL, so adapters need no support. A prefix must be a SQLite identifier and cannot begin with `sqlite_`. Changing the prefix does not migrate existing data. To share the database with host tables, open the database facade yourself and use it for both Pi and all host queries:

```python
from pi_durable.storage.sqlite import SqliteStorage
from pi_durable.storage.sqlite.stdlib import open_sqlite_database

database = await open_sqlite_database("./app.sqlite")
pi_storage = await SqliteStorage.open(database, prefix="pi_")
await database.exec("CREATE TABLE IF NOT EXISTS host_notes (text TEXT)")
await database.run("INSERT INTO host_notes (text) VALUES (?)", "a note")
```

Closing the Harness closes its storage, and with it the database.

One process owns a storage at a time; there is no cross-process locking. The portable cores (`pi_durable.storage.sqlite.SqliteStorage`, `pi_durable.storage.jsonl.JsonlStorage`) run over any asynchronous `SqliteDatabase` facade (`pi_chord.sqlite`) or any `FileSystem` (`pi_durable.env`): `await JsonlStorage.open(directory, fs, context)`.

SQLite facades implement async `exec`, `run`, `get`, `all`, `transaction`, and `close`. `run`, `get`, and `all` take SQL text plus positional bindings; rows are dicts keyed by column name. A transaction callback receives a transaction handle; all work in the transaction must use it, and the handle expires when the callback returns. Facades queue unrelated operations and other transactions until the transaction finishes, so calling `database` itself inside the callback never returns:

```python
async def create_example(transaction):
    await transaction.exec("CREATE TABLE example (value TEXT)")
    await transaction.run("INSERT INTO example (value) VALUES (?)", "stored atomically")


await database.transaction(create_example)
print(await database.all("SELECT value FROM example"))  # [{'value': 'stored atomically'}]
```

Custom backends can run the shared conformance suite under pytest. `register_storage_conformance()` defines one parametrized test in the module that calls it:

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

A custom `Storage` implements the protocol in `pi_durable.types` (`commit`, `mint_id`, the record reads and scans, `close`); `packages/durable/DESIGN.md` describes it.

## Errors

Three rules decide what an error does:

1. **A storage error ends the Harness.** The call that hit it raises the error. Everything after it, and every pending wait, raises `SessionFailed` (its `cause` and `__cause__` are that error); watches and event streams end with `session_failed`; running tasks get their abort signal; `onReport` gets the error once. The Harness then closes itself. Nothing is retried; a custom storage retries its own transient failures. Reopen to recover: commits are atomic, so the store is consistent, and resubmitting with the same `requestId` finds a submission that was already admitted.
2. **An invalid read fails only that call:** a storage read raises `StorageRequestError` for a bad cursor, an unknown conversation, or history a document does not keep. A read you cancelled fails only itself too. The exemption covers reads only: any error from `commit()` or `mint_id()`, `StorageRequestError` included, ends the Harness as in rule 1.
3. **Your code fails only its own unit:** a raising task phase faults that task, a raising tool gets an error result, a raising hook is reported and skipped (`before_tool` blocks the call), and a raising commit callback rolls back that commit. Listeners that raise, or return a failed awaitable, are reported and the others still run. The exception is a host `RegistryReader` whose `snapshot()` raises: the Harness cannot schedule without it, so it fails the Harness like a storage error.

`harness.closed` is a future that settles once the Harness has closed, by `close()` or after a storage error, so a host restarts on failure like this:

```python
async def supervise(harness):
    end = await harness.closed
    if end["reason"] == "failed":
        await reopen(end["error"])  # your code
```

A failure closes the Harness by itself; `closed` settles once that close is done, even if the storage could not close cleanly. Closing waits for running task code to return, so code that ignores its abort signal delays `closed`.

## Differences from the TypeScript Package

- **Names.** Functions, methods, attributes, and keyword arguments are snake_case (`createConversation` is `create_conversation`, `structuredOutputSchema=` is `structured_output_schema=`); dict keys keep upstream's camelCase, including option keys such as `"onReport"` and `"setTimer"`. Hook tables are keyed by snake_case hook names.
- **Definitions** (`define_tool`, `define_task`, `define_doc`, `section`, `hook`) take upstream's fields as keyword arguments and return dataclasses or token objects, not plain objects. Tool schemas are JSON Schema dicts (`pi_ai.Type`) or Pydantic models, not TypeBox; `args` is not typed from the schema.
- **Optional trailing options** become keyword-only parameters: `root(context, agent=..., init=...)`, `abort(context, background=True)`, `execute_tool(name, args, context, key=..., progress=...)`, `context(context, at=...)`, `section(key, render, tag=False)`.
- **Futures.** Promise-returning methods are `async def`, or plain functions that return a started future where upstream's work starts at the call (see [Two Python Rules](#two-python-rules)). `closed` properties are futures.
- **Callbacks** get upstream's arguments positionally: a state subscriber is `listener(value, context, delivery)`, a watch listener `listener(value, ops, context)`, an event listener `listener(events, context)`, a runtime commit change `change(tx, current)`, a section `render(input, context)`. State subscribers and watch listeners may declare fewer parameters, as in JavaScript (`lambda value: ...` gets only the value); the other callbacks must accept all of them. Each may be sync or async.
- **Timers** run on the asyncio loop and never keep a process alive by themselves; there is no maximum delay, and nothing is declined as in Workers.
- **Errors** keep upstream's classes and messages. A bare JavaScript `Error` is `RuntimeError`, `RangeError` is `ValueError`.
- **Storage and environment modules**: `openNodeSqliteStorage` is `open_sqlite_storage` (on `sqlite3`), `openNodeJsonlStorage` is `open_local_jsonl_storage`, `NodeExecutionEnv` is `LocalExecutionEnv` (id `"local"`), `Storage.entry(conversationId, id)` is `visible_entry()`, and `FileSystem.list_files()` is a required method that may return `not_supported`. The conformance suites register with pytest.
- **Binary data** is `bytes`; IDs are plain `int`s (`TaskId`, `EntryId`, ... are type-checker brands).
- **Not ported in phase 1**: the coding tools, tool search, images, code mode, and the Cloudflare adapters (see [What Phase 1 Includes](#what-phase-1-includes)).

## Design Notes

[`DESIGN.md`](DESIGN.md) is the porting contract: how each TypeScript type maps to Python, the persisted JSON rules, the Storage protocol, and the Session and Harness internals. Upstream's [specification](https://github.com/earendil-works/pi/blob/main/packages/durable/docs/spec.md) is the normative description of the behavior.

## License

MIT
