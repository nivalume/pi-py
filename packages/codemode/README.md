# pi_codemode

Runs model-written Python in [Monty](https://github.com/pydantic/monty), Pydantic's sandbox for AI-written Python, where the only capability is calling injected tools. Nested tool calls never enter the LLM context; only the script's output and return value do.

This is the Python port of [`@earendil-works/pi-codemode`](https://github.com/earendil-works/pi/tree/main/packages/codemode). The one intended difference: scripts are Python instead of JavaScript, and Monty replaces QuickJS. The API (in snake_case), the script facilities, the result format, the error kinds, the limits, deterministic runs and the declarations for the model follow upstream.

Scripts use `tools`, `ALL_TOOLS`, `text`, `image`, `print`, `exit`, `store`, and `load`, and may start with a `# @options:` line. The package has no pi dependencies beyond pi-ai's `AbortSignal` and can be used on its own to expose any functions (remote APIs, MCP servers, application services) to model-written scripts.

| Class | Import from | Where scripts run |
| --- | --- | --- |
| `CodemodeSandbox` | `pi_codemode` | a session of a Monty worker pool (one `monty` subprocess per execution) |
| `CodemodeSandboxBase` | `pi_codemode` or `pi_codemode.portable` | wherever its launcher puts them; the host side every sandbox shares |

Install the sandbox with the `codemode` extra:

```bash
pip install 'pi-ai-python[codemode]'
```

Without it, everything but `CodemodeSandbox` (types, source parsing, declarations, identifiers, `CodemodeSandboxBase`) still imports; `CodemodeSandbox` raises an `ImportError` that names the extra.

## Usage

```python
from pathlib import Path

from pi_codemode import CodemodeSandbox, CodemodeTool


async def read(args, context):
    return Path(args["path"]).read_text()


sandbox = CodemodeSandbox(
    timeout_ms=60_000,
    tools=[CodemodeTool("read", read, description="Read a file")],
)

result = await sandbox.execute("""
import tomllib
source = await tools.read(path="pyproject.toml")
text("bytes " + str(len(source)))
return tomllib.loads(source)["project"]["name"]
""")

print(result.output)  # [CodemodeTextOutput(text='bytes 24', console=False)]
if result.ok:
    print(result.value)  # demo
else:
    print(result.error.kind, result.error.message)

await sandbox.close()
```

`code` is the body of an async function: top-level `await` and `return` work. Inside the script:

- `tools.<name>(...)` returns an awaitable. Pass keyword arguments (`tools.read(path="x")`) or one positional value (`tools.read({"path": "x"})`); arguments and results make a JSON round trip. A tool that raises makes the `await` raise an `Exception` with the same message. Tool names are also exposed as identifiers: characters that are not valid in a Python identifier become `_` and keywords get a trailing `_` (`to_codemode_identifier`), so `my-tool` is `tools.my_tool` as well as `tools["my-tool"]`.
- Calls start when they are made, like JavaScript promises, so `asyncio.gather(tools.a(), tools.b())` runs both concurrently on the host, and a call the script never awaits is cancelled when it ends.
- `ALL_TOOLS` lists `{"name": ..., "description": ...}` for every tool, with `name` as the identifier.
- `text(value)` appends a text item to `result.output`; values other than strings are JSON-encoded. `print(...)` appends text items with `console=True`, one per call, formatted as Python's `print` formats them.
- `image(url_or_item)` appends an image item. It accepts a base64 `data:` URL, `{"image_url": ...}`, or an image block (`{"type": "image", "data": ..., "mimeType": ...}`, as MCP tools produce). Remote URLs are rejected.
- `exit()` ends the script successfully right away, keeping its output and store writes; no `except` can stop it.
- `globals` passed to the sandbox are called as top-level functions, for example a host helper `attach(ref)`. They behave like tools but are not recorded in `result.calls`. A name like `models.classify` puts the function on a `models` namespace object. With `spread=True`, `execute` receives the positional arguments as a list instead of the first one (keyword arguments, if any, as one trailing dict), and `signature` replaces the declaration generated from the schemas.
- `store(key, value)` and `load(key)` read and write JSON values synchronously. See [Store](#store).
- Monty's standard library: `json`, `re`, `math`, `datetime`, `random`, `asyncio.gather`/`sleep` and the like. Nothing else: no filesystem (`open()` raises `PermissionError`), environment (`os.getenv()` raises `RuntimeError`), network, subprocesses or host modules (`ModuleNotFoundError`). `eval` and `exec` work but only run more code inside the same sandbox.

`timeout_ms=math.inf` disables the deadline; the script then runs until it settles or `signal` aborts it. A script that waits on something nothing can settle (no tool call or sleep pending) fails right away instead of hanging.

`memory_limit_bytes` caps the sandbox's heap. Allocations beyond it fail the script with `MemoryError`.

`execute()` takes `signal` (a pi-ai `AbortSignal`) or `context` (a pi-chord `Context`, whose abort signal is used). Aborts, timeouts, and `close()` end the execution at once: a session that waits on tools is discarded, and a worker that is computing is killed. Results never wait for the worker to go away, and the pool replaces it.

## Workers and the pool

Each execution checks a session out of an `AsyncMonty` pool: a dedicated `monty` worker subprocess, so a script that computes never blocks the host's event loop. Executions run in parallel up to the pool's `max_processes` (default: the CPU count); later ones wait for a free worker. A worker that crashes fails only its own execution with `kind="sandbox"`; the pool replaces it and the host is unharmed.

By default a sandbox starts its own pool on first use and closes it in `close()` (or at the end of `async with`). To share one pool between sandboxes, pass it in; the sandbox then never closes it:

```python
import asyncio

from pydantic_monty import AsyncMonty

async with AsyncMonty(max_processes=4) as pool:
    async with CodemodeSandbox(pool=pool) as first, CodemodeSandbox(pool=pool) as second:
        results = await asyncio.gather(first.execute("return 1"), second.execute("return 2"))
        print([result.value for result in results])  # [1, 2]
```

`AsyncMontyWebsocket` pools work the same way. `binary_path` points the sandbox's own pool at another `monty` binary; a pool that cannot start fails executions with `Failed to start Monty: ...`.

## Store

`store`/`load` let scripts keep values across executions. The sandbox does not persist anything itself: pass the current values as `store`, and a successful result reports what the script changed as `result.store_writes` (`set` and `delete`). Failed executions report no writes.

```python
saved = {}
async with CodemodeSandbox() as sandbox:
    for _ in range(2):
        result = await sandbox.execute('store("runs", (load("runs") or 0) + 1)', store=saved)
        if result.ok:
            for key in result.store_writes.delete:
                saved.pop(key, None)
            saved.update(result.store_writes.set)
print(saved)  # {'runs': 2}
```

`load` returns a copy, so mutating it does not change the store; a missing key loads as `None`. Storing `None` deletes the key. A value may be at most `MAX_STORE_VALUE_CHARS` (256 Ki) characters of JSON and all values together at most `MAX_STORE_TOTAL_CHARS` (1 Mi); larger writes raise a `ValueError` inside the script.

## Source format

`parse_codemode_source()` accepts a script whose first line may be an options line:

```python
from pi_codemode import parse_codemode_source

source = parse_codemode_source('# @options: {"max_output_tokens": 2000, "timeout_ms": 30000}\nreturn 1')
print(source.options)  # CodemodeSourceOptions(max_output_tokens=2000, timeout_ms=30000)
print(repr(source.code))  # '\nreturn 1'
```

Supported fields are `max_output_tokens`, a token budget for the output, and `timeout_ms`, a hard deadline. The sandbox does not act on them; the caller decides. The options line is replaced by an empty line, so line numbers in tracebacks still match the input. Empty input, invalid JSON, unknown fields, an options line without code, or an options line on any line but the first raise `CodemodeSourceError`. `CODEMODE_SOURCE_GRAMMAR` is a Lark grammar for providers that support grammar-constrained tool input. Both are also available from the dependency-free `pi_codemode.source` module.

The script is the body of an async function, written in Python. It runs as its own Monty snippet after the prelude that defines the helpers, so tracebacks name `codemode.py` with the script's own line numbers. `return value` works anywhere at the top level and ends the script with that value; without one the value is `None` (a trailing expression is not a value, as in a function body). How upstream's JavaScript facilities map:

| Upstream (JavaScript) | Here (Python) |
| --- | --- |
| `// @options: {...}` first line | `# @options: {...}` first line |
| async function body: top-level `await`, `return` | the same; a trailing expression is not returned |
| `await tools.read({ path })` | `await tools.read(path=...)` or `await tools.read({"path": ...})` |
| `tools["my-tool"]`, `"x" in tools` | the same |
| `toCodemodeIdentifier`: invalid characters become `_` | `to_codemode_identifier`: ASCII identifier characters, and keywords get a trailing `_` (`class` is `tools.class_`) |
| `Promise.all([...])` | `asyncio.gather(...)`; calls start when made in both |
| `ALL_TOOLS` (frozen array) | `ALL_TOOLS` (tuple of dicts) |
| `text(value)`: JSON for non-strings, `undefined` prints `undefined` | `text(value)`: JSON for non-strings, `None` prints `null` |
| `console.log/info/warn/error/debug(...)` | `print(*values, sep=" ", end="\n")`, one console item per call |
| `image(urlOrItem)` | `image(url_or_item)` with dicts for the object forms |
| `exit()` | `exit()` |
| `store(key, undefined)` deletes; `load` of a missing key is `undefined` | `store(key, None)` deletes; `load` of a missing key is `None` |
| a failed tool rejects with `Error(message)` | the `await` raises `Exception(message)` |
| unknown tool: `TypeError` naming close matches | `AttributeError` naming close matches |
| `RangeError` (store size, output limit) | `ValueError` |
| a value JSON cannot hold: `TypeError` | `TypeError` |
| `models.classify` on a frozen namespace object | `models.classify` on a namespace object (`in` and `models["classify"]` work) |
| spread globals get the argument array | positional arguments as a list, keyword arguments as one trailing dict |
| stack `at f (codemode.js:2:31)` | CPython-style traceback, `File "codemode.py", line 2, in f` |
| `eval`/`Function` stay in the VM; no timers | `eval`/`exec` stay in Monty; `asyncio.sleep`/`time.sleep` wait on the host (virtual in `start()`) |

## Declarations for the model

Tools and globals can carry `description`, `input_schema`, and `output_schema` (JSON Schema). `render_declarations()` turns them into Python declarations for a model-facing tool description:

```python
from pi_codemode import CodemodeTool, render_declarations

read_tool = CodemodeTool(
    "read",
    read,
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace"},
            "limit": {"type": "integer", "minimum": 1, "default": 2000},
        },
        "required": ["path"],
    },
    output_schema={"type": "string"},
)
print(render_declarations(tools=[read_tool]))
```

prints

```text
class tools:
    async def read(
        self,
        *,
        # @minimum 1 @default 2000
        limit: int = ...,
        # Path relative to the workspace
        path: str,
    ) -> str:
        """Read a file"""
```

An object input whose property names are identifiers becomes keyword-only parameters; any other input is one positional `args` parameter. Nested objects render as inline `TypedDict[{"key": T, "other": NotRequired[T]}]` types (PEP 764 notation), arrays as `list[T]`, maps as `dict[str, T]`, constants and enums as `Literal[...]`. Property descriptions become `#` comments. Constraints the type cannot express follow as tags, such as `# @minimum 1 @default 20`; they are dropped first when a parameter list exceeds `DEFAULT_INPUT_SCHEMA_MAX_CHARS`, and past that it falls back to `args: Any = None, /, **kwargs: Any`. Schemas only shape the declarations; values are not validated against them. Local references (`#/$defs/...`, `#/definitions/...`) are expanded; recursive and remote references render as `Any`. `render_tool_sample()` renders one tool's description and declaration, for tool listings.

The declarations describe the script API for the model; they are not fed to Monty's type checker, which would reject dynamic but valid scripts and adds a type-checking pass to every execution.

## Using with pi-agent

`create_codemode_tool()` gives an `Agent` a `codemode` tool: it exposes the agent's other tools to the sandbox and wraps `execute()` as an `AgentTool`.

```python
from pi_agent import Agent, AgentInitialState, AgentTool, AgentToolResult
from pi_ai import TextContent, Type, create_models, faux_assistant_message, faux_provider, faux_text, faux_tool_call
from pi_codemode import create_codemode_tool


async def read_file(tool_call_id, params, signal=None, on_update=None):
    return AgentToolResult([TextContent(Path(params["path"]).read_text())])


agent_tools = [AgentTool("read", "Read a file", Type.Object({"path": Type.String()}), execute=read_file)]
codemode = create_codemode_tool(agent_tools)
print(codemode.description.splitlines()[0])  # Run Python that calls tools as `await tools.<name>(...)`. ...

# A scripted model calls it once: the script's output and value come back as the tool result.
models = create_models()
faux = faux_provider()
models.set_provider(faux.provider)
script = 'source = await tools.read(path="pyproject.toml")\nreturn source.splitlines()[0]'
faux.set_responses([
    faux_assistant_message([faux_tool_call("codemode", {"code": script})], stop_reason="toolUse"),
    faux_assistant_message([faux_text("done")]),
])
agent = Agent(initial_state=AgentInitialState(model=faux.get_model(), tools=[codemode]), stream_fn=models.stream_simple)
await agent.prompt("What is the first line of pyproject.toml?")
tool_result = next(message for message in agent.state.messages if message.role == "toolResult")
print([block.text for block in tool_result.content])  # ['"[project]"']
```

Each nested tool becomes a `CodemodeTool` (`pi_codemode.agent.codemode_tool_of()`): its arguments are validated against the agent tool's parameters (JSON Schema or Pydantic model), a tool with an `output_schema` resolves to its `structured_content` and any other to its text content, and a result with `is_error` raises in the script. The codemode tool's description is a one-line instruction followed by `render_declarations()`. Each call parses the input with `parse_codemode_source()` (honoring `timeout_ms`) and runs it in a fresh sandbox from `sandbox(tools)`, which defaults to `CodemodeSandbox(tools=tools)`; pass a factory that shares an `AsyncMonty` pool to avoid starting one per call. The result's content is `result.output` as pi-ai `TextContent`/`ImageContent` (`item.to_content()`), then the JSON value or the error's traceback, with `is_error` for failed scripts.

Calling `tool.execute()` directly skips the agent's `before_tool_call` and `after_tool_call` hooks for nested calls.

## Deterministic runs

`sandbox.start(code, ...)` runs a script so that a host can run it again after a restart and get the same behavior, for example to resume it from a durable log of its calls' results. `execute()` is unaffected. `pi_durable.codemode` builds its durable `codemode` tool on this (see the [pi_durable README](../durable/README.md#code-mode)).

The sandbox only runs when the host hands it something, so it is deterministic except for what it reads from its environment. `start()` fixes that through Monty's `os_policy`:

- **Time:** `time.time()`, `datetime.now()` and the other clocks read `started_at`, then the `now` of the result being processed, never going backwards. `time.monotonic()` and `time.perf_counter()` count from `started_at`. A sleep (`asyncio.sleep`, `time.sleep`) returns at once and advances the clock by its duration.
- **Randomness:** `random` is seeded with `seed` (as `random.seed(seed)`; unseeded `random.Random()` instances derive from it), and `os.urandom()` draws from a generator over `seed`.
- **Timezone:** UTC, whatever the host's (`execute()` uses the host's zone).
- **Result order:** the host answers each call with a result number (1, 2, ...), and the sandbox applies results one at a time, strictly in that order, whatever order they are delivered in.

The run declares its own tools and globals, and the host answers every call itself:

```python
from pi_codemode import CodemodeDeclaredTool

started_at = 1_768_471_200_000  # 2026-01-15T10:00:00Z

async with CodemodeSandbox() as sandbox:
    answered = []

    def on_call(call):
        # call.id is the sandbox's own number for the call, in the order the script made them.
        answered.append(call)
        seq = len(answered)  # result numbers run 1, 2, ...
        run.deliver(seq, started_at + 1000 * seq, call.id, {"ok": True, "value": f"contents of {call.args['path']}"})

    run = sandbox.start(
        'import time\ntext(str(time.time()))\nreturn await tools.read(path="a.txt")',
        started_at=started_at,
        seed=7,
        tools=[CodemodeDeclaredTool("read", description="Read a file")],
        on_started=lambda fingerprint: print(fingerprint.startswith("pydantic-monty")),  # Monty versions, prelude hash, limits
        on_call=on_call,
        on_idle=lambda seq, issued: print("idle", seq, issued),  # after result `seq` (0: the first run), `issued` calls in total
    )
    result = await run.result
    print(result.output[0].text, result.value, result.during)
```

A run has no deadline unless `timeout_ms` is given. Result numbers must run 1, 2, ... without gaps. The same script, `started_at`, `seed`, tools (with their `js_name`), globals and store, answered with the same results in the same order, makes the same calls with the same arguments and produces the same output on any sandbox with the same fingerprint. A host replays a run by starting it again from the same start data and delivering its journaled results in their order, each once the sandbox has made the call it answers; when a replayed script makes another call than the journal holds, or its `on_idle` counts differ from the journaled ones, it took a different path, and the host ends it. To end a run with an error the script cannot catch, abort its `signal`. A failed answer (`{"ok": False, "message": ...}`) raises `Exception(message)`; its optional `properties` are validated but not visible to the script, since Monty exceptions hold only a message.

## Telemetry

`execute()` and `create_codemode_tool()` take a [`pi_telemetry`](../telemetry/README.md) context as `telemetry_context` (an addition to upstream): each run then records a `pi.codemode.run` span with the script's line count, its deadline, the number of tool calls, the duration, and the outcome or error kind (`script`, `timeout`, `aborted`, `sandbox`), with a `pi.codemode.timeout` event when the deadline expires. Without a context nothing is recorded. The span never holds the script, its arguments, output or error text, unless the context opted in with `with_content_capture()`.

```python
from pi_codemode import CodemodeSandbox, CodemodeTool
from pi_telemetry import InMemoryTelemetryContext


async def double(args, context):
    return args["n"] * 2


telemetry = InMemoryTelemetryContext()
async with CodemodeSandbox(tools=[CodemodeTool("double", double)]) as traced:
    await traced.execute("return await tools.double(n=21)", telemetry_context=telemetry)
[span] = telemetry.get_spans()
print(span.name, span.attributes["pi.codemode.outcome"], span.attributes["pi.codemode.nested_call_count"])
```

Hosts of deterministic runs record the span with `pi_codemode.telemetry.trace_codemode_run()`; pi_durable's code mode tool does, adding whether the run replayed a journal after a restart and a `pi.codemode.different_path` event when a replay diverges. `pi_telemetry` is pi_codemode's only other pi dependency, and needs nothing.

## Results

`execute()` never raises for script failures. `result.error.kind` is one of:

| kind      | meaning                                                                     |
| --------- | --------------------------------------------------------------------------- |
| `script`  | the script raised or failed to parse; `stack` is a traceback of `codemode.py` |
| `timeout` | the deadline expired; the session was discarded                              |
| `aborted` | `signal` fired or `close()` was called; the session was discarded           |
| `sandbox` | the sandbox or its transport failed, for example a crashed worker, a pool that cannot start, or a broken bridge |

A result is a `CodemodeSuccess` (`value`, `output`, `calls`, `store_writes`) or a `CodemodeFailure` (`error`, `output`, `calls`); `result.ok` tells them apart (`isinstance` narrows the type). `result.output` holds the `CodemodeTextOutput` and `CodemodeImageOutput` items in the order the script produced them, also for failed executions. The host keeps all of it until the script ends, so output is limited to `MAX_OUTPUT_CHARS` (16 Mi) characters of text and base64 image data and `MAX_OUTPUT_ITEMS` (100000) items. Past either limit the script fails with a `ValueError`, even if it catches the error. `result.calls` lists every tool call with `status` `"ok"`, `"error"` or `"cancelled"`. A call that is still running when the script returns (not awaited) is aborted through the tool's `context.signal` and reported as `cancelled`.

## How it works

Each `execute()` checks a session out of the pool, feeds it the prelude, then starts the script with Monty's `feed_start` and answers every suspension itself. On a warm pool a script that only returns takes about 3 ms end to end. The worker is a separate process; the script reaches the host only through two channels:

- **Calls.** The prelude's helpers call names the sandbox does not define (`__codemode_call__`, `__codemode_global__`, `__codemode_exit__`, `__codemode_fail__`), which Monty hands to the host as function snapshots. The host checks the arguments are JSON, runs the tool, and answers with a pending future at once, so calls start eagerly and concurrent calls run concurrently. When every sandbox task waits, Monty reports a future snapshot, and the host hands in the results that are ready.
- **Records.** `text()`, `print()`, `image()` and `store()` print one JSON record per line with the builtin `print`, which the prelude keeps to itself. Monty streams printed output to the host and flushes it before every host call, so records arrive in order with the calls, and output a script printed before it was stopped is kept.

The prelude's helpers are closures of one factory, so their state and the builtins and module functions they use are out of the script's reach; the host validates every record and call anyway, and garbage fails the run as a broken bridge. OS calls get Monty's default denial, except sleeps and, in deterministic runs, the clock and `os.urandom()`.

The host owns the deadline and the abort signal. When either fires, it cancels the task that drives the session: Monty kills a worker that is computing, and the pool replaces it.

## Runtime notes

- Monty implements a subset of Python. Among what it lacks: class inheritance (so scripts cannot define exception classes), generators (`yield`), `del`, `globals()`, `datetime.fromtimestamp()`, and exceptions with more than one argument. Scripts get Monty's error for these.
- Monty has no read-only attributes, so a script can rebind `tools.echo` or `print` for itself; the host side is unaffected.
- `tools.missing` raises Monty's `AttributeError`; when the script does not catch it, the reported message names the close matches as upstream's does. `tools["missing"]` names them in the exception itself.
- Every host round trip costs about 0.3 ms, and an awaited tool call takes two (the call and its result), so a loop of sequential calls runs at about 0.6 ms per call; gathered calls overlap.
- A worker that dies while its script waits on tools is noticed when the next result is handed to it.
- Deeper recursion than `max_recursion_depth` (default 1000) raises a catchable `RecursionError`.
- Monty 1.2 resolves a top-level comprehension variable named like a builtin (`[id for id in ids]`, `{type: n for type, n in pairs}`) to the builtin instead of the loop value. Before feeding a script, the sandbox renames such variables inside their comprehension (to `__codemode_<name>`), in place, so the script works as written and line numbers stay the same; an error message that names the variable shows the new name.
