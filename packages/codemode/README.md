# pi_codemode

Runs model-written scripts in a sandbox whose only capability is calling injected tools. Nested tool calls never enter the LLM context; only the script's output and return value do.

This is the Python port of [`@earendil-works/pi-codemode`](https://github.com/earendil-works/pi/tree/main/packages/codemode). Scripts are JavaScript by default, run by QuickJS exactly as upstream runs them: the same QuickJS build (`quickjs-wasi` 3.6.2, QuickJS-NG compiled to WebAssembly), the same prelude, the same results, the same deterministic fingerprints. Python scripts are the alternative, run by [Monty](https://github.com/pydantic/monty), Pydantic's sandbox for AI-written Python. A sandbox runs one language, and a code mode tool built on it offers that language only.

| Class | Import from | Scripts | Where they run |
| --- | --- | --- | --- |
| `CodemodeSandbox` | `pi_codemode` | JavaScript | a fresh QuickJS VM (its own wasm instance, on wasmtime) on a thread of its own per execution |
| `MontyCodemodeSandbox` | `pi_codemode` | Python | a session of a Monty worker pool (one `monty` subprocess per execution) |
| `CodemodeSandboxBase` | `pi_codemode` or `pi_codemode.portable` | either | wherever its launcher puts them; the host side every sandbox shares |

Install the sandbox of the language you use:

```bash
pip install 'pi-ai-python[codemode]'         # JavaScript: wasmtime (QuickJS ships with the package)
pip install 'pi-ai-python[codemode-python]'  # Python: pydantic-monty
```

Without them, everything but the sandboxes (types, source parsing, declarations, identifiers, `CodemodeSandboxBase`) still imports; each sandbox raises an `ImportError` that names its extra.

The API (in snake_case), the script facilities, the result format, the error kinds, the limits, deterministic runs and the declarations for the model follow upstream. Functions whose output depends on the language (`parse_codemode_source`, `render_declarations` and the other renderers, `to_codemode_identifier`, `create_codemode_tool`) take `language="javascript"` (the default) or `language="python"`.

## Usage

```python
import json
from pathlib import Path

from pi_codemode import CodemodeSandbox, CodemodeTool


async def read(args, context):
    return Path(args["path"]).read_text()


sandbox = CodemodeSandbox(
    timeout_ms=60_000,
    tools=[CodemodeTool("read", read, description="Read a file")],
)

result = await sandbox.execute("""
const source = await tools.read({ path: "package.json" });
text("bytes " + source.length);
return JSON.parse(source).name;
""")

print(result.output)  # [CodemodeTextOutput(text='bytes 24', console=False)]
if result.ok:
    print(result.value)  # demo
else:
    print(result.error.kind, result.error.message)

await sandbox.close()
```

The same with a Python script:

```python
from pi_codemode import MontyCodemodeSandbox

sandbox = MontyCodemodeSandbox(tools=[CodemodeTool("read", read, description="Read a file")])
result = await sandbox.execute("""
import tomllib
source = await tools.read(path="pyproject.toml")
text("bytes " + str(len(source)))
return tomllib.loads(source)["project"]["name"]
""")
```

`code` is the body of an async function: top-level `await` and `return` work. A sandbox takes `tools`, `globals`, `timeout_ms` (default 300,000; `math.inf` disables the deadline, and the script then runs until it settles or `signal` aborts it) and `memory_limit_bytes`. A script that waits on something nothing can settle (no tool call pending) fails right away instead of hanging.

`execute()` takes `signal` (a pi-ai `AbortSignal`) or `context` (a pi-chord `Context`, whose abort signal is used), `timeout_ms`, `store` and `telemetry_context`. Aborts, timeouts, and `close()` end the execution at once: a VM that computes is interrupted (QuickJS) or a worker that computes is killed (Monty). Results never wait for the sandbox to go away.

### JavaScript scripts

Upstream's environment, unchanged:

- `tools.<name>(args)` returns a promise. Arguments and results make a JSON round trip; a failed call rejects with an `Error` carrying the message and the outcome's extra properties. Tool names are also identifiers (`to_codemode_identifier`: characters that are not valid in a JavaScript identifier become `_`), so `my-tool` is `tools.my_tool` as well as `tools["my-tool"]`. Reading a member that does not exist throws a `TypeError` naming close matches.
- `ALL_TOOLS` lists `{ name, description }` for every tool, with `name` as the identifier.
- `text(value)` appends a text item (non-strings are JSON); `console.log/info/warn/error/debug(...)` append items marked `console`; `image(urlOrItem)` appends an image from a base64 `data:` URL, `{ image_url }` or an image block (`{ type: "image", data, mimeType }`), checked against the PNG, JPEG, GIF and WebP signatures.
- `exit()` ends the script successfully at once, keeping its output and store writes.
- Globals are top-level functions; `a.b` names group into a frozen `a` object, and `spread` globals receive the whole argument list.
- `store(key, value)` and `load(key)`: see [Store](#store).
- Nothing else: no timers, `fetch`, modules, `require`, `process` or `WebAssembly`; `eval` only makes more code in the same VM. The built-ins are frozen.

`memory_limit_bytes` caps QuickJS's heap; allocations beyond it throw `InternalError: out of memory` in the script. Deeper recursion than the stack limit (`max_stack_size`, at most and by default 512 KiB) throws a catchable `RangeError`. `interrupt_budget` optionally ends a script that computes for more than that many interrupt polls (QuickJS polls about every 10,000 opcodes) with `kind="timeout"`, whatever the deadline. `wasm_path` loads another `quickjs.wasm`.

### Python scripts

The same facilities in Python:

- `tools.<name>(...)` returns an awaitable. Pass keyword arguments (`tools.read(path="x")`) or one positional value (`tools.read({"path": "x"})`). A tool that raises makes the `await` raise an `Exception` with the same message. In Python identifiers, keywords also get a trailing `_` (`class` is `tools.class_`).
- Calls start when they are made, like JavaScript promises, so `asyncio.gather(tools.a(), tools.b())` runs both concurrently on the host, and a call the script never awaits is cancelled when it ends.
- `print(...)` appends console items, one per call, formatted as Python's `print` formats them. `text`, `image` (with dicts for the object forms), `exit`, `store`, `load`, `ALL_TOOLS` and globals behave as in JavaScript. With `spread=True`, a global receives the positional arguments as a list (keyword arguments, if any, as one trailing dict), and `signature` holds a Python parameter list.
- Monty's standard library: `json`, `re`, `math`, `datetime`, `random`, `asyncio.gather`/`sleep` and the like. Nothing else: no filesystem (`open()` raises `PermissionError`), environment, network, subprocesses or host modules. `asyncio.sleep`/`time.sleep` wait on the host (virtual in `start()`).

`memory_limit_bytes` makes allocations beyond it fail with `MemoryError`; `max_recursion_depth` (default 1000) bounds recursion; `timezone` sets the zone of `execute()` runs. Each execution checks a session out of an `AsyncMonty` pool: a dedicated `monty` worker subprocess. Executions run in parallel up to the pool's `max_processes` (default: the CPU count). A worker that crashes fails only its own execution with `kind="sandbox"`. To share one pool between sandboxes, pass it in; the sandbox then never closes it:

```python
from pydantic_monty import AsyncMonty

async with AsyncMonty(max_processes=4) as pool:
    async with MontyCodemodeSandbox(pool=pool) as first, MontyCodemodeSandbox(pool=pool) as second:
        results = await asyncio.gather(first.execute("return 1"), second.execute("return 2"))
        print([result.value for result in results])  # [1, 2]
```

How upstream's JavaScript facilities map to Python:

| JavaScript | Python |
| --- | --- |
| `// @options: {...}` first line | `# @options: {...}` first line |
| `await tools.read({ path })` | `await tools.read(path=...)` or `await tools.read({"path": ...})` |
| `Promise.all([...])` | `asyncio.gather(...)`; calls start when made in both |
| `text(value)`: `undefined` prints `undefined` | `text(value)`: `None` prints `null` |
| `console.log/info/warn/error/debug(...)` | `print(*values, sep=" ", end="\n")` |
| `store(key, undefined)` deletes; `load` of a missing key is `undefined` | `store(key, None)` deletes; `load` of a missing key is `None` |
| a failed tool rejects with `Error(message)` and its extra properties | the `await` raises `Exception(message)`; extra properties are not visible |
| unknown tool: `TypeError` naming close matches | `AttributeError` naming close matches |
| `RangeError` (store size, output limit) | `ValueError` |
| stack `at f (codemode.js:2:31)` | CPython-style traceback, `File "codemode.py", line 2, in f` |

## Choosing the language

A sandbox has one language (`sandbox.language`), and the tools built on it follow it, so the model is offered one language, never both:

- `create_codemode_tool(tools, language="javascript")` (below) makes the tool description, its declarations and the source format of that language, and its default sandbox is a `CodemodeSandbox` or a `MontyCodemodeSandbox`. A sandbox factory of the other language is refused.
- pi_durable's code mode extension follows the sandbox it is given (see the [pi_durable README](../durable/README.md#code-mode)).

## Store

`store`/`load` let scripts keep values across executions. The sandbox does not persist anything itself: pass the current values as `store`, and a successful result reports what the script changed as `result.store_writes` (`set` and `delete`). Failed executions report no writes.

```python
saved = {}
async with CodemodeSandbox() as sandbox:
    for _ in range(2):
        result = await sandbox.execute('store("runs", (load("runs") ?? 0) + 1)', store=saved)
        if result.ok:
            for key in result.store_writes.delete:
                saved.pop(key, None)
            saved.update(result.store_writes.set)
print(saved)  # {'runs': 2}
```

`load` returns a copy, so mutating it does not change the store; a missing key loads as `undefined` (`None`). Storing `undefined` (`None`) deletes the key. A value may be at most `MAX_STORE_VALUE_CHARS` (256 Ki) characters of JSON and all values together at most `MAX_STORE_TOTAL_CHARS` (1 Mi); larger writes throw `RangeError` (`ValueError`) inside the script.

## Source format

`parse_codemode_source()` accepts a script whose first line may be an options line, `// @options:` in JavaScript and `# @options:` in Python:

```python
from pi_codemode import parse_codemode_source

source = parse_codemode_source('// @options: {"max_output_tokens": 2000, "timeout_ms": 30000}\nreturn 1')
print(source.options)  # CodemodeSourceOptions(max_output_tokens=2000, timeout_ms=30000)
print(repr(source.code))  # '\nreturn 1'
print(parse_codemode_source('# @options: {"timeout_ms": 10}\nreturn 1', "python").options.timeout_ms)  # 10
```

Supported fields are `max_output_tokens`, a token budget for the output, and `timeout_ms`, a hard deadline. The sandbox does not act on them; the caller decides. The options line is replaced by an empty line, so line numbers in stack traces still match the input. Empty input, invalid JSON, unknown fields, an options line without code, or an options line on any line but the first raise `CodemodeSourceError`. `CODEMODE_SOURCE_GRAMMAR` (upstream's, for JavaScript) and `PYTHON_CODEMODE_SOURCE_GRAMMAR` are Lark grammars for providers that support grammar-constrained tool input; `codemode_source_grammar(language)` picks one. All are also available from the dependency-free `pi_codemode.source` module.

## Declarations for the model

Tools and globals can carry `description`, `input_schema`, and `output_schema` (JSON Schema). `render_declarations()` turns them into declarations for a model-facing tool description: TypeScript for JavaScript scripts, upstream's output character for character, and Python stubs for Python scripts.

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
print(render_declarations(tools=[read_tool], language="python"))
```

prints

```text
declare const tools: {
  /** Read a file */
  read(args: {
  // @integer @minimum 1 @default 2000
  limit?: number;
  // Path relative to the workspace
  path: string;
}): Promise<string>;
};
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

In TypeScript, objects render on one line with properties sorted by name, or one property per line with `//` comments when a property has a description or constraints the type cannot express; unions come from `anyOf`/`oneOf`/`type` arrays, `enum` and `const` are literal types, arrays are `Array<T>` and tuples `[A, B]`. An input type longer than `DEFAULT_INPUT_SCHEMA_MAX_CHARS` (16,000) loses its tags, then renders as `unknown`. In Python, an object input whose property names are identifiers becomes keyword-only parameters, any other input one positional `args` parameter; nested objects render as inline `TypedDict[{"key": T, "other": NotRequired[T]}]` types (PEP 764 notation), arrays as `list[T]`, maps as `dict[str, T]`, constants and enums as `Literal[...]`, and past the budget the parameters fall back to `args: Any = None, /, **kwargs: Any`. In both, local references (`#/$defs/...`, `#/definitions/...`) are expanded and recursive or remote references render as `unknown`/`Any`. Schemas only shape the declarations; values are not validated against them. `render_tool_sample()` renders one tool's description and declaration, for tool listings.

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
print(codemode.description.splitlines()[0])  # Run JavaScript that calls tools as `await tools.<name>(args)`. ...

# A scripted model calls it once: the script's output and value come back as the tool result.
models = create_models()
faux = faux_provider()
models.set_provider(faux.provider)
script = 'const source = await tools.read({ path: "pyproject.toml" });\nreturn source.split("\\n")[0];'
faux.set_responses([
    faux_assistant_message([faux_tool_call("codemode", {"code": script})], stop_reason="toolUse"),
    faux_assistant_message([faux_text("done")]),
])
agent = Agent(initial_state=AgentInitialState(model=faux.get_model(), tools=[codemode]), stream_fn=models.stream_simple)
await agent.prompt("What is the first line of pyproject.toml?")
tool_result = next(message for message in agent.state.messages if message.role == "toolResult")
print([block.text for block in tool_result.content])  # ['"[project]"']
```

`create_codemode_tool(agent_tools, language="python")` makes the Python tool instead. Each nested tool becomes a `CodemodeTool` (`pi_codemode.agent.codemode_tool_of()`): its arguments are validated against the agent tool's parameters (JSON Schema or Pydantic model), a tool with an `output_schema` resolves to its `structured_content` and any other to its text content, and a result with `is_error` fails the call in the script. The codemode tool's description is a one-line instruction followed by `render_declarations()`. Each call parses the input with `parse_codemode_source()` (honoring `timeout_ms`) and runs it in a fresh sandbox from `sandbox(tools)`, which defaults to `CodemodeSandbox(tools=tools)` (`MontyCodemodeSandbox(tools=tools)` for Python; pass a factory that shares an `AsyncMonty` pool to avoid starting one per call). The result's content is `result.output` as pi-ai `TextContent`/`ImageContent` (`item.to_content()`), then the JSON value or the error's stack, with `is_error` for failed scripts.

Calling `tool.execute()` directly skips the agent's `before_tool_call` and `after_tool_call` hooks for nested calls.

## Deterministic runs

`sandbox.start(code, ...)` runs a script so that a host can run it again after a restart and get the same behavior, for example to resume it from a durable log of its calls' results. `execute()` is unaffected. `pi_durable.codemode` builds its durable `codemode` tool on this (see the [pi_durable README](../durable/README.md#code-mode)).

The sandbox only runs when the host hands it something, so it is deterministic except for what it reads from its environment, which `start()` fixes:

- **Time:** the clocks read `started_at`, then the `now` of the result being processed, never going backwards. In JavaScript `Date.now()`, `new Date()` and `performance.now()` read them through WASI, as upstream's; in Python `time.time()`, `datetime.now()` and the other clocks, and a sleep returns at once and advances the clock by its duration.
- **Randomness:** `Math.random()` (JavaScript) and `random`/`os.urandom()` (Python) derive from `seed`.
- **Timezone:** UTC, whatever the host's (`execute()` uses the host's zone).
- **Garbage collection:** JavaScript has no `WeakRef` or `FinalizationRegistry`.
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
        'text(String(Date.now()));\nreturn await tools.read({ path: "a.txt" });',
        started_at=started_at,
        seed=7,
        tools=[CodemodeDeclaredTool("read", description="Read a file")],
        on_started=print,  # quickjs-wasi 3.6.2, quickjs 0.17.0, prelude 8c6c9242, deterministic 1, stack 524288, heap none, budget none
        on_call=on_call,
        on_idle=lambda seq, issued: print("idle", seq, issued),  # after result `seq` (0: the first run), `issued` calls in total
    )
    result = await run.result
    print(result.output[0].text, result.value, result.during)
```

A run has no deadline unless `timeout_ms` is given. Result numbers must run 1, 2, ... without gaps. The same script, `started_at`, `seed`, tools (with their `js_name`), globals and store, answered with the same results in the same order, makes the same calls with the same arguments and produces the same output on any sandbox with the same fingerprint. A JavaScript run reports upstream's fingerprint and behaves as upstream's does on the same QuickJS build, call for call and value for value. A host replays a run by starting it again from the same start data and delivering its journaled results in their order, each once the sandbox has made the call it answers; when a replayed script makes another call than the journal holds, or its `on_idle` counts differ from the journaled ones, it took a different path, and the host ends it. To end a run with an error the script cannot catch, abort its `signal`. A failed answer (`{"ok": False, "message": ..., "properties": {...}}`) rejects with an `Error(message)` that has the properties in JavaScript; in Python it raises `Exception(message)`, since Monty exceptions hold only a message.

## Telemetry

`execute()` and `create_codemode_tool()` take a [`pi_telemetry`](../telemetry/README.md) context as `telemetry_context` (an addition to upstream): each run then records a `pi.codemode.run` span with the script's line count, its deadline, the number of tool calls, the duration, and the outcome or error kind (`script`, `timeout`, `aborted`, `sandbox`), with a `pi.codemode.timeout` event when the deadline expires. Without a context nothing is recorded. The span never holds the script, its arguments, output or error text, unless the context opted in with `with_content_capture()`.

```python
from pi_codemode import CodemodeSandbox, CodemodeTool
from pi_telemetry import InMemoryTelemetryContext


async def double(args, context):
    return args["n"] * 2


telemetry = InMemoryTelemetryContext()
async with CodemodeSandbox(tools=[CodemodeTool("double", double)]) as traced:
    await traced.execute("return await tools.double({ n: 21 })", telemetry_context=telemetry)
[span] = telemetry.get_spans()
print(span.name, span.attributes["pi.codemode.outcome"], span.attributes["pi.codemode.nested_call_count"])
```

Hosts of deterministic runs record the span with `pi_codemode.telemetry.trace_codemode_run()`; pi_durable's code mode tool does, adding whether the run replayed a journal after a restart and a `pi.codemode.different_path` event when a replay diverges. `pi_telemetry` is pi_codemode's only other pi dependency, and needs nothing.

## Results

`execute()` never raises for script failures. `result.error.kind` is one of:

| kind      | meaning                                                                     |
| --------- | --------------------------------------------------------------------------- |
| `script`  | the script threw or failed to parse; `stack` is its stack (`codemode.js`) or traceback (`codemode.py`) |
| `timeout` | the deadline expired, or the interrupt budget was used up                   |
| `aborted` | `signal` fired or `close()` was called                                      |
| `sandbox` | the sandbox failed outside the script's control, for example a trapped VM, a crashed worker, or a broken bridge |

A result is a `CodemodeSuccess` (`value`, `output`, `calls`, `store_writes`) or a `CodemodeFailure` (`error`, `output`, `calls`); `result.ok` tells them apart (`isinstance` narrows the type). A script that returns `null`/`undefined` (`None`) has the value `None`. `result.output` holds the `CodemodeTextOutput` and `CodemodeImageOutput` items in the order the script produced them, also for failed executions. The host keeps all of it until the script ends, so output is limited to `MAX_OUTPUT_CHARS` (16 Mi) characters of text and base64 image data and `MAX_OUTPUT_ITEMS` (100000) items. Past either limit the script fails, even if it catches the error. `result.calls` lists every tool call with `status` `"ok"`, `"error"` or `"cancelled"`. A call that is still running when the script returns (not awaited) is aborted through the tool's `context.signal` and reported as `cancelled`.

## How it works

The host side (`CodemodeSandboxBase`) owns the tool table, call records, output, the deadline and the abort signal; a launcher runs the script and exchanges upstream's protocol messages with it (calls, output, done, crash, and in deterministic runs started and idle). Only names depend on the language on the host side: the identifiers tools get and the reserved globals.

**QuickJS.** `quickjs.wasm` (QuickJS-NG built by `quickjs-wasi` 3.6.2, bundled with its MIT license) is compiled once per process on wasmtime, on first use (a few hundred milliseconds, off the event loop). Each execution instantiates it afresh on a thread of its own, evaluates upstream's prelude unchanged, and drives the script as upstream's `vm.ts` does: promise jobs run one at a time with an interrupt poll every 50 jobs, so `while (true) await null` stays interruptible. The VM's only imports are the six WASI functions the build needs (clocks, randomness, `fd_write` with the output discarded, three stubs) and QuickJS's host hooks; deterministic runs replace the clocks and randomness at that level, as upstream does. wasmtime releases the GIL while wasm runs, so a computing script does not hold up the event loop; the deadline, an abort and `close()` set the flag QuickJS polls, and the thread drops the VM. A wasm trap (for example a deeply nested `JSON.parse`, which QuickJS does not stack-check) fails only that execution. A warm `execute()` takes about 15 ms (most of it the prelude freezing the built-ins), an awaited tool call about 0.5 ms, a `text()` item about 0.1 ms.

**Monty.** Each execution checks a session out of the pool, feeds it the Python prelude, then starts the script with Monty's `feed_start` and answers every suspension itself. Calls are names the sandbox does not define, which Monty hands to the host as function snapshots; output is records the prelude prints. On a warm pool a script that only returns takes about 3 ms; every host round trip costs about 0.3 ms.

## Runtime notes

- QuickJS: `eval` and `Function` stay in the VM. A script can rebind `tools.echo` for itself only; the host checks every message it receives anyway, and garbage fails the run as a broken bridge.
- Monty implements a subset of Python. Among what it lacks: class inheritance (so scripts cannot define exception classes), generators (`yield`), `del`, `globals()`, `datetime.fromtimestamp()`, and exceptions with more than one argument. Scripts get Monty's error for these. Monty 1.2 resolves a top-level comprehension variable named like a builtin (`[id for id in ids]`) to the builtin; the sandbox renames such variables inside their comprehension before feeding the script, so it works as written and line numbers stay the same.
- A Monty worker that dies while its script waits on tools is noticed when the next result is handed to it.
