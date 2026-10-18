# pi_agent

Stateful agent with tool execution and event streaming, built on [`pi_ai`](../ai/README.md). Installed with `pip install pi-ai-python`.
Python port of `@earendil-works/pi-agent-core`; behavior and event order match the TypeScript package.

## Installation

```bash
pip install pi-ai-python             # contains both pi_agent and pi_ai
```

## Quick start

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

Listeners may be plain functions or coroutines; async listeners are awaited in
subscription order. `Agent.events()` is the same stream as an async iterator:

```python
async for event in agent.events("Summarize README.md"):
    if event.type == "tool_execution_start":
        print("running", event.tool_name, event.args)
```

## Core concepts

### Agent messages vs LLM messages

The transcript holds *agent messages*: the LLM messages from `pi_ai`
(`SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`) plus any
app-specific message object with a `role` attribute (for example a dataclass).
LLMs only understand the four standard roles; `convert_to_llm` bridges the gap before
each request:

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

## Event flow

`prompt("Hello")`:

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

With tool calls the loop continues with another turn:

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

| Event | Fields | Description |
| --- | --- | --- |
| `agent_start` | | Run begins |
| `agent_end` | `messages` | Last event of the run; awaited listeners still count toward settlement |
| `turn_start` | | One provider request plus its tool executions |
| `turn_end` | `message`, `tool_results` | Turn completed |
| `message_start` / `message_end` | `message` | Any message (system, user, assistant, tool result) |
| `message_update` | `message`, `assistant_message_event` | Assistant streaming only |
| `tool_execution_start` | `tool_call_id`, `tool_name`, `args` | Tool begins |
| `tool_execution_update` | `partial_result` | Tool progress |
| `tool_execution_end` | `result`, `is_error`, `duration_ms` | Tool finished |

Event classes (`MessageUpdateEvent`, `ToolExecutionEndEvent`, ...) are slotted
dataclasses, so you can also use structural pattern matching:

```python
from pi_agent import MessageUpdateEvent, ToolExecutionEndEvent
from pi_ai import TextDeltaEvent

match event:
    case MessageUpdateEvent(assistant_message_event=TextDeltaEvent(delta=delta)):
        print(delta, end="")
    case ToolExecutionEndEvent(tool_name=name, is_error=True):
        print(f"{name} failed")
```

### Tool execution modes

- `"parallel"` (default): tool calls are preflighted sequentially, allowed tools run
  concurrently, `tool_execution_end` is emitted as each tool finishes, and tool-result
  messages plus `turn_end.tool_results` follow assistant source order.
- `"sequential"`: one by one.

Set it with `Agent(tool_execution="sequential")` or per tool with
`execution_mode="sequential"` (any sequential tool in a batch makes the whole batch sequential).

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

- `before_tool_call` runs after `tool_execution_start` and argument validation, and can block execution.
- `after_tool_call` runs after execution, before `tool_execution_end`; omitted fields keep their values.
- `terminate=True` (from a tool result, a blocked call, or an override) skips the automatic
  follow-up request only when *every* finalized result in the batch terminates.
- With the `Agent` class, assistant `message_end` processing is a barrier before tool
  preflight, so `before_tool_call` sees state that already contains the assistant message.

### Request preparation and turn finalization

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

`finish_turn` may also return the strings `"continue"`/`"end"` or `{"action": ...}`.
`"end"` stops after `turn_end` without polling queues. `"continue"` guarantees one more
request; tool results, steering or follow-ups already scheduling a request satisfy it.
Error and aborted responses always end the run. Lifecycle per provider turn:

```
selected input events → prepare_request → provider response → tool results → finish_turn → turn_end → next turn or agent_end
```

### `continue_()` and queued input

`continue_()` (alias `resume()`; `continue` is a Python keyword) continues from the
existing transcript. Empty and system-only transcripts raise. A non-assistant tail is
retried; an assistant tail falls back to one queued steering batch, then one queued
follow-up batch.

## Agent options

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
)
```

## Agent state

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

The transcript owns the system prompt and tool declarations. Before every request the
loop diffs `state.tools` against the tools the transcript declares and announces the
change in a system message. To change the prompt mid-conversation, add a system message:

```python
from pi_ai import SystemMessage, UserMessage

await agent.prompt([
    SystemMessage("", sections={"skills": "<skills>...</skills>"}),
    UserMessage("Continue"),
])
```

## Methods

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

## Tools

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

Raise an exception when a tool fails (or return `AgentToolResult(..., is_error=True)`);
the error text is reported to the model as an error tool result.

### Tools from functions

`@agent_tool` builds the schema from type hints and the docstring:

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

Supported annotations: `str`, `int`, `float`, `bool`, `list[...]`, `dict[...]`,
`Literal`, `Enum`, `X | None`, `Annotated[..., "description"]`, dataclasses and Pydantic
models (a single Pydantic parameter becomes the whole schema and is passed validated).
`tool_call_id`, `signal` and `on_update` parameters are injected. `str` results become
text content, other values are sent as JSON. Sync functions run in a worker thread.

## Proxy usage

For apps that route LLM calls through a pi proxy server:

```python
from pi_agent import Agent, stream_proxy

agent = Agent(
    stream_fn=lambda model, context, options: stream_proxy(
        model, context, {**(options or {}), "auth_token": token, "proxy_url": "https://your-server.com"}
    ),
)
```

## Low-level API

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

These low-level streams are observational: they preserve event order but do not wait
for your event handling before later loop phases continue. Use `Agent` when message
processing must be a barrier before tool preflight. `run_tool_call()` runs one tool call
through the same preparation, validation and hooks without emitting events.

## License

MIT
