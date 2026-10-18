"""pi-agent-core: stateful agent with tool execution, steering/follow-up queues and
event streaming, built on pi-ai.

Python port of ``@earendil-works/pi-agent-core``.
"""

from __future__ import annotations

from .agent import Agent, AgentInitialState, AgentListener, AgentState, default_convert_to_llm
from .agent_loop import (
    AgentEventSink,
    RunToolCallOptions,
    ToolCallHooks,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
    run_tool_call,
)
from .proxy import ProxyStreamOptions, process_proxy_event, stream_proxy
from .stream_fn import get_default_stream_fn, set_default_stream_fn
from .tools import agent_tool, schema_from_type, tool_from_function
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentRequestUpdate,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolCallOutcome,
    AgentToolResult,
    AgentToolUpdateCallback,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCallContext,
    BeforeToolCallResult,
    FinishTurn,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    PrepareRequest,
    PrepareRequestContext,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__version__ = "1.3.0"

__all__ = [
    "__version__",
    # agent
    "Agent",
    "AgentState",
    "AgentInitialState",
    "AgentListener",
    "default_convert_to_llm",
    # loop
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "run_tool_call",
    "RunToolCallOptions",
    "ToolCallHooks",
    "AgentEventSink",
    # stream function
    "set_default_stream_fn",
    "get_default_stream_fn",
    "stream_proxy",
    "process_proxy_event",
    "ProxyStreamOptions",
    # tools
    "agent_tool",
    "tool_from_function",
    "schema_from_type",
    # types
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentRequestUpdate",
    "AgentTool",
    "AgentToolCall",
    "AgentToolCallOutcome",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "AgentTurnContext",
    "AgentTurnDecision",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "FinishTurn",
    "PrepareNextTurnContext",
    "PrepareRequest",
    "PrepareRequestContext",
    "QueueMode",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionMode",
    # events
    "AgentEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
]
