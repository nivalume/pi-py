"""Durable code mode (port of ``codemode/codemode.ts``): a tool that runs model-written Python in a Monty sandbox,
whose script survives a restart.

Problem: a script is in-memory sandbox state. A restart while

    results = await asyncio.gather(*(tools.read(path=path) for path in paths))

waits for the last of 200 reads would lose the script and every read it made.

Solution: deterministic replay (:meth:`pi_codemode.CodemodeSandboxBase.start`). The sandbox only runs when the host
hands it a result; given the same start data, clock, seed, and results in the same order, it makes the same calls
with the same arguments. The tool is replay-safe: after a restart, ``execute()`` runs again, starts the script again
from the stored start data, and hands it the results it got before, in the order it got them, from a journal written
before each delivery. Nested tool calls reattach to their tasks by key (``c<VM call id>``), so they never run twice
unless their own replay policy reruns them; their results come from ``NestedCallDoc``. Then the script continues live.

Documents, scoped to the code mode call's task and gone when it settles:

- ``pi.codemode.run``: the start data, written once before the sandbox starts, and the outcome, written once when
  the call decides it (with the store writes, in one transaction), so a rerun after that returns it without a sandbox.
- ``pi.codemode.journal``: one entry per delivered result, in delivery order: the VM call ID, the clock value, the
  calls the VM had issued after processing it (once acknowledged), and a global's outcome, which nothing else stores.

Each result's acknowledged call count is recorded with the next journal commit, or at once when the VM has processed
everything delivered: a script waiting on a long call is where a restart usually finds it.

A call is launched only once the count of calls issued in the drain that made it is committed. So every key an
earlier attempt admitted is at most its last recorded count; a replay checks every recorded count, so when it releases
its calls it has made every call an earlier attempt could have admitted, and later calls are new.

A rerun refuses to replay (no new nested calls; nested calls admitted earlier may still resume on their own and are
aborted by the call's cleanup) when this module's version or the sandbox's fingerprint changed, when the deadline
passed, or when the script diverges: it makes another call, or calls in another order, than the journal says. Each
ends the call with an error result.

The journal's format is upstream's, but a run cannot be replayed across languages: the scripts, the globals' names,
and the sandbox fingerprints (Monty's versus QuickJS's) differ, so a TypeScript run read here ends with "the sandbox
changed since it started".
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypedDict, cast

from pi_ai import TextContent, Type
from pi_ai.serialization import content_from_dict, to_json_value
from pi_chord.json import copy_json
from pi_codemode.identifier import to_codemode_identifier
from pi_codemode.source import CODEMODE_SOURCE_GRAMMAR, parse_codemode_source
from pi_codemode.types import (
    CodemodeDeclaredGlobal,
    CodemodeDeclaredTool,
    CodemodeFailure,
    CodemodeHostCall,
    CodemodeSuccess,
)

from ..documents import define_doc
from ..errors import error_message
from ..harness.define import define_extension, define_tool, section
from ..harness.tool import NestedCallDoc
from ..harness.util import json_equal
from ..search import SearchDocument, rank_documents
from .describe import (
    DEFAULT_CODEMODE_INLINE_BUDGET,
    CodemodeNamespace,
    codemode_description,
    codemode_section,
    namespace_of,
    sample_of,
)
from .models import (
    CATALOG_GLOBALS,
    CODEMODE_INTERNAL_PREFIX,
    GENERATE_IMAGES_TOOL,
    MAX_CONCURRENT_MODEL_CALLS,
    MODEL_CALLS,
    bind_arguments,
    catalog_call,
    create_model_tools,
)
from .output import DEFAULT_MAX_OUTPUT_TOKENS, OutputFile, OutputFileResult, format_result

if TYPE_CHECKING:
    from pi_chord.context import Context
    from pi_codemode.runtime.execution import CodemodeRun, CodemodeSandboxBase
    from pi_codemode.types import CodemodeOutcome, CodemodeResult

    from ..env.types import ExecutionEnv
    from ..harness.types import (
        Extension,
        NestedToolExecutionResult,
        PromptInput,
        ToolExecutionApi,
        ToolExecutionResult,
        ToolOutputLimits,
        ToolRegistration,
    )

__all__ = [
    "CODEMODE_MODULE_VERSION",
    "CODEMODE_TOOL_NAME",
    "CodemodeGlobal",
    "CodemodeJournalDoc",
    "CodemodeNestedCall",
    "CodemodeRunDoc",
    "CodemodeStoreDoc",
    "CodemodeToolDetails",
    "CodemodeToolInput",
    "create_codemode_extension",
    "create_codemode_tool",
]

#: Version of everything in this module that turns stored results into what a script sees: result mapping, globals,
#: start data. Bump it when that changes, so a call started before an upgrade is not replayed by different code.
CODEMODE_MODULE_VERSION: Final = 2

CODEMODE_TOOL_NAME: Final = "codemode"


class DeclaredTool(TypedDict):
    """One tool as the script sees it: ``tools.<jsName>``.

    ``structured``: the tool declares ``structured_output_schema``, so an error result with a value resolves with it.
    ``hidden``: an internal tool of this extension (``pi.codemode.*``), which scripts reach only through a global such
    as ``models.classify()``; never in ``tools``, ``ALL_TOOLS``, or tool search.
    """

    name: str
    jsName: str
    description: str
    structured: NotRequired[Literal[True]]
    hidden: NotRequired[Literal[True]]


class StartNamespace(TypedDict):
    name: str
    description: NotRequired[str]
    instructions: NotRequired[str]
    tools: list[str]


class StartGlobal(TypedDict):
    name: str
    spread: bool


class Start(TypedDict):
    """What a call starts with; a rerun starts the sandbox from exactly this."""

    version: int
    startedAt: float
    seed: int
    #: The sandbox's fingerprint, recorded by the first run before any call is admitted.
    fingerprint: NotRequired[str]
    #: ``startedAt + timeout_ms`` when the script set one.
    deadline: NotRequired[float]
    tools: list[DeclaredTool]
    #: The namespaces of the tools, as ``describe_namespace()`` and ``search_tools()`` see them.
    namespaces: list[StartNamespace]
    globals: list[StartGlobal]
    #: ``max_output_tokens`` of the script's options line, or the default.
    maxOutputTokens: int
    store: dict[str, Any]


class RunState(TypedDict, total=False):
    start: Start
    #: Directory of the result's files (the full output, images), created once: a rerun writes the same files again.
    outputDir: str
    #: The outcome, decided once.
    result: dict[str, Any]


CodemodeRunDoc = define_doc(kind="pi.codemode.run", version=1, scope="task", initial=lambda: cast("RunState", {}))


class JournalGlobal(TypedDict):
    name: str
    args: NotRequired[Any]
    outcome: dict[str, Any]


#: ``id``: the VM's call ID; its result is the entry's position + 1 (``seq``). ``at``: the VM clock while it processes
#: this result. ``issuedAfter``: calls the VM had issued after processing this result; absent until acknowledged.
#: ``global``: a global's call and outcome; a tool's result is its nested call's.
JournalEntry = TypedDict(
    "JournalEntry",
    {"id": int, "at": float, "issuedAfter": NotRequired[int], "global": NotRequired[JournalGlobal]},
)


class JournalState(TypedDict):
    entries: list[JournalEntry]
    #: Calls the VM had issued after the script's first run, before any result; absent until acknowledged.
    firstIssued: NotRequired[int]


CodemodeJournalDoc = define_doc(
    kind="pi.codemode.journal", version=1, scope="task", initial=lambda: cast("JournalState", {"entries": []})
)


class CodemodeStoreState(TypedDict):
    values: dict[str, Any]


#: What scripts keep with ``store()`` and read with ``load()``, per conversation. Rewindable and copied as of the fork
#: point, so a fork sees the values as they were when its entry was committed.
CodemodeStoreDoc = define_doc(
    kind="pi.codemode.store",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: cast("CodemodeStoreState", {"values": {}}),
)


class CodemodeNestedCall(TypedDict):
    """One tool call a script made, as the code mode call's details show it."""

    #: The VM's number for the call; its nested call ID is ``<callId>/c<id>``.
    id: int
    name: str
    #: Compact JSON of the arguments, cut for display.
    args: str
    status: Literal["running", "ok", "error", "cancelled"]
    durationMs: NotRequired[float]
    #: Error text, cut for display.
    error: NotRequired[str]


class CodemodeToolDetails(TypedDict):
    """The code mode call's details, in its result: its tool calls in the order it made them, with their status.

    Calls to globals are not listed. While the script runs, UIs show its nested calls from ``pi.live.nestedTools``,
    so the list is written once, with the result.
    """

    calls: list[CodemodeNestedCall]
    #: Temp file with the complete text output, when the output was cut to its token budget.
    fullOutputPath: NotRequired[str]


class CodemodeToolInput(TypedDict):
    code: str


_ARGS_PREVIEW_CHARS: Final = 200
_ERROR_PREVIEW_CHARS: Final = 500


def _cut(text: str, max: int) -> str:
    return f"{text[: max - 3]}..." if len(text) > max else text


def _compact(value: Any) -> str:
    """``JSON.stringify``: compact JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(slots=True, frozen=True)
class CodemodeGlobal:
    """Internal: a function scripts call as a top-level global; its result is journaled.

    ``execute(args, api, context)`` returns a JSON value, or ``None`` (the script gets ``None``). ``args`` is what the
    script passed: its argument, or with ``spread`` the list of its arguments.
    """

    name: str
    execute: Callable[[Any, ToolExecutionApi, Context], Awaitable[Any]]
    spread: bool = False


_CODEMODE_SCHEMA: Final = Type.Object({"code": Type.String(description="Raw Python source.")})

#: Outputs bounded by the script's own token budget; the Harness's limits only stop a runaway.
_OUTPUT_LIMITS: Final[ToolOutputLimits] = {"maxBytes": 4 * 1024 * 1024, "maxLines": 100_000}

#: Globals for finding tools, answered from the start data, so a replay sees what the first run saw.
DISCOVERY_GLOBALS: Final = ("search_tools", "describe_tool", "describe_namespace")
_DEFAULT_SEARCH_LIMIT: Final = 8


@dataclass(slots=True, frozen=True)
class _Options:
    sandbox: CodemodeSandboxBase
    namespaces: Sequence[CodemodeNamespace] = ()
    inline_budget: float | None = None
    models: bool = True
    globals: Sequence[CodemodeGlobal] = ()


def _tool_of(options: _Options) -> ToolRegistration:
    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        return await _CodemodeCall(options, api, context).execute(args["code"])

    return define_tool(
        name=CODEMODE_TOOL_NAME,
        description=codemode_description(options.models),
        parameters=_CODEMODE_SCHEMA,
        output_limits=_OUTPUT_LIMITS,
        # Capable models write the script as raw text instead of a JSON-escaped string; pi-ai applies it only for them.
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": CODEMODE_SOURCE_GRAMMAR}},
        callers=["model"],
        replay="safe",
        execute=execute,
    )


def create_codemode_tool(
    *,
    sandbox: CodemodeSandboxBase,
    namespaces: Sequence[CodemodeNamespace] = (),
    inline_budget: float | None = None,
    models: bool = True,
    globals: Sequence[CodemodeGlobal] = (),
) -> ToolRegistration:
    """The code mode tool. Scripts call the conversation's callable tools as ``await tools.<name>(...)``.

    Args:
        sandbox: Runs the scripts: a :class:`pi_codemode.CodemodeSandbox`, or any
            :class:`pi_codemode.CodemodeSandboxBase`.
        namespaces: Groups of tools by name prefix, such as MCP servers, for the ``codemode`` section and tool search.
        inline_budget: Estimated tokens the tool declarations in the ``codemode`` section may use. Default 3000.
        models: Give scripts ``models.*``: the model catalog, classifiers, and image generation. Default ``True``.
        globals: Internal: host functions scripts call as globals.
    """
    return _tool_of(_Options(sandbox, tuple(namespaces), inline_budget, models, tuple(globals)))


def create_codemode_extension(
    *,
    sandbox: CodemodeSandboxBase,
    namespaces: Sequence[CodemodeNamespace] = (),
    inline_budget: float | None = None,
    models: bool = True,
    globals: Sequence[CodemodeGlobal] = (),
) -> Extension:
    """The ``codemode`` extension: the code mode tool, the internal tools of ``models.*`` (unless ``models=False``),
    and the ``codemode`` prompt section listing the tools scripts can call. Arguments as :func:`create_codemode_tool`.
    """
    options = _Options(sandbox, tuple(namespaces), inline_budget, models, tuple(globals))
    budget = DEFAULT_CODEMODE_INLINE_BUDGET if inline_budget is None else inline_budget

    def render(input: PromptInput, _context: Context) -> str | None:
        # Deferred tools are found with `search_tools()`, as the coding agent leaves them out too.
        tools = [
            tool
            for tool in input["agent"]["callable"]
            if not tool.name.startswith(CODEMODE_INTERNAL_PREFIX) and tool.deferred is not True
        ]
        return codemode_section(tools, options.namespaces, budget)

    return define_extension(
        "codemode",
        tools=[_tool_of(options), *(create_model_tools() if models else [])],
        sections=[section("codemode", render)],
    )


def _ok(value: Any) -> CodemodeOutcome:
    """A value outcome; ``None`` is left out, as upstream leaves out ``undefined``."""
    return {"ok": True} if value is None else {"ok": True, "value": value}


def _outcome_of(name: str, structured: bool, result: NestedToolExecutionResult) -> CodemodeOutcome:
    """What a script gets for a nested call, as in the coding agent.

    A tool with a schema returns its value, also on an error result that carries one (``bash``'s ``{"output",
    "exitCode"}`` for a nonzero exit); a schema-less tool returns its text output, images left out, as its declaration
    says. Any other error raises, with its text: a schema-less tool's output, else its error diagnostics.
    """
    has_value = "structuredOutput" in result
    value = result.get("structuredOutput") if structured else _text_of(result.get("structuredOutput"))
    if not result["isError"] or (structured and has_value):
        return _ok(value)
    message = "" if structured else cast("str", value)
    diagnostics = "\n".join(
        diagnostic["message"] for diagnostic in result["diagnostics"] if diagnostic["severity"] == "error"
    )
    outcome: dict[str, Any] = {"ok": False, "message": message or diagnostics or f"Tool {name} failed"}
    if not structured or has_value:
        outcome["properties"] = {"structuredOutput": value}
    return cast("CodemodeOutcome", outcome)


def _text_of(value: Any) -> str:
    """The text of a schema-less tool's collapsed output: a string, one image, or a list of text and image blocks."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        item["text"]
        for item in cast("list[Any]", value)
        if isinstance(item, dict) and item.get("type") == "text"  # pyright: ignore[reportUnknownMemberType]
    )


def _error_result(message: str) -> ToolExecutionResult:
    return {"output": [TextContent(message)], "isError": True}


def _result_json(result: ToolExecutionResult) -> dict[str, Any]:
    """A tool result as the JSON the run document keeps."""
    value: dict[str, Any] = {key: item for key, item in result.items() if key != "output"}
    if "output" in result:
        value["output"] = [to_json_value(item) for item in result["output"]]
    return cast("dict[str, Any]", copy_json(value))


def _result_from_json(value: Any) -> ToolExecutionResult:
    result = dict(value)
    if "output" in result:
        result["output"] = [content_from_dict(item) for item in result["output"]]
    return cast("ToolExecutionResult", result)


@dataclass(slots=True)
class _Queued:
    id: int
    at: float
    outcome: CodemodeOutcome
    global_: JournalGlobal | None = None


class _CodemodeCall:
    """One execution of the code mode tool: a first run, or a rerun after a restart that replays the journal."""

    def __init__(self, options: _Options, api: ToolExecutionApi, context: Context) -> None:
        from pi_ai.abort import AbortController

        self._options = options
        self._api = api
        self._context = context
        self._controller = AbortController()
        self._run: CodemodeRun | None = None
        self._start: Start = cast("Start", {})
        #: The journal as this attempt found it; replay delivers exactly these.
        self._replayed: list[JournalEntry] = []
        #: Entries in the journal, replayed and live; the next delivery is ``length + 1``.
        self._journal_length = 0
        #: Replayed entries delivered so far.
        self._next_replayed = 0
        self._replaying = False
        #: Whether an earlier attempt ran this call: its nested call keys may already exist.
        self._rerun = False
        #: Whether live calls may launch; before, they are held (see `_release()`).
        self._released = False
        #: The count an earlier attempt recorded after the script's first run.
        self._first_issued: int | None = None
        #: The highest issued count known to be committed: calls up to it may launch.
        self._durable_issued = 0
        #: Live calls waiting for their issued count to be committed.
        self._waiting: list[CodemodeHostCall] = []
        #: Journal index of each replayed call ID.
        self._replay_index: dict[int, int] = {}
        #: Calls the VM issued so far; its call IDs are 1, 2, ...
        self._issued = 0
        #: Replay: results of journaled calls, ready to deliver in journal order.
        self._ready: dict[int, CodemodeOutcome] = {}
        #: Replay: calls the journal does not have, admitted once replay is done.
        self._held: list[CodemodeHostCall] = []
        #: Live: finished calls waiting for their journal commit.
        self._queue: list[_Queued] = []
        #: Acknowledged counts to record, by result number (0: the script's first run).
        self._acks: dict[int, int] = {}
        self._committing = False
        #: Set once the call decided an outcome other than the script's own (divergence, mismatch) or failed.
        self._ended: ToolExecutionResult | None = None
        self._failure: BaseException | None = None
        #: ``models.classify()``/``generate_images()`` calls in flight, and those waiting for a slot.
        self._model_calls = 0
        self._model_waiters: list[asyncio.Future[None]] = []
        #: Images ``models.generate_images()`` returned, to notice a script that never shows them.
        self._generated_images = 0
        #: The script's tool calls for ``details``, by VM call ID.
        self._rows: dict[int, CodemodeNestedCall] = {}
        #: Resolves once the VM's fingerprint is checked or recorded; nothing is delivered or admitted before.
        self._gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tasks: set[asyncio.Future[Any]] = set()

    # ─── background work ────────────────────────────────────────────────────

    def _spawn(self, work: Awaitable[Any]) -> None:
        """Run ``work`` on its own, as upstream's ``void promise``; an unexpected error fails the call."""
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._settled)

    def _settled(self, task: asyncio.Future[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._fail(error)

    def _open_gate(self) -> None:
        if not self._gate.done():
            self._gate.set_result(None)

    # ─── the call ───────────────────────────────────────────────────────────

    async def execute(self, source: str) -> ToolExecutionResult:
        parsed = parse_codemode_source(source)
        code, source_options = parsed.code, parsed.options
        api, context = self._api, self._context
        agent = await api.agent(context)
        tools: list[DeclaredTool] = []
        for tool in agent["callable"]:
            declared: DeclaredTool = {
                "name": tool.name,
                "jsName": to_codemode_identifier(tool.name),
                "description": sample_of(tool),
            }
            if tool.structured_output_schema is not None:
                declared["structured"] = True
            if tool.name.startswith(CODEMODE_INTERNAL_PREFIX):
                declared["hidden"] = True
            tools.append(declared)
        callable_ = [tool for tool in agent["callable"] if not tool.name.startswith(CODEMODE_INTERNAL_PREFIX)]
        models = [] if not self._options.models else [*CATALOG_GLOBALS, *MODEL_CALLS]
        globals_: list[StartGlobal] = [
            *({"name": name, "spread": True} for name in DISCOVERY_GLOBALS),
            *({"name": name, "spread": True} for name in models),
            *({"name": global_.name, "spread": global_.spread is True} for global_ in self._options.globals),
        ]
        namespaces: list[StartNamespace] = []
        for tool in callable_:
            namespace = namespace_of(tool.name, self._options.namespaces)
            if namespace is None:
                continue
            entry = next((candidate for candidate in namespaces if candidate["name"] == namespace["name"]), None)
            if entry is None:
                entry = StartNamespace(name=namespace["name"], tools=[])
                if "description" in namespace:
                    entry["description"] = namespace["description"]
                if "instructions" in namespace:
                    entry["instructions"] = namespace["instructions"]
                namespaces.append(entry)
            entry["tools"].append(tool.name)
        now = api.now()
        seed = secrets.randbits(32)

        async def find(tx: Any) -> dict[str, Any]:
            run = await tx.doc(CodemodeRunDoc, api.task_id)
            if "result" in run:
                return {"done": _result_from_json(copy_json(run["result"]))}
            rerun = "start" in run
            if not rerun:
                store = await tx.doc(CodemodeStoreDoc, api.conversation_id)
                start: dict[str, Any] = {
                    "version": CODEMODE_MODULE_VERSION,
                    "startedAt": now,
                    "seed": seed,
                }
                if source_options.timeout_ms is not None:
                    start["deadline"] = now + source_options.timeout_ms
                start.update(
                    {
                        "tools": tools,
                        "namespaces": namespaces,
                        "globals": globals_,
                        "maxOutputTokens": source_options.max_output_tokens
                        if source_options.max_output_tokens is not None
                        else DEFAULT_MAX_OUTPUT_TOKENS,
                        "store": copy_json(store["values"]),
                    }
                )
                run["start"] = copy_json(start)
            journal = await tx.doc(CodemodeJournalDoc, api.task_id)
            return {
                "start": copy_json(run["start"]),
                "journal": copy_json(journal["entries"]),
                "firstIssued": journal.get("firstIssued"),
                "rerun": rerun,
            }

        found = await api.commit(find, context)
        if "done" in found:
            return found["done"]
        self._start = found["start"]
        self._replayed = found["journal"]
        self._replay_index = {entry["id"]: index for index, entry in enumerate(self._replayed)}
        self._journal_length = len(self._replayed)
        self._replaying = len(self._replayed) > 0
        self._rerun = found["rerun"]
        self._first_issued = found["firstIssued"]
        self._durable_issued = max((entry.get("issuedAfter", 0) for entry in self._replayed), default=0)
        self._durable_issued = max(self._durable_issued, self._first_issued or 0)

        if self._start["version"] != CODEMODE_MODULE_VERSION:
            return await self._decide(
                _error_result(
                    f"The script cannot resume: it started under code mode version {self._start['version']}, "
                    f"this is {CODEMODE_MODULE_VERSION}."
                )
            )
        deadline = self._start.get("deadline")
        remaining = None if deadline is None else deadline - api.now()
        if remaining is not None and remaining <= 0:
            return await self._decide(_error_result("Script timed out: its deadline passed while it was interrupted."))

        signal = context.abort_signal

        def on_abort() -> None:
            self._controller.abort(None if signal is None else signal.reason)

        remove = signal.add_listener(on_abort) if signal is not None else None
        try:
            self._run = self._options.sandbox.start(
                code,
                started_at=self._start["startedAt"],
                seed=self._start["seed"],
                tools=[
                    CodemodeDeclaredTool(tool["name"], tool["jsName"], tool["description"])
                    for tool in self._start["tools"]
                    if tool.get("hidden") is not True
                ],
                globals=[
                    CodemodeDeclaredGlobal(global_["name"], global_["spread"]) for global_ in self._start["globals"]
                ],
                store=self._start["store"],
                signal=self._controller.signal,
                timeout_ms=remaining,
                on_started=lambda fingerprint: self._spawn(self._check_fingerprint(fingerprint)),
                on_call=self._on_call,
                on_idle=self._on_idle,
            )
            result = await self._run.result
            if self._failure is not None:
                raise self._failure
            if self._ended is not None:
                return self._ended
            context.throw_if_aborted()
            # Aborted, but not by this call: the host closed the sandbox, for example while shutting down before closing
            # the Harness. That decides nothing about the script: store nothing and end with this invocation, so a later
            # attempt replays it.
            if (
                isinstance(result, CodemodeFailure)
                and result.error.kind == "aborted"
                and not self._controller.signal.aborted
            ):
                if signal is not None and not signal.aborted:
                    await signal.wait()
                context.throw_if_aborted()
                raise RuntimeError("The code mode sandbox was closed while the script ran")
            # A replay that ends where the live run went on is a divergence.
            during = result.during
            # Within the replayed part, the first run included: a rerun without journal entries replays that much too.
            if self._rerun and during is not None and during.seq <= len(self._replayed):
                # The live run went on: it acknowledged that result, or it made a call that later results answer, which
                # the replay never made.
                if during.seq == 0:
                    acknowledged = len(self._replayed) > 0 or self._first_issued is not None
                else:
                    acknowledged = "issuedAfter" in self._replayed[during.seq - 1]
                continued = acknowledged or any(entry["id"] > during.issued for entry in self._replayed[during.seq :])
                if continued:
                    return await self._decide(
                        _error_result(
                            "The script took a different path when resumed after a restart: it ended after result "
                            f"{during.seq}, where it went on before."
                        )
                    )
            return await self._complete(result)
        finally:
            if remove is not None:
                remove()
            # Work still waiting on the gate or a model slot sees that the call ended, instead of waiting forever.
            self._open_gate()
            waiters, self._model_waiters = self._model_waiters, []
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)

    async def _decide(self, decided: ToolExecutionResult) -> ToolExecutionResult:
        """Store an outcome the call decided, and end the VM if it runs; returns the outcome."""
        result = cast("ToolExecutionResult", {**decided, "details": self._details()})

        async def store(tx: Any) -> None:
            run = await tx.doc(CodemodeRunDoc, self._api.task_id)
            if "result" not in run:
                run["result"] = _result_json(result)

        await self._api.commit(store, self._context)
        return result

    def _diverge(self, message: str) -> None:
        """End the call with ``message``: store it as the outcome, then stop the VM with an error the script cannot
        catch."""
        if self._ended is not None or self._failure is not None:
            return
        result = _error_result(f"The script took a different path when resumed after a restart: {message}")
        self._ended = result

        async def end() -> None:
            try:
                await self._decide(result)
            except Exception as error:
                self._fail(error)
                return
            self._controller.abort(RuntimeError(message))

        self._spawn(end())

    @property
    def _stopped(self) -> bool:
        """Whether nothing more may happen: the run has its outcome, or the call decided or failed."""
        return self._ended is not None or self._failure is not None or (self._run is not None and self._run.finished)

    def _fail(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        self._controller.abort(error)

    async def _complete(self, result: CodemodeResult) -> ToolExecutionResult:
        """The script's own outcome, stored with its store writes in one transaction."""
        details = self._details()
        formatted = await format_result(
            result,
            generated_images=self._generated_images,
            wall_ms=self._api.now() - self._start["startedAt"],
            max_output_tokens=self._start["maxOutputTokens"],
            calls=details["calls"],
            limits=_OUTPUT_LIMITS,
            write=self._output_files(),
        )
        if formatted.full_output_path is not None:
            details["fullOutputPath"] = formatted.full_output_path
        tool: ToolExecutionResult = {"output": formatted.content, "isError": not result.ok, "details": details}
        writes = result.store_writes if isinstance(result, CodemodeSuccess) else None

        async def store(tx: Any) -> ToolExecutionResult:
            run = await tx.doc(CodemodeRunDoc, self._api.task_id)
            if "result" in run:
                return _result_from_json(copy_json(run["result"]))
            run["result"] = _result_json(tool)
            if writes is not None and (len(writes.set) > 0 or len(writes.delete) > 0):
                values = (await tx.doc(CodemodeStoreDoc, self._api.conversation_id))["values"]
                for key in writes.delete:
                    values.pop(key, None)
                for key, value in writes.set.items():
                    values[key] = copy_json(value)
            return tool

        return await self._api.commit(store, self._context)

    def _output_files(self) -> OutputFile:
        """Writes the result's files under fixed names into the run's output directory, so a rerun overwrites them."""
        env = self._api.env
        directory: list[asyncio.Future[str]] = []

        async def write(name: str, data: str | bytes) -> OutputFileResult:
            if env is None:
                return {"error": "no execution environment"}
            try:
                if not directory:
                    directory.append(asyncio.ensure_future(self._output_dir(env)))
                path = f"{await directory[0]}/{name}"
                written = await env.write_file(path, data, self._context)
                return {"path": path} if written["ok"] is True else {"error": written["error"].message}
            except Exception as error:
                return {"error": error_message(error)}

        return write

    async def _output_dir(self, env: ExecutionEnv) -> str:
        """The run's output directory: the recorded one, else a new temp directory, recorded before any file is
        written."""

        async def read(tx: Any) -> str | None:
            return (await tx.doc(CodemodeRunDoc, self._api.task_id)).get("outputDir")

        recorded = await self._api.commit(read, self._context)
        if recorded is not None:
            return recorded
        created = await env.create_temp_dir("pi-codemode-", self._context)
        if created["ok"] is False:
            raise RuntimeError(created["error"].message)
        value = created["value"]

        async def record(tx: Any) -> str:
            run = await tx.doc(CodemodeRunDoc, self._api.task_id)
            if "outputDir" not in run:
                run["outputDir"] = value
            return run["outputDir"]

        return await self._api.commit(record, self._context)

    async def _check_fingerprint(self, fingerprint: str) -> None:
        recorded = self._start.get("fingerprint")
        if recorded is not None:
            if recorded != fingerprint:
                self._ended = _error_result(
                    f"The script cannot resume: the sandbox changed since it started ({recorded} -> {fingerprint})."
                )
                try:
                    await self._decide(self._ended)
                except Exception as error:
                    self._fail(error)
                self._controller.abort(RuntimeError("Sandbox fingerprint changed"))
                return
        else:

            async def record(tx: Any) -> None:
                run = await tx.doc(CodemodeRunDoc, self._api.task_id)
                if "start" in run:
                    run["start"]["fingerprint"] = fingerprint

            try:
                await self._api.commit(record, self._context)
            except Exception as error:
                self._fail(error)
                return
        self._open_gate()
        # Nothing was delivered before the gate: start now. A rerun without a journal releases its calls at the
        # script's first acknowledgment instead.
        if self._replaying:
            self._deliver_replayed()
        elif not self._rerun:
            self._spawn(self._release())

    def _on_call(self, made: CodemodeHostCall) -> None:
        if self._stopped:
            return
        call = _as_tool_call(made)
        if call.target == "tool":
            self._started(call)
        self._issued = max(self._issued, call.id)
        index = self._replay_index.get(call.id, -1) if self._replaying else -1
        if index == -1:
            if self._released:
                self._waiting.append(call)
                self._launch_covered()
            else:
                self._held.append(call)
            return
        entry = self._replayed[index]
        recorded = entry.get("global")
        if recorded is not None:
            if (
                call.target != "global"
                or call.name != recorded["name"]
                or not json_equal(call.args, recorded.get("args"))
            ):
                self._diverge(f"call {call.id} is {_describe(call)}, but was a call to {recorded['name']}")
                return
            self._ready[call.id] = cast("CodemodeOutcome", recorded["outcome"])
            self._gate.add_done_callback(lambda _gate: self._deliver_replayed())
            return
        if call.target != "tool":
            self._diverge(f"call {call.id} is {_describe(call)}, but was a tool call")
            return
        self._spawn(self._replay_tool(call))

    async def _replay_tool(self, call: CodemodeHostCall) -> None:
        await self._gate
        # The gate also opens when the call ends; a call that ended admits nothing.
        if self._stopped:
            return
        try:
            outcome = await self._call_tool(call)
        except Exception as error:
            if self._stopped:
                return
            signal = self._context.abort_signal
            if signal is not None and signal.aborted:
                self._fail(error)
            else:
                # The key names another tool or other arguments.
                self._diverge(f"call {call.id} is {_describe(call)}: {error_message(error)}")
            return
        self._ready[call.id] = outcome
        self._deliver_replayed()

    def _deliver_replayed(self) -> None:
        """Replay: deliver journaled results in journal order, each once the VM has issued its call."""
        while self._next_replayed < len(self._replayed) and not self._stopped:
            entry = self._replayed[self._next_replayed]
            outcome = self._ready.get(entry["id"])
            if entry["id"] > self._issued or outcome is None:
                return
            del self._ready[entry["id"]]
            self._next_replayed += 1
            assert self._run is not None
            self._run.deliver(self._next_replayed, entry["at"], entry["id"], outcome)

    def _on_idle(self, seq: int, issued: int) -> None:
        if self._stopped:
            return
        # A count an earlier attempt recorded must match; a missing one is recorded now.
        if self._rerun and seq <= len(self._replayed):
            recorded = self._first_issued if seq == 0 else self._replayed[seq - 1].get("issuedAfter")
            if recorded is not None and recorded != issued:
                self._diverge(
                    f"its first run made {issued} calls, where it had made {recorded}"
                    if seq == 0
                    else f"after result {seq} it had made {issued} calls, where it had made {recorded}"
                )
                return
        self._acks[seq] = issued
        if not self._replaying:
            # A rerun without a journal: every call an earlier attempt could have admitted is made by now.
            if seq == 0 and self._rerun:
                self._spawn(self._release())
            # Committed at once when the script has processed everything delivered: calls it made wait for it, and a
            # script waiting on a long call is where a restart usually finds it. Otherwise with the next commit; the VM
            # catches up without waiting on anything.
            if seq == self._journal_length:
                self._spawn(self._pump())
            return
        # Caught up: the VM processed everything delivered and waits.
        if seq != self._next_replayed:
            return
        if seq < len(self._replayed):
            following = self._replayed[seq]
            if following["id"] > issued:
                self._diverge(f"it waits without making call {following['id']}, which it made before")
            return
        # The last replayed result is acknowledged: continue live, admitting the calls the journal did not have once
        # their count is committed.
        self._replaying = False
        self._spawn(self._pump())
        self._spawn(self._release())

    async def _release(self) -> None:
        """Let the held calls launch, and later live calls, each once its issued count is committed.

        On a rerun, an earlier attempt may have admitted some of them under their keys, as other calls: when the replay
        releases (idle after the last journaled result, or after its first run without a journal) its counts matched
        every recorded one, so it has made every call an earlier attempt could have admitted. All held calls are
        checked first, whatever their target, and a divergence launches none. Calls made after this are new.
        """
        await self._gate
        if self._stopped or self._released:
            return
        calls, self._held = self._held, []
        if self._rerun:
            for call in calls:
                try:
                    other = await self._earlier(call)
                except Exception as error:
                    self._fail(error)
                    return
                if self._stopped:
                    return
                if other is not None:
                    self._diverge(f"call {call.id} is {_describe(call)}, but was {other}")
                    return
        self._released = True
        held, self._held = self._held, []
        self._waiting.extend([*calls, *held])
        self._launch_covered()

    def _launch_covered(self) -> None:
        """Launch the waiting calls whose issued count is committed."""
        if not self._released or self._stopped:
            return
        covered = [call for call in self._waiting if call.id <= self._durable_issued]
        if len(covered) == 0:
            return
        self._waiting = [call for call in self._waiting if call.id > self._durable_issued]
        for call in covered:
            self._launch(call)

    async def _earlier(self, call: CodemodeHostCall) -> str | None:
        """The call an earlier attempt made under this call's key, when it is another one: another target, tool, or
        arguments."""
        api, context = self._api, self._context
        state = await api.snapshot(NestedCallDoc, api.task_id, f"c{call.id}", context)
        task_id = None if state is None else state.get("taskId")
        if task_id is None:
            return None
        record = await api.get_task(task_id, context)
        task_input: Any = None if record is None else record["input"]
        if not isinstance(task_input, dict) or task_input.get("kind") != "nested":  # pyright: ignore[reportUnknownMemberType]
            return None
        made: dict[str, Any] = cast("dict[str, Any]", task_input)["call"]
        if (
            call.target == "tool"
            and made["name"] == call.name
            and json_equal(made["arguments"], call.args if call.args is not None else {})
        ):
            return None
        return f"tools.{made['name']}({_compact(made['arguments'])})"

    def _launch(self, call: CodemodeHostCall) -> None:
        """Run one live call and journal its result."""
        if self._stopped:
            return
        self._spawn(self._run_live(call))

    async def _run_live(self, call: CodemodeHostCall) -> None:
        recorded: JournalGlobal | None = None
        try:
            if call.target == "tool":
                outcome = await self._call_tool(call)
            else:
                outcome = await self._call_global(call)
                recorded = {"name": call.name, "outcome": cast("dict[str, Any]", outcome)}
                if call.args is not None:
                    recorded["args"] = call.args
        except Exception as error:
            # Stopped meanwhile: the outcome is decided, or a waiting model call gave up.
            if self._stopped:
                return
            signal = self._context.abort_signal
            if signal is not None and signal.aborted:
                self._fail(error)
            else:
                # A key an earlier attempt admitted with another tool or other arguments: this attempt took another path.
                self._diverge(f"call {call.id} is {_describe(call)}: {error_message(error)}")
            return
        # A result after the outcome is decided has nowhere to go, and the call's API has ended.
        if self._stopped:
            return
        self._queue.append(_Queued(call.id, self._api.now(), outcome, recorded))
        await self._pump()

    async def _call_tool(self, call: CodemodeHostCall) -> CodemodeOutcome:
        # Passed as the script gave them; validation reports arguments that are not an object.
        args: Any = call.args if call.args is not None else {}
        model = call.name.startswith(CODEMODE_INTERNAL_PREFIX)
        if model:
            await self._model_slot()
            # The call may have stopped while this one waited: a finished script admits no new paid call.
            if self._stopped:
                self._release_model_slot()
                raise RuntimeError("The script has ended")
        try:
            result = await self._api.execute_tool(call.name, args, self._context, key=f"c{call.id}")
        finally:
            if model:
                self._release_model_slot()
        if call.name == GENERATE_IMAGES_TOOL:
            self._generated_images += _count_images(result.get("structuredOutput"))
        structured = any(tool["name"] == call.name and tool.get("structured") is True for tool in self._start["tools"])
        outcome = _outcome_of(call.name, structured, result)
        self._finish_row(call.id, outcome, result)
        return outcome

    def _started(self, call: CodemodeHostCall) -> None:
        """The script made a tool call: its row appears, running, also while it waits to launch."""
        if call.id in self._rows:
            return
        args = "" if call.args is None else _compact(call.args)
        name = (
            call.name[len(CODEMODE_INTERNAL_PREFIX) :] if call.name.startswith(CODEMODE_INTERNAL_PREFIX) else call.name
        )
        self._rows[call.id] = {
            "id": call.id,
            "name": name,
            "args": _cut(args, _ARGS_PREVIEW_CHARS),
            "status": "running",
        }

    def _finish_row(self, id: int, outcome: CodemodeOutcome, result: NestedToolExecutionResult) -> None:
        """The call ended: an error result is an error row, also when the script got its value (bash's nonzero exit)."""
        row = self._rows.get(id)
        if row is None:
            return
        row["status"] = "error" if result["isError"] else "ok"
        if "durationMs" in result:
            row["durationMs"] = result["durationMs"]
        errors = [diagnostic["message"] for diagnostic in result["diagnostics"] if diagnostic["severity"] == "error"]
        error = outcome["message"] if outcome["ok"] is False else "\n".join(errors)
        if result["isError"] and error != "":
            row["error"] = _cut(error, _ERROR_PREVIEW_CHARS)

    def _details(self) -> CodemodeToolDetails:
        """The rows in call order; calls still running when the script ended were cancelled."""
        calls: list[CodemodeNestedCall] = []
        for id in sorted(self._rows):
            row: CodemodeNestedCall = {**self._rows[id]}
            if row["status"] == "running":
                row["status"] = "cancelled"
            calls.append(row)
        return {"calls": calls}

    async def _call_global(self, call: CodemodeHostCall) -> CodemodeOutcome:
        discovery = call.name in DISCOVERY_GLOBALS
        catalog = call.name in CATALOG_GLOBALS
        global_ = (
            None
            if discovery or catalog
            else next((candidate for candidate in self._options.globals if candidate.name == call.name), None)
        )
        if not discovery and not catalog and global_ is None:
            return {"ok": False, "message": f"Unknown global {call.name}"}
        try:
            # `None` stays `None`: the journal omits the value, and a replay resolves with `None` again.
            if discovery:
                value = self._discover(call)
            elif catalog:
                value = await catalog_call(call.name, call.args, self._api.models, self._context)
            else:
                assert global_ is not None
                value = await global_.execute(call.args, self._api, self._context)
            return _ok(value)
        except Exception as error:
            signal = self._context.abort_signal
            if signal is not None and signal.aborted:
                raise
            return {"ok": False, "message": error_message(error)}

    async def _model_slot(self) -> None:
        if self._model_calls >= MAX_CONCURRENT_MODEL_CALLS:
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._model_waiters.append(waiter)
            # A released slot passes to the first waiter without being freed, so no later call takes it meanwhile.
            await waiter
            return
        self._model_calls += 1

    def _release_model_slot(self) -> None:
        while self._model_waiters:
            waiter = self._model_waiters.pop(0)
            if not waiter.done():
                waiter.set_result(None)
                return
        self._model_calls -= 1

    def _discover(self, call: CodemodeHostCall) -> Any:
        """``search_tools()``, ``describe_tool()``, ``describe_namespace()``: answered from the start data."""
        namespaces = self._start["namespaces"]
        tools = [tool for tool in self._start["tools"] if tool.get("hidden") is not True]

        def namespace_of_tool(name: str) -> StartNamespace | None:
            return next((namespace for namespace in namespaces if name in namespace["tools"]), None)

        def matches_namespace(namespace: StartNamespace, query: str) -> bool:
            return namespace["name"] == query or to_codemode_identifier(namespace["name"]) == to_codemode_identifier(
                query
            )

        if call.name == "describe_tool":
            (name,) = bind_arguments(call.args, ("name",))
            if not isinstance(name, str):
                raise RuntimeError("describe_tool() expects a tool name")
            return next((tool["description"] for tool in tools if tool["name"] == name or tool["jsName"] == name), None)
        if call.name == "describe_namespace":
            (name,) = bind_arguments(call.args, ("name",))
            if not isinstance(name, str):
                raise RuntimeError("describe_namespace() expects a namespace name")
            found = next((candidate for candidate in namespaces if matches_namespace(candidate, name)), None)
            if found is None:
                return None
            described: dict[str, Any] = {"name": found["name"]}
            if "description" in found:
                described["description"] = found["description"]
            if "instructions" in found:
                described["instructions"] = found["instructions"]
            described["tools"] = [to_codemode_identifier(tool) for tool in found["tools"]]
            return described
        # search_tools(query, limit=8, namespace=None): the options as keyword arguments or one dict (upstream's).
        values: list[Any] = list(call.args) if isinstance(call.args, list) else []
        query = values[0] if len(values) > 0 else None
        options: Any = values[1] if len(values) > 1 else None
        if len(values) == 1 and isinstance(query, dict):
            options = query
            query = cast("dict[str, Any]", query).get("query")
        if not isinstance(query, str):
            raise RuntimeError("search_tools() expects a query string")
        options_dict: dict[str, Any] = cast("dict[str, Any]", options) if isinstance(options, dict) else {}
        limit = options_dict.get("limit")
        if limit is None:
            limit = _DEFAULT_SEARCH_LIMIT
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int | float)
            or (isinstance(limit, float) and not limit.is_integer())
            or limit <= 0
        ):
            raise RuntimeError("search_tools() limit must be a positive integer")
        namespace = options_dict.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise RuntimeError("search_tools() namespace must be a string")
        documents: list[SearchDocument] = []
        for tool in tools:
            own = namespace_of_tool(tool["name"])
            if isinstance(namespace, str) and (own is None or not matches_namespace(own, namespace)):
                continue
            text = [
                tool["name"],
                tool["name"].replace("_", " "),
                tool["description"],
                own["name"] if own is not None else "",
                (own.get("description") or "") if own is not None else "",
            ]
            documents.append({"name": tool["name"], "text": " ".join(text)})
        ranked = rank_documents(query, documents, int(limit))
        by_name = {tool["name"]: tool for tool in tools}
        return [{"name": by_name[name]["jsName"], "description": by_name[name]["description"]} for name in ranked]

    async def _pump(self) -> None:
        """Journal finished calls and acknowledged counts in one commit, then deliver the calls and launch the calls
        the counts cover; one commit in flight at a time, so what arrives meanwhile goes into the next."""
        if self._committing or self._stopped:
            return
        if len(self._queue) == 0 and len(self._acks) == 0:
            return
        self._committing = True
        batch, self._queue = self._queue, []
        acks = list(self._acks.items())
        self._acks.clear()

        async def journal_change(tx: Any) -> None:
            journal = await tx.doc(CodemodeJournalDoc, self._api.task_id)
            for seq, issued in acks:
                if seq == 0:
                    if "firstIssued" not in journal:
                        journal["firstIssued"] = issued
                elif seq - 1 < len(journal["entries"]):
                    entry = journal["entries"][seq - 1]
                    if "issuedAfter" not in entry:
                        entry["issuedAfter"] = issued
            for item in batch:
                entry: dict[str, Any] = {"id": item.id, "at": item.at}
                if item.global_ is not None:
                    entry["global"] = item.global_
                journal["entries"].append(copy_json(entry))

        try:
            await self._api.commit(journal_change, self._context)
        except Exception as error:
            self._fail(error)
            return
        for _seq, issued in acks:
            self._durable_issued = max(self._durable_issued, issued)
        assert self._run is not None
        for item in batch:
            self._journal_length += 1
            if not self._run.finished:
                self._run.deliver(self._journal_length, item.at, item.id, item.outcome)
        self._launch_covered()
        self._committing = False
        # What arrived meanwhile: finished calls, or the count of a VM that has caught up. Counts of results the VM is
        # still working through wait for the next journal commit or for it to catch up (`_on_idle`), as upstream's
        # rule says; upstream commits them here too, which with Monty's slower round trips costs a commit per few
        # results.
        if len(self._queue) > 0 or self._journal_length in self._acks:
            await self._pump()


def _as_tool_call(call: CodemodeHostCall) -> CodemodeHostCall:
    """A script's call of ``models.classify()`` or ``models.generate_images()`` as the call of its internal tool."""
    tool = MODEL_CALLS.get(call.name) if call.target == "global" else None
    if tool is None:
        return call
    model, context = bind_arguments(call.args, ("model", "context"))
    return CodemodeHostCall(call.id, "tool", tool, {"model": model, "context": context})


def _count_images(value: Any) -> int:
    """Image blocks in a ``models.generate_images()`` result."""
    output = value.get("output") if isinstance(value, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(output, list):
        return 0
    return sum(
        1
        for item in cast("list[Any]", output)
        if isinstance(item, dict) and item.get("type") == "image"  # pyright: ignore[reportUnknownMemberType]
    )


def _describe(call: CodemodeHostCall) -> str:
    args = "" if call.args is None else _compact(call.args)
    return f"{'tools.' if call.target == 'tool' else ''}{call.name}({args})"
