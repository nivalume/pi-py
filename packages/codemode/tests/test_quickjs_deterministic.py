"""Port of upstream ``test/deterministic.test.ts`` for JavaScript scripts, plus replay through a journal as pi-durable's
codemode does it.

``start()`` runs a script deterministically with the caller answering its calls, so a durable host can replay it after
a restart. Here on :class:`CodemodeSandbox` (QuickJS on wasmtime), which runs upstream's QuickJS build and prelude:
pinned values were recorded from upstream ``@earendil-works/pi-codemode`` 1.3.0 on Node, so a run started by either
replays on the other.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from codemode_helpers import wait_for

from pi_ai.abort import AbortController
from pi_codemode import (
    CodemodeDeclaredGlobal,
    CodemodeDeclaredTool,
    CodemodeDuring,
    CodemodeFailure,
    CodemodeHostCall,
    CodemodeOutcome,
    CodemodeResult,
    CodemodeRun,
    CodemodeSandbox,
    CodemodeSandboxBase,
    CodemodeSuccess,
    CodemodeTextOutput,
)

STARTED_AT = dt.datetime(2026, 1, 15, 10, 0, 0, tzinfo=dt.UTC).timestamp() * 1000
#: The first `Math.random()` with seed 1, as upstream's sandbox gives it.
PINNED_FIRST_RANDOM = 0.34227450605335275
#: The first two `Math.random()` values with seed 0, as upstream's sandbox gives them.
PINNED_SEED_ZERO = [0.6711372530266764, 0.7258461452833667]


@dataclass
class Recorded:
    run: CodemodeRun
    calls: list[CodemodeHostCall] = field(default_factory=list)
    idles: list[tuple[int, int]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    fingerprint: str | None = None


def start(
    sandbox: CodemodeSandboxBase,
    code: str,
    *,
    answer: Callable[[CodemodeHostCall], CodemodeOutcome] | None = None,
    **options: Any,
) -> Recorded:
    """Start ``code`` with tools ``echo`` and ``wait``. Every call but ``wait`` is answered soon after, by
    default with its argument, as result number 1, 2, ... with the clock at ``STARTED_AT + 1000 * seq``;
    ``wait`` is left for the test to answer."""
    loop = asyncio.get_running_loop()
    seq = 0

    def echo(call: CodemodeHostCall) -> CodemodeOutcome:
        return {"ok": True, "value": call.args}

    respond = answer or echo
    recorded: Recorded

    def on_started(fingerprint: str) -> None:
        recorded.fingerprint = fingerprint
        recorded.events.append("started")

    def deliver(call: CodemodeHostCall) -> None:
        nonlocal seq
        seq += 1
        recorded.run.deliver(seq, STARTED_AT + 1000 * seq, call.id, respond(call))

    def on_call(call: CodemodeHostCall) -> None:
        recorded.calls.append(call)
        recorded.events.append(f"call {call.id}")
        if call.name != "wait":
            loop.call_soon(deliver, call)

    def on_idle(seq: int, issued: int) -> None:
        recorded.idles.append((seq, issued))
        recorded.events.append(f"idle {seq}")

    settings: dict[str, Any] = {
        "started_at": STARTED_AT,
        "seed": 1,
        "tools": [CodemodeDeclaredTool("echo"), CodemodeDeclaredTool("wait")],
        "timeout_ms": 20_000,
        **options,
    }
    run = sandbox.start(code, on_started=on_started, on_call=on_call, on_idle=on_idle, **settings)
    recorded = Recorded(run)
    return recorded


@pytest.fixture
async def sandbox(new_quickjs_sandbox: Callable[..., CodemodeSandbox]) -> CodemodeSandbox:
    return new_quickjs_sandbox()


def timeless(result: CodemodeResult) -> CodemodeResult:
    """Call durations are wall-clock time; everything else is the same."""
    return replace(result, calls=[replace(call, duration_ms=0) for call in result.calls])


def value_of(result: CodemodeResult) -> Any:
    assert isinstance(result, CodemodeSuccess), result
    return result.value


async def test_makes_the_same_calls_and_output_for_the_same_start_seed_and_results(sandbox: CodemodeSandbox) -> None:
    code = """
const values = [Date.now(), performance.now(), Math.random(), Math.random(), new Date().toISOString()];
const first = await tools.echo(values);
text(JSON.stringify([Date.now(), performance.now(), Math.random()]));
return await tools.echo(first.concat(Math.random()));
"""
    one = start(sandbox, code)
    two = start(sandbox, code)
    a, b = await asyncio.gather(one.run.result, two.run.result)
    assert a.ok, a
    assert timeless(a) == timeless(b)
    assert one.calls == two.calls
    # The clock: started_at, then the time of the result being processed; performance.now() counts from started_at.
    args = one.calls[0].args
    assert isinstance(args, list)
    assert args[:2] == [STARTED_AT, 0]
    assert args[4] == "2026-01-15T10:00:00.000Z"
    assert isinstance(a.output[0], CodemodeTextOutput)
    assert json.loads(a.output[0].text)[:2] == [STARTED_AT + 1000, 1000]
    # Pinned, bit for bit: replays depend on it, so a change here must change the fingerprint too.
    assert args[2] == PINNED_FIRST_RANDOM
    assert value_of(a) == [STARTED_AT, 0, PINNED_FIRST_RANDOM, 0.4620546081390098, args[4], 0.056110785474501323]
    assert json.loads(a.output[0].text)[2] == 0.5807179129204683

    # The seed decides the random values, not the start time.
    later = start(sandbox, code, started_at=STARTED_AT + 5000)
    reseeded = start(sandbox, code, seed=2)
    await asyncio.gather(later.run.result, reseeded.run.result)
    assert isinstance(later.calls[0].args, list) and later.calls[0].args[2] == PINNED_FIRST_RANDOM
    assert isinstance(reseeded.calls[0].args, list) and reseeded.calls[0].args[2] != PINNED_FIRST_RANDOM
    zero = await start(sandbox, "return [Math.random(), Math.random()]", seed=0).run.result
    assert value_of(zero) == PINNED_SEED_ZERO


async def test_dates_read_the_virtual_clock_in_utc(sandbox: CodemodeSandbox) -> None:
    result = await start(
        sandbox,
        """
const local = new Date(2026, 0, 15, 10, 30);
return [
    new Date(0).getHours(), new Date(0).getTimezoneOffset(), typeof WeakRef, typeof FinalizationRegistry,
    new Date().getHours(), String(new Date(0)), new Date(0).toLocaleString(), local.toISOString(),
    new Date("2026-01-15T10:00:00").getTime() === Date.now(), Date.parse("2026-01-15") === Date.UTC(2026, 0, 15),
    Date(), new Date(0) instanceof Date, typeof Date.now
];
""",
    ).run.result
    assert value_of(result) == [
        0,
        0,
        "undefined",
        "undefined",
        10,
        "Thu Jan 01 1970 00:00:00 GMT+0000",
        "01/01/1970, 12:00:00 AM",
        "2026-01-15T10:30:00.000Z",
        True,
        True,
        "Thu Jan 15 2026 10:00:00 GMT+0000",
        True,
        "function",
    ]


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset() is POSIX only")
async def test_runs_in_utc_whatever_the_host_timezone(
    sandbox: CodemodeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        code = "return [new Date(0).getHours(), new Date(0).getTimezoneOffset(), typeof WeakRef, typeof FinalizationRegistry]"
        assert value_of(await start(sandbox, code).run.result) == [0, 0, "undefined", "undefined"]
        # execute() is unchanged: the host's timezone and every intrinsic.
        assert value_of(await sandbox.execute(code)) == [5, -330, "function", "function"]
    finally:
        monkeypatch.undo()
        time.tzset()


async def test_applies_results_in_seq_order_whatever_order_they_are_delivered_in(sandbox: CodemodeSandbox) -> None:
    recorded = start(
        sandbox,
        """const order = [];
await Promise.all([tools.wait("a").then(() => order.push("a")), tools.wait("b").then(() => order.push("b"))]);
return order;""",
    )
    await wait_for(lambda: len(recorded.calls) == 2 and len(recorded.idles) == 1)
    a, b = recorded.calls
    # b is result 1 and a result 2, but a arrives first.
    recorded.run.deliver(2, STARTED_AT + 2000, a.id, {"ok": True})
    recorded.run.deliver(1, STARTED_AT + 1000, b.id, {"ok": True})
    assert value_of(await recorded.run.result) == ["b", "a"]
    # No idle after the last result: the script finished in that drain, and the host stopped the VM.
    assert recorded.idles == [(0, 2), (1, 2)]


async def test_reports_the_fingerprint_before_any_call_and_the_calls_issued_after_each_result(
    new_quickjs_sandbox: Callable[..., CodemodeSandbox],
) -> None:
    sandbox = new_quickjs_sandbox(memory_limit_bytes=64 * 1024 * 1024)
    recorded = start(sandbox, "await tools.echo(1); await Promise.all([tools.echo(2), tools.echo(3)]);")
    assert (await recorded.run.result).ok
    assert recorded.events[:3] == ["started", "call 1", "idle 0"]
    assert recorded.idles == [(0, 1), (1, 3), (2, 3)]
    assert [(call.id, call.target, call.name, call.args) for call in recorded.calls] == [
        (1, "tool", "echo", 1),
        (2, "tool", "echo", 2),
        (3, "tool", "echo", 3),
    ]
    # Upstream's format and values, so a run started by upstream's sandbox on the same build replays here.
    assert recorded.fingerprint == (
        "quickjs-wasi 3.6.2, quickjs 0.17.0, prelude 8c6c9242, deterministic 1, stack 524288, heap 67108864, budget none"
    )

    same = start(sandbox, "return 1")
    other = start(new_quickjs_sandbox(memory_limit_bytes=32 * 1024 * 1024), "return 1")
    budget = start(new_quickjs_sandbox(interrupt_budget=1000), "return 1")
    stack = start(new_quickjs_sandbox(max_stack_size=256 * 1024), "return 1")
    await asyncio.gather(same.run.result, other.run.result, budget.run.result, stack.run.result)
    assert same.fingerprint == recorded.fingerprint
    assert other.fingerprint is not None and other.fingerprint.endswith("heap 33554432, budget none")
    assert budget.fingerprint is not None and budget.fingerprint.endswith("heap none, budget 1000")
    assert stack.fingerprint is not None and ", stack 262144, " in stack.fingerprint


async def test_numbers_calls_to_globals_with_tools_and_rejects_with_errors(sandbox: CodemodeSandbox) -> None:
    def answer(call: CodemodeHostCall) -> CodemodeOutcome:
        if call.args == "fail":
            return {"ok": False, "message": "no"}
        return {"ok": True, "value": ["found"] if call.name == "search" else call.args}

    recorded = start(
        sandbox,
        """const found = await search("x", 2);
try { await tools.echo("fail"); } catch (error) { text(error.message); }
return found;""",
        globals=[CodemodeDeclaredGlobal("search", spread=True)],
        answer=answer,
    )
    result = await recorded.run.result
    assert recorded.calls[0] == CodemodeHostCall(1, "global", "search", ["x", 2])
    assert isinstance(result, CodemodeSuccess)
    assert (result.value, result.output) == (["found"], [CodemodeTextOutput("no")])
    assert (recorded.calls[1].id, recorded.calls[1].target, recorded.calls[1].name) == (2, "tool", "echo")
    # Calls to globals count as issued calls too.
    assert recorded.idles[:2] == [(0, 1), (1, 2)]
    assert [(call.name, call.status) for call in result.calls] == [("echo", "error")]


async def test_keeps_the_clock_monotonic_and_rejects_a_clock_value_that_is_not_finite(sandbox: CodemodeSandbox) -> None:
    recorded = start(
        sandbox, "const before = Date.now(); await tools.wait(1); return [before, Date.now(), performance.now()]"
    )
    await wait_for(lambda: len(recorded.calls) == 1)
    with pytest.raises(ValueError, match="not a finite number"):
        recorded.run.deliver(1, math.nan, 1, {"ok": True})
    # A resumed host whose clock is behind the run's start.
    recorded.run.deliver(1, STARTED_AT - 2000, 1, {"ok": True})
    assert value_of(await recorded.run.result) == [STARTED_AT, STARTED_AT, 0]


async def test_reports_the_result_the_sandbox_was_processing_when_the_script_ended(sandbox: CodemodeSandbox) -> None:
    returned = start(sandbox, "await tools.echo(1); return await tools.echo(2)")
    thrown = start(
        sandbox, "await tools.echo(1); await Promise.all([tools.echo(2), tools.echo(3)]); throw new Error('x')"
    )
    first = start(sandbox, "return 1")
    returned_result = await returned.run.result
    assert (value_of(returned_result), returned_result.during) == (2, CodemodeDuring(2, 2))
    thrown_result = await thrown.run.result
    assert isinstance(thrown_result, CodemodeFailure) and thrown_result.during == CodemodeDuring(3, 3)
    first_result = await first.run.result
    assert first_result.ok and first_result.during == CodemodeDuring(0, 0)
    assert (await sandbox.execute("return 1")).during is None


async def test_leaves_a_call_waiting_when_its_answer_cannot_be_serialized(sandbox: CodemodeSandbox) -> None:
    recorded = start(sandbox, "return await tools.wait(1)")
    await wait_for(lambda: len(recorded.calls) == 1)
    with pytest.raises(TypeError):
        recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": {1}})  # type: ignore[typeddict-item]
    with pytest.raises(TypeError):
        recorded.run.deliver(1, STARTED_AT, 1, {"ok": False, "message": "x", "properties": {"big": {1}}})  # type: ignore[dict-item]
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": "fine"})
    assert value_of(await recorded.run.result) == "fine"


async def test_rejections_carry_the_answers_properties(sandbox: CodemodeSandbox) -> None:
    # Unlike Monty's exceptions, a JavaScript Error carries the answer's properties, except name, message and stack.
    recorded = start(
        sandbox,
        "try { await tools.echo(1); } catch (e) { return [e instanceof Error, e.message, e.name, e.structuredOutput]; }",
        answer=lambda _call: {
            "ok": False,
            "message": "denied",
            "properties": {"structuredOutput": {"code": 7}, "name": "Ignored", "message": "ignored"},
        },
    )
    assert value_of(await recorded.run.result) == [True, "denied", "Error", {"code": 7}]


async def test_rejects_rejection_properties_that_are_not_an_object_and_tools_sharing_a_script_name(
    sandbox: CodemodeSandbox,
) -> None:
    recorded = start(sandbox, "return await tools.wait(1)")
    await wait_for(lambda: len(recorded.calls) == 1)
    for properties in (None, [1]):
        with pytest.raises(ValueError, match="not an object"):
            recorded.run.deliver(1, STARTED_AT, 1, {"ok": False, "message": "x", "properties": properties})  # type: ignore[typeddict-item]
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": 1})
    assert value_of(await recorded.run.result) == 1
    with pytest.raises(ValueError, match="script names must be unique"):
        sandbox.start(
            "",
            started_at=STARTED_AT,
            seed=1,
            tools=[CodemodeDeclaredTool("a-b"), CodemodeDeclaredTool("a_b")],
            on_call=lambda _call: None,
        )


def test_validates_the_start_data(sandbox: CodemodeSandbox) -> None:
    for started_at in (-1, math.inf, math.nan):
        with pytest.raises(ValueError, match="started_at"):
            sandbox.start("", started_at=started_at, seed=1, on_call=lambda _call: None)
    for seed in (-1, 2**32, 1.5):
        with pytest.raises(ValueError, match="seed"):
            sandbox.start("", started_at=0, seed=seed, on_call=lambda _call: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Tool names must be unique"):
        sandbox.start("", started_at=0, seed=0, tools=[CodemodeDeclaredTool("a")] * 2, on_call=lambda _call: None)
    with pytest.raises(ValueError, match='Invalid tool script name "a-b"'):
        sandbox.start("", started_at=0, seed=0, tools=[CodemodeDeclaredTool("a", js_name="a-b")], on_call=print)


async def test_uses_the_script_names_the_run_passes(sandbox: CodemodeSandbox) -> None:
    recorded = start(
        sandbox,
        "return [typeof tools.renamed, Object.keys(tools), ALL_TOOLS.map((tool) => tool.name), 'class' in tools]",
        tools=[CodemodeDeclaredTool("my-tool", js_name="renamed"), CodemodeDeclaredTool("class")],
    )
    # JavaScript identifiers keep keywords as they are: `tools.class` works.
    assert value_of(await recorded.run.result) == [
        "function",
        ["renamed", "my-tool", "class"],
        ["renamed", "class"],
        True,
    ]


async def test_ends_with_an_error_the_script_cannot_catch_when_the_caller_aborts(sandbox: CodemodeSandbox) -> None:
    controller = AbortController()
    recorded = start(
        sandbox,
        'try { await tools.wait(1); } catch { text("caught"); }\ntext("after");',
        signal=controller.signal,
    )
    await wait_for(lambda: len(recorded.calls) == 1)
    assert recorded.run.finished is False
    controller.abort(RuntimeError("diverged"))
    result = await recorded.run.result
    assert isinstance(result, CodemodeFailure)
    assert (result.error.kind, result.error.message, result.output) == ("aborted", "diverged", [])
    assert recorded.run.finished is True
    # A finished run drops later answers.
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True})


async def test_is_finished_once_it_has_its_outcome(sandbox: CodemodeSandbox) -> None:
    recorded = start(sandbox, "await tools.wait(1); return 2")
    await wait_for(lambda: len(recorded.calls) == 1)
    assert recorded.run.finished is False
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True})
    assert value_of(await recorded.run.result) == 2
    assert recorded.run.finished is True


async def test_rejects_a_call_that_is_not_waiting_and_a_result_number_used_twice(sandbox: CodemodeSandbox) -> None:
    recorded = start(sandbox, "await Promise.all([tools.wait(1), tools.wait(2)])")
    await wait_for(lambda: len(recorded.calls) == 2)
    run = recorded.run
    with pytest.raises(ValueError, match="Call 99 is not waiting"):
        run.deliver(1, STARTED_AT, 99, {"ok": True})
    run.deliver(1, STARTED_AT, 1, {"ok": True})
    with pytest.raises(ValueError, match="used once"):
        run.deliver(1, STARTED_AT, 2, {"ok": True})
    with pytest.raises(ValueError, match="used once"):
        run.deliver(0, STARTED_AT, 2, {"ok": True})
    run.deliver(2, STARTED_AT, 2, {"ok": True})
    assert (await run.result).ok


async def test_execute_reads_the_real_clock_and_real_randomness(sandbox: CodemodeSandbox) -> None:
    before = time.time() * 1000
    code = "return [Date.now(), Math.random(), typeof WeakRef, typeof FinalizationRegistry]"
    first, second = value_of(await sandbox.execute(code)), value_of(await sandbox.execute(code))
    now, random_value, weak_ref, finalization = first
    assert before - 1 <= now <= time.time() * 1000 + 1
    assert random_value != second[1]
    assert (weak_ref, finalization) == ("function", "function")


async def test_matches_upstreams_sandbox_on_node(sandbox: CodemodeSandbox) -> None:
    # Recorded from upstream @earendil-works/pi-codemode 1.3.0 (quickjs-wasi 3.6.2) on Node: the same script, start,
    # seed and answers give the same fingerprint, calls, idles and value, bit for bit.
    expected = json.loads(
        '{"events":[["started","quickjs-wasi 3.6.2, quickjs 0.17.0, prelude 8c6c9242, deterministic 1, stack 524288, '
        'heap none, budget none"],["call",1,"read",{"path":"x"}],["idle",0,1],["call",2,"read",{"path":"y"}],'
        '["call",3,"read",{"path":"z"}],["idle",1,3],["idle",2,3]],"value":["R1",["R2","R3"],1768471203000,3000,'
        '0.3898226593579639,0.11240668117012764,"Thu Jan 15 2026 10:00:03 GMT+0000","2020-02-03T04:05:00.000Z",'
        '"undefined","undefined",0],"during":{"seq":3,"issued":3}}'
    )
    code = (
        'const a = await tools.read({path: "x"}); const b = await Promise.all([tools.read({path: "y"}), '
        'tools.read({path: "z"})]);\n'
        "return [a, b, Date.now(), performance.now(), Math.random(), Math.random(), new Date().toString(), "
        "new Date(2020, 1, 3, 4, 5).toISOString(), typeof WeakRef, typeof FinalizationRegistry, "
        "new Date().getTimezoneOffset()]"
    )
    loop = asyncio.get_running_loop()
    events: list[Any] = []
    seq = 0

    def on_call(call: CodemodeHostCall) -> None:
        nonlocal seq
        events.append(["call", call.id, call.name, call.args])
        seq += 1
        answered = seq
        loop.call_soon(
            lambda: run.deliver(
                answered, 1768471200000 + 1000 * answered, call.id, {"ok": True, "value": f"R{answered}"}
            )
        )

    run = sandbox.start(
        code,
        started_at=1768471200000,
        seed=7,
        tools=[CodemodeDeclaredTool("read")],
        on_started=lambda fingerprint: events.append(["started", fingerprint]),
        on_call=on_call,
        on_idle=lambda seq, issued: events.append(["idle", seq, issued]),
    )
    result = await run.result
    assert isinstance(result, CodemodeSuccess) and result.during is not None
    assert {
        "events": events,
        "value": result.value,
        "during": {"seq": result.during.seq, "issued": result.during.issued},
    } == expected


# ---------------------------------------------------------------------------
# Replay through a journal, the way pi-durable's codemode uses start() (a reduced copy of its logic).
# ---------------------------------------------------------------------------


@dataclass
class JournalEntry:
    #: The sandbox's call ID; its result is the entry's position + 1 (``seq``).
    id: int
    #: The clock while the sandbox processes this result.
    at: float
    outcome: CodemodeOutcome
    call: tuple[str, str, str]
    #: Calls issued after processing this result.
    issued_after: int | None = None


class ReplayHost:
    """Runs a script live, journaling each result before delivering it, or replays a journal first."""

    def __init__(self, sandbox: CodemodeSandboxBase, answer: Callable[[CodemodeHostCall], Any]) -> None:
        self.sandbox = sandbox
        self.answer = answer
        self.live_calls: list[CodemodeHostCall] = []
        self.divergence: str | None = None
        self.tasks: set[asyncio.Task[None]] = set()

    async def run(self, code: str, journal: list[JournalEntry], *, seed: int = 7) -> CodemodeResult:
        controller = AbortController()
        replayed = list(journal)
        by_id = {entry.id: entry for entry in replayed}
        loop = asyncio.get_running_loop()
        state = {"next": 0, "issued": 0}
        run: CodemodeRun

        def diverge(message: str) -> None:
            if self.divergence is None:
                self.divergence = f"The script took a different path when resumed after a restart: {message}"
                controller.abort(RuntimeError(message))

        def deliver_replayed() -> None:
            while state["next"] < len(replayed):
                entry = replayed[state["next"]]
                if entry.id > state["issued"]:
                    return
                state["next"] += 1
                run.deliver(state["next"], entry.at, entry.id, entry.outcome)

        async def live(call: CodemodeHostCall) -> None:
            value = await self.answer(call)
            outcome: CodemodeOutcome = {"ok": True, "value": value}
            journal.append(JournalEntry(call.id, STARTED_AT + 1000 * (len(journal) + 1), outcome, _describe(call)))
            if not run.finished:
                run.deliver(len(journal), journal[-1].at, call.id, outcome)

        def on_call(call: CodemodeHostCall) -> None:
            state["issued"] = max(state["issued"], call.id)
            entry = by_id.get(call.id)
            if entry is None:
                self.live_calls.append(call)
                task = loop.create_task(live(call))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                return
            if entry.call != _describe(call):
                diverge(f"call {call.id} is {_format(call)}, but was {entry.call[1]}({entry.call[2]})")
                return
            loop.call_soon(deliver_replayed)

        def on_idle(seq: int, issued: int) -> None:
            if 0 < seq <= len(replayed):
                recorded = replayed[seq - 1].issued_after
                if recorded is not None and recorded != issued:
                    diverge(f"after result {seq} it had made {issued} calls, where it had made {recorded}")
                    return
            if 0 < seq <= len(journal) and journal[seq - 1].issued_after is None:
                journal[seq - 1].issued_after = issued

        run = self.sandbox.start(
            code,
            started_at=STARTED_AT,
            seed=seed,
            tools=[CodemodeDeclaredTool("read"), CodemodeDeclaredTool("write")],
            on_call=on_call,
            on_idle=on_idle,
            signal=controller.signal,
            timeout_ms=20_000,
        )
        result = await run.result
        # A replay that ends where the live run went on is a divergence too.
        if self.divergence is None and result.during is not None and result.during.seq < len(replayed):
            self.divergence = (
                "The script took a different path when resumed after a restart: "
                f"it ended after result {result.during.seq}, where it went on before."
            )
        return result


def _describe(call: CodemodeHostCall) -> tuple[str, str, str]:
    return (call.target, call.name, json.dumps(call.args, sort_keys=True))


def _format(call: CodemodeHostCall) -> str:
    return f"{call.name}({json.dumps(call.args, sort_keys=True)})"


SCRIPT = """
const names = await Promise.all([tools.read("a"), tools.read("b")]);
const pick = names[Math.floor(Math.random() * names.length)];
await tools.write({ name: pick });
text("wrote " + pick);
return pick;
"""


async def test_replays_a_journal_to_the_same_calls_and_continues_live(sandbox: CodemodeSandbox) -> None:
    reads: list[Any] = []

    async def answer(call: CodemodeHostCall) -> Any:
        reads.append(call.args)
        await asyncio.sleep(0.01)
        return f"file-{call.args}" if call.name == "read" else "ok"

    journal: list[JournalEntry] = []
    first = await ReplayHost(sandbox, answer).run(SCRIPT, journal)
    assert isinstance(first, CodemodeSuccess)
    assert len(journal) == 3 and all(entry.issued_after is not None for entry in journal[:-1])

    # "Restart" after the two reads: replay them, then the write runs live.
    reads.clear()
    replay = ReplayHost(sandbox, answer)
    second = await replay.run(SCRIPT, journal[:2])
    assert replay.divergence is None
    assert isinstance(second, CodemodeSuccess)
    assert (second.value, second.output) == (first.value, first.output)
    assert [(call.name, call.args) for call in replay.live_calls] == [("write", {"name": first.value})]
    assert reads == [{"name": first.value}]


async def test_ends_a_replay_that_takes_another_path(sandbox: CodemodeSandbox) -> None:
    async def answer(call: CodemodeHostCall) -> Any:
        return f"file-{call.args}" if call.name == "read" else "ok"

    journal: list[JournalEntry] = []
    assert (await ReplayHost(sandbox, answer).run(SCRIPT, journal)).ok
    changed = SCRIPT.replace('tools.read("b")', 'tools.read("c")')
    replay = ReplayHost(sandbox, answer)
    result = await replay.run(changed, journal)
    assert isinstance(result, CodemodeFailure) and result.error.kind == "aborted"
    assert replay.divergence == (
        'The script took a different path when resumed after a restart: call 2 is read("c"), but was read("b")'
    )
    # A replay that ends where the live run went on diverged as well.
    stops = ReplayHost(sandbox, answer)
    ended = await stops.run(
        SCRIPT.replace("const pick = names[Math.floor(Math.random() * names.length)];", "return 'stopped';"), journal
    )
    assert ended.ok and ended.during == CodemodeDuring(2, 2)
    assert stops.divergence == (
        "The script took a different path when resumed after a restart: it ended after result 2, where it went on before."
    )
