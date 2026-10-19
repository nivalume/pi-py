"""Port of upstream ``test/deterministic.test.ts``, plus replay through a journal as pi-durable's codemode does it.

``start()`` runs a script deterministically with the caller answering its calls, so a durable host
can replay it after a restart. Upstream runs every test on the worker and the Cloudflare remote
sandbox; here on :class:`CodemodeSandbox`. The remote-transport reordering test is dropped (no
transport reorders here; ``test_applies_results_in_seq_order`` covers the VM side).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import re
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
#: The first two `random.random()` values with seed 0 (CPython's `random.seed(0)` too).
PINNED_SEED_ZERO = [0.8444218515250481, 0.7579544029403025]
#: The first `random.random()` with seed 1.
PINNED_FIRST_RANDOM = 0.13436424411240122


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
    respond = answer or (lambda call: {"ok": True, "value": call.args})
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
async def sandbox(new_sandbox: Callable[..., CodemodeSandbox]) -> CodemodeSandbox:
    return new_sandbox()


def timeless(result: CodemodeResult) -> CodemodeResult:
    """Call durations are wall-clock time; everything else is the same."""
    return replace(result, calls=[replace(call, duration_ms=0) for call in result.calls])


async def test_makes_the_same_calls_and_output_for_the_same_start_seed_and_results(sandbox: CodemodeSandbox) -> None:
    code = """
import datetime, json, random, time
values = [time.time(), time.monotonic(), random.random(), random.random(), datetime.datetime.now(datetime.timezone.utc).isoformat()]
first = await tools.echo(values)
text(json.dumps([time.time(), time.monotonic(), random.random()]))
return await tools.echo(first + [random.random()])
"""
    one = start(sandbox, code)
    two = start(sandbox, code)
    a, b = await asyncio.gather(one.run.result, two.run.result)
    assert timeless(a) == timeless(b)
    assert one.calls == two.calls
    # The clock: started_at, then the time of the result being processed; time.monotonic() counts from started_at.
    args = one.calls[0].args
    assert isinstance(args, list)
    assert args[:2] == [STARTED_AT / 1000, 0]
    assert args[4] == "2026-01-15T10:00:00+00:00"
    assert isinstance(a, CodemodeSuccess)
    assert isinstance(a.output[0], CodemodeTextOutput)
    assert json.loads(a.output[0].text)[:2] == [STARTED_AT / 1000 + 1, 1]
    # Pinned: a change here changes what replayed scripts see, so it must change the fingerprint too.
    assert args[2] == PINNED_FIRST_RANDOM

    # The seed decides the random values, not the start time.
    later = start(sandbox, code, started_at=STARTED_AT + 5000)
    reseeded = start(sandbox, code, seed=2)
    await asyncio.gather(later.run.result, reseeded.run.result)
    assert isinstance(later.calls[0].args, list) and later.calls[0].args[2] == PINNED_FIRST_RANDOM
    assert isinstance(reseeded.calls[0].args, list) and reseeded.calls[0].args[2] != PINNED_FIRST_RANDOM
    # Seed 0 has its own values.
    zero = await start(sandbox, "import random\nreturn [random.random(), random.random()]", seed=0).run.result
    assert isinstance(zero, CodemodeSuccess)
    assert zero.value == PINNED_SEED_ZERO
    assert PINNED_FIRST_RANDOM not in zero.value


async def test_runs_in_utc_whatever_the_host_timezone(
    sandbox: CodemodeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        code = """
import time
try:
    import weakref
    weak = "weakref"
except ImportError:
    weak = "none"
return [time.localtime().tm_hour - time.gmtime().tm_hour, time.timezone, time.tzname[0], weak]
"""
        result = await start(sandbox, code).run.result
        assert isinstance(result, CodemodeSuccess)
        assert result.value == [0, 0, "UTC", "none"]
        # execute() is unchanged: the host's timezone. Neither has weak references.
        executed = await sandbox.execute(code)
        assert isinstance(executed, CodemodeSuccess)
        assert isinstance(executed.value, list)
        assert executed.value[1:] == [-19800, "IST", "none"]
    finally:
        monkeypatch.undo()
        time.tzset()


async def test_applies_results_in_seq_order_whatever_order_they_are_delivered_in(sandbox: CodemodeSandbox) -> None:
    recorded = start(
        sandbox,
        """
import asyncio
order = []
async def wait(name):
    await tools.wait(name)
    order.append(name)
await asyncio.gather(wait("a"), wait("b"))
return order
""",
    )
    await wait_for(lambda: len(recorded.calls) == 2 and len(recorded.idles) == 1)
    a, b = recorded.calls
    # b is result 1 and a result 2, but a arrives first.
    recorded.run.deliver(2, STARTED_AT + 2000, a.id, {"ok": True, "value": None})
    recorded.run.deliver(1, STARTED_AT + 1000, b.id, {"ok": True, "value": None})
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess) and result.value == ["b", "a"]
    # No idle after the last result: the script finished in that drain, and the host stopped the sandbox.
    assert recorded.idles == [(0, 2), (1, 2)]


async def test_reports_the_fingerprint_before_any_call_and_the_calls_issued_after_each_result(
    new_sandbox: Callable[..., CodemodeSandbox],
) -> None:
    sandbox = new_sandbox(memory_limit_bytes=64 * 1024 * 1024)
    recorded = start(sandbox, "await tools.echo(1)\nimport asyncio\nawait asyncio.gather(tools.echo(2), tools.echo(3))")
    assert (await recorded.run.result).ok
    assert recorded.events[:3] == ["started", "call 1", "idle 0"]
    assert recorded.idles == [(0, 1), (1, 3), (2, 3)]
    assert [(call.id, call.target, call.name, call.args) for call in recorded.calls] == [
        (1, "tool", "echo", 1),
        (2, "tool", "echo", 2),
        (3, "tool", "echo", 3),
    ]
    fingerprint = recorded.fingerprint
    assert fingerprint is not None
    assert re.fullmatch(
        r"pydantic-monty \S+, monty \S+, prelude [0-9a-f]{8}, deterministic \d+, recursion \d+, heap 67108864",
        fingerprint,
    )

    same = start(sandbox, "return 1")
    other = start(new_sandbox(memory_limit_bytes=32 * 1024 * 1024), "return 1")
    await asyncio.gather(same.run.result, other.run.result)
    assert same.fingerprint == fingerprint
    assert other.fingerprint != fingerprint


async def test_numbers_calls_to_globals_with_tools_and_answers_errors_as_exceptions(sandbox: CodemodeSandbox) -> None:
    def answer(call: CodemodeHostCall) -> CodemodeOutcome:
        if call.args == "fail":
            return {"ok": False, "message": "no"}
        return {"ok": True, "value": ["found"] if call.name == "search" else call.args}

    recorded = start(
        sandbox,
        """
found = await search("x", 2)
try:
    await tools.echo("fail")
except Exception as error:
    text(str(error))
return found
""",
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
        sandbox,
        "import time\nbefore = time.time()\nawait tools.wait(1)\nreturn [before, time.time(), time.monotonic()]",
    )
    await wait_for(lambda: len(recorded.calls) == 1)
    with pytest.raises(ValueError, match="not a finite number"):
        recorded.run.deliver(1, math.nan, 1, {"ok": True, "value": None})
    # A resumed host whose clock is behind the run's start.
    recorded.run.deliver(1, STARTED_AT - 2000, 1, {"ok": True, "value": None})
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess)
    assert result.value == [STARTED_AT / 1000, STARTED_AT / 1000, 0]


async def test_advances_the_clock_on_sleeps_without_waiting(sandbox: CodemodeSandbox) -> None:
    # Not upstream (QuickJS has no timers): sleeps return at once and move the virtual clock.
    started = time.perf_counter()
    recorded = start(
        sandbox,
        "import asyncio, time\nbefore = time.time()\nawait asyncio.sleep(5)\ntime.sleep(1)\n"
        "await tools.echo(1)\nreturn [time.time() - before, time.monotonic()]",
    )
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess)
    # The result's clock (STARTED_AT + 1000) is behind the slept clock, which never goes back.
    assert result.value == [6, 6]
    assert time.perf_counter() - started < 3


async def test_derives_os_urandom_from_the_seed(sandbox: CodemodeSandbox) -> None:
    code = "import os\nreturn list(os.urandom(8))"
    one, two, other = await asyncio.gather(
        start(sandbox, code).run.result, start(sandbox, code).run.result, start(sandbox, code, seed=9).run.result
    )
    assert isinstance(one, CodemodeSuccess) and isinstance(two, CodemodeSuccess) and isinstance(other, CodemodeSuccess)
    assert one.value == two.value != other.value


async def test_reports_the_result_the_sandbox_was_processing_when_the_script_ended(sandbox: CodemodeSandbox) -> None:
    returned = start(sandbox, "await tools.echo(1)\nreturn await tools.echo(2)")
    thrown = start(
        sandbox,
        "await tools.echo(1)\nimport asyncio\nawait asyncio.gather(tools.echo(2), tools.echo(3))\nraise RuntimeError('x')",
    )
    first = start(sandbox, "return 1")
    returned_result = await returned.run.result
    assert isinstance(returned_result, CodemodeSuccess)
    assert (returned_result.value, returned_result.during) == (2, CodemodeDuring(2, 2))
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
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess) and result.value == "fine"


async def test_rejects_with_an_exception_carrying_the_answers_message(sandbox: CodemodeSandbox) -> None:
    # Upstream's Error carries the answer's properties; Monty exceptions hold only a message, so
    # `properties` are validated and accepted but not visible to the script.
    recorded = start(
        sandbox,
        "try:\n    await tools.echo(1)\nexcept Exception as error:\n"
        "    return [isinstance(error, Exception), str(error), type(error).__name__]",
        answer=lambda _call: {
            "ok": False,
            "message": "denied",
            "properties": {"structuredOutput": {"code": 7}, "name": "Ignored", "message": "ignored"},
        },
    )
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess) and result.value == [True, "denied", "Exception"]


async def test_rejects_rejection_properties_that_are_not_an_object_and_tools_sharing_a_script_name(
    sandbox: CodemodeSandbox,
) -> None:
    recorded = start(sandbox, "return await tools.wait(1)")
    await wait_for(lambda: len(recorded.calls) == 1)
    for properties in (None, [1]):
        with pytest.raises(ValueError, match="not an object"):
            recorded.run.deliver(1, STARTED_AT, 1, {"ok": False, "message": "x", "properties": properties})  # type: ignore[typeddict-item]
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": 1})
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess) and result.value == 1
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


async def test_uses_the_script_names_the_run_passes(sandbox: CodemodeSandbox) -> None:
    recorded = start(
        sandbox,
        "return [type(tools.renamed).__name__, 'renamed' in tools, 'my-tool' in tools, [tool['name'] for tool in ALL_TOOLS]]",
        tools=[CodemodeDeclaredTool("my-tool", js_name="renamed")],
    )
    result = await recorded.run.result
    assert isinstance(result, CodemodeSuccess)
    assert result.value == ["function", True, True, ["renamed"]]


async def test_ends_with_an_error_the_script_cannot_catch_when_the_caller_aborts(sandbox: CodemodeSandbox) -> None:
    controller = AbortController()
    recorded = start(
        sandbox,
        'try:\n    await tools.wait(1)\nexcept BaseException:\n    text("caught")\ntext("after")',
        signal=controller.signal,
    )
    await wait_for(lambda: len(recorded.calls) == 1)
    controller.abort(RuntimeError("diverged"))
    result = await recorded.run.result
    assert isinstance(result, CodemodeFailure)
    assert (result.error.kind, result.error.message, result.output) == ("aborted", "diverged", [])
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": None})


async def test_is_finished_once_it_has_its_outcome(sandbox: CodemodeSandbox) -> None:
    recorded = start(sandbox, "await tools.wait(1)\nreturn 2")
    await wait_for(lambda: len(recorded.calls) == 1)
    assert recorded.run.finished is False
    recorded.run.deliver(1, STARTED_AT, 1, {"ok": True, "value": None})
    await recorded.run.result
    assert recorded.run.finished is True


async def test_rejects_a_call_that_is_not_waiting_and_a_result_number_used_twice(sandbox: CodemodeSandbox) -> None:
    recorded = start(sandbox, "import asyncio\nawait asyncio.gather(tools.wait(1), tools.wait(2))")
    await wait_for(lambda: len(recorded.calls) == 2)
    run = recorded.run
    with pytest.raises(ValueError, match="Call 99 is not waiting"):
        run.deliver(1, STARTED_AT, 99, {"ok": True, "value": None})
    run.deliver(1, STARTED_AT, 1, {"ok": True, "value": None})
    with pytest.raises(ValueError, match="used once"):
        run.deliver(1, STARTED_AT, 2, {"ok": True, "value": None})
    with pytest.raises(ValueError, match="used once"):
        run.deliver(0, STARTED_AT, 2, {"ok": True, "value": None})
    run.deliver(2, STARTED_AT, 2, {"ok": True, "value": None})
    assert (await run.result).ok


async def test_execute_reads_the_real_clock_and_real_randomness(sandbox: CodemodeSandbox) -> None:
    before = time.time()
    first = await sandbox.execute("import time, random\nreturn [time.time(), random.random()]")
    second = await sandbox.execute("import time, random\nreturn [time.time(), random.random()]")
    assert isinstance(first, CodemodeSuccess) and isinstance(second, CodemodeSuccess)
    assert isinstance(first.value, list) and isinstance(second.value, list)
    now, random_value = first.value
    assert isinstance(now, float) and before <= now <= time.time()
    assert random_value not in second.value


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
import asyncio, random
names = await asyncio.gather(tools.read("a"), tools.read("b"))
pick = random.choice(names)
await tools.write({"name": pick})
text("wrote " + pick)
return pick
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
    ended = await stops.run(SCRIPT.replace("pick = random.choice(names)", "return 'stopped'"), journal)
    assert ended.ok and ended.during == CodemodeDuring(2, 2)
    assert stops.divergence == (
        "The script took a different path when resumed after a restart: it ended after result 2, where it went on before."
    )
