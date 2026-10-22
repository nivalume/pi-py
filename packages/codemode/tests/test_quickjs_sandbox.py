"""The QuickJS sandbox (:class:`pi_codemode.CodemodeSandbox`): JavaScript scripts, as upstream's.

Port of upstream ``test/sandbox.test.ts`` by way of delta's ``crates/codemode/tests/sandbox.rs``, which runs the same
prelude in the same engine. Added here: what the wasmtime host itself must get right (the memory limit, a trap that
ends only its own VM, prompt interrupts on close, an event loop that keeps running while a script computes).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest
from codemode_helpers import GIF, JPEG, PNG, WEBP, echo_tool, value_of

from pi_ai.abort import AbortController, AbortSignal
from pi_codemode import (
    MAX_OUTPUT_CHARS,
    MAX_OUTPUT_ITEMS,
    CodemodeFailure,
    CodemodeImageOutput,
    CodemodeSandbox,
    CodemodeSuccess,
    CodemodeTextOutput,
    CodemodeTool,
    CodemodeToolContext,
)

CreateSandbox = Callable[..., CodemodeSandbox]
NewSandbox = Callable[..., CodemodeSandbox]


def success(result: Any) -> CodemodeSuccess:
    assert isinstance(result, CodemodeSuccess), result
    return result


def failure(result: Any) -> CodemodeFailure:
    assert isinstance(result, CodemodeFailure), result
    return result


def text(value: str, *, console: bool = False) -> CodemodeTextOutput:
    return CodemodeTextOutput(value, console=console)


class TestScriptExecution:
    async def test_returns_values_after_a_json_round_trip(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = success(await sandbox.execute("return { a: 1, b: [true, 'x'] }"))
        assert (result.value, result.output, result.calls) == ({"a": 1, "b": [True, "x"]}, [], [])
        assert success(await sandbox.execute("return 'plain'")).value == "plain"
        assert success(await sandbox.execute("")).value is None
        assert success(await sandbox.execute("const x = await Promise.resolve(41); return x + 1")).value == 42
        # A trailing expression is not a return value, as in a function body.
        assert success(await sandbox.execute("1 + 1")).value is None

    async def test_collects_output_in_order(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = success(
            await sandbox.execute(
                f"""
console.log("hello", 1, {{ a: 1 }});
text({{ json: true }});
text(undefined);
text(7);
image("data:image/png;base64,{PNG}");
image({{ image_url: "data:image/jpeg;base64,{JPEG}" }});
image({{ type: "image", data: "{GIF}", mimeType: "image/gif" }});
image("data:image/png;base64,{WEBP}");
image({{ type: "image", data: "{PNG}" }});
console.error(new Error("bad"));
return null;
"""
            )
        )
        *rest, last = result.output
        assert rest == [
            text('hello 1 {"a":1}', console=True),
            text('{"json":true}'),
            text("undefined"),
            text("7"),
            CodemodeImageOutput(PNG, "image/png"),
            CodemodeImageOutput(JPEG, "image/jpeg"),
            CodemodeImageOutput(GIF, "image/gif"),
            # The MIME type comes from the data, not from the declared type.
            CodemodeImageOutput(WEBP, "image/webp"),
            CodemodeImageOutput(PNG, "image/png"),
        ]
        assert isinstance(last, CodemodeTextOutput) and last.console and last.text.startswith("Error: bad")

    async def test_rejects_invalid_text_and_image_arguments(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = success(
            await sandbox.execute(
                r"""
const errors = [];
const circular = {};
circular.self = circular;
for (const run of [
    () => text(circular),
    () => image(""),
    () => image("https://example.com/a.png"),
    () => image("data:image/png,raw"),
    () => image({ type: "text", text: "x" }),
    () => image({ type: "image", data: "" }),
    () => image(42),
    () => image("data:image/png;base64,AAAA!"),
    () => image("data:image/png;base64,AAAAA"),
    () => image("data:image/png;base64,AA=A"),
    () => image("data:image/png;base64,"),
    () => image("data:image/png;base64,AAAA\n[Output truncated]"),
    () => image({ type: "image", data: "AAAA!", mimeType: "image/png" }),
    () => image("data:image/png;base64,AAAA"),
    () => image("data:image/png;base64,QUJD"),
    () => image("data:image/jpeg;base64,/9j/9w=="),
]) {
    try { run(); errors.push("no error"); } catch (error) { errors.push(error.name + ": " + error.message); }
}
return errors;
"""
            )
        )
        assert result.output == []
        errors = result.value
        assert isinstance(errors, list) and isinstance(errors[0], str)
        assert errors[0].startswith("TypeError: ") and "ircular" in errors[0]
        expects = (
            "TypeError: image expects a non-empty image URL string, an object with image_url, "
            'or an image block { type: "image", data, mimeType }'
        )
        assert errors[1:] == [
            expects,
            "TypeError: remote image URLs are not supported in tool outputs. Pass a base64 data URI instead",
            "TypeError: invalid image output. Pass a base64 data URI instead",
            'TypeError: image only accepts image blocks, got "text"',
            "TypeError: image expected image block data",
            expects,
            *["TypeError: invalid image output. The image data is not valid base64 (truncated or corrupted?)"] * 6,
            *["TypeError: invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image"] * 3,
        ]

    async def test_accepts_wrapped_and_large_base64_images(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        large = "iVBORw0KGgoA" + "QUJD" * (256 * 1024)
        result = success(
            await sandbox.execute(
                f'image("data:image/png;base64,iVBORw0K\\r\\nGgo=\\n");\nimage("data:image/png;base64,{large}");'
            )
        )
        assert result.output == [CodemodeImageOutput(PNG, "image/png"), CodemodeImageOutput(large, "image/png")]

    async def test_exit_ends_successfully_keeping_output_and_store_writes(
        self, create_quickjs_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_quickjs_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                'text("before");\nstore("k", 1);\nawait tools.echo(1);\ntry { exit(); } catch {}\n'
                'text("after");\nreturn "unreachable";'
            )
        )
        assert result.value is None
        assert result.output == [text("before")]
        assert result.store_writes.set == {"k": 1}

    async def test_errors_keep_output_and_point_at_script_lines(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        partial = failure(await sandbox.execute('text("partial");\nthrow new Error("boom")'))
        assert partial.output == [text("partial")]

        syntax = failure(await sandbox.execute("const a = 1;\nconst b = ;\nreturn a")).error
        assert (syntax.kind, syntax.name) == ("script", "SyntaxError")
        assert "codemode.js:2" in (syntax.stack or "")

        thrown = failure(await sandbox.execute("const a = 1;\nthrow new TypeError('boom ' + a)")).error
        assert (thrown.kind, thrown.name, thrown.message) == ("script", "TypeError", "boom 1")
        assert "codemode.js:2" in (thrown.stack or "")

        stacked = failure(await sandbox.execute("console.log(new Error('inner'));\nthrow new RangeError('outer')"))
        stack = stacked.error.stack or ""
        assert stack.startswith("RangeError: outer\n    at ")
        assert "codemode.js:2" in stack and "codemode-prelude.js" not in stack
        inner = stacked.output[0]
        assert isinstance(inner, CodemodeTextOutput)
        assert inner.text.startswith("Error: inner\n    at ") and "codemode.js:1" in inner.text

        assert failure(await sandbox.execute("throw { code: 7 }")).error.message == '{"code":7}'
        assert failure(await sandbox.execute("return 10n")).error.name == "TypeError"
        numeric = failure(await sandbox.execute("const error = new Error('x'); error.message = 42; throw error;"))
        assert (numeric.error.name, numeric.error.message) == ("Error", "42")


class TestTools:
    async def test_tools_are_async_functions_with_call_records(self, create_quickjs_sandbox: CreateSandbox) -> None:
        seen: list[Any] = []

        def add(args: Any, _context: CodemodeToolContext) -> Any:
            seen.append(args)
            return {"sum": args.get("a", 0) + args.get("b", 0)}

        sandbox = create_quickjs_sandbox([CodemodeTool("add", add)])
        result = success(
            await sandbox.execute(
                "const first = await tools.add({ a: 1, b: 2 });\n"
                "const second = await tools.add({ a: first.sum, b: 10 });\nreturn second.sum;"
            )
        )
        assert result.value == 13
        assert seen == [{"a": 1, "b": 2}, {"a": 3, "b": 10}]
        assert [(call.name, call.status) for call in result.calls] == [("add", "ok"), ("add", "ok")]
        assert all(call.duration_ms >= 0 for call in result.calls)

    async def test_runs_concurrent_calls_and_normalizes_names(self, create_quickjs_sandbox: CreateSandbox) -> None:
        async def delay(args: Any, _context: CodemodeToolContext) -> Any:
            await asyncio.sleep(0.02)
            return args

        sandbox = create_quickjs_sandbox([echo_tool(), CodemodeTool("delay", delay)])
        result = success(
            await sandbox.execute(
                "const [a, b, c] = await Promise.all([tools.delay(1), tools.delay(2), tools.echo(3)]);\n"
                "return { values: [a, b, c], names: Object.keys(tools) };"
            )
        )
        assert result.value == {"values": [1, 2, 3], "names": ["echo", "delay"]}

        named = create_quickjs_sandbox(
            [
                CodemodeTool("my-tool", lambda _a, _c: "dash", description="Dashes"),
                CodemodeTool("my_tool", lambda _a, _c: "underscore", description="Shadowed"),
                CodemodeTool("mcp__docs__search", lambda _a, _c: "mcp"),
                CodemodeTool("class", lambda _a, _c: "keyword"),
            ]
        )
        result = success(
            await named.execute(
                "try { ALL_TOOLS.push({}); } catch {}\n"
                'return { all: ALL_TOOLS, calls: [await tools.my_tool(), await tools["my-tool"](), '
                "await tools.mcp__docs__search(), await tools.class()] };"
            )
        )
        # Upstream's identifiers: JavaScript reads `tools.class` as is, without Python's trailing `_`.
        assert result.value == {
            "all": [
                {"name": "my_tool", "description": "Dashes"},
                {"name": "mcp__docs__search", "description": ""},
                {"name": "class", "description": ""},
            ],
            "calls": ["dash", "dash", "mcp", "keyword"],
        }
        with pytest.raises(ValueError, match="already registered"):
            named.register_tool(CodemodeTool("class", lambda _a, _c: None))

    async def test_passes_undefined_through_and_turns_tool_errors_into_errors(
        self, create_quickjs_sandbox: CreateSandbox
    ) -> None:
        def fail(_args: Any, _context: CodemodeToolContext) -> Any:
            raise RuntimeError("tool exploded")

        sandbox = create_quickjs_sandbox([CodemodeTool("noop", lambda args, _c: args), CodemodeTool("fail", fail)])
        assert success(await sandbox.execute("return [await tools.noop(), await tools.noop(null)]")).value == [
            None,
            None,
        ]
        result = success(
            await sandbox.execute(
                "try { await tools.fail(); return 'no error'; } "
                "catch (error) { return { isError: error instanceof Error, message: error.message }; }"
            )
        )
        assert result.value == {"isError": True, "message": "tool exploded"}
        assert result.calls[0].status == "error"
        missing = failure(await sandbox.execute("return await tools.missing()")).error
        assert (missing.kind, missing.name) == ("script", "TypeError")

    async def test_cancels_unawaited_calls_when_the_script_returns(self, create_quickjs_sandbox: CreateSandbox) -> None:
        aborted = asyncio.Event()

        async def slow(_args: Any, context: CodemodeToolContext) -> Any:
            await context.signal.wait()
            aborted.set()
            raise RuntimeError("aborted")

        sandbox = create_quickjs_sandbox([CodemodeTool("slow", slow)])
        result = success(await sandbox.execute("tools.slow(); return 'early'"))
        assert result.value == "early"
        assert [(call.name, call.status) for call in result.calls] == [("slow", "cancelled")]
        await asyncio.wait_for(aborted.wait(), 5)

    async def test_names_close_matches_of_missing_members(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox([echo_tool(), CodemodeTool("web-search", lambda _a, _c: "")])

        async def attempt(expression: str) -> Any:
            return value_of(await sandbox.execute(f"return {expression};"))

        assert await attempt("tools.Echo") == (
            "tools.Echo does not exist. Did you mean tools.echo? ALL_TOOLS lists every tool; searchTools(query) "
            'finds tools by topic. Check for a member with "Echo" in tools.'
        )
        assert "Did you mean tools.web_search?" in await attempt("tools.websearch")
        assert "Available: echo, web_search." in await attempt("tools.nothing")
        assert await attempt(
            "['echo' in tools, 'nothing' in tools, String(tools.toString), JSON.stringify(tools)]"
        ) == [
            True,
            False,
            "undefined",
            "{}",
        ]


class TestStoreAndLoad:
    async def test_reads_the_snapshot_and_reports_writes(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = success(
            await sandbox.execute(
                """
const seen = load("counter");
store("counter", seen + 1);
store("list", [1, { a: null }]);
store("old", undefined);
const value = load("obj"); value.a = 2;
const kept = { b: 1 }; store("kept", kept); kept.b = 2;
return [seen, load("counter"), load("missing"), load("old"), load("obj").a, load("kept").b];
""",
                store={"counter": 41, "old": "x", "obj": {"a": 1}},
            )
        )
        assert result.value == [41, 42, None, None, 1, 1]
        assert result.store_writes.set == {"counter": 42, "list": [1, {"a": None}], "kept": {"b": 1}}
        assert result.store_writes.delete == ["old"]

    async def test_rejects_invalid_and_oversized_writes(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = success(
            await sandbox.execute(
                """
const attempt = (fn) => { try { fn(); return "ok"; } catch (error) { return error.name; } };
return [
    attempt(() => store(1, "x")),
    attempt(() => load({})),
    attempt(() => store("fn", () => 1)),
    attempt(() => store("big", "x".repeat(300 * 1024))),
    attempt(() => { for (let i = 0; i < 8; i++) store("k" + i, "x".repeat(200 * 1024)); }),
];
"""
            )
        )
        assert result.value == ["TypeError", "TypeError", "TypeError", "RangeError", "RangeError"]
        message = failure(await sandbox.execute('store("img", "x".repeat(300 * 1024));')).error.message
        assert 'store("img") value has 307202 characters of JSON' in message
        assert "Show images with image()" in message
        # A failed run reports no writes; a failure has no store_writes at all.
        assert not hasattr(failure(await sandbox.execute("store('a', 1); throw new Error('x')")), "store_writes")


class TestGlobals:
    async def test_exposes_globals_as_top_level_functions_without_call_records(
        self, create_quickjs_sandbox: CreateSandbox, new_quickjs_sandbox: NewSandbox
    ) -> None:
        seen: list[Any] = []

        def attach(args: Any, _context: CodemodeToolContext) -> Any:
            seen.append(args)
            return None

        sandbox = new_quickjs_sandbox(tools=[echo_tool()], globals=[CodemodeTool("attach", attach)])
        result = success(
            await sandbox.execute(
                'await attach({ ref: 1 });\nattach("not awaited");\n'
                "return [typeof attach, typeof globalThis.attach, await tools.echo(2)];"
            )
        )
        assert result.value == ["function", "function", 2]
        assert len(result.calls) == 1
        # Messages are handled in order, so an unawaited global still runs before the script ends.
        await asyncio.sleep(0)
        assert seen == [{"ref": 1}, "not awaited"]

    async def test_groups_namespaced_globals_and_spreads_arguments(self, new_quickjs_sandbox: NewSandbox) -> None:
        listed: list[Any] = []

        def models_list(args: Any, _context: CodemodeToolContext) -> Any:
            listed.append(args)
            return None

        sandbox = new_quickjs_sandbox(
            globals=[
                CodemodeTool("models.list", models_list, spread=True),
                CodemodeTool("models.first", lambda args, _c: args),
                CodemodeTool("__private", lambda _a, _c: "allowed in JavaScript"),
            ]
        )
        result = success(
            await sandbox.execute(
                'await models.list("classifier", undefined, 3);\nawait models.list();\n'
                "try { models.extra = 1; } catch {}\n"
                'return [Object.keys(models), await models.first("a", "ignored"), "extra" in models, await __private()];'
            )
        )
        assert result.value == [["list", "first"], "a", False, "allowed in JavaScript"]
        assert listed == [["classifier", None, 3], []]
        typo = failure(await sandbox.execute("await models.firstt();")).error.message
        assert typo == (
            'models.firstt does not exist. Did you mean models.first? Check for a member with "firstt" in models.'
        )

    @pytest.mark.parametrize(
        "name",
        [
            "a.b.c",
            "a.",
            ".a",
            "tools.x",
            "store.x",
            "a.not-valid",
            "not-valid",
            "tools",
            "console",
            "globalThis",
            "store",
            "load",
            "ALL_TOOLS",
        ],
    )
    def test_refuses_invalid_and_reserved_global_names(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid global name"):
            CodemodeSandbox(globals=[CodemodeTool(name, lambda _a, _c: None)])

    def test_refuses_conflicting_and_duplicate_globals(self) -> None:
        execute = lambda _a, _c: None  # noqa: E731
        with pytest.raises(ValueError, match='Global "models" conflicts with the namespace "models"'):
            CodemodeSandbox(globals=[CodemodeTool("models", execute), CodemodeTool("models.list", execute)])
        with pytest.raises(ValueError, match="already registered"):
            CodemodeSandbox(globals=[CodemodeTool("a", execute), CodemodeTool("a", execute)])
        # `print` is Python's; JavaScript scripts may have a global of that name.
        assert [global_.name for global_ in CodemodeSandbox(globals=[CodemodeTool("print", execute)]).globals] == [
            "print"
        ]


class TestLimitsAndLifetime:
    @pytest.mark.parametrize("code", ["while (true) {}", "while (true) await null"])
    async def test_loops_time_out(self, create_quickjs_sandbox: CreateSandbox, code: str) -> None:
        sandbox = create_quickjs_sandbox()
        started = time.perf_counter()
        error = failure(await sandbox.execute(code, timeout_ms=200)).error
        assert (error.kind, error.message) == ("timeout", "Execution timed out after 200 ms")
        assert time.perf_counter() - started < 5
        assert success(await sandbox.execute("return 1")).value == 1

    async def test_stalled_scripts_fail_at_once(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox([echo_tool()])
        started = time.perf_counter()
        stalled = failure(
            await sandbox.execute(
                "await tools.echo(1); await new Promise(() => {}); return 'never'", timeout_ms=float("inf")
            )
        ).error
        assert (stalled.kind, stalled.name) == ("script", "Error")
        assert "can never settle" in stalled.message
        assert time.perf_counter() - started < 5
        # Returning while a call is still pending is not a stall.
        assert success(await sandbox.execute("tools.echo(2); return 'early'")).value == "early"

    async def test_runs_without_a_deadline_when_timeout_ms_is_infinite(
        self, create_quickjs_sandbox: CreateSandbox
    ) -> None:
        async def wait(_args: Any, _context: CodemodeToolContext) -> Any:
            await asyncio.sleep(0.05)
            return "late"

        sandbox = create_quickjs_sandbox([CodemodeTool("wait", wait)], timeout_ms=float("inf"))
        assert success(await sandbox.execute("return await tools.wait()")).value == "late"

    async def test_interrupt_budgets_end_computing_scripts(self, new_quickjs_sandbox: NewSandbox) -> None:
        sandbox = new_quickjs_sandbox(timeout_ms=float("inf"), interrupt_budget=20_000)
        error = failure(await sandbox.execute("while (true) {}")).error
        assert error.kind == "timeout"
        assert error.message == "Execution exceeded its interrupt budget of 20000"
        # A script that computes little stays within the budget.
        assert success(await sandbox.execute("let s = 0; for (let i = 0; i < 1000; i++) s += i; return s")).value == (
            499500
        )

    async def test_aborts_via_signal_and_cancels_in_flight_calls(self, create_quickjs_sandbox: CreateSandbox) -> None:
        tool_signals: list[AbortSignal] = []
        called = asyncio.Event()

        async def hang(_args: Any, context: CodemodeToolContext) -> Any:
            tool_signals.append(context.signal)
            called.set()
            await asyncio.Event().wait()

        sandbox = create_quickjs_sandbox([CodemodeTool("hang", hang)])
        already = AbortController()
        already.abort(RuntimeError("user cancelled"))
        error = failure(await sandbox.execute("await tools.hang()", signal=already.signal)).error
        assert (error.kind, error.message) == ("aborted", "user cancelled")

        controller = AbortController()
        pending = asyncio.ensure_future(sandbox.execute("await tools.hang(); return 'never'", signal=controller.signal))
        await called.wait()
        controller.abort()
        aborted = failure(await pending)
        assert aborted.error.kind == "aborted"
        assert [(call.name, call.status) for call in aborted.calls] == [("hang", "cancelled")]
        assert tool_signals[-1].aborted is True

    async def test_close_interrupts_a_computing_script_at_once(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox(timeout_ms=float("inf"))
        pending = asyncio.ensure_future(sandbox.execute("while (true) {}"))
        await asyncio.sleep(0.2)
        started = time.perf_counter()
        await sandbox.close()
        error = failure(await asyncio.wait_for(pending, 5)).error
        assert (error.kind, error.message) == ("aborted", "Sandbox closed")
        assert time.perf_counter() - started < 1
        with pytest.raises(RuntimeError, match="Sandbox is closed"):
            await sandbox.execute("return 1")

    async def test_the_event_loop_keeps_running_while_a_script_computes(
        self, create_quickjs_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_quickjs_sandbox()
        pending = asyncio.ensure_future(sandbox.execute("while (true) {}", timeout_ms=500))
        ticks = 0
        while not pending.done():
            await asyncio.sleep(0.01)
            ticks += 1
        assert failure(pending.result()).error.kind == "timeout"
        # wasmtime releases the GIL while the script runs, so the loop ticked throughout.
        assert ticks >= 20

    # #10283: the host keeps all output, so a script that prints in a loop must not grow it without bound.
    @pytest.mark.parametrize("emit", ["text(s)", "console.log(s)", 'image("data:image/png;base64," + p)'])
    async def test_fails_a_script_whose_output_passes_the_limits(
        self, create_quickjs_sandbox: CreateSandbox, emit: str
    ) -> None:
        sandbox = create_quickjs_sandbox(timeout_ms=60_000)
        result = failure(
            await sandbox.execute(
                'const s = "x".repeat(1 << 20);\nconst p = "iVBORw0KGgoA" + "A".repeat(1 << 20);\n'
                f"for (;;) {{ try {{ {emit}; }} catch {{}} }}"
            )
        )
        assert (result.error.kind, result.error.name) == ("script", "RangeError")
        assert "script output exceeded" in result.error.message
        chars = sum(len(item.text if isinstance(item, CodemodeTextOutput) else item.data) for item in result.output)
        assert MAX_OUTPUT_CHARS - (2 << 20) < chars <= MAX_OUTPUT_CHARS

    async def test_limits_output_items(self, create_quickjs_sandbox: CreateSandbox) -> None:
        # About 0.1 ms per item: each text() is a bridge call that converts three strings.
        sandbox = create_quickjs_sandbox(timeout_ms=120_000)
        result = failure(await sandbox.execute('for (;;) text("");'))
        assert result.error.name == "RangeError"
        assert len(result.output) == MAX_OUTPUT_ITEMS

    async def test_runs_do_not_share_state(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        results = await asyncio.gather(
            sandbox.execute("globalThis.shared = 'a'; await null; return globalThis.shared"),
            sandbox.execute("globalThis.shared = 'b'; await null; return globalThis.shared"),
            sandbox.execute("return typeof globalThis.shared"),
        )
        assert [success(result).value for result in results] == ["a", "b", "undefined"]

    async def test_deep_recursion_is_a_catchable_range_error(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = await sandbox.execute(
            "let depth = 0;\nfunction dive() { depth++; dive(); }\n"
            "try { dive(); } catch (error) { return [error.name, depth > 1000]; }"
        )
        assert success(result).value == ["RangeError", True]

    async def test_a_trap_fails_only_its_own_execution(self, create_quickjs_sandbox: CreateSandbox) -> None:
        # QuickJS does not check its stack while parsing deeply nested JSON, so wasmtime's stack guard traps.
        sandbox = create_quickjs_sandbox()
        error = failure(await sandbox.execute('JSON.parse("[".repeat(200000))')).error
        assert (error.kind, error.message) == ("sandbox", "RuntimeError: wasm trap: call stack exhausted")
        assert success(await sandbox.execute("return 1")).value == 1

    async def test_caps_memory(self, new_quickjs_sandbox: NewSandbox) -> None:
        sandbox = new_quickjs_sandbox(memory_limit_bytes=32 * 1024 * 1024)
        result = await sandbox.execute(
            "const chunks = [];\n"
            "try { for (;;) chunks.push('x'.repeat(1 << 20) + chunks.length); } catch (error) { return String(error); }"
        )
        assert "out of memory" in str(success(result).value)
        # The limit is the VM's own: the next run starts with a fresh heap.
        assert success(await sandbox.execute("return 'x'.repeat(1 << 20).length")).value == 1 << 20

    @pytest.mark.parametrize("size", [-1, 512 * 1024 + 1])
    def test_refuses_a_stack_size_out_of_range(self, size: int) -> None:
        with pytest.raises(ValueError, match="max_stack_size must be an integer between 0 and 524288"):
            CodemodeSandbox(max_stack_size=size)

    async def test_reports_a_wasm_file_that_cannot_load(self, new_quickjs_sandbox: NewSandbox, tmp_path: Any) -> None:
        broken = tmp_path / "quickjs.wasm"
        broken.write_bytes(b"not wasm")
        error = failure(await new_quickjs_sandbox(wasm_path=broken).execute("return 1")).error
        assert error.kind == "sandbox"
        assert error.message.startswith("Failed to load QuickJS: ")


class TestEscapeHatches:
    async def test_has_no_host_globals(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        result = await sandbox.execute(
            "return [typeof process, typeof require, typeof module, typeof setTimeout, typeof fetch, "
            "typeof WebAssembly, typeof std, typeof os, typeof globalThis.constructor]"
        )
        assert success(result).value == ["undefined"] * 8 + ["function"]

    async def test_keeps_eval_and_imports_inside_the_vm(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox([echo_tool()])
        result = await sandbox.execute(
            'return [eval("typeof process"), new Function("return typeof process")(), '
            'tools.echo.constructor("return typeof require")(), '
            '(async () => {}).constructor("return typeof setTimeout")() instanceof Promise];'
        )
        assert success(result).value == ["undefined", "undefined", "undefined", True]
        imported = await sandbox.execute(
            'try { await import("node:fs"); return "imported"; } catch (error) { return error.constructor.name; }'
        )
        assert value_of(imported) != "imported"

    async def test_ignores_patched_built_ins(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox([echo_tool()])
        patched = success(
            await sandbox.execute(
                """
Array.prototype.toJSON = () => null;
Object.prototype.toJSON = () => 5;
Promise.prototype.then = () => {};
Map.prototype.get = () => undefined;
globalThis.JSON = { stringify: () => "x", parse: () => "x" };
store("k", [1]);
return [await tools.echo([2]), JSON.stringify({ a: 1 })];
"""
            )
        )
        assert patched.value == [[2], '{"a":1}']
        assert patched.store_writes.set == {"k": [1]}

    async def test_freezes_intrinsics_but_keeps_common_overrides(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox()
        frozen = await sandbox.execute(
            """
return [
    Object.getPrototypeOf(function* () {}).prototype,
    Object.getPrototypeOf(async function () {}),
    Object.getPrototypeOf(Int8Array).prototype,
    Object.getPrototypeOf([][Symbol.iterator]()),
    Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]())),
    Object.getPrototypeOf(new Map()[Symbol.iterator]()),
    Object.getPrototypeOf(/a/[Symbol.matchAll]("")),
].every((object) => Object.isFrozen(object));
"""
        )
        assert success(frozen).value is True
        overrides = await sandbox.execute(
            """
const object = {};
object.toString = () => "custom";
function Legacy() {}
Legacy.prototype = Object.create(Error.prototype);
Legacy.prototype.constructor = Legacy;
const bare = new Error();
bare.message = "set later";
class MyError extends Error {
    constructor(message) {
        super(message);
        this.name = "MyError";
    }
}
let patched = "silent";
try { Error.prototype.name = "Patched"; } catch (error) { patched = error.constructor.name; }
return [String(object), new Legacy().constructor === Legacy, bare.message, new MyError("x").name, Error.prototype.name, patched];
"""
        )
        assert success(overrides).value == ["custom", True, "set later", "MyError", "Error", "TypeError"]

    async def test_tampering_with_tools_only_affects_the_script(self, create_quickjs_sandbox: CreateSandbox) -> None:
        sandbox = create_quickjs_sandbox([echo_tool()])
        result = await sandbox.execute(
            "try { tools.echo = () => 'nope'; } catch {}\ntry { tools.extra = () => 'nope'; } catch {}\n"
            "try { globalThis.tools = null; } catch {}\nreturn [\"extra\" in tools, await tools.echo('still')];"
        )
        assert success(result).value == [False, "still"]
