"""Port of upstream ``test/sandbox.test.ts``.

Upstream runs every behavior test against three backends (worker thread, in-thread QuickJS,
Cloudflare remote); here there is one, :class:`CodemodeSandbox` on a Monty pool, and the
scripts are Python. Each test keeps upstream's assertions on the Python equivalent of its
script; differences are noted where they are made.
"""

from __future__ import annotations

import asyncio
import json
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
    CodemodeStoreWrites,
    CodemodeSuccess,
    CodemodeTextOutput,
    CodemodeTool,
    CodemodeToolContext,
)
from pi_codemode.runtime.prelude import PRELUDE_SOURCE

CreateSandbox = Callable[..., CodemodeSandbox]


def success(result: Any) -> CodemodeSuccess:
    assert isinstance(result, CodemodeSuccess), result
    return result


def failure(result: Any) -> CodemodeFailure:
    assert isinstance(result, CodemodeFailure), result
    return result


def test_embedded_prelude_parses_as_python() -> None:
    import ast

    ast.parse(PRELUDE_SOURCE)


class TestScriptExecution:
    async def test_returns_the_scripts_return_value_after_a_json_round_trip(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox()
        result = success(await sandbox.execute("return {'a': 1, 'b': [True, 'x']}"))
        assert (result.value, result.output, result.calls) == ({"a": 1, "b": [True, "x"]}, [], [])
        assert success(await sandbox.execute("return 'plain'")).value == "plain"
        assert success(await sandbox.execute("")).value is None
        # Python: tuples cross as lists, and a trailing expression is not a return value.
        assert success(await sandbox.execute("return (1, 2)")).value == [1, 2]
        assert success(await sandbox.execute("1 + 1")).value is None
        assert success(await sandbox.execute("x = 2\nif x > 1:\n    return 'big'\nreturn 'small'")).value == "big"

    async def test_comprehension_variables_may_shadow_builtins(self, create_sandbox: CreateSandbox) -> None:
        # Monty 1.2 resolves such top-level comprehension variables to the builtin; the sandbox renames them.
        sandbox = create_sandbox()
        script = """
pairs = [("a", 1), ("b", 2)]
ids = [id for id in [1, 2, 3]]
by_type = {type: n for type, n in pairs}
nested = [[id * max for max in range(1, 3)] for id in [2, 3]]
lambdas = [(lambda id: id * 10)(id) for id in [1]]
first = list(id for id in range(id(ids) % 1, 2))
sums = [sum for sum in [5]]
return [ids, by_type, nested, lambdas, first, sums, len(ids), max(ids), type(ids).__name__]
"""
        assert success(await sandbox.execute(script)).value == [
            [1, 2, 3],
            {"a": 1, "b": 2},
            [[2, 4], [3, 6]],
            [10],
            [0, 1],
            [5],
            3,
            3,
            "list",
        ]

        failure = await sandbox.execute("bad = [id for id in [1]]\nraise ValueError('boom')")
        assert not failure.ok
        assert 'codemode.py", line 2' in (failure.error.stack or "")

    async def test_supports_top_level_await(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = await sandbox.execute("async def f():\n    return 41\nx = await f()\nreturn x + 1")
        assert success(result).value == 42

    async def test_collects_text_image_and_print_output_in_order(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                f"""
print("hello", 1, {{"a": 1}})
text({{"json": True}})
text(None)
text(7)
image("data:image/png;base64,{PNG}")
image({{"image_url": "data:image/jpeg;base64,{JPEG}"}})
image({{"type": "image", "data": "{GIF}", "mimeType": "image/gif"}})
image("data:image/png;base64,{WEBP}")
image({{"type": "image", "data": "{PNG}"}})
print(ValueError("bad"))
print("a", "b", sep="-", end="!")
return None
"""
            )
        )
        assert result.output == [
            # print() formats like Python's print (str of each value), not as JSON.
            CodemodeTextOutput("hello 1 {'a': 1}", console=True),
            CodemodeTextOutput('{"json":true}'),
            # None is JSON null (JavaScript's text(undefined) prints "undefined").
            CodemodeTextOutput("null"),
            CodemodeTextOutput("7"),
            CodemodeImageOutput(PNG, "image/png"),
            CodemodeImageOutput(JPEG, "image/jpeg"),
            CodemodeImageOutput(GIF, "image/gif"),
            # The MIME type comes from the data, not from the declared type.
            CodemodeImageOutput(WEBP, "image/webp"),
            CodemodeImageOutput(PNG, "image/png"),
            # Upstream prints an Error's stack; Python's print shows str(error).
            CodemodeTextOutput("bad", console=True),
            CodemodeTextOutput("a-b!", console=True),
        ]

    async def test_rejects_invalid_text_and_image_arguments(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                """
errors = []
circular = {}
circular["self"] = circular
runs = [
    lambda: text(circular),
    lambda: image(""),
    lambda: image("https://example.com/a.png"),
    lambda: image("data:image/png,raw"),
    lambda: image({"type": "text", "text": "x"}),
    lambda: image({"type": "image", "data": ""}),
    lambda: image(42),
    lambda: image("data:image/png;base64,AAAA!"),
    lambda: image("data:image/png;base64,AAAAA"),
    lambda: image("data:image/png;base64,AA=A"),
    lambda: image("data:image/png;base64,"),
    lambda: image("data:image/png;base64,AAAA\\n[Output truncated]"),
    lambda: image({"type": "image", "data": "AAAA!", "mimeType": "image/png"}),
    lambda: image("data:image/png;base64,AAAA"),
    lambda: image("data:image/png;base64,QUJD"),
    lambda: image("data:image/jpeg;base64,/9j/9w=="),
]
for run in runs:
    try:
        run()
        errors.append("no error")
    except Exception as error:
        errors.append(type(error).__name__ + ": " + str(error))
return errors
"""
            )
        )
        assert result.output == []
        errors = result.value
        assert isinstance(errors, list)
        assert str(errors[0]).startswith("TypeError: ") and "ircular" in str(errors[0])
        expects = (
            "TypeError: image expects a non-empty image URL string, a dict with image_url, "
            'or an image block {"type": "image", "data": ..., "mimeType": ...}'
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

    # https://github.com/earendil-works/pi/issues/10215
    async def test_accepts_wrapped_and_large_base64_image_data(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        large = "iVBORw0KGgoA" + "QUJD" * (256 * 1024)
        result = success(
            await sandbox.execute(
                f'image("data:image/png;base64,iVBORw0K\\r\\nGgo=\\n")\nimage("data:image/png;base64,{large}")\n'
            )
        )
        assert result.output == [CodemodeImageOutput(PNG, "image/png"), CodemodeImageOutput(large, "image/png")]

    async def test_ends_the_script_successfully_on_exit_keeping_output_and_store_writes(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                """
text("before")
store("k", 1)
await tools.echo(1)
try:
    exit()
except BaseException:
    pass
text("after")
return "unreachable"
"""
            )
        )
        assert result.value is None
        assert result.output == [CodemodeTextOutput("before")]
        assert result.store_writes == CodemodeStoreWrites({"k": 1}, [])

    async def test_keeps_output_produced_before_a_failure(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = failure(await sandbox.execute('text("partial")\nraise RuntimeError("boom")'))
        assert result.output == [CodemodeTextOutput("partial")]

    async def test_reports_syntax_errors_with_the_scripts_line_number(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        error = failure(await sandbox.execute("a = 1\nb = \nreturn a")).error
        assert (error.kind, error.name) == ("script", "SyntaxError")
        assert error.stack is not None and 'File "codemode.py", line 2' in error.stack

    async def test_reports_raised_errors_with_the_scripts_line_number(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        error = failure(await sandbox.execute("a = 1\nraise TypeError('boom ' + str(a))")).error
        assert (error.kind, error.name, error.message) == ("script", "TypeError", "boom 1")
        assert error.stack is not None and 'File "codemode.py", line 2' in error.stack

    async def test_formats_stacks_like_cpython_without_prelude_frames(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = failure(await sandbox.execute("print(ValueError('inner'))\nraise ValueError('outer')"))
        assert result.error.stack == "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "codemode.py", line 2, in <module>',
                "    raise ValueError('outer')",
                "ValueError: outer",
            ]
        )
        assert result.output == [CodemodeTextOutput("inner", console=True)]
        # An error raised inside a helper shows the script's frames only.
        nested = failure(await sandbox.execute("def show():\n    image(42)\n\nshow()")).error
        assert nested.stack is not None
        assert nested.stack.splitlines()[1:5] == [
            '  File "codemode.py", line 4, in <module>',
            "    show()",
            '  File "codemode.py", line 2, in show',
            "    image(42)",
        ]
        assert "python-input" not in nested.stack and "__codemode" not in nested.stack

    async def test_reports_non_exception_raises(self, create_sandbox: CreateSandbox) -> None:
        # Upstream throws a plain object; Python only raises exceptions.
        sandbox = create_sandbox()
        error = failure(await sandbox.execute("raise {'code': 7}")).error
        assert (error.kind, error.name) == ("script", "TypeError")
        assert "exceptions must derive from BaseException" in error.message

    async def test_reports_a_non_serializable_return_value_as_a_script_error(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox()
        for code in ("return {1, 2}", "return b'x'"):
            error = failure(await sandbox.execute(code)).error
            assert (error.kind, error.name) == ("script", "TypeError")


class TestTools:
    async def test_exposes_tools_as_async_functions_and_records_calls(self, create_sandbox: CreateSandbox) -> None:
        seen: list[Any] = []

        def add(args: Any, _context: CodemodeToolContext) -> Any:
            seen.append(args)
            return {"sum": args["a"] + args["b"]}

        sandbox = create_sandbox([CodemodeTool("add", add)])
        result = success(
            await sandbox.execute(
                "first = await tools.add({'a': 1, 'b': 2})\n"
                "second = await tools.add(a=first['sum'], b=10)\n"
                "return second['sum']"
            )
        )
        assert result.value == 13
        assert seen == [{"a": 1, "b": 2}, {"a": 3, "b": 10}]
        assert [(call.name, call.status) for call in result.calls] == [("add", "ok"), ("add", "ok")]
        assert all(call.duration_ms >= 0 for call in result.calls)

    async def test_rejects_positional_and_keyword_arguments_together(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                "try:\n    tools.echo({'a': 1}, b=2)\nexcept TypeError as error:\n    return str(error)"
            )
        )
        assert result.value == "pass the arguments as one positional value or as keyword arguments, not both"
        assert result.calls == []

    async def test_runs_concurrent_calls_and_lists_tool_names(self, create_sandbox: CreateSandbox) -> None:
        running = 0
        most = 0

        async def delay(args: Any, _context: CodemodeToolContext) -> Any:
            nonlocal running, most
            running += 1
            most = max(most, running)
            await asyncio.sleep(0.02)
            running -= 1
            return args

        sandbox = create_sandbox([echo_tool(), CodemodeTool("delay", delay)])
        result = success(
            await sandbox.execute(
                "import asyncio\n"
                "a, b, c = await asyncio.gather(tools.delay(1), tools.delay(2), tools.echo(3))\n"
                "return {'values': [a, b, c], 'names': [tool['name'] for tool in ALL_TOOLS]}"
            )
        )
        assert result.value == {"values": [1, 2, 3], "names": ["echo", "delay"]}
        # asyncio.gather over tool calls runs them concurrently on the host.
        assert most == 2

    async def test_exposes_tools_under_normalized_identifiers_and_lists_them_in_all_tools(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox(
            [
                CodemodeTool("my-tool", lambda _a, _c: "dash", description="Dashes"),
                CodemodeTool("my_tool", lambda _a, _c: "underscore", description="Shadowed"),
                CodemodeTool("mcp__docs__search", lambda _a, _c: "mcp"),
                CodemodeTool("class", lambda _a, _c: "keyword"),
            ]
        )
        result = success(
            await sandbox.execute(
                """
try:
    ALL_TOOLS.append({})
except AttributeError:
    pass
return {
    "all": list(ALL_TOOLS),
    "calls": [await tools.my_tool(), await tools["my-tool"](), await tools["my_tool"](), await tools.mcp__docs__search(), await tools.class_()],
}
"""
            )
        )
        assert result.value == {
            "all": [
                {"name": "my_tool", "description": "Dashes"},
                {"name": "mcp__docs__search", "description": ""},
                {"name": "class_", "description": ""},
            ],
            "calls": ["dash", "dash", "dash", "mcp", "keyword"],
        }

    async def test_passes_missing_arguments_and_results_through_as_none(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox([CodemodeTool("noop", lambda args, _context: args)])
        assert success(await sandbox.execute("return [await tools.noop(), await tools.noop(None)]")).value == [
            None,
            None,
        ]

    async def test_turns_tool_errors_into_catchable_exceptions_in_the_script(
        self, create_sandbox: CreateSandbox
    ) -> None:
        def fail(_args: Any, _context: CodemodeToolContext) -> Any:
            raise RuntimeError("tool exploded")

        sandbox = create_sandbox([CodemodeTool("fail", fail)])
        result = success(
            await sandbox.execute(
                """
try:
    await tools.fail()
    return "no error"
except Exception as error:
    return {"is_error": isinstance(error, Exception), "message": str(error), "name": type(error).__name__}
"""
            )
        )
        assert result.value == {"is_error": True, "message": "tool exploded", "name": "Exception"}
        assert [(call.name, call.status) for call in result.calls] == [("fail", "error")]

    async def test_reports_a_tool_result_json_cannot_hold_as_a_failed_call(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox([CodemodeTool("odd", lambda _a, _c: {1, 2})])
        result = success(
            await sandbox.execute("try:\n    await tools.odd()\nexcept Exception as error:\n    return str(error)")
        )
        assert result.value == "Object of type set is not JSON serializable"
        assert result.calls[0].status == "error"

    async def test_rejects_calls_to_unknown_tools(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        error = failure(await sandbox.execute("return await tools.missing()")).error
        assert (error.kind, error.name) == ("script", "AttributeError")

    async def test_aborts_unawaited_calls_when_the_script_returns(self, create_sandbox: CreateSandbox) -> None:
        aborted: list[bool] = []

        async def slow(_args: Any, context: CodemodeToolContext) -> Any:
            await context.signal.wait()
            aborted.append(True)
            raise RuntimeError("aborted")

        sandbox = create_sandbox([CodemodeTool("slow", slow)])
        result = success(await sandbox.execute("tools.slow()\nreturn 'early'"))
        assert result.value == "early"
        assert [(call.name, call.status) for call in result.calls] == [("slow", "cancelled")]
        await asyncio.sleep(0)
        assert aborted == [True]

    async def test_names_close_matches_when_a_script_reads_a_tool_that_does_not_exist(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox([echo_tool(), CodemodeTool("web-search", lambda _a, _c: "")])

        async def attempt(expression: str) -> Any:
            return value_of(await sandbox.execute(f"return {expression}"))

        expected = (
            "tools.Echo does not exist. Did you mean tools.echo? ALL_TOOLS lists every tool; search_tools(query) "
            'finds tools by topic. Check for a member with "Echo" in tools.'
        )
        assert await attempt("tools.Echo") == expected
        # Subscripts raise the same message, also when the script catches it.
        assert (
            value_of(
                await sandbox.execute("try:\n    tools['Echo']\nexcept AttributeError as error:\n    return str(error)")
            )
            == expected
        )
        assert "Did you mean tools.web_search?" in await attempt("tools.websearch")
        assert "Available: echo, web_search." in await attempt("tools.nothing")
        assert await attempt("['echo' in tools, 'nothing' in tools, 'web-search' in tools, 1 in tools]") == [
            True,
            False,
            True,
            False,
        ]

    async def test_supports_register_and_unregister_between_executions(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        echo = echo_tool()
        sandbox.register_tool(echo)
        with pytest.raises(ValueError, match="already registered"):
            sandbox.register_tool(echo)
        assert [tool.name for tool in sandbox.tools] == ["echo"]
        assert success(await sandbox.execute("return await tools.echo('a')")).value == "a"
        assert sandbox.unregister_tool("echo") is True
        assert success(await sandbox.execute("return 'echo' in tools")).value is False


class TestStoreAndLoad:
    async def test_reads_the_snapshot_and_reports_writes(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                """
seen = load("counter")
store("counter", seen + 1)
store("list", [1, {"a": None}])
store("old", None)
return [seen, load("counter"), load("missing"), load("old")]
""",
                store={"counter": 41, "old": "x"},
            )
        )
        assert result.value == [41, 42, None, None]
        # store(key, None) deletes the key, as storing undefined does upstream.
        assert result.store_writes == CodemodeStoreWrites({"counter": 42, "list": [1, {"a": None}]}, ["old"])

    async def test_returns_copies_so_mutating_a_loaded_value_does_not_change_the_store(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                'value = load("obj")\nvalue["a"] = 2\nkept = {"b": 1}\nstore("kept", kept)\nkept["b"] = 2\n'
                'return [load("obj")["a"], load("kept")["b"]]',
                store={"obj": {"a": 1}},
            )
        )
        assert result.value == [1, 1]
        assert result.store_writes.set == {"kept": {"b": 1}}

    async def test_rejects_invalid_keys_values_and_oversized_writes_inside_the_script(
        self, create_sandbox: CreateSandbox
    ) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                """
def attempt(fn):
    try:
        fn()
        return "ok"
    except Exception as error:
        return type(error).__name__

def fill():
    for i in range(8):
        store("k" + str(i), "x" * (200 * 1024))

return [
    attempt(lambda: store(1, "x")),
    attempt(lambda: load({})),
    attempt(lambda: store("fn", lambda: 1)),
    attempt(lambda: store("big", "x" * (300 * 1024))),
    attempt(fill),
    attempt(lambda: store("nan", float("nan"))),
]
"""
            )
        )
        # RangeError upstream is ValueError here.
        assert result.value == ["TypeError", "TypeError", "TypeError", "ValueError", "ValueError", "TypeError"]

    async def test_explains_oversized_writes(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        error = failure(await sandbox.execute('store("img", "x" * (300 * 1024))')).error
        assert 'store("img") value has 307202 characters of JSON' in error.message
        assert "Show images with image()" in error.message

    def test_reserves_the_store_and_load_names(self) -> None:
        execute = lambda _a, _c: None  # noqa: E731
        with pytest.raises(ValueError, match="Invalid global"):
            CodemodeSandbox(globals=[CodemodeTool("store", execute)])
        with pytest.raises(ValueError, match="Invalid global"):
            CodemodeSandbox(globals=[CodemodeTool("load", execute)])


class TestGlobals:
    async def test_exposes_globals_as_top_level_functions_without_recording_them_as_calls(
        self, new_sandbox: Callable[..., CodemodeSandbox]
    ) -> None:
        seen: list[Any] = []
        sandbox = new_sandbox(tools=[echo_tool()], globals=[CodemodeTool("attach", lambda args, _c: seen.append(args))])
        result = success(
            await sandbox.execute(
                """
await attach({"ref": 1})
attach("not awaited")
return [type(attach).__name__, await tools.echo(2)]
"""
            )
        )
        assert result.value == ["function", 2]
        assert [call.name for call in result.calls] == ["echo"]
        # Calls are handled in order, so an unawaited global still runs before the script settles.
        assert seen == [{"ref": 1}, "not awaited"]

    async def test_groups_namespaced_globals_and_spreads_arguments_on_request(
        self, new_sandbox: Callable[..., CodemodeSandbox]
    ) -> None:
        seen: list[Any] = []
        sandbox = new_sandbox(
            globals=[
                CodemodeTool("models.list", lambda args, _c: seen.append(args), spread=True),
                CodemodeTool("models.first", lambda args, _c: args),
            ]
        )
        result = success(
            await sandbox.execute(
                """
await models.list("classifier", None, 3)
await models.list()
await models.list("x", limit=2)
try:
    models.extra = 1
except Exception:
    pass
return ["list" in models, "first" in models, await models.first("a"), "extra" in models, repr(models)]
"""
            )
        )
        # Python: a spread global gets keyword arguments as one trailing object.
        assert result.value == [True, True, "a", False, "<codemode namespace models>"]
        assert seen == [["classifier", None, 3], [], ["x", {"limit": 2}]]

    async def test_names_the_members_of_a_namespace_when_a_script_reads_one_that_does_not_exist(
        self, new_sandbox: Callable[..., CodemodeSandbox]
    ) -> None:
        execute = lambda _a, _c: None  # noqa: E731
        sandbox = new_sandbox(
            globals=[CodemodeTool("models.classify", execute), CodemodeTool("models.generateImages", execute)]
        )
        error = failure(await sandbox.execute("await models.generateImage()")).error
        assert error.message == (
            "models.generateImage does not exist. Did you mean models.generateImages? "
            'Check for a member with "generateImage" in models.'
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
            "print",
            "class",
            "a.__x",
            "__x",
        ],
    )
    def test_rejects_invalid_and_reserved_global_names(self, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid global"):
            CodemodeSandbox(globals=[CodemodeTool(name, lambda _a, _c: None)])

    def test_rejects_a_global_that_conflicts_with_a_namespace(self) -> None:
        execute = lambda _a, _c: None  # noqa: E731
        with pytest.raises(ValueError, match="conflicts with the namespace"):
            CodemodeSandbox(globals=[CodemodeTool("models", execute), CodemodeTool("models.list", execute)])
        with pytest.raises(ValueError, match="already registered"):
            CodemodeSandbox(globals=[CodemodeTool("a", execute), CodemodeTool("a", execute)])


class TestLimitsAndLifetime:
    async def test_terminates_a_synchronous_infinite_loop_on_timeout(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        started = time.perf_counter()
        error = failure(await sandbox.execute("while True:\n    pass", timeout_ms=200)).error
        assert error.kind == "timeout"
        assert time.perf_counter() - started < 5
        # The pool replaces the killed worker.
        assert success(await sandbox.execute("return 1")).value == 1

    async def test_runs_without_a_deadline_when_timeout_ms_is_infinite(self, create_sandbox: CreateSandbox) -> None:
        async def wait(_args: Any, _context: CodemodeToolContext) -> Any:
            await asyncio.sleep(0.05)
            return "late"

        sandbox = create_sandbox([CodemodeTool("wait", wait)])
        result = await sandbox.execute("return await tools.wait()", timeout_ms=float("inf"))
        assert success(result).value == "late"

    async def test_fails_a_script_whose_wait_nothing_can_end_instead_of_hanging(
        self, create_sandbox: CreateSandbox
    ) -> None:
        # JavaScript can await a promise nothing settles; Monty's awaitables are calls, sleeps and
        # coroutines, and it rejects a coroutine awaited twice itself. Either way the script fails at once.
        sandbox = create_sandbox([echo_tool()])
        started = time.perf_counter()
        result = await sandbox.execute(
            "await tools.echo(1)\nasync def f():\n    return 1\nc = f()\nawait c\nawait c\nreturn 'never'",
            timeout_ms=float("inf"),
        )
        error = failure(result).error
        assert (error.kind, error.name) == ("script", "RuntimeError")
        assert "cannot reuse already awaited coroutine" in error.message
        assert time.perf_counter() - started < 5
        # Returning while a call is still pending is not a stall.
        returned = await sandbox.execute("tools.echo(2)\nreturn 'early'", timeout_ms=float("inf"))
        assert success(returned).value == "early"

    async def test_terminates_an_await_spinning_loop_on_timeout(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        for code in (
            "import asyncio\nwhile True:\n    await asyncio.sleep(0)",
            "async def f():\n    pass\nwhile True:\n    await f()",
        ):
            assert failure(await sandbox.execute(code, timeout_ms=200)).error.kind == "timeout"

    async def test_sleeps_for_real_in_execute(self, create_sandbox: CreateSandbox) -> None:
        # Not upstream: QuickJS has no timers; Monty's sleeps wait on the host and gathered ones overlap.
        sandbox = create_sandbox()
        started = time.perf_counter()
        result = await sandbox.execute(
            "import asyncio, time\nstart = time.monotonic()\n"
            "await asyncio.gather(asyncio.sleep(0.1), asyncio.sleep(0.1))\ntime.sleep(0.05)\n"
            "return time.monotonic() - start"
        )
        elapsed = success(result).value
        assert isinstance(elapsed, float) and 0.14 <= elapsed < 1
        assert time.perf_counter() - started < 2

    async def test_aborts_via_signal_and_cancels_in_flight_calls(self, create_sandbox: CreateSandbox) -> None:
        controller = AbortController()
        tool_signal: list[AbortSignal] = []
        called = asyncio.Event()

        async def hang(_args: Any, context: CodemodeToolContext) -> Any:
            tool_signal.append(context.signal)
            called.set()
            await asyncio.Event().wait()

        sandbox = create_sandbox([CodemodeTool("hang", hang)])
        first = asyncio.ensure_future(sandbox.execute("await tools.hang()\nreturn 'never'"))
        await called.wait()
        called.clear()
        controller.abort(RuntimeError("user cancelled"))
        assert tool_signal[0].aborted is False
        result = await sandbox.execute("await tools.hang()", signal=controller.signal)
        error = failure(result).error
        assert (error.kind, error.message) == ("aborted", "user cancelled")

        second = AbortController()
        pending = asyncio.ensure_future(sandbox.execute("await tools.hang()\nreturn 'never'", signal=second.signal))
        await called.wait()
        second.abort()
        aborted = failure(await pending)
        assert aborted.error.kind == "aborted"
        assert [(call.name, call.status) for call in aborted.calls] == [("hang", "cancelled")]
        assert tool_signal[-1].aborted is True

        await sandbox.close()
        closed = failure(await first).error
        assert (closed.kind, closed.message) == ("aborted", "Sandbox closed")

    # #10283: the host keeps all output, so a script that prints in a loop must not grow it without bound.
    @pytest.mark.parametrize("emit", ["text(s)", "print(s)", 'image("data:image/png;base64," + p)'])
    async def test_fails_a_script_whose_output_passes_the_limits_even_if_it_catches_the_error(
        self, create_sandbox: CreateSandbox, emit: str
    ) -> None:
        sandbox = create_sandbox(timeout_ms=60_000)
        result = failure(
            await sandbox.execute(
                f"""
s = "x" * (1 << 20)
p = "iVBORw0KGgoA" + "A" * (1 << 20)
while True:
    try:
        {emit}
    except BaseException:
        pass
"""
            )
        )
        assert (result.error.kind, result.error.name) == ("script", "ValueError")
        assert "script output exceeded" in result.error.message
        chars = sum(
            len(item.text) if isinstance(item, CodemodeTextOutput) else len(item.data) for item in result.output
        )
        assert MAX_OUTPUT_CHARS - (2 << 20) < chars <= MAX_OUTPUT_CHARS

    async def test_limits_output_items(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox(timeout_ms=60_000)
        empty = failure(await sandbox.execute('while True:\n    text("")'))
        assert empty.error.name == "ValueError"
        assert len(empty.output) == MAX_OUTPUT_ITEMS

    async def test_rejects_execute_after_close(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        await sandbox.close()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.execute("return 1")

    async def test_runs_executions_in_parallel_without_sharing_state(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        results = await asyncio.gather(
            sandbox.execute("import asyncio\nshared = 'a'\nawait asyncio.sleep(0.01)\nreturn shared"),
            sandbox.execute("import asyncio\nshared = 'b'\nawait asyncio.sleep(0.01)\nreturn shared"),
            sandbox.execute("try:\n    return shared\nexcept NameError:\n    return 'undefined'"),
        )
        assert [value_of(result) for result in results] == ["a", "b", "undefined"]

    async def test_turns_deep_recursion_into_a_catchable_recursion_error(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = await sandbox.execute(
            """
depth = 0
def dive():
    global depth
    depth += 1
    dive()
try:
    dive()
except RecursionError as error:
    return [type(error).__name__, depth > 500]
"""
        )
        assert success(result).value == ["RecursionError", True]

    async def test_caps_memory(self, new_sandbox: Callable[..., CodemodeSandbox]) -> None:
        sandbox = new_sandbox(memory_limit_bytes=8 * 1024 * 1024)
        result = failure(
            await sandbox.execute("parts = []\nwhile True:\n    parts.append('x' * (1 << 20) + str(len(parts)))")
        )
        assert (result.error.kind, result.error.name) == ("script", "MemoryError")
        assert "memory" in json.dumps(result.error.message)


class TestEscapeHatches:
    async def test_has_no_filesystem_network_subprocess_or_host_modules(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox()
        result = success(
            await sandbox.execute(
                """
import os
from pathlib import Path
attempts = {
    "open": lambda: open("/etc/passwd").read(),
    "write": lambda: open("/tmp/pi-codemode-escape", "w").write("x"),
    "listdir": lambda: os.listdir("/"),
    "getenv": lambda: os.getenv("HOME"),
    "environ": lambda: os.environ,
    "path": lambda: Path("/etc/passwd").read_text(),
    "path_write": lambda: Path("/tmp/pi-codemode-escape").write_text("x"),
    "globals": lambda: globals(),
    "builtins": lambda: __builtins__,
    "import": lambda: __import__("os"),
}
outcomes = {}
for name, attempt in attempts.items():
    try:
        attempt()
        outcomes[name] = "succeeded"
    except Exception as error:
        outcomes[name] = type(error).__name__
for module in ["subprocess", "socket", "importlib", "builtins", "ctypes", "threading", "multiprocessing", "urllib", "http", "shutil", "signal", "inspect", "weakref", "gc", "pydantic_monty"]:
    try:
        exec("import " + module)
        outcomes[module] = "succeeded"
    except Exception as error:
        outcomes[module] = type(error).__name__
return outcomes
"""
            )
        )
        outcomes = result.value
        assert isinstance(outcomes, dict)
        assert "succeeded" not in outcomes.values(), outcomes
        assert outcomes["open"] == outcomes["listdir"] == outcomes["path"] == "PermissionError"
        assert outcomes["subprocess"] == outcomes["socket"] == "ModuleNotFoundError"

    async def test_keeps_eval_and_exec_inside_the_sandbox(self, create_sandbox: CreateSandbox) -> None:
        # Code generation is allowed: it can only produce more code in the same sandbox.
        sandbox = create_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                """
def outcome(code):
    try:
        exec(code)
        return "ran"
    except Exception as error:
        return type(error).__name__
return [eval("1 + 1"), outcome("import subprocess"), outcome("open('/etc/passwd')"), await eval("tools.echo(3)")]
"""
            )
        )
        assert result.value == [2, "ModuleNotFoundError", "PermissionError", 3]

    # #10444: the prelude shares globals with the script, so rebinding them must not corrupt what reaches the host.
    async def test_ignores_rebound_builtins_and_patched_modules(self, create_sandbox: CreateSandbox) -> None:
        sandbox = create_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                """
import json
patched = []
try:
    json.dumps = lambda *args, **kwargs: "x"
except AttributeError:
    patched.append("json is read-only")
len = lambda value: 0
str = lambda value: "s"
isinstance = lambda value, kind: False
dict = None
store("k", [1])
text("still")
return [await tools.echo([2]), json.loads('{"a": 1}'), patched]
"""
            )
        )
        assert result.value == [[2], {"a": 1}, ["json is read-only"]]
        assert result.output == [CodemodeTextOutput("still")]
        assert result.store_writes.set == {"k": [1]}

    async def test_reports_forged_host_calls_as_a_broken_bridge(self, create_sandbox: CreateSandbox) -> None:
        # The prelude keeps the raw print to itself, so a script reaches the host only through calls; calls the
        # prelude never makes fail the run as a sandbox error.
        sandbox = create_sandbox([echo_tool()])
        for code, reason in (
            ("__codemode_fail__(1, 2)", "malformed failure"),
            ("await __codemode_call__(7, None, {})", "malformed tool call"),
            ("await __codemode_global__(0, None, {})", "malformed global call"),
        ):
            error = failure(await sandbox.execute(code)).error
            assert error.kind == "sandbox"
            assert error.message.startswith(f"Sandbox bridge broken: {reason}")
        unknown = failure(await sandbox.execute("not_defined_anywhere()")).error
        assert (unknown.kind, unknown.name) == ("script", "NameError")

    async def test_tampering_with_tools_only_affects_the_script(self, create_sandbox: CreateSandbox) -> None:
        # Upstream freezes `tools`; Monty has no read-only attributes, so a script can rebind its own view.
        sandbox = create_sandbox([echo_tool()])
        result = success(
            await sandbox.execute(
                """
original = tools.echo
tools.echo = lambda *args: "nope"
tools.extra = lambda *args: "nope"
return ["extra" in tools, tools.echo(), await tools["echo"]("still"), await original("also")]
"""
            )
        )
        assert result.value == [False, "nope", "still", "also"]
        assert [call.name for call in result.calls] == ["echo", "echo"]

    async def test_reports_errors_whose_message_is_not_a_string(self, create_sandbox: CreateSandbox) -> None:
        # Monty's exceptions take at most one string argument; anything else fails where it is raised.
        sandbox = create_sandbox()
        error = failure(await sandbox.execute("raise ValueError(42)")).error
        assert error.kind == "script"
        assert isinstance(error.name, str) and isinstance(error.message, str)
        assert "one string argument" in error.message


async def test_the_driver_fails_a_wait_that_no_pending_call_or_sleep_can_end() -> None:
    # Monty rejects the awaitables that could wait forever (see above), so the guard is checked directly:
    # a sandbox waiting on futures the host is not settling fails at once.
    from pi_codemode.runtime.monty import _Driver, _Ended
    from pi_codemode.runtime.protocol import DoneMessage, VmStartData

    messages: list[object] = []

    class Events:
        def message(self, message: object) -> None:
            messages.append(message)

        def failure(self, error: object) -> None:
            messages.append(error)

    class Waiting:
        pending_call_ids = (5,)

    driver = _Driver(None, VmStartData("", (), (), None, {}), Events(), 1000, "utc")
    with pytest.raises(_Ended):
        await driver._wait(Waiting())
    (done,) = messages
    assert isinstance(done, DoneMessage) and done.error is not None
    assert done.error["name"] == "RuntimeError"
    assert "can never finish" in str(done.error["message"])
