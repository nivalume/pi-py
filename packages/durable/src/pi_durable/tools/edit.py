"""The ``edit`` tool (port of ``tools/edit.ts``)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from pi_ai import TextContent, Type

from ..harness.define import define_tool
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .env import require_env
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import path_subject, resolve_tool_path

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..env.types import FileError
    from ..harness.types import ToolExecutionApi, ToolExecutionResult, ToolRegistration

__all__ = ["EditToolDetails", "EditToolInput", "create_edit_tool"]

_REPLACE_EDIT_SCHEMA = Type.Object(
    {
        "oldText": Type.String(
            description=(
                "Exact text for one targeted replacement. It must be unique in the original file and must not "
                "overlap with any other edits[].oldText in the same call."
            )
        ),
        "newText": Type.String(description="Replacement text for this targeted edit."),
    }
)

_EDIT_SCHEMA = Type.Object(
    {
        "path": Type.String(description="Path to the file to edit (relative or absolute)"),
        "edits": Type.Array(
            _REPLACE_EDIT_SCHEMA,
            description=(
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same block or "
                "nearby lines, merge them into one edit instead."
            ),
        ),
    }
)


class EditToolInput(TypedDict):
    path: str
    edits: list[Edit]


class EditToolDetails(TypedDict):
    diff: str
    patch: str
    firstChangedLine: NotRequired[int]


def _is_single_edit_input(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("oldText"), str)  # pyright: ignore[reportUnknownMemberType]
        and isinstance(value.get("newText"), str)  # pyright: ignore[reportUnknownMemberType]
    )


def _reject_constant(name: str) -> Any:
    # JSON.parse has no NaN or Infinity.
    raise ValueError(f"Unexpected token {name}")


def _prepare_edit_arguments(input: Any) -> Any:
    """Repair shapes models commonly send.

    ``edits`` as a JSON string or as a single edit object, and a top-level ``oldText``/``newText`` pair.
    Works on a copy; the call's arguments stay unchanged.
    """
    if not isinstance(input, Mapping):
        return input
    args: dict[str, Any] = dict(input)  # pyright: ignore[reportUnknownArgumentType]
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed: Any = json.loads(edits, parse_constant=_reject_constant)
        except ValueError:
            pass
        else:
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit_input(parsed):
                args["edits"] = [parsed]
    elif _is_single_edit_input(edits):
        args["edits"] = [edits]

    if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
        return args
    current = args.get("edits")
    repaired: list[Any] = list(current) if isinstance(current, list) else []  # pyright: ignore[reportUnknownArgumentType]
    repaired.append({"oldText": args["oldText"], "newText": args["newText"]})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    rest["edits"] = repaired
    return rest


def _validate_edit_input(input: Mapping[str, Any]) -> tuple[str, Sequence[Edit]]:
    edits = input.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:  # pyright: ignore[reportUnknownArgumentType]
        raise RuntimeError("Edit tool input is invalid. edits must contain at least one replacement.")
    return input["path"], edits  # pyright: ignore[reportUnknownVariableType]


def _edit_access_error(path: str, error: FileError) -> RuntimeError:
    exception = RuntimeError(f"Could not edit file: {path}. Error code: {error.code}.")
    exception.__cause__ = error
    return exception


def _check_aborted(context: Context) -> None:
    signal = context.abort_signal
    if signal is not None and signal.aborted:
        raise RuntimeError("Operation aborted")


def create_edit_tool() -> ToolRegistration:
    """Edits one file by exact, unique, non-overlapping text replacements; details carry a diff and a patch."""

    async def execute(args: Any, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        path, edits = _validate_edit_input(args)
        env = require_env(api)
        absolute_path = await resolve_tool_path(env, path, context)

        async def mutate() -> ToolExecutionResult:
            _check_aborted(context)
            info = await env.file_info(absolute_path, context)
            if info["ok"] is False:
                raise _edit_access_error(path, info["error"])
            if info["value"]["kind"] not in ("file", "symlink"):
                raise RuntimeError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, context)
            if read_result["ok"] is False:
                raise _edit_access_error(path, read_result["error"])
            _check_aborted(context)

            split = strip_bom(read_result["value"])
            content = split["text"]
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            base_content = applied["baseContent"]
            new_content = applied["newContent"]
            _check_aborted(context)

            final_content = split["bom"] + restore_line_endings(new_content, original_ending)
            write_result = await env.write_file(absolute_path, final_content, context)
            if write_result["ok"] is False:
                raise _edit_access_error(path, write_result["error"])
            _check_aborted(context)

            diff_result = generate_diff_string(base_content, new_content)
            details: EditToolDetails = {
                "diff": diff_result["diff"],
                "patch": generate_unified_patch(path, base_content, new_content),
            }
            if diff_result["firstChangedLine"] is not None:
                details["firstChangedLine"] = diff_result["firstChangedLine"]
            return {
                "output": [TextContent(f"Successfully replaced {len(edits)} block(s) in {path}.")],
                "details": details,
            }

        return await with_file_mutation_queue(env, absolute_path, mutate, context)

    return define_tool(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
            "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions "
            "just to connect distant changes."
        ),
        parameters=_EDIT_SCHEMA,
        prepare_arguments=_prepare_edit_arguments,
        execute=execute,
        permission_subject=path_subject("write"),
    )
