"""Port of ``test/types.test.ts``.

Upstream's assertions are mostly compile-time (``expectTypeOf`` and ``@ts-expect-error``). Their
Python counterparts live in ``typing_cases.py``, which :func:`test_static_typing_cases` type-checks
with pyright; the runtime assertions are here.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from pi_durable import (
    ROOT_CONVERSATION_ID,
    ConversationDocToken,
    Entry,
    RewindableConversationDocFamilyToken,
    SessionDocToken,
    TaskDocToken,
    ToolResultEntry,
    UserEntry,
    define_doc,
    define_doc_family,
    define_entry,
    define_extension,
    hook,
    id_from_number,
    seq_from_number,
)
from pi_durable.types import ConversationId, TaskId

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_typing_cases() -> Any:
    spec = importlib.util.spec_from_file_location("durable_typing_cases", HERE / "typing_cases.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_brands_numeric_ids_by_record_kind_and_carries_task_result_types() -> None:
    conversation_id: ConversationId = id_from_number(1)
    task_id: TaskId[int] = id_from_number(4)
    assert type(conversation_id) is int
    assert json.dumps(task_id) == "4"
    assert TaskId(4) == 4 and type(TaskId(4)) is int
    assert seq_from_number(1) == 1
    assert ROOT_CONVERSATION_ID == 1


def test_types_tasks_erased_into_extensions_and_hooks() -> None:
    cases = load_typing_cases()
    stepper = cases.Stepper
    assert stepper.definition.name == "test.stepper"
    assert stepper.definition.version == 1
    assert stepper.definition.initial({"steps": 2}) == {"phase": "plan", "steps": 2}
    assert set(stepper.definition.phases) == {"plan", "run"}
    registration = hook(stepper, {"before_step": cases.before_step})
    assert registration.task == "test.stepper"
    assert registration.handlers["before_step"] is cases.before_step
    extension = define_extension("stepper", tasks=[stepper], hooks=[registration])
    assert extension.tasks == [stepper] and extension.hooks == [registration]


def test_defines_entries_with_narrowing_guards() -> None:
    notes: Entry[dict[str, str]] = define_entry("app.note")
    note = {"id": id_from_number(2), "conversationId": ROOT_CONVERSATION_ID, "kind": "app.note", "data": {"a": "b"}}
    assert notes.is_(note)  # type: ignore[arg-type]
    assert not notes.is_(None)
    assert not UserEntry.is_(note)  # type: ignore[arg-type]
    assert ToolResultEntry.kind == "pi.tool-result"
    with pytest.raises(TypeError, match="Entry kind must be a non-empty string"):
        define_entry("")


def test_defines_documents_with_scope_tokens() -> None:
    session = define_doc(kind="app.session", version=1, scope="session", initial=lambda: {"n": 0})
    latest = define_doc(
        kind="app.latest", version=1, scope="conversation", history="latest", fork="initial", initial=lambda: {}
    )
    task = define_doc(kind="app.task", version=2, scope="task", initial=lambda: {}, migrate=lambda value, _: value)
    family = define_doc_family(
        kind="app.family",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="asOf",
        initial=lambda seed: {"seed": seed},
    )
    assert isinstance(session, SessionDocToken)
    assert type(latest) is ConversationDocToken
    assert isinstance(task, TaskDocToken)
    assert isinstance(family, RewindableConversationDocFamilyToken)
    assert family.definition.family and family.definition.initial(3) == {"seed": 3}
    with pytest.raises(TypeError, match="version must be a positive integer"):
        define_doc(kind="app.bad", version=0, scope="session", initial=lambda: {})
    with pytest.raises(TypeError, match="requires history and fork"):
        define_doc(kind="app.bad", version=1, scope="conversation", initial=lambda: {})  # type: ignore[call-overload]


def test_static_typing_cases(tmp_path: Path) -> None:
    """Every ``# pyright: ignore`` in ``typing_cases.py`` marks a type error, and every ``assert_type`` holds."""
    pyright = Path(sys.executable).with_name("pyright")
    if not pyright.exists():
        found = shutil.which("pyright")
        if found is None:
            pytest.skip("pyright unavailable")
        pyright = Path(found)
    config = tmp_path / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "venvPath": str(ROOT),
                "venv": ".venv",
                "pythonVersion": "3.11",
                "typeCheckingMode": "basic",
                "reportUnnecessaryTypeIgnoreComment": "error",
            }
        )
    )
    result = subprocess.run(
        [str(pyright), "-p", str(config), "--outputjson", str(HERE / "typing_cases.py")],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    report = json.loads(result.stdout)
    problems = [
        f"{item['range']['start']['line'] + 1}: {item['message']}"
        for item in report["generalDiagnostics"]
        if item["severity"] == "error"
    ]
    assert report["summary"]["filesAnalyzed"] == 1
    assert problems == []
