"""Port of ``test/session-definitions.test.ts``.

Upstream's typing assertions (``expectTypeOf``, ``@ts-expect-error``) live in ``session_typing_cases.py``,
which :func:`test_static_session_typing_cases` type-checks with pyright; the runtime assertions are here,
plus a runtime pass over every owner, key, and seed overload.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from durable_session_support import StorageBackend, context

from pi_durable import MemoryStorage, create_session, define_doc, define_doc_family, define_entry, define_task

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def initial() -> dict[str, Any]:
    return {"value": 0}


SessionDoc = define_doc(kind="t.session", version=1, scope="session", initial=initial)
LatestDoc = define_doc(
    kind="t.latest", version=1, scope="conversation", history="latest", fork="current", initial=initial
)
RewindableDoc = define_doc(
    kind="t.rewindable", version=1, scope="conversation", history="rewindable", fork="asOf", initial=initial
)
TaskDoc = define_doc(kind="t.task", version=1, scope="task", initial=initial)
SessionFamily = define_doc_family(
    kind="t.session-family", version=1, scope="session", initial=lambda seed: {"value": seed}
)
ConversationFamily = define_doc_family(
    kind="t.conversation-family",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="initial",
    initial=lambda seed: {"value": seed},
)
TaskFamily = define_doc_family(kind="t.task-family", version=1, scope="task", initial=lambda seed: {"value": seed})


async def _noop(*_args: Any) -> None:
    return None


Work = define_task(
    name="t.work", version=1, initial=lambda _input: {"phase": "start"}, phases={"start": _noop}, abort=_noop
)


def test_validates_persisted_version_semantics() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        define_doc(kind="k", version=0, scope="session", initial=initial)
    with pytest.raises(TypeError, match="positive integer"):
        define_doc(kind="k", version=1.5, scope="session", initial=initial)  # type: ignore[arg-type]
    assert define_doc(kind="", version=1, scope="session", initial=initial).definition.kind == ""
    assert SessionDoc.definition.kind == "t.session"


async def test_resolves_every_owner_key_and_seed_overload(storage_backend: StorageBackend) -> None:
    session = create_session(await storage_backend.open())

    async def change(tx: Any) -> Any:
        conversation = await tx.create_conversation({"ownership": {"kind": "ownerless"}})
        conversation_id = conversation["id"]
        task_id = await tx.create_task(
            Work, None, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
        )
        (await tx.doc(SessionDoc))["value"] = 1
        (await tx.doc(LatestDoc, conversation_id))["value"] = 2
        (await tx.doc(RewindableDoc, conversation_id))["value"] = 3
        (await tx.doc(TaskDoc, task_id))["value"] = 4
        assert (await tx.doc(SessionFamily, "k", 5))["value"] == 5
        assert (await tx.doc(ConversationFamily, conversation_id, "k", 6))["value"] == 6
        assert (await tx.doc(TaskFamily, task_id, "k", 7))["value"] == 7
        return conversation_id, task_id

    conversation_id, task_id = await session.commit(change, context)
    assert await session.snapshot(SessionDoc, context) == {"value": 1}
    assert await session.snapshot(LatestDoc, conversation_id, context) == {"value": 2}
    assert await session.snapshot(RewindableDoc, conversation_id, context) == {"value": 3}
    assert await session.snapshot(TaskDoc, task_id, context) == {"value": 4}
    assert await session.snapshot(SessionFamily, "k", context) == {"value": 5}
    assert await session.snapshot(ConversationFamily, conversation_id, "k", context) == {"value": 6}
    assert await session.snapshot(TaskFamily, task_id, "k", context) == {"value": 7}

    async def retire(tx: Any) -> None:
        await tx.retire_doc(SessionDoc)
        await tx.retire_doc(LatestDoc, conversation_id)
        await tx.retire_doc(TaskDoc, task_id)
        await tx.retire_doc(SessionFamily, "k")
        await tx.retire_doc(ConversationFamily, conversation_id, "k")
        await tx.retire_doc(TaskFamily, task_id, "k")

    await session.commit(retire, context)
    assert await session.snapshot(SessionDoc, context) is None
    assert await session.snapshot(LatestDoc, conversation_id, context) is None
    assert await session.snapshot(TaskDoc, task_id, context) is None
    assert await session.snapshot(SessionFamily, "k", context) is None
    assert await session.snapshot(ConversationFamily, conversation_id, "k", context) is None
    assert await session.snapshot(TaskFamily, task_id, "k", context) is None
    assert await session.snapshot(RewindableDoc, conversation_id, context) == {"value": 3}

    # Arguments the token's scope requires are checked at runtime too.
    with pytest.raises(TypeError, match="requires a conversation ID"):
        await session.snapshot(LatestDoc, "k", context)
    with pytest.raises(TypeError, match="requires a conversation document"):
        await session.snapshot_as_of(SessionDoc, conversation_id, context)
    with pytest.raises(TypeError, match="requires an entry ID"):
        await session.snapshot_as_of(RewindableDoc, conversation_id, "entry", context)
    await session.close(context)


async def test_types_typed_entries_at_runtime() -> None:
    Note = define_entry("t.note")
    session = create_session(MemoryStorage())

    async def change(tx: Any) -> Any:
        conversation_id = (await tx.create_conversation({"ownership": {"kind": "ownerless"}}))["id"]
        note = await tx.append_entry(Note, conversation_id, {"data": {"text": "x"}})
        raw = await tx.append_entry(conversation_id, {"kind": "t.raw"})
        return note, raw

    note, raw = await session.commit(change, context)
    assert note["kind"] == "t.note" and note["data"] == {"text": "x"}

    async def read(tx: Any) -> Any:
        return await tx.entry(Note, note["id"]), await tx.entry(Note, raw["id"]), await tx.entry(raw["id"])

    typed, other_kind, untyped = await session.commit(read, context)
    assert typed == note
    assert other_kind is None
    assert untyped == raw
    await session.close(context)


def test_static_session_typing_cases(tmp_path: Path) -> None:
    """Every ``# pyright: ignore`` in ``session_typing_cases.py`` marks a type error, and every ``assert_type`` holds."""
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
        [str(pyright), "-p", str(config), "--outputjson", str(HERE / "session_typing_cases.py")],
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
