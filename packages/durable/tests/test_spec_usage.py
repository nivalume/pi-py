"""Port of the session-only parts of ``test/spec-usage.test.ts`` (the usage examples of ``docs/spec.md``).

Upstream only compile-checks the spec's examples; the sequences that need nothing but a Session run
here, against a stand-in for the Harness's ``pi.live`` document (``harness/live.ts``, stage 3).

Deferred to stage 3 (they need the Harness, its registry, ``configure()``, or the tools of
``src/tools/``, which phase 1 does not port): section 7.1 extensions and host setup (``ContextFiles``,
``Skills``, ``Coding``, ``Permissions``, ``Reviewer``, ``Timing``, ``Venv``), section 7.2 plan mode,
section 7.3 subagents and the named-subagent spawn commit, section 7.4 ``Chat``, and the sequences
``childInToolCommit``, ``liveSettings``, ``containerEnv`` and ``host``. Section 5.1's ``Payment`` task is
defined here; running it needs the Harness.
"""

from __future__ import annotations

from typing import Any

import pytest
from durable_session_support import StorageBackend, context, create_conversation, open_test_session

from pi_durable import define_doc, define_task

#: Stand-in for the Harness's ``LiveDoc`` (``pi.live``): same kind and semantics, opaque state.
LiveDoc = define_doc(
    kind="pi.live", version=1, scope="conversation", history="latest", fork="initial", initial=lambda: {}
)


async def _noop(*_args: Any) -> None:
    return None


Follow = define_task(
    name="app.follow", version=1, initial=lambda _input: {"phase": "follow"}, phases={"follow": _noop}, abort=_noop
)


def test_defines_the_section_5_1_payment_task() -> None:
    """Section 5.1: a task with intent, effect, outcome."""

    async def prepare(_task: Any, runtime: Any, context: Any) -> None:
        await runtime.commit(
            lambda tx, current: {"status": "running", "checkpoint": {"phase": "charge", "key": "k"}}, context
        )

    async def charge(task: Any, runtime: Any, context: Any) -> None:
        key = task["state"]["checkpoint"]["key"]  # idempotent by key

        async def outcome(tx: Any, current: Any) -> Any:
            entry = await tx.append_entry(current["conversationId"], {"kind": "app.receipt", "data": {"id": key}})
            return {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": entry["id"]}}}

        await runtime.commit(outcome, context)

    async def abort(_task: Any, runtime: Any, context: Any) -> None:
        # The abort handler decides the outcome; returning without one faults the task.
        await runtime.commit(
            lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted", "reason": "user"}}, context
        )

    Payment = define_task(
        name="app.payment",
        version=1,
        initial=lambda _input: {"phase": "prepare"},
        phases={"prepare": prepare, "charge": charge},
        abort=abort,
    )
    assert Payment.definition.initial(None) == {"phase": "prepare"}
    assert set(Payment.definition.phases) == {"prepare", "charge"}


async def test_section_4_table_reads_before_the_first_table_write_documents_stay_usable(
    storage_backend: StorageBackend,
) -> None:
    storage, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)

    async def seed(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation_id))["generation"] = {"step": 1}

    await session.commit(seed, context)

    async def change(tx: Any) -> None:
        conversation = await tx.conversation(conversation_id)  # table read
        live = await tx.doc(LiveDoc, conversation_id)

        await tx.append_entry(conversation_id, {"kind": "app.message"})  # first table write
        del live["generation"]  # document mutation remains valid
        await tx.create_task(
            Follow, {}, {"ownership": {"kind": "conversation"}, "conversationId": conversation_id}
        )  # further table writes are fine
        assert conversation == {"id": conversation_id}

    await session.commit(change, context)
    assert await session.snapshot(LiveDoc, conversation_id, context) == {}
    kinds = sorted(write["type"] for write in storage.commits[-1])
    assert kinds == ["document.change", "entry", "task"]


async def test_section_3_4_drafts_are_revoked_after_their_commit(storage_backend: StorageBackend) -> None:
    _, session, _ = await open_test_session(storage_backend)
    conversation_id = await create_conversation(session)
    escaped: list[Any] = []

    async def change(tx: Any) -> None:
        escaped.append(await tx.doc(LiveDoc, conversation_id))

    await session.commit(change, context)
    with pytest.raises(TypeError):  # raises: the draft was revoked
        escaped[0]["generation"] = None
