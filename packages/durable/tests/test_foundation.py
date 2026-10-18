"""Tests of the stage-1 helpers that upstream covers only through larger suites (Python-only)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pi_ai.types import TextContent, ToolResultMessage, UserMessage
from pi_chord.delta import track
from pi_chord.json import copy_json
from pi_durable import StorageRequestError, define_doc, define_doc_family, id_from_number
from pi_durable.documents import (
    address_id,
    check_record_scope,
    check_record_version,
    document_create,
    materialize_document_value,
    resolve_address,
)
from pi_durable.harness.json import assign_json
from pi_durable.search import rank_documents, tokenize
from pi_durable.storage.codec import (
    clone_value,
    decode_entry,
    decode_storage_write,
    encode_entry,
    encode_storage_write,
)
from pi_durable.storage.scan import next_cursor, scan_start
from pi_durable.timers import host_timers
from pi_durable.types import EntryRecord, StorageWrite


def test_assign_json_writes_leaf_by_leaf_into_a_draft() -> None:
    tracker = track({"message": {"text": "he", "parts": [1], "gone": True}})
    change = tracker.begin_change()
    assign_json(change.state, "message", {"text": "hello", "parts": [1, 2]})
    prepared = change.prepare()
    assert prepared.value == {"message": {"text": "hello", "parts": [1, 2]}}
    # The longer string is an append, the new element a splice, the dropped key a delete: no full set.
    assert sorted(op[0] for op in prepared.ops) == ["a", "d", "p"]


def test_assign_json_compares_primitives_like_javascript() -> None:
    target: dict[str, Any] = {"flag": 1}
    assign_json(target, "flag", True)
    assert target["flag"] is True
    same = {"n": 1}
    assign_json(same, "n", 1.0)
    assert type(same["n"]) is int


def test_codec_round_trips_entries_and_writes() -> None:
    entry: EntryRecord = {
        "id": id_from_number(2),
        "conversationId": id_from_number(1),
        "kind": "pi.user",
        "model": [UserMessage([TextContent("hi")], timestamp=1)],
        "edits": [
            {"target": id_from_number(1), "action": "omit"},
            {
                "target": id_from_number(1),
                "action": "replace",
                "messages": [ToolResultMessage("c", "t", [TextContent("x")], timestamp=2)],
            },
        ],
        "data": {"a": [1]},
    }
    encoded = encode_entry(entry)
    assert copy_json(encoded) == encoded
    assert encoded["model"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1}]
    assert json.loads(json.dumps(encoded)) == encoded
    assert decode_entry(json.loads(json.dumps(encoded))) == entry

    write: StorageWrite = {
        "type": "document.change",
        "id": id_from_number(3),
        "content": {"kind": "delta", "version": 1, "ops": [("s", ("a", 0), {"b": 1}), ("r", {})]},
    }
    stored = json.loads(json.dumps(encode_storage_write(write)))
    assert stored["content"]["ops"] == [["s", ["a", 0], {"b": 1}], ["r", {}]]
    assert decode_storage_write(stored) == write


def test_clone_value_detaches_and_rejects_non_json() -> None:
    message = UserMessage("hi", timestamp=1)
    value = {"ops": [("s", ("a",), [1])], "model": [message]}
    cloned = clone_value(value)
    assert cloned == value
    assert cloned["ops"][0][2] is not value["ops"][0][2]
    assert cloned["model"][0] is not message
    with pytest.raises(TypeError):
        clone_value({"bad": object()})
    with pytest.raises(TypeError):
        clone_value({1: "non-string key"})


def test_scan_start_resolves_order_and_cursor() -> None:
    assert scan_start(None, None, "descending") == ("descending", None)
    assert scan_start("ascending", next_cursor(5, "ascending"), "descending") == ("ascending", 5)
    assert scan_start(None, {"after": 5}, "descending") == ("descending", 5)
    with pytest.raises(StorageRequestError, match="Invalid scan order"):
        scan_start("sideways", None, "ascending")
    with pytest.raises(StorageRequestError, match="Invalid storage cursor"):
        scan_start(None, {"after": "5"}, "ascending")
    with pytest.raises(
        StorageRequestError, match="The cursor continues a ascending scan; the query asks for descending"
    ):
        scan_start("descending", next_cursor(5, "ascending"), "ascending")


def test_resolves_document_addresses_and_create_records() -> None:
    family = define_doc_family(
        kind="app.notes", version=2, scope="conversation", history="latest", fork="current", initial=lambda seed: {}
    )
    resolved = resolve_address(family.definition, [7, "k", {"seed": 1}])
    assert resolved.address == {"kind": "app.notes", "scope": {"kind": "conversation", "conversationId": 7}, "key": "k"}
    assert resolved.id == '["app.notes","conversation",7,"k"]'
    assert resolved.next_argument == 2
    create = document_create(family.definition, resolved.address, id_from_number(9))
    assert create == {
        "id": 9,
        "kind": "app.notes",
        "key": "k",
        "scope": {"kind": "conversation", "conversationId": 7},
        "history": "latest",
        "fork": "current",
    }
    check_record_scope(family.definition, create)
    with pytest.raises(RuntimeError, match="requires migration from version 1"):
        check_record_version(family.definition, create, 1)
    with pytest.raises(RuntimeError, match="has newer version 3 than 2"):
        check_record_version(family.definition, create, 3)
    with pytest.raises(TypeError, match="requires a conversation ID"):
        resolve_address(family.definition, ["7"])

    session = define_doc(
        kind="app.counter",
        version=2,
        scope="session",
        initial=lambda: {"count": 0},
        migrate=lambda value, version: {"count": value["n"]},
    )
    session_address = resolve_address(session.definition, []).address
    assert address_id(session_address) == '["app.counter","session",null,null]'
    record = document_create(session.definition, session_address, id_from_number(10))
    assert materialize_document_value(session.definition, record, 1, {"n": 4}) == {"count": 4}
    with pytest.raises(TypeError, match="does not match the supplied definition semantics"):
        check_record_scope(family.definition, record)


def test_tokenizes_and_ranks_tool_search_documents() -> None:
    assert tokenize("searchGitHubIssues for the repo") == ["search", "git", "hub", "issue", "repo"]
    documents = [
        {"name": "read", "text": "Read a file from disk"},
        {"name": "issues", "text": "Search GitHub issues and pull requests"},
        {"name": "search", "text": "Search the web"},
    ]
    assert rank_documents("github issue search", documents, 2) == ["issues", "search"]  # type: ignore[arg-type]
    assert rank_documents("the", documents, 2) == []  # type: ignore[arg-type]


async def test_host_timers_fall_back_to_the_native_timer() -> None:
    reports: list[BaseException] = []
    fired: list[str] = []
    timers = host_timers(lambda callback, delay, options: None, reports.append)
    timers.foreground(lambda: fired.append("foreground"), 0)
    assert timers.background(lambda: fired.append("background"), 0) is None
    timers.foreground(lambda: fired.append("again"), 0)
    await asyncio.sleep(0.01)
    assert fired == ["foreground", "again"]
    assert len(reports) == 1 and "declined a timer that paces work" in str(reports[0])

    def raising(callback: Any, delay: float, options: Any) -> Any:
        raise RuntimeError("no timers")

    failing = host_timers(raising, reports.append)
    assert failing.background(lambda: None, 0) is None
    assert str(reports[-1]) == "no timers"
    disarm = host_timers(None, reports.append).foreground(lambda: fired.append("disarmed"), 0)
    disarm()
    await asyncio.sleep(0.01)
    assert "disarmed" not in fired
