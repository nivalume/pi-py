"""Regenerate ``commits.json`` (the dataset) and ``reads.json`` (the reads compared) of this fixture.

``python generate.py <this directory>``. The TypeScript side of the fixture was then produced with upstream
pi-durable 1.3.0 under Node 22 (``node --conditions=source --experimental-strip-types``) by ``interop.mjs``::

    node interop.mjs write sqlite ts.sqlite .            # then: gzip -9 ts.sqlite
    node interop.mjs write jsonl ts_jsonl .
    node interop.mjs read sqlite <copy of ts.sqlite> . expected.json

``expected.json`` is what TypeScript reads back; ``test_storage_interop.py`` checks Python reads the same from
the TypeScript stores and from stores Python writes.
"""

from __future__ import annotations

import json
import sys

from pi_ai.serialization import to_json_value
from pi_ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage

out = sys.argv[1]
user = to_json_value(UserMessage([TextContent("héllo \ud800 😀")], timestamp=1700000000000))
assistant = to_json_value(
    AssistantMessage(
        [TextContent("hi"), ToolCall(id="c1", name="ls", arguments={"path": "."})],
        api="anthropic-messages",
        provider="anthropic",
        model="claude",
        usage=Usage(1, 2, 0, 0, 3),
        timestamp=1700000000001,
    )
)
tool = to_json_value(
    ToolResultMessage(
        tool_call_id="c1", tool_name="ls", content=[TextContent("a\nb")], is_error=False, timestamp=1700000000002
    )
)
commits = [
    [{"type": "conversation", "value": {"id": 1}}],
    [
        {
            "type": "entry",
            "value": {"id": 2, "conversationId": 1, "kind": "pi.user", "model": [user], "data": {"text": "héllo"}},
        },
        {"type": "entry", "value": {"id": 3, "conversationId": 1, "kind": "pi.assistant", "model": [assistant]}},
        {
            "type": "entry",
            "value": {
                "id": 4,
                "conversationId": 1,
                "kind": "marker",
                "head": 2,
                "edits": [{"kind": "replace", "from": 2, "to": 3, "messages": [tool]}],
            },
        },
    ],
    [
        {
            "type": "task",
            "value": {
                "id": 5,
                "conversationId": 1,
                "kind": "t.\ud801",
                "version": 1,
                "input": {"n": 1.5},
                "state": {"status": "pending", "checkpoint": {"phase": "a"}},
                "background": True,
                "abortRequested": False,
            },
        },
        {
            "type": "task",
            "value": {
                "id": 6,
                "conversationId": 1,
                "kind": "t.sleep",
                "version": 2,
                "input": None,
                "state": {"status": "sleeping", "checkpoint": {"phase": "b"}, "until": 61000},
                "background": False,
                "abortRequested": True,
                "memos": {"m": [1, 2]},
            },
        },
        {
            "type": "submission",
            "value": {
                "id": 7,
                "conversationId": 1,
                "requestId": "req-\ud800",
                "type": "input",
                "status": "placed",
                "entry": 2,
            },
        },
        {"type": "submission", "value": {"id": 8, "conversationId": 1, "type": "write", "status": "queued"}},
    ],
    [
        {"type": "conversation", "value": {"id": 9, "parent": {"conversationId": 1, "at": 3}}},
        {"type": "conversation", "value": {"id": 10, "owner": {"conversationId": 1, "taskId": 5}}},
    ],
    [
        {"type": "entry", "value": {"id": 11, "conversationId": 9, "kind": "fork", "head": 11}},
        {
            "type": "document.create",
            "record": {"id": 12, "kind": "session.settings", "scope": {"kind": "session"}},
            "content": {"kind": "base", "version": 1, "value": {"count": 0, "items": []}},
        },
        {
            "type": "document.create",
            "record": {
                "id": 13,
                "kind": "notes",
                "scope": {"kind": "conversation", "conversationId": 1},
                "history": "rewindable",
                "fork": "asOf",
            },
            "content": {"kind": "base", "version": 1, "value": {"text": "a"}},
        },
        {
            "type": "document.create",
            "record": {"id": 14, "kind": "fam", "key": "k-\ud800", "scope": {"kind": "task", "taskId": 5}},
            "content": {"kind": "base", "version": 1, "value": {}},
        },
        {
            "type": "document.create",
            "record": {
                "id": 15,
                "kind": "latest",
                "scope": {"kind": "conversation", "conversationId": 1},
                "history": "latest",
                "fork": "current",
            },
            "content": {"kind": "base", "version": 3, "value": {"v": 1}},
        },
    ],
    [
        {
            "type": "document.change",
            "id": 12,
            "content": {
                "kind": "delta",
                "version": 1,
                "ops": [["s", ["count"], 1], ["p", ["items"], 0, 0, ["x", "y"]]],
            },
        },
        {
            "type": "document.change",
            "id": 13,
            "content": {"kind": "delta", "version": 1, "ops": [["a", ["text"], "b"]]},
        },
    ],
    [
        {"type": "document.change", "id": 13, "content": {"kind": "base", "version": 2, "value": {"text": "ab!"}}},
        {"type": "document.change", "id": 15, "content": {"kind": "base", "version": 3, "value": {"v": 2}}},
        {
            "type": "task",
            "value": {
                "id": 5,
                "conversationId": 1,
                "kind": "t.\ud801",
                "version": 1,
                "input": {"n": 1.5},
                "state": {"status": "terminal", "outcome": {"status": "completed", "result": {"ok": True}}},
                "background": True,
                "abortRequested": False,
            },
        },
    ],
    [
        {
            "type": "document.change",
            "id": 13,
            "content": {"kind": "delta", "version": 2, "ops": [["r", {"text": "replaced"}]]},
        }
    ],
    [
        {
            "type": "document.copy",
            "record": {
                "id": 16,
                "kind": "notes",
                "scope": {"kind": "conversation", "conversationId": 9},
                "history": "rewindable",
                "fork": "asOf",
            },
            "source": {"id": 13, "at": 6},
        },
        {"type": "document.retire", "id": 14},
        {
            "type": "submission",
            "value": {
                "id": 7,
                "conversationId": 1,
                "requestId": "req-\ud800",
                "type": "input",
                "status": "done",
                "entry": 2,
                "answer": 3,
            },
        },
    ],
]
reads = []
for cid in (1, 9, 10, 99):
    reads.append(["conversation", cid])
reads += [
    ["scanConversations", {}, 2],
    ["scanConversations", {"ownerConversationId": 1}, 10],
    ["scanConversations", {"order": "descending"}, 10],
]
for eid in (2, 3, 4, 11, 50):
    reads.append(["entry", eid])
for cid, eid in ((9, 2), (9, 4), (9, 11), (1, 11)):
    reads.append(["visibleEntry", cid, eid])
for cid in (1, 9):
    reads += [
        ["findLatestHeadMarker", cid, None],
        ["scanEntries", {"conversationId": cid}, 100],
        ["scanEntries", {"conversationId": cid, "order": "ascending"}, 2],
    ]
reads.append(["findLatestHeadMarker", 9, 3])
for tid in (5, 6):
    reads.append(["task", tid])
reads += [
    ["scanTasks", {"status": "terminal"}, 10],
    ["scanTasks", {"kind": "t.\ud801"}, 10],
    ["scanTasks", {"background": False}, 10],
    ["scanTasks", {"status": "sleeping"}, 10],
]
reads += [
    ["submission", 7],
    ["submission", 8],
    ["scanSubmissions", {}, 10],
    ["scanSubmissions", {"status": "queued"}, 10],
    ["submissionByRequest", 1, "req-\ud800"],
    ["submissionByRequest", 1, "req-\ud801"],
]
for doc in (12, 13, 14, 15, 16):
    reads.append(["document", doc, "current"])
for at in (5, 6, 7, 8, 9):
    reads.append(["document", 13, at])
reads += [
    ["findDocument", {"kind": "notes", "scope": {"kind": "conversation", "conversationId": 1}}, "current"],
    ["findDocument", {"kind": "fam", "key": "k-\ud800", "scope": {"kind": "task", "taskId": 5}}, 6],
    ["findDocument", {"kind": "fam", "key": "k-\ud800", "scope": {"kind": "task", "taskId": 5}}, "current"],
    ["scanDocuments", {"scope": {"kind": "conversation", "conversationId": 1}, "at": "current"}, 10],
    ["scanDocuments", {"scope": {"kind": "session"}, "at": "current", "kind": "session.settings"}, 10],
    ["scanDocuments", {"scope": {"kind": "task", "taskId": 5}, "at": 7}, 10],
    ["mintId"],
    ["commit", [{"type": "entry", "value": {"id": 100, "conversationId": 9, "kind": "after"}}]],
    ["entry", 100],
    ["mintId"],
]
with open(f"{out}/commits.json", "w") as file:
    json.dump(commits, file)
with open(f"{out}/reads.json", "w") as file:
    json.dump(reads, file)
