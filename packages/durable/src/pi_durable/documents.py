"""Document definitions and address helpers (port of ``documents.ts``).

``define_doc()`` and ``define_doc_family()`` take upstream's definition fields as keyword arguments
(snake_case: ``checkpoint_when``) and return a token whose class records the scope, so typed access
(``tx.doc(token, ...)``) resolves the right overload::

    class Todos(TypedDict):
        items: list[str]

    TodosDoc: ConversationDocToken[Todos] = define_doc(
        kind="app.todos",
        version=1,
        scope="conversation",
        history="latest",  # or "rewindable" to read old values with snapshot_as_of()
        fork="initial",  # what a fork starts with: "initial", "current", or "asOf"
        initial=lambda: {"items": []},
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, TypeVar, overload

from pi_chord.delta import Op
from pi_chord.json import JsonObject, copy_json

from .ids import id_from_number
from .types import (
    CheckpointInfo,
    ConversationDocFamilyToken,
    ConversationDocToken,
    DocDefinition,
    DocFamilyToken,
    DocToken,
    DocumentAddress,
    DocumentCreate,
    DocumentFork,
    DocumentHistory,
    DocumentId,
    DocumentRecord,
    DocumentScope,
    DocumentScopeName,
    RewindableConversationDocFamilyToken,
    RewindableConversationDocToken,
    SessionDocFamilyToken,
    SessionDocToken,
    StoredDocument,
    TaskDocFamilyToken,
    TaskDocToken,
)

__all__ = [
    "AnyDocDefinition",
    "AnyDocToken",
    "ResolvedAddress",
    "address_id",
    "check_record_scope",
    "check_record_version",
    "define_doc",
    "define_doc_family",
    "document_create",
    "materialize_document",
    "materialize_document_value",
    "resolve_address",
]

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741 - upstream's name for family seeds

#: Erased definition shape used by the Session after overload resolution.
AnyDocDefinition = DocDefinition
#: Erased singleton or family token.
AnyDocToken = DocToken[Any] | DocFamilyToken[Any, Any]

_SCOPES = ("session", "conversation", "task")
_HISTORIES = ("latest", "rewindable")
_FORKS = ("asOf", "current", "initial")


@overload
def define_doc(
    *,
    kind: str,
    version: int,
    scope: Literal["session"],
    initial: Callable[[], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> SessionDocToken[T]: ...
@overload
def define_doc(
    *,
    kind: str,
    version: int,
    scope: Literal["conversation"],
    history: Literal["latest"],
    fork: Literal["current", "initial"],
    initial: Callable[[], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> ConversationDocToken[T]: ...
@overload
def define_doc(
    *,
    kind: str,
    version: int,
    scope: Literal["conversation"],
    history: Literal["rewindable"],
    fork: DocumentFork,
    initial: Callable[[], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> RewindableConversationDocToken[T]: ...
@overload
def define_doc(
    *,
    kind: str,
    version: int,
    scope: Literal["task"],
    initial: Callable[[], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> TaskDocToken[T]: ...
def define_doc(
    *,
    kind: str,
    version: int,
    scope: DocumentScopeName,
    initial: Callable[[], Any],
    history: DocumentHistory | None = None,
    fork: DocumentFork | None = None,
    migrate: Callable[[JsonObject, int], Any] | None = None,
    checkpoint_when: Callable[[Any, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> DocToken[Any]:
    """Define a singleton document of one scope; see the module docstring for an example."""
    definition = DocDefinition(
        kind=kind,
        version=version,
        scope=scope,
        initial=initial,
        history=history,
        fork=fork,
        migrate=migrate,
        checkpoint_when=checkpoint_when,
    )
    _validate_definition(definition)
    if scope == "session":
        return SessionDocToken(definition)
    if scope == "task":
        return TaskDocToken(definition)
    if history == "rewindable":
        return RewindableConversationDocToken(definition)
    return ConversationDocToken(definition)


@overload
def define_doc_family(
    *,
    kind: str,
    version: int,
    scope: Literal["session"],
    initial: Callable[[I], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> SessionDocFamilyToken[T, I]: ...
@overload
def define_doc_family(
    *,
    kind: str,
    version: int,
    scope: Literal["conversation"],
    history: Literal["latest"],
    fork: Literal["current", "initial"],
    initial: Callable[[I], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> ConversationDocFamilyToken[T, I]: ...
@overload
def define_doc_family(
    *,
    kind: str,
    version: int,
    scope: Literal["conversation"],
    history: Literal["rewindable"],
    fork: DocumentFork,
    initial: Callable[[I], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> RewindableConversationDocFamilyToken[T, I]: ...
@overload
def define_doc_family(
    *,
    kind: str,
    version: int,
    scope: Literal["task"],
    initial: Callable[[I], T],
    migrate: Callable[[JsonObject, int], T] | None = None,
    checkpoint_when: Callable[[T, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> TaskDocFamilyToken[T, I]: ...
def define_doc_family(
    *,
    kind: str,
    version: int,
    scope: DocumentScopeName,
    initial: Callable[[Any], Any],
    history: DocumentHistory | None = None,
    fork: DocumentFork | None = None,
    migrate: Callable[[JsonObject, int], Any] | None = None,
    checkpoint_when: Callable[[Any, Sequence[Op], CheckpointInfo], bool] | None = None,
) -> DocFamilyToken[Any, Any]:
    """Define a keyed document family; ``initial(seed)`` runs only when a member is absent."""
    definition = DocDefinition(
        kind=kind,
        version=version,
        scope=scope,
        initial=initial,
        history=history,
        fork=fork,
        family=True,
        migrate=migrate,
        checkpoint_when=checkpoint_when,
    )
    _validate_definition(definition)
    if scope == "session":
        return SessionDocFamilyToken(definition)
    if scope == "task":
        return TaskDocFamilyToken(definition)
    if history == "rewindable":
        return RewindableConversationDocFamilyToken(definition)
    return ConversationDocFamilyToken(definition)


def _validate_definition(definition: DocDefinition) -> None:
    version = definition.version
    if type(version) is not int or version < 1 or version > 2**53 - 1:
        raise TypeError(f"Document {definition.kind} version must be a positive integer")
    # Python callers have no overloads to check these, so the definition is validated at runtime.
    if definition.scope not in _SCOPES:
        raise TypeError(f"Document {definition.kind} has an invalid scope")
    if definition.scope == "conversation":
        if definition.history not in _HISTORIES or definition.fork not in _FORKS:
            raise TypeError(f"Document {definition.kind} requires history and fork semantics")
        if definition.history == "latest" and definition.fork == "asOf":
            raise TypeError(f"Document {definition.kind} cannot fork asOf without rewindable history")
    elif definition.history is not None or definition.fork is not None:
        raise TypeError(f"Document {definition.kind} declares history or fork outside a conversation scope")


class ResolvedAddress(NamedTuple):
    """Logical address plus its string identity for maps."""

    address: DocumentAddress
    id: str
    #: The index of ``args`` after the owner and family key.
    next_argument: int


def resolve_address(definition: DocDefinition, args: Sequence[Any]) -> ResolvedAddress:
    """Resolve an overloaded argument list (after the token) and return the index after the owner and family key."""
    index = 0
    scope: DocumentScope
    if definition.scope == "session":
        scope = {"kind": "session"}
    elif definition.scope == "conversation":
        scope = {"kind": "conversation", "conversationId": _owner_id(_arg(args, index), definition)}
        index += 1
    else:
        scope = {"kind": "task", "taskId": _owner_id(_arg(args, index), definition)}
        index += 1
    key: str | None = None
    if definition.family:
        key = _arg(args, index)
        index += 1
    address: DocumentAddress = (
        {"kind": definition.kind, "scope": scope}
        if key is None
        else {"kind": definition.kind, "scope": scope, "key": key}
    )
    return ResolvedAddress(address, address_id(address), index)


def _arg(args: Sequence[Any], index: int) -> Any:
    return args[index] if index < len(args) else None


def _owner_id(value: Any, definition: DocDefinition) -> Any:
    if type(value) is not int or not -(2**53 - 1) <= value <= 2**53 - 1:
        raise TypeError(f"Document {definition.kind} requires a {definition.scope} ID")
    return id_from_number(value)


def address_id(address: DocumentAddress) -> str:
    """Stable string identity of one logical address."""
    scope = address["scope"]
    if scope["kind"] == "session":
        owner = None
    elif scope["kind"] == "conversation":
        owner = scope["conversationId"]
    else:
        owner = scope["taskId"]
    return json.dumps(
        [address["kind"], scope["kind"], owner, address.get("key")], separators=(",", ":"), ensure_ascii=False
    )


def document_create(definition: DocDefinition, address: DocumentAddress, id: DocumentId) -> DocumentCreate:
    """Build the storage create record for a new incarnation at an address."""
    scope = address["scope"]
    record: dict[str, Any] = {"id": id, "kind": address["kind"]}
    if "key" in address:
        record["key"] = address["key"]
    record["scope"] = scope
    if scope["kind"] == "conversation":
        record["history"] = definition.history
        record["fork"] = definition.fork
    return record  # type: ignore[return-value]


def check_record_scope(definition: DocDefinition, record: DocumentCreate | DocumentRecord) -> None:
    """Reject typed access whose token disagrees with the persisted scope, history, or fork semantics."""
    scope = record["scope"]
    if scope["kind"] != definition.scope or (
        scope["kind"] == "conversation"
        and (record.get("history") != definition.history or record.get("fork") != definition.fork)
    ):
        raise TypeError(f"Document {record['id']} ({record['kind']}) does not match the supplied definition semantics")


def check_record_version(definition: DocDefinition, record: DocumentCreate | DocumentRecord, version: int) -> None:
    """Reject typed access to a stored version the supplied definition cannot use."""
    if version > definition.version:
        raise RuntimeError(
            f"Document {record['id']} ({record['kind']}) has newer version {version} than {definition.version}"
        )
    if version < definition.version and definition.migrate is None:
        raise RuntimeError(f"Document {record['id']} ({record['kind']}) requires migration from version {version}")


def materialize_document(definition: DocDefinition, stored: StoredDocument) -> JsonObject:
    """Validate and materialize a detached stored value for typed access."""
    return materialize_document_value(definition, stored["record"], stored["version"], stored["value"])


def materialize_document_value(
    definition: DocDefinition, record: DocumentCreate | DocumentRecord, version: int, value: JsonObject
) -> JsonObject:
    """Validate and materialize one detached value before its first persisted incarnation."""
    check_record_scope(definition, record)
    check_record_version(definition, record, version)
    if version == definition.version:
        return value
    assert definition.migrate is not None
    return copy_json(definition.migrate(value, version))  # type: ignore[return-value]
