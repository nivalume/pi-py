"""Messages between the host side of an execution and the sandbox that runs it (port of ``protocol.ts``).

Tool arguments, results and values cross as JSON text, as upstream: the sandbox side
serializes them (and so checks they are JSON), the host parses them, and each side gets its
own copy. Both sides run on one event loop here, so the messages are plain dataclasses
rather than structured-clone payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..types import CodemodeDuring, CodemodeOutputItem

__all__ = [
    "CallMessage",
    "CrashMessage",
    "DoneMessage",
    "HostToVmMessage",
    "IdleMessage",
    "OutputMessage",
    "ResultMessage",
    "StartedMessage",
    "VmStartData",
    "VmToHostMessage",
    "is_vm_to_host_message",
]


@dataclass(slots=True, frozen=True)
class VmStartData:
    """Everything a sandbox needs to run one script, independent of where it runs."""

    code: str
    #: ``(name, js_name, description)``: ``tools.<js_name>`` and ``tools["<name>"]`` call the tool.
    tools: tuple[tuple[str, str, str], ...]
    #: ``(name, spread)``; ``a.b`` names are grouped into a namespace object ``a``.
    globals: tuple[tuple[str, bool], ...]
    memory_limit_bytes: int | None
    #: Snapshot for ``load()``: key to JSON text.
    store: dict[str, str]
    #: Run deterministically (``CodemodeSandboxBase.start()``): ``(started_at, seed)``. UTC, a
    #: virtual clock starting at ``started_at`` (ms since the epoch) that moves only with
    #: delivered results and sleeps, randomness from ``seed``, results applied in ``seq``
    #: order, and ``started``/``idle`` messages.
    deterministic: tuple[float, int] | None = None


@dataclass(slots=True, frozen=True)
class CallMessage:
    """The script called a tool or a global. ``args`` is JSON text, ``None`` without an argument."""

    id: int
    target: Literal["tool", "global"]
    name: str
    args: str | None


@dataclass(slots=True, frozen=True)
class OutputMessage:
    item: CodemodeOutputItem


@dataclass(slots=True, frozen=True)
class DoneMessage:
    """The script finished.

    Success: ``value`` is JSON text (``None`` for no value) and ``writes`` lists ``(key, json)``
    for ``store()`` and ``(key, None)`` for deletions. Failure: ``error`` holds ``name``,
    ``message`` and ``stack``. Deterministic mode: ``during`` is the result the sandbox was
    processing when the script finished (0: its first run) and the calls issued by then.
    """

    ok: bool
    value: str | None = None
    writes: tuple[tuple[str, str | None], ...] = ()
    error: dict[str, str | None] | None = None
    during: CodemodeDuring | None = None


@dataclass(slots=True, frozen=True)
class CrashMessage:
    """The sandbox failed outside the script's control, for example a crashed worker."""

    message: str


@dataclass(slots=True, frozen=True)
class StartedMessage:
    """Deterministic mode, before the script runs: what the sandbox is, so a replay can refuse a different one."""

    fingerprint: str


@dataclass(slots=True, frozen=True)
class IdleMessage:
    """Deterministic mode, after the script waits again: having processed result ``seq`` (0: the
    script's first run), it has issued ``issued`` calls in total."""

    seq: int
    issued: int


VmToHostMessage: TypeAlias = CallMessage | OutputMessage | DoneMessage | CrashMessage | StartedMessage | IdleMessage


@dataclass(slots=True, frozen=True)
class ResultMessage:
    """A call's answer. ``payload`` is the JSON result when ``ok`` (``None``: no value), otherwise
    the error message. Deterministic mode: ``seq`` numbers the results from 1 in the order the
    sandbox applies them, and its clock reads ``now`` while it processes this one;
    ``properties`` is the JSON object of a failure's extra properties."""

    id: int
    ok: bool
    payload: str | None
    seq: int | None = None
    now: float | None = None
    properties: str | None = None


HostToVmMessage: TypeAlias = ResultMessage

_VM_TO_HOST = (CallMessage, OutputMessage, DoneMessage, CrashMessage, StartedMessage, IdleMessage)


def is_vm_to_host_message(value: object) -> bool:
    return isinstance(value, _VM_TO_HOST)
