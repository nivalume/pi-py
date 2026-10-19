"""Public data types of pi_codemode (port of ``types.ts``).

Everything here is plain data and needs no sandbox: tools and globals the host exposes, the
items a script outputs, the result of an execution, and the records of a deterministic run
(:meth:`pi_codemode.CodemodeSandboxBase.start`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NotRequired, Protocol, TypeAlias, TypedDict

from pi_ai.abort import AbortSignal
from pi_chord.context import BACKGROUND_CONTEXT, Context, with_abort_signal
from pi_chord.json import JsonValue

if TYPE_CHECKING:
    from pi_ai.types import ImageContent, TextContent

__all__ = [
    "CodemodeCall",
    "CodemodeCallStatus",
    "CodemodeDeclaredGlobal",
    "CodemodeDeclaredTool",
    "CodemodeDuring",
    "CodemodeError",
    "CodemodeErrorKind",
    "CodemodeErrorOutcome",
    "CodemodeFailure",
    "CodemodeHostCall",
    "CodemodeImageOutput",
    "CodemodeJsonSchema",
    "CodemodeOkOutcome",
    "CodemodeOutcome",
    "CodemodeOutputItem",
    "CodemodeResult",
    "CodemodeStoreWrites",
    "CodemodeSuccess",
    "CodemodeTextOutput",
    "CodemodeTool",
    "CodemodeToolContext",
    "CodemodeToolDeclaration",
    "CodemodeToolExecute",
    "CodemodeToolLike",
]

#: A JSON Schema document. Only used to render declarations; values are not validated against it.
CodemodeJsonSchema: TypeAlias = "Mapping[str, Any] | bool"


@dataclass(slots=True, frozen=True)
class CodemodeToolContext:
    """What a tool's ``execute`` gets besides its arguments."""

    #: Aborted when the script finishes (including unawaited calls), the execution times
    #: out, the caller aborts, or the sandbox is closed.
    signal: AbortSignal

    @property
    def context(self) -> Context:
        """A pi_chord :class:`~pi_chord.context.Context` cancelled with :attr:`signal`."""
        return with_abort_signal(self.signal, BACKGROUND_CONTEXT)


#: ``execute(args, context)``: ``args`` is whatever the script passed, after a JSON round
#: trip (``None`` without arguments). The return value must be JSON-serializable; a raised
#: exception surfaces in the script as an ``Exception`` with the same message.
CodemodeToolExecute: TypeAlias = Callable[[JsonValue, CodemodeToolContext], "Awaitable[Any] | Any"]


class CodemodeToolLike(Protocol):
    """What declarations read from a tool or global: :class:`CodemodeTool` or :class:`CodemodeToolDeclaration`."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str | None: ...
    @property
    def input_schema(self) -> CodemodeJsonSchema | None: ...
    @property
    def output_schema(self) -> CodemodeJsonSchema | None: ...
    @property
    def signature(self) -> str | None: ...


@dataclass(slots=True)
class CodemodeTool:
    """A function exposed to scripts, as a tool (``tools.<name>``) or a global.

    The script calls tools as ``tools.<id>(...)``, where ``<id>`` is the name with characters
    that are not valid in identifiers replaced by ``_`` (see
    :func:`~pi_codemode.to_codemode_identifier`), and also as ``tools["<name>"](...)``.
    Globals are called as ``<name>(...)`` and must be identifiers, or ``<namespace>.<member>``,
    which groups them into a namespace object.
    """

    name: str
    execute: CodemodeToolExecute
    #: Shown as a docstring by :func:`~pi_codemode.render_declarations`, and listed in ``ALL_TOOLS`` for tools.
    description: str | None = None
    #: Schema of the argument. Rendered as the parameters; ``Any`` when omitted.
    input_schema: CodemodeJsonSchema | None = None
    #: Schema of the result. Rendered as the return type; ``Any`` when omitted.
    output_schema: CodemodeJsonSchema | None = None
    #: Globals only: ``execute`` receives all positional arguments as a list instead of the
    #: first one (keyword arguments, if any, as one trailing object).
    spread: bool = False
    #: Globals only: Python parameter list and return annotation for the declarations, for
    #: example ``"(type: str, id: str | None = None) -> list[Model]"``. Replaces the
    #: rendering from the schemas.
    signature: str | None = None


@dataclass(slots=True, frozen=True)
class CodemodeToolDeclaration:
    """A tool or global as declarations see it, without an implementation."""

    name: str
    description: str | None = None
    input_schema: CodemodeJsonSchema | None = None
    output_schema: CodemodeJsonSchema | None = None
    spread: bool = False
    signature: str | None = None


@dataclass(slots=True, frozen=True)
class CodemodeTextOutput:
    """A text item: from ``text()``, or from ``print()`` with ``console=True``."""

    type: ClassVar[Literal["text"]] = "text"
    text: str
    console: bool = False

    def to_content(self) -> TextContent:
        """The item as pi_ai ``TextContent``."""
        from pi_ai.types import TextContent

        return TextContent(self.text)


@dataclass(slots=True, frozen=True)
class CodemodeImageOutput:
    """An image item from ``image()``. ``data`` is base64; ``mime_type`` is detected from it."""

    type: ClassVar[Literal["image"]] = "image"
    data: str
    mime_type: str

    def to_content(self) -> ImageContent:
        """The item as pi_ai ``ImageContent``."""
        from pi_ai.types import ImageContent

        return ImageContent(self.data, self.mime_type)


#: One item of the script's output, in the order the script produced it.
CodemodeOutputItem: TypeAlias = "CodemodeTextOutput | CodemodeImageOutput"

CodemodeCallStatus: TypeAlias = Literal["ok", "error", "cancelled"]


@dataclass(slots=True)
class CodemodeCall:
    """One tool call the script made. Calls to globals are not recorded."""

    name: str
    status: CodemodeCallStatus
    duration_ms: float


#: ``script``: the script raised or failed to parse; ``name`` and ``stack`` come from its error.
#: ``timeout``: the deadline expired; the execution ended and the sandbox was asked to stop.
#: ``aborted``: the caller's signal fired or the sandbox was closed.
#: ``sandbox``: the sandbox or its transport failed outside the script's control (a crashed
#: worker, a pool that cannot start, a broken bridge).
CodemodeErrorKind: TypeAlias = Literal["script", "timeout", "aborted", "sandbox"]


@dataclass(slots=True, frozen=True)
class CodemodeError:
    kind: CodemodeErrorKind
    message: str
    name: str | None = None
    stack: str | None = None


@dataclass(slots=True)
class CodemodeStoreWrites:
    """Keys the script changed with ``store()``. Only successful executions report writes."""

    set: dict[str, JsonValue] = field(default_factory=dict)
    #: Keys stored as ``None``.
    delete: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class CodemodeDuring:
    """In a :meth:`~pi_codemode.CodemodeSandboxBase.start` run that the script itself ended:
    the result being processed when it did (``0`` for its first run), and the calls issued by then."""

    seq: int
    issued: int


@dataclass(slots=True)
class CodemodeSuccess:
    """The script returned (``value``), ran to its end (``None``), or called ``exit()`` (``None``)."""

    value: JsonValue
    output: list[CodemodeOutputItem] = field(default_factory=list)
    calls: list[CodemodeCall] = field(default_factory=list)
    store_writes: CodemodeStoreWrites = field(default_factory=CodemodeStoreWrites)
    during: CodemodeDuring | None = None
    ok: Literal[True] = field(default=True, init=False)


@dataclass(slots=True)
class CodemodeFailure:
    """The execution failed; ``output`` is kept up to the failure."""

    error: CodemodeError
    output: list[CodemodeOutputItem] = field(default_factory=list)
    calls: list[CodemodeCall] = field(default_factory=list)
    during: CodemodeDuring | None = None
    ok: Literal[False] = field(default=False, init=False)


#: ``execute()`` and ``start()`` never raise for script failures; check ``result.ok``.
CodemodeResult: TypeAlias = "CodemodeSuccess | CodemodeFailure"


@dataclass(slots=True, frozen=True)
class CodemodeHostCall:
    """A call the script made in a ``start()`` run, answered with :meth:`~pi_codemode.CodemodeRun.deliver`."""

    #: The sandbox's own number for the call, ``1``, ``2``, ... in the order the script made its
    #: calls (tools and globals).
    id: int
    target: Literal["tool", "global"]
    name: str
    #: What the script passed, after a JSON round trip; for spread globals the argument list.
    args: JsonValue


@dataclass(slots=True, frozen=True)
class CodemodeDeclaredTool:
    """A tool as a ``start()`` run's script sees it: ``tools.<js_name>`` and its ``ALL_TOOLS`` entry."""

    name: str
    #: The identifier the script uses. Default: ``to_codemode_identifier(name)``. A host that
    #: replays a run passes the one the run started with, so a changed mapping cannot rename a
    #: tool under a resumed script.
    js_name: str | None = None
    description: str | None = None


@dataclass(slots=True, frozen=True)
class CodemodeDeclaredGlobal:
    """A global as a ``start()`` run's script sees it; the same names as ``globals`` allow."""

    name: str
    spread: bool = False


class CodemodeOkOutcome(TypedDict):
    """A call's value. Without ``value`` the call resolves with ``None``."""

    ok: Literal[True]
    value: NotRequired[JsonValue]


class CodemodeErrorOutcome(TypedDict):
    """A call's failure: the script's ``await`` raises ``Exception(message)``.

    ``properties`` (a JSON object) is validated and accepted for parity with upstream, but
    Monty exceptions carry only a message, so the script does not see them.
    """

    ok: Literal[False]
    message: str
    properties: NotRequired[dict[str, JsonValue]]


#: A call's answer in a ``start()`` run; plain JSON, so a host can journal it as is.
CodemodeOutcome: TypeAlias = "CodemodeOkOutcome | CodemodeErrorOutcome"
