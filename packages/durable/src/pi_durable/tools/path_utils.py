"""Path resolution of tool arguments (port of ``tools/path-utils.ts``)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..env.types import get_or_throw

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..approval import PathAccess, PermissionSubject
    from ..env.types import ExecutionEnv

__all__ = ["path_subject", "resolve_read_tool_path", "resolve_tool_path"]

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def _normalize_tool_path(path: str) -> str:
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    """The absolute path a tool argument names: special spaces as spaces, a leading ``@`` dropped."""
    return get_or_throw(await env.absolute_path(_normalize_tool_path(path), context))


def path_subject(
    access: PathAccess,
) -> Callable[[dict[str, Any], ExecutionEnv | None, Context], Awaitable[PermissionSubject | None]]:
    """``permission_subject`` of a tool whose ``path`` argument it reads or writes: the path made absolute with the
    environment; without one, only an absolute path. A path that does not resolve has no subject."""

    async def subject(args: dict[str, Any], env: ExecutionEnv | None, context: Context) -> PermissionSubject | None:
        path = args.get("path")
        if not isinstance(path, str):
            return None
        normalized = _normalize_tool_path(path)
        if env is None:
            return {"type": "path", "path": normalized, "access": access} if normalized.startswith("/") else None
        resolved = await env.absolute_path(normalized, context)
        if resolved["ok"] is False:
            return None
        return {"type": "path", "path": resolved["value"], "access": access}

    return subject


async def resolve_read_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    """``resolve_tool_path``, then the first existing variant of it.

    Variants: a narrow no-break space before ``AM.``/``PM.`` (macOS screenshot names), NFD, and curly
    apostrophes.
    """
    resolved = await resolve_tool_path(env, path, context)
    nfd = unicodedata.normalize("NFD", resolved)
    variants = [
        resolved,
        _AM_PM.sub(lambda match: f"{_NARROW_NO_BREAK_SPACE}{match.group(1)}.", resolved),
        nfd,
        resolved.replace("'", "\u2019"),
        nfd.replace("'", "\u2019"),
    ]
    for variant in dict.fromkeys(variants):
        if get_or_throw(await env.exists(variant, context)):
            return variant
    return resolved
