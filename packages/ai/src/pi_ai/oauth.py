"""Type-only compatibility entry point for coding-agent extension OAuth declarations.

These legacy callback shapes are kept so extensions written against the older
``OAuthLoginCallbacks`` surface keep type-checking. Built-in flows use
:class:`pi_ai.OAuthAuth` with :class:`pi_ai.ProviderAuthInteraction` instead, and
this module never imports them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .auth.oauth._common import OAuthError
from .auth.types import OAuthCredentials

if TYPE_CHECKING:
    from .abort import AbortSignal

__all__ = [
    "OAuthError",
    "OAuthAuthInfo",
    "OAuthCredentials",
    "OAuthDeviceCodeInfo",
    "OAuthLoginCallbacks",
    "OAuthPrompt",
    "OAuthSelectOption",
    "OAuthSelectPrompt",
]


@dataclass(slots=True)
class OAuthPrompt:
    """Legacy extension OAuth prompt."""

    message: str
    placeholder: str | None = None
    allow_empty: bool | None = None


@dataclass(slots=True)
class OAuthAuthInfo:
    """Legacy extension OAuth authorization link."""

    url: str
    instructions: str | None = None


@dataclass(slots=True)
class OAuthDeviceCodeInfo:
    """Legacy extension OAuth device-code notification."""

    user_code: str
    verification_uri: str
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


@dataclass(slots=True)
class OAuthSelectOption:
    id: str
    label: str


@dataclass(slots=True)
class OAuthSelectPrompt:
    message: str
    options: list[OAuthSelectOption] = field(default_factory=list)


class OAuthLoginCallbacks(Protocol):
    """Callback surface retained only for coding-agent extension compatibility.

    ``on_progress`` and ``on_manual_code_input`` are optional (``None`` when unsupported).
    """

    #: Optional progress message sink.
    on_progress: Callable[[str], None] | None
    #: Optional manual code entry that races the browser callback.
    on_manual_code_input: Callable[[], Awaitable[str]] | None
    signal: AbortSignal | None

    def on_auth(self, info: OAuthAuthInfo) -> None: ...

    def on_device_code(self, info: OAuthDeviceCodeInfo) -> None: ...

    async def on_prompt(self, prompt: OAuthPrompt) -> str: ...

    async def on_select(self, prompt: OAuthSelectPrompt) -> str | None: ...
