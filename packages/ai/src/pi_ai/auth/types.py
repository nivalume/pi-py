"""Auth types: credentials, auth methods, login interaction protocol.

Credentials are plain dicts in exactly the shape of pi's ``auth.json``::

    {"type": "api_key", "key": "sk-...", "env": {"CLOUDFLARE_ACCOUNT_ID": "..."}}
    {"type": "oauth", "access": "...", "refresh": "...", "expires": 1767225600000, ...}

so credential files are shared with the TypeScript tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from ..abort import AbortSignal

__all__ = [
    "ModelAuth",
    "ApiKeyCredential",
    "OAuthCredential",
    "OAuthCredentials",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "AuthContext",
    "AuthResult",
    "AuthCheck",
    "AuthType",
    "AuthSelectOption",
    "AuthPrompt",
    "AuthInfoLink",
    "AuthEvent",
    "AuthInteraction",
    "ProviderAuthInteraction",
    "ApiKeyAuth",
    "OAuthAuth",
    "LoginOptions",
    "ProviderAuth",
]

AuthType: TypeAlias = Literal["api_key", "oauth"]


@dataclass(slots=True)
class ModelAuth:
    """Request auth for a single model request: an API key, headers, and/or a base URL."""

    api_key: str | None = None
    headers: dict[str, str | None] | None = None
    base_url: str | None = None


class ApiKeyCredential(TypedDict, total=False):
    type: Literal["api_key"]
    key: str
    #: Provider-scoped environment/config values such as Cloudflare account ids.
    env: dict[str, str]


class OAuthCredentials(TypedDict, total=False):
    refresh: str
    access: str
    #: Expiry as a Unix timestamp in milliseconds.
    expires: int


class OAuthCredential(OAuthCredentials, total=False):
    type: Literal["oauth"]


Credential: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class CredentialInfo:
    """Non-secret credential metadata for account/status enumeration."""

    provider_id: str
    type: AuthType


class CredentialStore(Protocol):
    """App-owned credential storage keyed by provider id, one credential per provider.

    ``modify`` is the only write path: a serialized read-modify-write, so OAuth refresh
    under the lock cannot double-refresh a rotated token. Methods raise only on storage
    failure; ``read`` returns ``None`` for missing entries.
    """

    async def read(self, provider_id: str, *, signal: AbortSignal | None = None) -> Credential | None: ...

    async def list(self, *, signal: AbortSignal | None = None) -> Sequence[CredentialInfo]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        *,
        signal: AbortSignal | None = None,
    ) -> Credential | None: ...

    async def delete(self, provider_id: str, *, signal: AbortSignal | None = None) -> None: ...


class AuthContext(Protocol):
    """Environment access for auth resolution. Injectable for tests and sandboxes."""

    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...


@dataclass(slots=True)
class AuthResult:
    auth: ModelAuth
    #: Provider-scoped environment/config values resolved from credentials and ambient context.
    env: dict[str, str] | None = None
    #: Human-readable source for status UI: ``"ANTHROPIC_API_KEY"``, ``"OAuth"``, ``"~/.aws/credentials"``.
    source: str | None = None


@dataclass(slots=True)
class AuthCheck:
    type: AuthType
    source: str | None = None


@dataclass(slots=True)
class AuthSelectOption:
    id: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class AuthPrompt:
    """A prompt shown during login. ``signal`` aborts when an out-of-band event resolves the step."""

    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str | None = None
    options: list[AuthSelectOption] | None = None
    signal: AbortSignal | None = None


@dataclass(slots=True)
class AuthInfoLink:
    url: str
    label: str | None = None


@dataclass(slots=True)
class AuthEvent:
    """Login notification: ``info``, ``auth_url``, ``device_code`` or ``progress``."""

    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str | None = None
    links: list[AuthInfoLink] | None = None
    url: str | None = None
    instructions: str | None = None
    user_code: str | None = None
    verification_uri: str | None = None
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


class AuthInteraction(Protocol):
    """Login callbacks for API-key and OAuth flows.

    ``prompt()`` returns the entered string (``select`` returns the option id) and
    raises on cancel. ``signal`` aborts the whole login flow.
    """

    signal: AbortSignal | None

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


class ProviderAuthInteraction:
    """Normalized interaction passed to provider login implementations (signal always present)."""

    __slots__ = ("_inner", "signal")

    def __init__(self, inner: AuthInteraction, signal: AbortSignal) -> None:
        self._inner = inner
        self.signal = signal

    async def prompt(self, prompt: AuthPrompt) -> str:
        return await self._inner.prompt(prompt)

    def notify(self, event: AuthEvent) -> None:
        self._inner.notify(event)


@dataclass(slots=True)
class LoginOptions:
    #: Stable id of this app installation (e.g. OpenAI's agent host id).
    get_device_id: Callable[[], str] | None = None
    #: Name this app introduces itself with during login. Defaults to pi's own name.
    agent_name: str | None = None


ResolveFn: TypeAlias = Callable[..., Awaitable["AuthResult | None"]]
CheckFn: TypeAlias = Callable[..., Awaitable["AuthCheck | None"]]
LoginFn: TypeAlias = Callable[..., Awaitable[Credential]]


class ApiKeyAuth:
    """API-key auth: stored key/provider env plus ambient sources (env vars, profiles, files).

    Either pass callables or subclass and override :meth:`resolve` (and optionally
    ``login``/``check``)::

        ApiKeyAuth("llama.cpp", resolve=lambda **_: AuthResult(ModelAuth()))

    ``resolve`` receives keyword arguments ``ctx``, ``credential`` and ``signal`` and
    returns ``None`` when the provider is not configured.
    """

    #: Interactive setup returning an ``ApiKeyCredential``. ``None`` means ambient-only.
    login: Callable[[ProviderAuthInteraction], Awaitable[Credential]] | None = None
    #: Optional side-effect-free availability check (use when ``resolve`` runs commands).
    check: CheckFn | None = None

    def __init__(
        self,
        name: str,
        resolve: ResolveFn | None = None,
        *,
        login: Callable[[ProviderAuthInteraction], Awaitable[Credential]] | None = None,
        check: CheckFn | None = None,
    ) -> None:
        self.name = name
        if resolve is not None:
            self._resolve_fn: ResolveFn | None = resolve
        else:
            self._resolve_fn = None
        if login is not None:
            self.login = login
        if check is not None:
            self.check = check

    async def resolve(
        self, *, ctx: AuthContext, credential: Credential | None, signal: AbortSignal
    ) -> AuthResult | None:
        if self._resolve_fn is None:
            raise NotImplementedError(f"{type(self).__name__}.resolve() is not implemented")
        from ..utils.callbacks import maybe_await

        return await maybe_await(self._resolve_fn(ctx=ctx, credential=credential, signal=signal))

    def __repr__(self) -> str:
        return f"ApiKeyAuth(name={self.name!r})"


class OAuthAuth:
    """OAuth auth. ``refresh`` produces a credential and ``to_auth`` derives request auth from it.

    ``Models`` runs ``refresh`` under the credential-store lock, so concurrent
    requests and processes refresh only once.
    """

    name: str = "OAuth"
    #: Whether access through this method is backed by a provider subscription.
    is_subscription: bool = False
    #: Selector label for the login option, e.g. "Sign in with ChatGPT".
    login_label: str | None = None

    async def login(self, interaction: ProviderAuthInteraction, options: LoginOptions | None = None) -> Credential:
        raise NotImplementedError

    async def refresh(self, credential: Credential, signal: AbortSignal) -> Credential:
        raise NotImplementedError

    async def to_auth(self, credential: Credential) -> ModelAuth:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


@dataclass(slots=True)
class ProviderAuth:
    """Provider auth. At least one of ``api_key``/``oauth`` must be present."""

    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None
