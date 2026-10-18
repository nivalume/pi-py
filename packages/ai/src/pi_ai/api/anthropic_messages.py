# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# Converters branch on ``role``/``type`` discriminators held in locals, which pyright does not narrow.
"""Anthropic Messages API (``anthropic-messages``).

Port of pi's ``api/anthropic-messages.ts``. pi drives the Anthropic TypeScript SDK
(``client.beta.messages.create(...).asResponse()``) and parses the SSE body itself;
this module talks HTTP directly and reproduces the SDK's wire format: ``POST
{base_url}/v1/messages?beta=true`` with ``x-api-key`` (or ``Authorization: Bearer``),
``anthropic-version: 2023-06-01``, ``anthropic-beta`` built from the request ``betas``,
and SDK-style error messages (``400 {"type":"error",...}``).

Workload identity federation (``ANTHROPIC_FEDERATION_RULE_ID`` & co.) is handled the
way the SDK does it: the identity token is exchanged at ``/v1/oauth/token`` (RFC 7523
jwt-bearer grant) and the access token is cached across requests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import httpx

from ..abort import AbortError, abort_scope
from ..env_api_keys import (
    ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
)
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..serialization import from_dict
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
    ProviderResponse,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    now_ms,
)
from ..utils.callbacks import maybe_await
from ..utils.diagnostics import AssistantMessageDiagnostic, DiagnosticErrorInfo, append_assistant_message_diagnostic
from ..utils.http import DEFAULT_TIMEOUT_MS, ProviderHttpError, build_headers, get_http_client, send_request
from ..utils.json_parse import StreamingJsonParser, parse_json_with_repair, parse_tool_call_arguments
from ..utils.misc import get_pi_user_agent, get_provider_env_value, headers_to_record, sanitize_surrogates
from ..utils.text import append_text, append_thinking, get_system_message_text, render_system_message_update
from ..utils.transcript import get_current_tools, get_initial_system_message, resolve_transcript
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context
from .transform_messages import split_error_tool_result_images, transform_messages

__all__ = [
    "AnthropicEffort",
    "AnthropicThinkingDisplay",
    "AnthropicToolChoice",
    "AnthropicOptions",
    "AnthropicParams",
    "ConvertedAnthropicMessages",
    "WorkloadIdentityError",
    "CLAUDE_CODE_VERSION",
    "CLAUDE_CODE_TOOLS",
    "CLAUDE_CODE_SYSTEM_PROMPT",
    "stream",
    "stream_simple",
    "build_params",
    "convert_messages",
    "convert_tools",
    "get_beta_features",
    "insert_thinking_level_messages",
    "map_stop_reason",
    "normalize_tool_call_id",
    "is_oauth_token",
    "to_claude_code_name",
    "from_claude_code_name",
]

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
AnthropicThinkingDisplay = Literal["summarized", "omitted"]


class _ToolChoiceTool(TypedDict):
    type: Literal["tool"]
    name: str


AnthropicToolChoice = Literal["auto", "any", "none"] | _ToolChoiceTool


class AnthropicOptions(StreamOptions, total=False):
    """Options for :func:`stream` (pi's ``AnthropicOptions``)."""

    #: Enable extended thinking. Adaptive thinking models decide when/how much to think;
    #: older models use budget-based thinking with ``thinking_budget_tokens``.
    thinking_enabled: bool
    #: Token budget for extended thinking (older models only). Default 1024 when thinking is enabled.
    thinking_budget_tokens: int
    #: Effort level for adaptive thinking models. Ignored for older models.
    effort: AnthropicEffort
    #: How thinking content is returned. Default ``"summarized"`` when thinking is enabled
    #: (Anthropic's API default for Opus 4.7+ is ``"omitted"``).
    thinking_display: AnthropicThinkingDisplay
    #: Request the interleaved thinking beta for non-adaptive thinking models. Default True.
    interleaved_thinking: bool
    #: Anthropic tool choice. Strings map to built-in choices; ``{"type": "tool", "name": ...}`` forces a tool.
    tool_choice: AnthropicToolChoice


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MID_CONVERSATION_OUTPUT_CONFIG_BETA = "mid-conversation-output-config-2026-07-01"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
INLINE_TOOLS_BETA = "inline-tools-2026-09-15"
_CLAUDE_CODE_BETA = "claude-code-20250219"
_OAUTH_BETA = "oauth-2025-04-20"
_FEDERATION_BETA = "oidc-federation-2026-04-01"

# Stealth mode: Mimic Claude Code's tool naming exactly
CLAUDE_CODE_VERSION = "2.1.280"
CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."

# Claude Code 2.x tool names (canonical casing)
# Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
# To update: https://github.com/badlogic/cchistory
CLAUDE_CODE_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)

_CC_TOOL_LOOKUP = {name.lower(): name for name in CLAUDE_CODE_TOOLS}

_ANTHROPIC_MESSAGE_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)

# Errors for which a request with preferred (non-required) strict tools is retried non-strict.
_STRICT_TOOL_LIMIT_MESSAGES = (
    "The compiled grammar is too large",
    "Too many strict tools (",
    "Schemas contains too many parameters with union types (",
)

# Keywords Anthropic strict tool use rejects with a 400 for the whole request.
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "maxItems",
        "uniqueItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
    }
)
_STRICT_STRING_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)

_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_TOOL_CALL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _deferred_tool_placeholder() -> dict[str, Any]:
    """Stable deferred tool declared whenever native tool changes are in use.

    Anthropic adds hidden prompt scaffolding for mid-conversation tool changes; declaring
    this placeholder from the first request keeps that scaffolding in the cached prefix, so
    the first tool change does not invalidate the cache (measured: full miss without it).
    It is never activated and the model cannot see it.
    """
    return {
        "name": "__pi_deferred_placeholder__",
        "description": "Reserved placeholder. Never available. Never call this.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "defer_loading": True,
    }


# ---------------------------------------------------------------------------
# Compat, cache control and auth helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _AnthropicCompat:
    supports_eager_tool_input_streaming: bool
    supports_long_cache_retention: bool
    send_session_affinity_headers: bool
    session_affinity_format: str | None
    supports_cache_control_on_tools: bool
    supports_temperature: bool
    allow_empty_signature: bool
    supports_strict_tools: bool
    supports_mid_convo_system_messages: bool
    supports_mid_convo_tool_changes: bool


def _flag(compat: Mapping[str, Any], name: str, default: Any) -> Any:
    value = compat.get(name)
    return default if value is None else value


def _get_compat(model: Model) -> _AnthropicCompat:
    compat = model.compat or {}
    is_openrouter = model.provider == "openrouter" or "openrouter.ai" in model.base_url
    return _AnthropicCompat(
        supports_eager_tool_input_streaming=_flag(compat, "supportsEagerToolInputStreaming", True),
        supports_long_cache_retention=_flag(compat, "supportsLongCacheRetention", True),
        send_session_affinity_headers=_flag(compat, "sendSessionAffinityHeaders", is_openrouter),
        session_affinity_format=_flag(compat, "sessionAffinityFormat", "openrouter" if is_openrouter else None),
        supports_cache_control_on_tools=_flag(compat, "supportsCacheControlOnTools", True),
        supports_temperature=_flag(compat, "supportsTemperature", True),
        allow_empty_signature=_flag(compat, "allowEmptySignature", False),
        supports_strict_tools=_flag(compat, "supportsStrictTools", False),
        supports_mid_convo_system_messages=_flag(compat, "supportsMidConvoSystemMessages", False),
        supports_mid_convo_tool_changes=_flag(compat, "supportsMidConvoToolChanges", False),
    )


def _resolve_cache_retention(cache_retention: str | None, env: Mapping[str, str] | None) -> str:
    """Resolve the cache retention preference. Defaults to ``"short"``; honors ``PI_CACHE_RETENTION``."""
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _get_cache_control(
    model: Model, cache_retention: str | None, env: Mapping[str, str] | None
) -> dict[str, Any] | None:
    retention = _resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return None
    if retention == "long" and _get_compat(model).supports_long_cache_retention:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _has_header(headers: Mapping[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() != "" for key, value in headers.items())


def _has_request_auth(api_key: str | None, headers: Mapping[str, str | None] | None) -> bool:
    return (
        bool(api_key)
        or _has_header(headers, "authorization")
        or _has_header(headers, "x-api-key")
        or _has_header(headers, "cf-aig-authorization")
    )


def _assert_request_auth(provider: str, api_key: str | None, headers: Mapping[str, str | None] | None) -> None:
    if not _has_request_auth(api_key, headers):
        raise ValueError(f"No API key for provider: {provider}")


def is_oauth_token(api_key: str) -> bool:
    """Whether ``api_key`` is a Claude Code OAuth access token."""
    return "sk-ant-oat" in api_key


def to_claude_code_name(name: str) -> str:
    """Convert a tool name to Claude Code's canonical casing if it matches one (case-insensitive)."""
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def from_claude_code_name(name: str, tools: Sequence[Tool] | None = None) -> str:
    """Map a tool name returned in OAuth mode back to the caller's tool name."""
    if tools:
        lower = name.lower()
        for tool in tools:
            if tool.name.lower() == lower:
                return tool.name
    return name


def normalize_tool_call_id(tool_call_id: str, *_: Any) -> str:
    """Normalize tool call ids to Anthropic's required pattern and length."""
    return _TOOL_CALL_ID_RE.sub("_", tool_call_id)[:64]


def _base_url(model: Model) -> str:
    return model.base_url or os.environ.get("ANTHROPIC_BASE_URL") or _DEFAULT_BASE_URL


def _messages_url(base_url: str) -> str:
    # Same joining rule as the SDK's buildURL; beta.messages.create posts to /v1/messages?beta=true.
    path = "v1/messages?beta=true" if base_url.endswith("/") else "/v1/messages?beta=true"
    return base_url + path


# ---------------------------------------------------------------------------
# Workload identity federation
# ---------------------------------------------------------------------------

_TOKEN_ENDPOINT = "/v1/oauth/token"
_GRANT_TYPE_JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_ADVISORY_REFRESH_THRESHOLD_S = 120
_MANDATORY_REFRESH_THRESHOLD_S = 30
_ADVISORY_REFRESH_BACKOFF_S = 5
_MAX_TOKEN_RESPONSE_BYTES = 1 << 20
_MAX_ERROR_BODY_CHARS = 2000
_SAFE_ERROR_KEYS = ("error", "error_description", "error_uri")


class WorkloadIdentityError(Exception):
    """A failed workload identity federation token exchange (the SDK's ``WorkloadIdentityError``)."""

    def __init__(
        self, message: str, status_code: int | None = None, body: Any = None, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


def _get_anthropic_federation(
    model: Model,
    api_key: str | None,
    headers: Mapping[str, str | None] | None,
    env: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    """Workload identity federation config from the ``ANTHROPIC_*`` variables the SDK documents.

    Only for the anthropic provider, since the exchange is an Anthropic API endpoint, and
    only when no key or auth header was resolved.
    """
    if model.provider != "anthropic" or _has_request_auth(api_key, headers):
        return None
    federation_rule_id = get_provider_env_value(ANTHROPIC_FEDERATION_RULE_ID_ENV, env)
    organization_id = get_provider_env_value(ANTHROPIC_ORGANIZATION_ID_ENV, env)
    identity_token_file = get_provider_env_value(ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, env)
    if not federation_rule_id or not organization_id or not identity_token_file:
        return None
    return {
        "organization_id": organization_id,
        "workspace_id": get_provider_env_value(ANTHROPIC_WORKSPACE_ID_ENV, env),
        "authentication": {
            "type": "oidc_federation",
            "federation_rule_id": federation_rule_id,
            "service_account_id": get_provider_env_value(ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, env),
            "identity_token": {"source": "file", "path": identity_token_file},
        },
    }


@dataclass(slots=True)
class _AccessToken:
    token: str
    #: Unix epoch seconds; ``None`` caches forever.
    expires_at: float | None


def _redact_sensitive(body: Any) -> Any:
    """Redacted copy of a token-endpoint error body (only RFC 6749 §5.2 error fields survive)."""
    if body is None:
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except ValueError:
            if len(body) <= _MAX_ERROR_BODY_CHARS:
                return body
            return f"{body[:_MAX_ERROR_BODY_CHARS]}... <{len(body) - _MAX_ERROR_BODY_CHARS} more chars>"
        return json.dumps(_redact_sensitive(parsed), separators=(",", ":"), ensure_ascii=False)
    if isinstance(body, dict):
        return {key: body[key] for key in body if key in _SAFE_ERROR_KEYS}
    return None


def _require_secure_token_endpoint(base_url: str) -> None:
    """Reject base URLs that would send a JWT assertion over cleartext HTTP (loopback allowed)."""
    if not base_url:
        return
    try:
        url = httpx.URL(base_url)
    except Exception as error:
        raise WorkloadIdentityError(f'Invalid token endpoint base URL "{base_url}": {error}') from error
    if url.scheme == "https":
        return
    host = url.host.lower().strip("[]")
    if url.scheme == "http" and host in ("localhost", "127.0.0.1", "::1"):
        return
    raise WorkloadIdentityError(f'Refusing to send credential over non-https token endpoint "{base_url}"')


def _read_identity_token(path: str) -> str:
    """Read the JWT from ``path`` on every exchange (supports rotation, e.g. projected k8s tokens)."""
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
    except OSError as error:
        raise WorkloadIdentityError(f"Failed to read identity token file at {path}: {error}") from error
    token = content.strip()
    if not token:
        raise WorkloadIdentityError(f"Identity token file at {path} is empty")
    return token


def _js_number(data: Mapping[str, Any], key: str) -> float:
    """``Number(data[key])`` semantics for the token response's ``expires_in``."""
    if key not in data:
        return math.nan
    value = data[key]
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return math.nan
    return math.nan


async def _exchange_federation_token(
    config: Mapping[str, Any], base_url: str, request_options: Mapping[str, Any]
) -> _AccessToken:
    """Exchange the external OIDC JWT for an Anthropic access token (RFC 7523 jwt-bearer grant)."""
    _require_secure_token_endpoint(base_url)
    authentication = config["authentication"]
    jwt = _read_identity_token(authentication["identity_token"]["path"])
    # The token endpoint enforces a 16 KiB assertion limit; fail clearly before the round-trip.
    if len(jwt) > 16 * 1024:
        raise WorkloadIdentityError(
            f"Identity token is {math.ceil(len(jwt) / 1024)} KiB, exceeds the 16 KiB assertion limit"
        )
    body: dict[str, str] = {
        "grant_type": _GRANT_TYPE_JWT_BEARER,
        "assertion": jwt,
        "federation_rule_id": authentication["federation_rule_id"],
        "organization_id": config["organization_id"],
    }
    if authentication.get("service_account_id"):
        body["service_account_id"] = authentication["service_account_id"]
    if config.get("workspace_id"):
        body["workspace_id"] = config["workspace_id"]

    url = f"{base_url}{_TOKEN_ENDPOINT}"
    client = get_http_client(url, request_options)
    timeout_ms = request_options.get("timeout_ms")
    try:
        response = await client.post(
            url,
            content=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "anthropic-beta": f"{_OAUTH_BETA},{_FEDERATION_BETA}",
                "User-Agent": get_pi_user_agent(),
            },
            timeout=(timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000,
        )
    except httpx.HTTPError as error:
        raise WorkloadIdentityError(f"Failed to reach token endpoint {url}: {error}") from error

    request_id = response.headers.get("request-id")
    text = response.content[:_MAX_TOKEN_RESPONSE_BYTES].decode("utf-8", "replace")
    if not response.is_success:
        redacted = _redact_sensitive(text)
        hint = ""
        if response.status_code == 401:
            middle = (
                ""
                if config.get("workspace_id")
                else "If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID "
                "environment variable, the 'workspace_id' config key, or the `workspaceId` option. "
            )
            hint = (
                f" Ensure your federation rule matches your identity token. {middle}View your authentication "
                "events in the Workload identity page of Claude Console for more details."
            )
        request_info = f" (request-id {request_id})" if request_id else ""
        raise WorkloadIdentityError(
            f"Token exchange failed with status {response.status_code}{request_info}: {redacted}{hint}",
            response.status_code,
            redacted,
            request_id,
        )

    try:
        data = json.loads(text)
    except ValueError as error:
        raise WorkloadIdentityError(
            f"Token endpoint returned non-JSON response (status {response.status_code})",
            response.status_code,
            _redact_sensitive(text),
            request_id,
        ) from error
    if not isinstance(data, dict) or not data.get("access_token"):
        redacted_data = _redact_sensitive(data)
        raise WorkloadIdentityError(
            f"Token endpoint response missing access_token: {json.dumps(redacted_data, separators=(',', ':'))}",
            response.status_code,
            redacted_data,
            request_id,
        )
    token_type = data.get("token_type")
    if token_type and str(token_type).lower() != "bearer":
        raise WorkloadIdentityError(
            f'Token endpoint response: unsupported token_type "{token_type}" (want Bearer)',
            response.status_code,
            _redact_sensitive(data),
            request_id,
        )
    expires_in = _js_number(data, "expires_in")
    if not math.isfinite(expires_in):
        redacted_data = _redact_sensitive(data)
        raise WorkloadIdentityError(
            f"Token endpoint response missing required fields: {json.dumps(redacted_data, separators=(',', ':'))}",
            response.status_code,
            redacted_data,
            request_id,
        )
    return _AccessToken(str(data["access_token"]), time.time() + expires_in)


class _FederationTokenCache:
    """Access token cache with the SDK's two-tier proactive refresh and request deduplication.

    * No cached token: exchange (blocking).
    * More than 120s left: cached token.
    * 30-120s left: cached token, refresh in the background (failures keep the stale token).
    * Under 30s left: block and refresh.
    """

    __slots__ = ("_cached", "_last_advisory_error", "_next_force", "_pending", "_provider")

    def __init__(self, provider: Callable[[], Awaitable[_AccessToken]]) -> None:
        self._provider = provider
        self._cached: _AccessToken | None = None
        self._pending: asyncio.Task[_AccessToken] | None = None
        self._next_force = False
        self._last_advisory_error = 0.0

    async def get_token(self) -> str:
        force = self._next_force
        self._next_force = False
        cached = self._cached
        if force or cached is None:
            return (await self._refresh(force)).token
        if cached.expires_at is None:
            return cached.token
        remaining = cached.expires_at - time.time()
        if remaining > _ADVISORY_REFRESH_THRESHOLD_S:
            return cached.token
        if remaining > _MANDATORY_REFRESH_THRESHOLD_S:
            self._background_refresh()
            return cached.token
        return (await self._refresh()).token

    def invalidate(self) -> None:
        """Drop the cached token after a 401 so the next request exchanges again."""
        self._cached = None
        self._next_force = True

    def _current_pending(self) -> asyncio.Task[_AccessToken] | None:
        pending = self._pending
        if pending is None or pending.done() or pending.get_loop() is not asyncio.get_running_loop():
            return None
        return pending

    async def _refresh(self, force: bool = False) -> _AccessToken:
        pending = None if force else self._current_pending()
        if pending is None:
            pending = self._start_refresh()
        # Shield the shared exchange so one aborted request does not cancel it for the others.
        return await asyncio.shield(pending)

    def _background_refresh(self) -> None:
        if self._current_pending() is not None:
            return
        if time.time() - self._last_advisory_error < _ADVISORY_REFRESH_BACKOFF_S:
            return
        self._start_refresh().add_done_callback(self._on_background_done)

    def _on_background_done(self, task: asyncio.Task[_AccessToken]) -> None:
        if not task.cancelled() and task.exception() is not None:
            self._last_advisory_error = time.time()

    def _start_refresh(self) -> asyncio.Task[_AccessToken]:
        async def run() -> _AccessToken:
            try:
                token = await self._provider()
                self._cached = token
                return token
            finally:
                if self._pending is task:
                    self._pending = None

        task = asyncio.ensure_future(run())
        task.add_done_callback(_consume_task_exception)
        self._pending = task
        return task


def _consume_task_exception(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


#: The SDK caches the federated access token per client. pi keeps one client for the current
#: federation config and fetch implementation; this keeps one token cache for the current
#: config and ``http_client``.
_federation_cache: tuple[str, Any, _FederationTokenCache] | None = None


def _federation_token_cache(
    model: Model, base_url: str, federation: Mapping[str, Any], options: Mapping[str, Any]
) -> _FederationTokenCache:
    global _federation_cache
    key = json.dumps([model.base_url, federation], sort_keys=True)
    http_client = options.get("http_client")
    cached = _federation_cache
    if cached is not None and cached[0] == key and cached[1] is http_client:
        return cached[2]
    token_base_url = re.sub(r"/+$", "", base_url)
    exchange_options = {
        name: options[name] for name in ("http_client", "env", "timeout_ms") if options.get(name) is not None
    }
    cache = _FederationTokenCache(lambda: _exchange_federation_token(federation, token_base_url, exchange_options))
    _federation_cache = (key, http_client, cache)
    return cache


# ---------------------------------------------------------------------------
# Client setup (headers and auth)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Client:
    base_url: str
    #: Headers the SDK derives from apiKey/authToken (sent before the client default headers).
    auth_headers: dict[str, str]
    #: Client default headers in merge order (pi's ``defaultHeaders``); ``None`` values delete.
    default_headers: list[Mapping[str, str | None] | None]
    is_oauth_token: bool
    federation: _FederationTokenCache | None = None


def _create_client(
    model: Model,
    api_key: str | None,
    options_headers: Mapping[str, str | None] | None,
    dynamic_headers: Mapping[str, str] | None,
    session_id: str | None,
    federation: Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> _Client:
    base_url = _base_url(model)
    user_agent = {"User-Agent": get_pi_user_agent()}

    # Copilot: Bearer auth.
    if model.provider == "github-copilot":
        return _Client(
            base_url,
            {"Authorization": f"Bearer {api_key}"} if api_key is not None else {},
            [
                user_agent,
                {"accept": "application/json", "anthropic-dangerous-direct-browser-access": "true"},
                model.headers,
                dynamic_headers,
                options_headers,
            ],
            False,
        )

    # OAuth: Bearer auth, Claude Code identity headers
    if api_key and is_oauth_token(api_key):
        return _Client(
            base_url,
            {"Authorization": f"Bearer {api_key}"},
            [
                user_agent,
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                    "x-app": "cli",
                },
                model.headers,
                options_headers,
            ],
            True,
        )

    # API key, header-owned auth, or workload identity federation.
    compat = _get_compat(model)
    session_affinity: dict[str, str] = {}
    if session_id and compat.send_session_affinity_headers:
        header = "x-session-id" if compat.session_affinity_format == "openrouter" else "x-session-affinity"
        session_affinity[header] = session_id
    default_headers: list[Mapping[str, str | None] | None] = [
        user_agent,
        {"accept": "application/json", "anthropic-dangerous-direct-browser-access": "true"},
        session_affinity,
        model.headers,
        options_headers,
    ]
    if federation is not None:
        return _Client(
            base_url, {}, default_headers, False, _federation_token_cache(model, base_url, federation, options)
        )
    return _Client(base_url, {"X-Api-Key": api_key} if api_key is not None else {}, default_headers, False)


def _split_request(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Split request params into the JSON body and per-request headers, like ``beta.messages.create``."""
    body = dict(params)
    betas = body.pop("betas", None)
    user_profile_id = body.pop("user_profile_id", None)
    workspace_id = body.pop("workspace_id", None)
    # The SDK maps the deprecated output_format to output_config.format.
    output_format = body.pop("output_format", None)
    if output_format:
        output_config = body.get("output_config")
        if isinstance(output_config, Mapping) and output_config.get("format"):
            raise ValueError(
                "Both output_format and output_config.format were provided. "
                "Please use only output_config.format (output_format is deprecated)."
            )
        body["output_config"] = {
            **(output_config if isinstance(output_config, Mapping) else {}),
            "format": output_format,
        }
    elif output_format is not None:
        body["output_format"] = output_format
    headers: dict[str, str] = {}
    if betas is not None:
        headers["anthropic-beta"] = (
            ",".join(str(beta) for beta in betas) if isinstance(betas, (list, tuple)) else str(betas)
        )
    if user_profile_id is not None:
        headers["anthropic-user-profile-id"] = str(user_profile_id)
    if workspace_id is not None:
        headers["anthropic-workspace-id"] = str(workspace_id)
    return body, headers


async def _request_headers(client: _Client, request_headers: Mapping[str, str]) -> dict[str, str]:
    auth_headers: Mapping[str, str] = client.auth_headers
    if client.federation is not None:
        auth_headers = {"Authorization": f"Bearer {await client.federation.get_token()}"}
    headers = build_headers(
        {
            "Accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-version": _ANTHROPIC_VERSION,
        },
        auth_headers,
        *client.default_headers,
        {"content-type": "application/json"},
        request_headers,
    )
    if client.federation is not None:
        # Token auth appends the OAuth beta after all header merging (the SDK's prepareRequest).
        existing_key = next((name for name in headers if name.lower() == "anthropic-beta"), None)
        existing = headers.pop(existing_key) if existing_key is not None else None
        parts = [part.strip() for part in existing.split(",")] if existing is not None else []
        if existing is None or _OAUTH_BETA not in parts:
            parts.append(_OAUTH_BETA)
        headers["anthropic-beta"] = ",".join(parts)
    return headers


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def _convert_image_block(image: ImageContent) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": image.mime_type, "data": image.data}}


def _convert_content_blocks(content: Sequence[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    """Tool result content: a string when text-only, otherwise content blocks."""
    if not any(block.type == "image" for block in content):
        return sanitize_surrogates("\n".join(block.text for block in content))  # type: ignore[union-attr]
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": sanitize_surrogates(block.text)}
        if block.type == "text"
        else _convert_image_block(block)  # type: ignore[arg-type]
        for block in content
    ]
    # If only images (no text), add placeholder text block
    if not any(block["type"] == "text" for block in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def _convert_tool_result(message: ToolResultMessage, content: Sequence[TextContent | ImageContent]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": _convert_content_blocks(content),
        "is_error": message.is_error,
    }


def _is_strict_unsupported_keyword(key: str, value: Any) -> bool:
    if key in _STRICT_UNSUPPORTED_KEYWORDS:
        return True
    if key == "minItems":
        return not (isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1))
    if key == "format":
        return not isinstance(value, str) or value not in _STRICT_STRING_FORMATS
    return False


def _is_preferred_strict(tool: Tool) -> bool:
    config = tool.constrained_sampling
    return bool(config) and config.get("type") == "json_schema" and config.get("strict") == "prefer"  # type: ignore[union-attr]


def convert_tools(
    tools: Sequence[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Mapping[str, Any] | None = None,
    disable_preferred_strict_tools: bool = False,
) -> list[dict[str, Any]]:
    """Convert tools to Anthropic tool definitions (cache breakpoint on the last one)."""
    if not tools:
        return []
    last = len(tools) - 1
    definitions: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if disable_preferred_strict_tools and _is_preferred_strict(tool):
            strict: bool | None = None
        else:
            strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools, _is_strict_unsupported_keyword)
        parameters = get_json_schema_tool_parameters(tool, strict)
        properties = parameters.get("properties")
        required = parameters.get("required")
        legacy_input_schema = {
            "type": "object",
            "properties": properties if properties is not None else {},
            "required": required if required is not None else [],
        }
        definition: dict[str, Any] = {
            "name": to_claude_code_name(tool.name) if is_oauth_token else tool.name,
            "description": tool.description,
        }
        if supports_eager_tool_input_streaming:
            definition["eager_input_streaming"] = True
        if strict is True:
            definition["strict"] = True
        definition["input_schema"] = {**parameters, **legacy_input_schema} if strict is True else legacy_input_schema
        if cache_control and index == last:
            definition["cache_control"] = cache_control
        definitions.append(definition)
    return definitions


@dataclass(slots=True)
class ConvertedAnthropicMessages:
    messages: list[dict[str, Any]]
    #: Wire index of each replayed assistant message -> the effort it was generated with.
    assistant_levels: dict[int, str] = field(default_factory=dict)


def convert_messages(
    transformed_messages: Sequence[Message],
    is_oauth_token: bool,
    cache_control: Mapping[str, Any] | None = None,
    allow_empty_signature: bool = False,
    managed_provider: str | None = None,
    convert_tool_definitions: Callable[[list[Tool]], list[dict[str, Any]]] | None = None,
) -> ConvertedAnthropicMessages:
    """Convert transformed pi messages (without the leading system message) to Anthropic messages.

    ``convert_tool_definitions`` converts tool definitions for native ``tool_addition``
    blocks; ``None`` when tool changes are not native.
    """
    params: list[dict[str, Any]] = []
    assistant_levels: dict[int, str] = {}
    # Later system messages are held back and emitted directly before the next assistant
    # message (or at the end of the transcript). Anthropic requires `tool_result` blocks to
    # immediately follow their `tool_use`, so a system message between them is rejected; this
    # also mirrors where the managed-effort system messages are inserted. As a result an
    # update placed before a user message in the transcript lands after it on the wire.
    pending_system: list[dict[str, Any]] = []

    count = len(transformed_messages)
    i = 0
    while i < count:
        message = transformed_messages[i]
        role = message.role
        if role == "system":
            # Later system messages only reach this point when the model accepts them natively;
            # otherwise the transcript was collapsed into the leading message before conversion.
            text = render_system_message_update(message)  # type: ignore[arg-type]
            blocks: list[dict[str, Any]] = []
            if text:
                blocks.append({"type": "text", "text": sanitize_surrogates(text)})
            if convert_tool_definitions is not None:
                added = list(message.tools_added or [])  # type: ignore[union-attr]
                redefined = {tool.name for tool in added}
                for removed in message.tools_removed or ():  # type: ignore[union-attr]
                    # A new definition under the same name replaces the old one, so no removal is needed.
                    if removed.name in redefined:
                        continue
                    name = to_claude_code_name(removed.name) if is_oauth_token else removed.name
                    blocks.append({"type": "tool_removal", "tool": {"type": "tool_reference", "name": name}})
                for definition in convert_tool_definitions(added):
                    blocks.append(
                        {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}
                    )
            if blocks:
                pending_system.append({"role": "system", "content": blocks})
        elif role == "user":
            content = message.content
            if isinstance(content, str):
                if content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(content)})
            else:
                user_blocks: list[dict[str, Any]] = []
                for item in content:
                    if item.type == "text":
                        if item.text.strip():
                            user_blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        user_blocks.append(_convert_image_block(item))  # type: ignore[arg-type]
                if user_blocks:
                    params.append({"role": "user", "content": user_blocks})
        elif role == "assistant":
            params.extend(pending_system)
            pending_system.clear()
            assistant_blocks: list[dict[str, Any]] = []
            for block in message.content:  # type: ignore[union-attr]
                kind = block.type
                if kind == "text":
                    if not block.text.strip():
                        continue
                    assistant_blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif kind == "thinking":
                    # Redacted thinking: pass the opaque payload back as redacted_thinking
                    if block.redacted:
                        assistant_blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                        continue
                    signature = block.thinking_signature
                    has_signature = bool(signature) and bool(signature.strip())
                    if not block.thinking.strip() and not has_signature:
                        continue
                    # If thinking signature is missing/empty (e.g., from aborted stream),
                    # convert to plain text for Anthropic. Some compatible providers emit
                    # and accept empty signatures, so let marked models preserve the block.
                    if not has_signature:
                        if allow_empty_signature:
                            assistant_blocks.append(
                                {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                            )
                        else:
                            assistant_blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                    else:
                        assistant_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": signature,
                            }
                        )
                elif kind == "toolCall":
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": to_claude_code_name(block.name) if is_oauth_token else block.name,
                            "input": block.arguments if block.arguments is not None else {},
                        }
                    )
            if assistant_blocks:
                message_index = len(params)
                params.append({"role": "assistant", "content": assistant_blocks})
                level = message.provider_thinking_level  # type: ignore[union-attr]
                if (
                    managed_provider is not None
                    and message.api == "anthropic-messages"  # type: ignore[union-attr]
                    and message.provider == managed_provider  # type: ignore[union-attr]
                    and level in _ANTHROPIC_EFFORTS
                ):
                    assistant_levels[message_index] = level  # type: ignore[assignment]
        elif role == "toolResult":
            # Collect all consecutive toolResult messages, needed for z.ai Anthropic endpoint.
            tool_results: list[dict[str, Any]] = []
            # Anthropic rejects error tool results that contain anything but text, so their images follow
            # all tool results of the turn, which must come first in the user message (#10794).
            error_images: list[dict[str, Any]] = []
            j = i
            while j < count and transformed_messages[j].role == "toolResult":
                tool_message: ToolResultMessage = transformed_messages[j]  # type: ignore[assignment]
                split = split_error_tool_result_images(tool_message)
                tool_results.append(
                    _convert_tool_result(tool_message, split.content if split is not None else tool_message.content)
                )
                if split is not None:
                    error_images.append({"type": "text", "text": split.caption})
                    error_images.extend(_convert_image_block(image) for image in split.images)
                j += 1
            # Skip the messages we've already processed.
            i = j - 1
            params.append({"role": "user", "content": [*tool_results, *error_images]})
        i += 1

    params.extend(pending_system)

    # Add cache_control to the last user or system message to cache conversation history
    if cache_control and params:
        last_message = params[-1]
        if last_message["role"] in ("user", "system"):
            content = last_message["content"]
            if isinstance(content, list):
                last_block = content[-1] if content else None
                if last_block is not None and last_block.get("type") in (
                    "text",
                    "image",
                    "tool_result",
                    "tool_addition",
                    "tool_removal",
                ):
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last_message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]

    return ConvertedAnthropicMessages(params, assistant_levels)


def insert_thinking_level_messages(converted: ConvertedAnthropicMessages, active_effort: str) -> list[dict[str, Any]]:
    """Insert effort-only system messages before replayed assistants and a final one for this turn."""
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(converted.messages):
        historical = converted.assistant_levels.get(index)
        if historical is not None:
            messages.append({"role": "system", "content": [], "output_config": {"effort": historical}})
        messages.append(message)
    messages.append({"role": "system", "content": [], "output_config": {"effort": active_effort}})
    return messages


def _should_use_fine_grained_tool_streaming_beta(model: Model, context: TranscriptContext) -> bool:
    return len(get_current_tools(context.messages)) > 0 and not _get_compat(model).supports_eager_tool_input_streaming


def get_beta_features(
    model: Model,
    context: TranscriptContext,
    is_oauth_token: bool,
    native_tool_changes: bool,
    options: Mapping[str, Any] | None = None,
) -> list[str]:
    """The ``betas`` for a request. An ``anthropic-beta`` header in model/options headers replaces them."""
    options = options or {}
    found = False
    configured: str | None = None
    for headers in (model.headers, options.get("headers")):
        for name, value in (headers or {}).items():
            if name.lower() == "anthropic-beta":
                found = True
                configured = value
    if found:
        if configured is None:
            return []
        return list(dict.fromkeys(part.strip() for part in configured.split(",") if part.strip()))

    model_compat = model.compat or {}
    features: list[str] = []
    if is_oauth_token:
        features.extend((_CLAUDE_CODE_BETA, _OAUTH_BETA))
    if _should_use_fine_grained_tool_streaming_beta(model, context):
        features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    if (
        model.reasoning
        and options.get("thinking_enabled") is True
        and _flag(options, "interleaved_thinking", True)
        and model_compat.get("forceAdaptiveThinking") is not True
    ):
        features.append(INTERLEAVED_THINKING_BETA)
    if model_compat.get("allowedFallbackModels"):
        features.append(SERVER_SIDE_FALLBACK_BETA)
    if model_compat.get("supportsMidConvoEffort") is True:
        features.extend((MID_CONVERSATION_OUTPUT_CONFIG_BETA, THINKING_BINDING_CONTROLS_BETA))
    if native_tool_changes:
        features.append(INLINE_TOOLS_BETA)
    return list(dict.fromkeys(features))


@dataclass(slots=True)
class AnthropicParams:
    """Result of :func:`build_params`."""

    params: dict[str, Any]
    has_preferred_strict_tools: bool
    #: Effort of the final effort marker, recorded on the reply so later requests can replay it.
    provider_thinking_level: str | None


def build_params(
    model: Model,
    context: TranscriptContext,
    is_oauth_token: bool,
    options: Mapping[str, Any] | None = None,
    disable_preferred_strict_tools: bool = False,
) -> AnthropicParams:
    """Build the ``beta.messages.create`` params (including ``betas``) for a resolved transcript."""
    options = options or {}
    cache_control = _get_cache_control(model, options.get("cache_retention"), options.get("env"))
    compat = _get_compat(model)
    model_compat = model.compat or {}
    initial = get_initial_system_message(context.messages)
    initial_text = get_system_message_text(initial) if initial is not None else ""
    transformed = transform_messages(context.messages, model, normalize_tool_call_id)
    conversation = transformed[1:] if initial is not None else transformed

    # Native tool changes keep the request-level tool list fixed and define every later tool
    # by value in a `tool_addition` block, which also expresses same-name redefinitions.
    # Anthropic rejects a tool list where every tool is deferred, so there must be an initial
    # active tool to anchor the placeholder. Otherwise the current tool list is sent.
    has_preferred_strict_tools = False

    def convert_tool_definitions(
        tools: list[Tool], tool_cache_control: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        nonlocal has_preferred_strict_tools
        definitions = convert_tools(
            tools,
            is_oauth_token,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            tool_cache_control,
            disable_preferred_strict_tools,
        )
        if any(
            definition.get("strict") is True and _is_preferred_strict(tool)
            for definition, tool in zip(definitions, tools, strict=False)
        ):
            has_preferred_strict_tools = True
        return definitions

    initial_tools = list(initial.tools_added or []) if initial is not None else []
    native_tool_changes = (
        compat.supports_mid_convo_system_messages and compat.supports_mid_convo_tool_changes and len(initial_tools) > 0
    )
    level_map = model.thinking_level_map
    off_unsupported = level_map is not None and "off" in level_map and level_map["off"] is None
    thinking_disabled = bool(model.reasoning) and options.get("thinking_enabled") is False and not off_unsupported
    # Effort cannot change between messages while thinking is disabled, so "off" sends no effort markers.
    effort_messages = model_compat.get("supportsMidConvoEffort") is True and not thinking_disabled
    converted = convert_messages(
        conversation,
        is_oauth_token,
        cache_control,
        compat.allow_empty_signature,
        model.provider if effort_messages else None,
        convert_tool_definitions if native_tool_changes else None,
    )
    effort = options.get("effort")
    active_effort = effort if effort is not None else "high"
    betas = get_beta_features(model, context, is_oauth_token, native_tool_changes, options)
    max_tokens = options.get("max_tokens")
    params: dict[str, Any] = {
        "model": model.id,
        "messages": insert_thinking_level_messages(converted, active_effort) if effort_messages else converted.messages,
        "max_tokens": max_tokens if max_tokens is not None else model.max_tokens,
        "stream": True,
    }
    if betas:
        params["betas"] = betas

    def system_block(text: str) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "text", "text": text}
        if cache_control:
            block["cache_control"] = cache_control
        return block

    # For OAuth tokens, we MUST include Claude Code identity
    if is_oauth_token:
        system = [system_block(CLAUDE_CODE_SYSTEM_PROMPT)]
        if initial_text:
            system.append(system_block(sanitize_surrogates(initial_text)))
        params["system"] = system
    elif initial_text:
        # Add cache control to system prompt for non-OAuth tokens
        params["system"] = [system_block(sanitize_surrogates(initial_text))]

    # Temperature is incompatible with extended thinking and unsupported on Claude Opus 4.7+.
    temperature = options.get("temperature")
    if (
        temperature is not None
        and not options.get("thinking_enabled")
        and model_compat.get("supportsMidConvoEffort") is not True
        and compat.supports_temperature
    ):
        params["temperature"] = temperature

    tool_cache_control = cache_control if compat.supports_cache_control_on_tools else None
    if native_tool_changes:
        # Initial tools stay active with the cache breakpoint on the last one, followed by the
        # placeholder. The list never changes afterwards: later tools are defined by value in
        # `tool_addition` blocks and withdrawn by `tool_removal`, so the cached prefix survives
        # every tool change.
        params["tools"] = [
            *convert_tool_definitions(initial_tools, tool_cache_control),
            _deferred_tool_placeholder(),
        ]
    else:
        tools = get_current_tools(context.messages)
        if tools:
            params["tools"] = convert_tool_definitions(tools, tool_cache_control)

    if thinking_disabled:
        # No effort is sent, so the model default applies. Anthropic rejects `block_binding`
        # with disabled thinking.
        params["thinking"] = {"type": "disabled"}
    elif effort_messages:
        # Managed effort models use adaptive thinking so prefix mismatches can
        # be dropped instead of surfacing as persistent 400 responses.
        params["thinking"] = {
            "type": "adaptive",
            "display": options.get("thinking_display") or "summarized",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        params["output_config"] = {"effort": "high"}
    elif model.reasoning and options.get("thinking_enabled"):
        # Default to "summarized" so Opus 4.7 and Mythos Preview behave like
        # older Claude 4 models (whose API default is also "summarized").
        display = options.get("thinking_display") or "summarized"
        if model_compat.get("forceAdaptiveThinking") is True:
            # Adaptive thinking: Claude decides when and how much to think.
            params["thinking"] = {"type": "adaptive", "display": display}
            if effort:
                params["output_config"] = {"effort": effort}
        else:
            # Budget-based thinking for older models
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": options.get("thinking_budget_tokens") or 1024,
                "display": display,
            }

    metadata = options.get("metadata")
    if metadata:
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    tool_choice = options.get("tool_choice")
    if tool_choice:
        params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice

    fallbacks = model_compat.get("allowedFallbackModels")
    if fallbacks:
        params["fallbacks"] = [{"model": fallback["model"]} for fallback in fallbacks]

    return AnthropicParams(params, has_preferred_strict_tools, active_effort if effort_messages else None)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def map_stop_reason(reason: str, stop_details: Mapping[str, Any] | None = None) -> tuple[str, str | None]:
    """Map an Anthropic stop reason to ``(pi stop reason, error message)``."""
    match reason:
        case "end_turn":
            return "stop", None
        case "max_tokens":
            return "length", None
        case "tool_use":
            return "toolUse", None
        case "refusal":
            explanation = stop_details.get("explanation") if isinstance(stop_details, Mapping) else None
            return "error", explanation or "The model refused to complete the request"
        case "pause_turn":  # Stop is good enough -> resubmit
            return "stop", None
        case "stop_sequence":
            return "stop", None  # We don't supply stop sequences, so this should never happen
        case "sensitive":  # Content flagged by safety filters
            return "error", "Provider stopped with: sensitive"
        case _:
            # Handle unknown stop reasons gracefully (API may add new values)
            raise ValueError(f"Unhandled stop reason: {reason}")


async def _iter_sse_messages(response: httpx.Response) -> AsyncIterator[tuple[str | None, str, list[str]]]:
    """Decode SSE messages as ``(event, data, raw lines)``, keeping raw lines for parse errors."""
    event: str | None = None
    data: list[str] = []
    raw: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if event or data:
                yield event, "\n".join(data), raw
                event = None
                data = []
                raw = []
            continue
        raw.append(line)
        if line[0] == ":":
            continue
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if event or data:
        yield event, "\n".join(data), raw


async def _iter_anthropic_events(response: httpx.Response) -> AsyncIterator[Any]:
    saw_message_start = False
    saw_message_end = False
    async for event_name, data, raw in _iter_sse_messages(response):
        if event_name == "error":
            raise RuntimeError(data)
        if event_name not in _ANTHROPIC_MESSAGE_EVENTS:
            continue
        try:
            event = parse_json_with_repair(data)
            if event is None:
                raise ValueError("Cannot read properties of null (reading 'type')")
        except ValueError as error:
            # Raw lines are joined with a literal backslash-n, as upstream does.
            raw_text = "\\n".join(raw)
            raise RuntimeError(
                f"Could not parse Anthropic SSE event {event_name}: {error}; data={data}; raw={raw_text}"
            ) from error
        if isinstance(event, dict):
            kind = event.get("type")
            if kind == "message_start":
                saw_message_start = True
            elif kind == "message_stop":
                saw_message_end = True
        yield event
    if saw_message_start and not saw_message_end:
        raise RuntimeError("Anthropic stream ended before message_stop")


def _usage_model(model: Model, response_model: str | None) -> Model:
    """The model used for pricing: a returned fallback model's pricing when it differs."""
    if response_model == model.id:
        return model
    for fallback in (model.compat or {}).get("allowedFallbackModels") or ():
        if fallback.get("provider") == model.provider and fallback.get("model") == response_model:
            cost = fallback.get("cost")
            if not cost:
                return model
            fallback_cost = cost if isinstance(cost, ModelCost) else from_dict(ModelCost, cost)
            return dataclasses.replace(model, id=response_model, cost=fallback_cost)
    return model


def _update_total(usage: Usage) -> None:
    # Anthropic doesn't provide total_tokens, compute from components
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write


def _is_strict_tool_limit_error(error: BaseException) -> bool:
    if not isinstance(error, ProviderHttpError) or error.status != 400:
        return False
    body = error.error
    inner = body.get("error") if isinstance(body, dict) else None
    if not isinstance(inner, dict) or inner.get("type") != "invalid_request_error":
        return False
    message = str(error)
    return any(text in message for text in _STRICT_TOOL_LIMIT_MESSAGES)


def _input_transformations_diagnostic(transformations: list[Any]) -> AssistantMessageDiagnostic:
    details: list[dict[str, Any]] = []
    for transformation in transformations:
        entry: dict[str, Any] = {}
        if isinstance(transformation, Mapping):
            for key in ("type", "path", "reason"):
                value = transformation.get(key)
                if value is not None:
                    entry[key] = value
        details.append(entry)
    return AssistantMessageDiagnostic(
        type="anthropic_input_transformations", timestamp=now_ms(), details={"transformations": details}
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a response from the Anthropic Messages API (options: :class:`AnthropicOptions`)."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    normalized = resolve_transcript(context, _get_compat(model).supports_mid_convo_system_messages)
    current_tools = get_current_tools(normalized.messages)
    out.start(_run(model, normalized, current_tools, opts, out))
    return out


async def _run(
    model: Model,
    context: TranscriptContext,
    current_tools: list[Tool],
    options: dict[str, Any],
    out: AssistantMessageEventStream,
) -> None:
    output = AssistantMessage(
        [], api=model.api, provider=model.provider, model=model.id, usage=Usage(), stop_reason="pending"
    )
    signal = options.get("signal")
    try:
        async with abort_scope(signal):
            await _stream_response(model, context, current_tools, options, output, out)
        if signal is not None and signal.aborted:
            raise AbortError(signal.reason)
        if output.stop_reason == "pending":
            raise RuntimeError("Anthropic stream ended without a stop reason")
        if output.stop_reason in ("aborted", "error"):
            raise RuntimeError(output.error_message or "An unknown error occurred")
        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        aborted = signal is not None and signal.aborted
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = (
            "Request was aborted" if isinstance(error, AbortError) else (str(error) or type(error).__name__)
        )
        out.push(ErrorEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()


async def _stream_response(
    model: Model,
    context: TranscriptContext,
    current_tools: list[Tool],
    options: dict[str, Any],
    output: AssistantMessage,
    out: AssistantMessageEventStream,
) -> None:
    signal = options.get("signal")
    api_key: str | None = options.get("api_key")
    options_headers: Mapping[str, str | None] | None = options.get("headers")
    federation = _get_anthropic_federation(model, api_key, options_headers, options.get("env"))
    if federation is None:
        _assert_request_auth(model.provider, api_key, options_headers)

    dynamic_headers: dict[str, str] | None = None
    if model.provider == "github-copilot":
        dynamic_headers = build_copilot_dynamic_headers(
            messages=context.messages, has_images=has_copilot_vision_input(context.messages)
        )
    cache_retention = _resolve_cache_retention(options.get("cache_retention"), options.get("env"))
    cache_session_id = None if cache_retention == "none" else options.get("session_id")
    client = _create_client(model, api_key, options_headers, dynamic_headers, cache_session_id, federation, options)
    is_oauth = client.is_oauth_token
    url = _messages_url(client.base_url)

    disable_preferred_strict_tools = False
    while True:
        if signal is not None:
            signal.throw_if_aborted()
        built = build_params(model, context, is_oauth, options, disable_preferred_strict_tools)
        if built.provider_thinking_level:
            output.provider_thinking_level = built.provider_thinking_level
        params = built.params
        on_payload = options.get("on_payload")
        if on_payload is not None:
            replaced = await maybe_await(on_payload(params, model))
            if replaced is not None:
                params = dict(replaced)
                params["stream"] = True
        body, request_headers = _split_request(params)
        try:
            headers = await _request_headers(client, request_headers)
            try:
                response = await send_request(
                    "POST", url, headers=headers, body=body, options=options, error_style="anthropic"
                )
            except ProviderHttpError as error:
                if client.federation is not None and error.status == 401:
                    # The server rejected the token even if it still looks fresh.
                    client.federation.invalidate()
                raise
            break
        except Exception as error:
            # Only downgrade a rejected request, never an already-started stream.
            if (
                (signal is not None and signal.aborted)
                or disable_preferred_strict_tools
                or not built.has_preferred_strict_tools
                or not _is_strict_tool_limit_error(error)
            ):
                raise
            disable_preferred_strict_tools = True
            append_assistant_message_diagnostic(
                output,
                AssistantMessageDiagnostic(
                    type="anthropic_strict_tools_fallback",
                    timestamp=now_ms(),
                    error=DiagnosticErrorInfo(message=str(error)),
                    details={"reason": "strict_tool_limits_exceeded"},
                ),
            )

    try:
        on_response = options.get("on_response")
        if on_response is not None:
            await maybe_await(
                on_response(ProviderResponse(response.status_code, headers_to_record(response.headers)), model)
            )
        out.push(StartEvent(output))
        await _consume_events(model, current_tools, is_oauth, options, output, out, response)
    finally:
        await response.aclose()


async def _consume_events(
    model: Model,
    current_tools: list[Tool],
    is_oauth: bool,
    options: Mapping[str, Any],
    output: AssistantMessage,
    out: AssistantMessageEventStream,
    response: httpx.Response,
) -> None:
    content = output.content
    usage = output.usage
    usage_model = model
    input_transformations: list[Any] | None = None
    # Anthropic content block index -> positions in output.content of blocks not yet stopped.
    open_blocks: dict[Any, list[int]] = {}
    # Streaming argument parsers of open tool calls, keyed by content position.
    parsers: dict[int, StreamingJsonParser] = {}
    on_event = options.get("on_provider_stream_event")

    def find(index: Any) -> int:
        positions = open_blocks.get(index)
        return positions[0] if positions else -1

    def open_block(index: Any) -> int:
        position = len(content) - 1
        open_blocks.setdefault(index, []).append(position)
        return position

    async for event in _iter_anthropic_events(response):
        if on_event is not None:
            await maybe_await(on_event(event, model))
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "content_block_delta":
            position = find(event.get("index"))
            if position < 0:
                continue
            block = content[position]
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                if block.type == "text":
                    text = delta.get("text") or ""
                    append_text(block, text)
                    out.push(TextDeltaEvent(position, text, output))
            elif delta_type == "thinking_delta":
                if block.type == "thinking":
                    thinking = delta.get("thinking") or ""
                    append_thinking(block, thinking)
                    out.push(ThinkingDeltaEvent(position, thinking, output))
            elif delta_type == "input_json_delta":
                if block.type == "toolCall":
                    partial_json = delta.get("partial_json") or ""
                    parser = parsers.get(position)
                    if parser is None:
                        parser = parsers[position] = StreamingJsonParser()
                    parser.feed(partial_json)
                    block.arguments = parser.value()
                    out.push(ToolCallDeltaEvent(position, partial_json, output))
            elif delta_type == "signature_delta":
                if block.type == "thinking":
                    block.thinking_signature = (block.thinking_signature or "") + (delta.get("signature") or "")
        elif kind == "content_block_start":
            content_block = event.get("content_block") or {}
            block_type = content_block.get("type")
            index = event.get("index")
            if block_type == "fallback":
                if content:
                    raise RuntimeError("Anthropic performed an unsupported mid-output model fallback")
                continue
            if block_type == "text":
                text = content_block.get("text")
                content.append(TextContent(text if text is not None else ""))
                out.push(TextStartEvent(open_block(index), output))
            elif block_type == "thinking":
                thinking = content_block.get("thinking")
                signature = content_block.get("signature")
                content.append(
                    ThinkingContent(
                        thinking if thinking is not None else "",
                        thinking_signature=signature if signature is not None else "",
                    )
                )
                out.push(ThinkingStartEvent(open_block(index), output))
            elif block_type == "redacted_thinking":
                content.append(
                    ThinkingContent("[Reasoning redacted]", thinking_signature=content_block.get("data"), redacted=True)
                )
                out.push(ThinkingStartEvent(open_block(index), output))
            elif block_type == "tool_use":
                name = content_block.get("name") or ""
                arguments = content_block.get("input")
                content.append(
                    ToolCall(
                        content_block.get("id") or "",
                        from_claude_code_name(name, current_tools) if is_oauth else name,
                        arguments if isinstance(arguments, dict) else {},
                    )
                )
                out.push(ToolCallStartEvent(open_block(index), output))
        elif kind == "content_block_stop":
            positions = open_blocks.get(event.get("index"))
            if not positions:
                continue
            position = positions.pop(0)
            block = content[position]
            if block.type == "text":
                out.push(TextEndEvent(position, block.text, output))
            elif block.type == "thinking":
                out.push(ThinkingEndEvent(position, block.thinking, output))
            elif block.type == "toolCall":
                parser = parsers.pop(position, None)
                arguments, arguments_error = parse_tool_call_arguments(parser.text if parser is not None else "")
                block.arguments = arguments
                if arguments_error is not None:
                    block.arguments_error = arguments_error
                out.push(ToolCallEndEvent(position, block, output))
        elif kind == "message_start":
            message = event.get("message") or {}
            output.response_id = message.get("id")
            transformations = message.get("input_transformations")
            if isinstance(transformations, list):
                input_transformations = transformations
            response_model = message.get("model")
            if response_model != model.id:
                output.response_model = response_model
            usage_model = _usage_model(model, response_model)
            # Capture initial token usage from message_start event
            # This ensures we have input token counts even if the stream is aborted early
            start_usage = message.get("usage") or {}
            usage.input = start_usage.get("input_tokens") or 0
            usage.output = start_usage.get("output_tokens") or 0
            usage.cache_read = start_usage.get("cache_read_input_tokens") or 0
            usage.cache_write = start_usage.get("cache_creation_input_tokens") or 0
            usage.cache_write_1h = (start_usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
            _update_total(usage)
            calculate_cost(usage_model, usage)
        elif kind == "message_delta":
            transformations = event.get("input_transformations")
            if isinstance(transformations, list):
                input_transformations = transformations
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                output.raw_stop_reason = stop_reason
                mapped, error_message = map_stop_reason(stop_reason, delta.get("stop_details"))
                output.stop_reason = mapped  # type: ignore[assignment]
                if error_message:
                    output.error_message = error_message
            # Only update usage fields if present (not null).
            # Preserves input_tokens from message_start when proxies omit it in message_delta.
            delta_usage = event.get("usage")
            if delta_usage:
                if delta_usage.get("input_tokens") is not None:
                    usage.input = delta_usage["input_tokens"]
                if delta_usage.get("output_tokens") is not None:
                    usage.output = delta_usage["output_tokens"]
                if delta_usage.get("cache_read_input_tokens") is not None:
                    usage.cache_read = delta_usage["cache_read_input_tokens"]
                if delta_usage.get("cache_creation_input_tokens") is not None:
                    usage.cache_write = delta_usage["cache_creation_input_tokens"]
                # Vercel AI Gateway includes the TTL breakdown in deltas, though the SDK only types it on message_start.
                cache_creation = delta_usage.get("cache_creation")
                if isinstance(cache_creation, dict) and cache_creation.get("ephemeral_1h_input_tokens") is not None:
                    usage.cache_write_1h = cache_creation["ephemeral_1h_input_tokens"]
                # Anthropic reports reasoning tokens as a subset of output tokens.
                details = delta_usage.get("output_tokens_details")
                if isinstance(details, dict) and details.get("thinking_tokens") is not None:
                    usage.reasoning = details["thinking_tokens"]
            _update_total(usage)
            calculate_cost(usage_model, usage)

    if input_transformations:
        append_assistant_message_diagnostic(output, _input_transformations_diagnostic(input_transformations))


# ---------------------------------------------------------------------------
# Simple options
# ---------------------------------------------------------------------------


def _map_thinking_level_to_effort(model: Model, level: str | None) -> str:
    """Map a pi thinking level to an Anthropic effort for adaptive thinking.

    Effort "max" is available on all adaptive-thinking Claude models, while native
    "xhigh" is only available on Opus 4.7/4.8, Sonnet 5, and Fable 5.
    """
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream with provider-neutral options (``reasoning`` maps to effort or a thinking budget).

    Raises ``ValueError`` synchronously when no API key, auth header or federation config is available.
    """
    options = options or {}
    api_key = options.get("api_key")
    if _get_anthropic_federation(model, api_key, options.get("headers"), options.get("env")) is None:
        _assert_request_auth(model.provider, api_key, options.get("headers"))

    base: dict[str, Any] = build_base_options(model, context, options, api_key)
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    reasoning = options.get("reasoning")
    if not reasoning:
        return stream(model, context, {**base, "thinking_enabled": False})

    # For models with adaptive thinking: use an effort level.
    # For older models: use budget-based thinking.
    if (model.compat or {}).get("forceAdaptiveThinking") is True:
        return stream(
            model,
            context,
            {**base, "thinking_enabled": True, "effort": _map_thinking_level_to_effort(model, reasoning)},
        )

    adjusted_max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
        base.get("max_tokens"), model.max_tokens, reasoning, options.get("thinking_budgets")
    )
    max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)
    return stream(
        model,
        context,
        {
            **base,
            "max_tokens": max_tokens,
            "thinking_enabled": True,
            "thinking_budget_tokens": min(thinking_budget, max(0, max_tokens - 1024)),
        },
    )
