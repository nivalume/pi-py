"""Amazon Bedrock Converse streaming (``bedrock-converse-stream`` API).

Port of pi's ``api/bedrock-converse-stream.ts``, which drives the AWS SDK's
``ConverseStreamCommand``. This module uses the AWS SDK for Python: a boto3
``bedrock-runtime`` client and ``converse_stream``. boto3 is an optional extra
(``pip install "pi-ai[bedrock]"``).

boto3 is synchronous, so each request runs the call and iterates its
``EventStream`` in a worker thread; events reach the event loop in batches.

The payload passed to ``on_payload`` is the ``converse_stream`` input: ``modelId``
plus the request members, with binary members (image bytes, redacted reasoning)
as ``bytes``.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import httpx

from ..abort import abort_scope
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    now_ms,
)
from ..utils import fastjson
from ..utils.callbacks import maybe_await
from ..utils.diagnostics import AssistantMessageDiagnostic, append_assistant_message_diagnostic
from ..utils.error_body import normalize_provider_error
from ..utils.http import DEFAULT_TIMEOUT_MS, _proxy_for
from ..utils.json_parse import StreamingJsonParser, parse_tool_call_arguments
from ..utils.misc import get_provider_env_value, provider_headers_to_record, sanitize_surrogates
from ..utils.text import append_text, append_thinking, get_system_message_text
from ..utils.transcript import (
    collapse_system_messages,
    get_current_tools,
    get_initial_system_message,
    without_initial_system_message,
)
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
)
from .transform_messages import split_error_tool_result_images, transform_messages

__all__ = [
    "BedrockThinkingDisplay",
    "BedrockOptions",
    "BedrockClientConfig",
    "BedrockCredentials",
    "BedrockServiceError",
    "BedrockStreamError",
    "BOTO3_INSTALL_HINT",
    "resolve_bedrock_client_config",
    "stream",
    "stream_simple",
]

BedrockThinkingDisplay = Literal["summarized", "omitted"]


class _ToolChoiceTool(TypedDict):
    type: Literal["tool"]
    name: str


class BedrockOptions(StreamOptions, total=False):
    region: str
    profile: str
    tool_choice: Literal["auto", "any", "none"] | _ToolChoiceTool
    #: See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html for supported models.
    reasoning: ThinkingLevel
    #: Custom token budgets per thinking level. Overrides default budgets.
    thinking_budgets: ThinkingBudgets
    #: Only supported by Claude 4.x models (interleaved thinking beta).
    interleaved_thinking: bool
    #: How Claude's thinking is returned: "summarized" (default here) or "omitted" (redacted but the
    #: signature still travels back). Anthropic's own default for Opus 4.8 is "omitted"; pi defaults to
    #: "summarized" to keep behavior consistent with older Claude 4 models. Claude models only.
    thinking_display: BedrockThinkingDisplay
    #: Key-value pairs attached to the request for cost allocation tagging.
    request_metadata: dict[str, str]
    #: Bedrock API key. When set, SigV4 signing is skipped and ``Authorization: Bearer`` is sent.
    #: Also read from ``AWS_BEARER_TOKEN_BEDROCK``.
    bearer_token: str
    #: Python extension: explicit SigV4 credentials (a :class:`BedrockCredentials`, a mapping with
    #: ``access_key_id``/``secret_access_key``/``session_token`` or the SDK's camelCase keys, a botocore
    #: ``Credentials`` object, or a sync/async callable returning one). Takes precedence over the
    #: credential chain.
    credentials: Any


EMPTY_TEXT_PLACEHOLDER = "<empty>"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
#: Matches the placeholder the Anthropic API path uses for redacted thinking.
REDACTED_THINKING_PLACEHOLDER = "[Reasoning redacted]"
_AMBIENT_AUTH_MARKER = "<authenticated>"
BOTO3_INSTALL_HINT = (
    'Amazon Bedrock requires the AWS SDK for Python (boto3). Install it with: pip install "pi-ai[bedrock]"'
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BedrockServiceError(Exception):
    """A modeled Bedrock Runtime error (the SDK's ``BedrockRuntimeServiceException``).

    ``name`` is the AWS error code (``ThrottlingException``, ...). ``status``/``request_id``
    carry the HTTP metadata of a failed request (``None`` for errors inside the stream).
    """

    def __init__(
        self,
        name: str,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        body: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.message = message
        self.status = status
        self.request_id = request_id
        #: Raw response body, only when the message could not be taken from it.
        self.body = body
        self.fields = fields or {}


class BedrockStreamError(Exception):
    """An unmodeled error frame in the response stream, named after its error code."""

    def __init__(self, name: str | None, message: str) -> None:
        super().__init__(message)
        self.name = name


#: Human-readable prefixes for Bedrock exception names. Retry logic matches patterns like
#: `server.?error` and `service.?unavailable`, and overflow detection excludes the throttling and
#: service-unavailable prefixes, so keep this format rather than the raw exception name.
BEDROCK_ERROR_PREFIXES: dict[str, str] = {
    "InternalServerException": "Internal server error",
    "ModelStreamErrorException": "Model stream error",
    "ValidationException": "Validation error",
    "ThrottlingException": "Throttling error",
    "ServiceUnavailableException": "Service unavailable",
}

#: Some models reject the account's configured data retention mode; point users at the AWS docs.
BEDROCK_DATA_RETENTION_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html"
_DATA_RETENTION = re.compile(r"data retention mode", re.IGNORECASE)

#: Modeled exception members of the ConverseStream output union.
_STREAM_EXCEPTIONS = frozenset(
    {
        "internalServerException",
        "modelStreamErrorException",
        "validationException",
        "throttlingException",
        "serviceUnavailableException",
    }
)


def _format_bedrock_error(error: BaseException) -> str:
    norm = normalize_provider_error(error)
    # Surface the raw HTTP body (with status) when it is not folded into the message; this stops a
    # gateway 403 from collapsing to `Unknown: UnknownError`.
    if not norm.message_carries_body and norm.status is not None and norm.body is not None:
        core = f"{norm.status}: {norm.body}"
    else:
        core = norm.message
    hint = (
        f" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes."
        if _DATA_RETENTION.search(core)
        else ""
    )
    if isinstance(error, BedrockServiceError):
        prefix = BEDROCK_ERROR_PREFIXES.get(error.name, error.name)
        return f"{prefix}: {core}{hint}"
    return f"{core}{hint}"


#: Over-long header values are dropped rather than truncated: a truncated request id is not a request id.
_MAX_DIAGNOSTIC_VALUE_CHARS = 200


def _normalize_diagnostic_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or len(trimmed) > _MAX_DIAGNOSTIC_VALUE_CHARS:
        return None
    return trimmed


def _extract_error_code(error: BaseException) -> str | None:
    # Modeled Bedrock errors all end in `Exception`, unlike transport names such as `TimeoutError`.
    name = getattr(error, "name", None)
    if not isinstance(name, str):
        name = type(error).__name__
    if not name.endswith("Exception"):
        return None
    return _normalize_diagnostic_value(name)


def _append_failure_diagnostic(output: AssistantMessage, error: BaseException, fallback_request_id: str | None) -> None:
    """Structured metadata alongside ``error_message``, which stays untouched for retry classification."""
    details: dict[str, Any] = {}
    status = getattr(error, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        details["status"] = status
    error_code = _extract_error_code(error)
    if error_code is not None:
        details["errorCode"] = error_code
    request_id = _normalize_diagnostic_value(getattr(error, "request_id", None)) or fallback_request_id
    if request_id is not None:
        details["requestId"] = request_id
    if not details:
        return
    append_assistant_message_diagnostic(
        output, AssistantMessageDiagnostic(type="bedrock_response_failure", timestamp=now_ms(), details=details)
    )


def _sanitize_error_code(raw: Any) -> str | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = str(raw)
    if not isinstance(raw, str) or not raw:
        return None
    value = raw.split(",")[0].split(":")[0]
    if "#" in value:
        value = value.split("#")[1]
    return value or None


def _request_id(headers: Mapping[str, Any] | None) -> str | None:
    if headers is None:
        return None
    for name in ("x-amzn-requestid", "x-amzn-request-id", "x-amz-request-id"):
        value = headers.get(name)
        if value:
            return value
    return None


def _http_error(status: int, headers: Mapping[str, str], text: str) -> BedrockServiceError:
    """Deserialize a non-2xx response like the JS SDK's REST-JSON error parser.

    botocore's own parser turns unrecognized bodies into ``"403"``/``"Forbidden"``; pi's
    error messages (and retry/overflow detection) follow the JS SDK, so the raw body is used.
    """
    parsed: Any = None
    if text.strip():
        try:
            parsed = fastjson.loads(text)
        except ValueError:
            parsed = None
    body = parsed if isinstance(parsed, dict) else {}
    code = _sanitize_error_code(headers.get("x-amzn-errortype"))
    if code is None:
        code = _sanitize_error_code(body.get("code")) or _sanitize_error_code(body.get("__type"))
    message = body.get("message") or body.get("Message")
    if not isinstance(message, str) or not message:
        message = None
    return BedrockServiceError(
        code or "Unknown",
        message or "UnknownError",
        status=status,
        request_id=_request_id(headers),
        body=None if message else (text or None),
        fields={key: value for key, value in body.items() if key not in ("message", "Message")},
    )


#: Key under which the ``after-call`` hook stores the raw error response on botocore's parsed dict.
_RAW_ERROR_KEY = "PiRawErrorResponse"


def _client_error(error: Any) -> BedrockServiceError:
    """Map a botocore ``ClientError`` raised by ``converse_stream``."""
    response: Mapping[str, Any] = getattr(error, "response", None) or {}
    raw = response.get(_RAW_ERROR_KEY)
    if raw is not None:
        status, headers, content = raw
        return _http_error(status, headers, content.decode("utf-8", "replace"))
    metadata = response.get("ResponseMetadata") or {}
    info = response.get("Error") or {}
    status = metadata.get("HTTPStatusCode")
    return BedrockServiceError(
        str(info.get("Code") or "Unknown"),
        str(info.get("Message") or "UnknownError"),
        status=status if isinstance(status, int) else None,
        request_id=metadata.get("RequestId") or None,
    )


def _convert_sdk_error(error: Exception) -> tuple[Exception, str | None]:
    """Turn a botocore exception into pi's error types.

    Returns the error and, for modeled stream exceptions, the stream member name
    (``throttlingException``) under which it is reported to ``on_provider_stream_event``.
    """
    try:
        from botocore.exceptions import ClientError, EventStreamError, NoCredentialsError, NoRegionError
    except ImportError:
        return error, None
    if isinstance(error, EventStreamError):
        info = error.response.get("Error") or {}
        code = info.get("Code") or None
        message = info.get("Message") or "UnknownError"
        if code in _STREAM_EXCEPTIONS:
            return BedrockServiceError(code[0].upper() + code[1:], message), code
        return BedrockStreamError(code, message), None
    if isinstance(error, ClientError):
        return _client_error(error), None
    # Same wording as the JS SDK.
    if isinstance(error, NoRegionError):
        return RuntimeError("Region is missing"), None
    if isinstance(error, NoCredentialsError):
        return RuntimeError("Could not load credentials from any providers"), None
    return error, None


# ---------------------------------------------------------------------------
# Client configuration (region, endpoint, auth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BedrockCredentials:
    """Static SigV4 credentials."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


@dataclass(slots=True)
class BedrockClientConfig:
    """What pi would pass to ``new BedrockRuntimeClient(config)``."""

    region: str | None = None
    #: Explicit endpoint; ``None`` lets the SDK derive it from the region.
    endpoint: str | None = None
    profile: str | None = None
    #: Static SigV4 credentials (ambient access keys, explicit option, or skip-auth dummies).
    credentials: Any = None
    #: Bearer token (Bedrock API key); takes precedence over SigV4.
    token: str | None = None
    #: ``AWS_BEDROCK_SKIP_AUTH=1``: sign with dummy credentials, never with a bearer token.
    skip_auth: bool = False


_ARN_REGION = re.compile(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):")
_STANDARD_ENDPOINT = re.compile(r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$")


def _configured_region(options: Mapping[str, Any]) -> str | None:
    env = options.get("env")
    return (
        options.get("region")
        or get_provider_env_value("AWS_REGION", env)
        or get_provider_env_value("AWS_DEFAULT_REGION", env)
        or None
    )


def _configured_credentials(env: Mapping[str, str] | None) -> BedrockCredentials | None:
    access_key_id = get_provider_env_value("AWS_ACCESS_KEY_ID", env)
    secret_access_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    if not access_key_id or not secret_access_key:
        return None
    return BedrockCredentials(access_key_id, secret_access_key, get_provider_env_value("AWS_SESSION_TOKEN", env))


def _standard_endpoint_region(base_url: str | None) -> str | None:
    if not base_url:
        return None
    try:
        host = httpx.URL(base_url).host.lower()
    except (httpx.InvalidURL, ValueError):
        return None
    match = _STANDARD_ENDPOINT.match(host)
    return match.group(1) if match else None


def _should_use_explicit_endpoint(base_url: str, configured_region: str | None, has_ambient_profile: bool) -> bool:
    if not _standard_endpoint_region(base_url):
        return True
    return not configured_region and not has_ambient_profile


def resolve_bedrock_client_config(model: Model, options: Mapping[str, Any] | None = None) -> BedrockClientConfig:
    """Resolve region, endpoint, profile and auth exactly like pi configures the SDK client."""
    options = options or {}
    env = options.get("env")
    # A profile explicitly configured through pi's auth flow (the `profile` option or scoped
    # `AWS_PROFILE` on the stored credential's env) must win over ambient access keys (#6957).
    options_profile = options.get("profile") or (env or {}).get("AWS_PROFILE") or None
    config = BedrockClientConfig(profile=options_profile or get_provider_env_value("AWS_PROFILE", env))
    configured_region = _configured_region(options)
    has_ambient_profile = bool(get_provider_env_value("AWS_PROFILE"))
    endpoint_region = _standard_endpoint_region(model.base_url)
    use_explicit_endpoint = _should_use_explicit_endpoint(model.base_url, configured_region, has_ambient_profile)

    # Only pin standard AWS Bedrock runtime endpoints when no region or ambient AWS_PROFILE is configured.
    # This preserves custom endpoints (VPC/proxy) without forcing built-in catalog defaults such as
    # us-east-1 to override AWS_REGION/AWS_PROFILE.
    if use_explicit_endpoint and model.base_url:
        config.endpoint = model.base_url

    skip_auth = get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", env) == "1"
    config.skip_auth = skip_auth
    api_key = options.get("api_key")
    if api_key == _AMBIENT_AUTH_MARKER:
        api_key = None
    bearer_token = (
        options.get("bearer_token") or api_key or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env) or None
    )

    # Region resolution: ARN-embedded > explicit option > env vars > endpoint > profile config.
    # An inference profile ARN carries its region, which avoids conflicts with AWS_REGION set for other services.
    arn_match = _ARN_REGION.match(model.id)
    if arn_match:
        config.region = arn_match.group(1)
    elif configured_region:
        config.region = configured_region
    elif endpoint_region and use_explicit_endpoint:
        config.region = endpoint_region
    elif not has_ambient_profile:
        config.region = "us-east-1"

    if skip_auth:
        # Support proxies that don't need authentication.
        config.credentials = BedrockCredentials("dummy-access-key", "dummy-secret-key")
    else:
        explicit = options.get("credentials")
        if explicit is not None:
            config.credentials = explicit
        else:
            credentials = _configured_credentials(env)
            if credentials is not None and not options_profile:
                config.credentials = credentials
        if bearer_token is not None:
            config.token = bearer_token
    return config


def _coerce_credentials(value: Any) -> tuple[str, str, str | None]:
    if isinstance(value, BedrockCredentials):
        return value.access_key_id, value.secret_access_key, value.session_token
    if isinstance(value, Mapping):
        access_key_id = value.get("access_key_id") or value.get("accessKeyId")
        secret_access_key = value.get("secret_access_key") or value.get("secretAccessKey")
        if access_key_id and secret_access_key:
            return access_key_id, secret_access_key, value.get("session_token") or value.get("sessionToken")
    else:
        # botocore `Credentials` (refreshable ones are frozen per request).
        frozen = getattr(value, "get_frozen_credentials", None)
        if callable(frozen):
            value = frozen()
        access_key = getattr(value, "access_key", None) or getattr(value, "access_key_id", None)
        secret_key = getattr(value, "secret_key", None) or getattr(value, "secret_access_key", None)
        if access_key and secret_key:
            token = getattr(value, "token", None) or getattr(value, "session_token", None)
            return access_key, secret_key, token
    raise TypeError("Bedrock credentials must provide access_key_id and secret_access_key")


@dataclass(frozen=True, slots=True)
class _ClientKey:
    """Everything a cached boto3 client is configured with."""

    region: str | None
    endpoint: str | None
    profile: str | None
    credentials: tuple[str, str, str | None] | None
    token: str | None
    #: Prefer SigV4 even when ``AWS_BEARER_TOKEN_BEDROCK`` is set (skip-auth proxies).
    force_sigv4: bool
    #: Total attempts; ``None`` keeps the SDK default (standard mode: 3) or ``AWS_MAX_ATTEMPTS``.
    max_attempts: int | None
    timeout: float
    proxy: str | None


async def _client_key(model: Model, config: BedrockClientConfig, options: Mapping[str, Any]) -> _ClientKey:
    credentials = config.credentials
    if callable(credentials):
        credentials = await maybe_await(credentials())
    max_retries = options.get("max_retries")
    timeout_ms = options.get("timeout_ms")
    return _ClientKey(
        region=config.region,
        endpoint=config.endpoint,
        profile=config.profile,
        credentials=_coerce_credentials(credentials) if credentials is not None else None,
        token=config.token,
        force_sigv4=config.skip_auth,
        max_attempts=None if max_retries is None else max(0, int(max_retries)) + 1,
        timeout=(timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000,
        proxy=_proxy_for(config.endpoint or model.base_url or "https://", options.get("env")),
    )


# ---------------------------------------------------------------------------
# boto3 clients
# ---------------------------------------------------------------------------

#: Client creation loads the service model and resolves credentials, which is slow; clients are
#: thread-safe, so they are shared per configuration.
_MAX_CACHED_CLIENTS = 16
_client_cache: OrderedDict[_ClientKey, Any] = OrderedDict()
_client_cache_lock = threading.Lock()
#: Per-call state for the botocore event hooks, which run in the calling (worker) thread.
_call_state = threading.local()


def _load_boto3() -> Any:
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(BOTO3_INSTALL_HINT) from error
    return boto3


class _StaticTokenProvider:
    """botocore token provider for an explicit Bedrock API key."""

    def __init__(self, token: Any) -> None:
        self._token = token

    def load_token(self, **_kwargs: Any) -> Any:
        return self._token


#: `host` and `x-amz-*` participate in the SigV4 canonical request; `authorization` is owned by
#: SigV4 or the bearer-token path. Compared case-insensitively.
_RESERVED_HEADER_EXACT = frozenset({"authorization", "host"})


def _is_reserved_header(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("x-amz-") or lower in _RESERVED_HEADER_EXACT


def _apply_custom_headers(request: Any, **_kwargs: Any) -> None:
    """``before-sign`` hook: caller headers override same-named request headers, covered by the signature."""
    headers: Mapping[str, str] | None = getattr(_call_state, "headers", None)
    if not headers:
        return
    for name, value in headers.items():
        if name in request.headers:
            del request.headers[name]
        request.headers[name] = value


def _apply_custom_user_agent(request: Any, **_kwargs: Any) -> None:
    """``request-created`` hook (last): botocore rebuilds ``User-Agent`` after signing, so reapply it."""
    headers: Mapping[str, str] | None = getattr(_call_state, "headers", None)
    if not headers:
        return
    for name, value in headers.items():
        if name.lower() == "user-agent":
            if name in request.headers:
                del request.headers[name]
            request.headers[name] = value


def _capture_error_response(http_response: Any, parsed: Any, **_kwargs: Any) -> None:
    """``after-call`` hook: keep the raw error response, which botocore's parser discards."""
    if http_response.status_code >= 300 and isinstance(parsed, dict):
        headers = {str(name).lower(): str(value) for name, value in http_response.headers.items()}
        parsed[_RAW_ERROR_KEY] = (http_response.status_code, headers, http_response.content or b"")


def _new_client(key: _ClientKey) -> Any:
    boto3 = _load_boto3()
    import botocore.session
    from botocore.config import Config
    from botocore.tokens import FrozenAuthToken

    core = botocore.session.Session()
    if key.token is not None:
        core.register_component("token_provider", _StaticTokenProvider(FrozenAuthToken(key.token)))
    session = boto3.session.Session(botocore_session=core, profile_name=key.profile)
    # The JS SDK defaults to the standard retry mode (3 attempts); botocore defaults to legacy.
    retries: dict[str, Any] = {}
    if key.max_attempts is not None:
        retries["total_max_attempts"] = key.max_attempts
    if not os.environ.get("AWS_RETRY_MODE") and not core.get_scoped_config().get("retry_mode"):
        retries["mode"] = "standard"
    config: dict[str, Any] = {"retries": retries, "connect_timeout": key.timeout, "read_timeout": key.timeout}
    if key.token is not None:
        config["auth_scheme_preference"] = "httpBearerAuth"
    elif key.force_sigv4:
        config["auth_scheme_preference"] = "sigv4"
    if key.proxy is not None:
        config["proxies"] = {"http": key.proxy, "https": key.proxy}
    access_key, secret_key, session_token = key.credentials or (None, None, None)
    client = session.client(
        "bedrock-runtime",
        region_name=key.region,
        endpoint_url=key.endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        config=Config(**config),
    )
    client.meta.events.register("before-sign.bedrock-runtime.ConverseStream", _apply_custom_headers)
    client.meta.events.register("after-call.bedrock-runtime.ConverseStream", _capture_error_response)
    client.meta.events.register_last("request-created.bedrock-runtime", _apply_custom_user_agent)
    return client


def _get_client(key: _ClientKey) -> Any:
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is not None:
            _client_cache.move_to_end(key)
            return client
        client = _new_client(key)
        _client_cache[key] = client
        if len(_client_cache) > _MAX_CACHED_CLIENTS:
            _client_cache.popitem(last=False)
        return client


# ---------------------------------------------------------------------------
# Worker thread <-> event loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Response:
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class _Failure:
    error: Exception
    #: Stream member name of a modeled stream exception (``throttlingException``).
    stream_item: str | None


_END = object()


def _interrupt_stream(event_stream: Any) -> None:
    """Unblock a worker reading ``event_stream`` from another thread."""
    raw = getattr(event_stream, "_raw_stream", None)
    shutdown = getattr(raw, "shutdown", None)  # urllib3 >= 2.3
    try:
        if callable(shutdown):
            shutdown()
        else:
            event_stream.close()
    except Exception:
        pass


class _StreamBridge:
    """Hands items from the worker thread to the event loop.

    Items accumulate in a list; the worker schedules at most one wake-up per batch, so a
    fast stream costs one ``call_soon_threadsafe`` per batch rather than per event.
    """

    __slots__ = ("_items", "_lock", "_loop", "_scheduled", "_stream", "_waiter", "closed")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._items: list[Any] = []
        self._scheduled = False
        self._waiter: asyncio.Future[None] | None = None
        self._stream: Any = None
        self.closed = False

    # -- worker side -----------------------------------------------------------------------

    def attach(self, event_stream: Any) -> bool:
        """Register the response stream; ``False`` once the consumer has gone away."""
        with self._lock:
            if self.closed:
                return False
            self._stream = event_stream
            return True

    def put(self, item: Any) -> None:
        with self._lock:
            if self.closed:
                return
            self._items.append(item)
            if self._scheduled:
                return
            self._scheduled = True
        try:
            self._loop.call_soon_threadsafe(self._wake)
        except RuntimeError:  # the event loop is closed
            self.closed = True

    # -- event loop side -------------------------------------------------------------------

    def _wake(self) -> None:
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    async def get(self) -> list[Any]:
        while True:
            with self._lock:
                if self._items:
                    items, self._items = self._items, []
                    self._scheduled = False
                    return items
            waiter = self._loop.create_future()
            self._waiter = waiter
            try:
                await waiter
            finally:
                self._waiter = None

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self._items = []
            event_stream = self._stream
        if event_stream is not None:
            _interrupt_stream(event_stream)


def _stream_worker(bridge: _StreamBridge, key: _ClientKey, params: dict[str, Any], headers: Any) -> None:
    event_stream: Any = None
    try:
        client = _get_client(key)
        _call_state.headers = headers
        try:
            response = client.converse_stream(**params)
        finally:
            _call_state.headers = None
        event_stream = response["stream"]
        if not bridge.attach(event_stream):
            return
        bridge.put(_Response(response.get("ResponseMetadata") or {}))
        for event in event_stream:
            if bridge.closed:
                return
            bridge.put(event)
        bridge.put(_END)
    except Exception as error:
        if not bridge.closed:
            bridge.put(_Failure(*_convert_sdk_error(error)))
    finally:
        if event_stream is not None:
            try:
                event_stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Stream a Bedrock ConverseStream response."""
    opts: dict[str, Any] = dict(options or {})
    out = AssistantMessageEventStream()
    # Bedrock has no mid-conversation system messages; fold them into the leading prompt.
    normalized = collapse_system_messages(context)
    out.start(_run(model, normalized, opts, out))
    return out


def stream_simple(
    model: Model, context: TranscriptContext, options: Mapping[str, Any] | None = None
) -> AssistantMessageEventStream:
    """Map provider-agnostic simple options to Bedrock options."""
    options = options or {}
    base: dict[str, Any] = {**build_base_options(model, context, options, None)}
    if options.get("tool_choice") is not None:
        base["tool_choice"] = options["tool_choice"]
    reasoning = options.get("reasoning")
    budgets = options.get("thinking_budgets")
    if not reasoning:
        return stream(model, context, base)

    if _is_anthropic_claude_model(model):
        if _supports_adaptive_thinking(model.id, model.name):
            return stream(model, context, {**base, "reasoning": reasoning, "thinking_budgets": budgets})
        # None would mean no caller cap; build_base_options always sets one, so never coerce to 0
        # (that would make the thinking budget the entire max_tokens value).
        adjusted_max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
            base.get("max_tokens"), model.max_tokens, reasoning, budgets
        )
        max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)
        level = clamp_reasoning(reasoning) or "medium"
        return stream(
            model,
            context,
            {
                **base,
                "max_tokens": max_tokens,
                "reasoning": reasoning,
                "thinking_budgets": {**(budgets or {}), level: min(thinking_budget, max(0, max_tokens - 1024))},
            },
        )

    return stream(model, context, {**base, "reasoning": reasoning, "thinking_budgets": budgets})


@dataclass(slots=True)
class _BlockState:
    """Streaming scratch state of one content block (never stored on the block)."""

    #: Bedrock contentBlockIndex while the block is open.
    index: int | None = None
    parser: StreamingJsonParser | None = None
    #: Encrypted reasoning deltas, joined into `thinking_signature` when the block ends.
    redacted_chunks: list[bytes] | None = None


class _Blocks:
    __slots__ = ("output", "states")

    def __init__(self, output: AssistantMessage) -> None:
        self.output = output
        self.states: dict[int, _BlockState] = {}

    def find(self, bedrock_index: Any) -> int:
        for position, state in self.states.items():
            if state.index is not None and state.index == bedrock_index:
                return position
        return -1

    def add(self, block: Any, bedrock_index: Any, **state: Any) -> int:
        self.output.content.append(block)
        position = len(self.output.content) - 1
        self.states[position] = _BlockState(index=bedrock_index, **state)
        return position

    def flush_redacted(self, position: int) -> None:
        state = self.states.get(position)
        if state is None or state.redacted_chunks is None:
            return
        block = self.output.content[position]
        if block.type == "thinking":
            block.thinking_signature = base64.b64encode(b"".join(state.redacted_chunks)).decode("ascii")
        state.redacted_chunks = None

    def finalize_all(self) -> None:
        """Strip every scratch field; a stream can settle without stopping each block."""
        for position in list(self.states):
            self.flush_redacted(position)
        self.states.clear()


async def _run(
    model: Model, context: TranscriptContext, options: dict[str, Any], out: AssistantMessageEventStream
) -> None:
    output = AssistantMessage(
        [],
        api="bedrock-converse-stream",
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
    )
    signal = options.get("signal")
    env = options.get("env")
    blocks = _Blocks(output)
    # Kept outside the try so the handler can still correlate a mid-stream failure:
    # exceptions delivered as stream events carry no HTTP metadata of their own.
    response_request_id: str | None = None
    try:
        config = resolve_bedrock_client_config(model, options)
        supports_strict_mode = bool((model.compat or {}).get("supportsStrictMode", False))
        cache_retention = _resolve_cache_retention(options.get("cache_retention"), env)
        max_tokens = options.get("max_tokens")
        if max_tokens is None and _is_anthropic_claude_model(model):
            max_tokens = model.max_tokens
        initial = get_initial_system_message(context.messages)
        system_prompt = get_system_message_text(initial) if initial else None

        payload: Any = {
            "modelId": model.id,
            "messages": _convert_messages(context, model, cache_retention, env),
        }
        system = _build_system_prompt(system_prompt, model, cache_retention, env)
        if system is not None:
            payload["system"] = system
        inference_config: dict[str, Any] = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if options.get("temperature") is not None:
            inference_config["temperature"] = options["temperature"]
        payload["inferenceConfig"] = inference_config
        tool_config = _convert_tool_config(
            get_current_tools(context.messages), options.get("tool_choice"), supports_strict_mode
        )
        if tool_config is not None:
            payload["toolConfig"] = tool_config
        additional = _build_additional_model_request_fields(model, options)
        if additional is not None:
            payload["additionalModelRequestFields"] = additional
        if options.get("request_metadata") is not None:
            payload["requestMetadata"] = options["request_metadata"]
        on_payload = options.get("on_payload")
        if on_payload is not None:
            replaced = await maybe_await(on_payload(payload, model))
            if replaced is not None:
                payload = replaced

        async with abort_scope(signal):
            key = await _client_key(model, config, options)
            params = {name: value for name, value in payload.items() if value is not None}
            custom_headers = provider_headers_to_record(options.get("headers"))
            headers = (
                {name: value for name, value in custom_headers.items() if not _is_reserved_header(name)}
                if custom_headers
                else None
            )
            on_response = options.get("on_response")
            on_event = options.get("on_provider_stream_event")
            bridge = _StreamBridge(asyncio.get_running_loop())
            worker = threading.Thread(
                target=_stream_worker, args=(bridge, key, params, headers), name="pi-ai-bedrock", daemon=True
            )
            worker.start()
            try:
                finished = False
                while not finished:
                    for item in await bridge.get():
                        if type(item) is dict:
                            if on_event is not None:
                                await maybe_await(on_event(item, model))
                            _handle_item(item, model, output, blocks, out)
                        elif item is _END:
                            finished = True
                        elif isinstance(item, _Response):
                            metadata = item.metadata
                            response_request_id = _normalize_diagnostic_value(metadata.get("RequestId"))
                            status = metadata.get("HTTPStatusCode")
                            if on_response is not None and isinstance(status, int):
                                headers_record = dict(metadata.get("HTTPHeaders") or {})
                                await maybe_await(on_response(ProviderResponse(status, headers_record), model))
                        else:
                            # Modeled stream exceptions are stream items in the SDK: observe, then raise.
                            if item.stream_item is not None and on_event is not None:
                                await maybe_await(on_event({item.stream_item: item.error}, model))
                            raise item.error
            finally:
                bridge.close()

        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        if output.stop_reason == "pending":
            raise RuntimeError("Bedrock stream ended without a stop reason")
        if output.stop_reason in ("error", "aborted"):
            raise RuntimeError(output.error_message or "An unknown error occurred")

        blocks.finalize_all()
        out.push(DoneEvent(output.stop_reason, output))  # type: ignore[arg-type]
        out.end()
    except Exception as error:
        blocks.finalize_all()
        output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
        output.error_message = _format_bedrock_error(error)
        if output.stop_reason == "error":
            _append_failure_diagnostic(output, error, response_request_id)
        out.push(ErrorEvent(output.stop_reason, output))
        out.end()


# ---------------------------------------------------------------------------
# Response stream
# ---------------------------------------------------------------------------


def _handle_item(
    item: Mapping[str, Any],
    model: Model,
    output: AssistantMessage,
    blocks: _Blocks,
    out: AssistantMessageEventStream,
) -> None:
    if (start := item.get("messageStart")) is not None:
        if start.get("role") != "assistant":
            raise RuntimeError("Unexpected assistant message start but got user message start instead")
        out.push(StartEvent(output))
    elif (event := item.get("contentBlockStart")) is not None:
        _handle_content_block_start(event, blocks, out)
    elif (event := item.get("contentBlockDelta")) is not None:
        _handle_content_block_delta(event, blocks, out)
    elif (event := item.get("contentBlockStop")) is not None:
        _handle_content_block_stop(event, blocks, out)
    elif (stop := item.get("messageStop")) is not None:
        raw = stop.get("stopReason")
        output.raw_stop_reason = raw
        stop_reason, error_message = _map_stop_reason(raw)
        output.stop_reason = stop_reason
        if error_message:
            output.error_message = error_message
    elif (metadata := item.get("metadata")) is not None:
        _handle_metadata(metadata, model, output)


def _handle_content_block_start(event: Mapping[str, Any], blocks: _Blocks, out: AssistantMessageEventStream) -> None:
    tool_use = (event.get("start") or {}).get("toolUse")
    if tool_use:
        block = ToolCall(tool_use.get("toolUseId") or "", tool_use.get("name") or "", {})
        position = blocks.add(block, event.get("contentBlockIndex"), parser=StreamingJsonParser())
        out.push(ToolCallStartEvent(position, blocks.output))


def _handle_content_block_delta(event: Mapping[str, Any], blocks: _Blocks, out: AssistantMessageEventStream) -> None:
    output = blocks.output
    bedrock_index = event.get("contentBlockIndex")
    delta = event.get("delta") or {}
    position = blocks.find(bedrock_index)
    block: Any = output.content[position] if position >= 0 else None

    if delta.get("text") is not None:
        # handleContentBlockStart is not sent for text blocks, so create one on the first delta.
        if block is None:
            block = TextContent("")
            position = blocks.add(block, bedrock_index)
            out.push(TextStartEvent(position, output))
        if block.type == "text":
            append_text(block, delta["text"])
            out.push(TextDeltaEvent(position, delta["text"], output))
    elif delta.get("toolUse") and block is not None and block.type == "toolCall":
        partial = delta["toolUse"].get("input") or ""
        parser = blocks.states[position].parser
        assert parser is not None
        parser.feed(partial)
        block.arguments = parser.value()
        out.push(ToolCallDeltaEvent(position, partial, output))
    elif delta.get("reasoningContent"):
        reasoning = delta["reasoningContent"]
        if block is None:
            block = ThinkingContent("", thinking_signature="")
            position = blocks.add(block, bedrock_index)
            out.push(ThinkingStartEvent(position, output))
        if block.type != "thinking":
            return
        text = reasoning.get("text")
        if text:
            append_thinking(block, text)
            out.push(ThinkingDeltaEvent(position, text, output))
        # `thinking_signature` holds either an Anthropic signature or an opaque redacted payload,
        # never both: mixing them would corrupt whichever arrived first.
        signature = reasoning.get("signature")
        if signature and not block.redacted:
            block.thinking_signature = (block.thinking_signature or "") + signature
        redacted = reasoning.get("redactedContent")
        if redacted:
            # Encrypted reasoning from non-Anthropic models (e.g. OpenAI GPT-5.6). The payload is opaque,
            # so keep it verbatim in `thinking_signature` like Anthropic redacted thinking, and replay it.
            if not block.redacted:
                block.redacted = True
                block.thinking_signature = ""
                append_thinking(block, REDACTED_THINKING_PLACEHOLDER)
                out.push(ThinkingDeltaEvent(position, REDACTED_THINKING_PLACEHOLDER, output))
            state = blocks.states[position]
            if state.redacted_chunks is None:
                state.redacted_chunks = []
            state.redacted_chunks.append(bytes(redacted))


def _handle_content_block_stop(event: Mapping[str, Any], blocks: _Blocks, out: AssistantMessageEventStream) -> None:
    output = blocks.output
    position = blocks.find(event.get("contentBlockIndex"))
    if position < 0:
        return
    block: Any = output.content[position]
    state = blocks.states[position]
    state.index = None
    kind = block.type
    if kind == "text":
        out.push(TextEndEvent(position, block.text, output))
    elif kind == "thinking":
        blocks.flush_redacted(position)
        out.push(ThinkingEndEvent(position, block.thinking, output))
    elif kind == "toolCall":
        arguments, arguments_error = parse_tool_call_arguments(state.parser.text if state.parser else "")
        block.arguments = arguments
        if arguments_error is not None:
            block.arguments_error = arguments_error
        state.parser = None
        out.push(ToolCallEndEvent(position, block, output))


def _handle_metadata(event: Mapping[str, Any], model: Model, output: AssistantMessage) -> None:
    usage = event.get("usage")
    if not usage:
        return
    output.usage.input = usage.get("inputTokens") or 0
    output.usage.output = usage.get("outputTokens") or 0
    output.usage.cache_read = usage.get("cacheReadInputTokens") or 0
    output.usage.cache_write = usage.get("cacheWriteInputTokens") or 0
    details = usage.get("cacheDetails")
    output.usage.cache_write_1h = (
        sum((detail.get("inputTokens") or 0) for detail in details if detail.get("ttl") == "1h")
        if isinstance(details, list)
        else None
    )
    output.usage.total_tokens = usage.get("totalTokens") or (output.usage.input + output.usage.output)
    calculate_cost(model, output.usage)


def _map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason in ("end_turn", "stop_sequence"):
        return "stop", None
    if reason in ("max_tokens", "model_context_window_exceeded"):
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    return "error", (f"Provider stopped with: {reason}" if reason else None)


# ---------------------------------------------------------------------------
# Model capabilities
# ---------------------------------------------------------------------------

_MATCH_SEPARATORS = re.compile(r"[\s_.:]+")


def _model_match_candidates(model_id: str, model_name: str | None = None) -> list[str]:
    """Model id and name variants; names cover application inference profile ARNs."""
    values = [model_id, model_name] if model_name else [model_id]
    candidates: list[str] = []
    for value in values:
        lower = value.lower()
        candidates.append(lower)
        candidates.append(_MATCH_SEPARATORS.sub("-", lower))
    return candidates


def _matches_any(candidates: list[str], needles: tuple[str, ...]) -> bool:
    return any(needle in candidate for candidate in candidates for needle in needles)


_ADAPTIVE_FAMILIES = ("opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "sonnet-4-6", "sonnet-5", "haiku-5", "fable-5")
_XHIGH_FAMILIES = ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "haiku-5", "fable-5")


def _supports_adaptive_thinking(model_id: str, model_name: str | None = None) -> bool:
    """Opus 4.6+, Sonnet 4.6 and Claude 5 models."""
    return _matches_any(_model_match_candidates(model_id, model_name), _ADAPTIVE_FAMILIES)


def _supports_native_xhigh_effort(model: Model) -> bool:
    return _matches_any(_model_match_candidates(model.id, model.name), _XHIGH_FAMILIES)


def _supports_thinking_block_binding(model: Model) -> bool:
    """Opus 4.6 and Sonnet 4.6 reject `thinking.block_binding` ("Extra inputs are not permitted")."""
    return _matches_any(_model_match_candidates(model.id, model.name), _XHIGH_FAMILIES)


def _map_thinking_level_to_effort(model: Model, level: str | None) -> str:
    if level == "xhigh" and _supports_native_xhigh_effort(model):
        return "xhigh"
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def _resolve_cache_retention(cache_retention: str | None, env: Mapping[str, str] | None) -> str:
    """Defaults to "short"; PI_CACHE_RETENTION=long is honored for backward compatibility."""
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _is_anthropic_claude_model(model: Model) -> bool:
    model_id = model.id.lower()
    name = (model.name or "").lower()
    return (
        "anthropic.claude" in model_id
        or "anthropic/claude" in model_id
        or "anthropic.claude" in name
        or "anthropic/claude" in name
        or "claude" in name
    )


def _supports_prompt_caching(model: Model, env: Mapping[str, str] | None) -> bool:
    """Claude 3.5 Haiku, 3.7 Sonnet, 4.x and 5 models.

    Application inference profiles may only carry the model in ``model.name``; as a last
    resort AWS_BEDROCK_FORCE_CACHE=1 enables cache points. Nova caches automatically.
    """
    candidates = _model_match_candidates(model.id, model.name)
    if not _matches_any(candidates, ("claude",)):
        return get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env) == "1"
    if _matches_any(candidates, ("fable-5", "opus-5", "sonnet-5", "haiku-5")):
        return True
    return _matches_any(candidates, ("-4-", "claude-3-7-sonnet", "claude-3-5-haiku"))


def _supports_thinking_signature(model: Model) -> bool:
    """Only Anthropic models accept `reasoningContent.reasoningText.signature`."""
    return _is_anthropic_claude_model(model)


def _is_gov_cloud_target(model: Model, options: Mapping[str, Any]) -> bool:
    region = _configured_region(options)
    if region and region.lower().startswith("us-gov-"):
        return True
    model_id = model.id.lower()
    return model_id.startswith("us-gov.") or model_id.startswith("arn:aws-us-gov:")


_DEFAULT_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 16384,  # Budget-based Claude clamps extended levels to high
    "max": 16384,
}
#: OpenAI GPT models (GPT-5.x, GPT-6) take a nested `reasoning.effort` and reject `minimal`.
_OPENAI_GPT_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
#: gpt-oss takes a flat `reasoning_effort` and only accepts low, medium and high.
_OPENAI_GPT_OSS_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _build_additional_model_request_fields(model: Model, options: Mapping[str, Any]) -> dict[str, Any] | None:
    reasoning = options.get("reasoning")
    if not reasoning or not model.reasoning:
        return None

    if _is_anthropic_claude_model(model):
        # GovCloud Bedrock rejects the Claude thinking.display field; omit it there.
        gov_cloud = _is_gov_cloud_target(model, options)
        display = None
        if not gov_cloud:
            display = options.get("thinking_display")
            if display is None:
                display = "summarized"
        # Replayed signed thinking blocks are bound to the system prompt and tools they were created
        # with. Bedrock 400s on replay after either changes unless stale blocks are dropped, matching
        # the Anthropic provider. Skipped on GovCloud like display.
        use_block_binding = not gov_cloud and _supports_thinking_block_binding(model)
        adaptive = _supports_adaptive_thinking(model.id, model.name)
        result: dict[str, Any]
        if adaptive:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if display is not None:
                thinking["display"] = display
            if use_block_binding:
                thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            result = {
                "thinking": thinking,
                "output_config": {"effort": _map_thinking_level_to_effort(model, reasoning)},
            }
            if use_block_binding:
                result["anthropic_beta"] = [THINKING_BINDING_CONTROLS_BETA]
        else:
            # Custom budgets only cover token-based levels through high.
            level = "high" if reasoning in ("xhigh", "max") else reasoning
            budget = (options.get("thinking_budgets") or {}).get(level)
            if budget is None:
                budget = _DEFAULT_THINKING_BUDGETS.get(reasoning)
            thinking = {"type": "enabled"}
            if budget is not None:
                thinking["budget_tokens"] = budget
            if display is not None:
                thinking["display"] = display
            result = {"thinking": thinking}
        interleaved = options.get("interleaved_thinking")
        if not adaptive and (interleaved is None or interleaved):
            result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]
        return result

    candidates = _model_match_candidates(model.id, model.name)
    if _matches_any(candidates, ("gpt-oss",)):
        return {"reasoning_effort": _OPENAI_GPT_OSS_EFFORT.get(reasoning)}
    if _matches_any(candidates, ("gpt-",)):
        mapped = (model.thinking_level_map or {}).get(reasoning)
        return {"reasoning": {"effort": mapped if isinstance(mapped, str) else _OPENAI_GPT_EFFORT.get(reasoning)}}
    return None


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------


def _cache_point(cache_retention: str) -> dict[str, Any]:
    point: dict[str, Any] = {"type": "default"}
    if cache_retention == "long":
        point["ttl"] = "1h"
    return {"cachePoint": point}


def _build_system_prompt(
    system_prompt: str | None, model: Model, cache_retention: str, env: Mapping[str, str] | None
) -> list[dict[str, Any]] | None:
    if not system_prompt:
        return None
    blocks: list[dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]
    if cache_retention != "none" and _supports_prompt_caching(model, env):
        blocks.append(_cache_point(cache_retention))
    return blocks


_TOOL_CALL_ID_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


def _normalize_tool_call_id(id_: str) -> str:
    return _TOOL_CALL_ID_INVALID.sub("_", id_)[:64]


def _non_blank_text_block(text: str) -> dict[str, Any] | None:
    sanitized = sanitize_surrogates(text)
    return None if not sanitized.strip() else {"text": sanitized}


def _required_text_block(text: str) -> dict[str, Any]:
    return _non_blank_text_block(text) or {"text": EMPTY_TEXT_PLACEHOLDER}


def _sanitize_bedrock_document(value: Any) -> Any:
    """Copy a JSON document without empty property names, which Bedrock rejects."""
    if isinstance(value, (list, tuple)):
        return [_sanitize_bedrock_document(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _sanitize_bedrock_document(entry) for key, entry in value.items() if len(key) > 0}
    return value


_IMAGE_FORMATS = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


_BASE64_WHITESPACE = re.compile(r"[\t\n\f\r ]+")
_BASE64_INVALID = re.compile(r"[^A-Za-z0-9+/]")


def _b64decode_forgiving(data: str) -> bytes:
    """Base64 decode like the browser ``atob``: whitespace and missing padding are tolerated.

    Raises ``ValueError`` for characters outside the base64 alphabet.
    """
    text = _BASE64_WHITESPACE.sub("", data)
    if len(text) % 4 == 0 and text.endswith("="):
        text = text[:-2] if text.endswith("==") else text[:-1]
    if len(text) % 4 == 1 or _BASE64_INVALID.search(text):
        raise ValueError("Invalid base64 data")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def _image_block(mime_type: str, data: str) -> dict[str, Any]:
    image_format = _IMAGE_FORMATS.get(mime_type)
    if image_format is None:
        raise ValueError(f"Unknown image type: {mime_type}")
    return {"source": {"bytes": _b64decode_forgiving(data)}, "format": image_format}


def _decode_redacted_content(signature: str | None) -> bytes | None:
    """A persisted session carries the blob as base64; drop a block whose signature is not base64."""
    if not signature:
        return None
    try:
        return _b64decode_forgiving(signature)
    except ValueError:
        return None


def _tool_result_content(content: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "image":
            result.append({"image": _image_block(block.mime_type, block.data)})
        elif kind == "text":
            text_block = _non_blank_text_block(block.text)
            if text_block is not None:
                result.append(text_block)
    if not result:
        result.append({"text": EMPTY_TEXT_PLACEHOLDER})
    return result


def _convert_assistant_content(message: Any, model: Model) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_block = _non_blank_text_block(block.text)
            if text_block is not None:
                blocks.append(text_block)
        elif kind == "toolCall":
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": block.id,
                        "name": block.name,
                        "input": _sanitize_bedrock_document(block.arguments),
                    }
                }
            )
        elif kind == "thinking":
            # Encrypted reasoning is opaque: replay the stored payload as the `redactedContent`
            # member instead of lowering it to reasoning text.
            if block.redacted:
                redacted = _decode_redacted_content(block.thinking_signature)
                if redacted:
                    blocks.append({"reasoningContent": {"redactedContent": redacted}})
                continue
            thinking = sanitize_surrogates(block.thinking)
            if not thinking.strip():
                continue
            if _supports_thinking_signature(model):
                # Signatures arrive after thinking deltas. Bedrock rejects a replayed reasoning block
                # without one, so fall back to plain text, matching Anthropic.
                signature = block.thinking_signature
                if not signature or not signature.strip():
                    blocks.append({"text": thinking})
                else:
                    blocks.append({"reasoningContent": {"reasoningText": {"text": thinking, "signature": signature}}})
            else:
                # Other models reject the signature field ("doesn't support ... reasoningText.signature").
                blocks.append({"reasoningContent": {"reasoningText": {"text": thinking}}})
    return blocks


def _convert_messages(
    context: TranscriptContext, model: Model, cache_retention: str, env: Mapping[str, str] | None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    messages: list[Any] = transform_messages(
        without_initial_system_message(context.messages),
        model,
        lambda id_, _model, _message: _normalize_tool_call_id(id_),
    )
    claude = _is_anthropic_claude_model(model)
    index = 0
    count = len(messages)
    while index < count:
        message = messages[index]
        role = message.role
        if role == "user":
            content: list[dict[str, Any]] = []
            if isinstance(message.content, str):
                content.append(_required_text_block(message.content))
            else:
                for block in message.content:
                    kind = getattr(block, "type", None)
                    if kind == "text":
                        text_block = _non_blank_text_block(block.text)
                        if text_block is not None:
                            content.append(text_block)
                    elif kind == "image":
                        content.append({"image": _image_block(block.mime_type, block.data)})
                if not content:
                    content.append({"text": EMPTY_TEXT_PLACEHOLDER})
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            # Bedrock rejects messages with empty content arrays (e.g. from aborted requests).
            if message.content:
                blocks = _convert_assistant_content(message, model)
                if blocks:
                    result.append({"role": "assistant", "content": blocks})
        elif role == "toolResult":
            # Bedrock requires all consecutive tool results in one user message. Claude rejects
            # error tool results with anything but text, so their images follow the results (#10794).
            tool_results: list[dict[str, Any]] = []
            error_images: list[dict[str, Any]] = []
            while index < count and messages[index].role == "toolResult":
                tool_message = messages[index]
                split = split_error_tool_result_images(tool_message) if claude else None
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_message.tool_call_id,
                            "content": _tool_result_content(split.content if split else tool_message.content),
                            "status": "error" if tool_message.is_error else "success",
                        }
                    }
                )
                if split is not None:
                    error_images.append({"text": split.caption})
                    error_images.extend({"image": _image_block(image.mime_type, image.data)} for image in split.images)
                index += 1
            index -= 1
            result.append({"role": "user", "content": [*tool_results, *error_images]})
        index += 1

    # Cache point on the last user message for supported Claude models.
    if cache_retention != "none" and _supports_prompt_caching(model, env) and result:
        last = result[-1]
        if last["role"] == "user":
            last["content"].append(_cache_point(cache_retention))
    return result


def _convert_tool_config(tools: list[Tool], tool_choice: Any, supports_strict_mode: bool) -> dict[str, Any] | None:
    if not tools or tool_choice == "none":
        return None
    bedrock_tools: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        spec: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": get_json_schema_tool_parameters(tool, strict)},
        }
        if strict is True:
            spec["strict"] = True
        bedrock_tools.append({"toolSpec": spec})
    config: dict[str, Any] = {"tools": bedrock_tools}
    if tool_choice == "auto":
        config["toolChoice"] = {"auto": {}}
    elif tool_choice == "any":
        config["toolChoice"] = {"any": {}}
    elif isinstance(tool_choice, Mapping) and tool_choice.get("type") == "tool":
        config["toolChoice"] = {"tool": {"name": tool_choice["name"]}}
    return config
