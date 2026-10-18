"""pi-ai: unified LLM API with provider collections, auth resolution, streaming events,
token/cost tracking and cross-provider context hand-off.

Python port of ``@earendil-works/pi-ai``. The core entry point does not import
built-in catalogs or provider implementations; use :mod:`pi_ai.providers.all`
(``builtin_models()``) or individual provider factories such as
:func:`pi_ai.providers.anthropic.anthropic_provider`.
"""

from __future__ import annotations

from .abort import AbortController, AbortError, AbortSignal, abort_scope
from .api.lazy import LazyApi, lazy_api, lazy_stream
from .auth.context import DefaultAuthContext, default_provider_auth_context
from .auth.credential_store import FileCredentialStore, InMemoryCredentialStore
from .auth.helpers import env_api_key_auth, keyless_auth, lazy_oauth
from .auth.resolve import AuthResolutionOverrides
from .auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    AuthSelectOption,
    AuthType,
    Credential,
    CredentialInfo,
    CredentialStore,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    OAuthCredentials,
    ProviderAuth,
    ProviderAuthInteraction,
)
from .event_stream import AssistantMessageEventStream, EventStream, create_assistant_message_event_stream
from .models import (
    UNCHANGED,
    Models,
    ModelsError,
    ModelsErrorCode,
    ModelsPublication,
    ModelsRefreshResult,
    MutableModels,
    Provider,
    RefreshModelsContext,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    create_provider,
    get_model_type,
    get_supported_thinking_levels,
    has_api,
    is_model_type,
    models_are_equal,
)
from .models_store import FileModelsStore, InMemoryModelsStore, ModelsStore, ModelsStoreEntry
from .providers.faux import (
    FauxModelDefinition,
    FauxProviderHandle,
    FauxProviderState,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from .session_resources import cleanup_session_resources, register_session_resource_cleanup
from .types import *  # noqa: F403
from .types import __all__ as _types_all
from .utils.diagnostics import (
    AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from .utils.http import ProviderHttpError, aclose_http_clients
from .utils.json_parse import (
    StreamingJsonParser,
    parse_json_with_repair,
    parse_streaming_json,
    parse_tool_call_arguments,
    repair_json,
)
from .utils.misc import uuidv7
from .utils.overflow import get_overflow_patterns, is_context_overflow, is_recoverable_length
from .utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)
from .utils.schema import StringEnum, TSchema, Type
from .utils.text import content_text, get_system_message_text, render_system_message_update
from .utils.transcript import (
    ToolStateChanges,
    TranscriptTools,
    collapse_system_messages,
    create_initial_system_message,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    resolve_transcript,
    resolve_transcript_tools,
    to_tool_declaration,
    without_initial_system_message,
)
from .utils.validation import ToolValidationError, validate_tool_arguments, validate_tool_call

__version__ = "1.3.0"

__all__ = [
    *_types_all,
    "__version__",
    # cancellation
    "AbortController",
    "AbortError",
    "AbortSignal",
    "abort_scope",
    # streams
    "EventStream",
    "AssistantMessageEventStream",
    "create_assistant_message_event_stream",
    "lazy_stream",
    "lazy_api",
    "LazyApi",
    # models runtime
    "Models",
    "MutableModels",
    "Provider",
    "ModelsError",
    "ModelsErrorCode",
    "ModelsPublication",
    "ModelsRefreshResult",
    "RefreshModelsContext",
    "UNCHANGED",
    "create_models",
    "create_provider",
    "has_api",
    "get_model_type",
    "is_model_type",
    "calculate_cost",
    "get_supported_thinking_levels",
    "clamp_thinking_level",
    "models_are_equal",
    "ModelsStore",
    "ModelsStoreEntry",
    "InMemoryModelsStore",
    "FileModelsStore",
    # auth
    "ApiKeyAuth",
    "ApiKeyCredential",
    "AuthCheck",
    "AuthContext",
    "AuthEvent",
    "AuthInfoLink",
    "AuthInteraction",
    "AuthPrompt",
    "AuthResult",
    "AuthResolutionOverrides",
    "AuthSelectOption",
    "AuthType",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "LoginOptions",
    "ModelAuth",
    "OAuthAuth",
    "OAuthCredential",
    "OAuthCredentials",
    "ProviderAuth",
    "ProviderAuthInteraction",
    "DefaultAuthContext",
    "default_provider_auth_context",
    "InMemoryCredentialStore",
    "FileCredentialStore",
    "env_api_key_auth",
    "keyless_auth",
    "lazy_oauth",
    # faux
    "FauxModelDefinition",
    "FauxProviderHandle",
    "FauxProviderState",
    "faux_provider",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    # tools & schemas
    "Type",
    "TSchema",
    "StringEnum",
    "validate_tool_call",
    "validate_tool_arguments",
    "ToolValidationError",
    # json
    "parse_streaming_json",
    "parse_json_with_repair",
    "parse_tool_call_arguments",
    "repair_json",
    "StreamingJsonParser",
    # transcript
    "normalize_context",
    "create_initial_system_message",
    "get_initial_system_message",
    "without_initial_system_message",
    "get_current_tools",
    "get_current_system_message",
    "get_current_system_prompt",
    "collapse_system_messages",
    "resolve_transcript",
    "to_tool_declaration",
    "declarations_equal",
    "ToolStateChanges",
    "get_tool_state_changes",
    "get_declared_tools",
    "has_tool_redefinitions",
    "has_non_additive_tool_changes",
    "TranscriptTools",
    "resolve_transcript_tools",
    "content_text",
    "get_system_message_text",
    "render_system_message_update",
    # errors, retry, overflow
    "ProviderHttpError",
    "aclose_http_clients",
    "is_context_overflow",
    "is_recoverable_length",
    "get_overflow_patterns",
    "RetryPolicy",
    "RetryCallbacks",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "retry_delay_ms",
    "retry_assistant_call",
    "is_retryable_assistant_error",
    # diagnostics
    "AssistantMessageDiagnostic",
    "DiagnosticErrorInfo",
    "append_assistant_message_diagnostic",
    "create_assistant_message_diagnostic",
    "extract_diagnostic_error",
    "format_thrown_value",
    # misc
    "uuidv7",
    "cleanup_session_resources",
    "register_session_resource_cleanup",
]
