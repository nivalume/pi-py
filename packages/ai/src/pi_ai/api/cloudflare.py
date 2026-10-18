"""Cloudflare endpoint templates.

``{CLOUDFLARE_ACCOUNT_ID}`` and ``{CLOUDFLARE_GATEWAY_ID}`` are placeholders that
the Cloudflare providers fill from the resolved provider env before dispatch.
"""

from __future__ import annotations

__all__ = [
    "CLOUDFLARE_WORKERS_AI_BASE_URL",
    "CLOUDFLARE_WORKERS_AI_REST_BASE_URL",
    "CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL",
    "CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL",
    "CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL",
]

#: Workers AI direct endpoint.
CLOUDFLARE_WORKERS_AI_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"

#: Workers AI REST endpoint root; model runs are posted to ``run``.
CLOUDFLARE_WORKERS_AI_REST_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai"

#: AI Gateway Unified API. https://developers.cloudflare.com/ai-gateway/usage/unified-api/
CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
)

#: AI Gateway -> OpenAI passthrough. Used until /compat supports /v1/responses.
CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
)

#: AI Gateway -> Anthropic passthrough.
CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
)
