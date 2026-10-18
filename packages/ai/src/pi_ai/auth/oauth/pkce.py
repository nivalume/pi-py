"""PKCE (RFC 7636) code verifier and S256 challenge."""

from __future__ import annotations

import hashlib
import secrets
from typing import NamedTuple

from ._common import base64url_encode

__all__ = ["PKCE", "generate_pkce"]


class PKCE(NamedTuple):
    verifier: str
    challenge: str


def generate_pkce() -> PKCE:
    """Generate a random code verifier and its SHA-256 challenge (both base64url, unpadded)."""
    verifier = base64url_encode(secrets.token_bytes(32))
    challenge = base64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return PKCE(verifier, challenge)
