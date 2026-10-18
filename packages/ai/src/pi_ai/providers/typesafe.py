"""TypeSafe provider (System One classifiers)."""

from __future__ import annotations

from ..api.registry import typesafe_system_one_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import Provider, create_provider
from .catalog import catalog_models

__all__ = ["typesafe_provider"]


def typesafe_provider() -> Provider:
    """TypeSafe System One classifiers, authenticated with ``TYPESAFE_API_KEY``."""
    return create_provider(
        id="typesafe",
        name="TypeSafe",
        auth=ProviderAuth(api_key=env_api_key_auth("TypeSafe API key", ["TYPESAFE_API_KEY"])),
        models=lambda: catalog_models("typesafe"),
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
