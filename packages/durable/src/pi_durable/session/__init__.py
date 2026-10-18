"""The Session kernel (port of upstream ``src/session/``).

:func:`create_session` opens a :class:`~pi_durable.types.Session` over one Storage. The other names are
internal building blocks the Harness uses: the concrete :class:`SessionImpl` and :class:`Transaction`,
and the committed-state observers.
"""

from __future__ import annotations

from .forks import ForkDocumentCopy, prepare_fork_document_copies
from .observation import (
    MAX_PENDING_WATCH_FRAMES,
    RETIREMENT_OPERATIONS,
    CommittedStateSource,
    CommittedWatch,
    ObservedDocumentValue,
)
from .session import (
    DocumentCacheSizes,
    GuardedStorage,
    SessionFailure,
    SessionImpl,
    contained,
    create_session,
    date_now,
    document_retention,
    set_default_document_retention,
)
from .transaction import LoadedDocument, Transaction, TransactionHost, TransactionScope

__all__ = [
    "MAX_PENDING_WATCH_FRAMES",
    "RETIREMENT_OPERATIONS",
    "CommittedStateSource",
    "CommittedWatch",
    "DocumentCacheSizes",
    "ForkDocumentCopy",
    "GuardedStorage",
    "LoadedDocument",
    "ObservedDocumentValue",
    "SessionFailure",
    "SessionImpl",
    "Transaction",
    "TransactionHost",
    "TransactionScope",
    "contained",
    "create_session",
    "date_now",
    "document_retention",
    "prepare_fork_document_copies",
    "set_default_document_retention",
]
