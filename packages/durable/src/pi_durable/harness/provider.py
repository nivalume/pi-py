"""The provider identity document (port of ``harness/provider.ts``)."""

from __future__ import annotations

from typing import Any, TypedDict

from pi_ai.utils.misc import uuidv7
from pi_chord.context import Context

from ..documents import define_doc
from ..types import ConversationDocToken, TaskRuntime

__all__ = ["ProviderDoc", "ProviderState", "ensure_provider_session_id"]


class ProviderState(TypedDict):
    """Stable provider-facing identity of one conversation."""

    sessionId: str


#: Built-in provider state; every fork starts with a fresh identity instead of copying its parent.
ProviderDoc: ConversationDocToken[ProviderState] = define_doc(
    kind="pi.provider",
    version=1,
    scope="conversation",
    history="latest",
    fork="initial",
    initial=lambda: {"sessionId": uuidv7()},
    checkpoint_when=lambda *_args: True,
)


async def ensure_provider_session_id(runtime: TaskRuntime[Any, Any, Any, Any], context: Context) -> str:
    """The persisted identity, without writing in the normal path.

    A legacy conversation without ``pi.provider`` gets one migration commit whose ``tx.doc()`` runs
    ``initial()`` before the provider request starts.
    """
    existing = await runtime.snapshot(ProviderDoc, runtime.conversation_id, context)
    if existing is not None:
        return existing["sessionId"]
    created: list[str] = []

    async def change(tx: Any, _current: Any) -> None:
        created.append((await tx.doc(ProviderDoc, runtime.conversation_id))["sessionId"])

    await runtime.commit(change, context)
    if not created:
        raise RuntimeError(f"Conversation {runtime.conversation_id} has no provider session ID")
    return created[0]
