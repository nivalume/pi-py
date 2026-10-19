"""Durable code mode (port of ``@earendil-works/pi-durable/codemode``).

:func:`create_codemode_extension` adds a ``codemode`` tool: the model writes Python that calls the conversation's
tools as ``await tools.<name>(...)``, run by a :mod:`pi_codemode` sandbox (Monty). Each call is a nested tool call,
and only the script's output reaches the model. The script survives a restart: it resumes where it was, without
running its finished calls again. ::

    from pi_codemode import CodemodeSandbox
    from pi_durable.codemode import create_codemode_extension

    registry.install(create_codemode_extension(sandbox=CodemodeSandbox(memory_limit_bytes=256 * 1024 * 1024)))

Script globals use snake_case names where upstream's are camelCase: ``search_tools``, ``describe_tool``,
``describe_namespace``, ``models.get_models_of_type``, ``models.get_available_of_type``, ``models.get_model_of_type``,
``models.classify``, ``models.generate_images``; ``print`` replaces ``console.log``. Tool names are unchanged.

This module imports only pi_codemode's portable parts; the sandbox (and Monty) comes from the caller. Importing
:mod:`pi_durable` does not load it.
"""

from __future__ import annotations

from .codemode import (
    CODEMODE_MODULE_VERSION,
    CODEMODE_TOOL_NAME,
    CodemodeGlobal,
    CodemodeNestedCall,
    CodemodeStoreDoc,
    CodemodeToolDetails,
    CodemodeToolInput,
    create_codemode_extension,
    create_codemode_tool,
)
from .describe import DEFAULT_CODEMODE_INLINE_BUDGET, CodemodeNamespace

__all__ = [
    "CODEMODE_MODULE_VERSION",
    "CODEMODE_TOOL_NAME",
    "DEFAULT_CODEMODE_INLINE_BUDGET",
    "CodemodeGlobal",
    "CodemodeNamespace",
    "CodemodeNestedCall",
    "CodemodeStoreDoc",
    "CodemodeToolDetails",
    "CodemodeToolInput",
    "create_codemode_extension",
    "create_codemode_tool",
]
