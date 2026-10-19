"""The coding tools: ``read``, ``write``, ``edit``, ``bash``, and ``powershell`` (port of ``tools/index.ts``).

Tools take their environment from ``api.env`` (a tool call without one fails), stream through
``api.output()``, and report remarks as diagnostics. Not imported by :mod:`pi_durable`::

    from pi_durable.tools import CodingTools, create_coding_tools

    registry.install(CodingTools)                      # read, write, edit, bash
    # With a Pillow image processor (the ``images`` extra), ``read`` converts and shrinks images:
    # registry.install(create_coding_tools(images=create_pillow_images()))
"""

from __future__ import annotations

from ..harness.define import define_extension
from ..harness.types import Extension
from .bash import (
    BashExecution,
    BashPrepare,
    BashToolInput,
    PowerShellToolInput,
    ShellStructuredOutput,
    create_bash_tool,
    create_powershell_tool,
)
from .edit import EditToolDetails, EditToolInput, create_edit_tool
from .image_processor import DEFAULT_IMAGE_LIMITS, ImageLimits, ImageProcessor, PreparedImage
from .read import ReadToolDetails, ReadToolInput, create_read_tool
from .write import WriteToolInput, create_write_tool

__all__ = [
    "DEFAULT_IMAGE_LIMITS",
    "BashExecution",
    "BashPrepare",
    "BashToolInput",
    "CodingTools",
    "EditToolDetails",
    "EditToolInput",
    "ImageLimits",
    "ImageProcessor",
    "PowerShellToolInput",
    "PreparedImage",
    "ReadToolDetails",
    "ReadToolInput",
    "ShellStructuredOutput",
    "WriteToolInput",
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_powershell_tool",
    "create_read_tool",
    "create_write_tool",
]


def create_coding_tools(*, images: ImageProcessor | None = None) -> Extension:
    """The ``coding-tools`` extension: ``read``, ``write``, ``edit``, and ``bash``; nothing installs it automatically.

    ``create_powershell_tool()`` adds ``powershell``. ``images`` prepares the images ``read`` returns (see
    :func:`create_read_tool`).
    """
    return define_extension(
        "coding-tools",
        tools=[create_read_tool(images=images), create_write_tool(), create_edit_tool(), create_bash_tool()],
    )


#: ``create_coding_tools()`` without an image processor.
CodingTools: Extension = create_coding_tools()
