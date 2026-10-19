"""Port of ``test/storage-runtime-boundary.test.ts``.

Upstream walks the import graph for ``node:`` built-ins; here each portable entry point is imported in a fresh
interpreter and must not load the local (host OS) adapters, ``sqlite3``, or the backends the package root
keeps out. The tools, tool search, and image processor (phase 2) are their own entry points as well, and so is code
mode, which loads only pi_codemode's portable parts: the sandbox (and Monty) comes from the caller.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest


def loaded_after(module: str) -> set[str]:
    code = f"import json, sys\nimport {module}\nprint(json.dumps(sorted(sys.modules)))\n"
    output = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    return set(json.loads(output))


def test_keeps_the_package_root_limited_to_portable_core_storage() -> None:
    modules = loaded_after("pi_durable")
    assert not any(name.startswith("pi_durable.env") for name in modules)
    assert not any(name.startswith("pi_durable.storage.jsonl") for name in modules)
    assert not any(name.startswith("pi_durable.storage.sqlite") for name in modules)
    assert not any(name.startswith("pi_durable.testing") for name in modules)
    assert not any(name.startswith("pi_durable.tools") for name in modules)
    assert not any(name.startswith("pi_durable.tool_search") for name in modules)
    assert not any(name.startswith("pi_durable.images") for name in modules)
    assert not any(name == "PIL" or name.startswith("PIL.") for name in modules)
    assert not any(name.startswith("pi_durable.codemode") for name in modules)
    assert not any(name == "pi_codemode" or name.startswith("pi_codemode.") for name in modules)
    assert not any(name.startswith("pydantic_monty") for name in modules)
    assert "sqlite3" not in modules


def test_keeps_the_portable_sqlite_subpath_free_of_the_stdlib_driver() -> None:
    modules = loaded_after("pi_durable.storage.sqlite")
    assert "pi_durable.storage.sqlite.stdlib" not in modules
    assert "pi_chord.sqlite.stdlib" not in modules
    assert "sqlite3" not in modules


def test_keeps_the_portable_environment_subpath_free_of_the_local_environment() -> None:
    modules = loaded_after("pi_durable.env")
    assert "pi_durable.env.local" not in modules
    assert "pi_durable.env.local_watch" not in modules


def test_keeps_the_portable_jsonl_subpath_free_of_the_local_environment() -> None:
    modules = loaded_after("pi_durable.storage.jsonl")
    assert "pi_durable.storage.jsonl.local" not in modules
    assert "pi_durable.env.local" not in modules


def test_keeps_the_tools_free_of_the_image_processor_pillow_and_the_local_environment() -> None:
    modules = loaded_after("pi_durable.tools")
    assert not any(name.startswith("pi_durable.images") for name in modules)
    assert not any(name == "PIL" or name.startswith("PIL.") for name in modules)
    assert "pi_durable.env.local" not in modules
    assert not any(name.startswith("pi_durable.storage.sqlite") for name in modules)
    assert "sqlite3" not in modules


def test_keeps_tool_search_free_of_the_coding_tools_and_the_image_processor() -> None:
    modules = loaded_after("pi_durable.tool_search")
    assert not any(name.startswith("pi_durable.tools") for name in modules)
    assert not any(name.startswith("pi_durable.images") for name in modules)
    assert not any(name == "PIL" or name.startswith("PIL.") for name in modules)
    assert "pi_durable.env.local" not in modules
    assert "sqlite3" not in modules


def test_keeps_code_mode_free_of_the_sandbox_monty_and_the_local_environment() -> None:
    modules = loaded_after("pi_durable.codemode")
    assert "pi_codemode.sandbox" not in modules
    assert "pi_codemode.runtime.monty" not in modules
    assert not any(name.startswith("pydantic_monty") for name in modules)
    assert not any(name == "pi_agent" or name.startswith("pi_agent.") for name in modules)
    assert not any(name.startswith("pi_durable.tools") for name in modules)
    assert not any(name.startswith("pi_durable.images") for name in modules)
    assert "pi_durable.env.local" not in modules
    assert "sqlite3" not in modules


def test_imports_code_mode_without_monty() -> None:
    # A None entry in sys.modules makes `import pydantic_monty` raise ImportError, as if the extra were not installed.
    code = "import sys\nsys.modules['pydantic_monty'] = None\nimport pi_durable.codemode\n"
    assert subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).returncode == 0


@pytest.mark.skipif(importlib.util.find_spec("PIL") is None, reason="needs Pillow (the images extra)")
def test_keeps_the_image_processor_free_of_the_local_environment_and_sqlite() -> None:
    modules = loaded_after("pi_durable.images")
    assert "pi_durable.env.local" not in modules
    assert "sqlite3" not in modules


def test_names_the_images_extra_when_pillow_is_missing() -> None:
    # A None entry in sys.modules makes `import PIL` raise ImportError, as if Pillow were not installed.
    code = (
        "import sys\nsys.modules['PIL'] = None\n"
        "try:\n    import pi_durable.images\nexcept ImportError as error:\n    print(error)\n"
    )
    output = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert 'pip install "pi-ai-python[images]"' in output
    # The tools work without it.
    tools = subprocess.run(
        [sys.executable, "-c", "import sys\nsys.modules['PIL'] = None\nimport pi_durable.tools"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert tools.returncode == 0
