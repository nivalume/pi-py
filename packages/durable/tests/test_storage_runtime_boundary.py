"""Port of ``test/storage-runtime-boundary.test.ts``.

Upstream walks the import graph for ``node:`` built-ins; here each portable entry point is imported in a fresh
interpreter and must not load the local (host OS) adapters, ``sqlite3``, or the backends the package root
keeps out.
"""

from __future__ import annotations

import json
import subprocess
import sys


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
