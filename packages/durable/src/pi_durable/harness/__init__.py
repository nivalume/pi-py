"""The durable agent harness (port of ``harness/``).

:class:`pi_durable.harness.harness.Harness` extends the Session kernel with conversations, a registry
(:mod:`~pi_durable.harness.registry`), and a task scheduler (:mod:`~pi_durable.harness.scheduler`) that runs
the built-in generation, tool, and compaction tasks and the application's own. The package root re-exports
the public names; the modules keep upstream's layout (``task-graph.ts`` is ``task_graph.py``).
"""

from __future__ import annotations
