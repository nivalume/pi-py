"""Adapter conformance suite and schema checks for tests (upstream's ``@earendil-works/pi-telemetry/testing``).

The cases are runner-independent; register them with pytest::

    from pi_telemetry.testing import create_telemetry_adapter_conformance

    CASES = create_telemetry_adapter_conformance(make_fixture)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case.group}: {case.name}")
    async def test_conformance(case):
        await case.run()

The cases use plain ``assert`` statements and need a running event loop; they import no test framework.
"""

from .conformance import UnreadableError, create_telemetry_adapter_conformance, unreadable_mapping, unreadable_sequence
from .schemas import assert_spans_match_schemas, check_spans_against_schemas
from .types import TelemetryAdapterConformanceCase, TelemetryAdapterFixture, TelemetryAdapterFixtureFactory

__all__ = [
    "create_telemetry_adapter_conformance",
    "TelemetryAdapterFixture",
    "TelemetryAdapterFixtureFactory",
    "TelemetryAdapterConformanceCase",
    "UnreadableError",
    "unreadable_mapping",
    "unreadable_sequence",
    "check_spans_against_schemas",
    "assert_spans_match_schemas",
]
