"""Issue #14/#15 equivalents use the public HTTP service and replay transactions."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

driver_path = Path(__file__).resolve().parents[1] / "experiments/reliability"
if not driver_path.exists():
    pytest.skip("source-only correctness matrix", allow_module_level=True)
sys.path.insert(0, str(driver_path))
from control_fault_matrix import (  # noqa: E402
    FAULTS,
    MODES,
    PATTERNS,
    cell,
    trace,
    trace_digest,
    validate_cell,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("fault", FAULTS)
async def test_matched_fault_modes_recover_without_stale_hint_or_duplicate(
    pattern, fault
):
    results = [await cell(pattern, fault, mode) for mode in MODES]
    assert len({r["trace_sha256"] for r in results}) == 1
    for result in results:
        validate_cell(result)
        assert result["result"] == "pass"
        assert not result["engine_reuse_qualified"]
        assert not result["performance_qualified"]
        invalid = [r for r in result["rows"] if r["affected"]]
        assert invalid and all(r["unsafe_retained_hint_tokens"] == 128 for r in invalid)
        assert all(not r["view"]["ready"] for r in invalid)
        assert (
            result["rows"][0]["receipt"]["generation"]
            != result["rows"][-1]["receipt"]["generation"]
        )


def test_fixed_traces_have_distinct_shapes_and_repeatable_seed():
    assert len({trace_digest(trace(p)) for p in PATTERNS}) == 4
    for pattern in PATTERNS:
        assert trace(pattern) == trace(pattern)
    assert trace("steady", 17) != trace("steady", 18)
    assert len({r["prefix"] for r in trace("hotspot")}) == 1
    assert [r["prefix"] for r in trace("phase-shift")] == ["a"] * 6 + ["b"] * 6
    assert len({r["arrival_s"] for r in trace("microburst")}) < 12


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["hint", "attempt", "missing", "claim"])
async def test_matrix_validator_rejects_forged_success(fault):
    result = deepcopy(await cell("steady", "replay-timeout", "safe-fallback"))
    if fault == "hint":
        result["rows"][3]["receipt"]["predicted_cached_tokens"] = 128
    elif fault == "attempt":
        result["rows"][0]["receipt"]["replay_attempt_id"] = "different"
    elif fault == "missing":
        result["rows"].pop()
    else:
        result["engine_reuse_qualified"] = True
    with pytest.raises(ValueError):
        validate_cell(result)
