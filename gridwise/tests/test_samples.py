# tests/test_samples.py
"""Replay the 10 public sample cases against the running service.

Usage:
    pytest -q tests/test_samples.py
or:
    python -m tests.test_samples path/to/public_samples.json
"""
import json
import os
import sys
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    OptimizeRequest, DirectiveInterpretation, HourlyPlanEntry,
)
from app.replay import replay_and_check

client = TestClient(app)


def _load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def _semantic_interpretation_match(
    expected: list[dict], actual: list[dict]
) -> list[str]:
    """Compare only the machine-checkable fields (not explanation text)."""
    errs = []
    if len(expected) != len(actual):
        return [f"entry count {len(actual)} != {len(expected)}"]
    for e, a in zip(expected, actual):
        if e["note_index"] != a["note_index"]:
            errs.append(f"note_index mismatch at {e['note_index']}")
        if e["applies"] != a["applies"]:
            errs.append(f"applies mismatch at {e['note_index']}")
        if e["directive_type"] != a["directive_type"]:
            errs.append(
                f"type mismatch at {e['note_index']}: "
                f"{a['directive_type']} != {e['directive_type']}"
            )
        es, as_ = e["structured_adjustment"], a["structured_adjustment"]
        if (es is None) != (as_ is None):
            errs.append(f"adjustment nullness mismatch at {e['note_index']}")
        elif es is not None:
            if es.get("hours") != as_.get("hours"):
                errs.append(
                    f"hours mismatch at {e['note_index']}: "
                    f"{as_.get('hours')} != {es.get('hours')}"
                )
            for k, v in es.items():
                if k == "hours":
                    continue
                if abs(float(as_.get(k, 0)) - float(v)) > 1e-6:
                    errs.append(
                        f"{k} mismatch at {e['note_index']}: "
                        f"{as_.get(k)} != {v}"
                    )
    return errs


def _run_case(case: dict) -> list[str]:
    """Return list of error strings; empty means pass."""
    errs: list[str] = []
    resp = client.post("/optimize-energy", json=case["input"])
    if resp.status_code != 200:
        return [f"HTTP {resp.status_code}: {resp.text[:200]}"]

    body = resp.json()
    exp = case["expected_output"]

    # 1. Interpretation semantics
    errs += _semantic_interpretation_match(
        exp["directive_interpretation"], body["directive_interpretation"]
    )

    # 2. Replay schedule and validate constraints
    req = OptimizeRequest(**case["input"])
    dirs = [DirectiveInterpretation(**d) for d in body["directive_interpretation"]]
    plan = [HourlyPlanEntry(**p) for p in body["hourly_plan"]]
    errs += replay_and_check(req.hours, req.battery, dirs, plan)

    # 3. Totals self-consistency
    recalc_grid = sum(p.grid_kwh for p in plan)
    recalc_cost = sum(
        p.grid_kwh * req.hours[p.hour].tariff_bdt_per_kwh for p in plan
    )
    recalc_peak = max(p.grid_kwh for p in plan)
    if abs(body["total_grid_kwh"] - recalc_grid) > 0.01:
        errs.append("total_grid_kwh mismatch")
    if abs(body["total_cost_bdt"] - recalc_cost) > 0.01:
        errs.append("total_cost_bdt mismatch")
    if abs(body["peak_grid_kwh"] - recalc_peak) > 0.01:
        errs.append("peak_grid_kwh mismatch")

    # 4. Optimization quality — must be <= reference cost + tolerance
    ref_cost = exp["total_cost_bdt"]
    if recalc_cost > ref_cost + 0.01:
        errs.append(
            f"cost {recalc_cost:.2f} worse than reference {ref_cost:.2f}"
        )

    return errs


def main(path: str) -> int:
    cases = _load_cases(path)
    failures = 0
    for case in cases:
        errs = _run_case(case)
        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {case['id']} — {case['label']}")
        for e in errs:
            print(f"        - {e}")
        failures += bool(errs)
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m tests.test_samples public_samples.json")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))