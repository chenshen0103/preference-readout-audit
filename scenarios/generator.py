"""Deterministic scenario generation (SPEC §10, §18; DECISIONS DR-2, DR-10).

The final set is a fixed structured grid, not random draws: the
near-indifference family is the cost sweep that the primary figure (§49)
reads off directly. Generation is fully deterministic; rerunning this module
always yields identical scenario IDs (test T4/T5).
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Scenario, make_scenario

# Sweep grid (DR-2): delta = benefit forgone by choosing the lower-cost option b.
# Levels chosen so that switch points delta* = (beta/alpha) * (cost_a - cost_b)
# fall inside the delta grid for plausible beta/alpha in [0.5, 2]:
# delta* in {7.5..12.5, 15..25, 30..50}. Recalibrate at Day-1 pilot if curves
# are flat, BEFORE SPEC_FREEZE (see OQ-8).
SWEEP_DELTAS = (2, 5, 10, 20, 40)
SWEEP_LEVELS = ((70, 25), (80, 30), (90, 35))  # (benefit_a, cost_a) for the risky option
SWEEP_COST_B = 10                              # fixed low cost of the safer option


def _near_indifference(set_name: str, start_idx: int, levels, deltas,
                       family: str = "near_indifference",
                       stress: int = 0) -> list[Scenario]:
    out = []
    idx = start_idx
    for (ba, ca) in levels:
        for d in deltas:
            out.append(make_scenario(family, idx, set_name,
                                     ba=ba, ca=ca, bb=ba - d, cb=SWEEP_COST_B,
                                     stress=stress, delta=d))
            idx += 1
    return out


def final_set() -> list[Scenario]:
    scenarios: list[Scenario] = []

    dominance = [(90, 10, 50, 40), (80, 15, 55, 45), (85, 20, 40, 60),
                 (95, 5, 60, 30), (75, 10, 45, 50)]
    for i, (ba, ca, bb, cb) in enumerate(dominance):
        scenarios.append(make_scenario("dominance", i, "final", ba, ca, bb, cb))

    clear = [(90, 80, 50, 10), (85, 70, 45, 15), (95, 85, 55, 20),
             (80, 75, 40, 12), (90, 70, 45, 11), (85, 80, 50, 15),
             (95, 75, 60, 25), (88, 72, 42, 12), (92, 78, 52, 18),
             (86, 68, 46, 14)]
    for i, (ba, ca, bb, cb) in enumerate(clear):
        scenarios.append(make_scenario("clear_tradeoff", i, "final", ba, ca, bb, cb))

    scenarios.extend(_near_indifference("final", 0, SWEEP_LEVELS, SWEEP_DELTAS))

    # Stress family: mid sweep level, delta x stress_level combinations (10 total).
    stress_combos = [(5, 1), (5, 2), (5, 3), (20, 1), (20, 2), (20, 3),
                     (10, 1), (10, 3), (40, 1), (40, 3)]
    ba, ca = 80, 30
    for i, (d, lvl) in enumerate(stress_combos):
        scenarios.append(make_scenario("stress", i, "final",
                                       ba=ba, ca=ca, bb=ba - d, cb=SWEEP_COST_B,
                                       stress=lvl, delta=d))
    return scenarios


def pilot_set() -> list[Scenario]:
    """10 scenarios for debugging only. MUST NOT enter final hypothesis tests (§18)."""
    scenarios: list[Scenario] = []
    for i, (ba, ca, bb, cb) in enumerate([(88, 12, 52, 42), (78, 14, 50, 46)]):
        scenarios.append(make_scenario("dominance", i, "pilot", ba, ca, bb, cb))
    for i, (ba, ca, bb, cb) in enumerate([(89, 79, 49, 11), (84, 69, 44, 16),
                                          (94, 84, 54, 21)]):
        scenarios.append(make_scenario("clear_tradeoff", i, "pilot", ba, ca, bb, cb))
    scenarios.extend(_near_indifference("pilot", 0, ((75, 28),), (3, 15, 30)))
    for i, (d, lvl) in enumerate([(15, 1), (15, 3)]):
        scenarios.append(make_scenario("stress", i, "pilot",
                                       ba=75, ca=28, bb=75 - d, cb=SWEEP_COST_B,
                                       stress=lvl, delta=d))
    return scenarios


def write_frozen(out_dir: str | Path) -> dict[str, Path]:
    """Serialize both sets to scenarios/frozen/ as JSON for the freeze commit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, scs in (("final", final_set()), ("pilot", pilot_set())):
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps([s.to_dict() for s in scs], indent=2),
                     encoding="utf-8")
        paths[name] = p
    return paths
