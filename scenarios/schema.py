"""Abstract scenario schema (SPEC §9). The abstract schema is the source of truth;
renderers may only translate labels and narrative presentation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

FAMILIES = ("dominance", "clear_tradeoff", "near_indifference", "stress")
PAYOFF_MIN, PAYOFF_MAX = 0, 100


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    benefit_a: float
    cost_a: float
    benefit_b: float
    cost_b: float
    stress_level: int = 0
    uncertainty_a: float | None = None
    uncertainty_b: float | None = None
    seed: int = 0
    set_name: str = "final"  # "pilot" | "final"
    # Sweep metadata: benefit forgone by choosing the lower-cost option b (DR-2).
    delta: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def scenario_hash(family: str, ba: float, ca: float, bb: float, cb: float,
                  stress: int, set_name: str) -> str:
    payload = json.dumps([family, ba, ca, bb, cb, stress, set_name])
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def make_scenario(family: str, idx: int, set_name: str,
                  ba: float, ca: float, bb: float, cb: float,
                  stress: int = 0, seed: int = 0,
                  delta: float | None = None) -> Scenario:
    h = scenario_hash(family, ba, ca, bb, cb, stress, set_name)
    sid = f"{set_name}-{family}-{idx:02d}-{h}"
    return Scenario(scenario_id=sid, family=family,
                    benefit_a=ba, cost_a=ca, benefit_b=bb, cost_b=cb,
                    stress_level=stress, seed=seed, set_name=set_name,
                    delta=delta)
