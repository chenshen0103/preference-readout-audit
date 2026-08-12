"""Scenario validation and dominance labeling (SPEC §10.1, §32 SC-checks)."""
from __future__ import annotations

from .schema import Scenario, FAMILIES, PAYOFF_MIN, PAYOFF_MAX

DOMINANCE_MARGIN = 20  # DR-10


def dominance_label(s: Scenario) -> str | None:
    """Return 'a' or 'b' if that option dominates (>= on benefit, <= on cost,
    at least one strict), else None."""
    if s.benefit_a >= s.benefit_b and s.cost_a <= s.cost_b and (
            s.benefit_a > s.benefit_b or s.cost_a < s.cost_b):
        return "a"
    if s.benefit_b >= s.benefit_a and s.cost_b <= s.cost_a and (
            s.benefit_b > s.benefit_a or s.cost_b < s.cost_a):
        return "b"
    return None


def validate_scenario(s: Scenario) -> list[str]:
    """Return a list of violation strings; empty list means valid."""
    errors = []
    if s.family not in FAMILIES:
        errors.append(f"{s.scenario_id}: unknown family {s.family}")
    for name in ("benefit_a", "cost_a", "benefit_b", "cost_b"):
        v = getattr(s, name)
        if not (PAYOFF_MIN <= v <= PAYOFF_MAX):
            errors.append(f"{s.scenario_id}: {name}={v} outside [{PAYOFF_MIN},{PAYOFF_MAX}]")
        if float(v) != int(v):
            errors.append(f"{s.scenario_id}: {name}={v} not an integer (DR-10)")
    if not (0 <= s.stress_level <= 3):
        errors.append(f"{s.scenario_id}: stress_level={s.stress_level} outside 0..3")
    if s.family == "stress" and s.stress_level == 0:
        errors.append(f"{s.scenario_id}: stress family requires stress_level >= 1")
    if s.family != "stress" and s.stress_level != 0:
        errors.append(f"{s.scenario_id}: non-stress family must have stress_level 0")

    dom = dominance_label(s)
    if s.family == "dominance":
        if dom != "a":
            errors.append(f"{s.scenario_id}: dominance family must have option a dominant")
        elif (s.benefit_a - s.benefit_b < DOMINANCE_MARGIN
              or s.cost_b - s.cost_a < DOMINANCE_MARGIN):
            errors.append(f"{s.scenario_id}: dominance margin < {DOMINANCE_MARGIN}")
    else:
        if dom is not None:
            errors.append(f"{s.scenario_id}: family {s.family} must not contain a dominant option")
    return errors


def validate_set(scenarios: list[Scenario]) -> list[str]:
    errors = []
    seen = set()
    for s in scenarios:
        if s.scenario_id in seen:
            errors.append(f"duplicate scenario_id {s.scenario_id}")
        seen.add(s.scenario_id)
        errors.extend(validate_scenario(s))
    return errors
