"""Domain renderers (SPEC §4, §8, §35; DECISIONS DR-5, DR-11, DR-13).

Renderers translate the abstract scenario into domain-flavored text.
They MUST preserve numeric payoff values verbatim and MUST NOT add
information that changes the decision problem. Stress values come from
the abstract schema via the frozen stress mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from scenarios.schema import Scenario

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))


def fmt_num(x: float) -> str:
    """Render payoff numbers verbatim; integers without a decimal point."""
    return str(int(x)) if float(x).is_integer() else str(x)


@dataclass(frozen=True)
class Rendered:
    text: str
    # Maps displayed label -> schema option ("a"/"b"), e.g. {"A": "b", "B": "a"}.
    displayed_to_schema: dict


class DomainRenderer:
    def __init__(self, domain: str):
        domains = _load_yaml("domains.yaml")
        if domain not in domains:
            raise KeyError(f"unknown domain: {domain}")
        cfg = domains[domain]
        self.domain = domain
        self.benefit_label = cfg["benefit_label"]
        self.cost_label = cfg["cost_label"]
        self.context_line = cfg["context_line"]
        self.harm_event_label = cfg["harm_event_label"]
        self.stress_levels = _load_yaml("stress.yaml")["levels"]

    def _option_block(self, label: str, benefit: float, cost: float,
                      presentation_order: str) -> str:
        b_line = f"{self.benefit_label}: {fmt_num(benefit)}"
        c_line = f"{self.cost_label}: {fmt_num(cost)}"
        first, second = ((b_line, c_line) if presentation_order == "benefit_first"
                         else (c_line, b_line))
        return f"Option {label}:\n{first}\n{second}"

    def _stress_block(self, level: int) -> str:
        p = self.stress_levels[level]
        return (f"Probability of a {self.harm_event_label}: "
                f"{p['harm_probability_pct']}%\n"
                f"Severity multiplier if the event occurs: "
                f"x{p['severity_multiplier']}")

    def render(self, s: Scenario, choice_order: str = "ab",
               presentation_order: str = "benefit_first") -> Rendered:
        if choice_order not in ("ab", "ba"):
            raise ValueError(f"choice_order must be 'ab' or 'ba', got {choice_order}")
        if presentation_order not in ("benefit_first", "cost_first"):
            raise ValueError(f"bad presentation_order: {presentation_order}")

        if choice_order == "ab":
            displayed = [("A", s.benefit_a, s.cost_a), ("B", s.benefit_b, s.cost_b)]
            mapping = {"A": "a", "B": "b"}
        else:
            displayed = [("A", s.benefit_b, s.cost_b), ("B", s.benefit_a, s.cost_a)]
            mapping = {"A": "b", "B": "a"}

        parts = [self.context_line, ""]
        for label, benefit, cost in displayed:
            parts.append(self._option_block(label, benefit, cost, presentation_order))
            parts.append("")
        if s.stress_level > 0:
            parts.append(self._stress_block(s.stress_level))
            parts.append("")
        return Rendered(text="\n".join(parts).strip(), displayed_to_schema=mapping)
