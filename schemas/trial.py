"""Trial record schema and deterministic trial IDs (SPEC §41-§42, DR-11)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class Condition:
    scenario_id: str
    domain: str
    persona: str
    elicitation_method: str
    prompt_template_id: str
    stress_level: int
    choice_order: str          # "ab" | "ba"
    presentation_order: str    # "benefit_first" | "cost_first"


def compute_trial_id(experiment_version: str, model_revision: str,
                     cond: Condition, seed: int) -> str:
    payload = "|".join([
        experiment_version, cond.scenario_id, str(model_revision), cond.domain,
        cond.persona, cond.elicitation_method, cond.prompt_template_id,
        str(cond.stress_level), cond.choice_order, cond.presentation_order,
        str(seed),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class TrialRecord:
    trial_id: str
    experiment_version: str
    git_commit: str
    scenario_id: str
    model: dict
    condition: dict
    payoff: dict
    generation: dict
    response: dict
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)
