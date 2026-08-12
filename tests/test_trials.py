"""T9 — deterministic trial IDs, no duplication (SPEC §42)."""
import dataclasses

from schemas.trial import Condition, compute_trial_id

BASE = Condition(scenario_id="final-dominance-00-abc", domain="generic",
                 persona="P0", elicitation_method="forced_choice_logprob",
                 prompt_template_id="fc_v1", stress_level=0,
                 choice_order="ab", presentation_order="benefit_first")


def test_id_deterministic():
    a = compute_trial_id("0.1.0", "rev1", BASE, 0)
    b = compute_trial_id("0.1.0", "rev1", BASE, 0)
    assert a == b


def test_id_changes_with_every_condition_field():
    base_id = compute_trial_id("0.1.0", "rev1", BASE, 0)
    variants = [
        dataclasses.replace(BASE, scenario_id="other"),
        dataclasses.replace(BASE, domain="finance"),
        dataclasses.replace(BASE, persona="P1"),
        dataclasses.replace(BASE, elicitation_method="rating"),
        dataclasses.replace(BASE, prompt_template_id="fc_v2"),
        dataclasses.replace(BASE, stress_level=1),
        dataclasses.replace(BASE, choice_order="ba"),
        dataclasses.replace(BASE, presentation_order="cost_first"),
    ]
    ids = {compute_trial_id("0.1.0", "rev1", v, 0) for v in variants}
    ids.add(compute_trial_id("0.2.0", "rev1", BASE, 0))
    ids.add(compute_trial_id("0.1.0", "rev2", BASE, 0))
    ids.add(compute_trial_id("0.1.0", "rev1", BASE, 1))
    assert base_id not in ids
    assert len(ids) == 11
