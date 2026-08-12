"""T1, T2, T3 + neutral-naming and stress-block checks (SPEC §32 SC5, §35, §45)."""
import itertools
from pathlib import Path

import pytest
import yaml

from renderers import get_renderer
from renderers.base import fmt_num
from renderers.biosafety import contains_forbidden
from scenarios.generator import final_set, pilot_set

DOMAINS = ("generic", "finance", "biosafety")
ORDERS = ("ab", "ba")
PRESENTATIONS = ("benefit_first", "cost_first")
ALL_SCENARIOS = final_set() + pilot_set()
VALUE_LOADED = ("safe", "dangerous", "selfish", "ethical", "reckless")


def _all_renderings():
    for s, domain, co, po in itertools.product(
            ALL_SCENARIOS, DOMAINS, ORDERS, PRESENTATIONS):
        yield s, domain, co, po, get_renderer(domain).render(
            s, choice_order=co, presentation_order=po)


def test_t1_payoffs_preserved_verbatim():
    for s, domain, co, po, r in _all_renderings():
        for v in (s.benefit_a, s.cost_a, s.benefit_b, s.cost_b):
            assert f": {fmt_num(v)}" in r.text, (
                f"{s.scenario_id}/{domain}/{co}/{po}: value {v} missing")


def test_t2_ab_swap_swaps_values_and_mapping():
    r = get_renderer("generic")
    for s in ALL_SCENARIOS:
        ab = r.render(s, choice_order="ab")
        ba = r.render(s, choice_order="ba")
        assert ab.displayed_to_schema == {"A": "a", "B": "b"}
        assert ba.displayed_to_schema == {"A": "b", "B": "a"}
        a_block_ab = ab.text.split("Option B:")[0]
        a_block_ba = ba.text.split("Option B:")[0]
        assert f": {fmt_num(s.benefit_a)}" in a_block_ab
        assert f": {fmt_num(s.benefit_b)}" in a_block_ba


def test_t3_biosafety_blocklist():
    for s in ALL_SCENARIOS:
        for co in ORDERS:
            text = get_renderer("biosafety").render(s, choice_order=co).text
            assert contains_forbidden(text) == [], (
                f"forbidden terms in {s.scenario_id}: {contains_forbidden(text)}")


def test_t3_blocklist_covers_templates_and_personas():
    root = Path(__file__).resolve().parent.parent
    for p in (root / "prompts").rglob("*.txt"):
        text = p.read_text(encoding="utf-8")
        assert contains_forbidden(text) == [], (
            f"forbidden terms in {p.name}: {contains_forbidden(text)}")
    personas = yaml.safe_load(
        (root / "configs" / "personas.yaml").read_text(encoding="utf-8"))
    bio_texts = " ".join(v["biosafety"] for v in personas.values())
    assert contains_forbidden(bio_texts) == []


def test_neutral_naming():
    import re
    for s, domain, co, po, r in _all_renderings():
        low = r.text.lower()
        for word in VALUE_LOADED:
            # Word-boundary match: "Safety risk" (metric label) is allowed;
            # calling an OPTION "safe"/"dangerous" is not (SPEC §15.4).
            assert not re.search(rf"\b{word}\b", low), (
                f"value-loaded word '{word}' in {domain}")


def test_stress_block_presence():
    r = get_renderer("generic")
    for s in ALL_SCENARIOS:
        text = r.render(s).text
        if s.stress_level > 0:
            assert "Probability of a" in text
            assert "Severity multiplier" in text
        else:
            assert "Probability" not in text


def test_stress_values_match_config():
    import yaml
    root = Path(__file__).resolve().parent.parent
    levels = yaml.safe_load(
        (root / "configs" / "stress.yaml").read_text(encoding="utf-8"))["levels"]
    r = get_renderer("finance")
    for s in ALL_SCENARIOS:
        if s.stress_level > 0:
            text = r.render(s).text
            cfg = levels[s.stress_level]
            assert f"{cfg['harm_probability_pct']}%" in text
            assert f"x{cfg['severity_multiplier']}" in text


def test_bad_choice_order_rejected():
    with pytest.raises(ValueError):
        get_renderer("generic").render(ALL_SCENARIOS[0], choice_order="xx")
