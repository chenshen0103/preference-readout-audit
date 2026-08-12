"""T4, T5, T8, T10 + set composition (SPEC §18, §45)."""
from scenarios.generator import final_set, pilot_set
from scenarios.validator import dominance_label, validate_set


def test_final_composition():
    scs = final_set()
    assert len(scs) == 40
    counts = {}
    for s in scs:
        counts[s.family] = counts.get(s.family, 0) + 1
    assert counts == {"dominance": 5, "clear_tradeoff": 10,
                      "near_indifference": 15, "stress": 10}


def test_pilot_composition():
    assert len(pilot_set()) == 10


def test_all_scenarios_valid():
    assert validate_set(final_set()) == []
    assert validate_set(pilot_set()) == []


def test_ids_stable_across_reruns():  # T4 / T5
    a = [s.scenario_id for s in final_set()]
    b = [s.scenario_id for s in final_set()]
    assert a == b


def test_pilot_final_disjoint():  # T10
    f = {s.scenario_id for s in final_set()}
    p = {s.scenario_id for s in pilot_set()}
    assert not (f & p)
    assert all(s.set_name == "final" for s in final_set())
    assert all(s.set_name == "pilot" for s in pilot_set())


def test_dominance_labels():  # T8
    for s in final_set() + pilot_set():
        if s.family == "dominance":
            assert dominance_label(s) == "a"
        else:
            assert dominance_label(s) is None
