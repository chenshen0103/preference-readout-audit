"""Runner integration: resume behavior (SPEC §43), no duplicates (T9),
diagnostics wiring (SC1-SC3)."""
import json

import pytest

from analysis.clean import load_results
from analysis.diagnostics import run_all
from runners.batch import load_configs, run_batch
from scenarios.generator import pilot_set


@pytest.fixture
def small_cfg():
    exp, models, personas = load_configs()
    exp = dict(exp, experiment_version="test", domains=["generic", "finance"],
               personas=["P0"], prompt_templates=["fc_v1", "fc_v2"],
               choice_orders=["ab", "ba"])
    return exp, models["mock"], personas


def test_resume_skips_completed(tmp_path, small_cfg):
    exp, model_cfg, personas = small_cfg
    out = tmp_path / "results.jsonl"
    scs = pilot_set()

    s1 = run_batch(exp, model_cfg, scs, personas, out)
    assert s1["completed"] == s1["planned"] == 10 * 2 * 2 * 2
    assert s1["failed"] == 0

    s2 = run_batch(exp, model_cfg, scs, personas, out)
    assert s2["skipped"] == s2["planned"]
    assert s2["completed"] == 0

    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [json.loads(l)["trial_id"] for l in lines]
    assert len(ids) == len(set(ids)) == s1["planned"]


def test_mock_run_passes_sanity_checks(tmp_path, small_cfg):
    exp, model_cfg, personas = small_cfg
    out = tmp_path / "results.jsonl"
    scs = pilot_set()
    run_batch(exp, model_cfg, scs, personas, out)
    df = load_results(out, scs)
    diag = run_all(df, templates=["fc_v1", "fc_v2"])
    assert diag["SC1"]["pass"], diag["SC1"]   # mock must ace dominance
    assert diag["SC2"]["pass"], diag["SC2"]
    assert diag["SC3"]["pass"], diag["SC3"]   # unbiased mock + counterbalancing
