"""V1 — planted-preference recovery (the end-to-end measurement-integrity test).

A mock model with a PLANTED utility U = alpha*B - beta*C decides purely from
the rendered prompt text. The full pipeline (generator -> renderer -> runner
-> result store -> analysis fit) must recover beta/alpha. If any stage
distorts payoffs, swaps mappings, or mislabels options, recovery fails.
"""
import math

import pytest

from analysis.clean import load_results
from analysis.metrics import m1_invariance, m5_prompt_sensitivity
from analysis.utility_fit import fit_choice_logistic, fit_exact_from_probability
from runners.batch import load_configs, run_batch
from scenarios.generator import final_set

PLANTED_ALPHA, PLANTED_BETA = 1.0, 2.0


@pytest.fixture(scope="module")
def results_df(tmp_path_factory):
    exp, models, personas = load_configs()
    exp = dict(exp, experiment_version="recovery-test")
    model_cfg = dict(models["mock"], alpha=PLANTED_ALPHA, beta=PLANTED_BETA)
    out = tmp_path_factory.mktemp("recovery") / "results.jsonl"
    scs = final_set()
    summary = run_batch(exp, model_cfg, scs, personas, out)
    assert summary["failed"] == 0
    assert summary["completed"] == 40 * 3 * 3 * 3 * 2  # DR-1 primary matrix
    return load_results(out, scs)


def test_exact_ratio_recovery(results_df):
    fit = fit_exact_from_probability(results_df)
    assert math.isclose(fit["ratio"], PLANTED_BETA / PLANTED_ALPHA,
                        rel_tol=1e-6), fit


def test_mle_ratio_recovery_binary(results_df):
    fit = fit_choice_logistic(results_df)
    # Binary argmax choices lose information vs continuous P; the boundary
    # (ratio) must still be recovered within a loose tolerance.
    assert abs(fit["ratio"] - PLANTED_BETA / PLANTED_ALPHA) < 0.3, fit


def test_domain_invariance_of_planted_model(results_df):
    # The mock is domain-blind by construction -> M1 must be ~0 and
    # per-domain fitted ratios identical.
    for domain in ("generic", "finance", "biosafety"):
        fit = fit_exact_from_probability(results_df[results_df["domain"] == domain])
        assert math.isclose(fit["ratio"], 2.0, rel_tol=1e-6), (domain, fit)
    m1 = m1_invariance(results_df, ["generic", "finance", "biosafety"], n_boot=20)
    assert m1["m1"] < 0.02, m1


def test_prompt_insensitivity_of_planted_model(results_df):
    m5 = m5_prompt_sensitivity(results_df, ["fc_v1", "fc_v2", "fc_v3"])
    assert m5["range"] < 1e-6, m5
