"""Decision-model fitting (SPEC §28-§29, DR-9).

Terminology: fitted beta/alpha is the INFERRED TRADE-OFF PARAMETER,
never 'true utility' (SPEC §29).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d_benefit = (df["benefit_a"] - df["benefit_b"]).to_numpy(dtype=float)
    d_cost = (df["cost_a"] - df["cost_b"]).to_numpy(dtype=float)
    return d_benefit, d_cost


def fit_exact_from_probability(df: pd.DataFrame) -> dict:
    """Linear regression on logit(P(choose a)) — exact recovery when the
    generating process is logistic in (dBenefit, dCost). Used by the
    planted-preference recovery test (V1) and as a descriptive fit for
    logprob-scored trials (continuous P)."""
    ok = df[df["parse_status"] == "ok"].copy()
    p = ok["p_choose_a"].clip(1e-9, 1 - 1e-9).to_numpy(dtype=float)
    y = np.log(p / (1 - p))
    db, dc = _design(ok)
    X = np.column_stack([np.ones_like(db), db, dc])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, c_benefit, c_cost = coef
    alpha, beta = c_benefit, -c_cost
    return {"alpha": alpha, "beta": beta,
            "ratio": beta / alpha if alpha != 0 else np.nan,
            "intercept": intercept, "n": len(ok)}


def fit_choice_logistic(df: pd.DataFrame) -> dict:
    """Logistic MLE on binary choices with cluster-robust SEs by scenario (DR-9)."""
    import statsmodels.api as sm

    ok = df[df["parse_status"] == "ok"].copy()
    y = (ok["choice"] == "a").astype(int).to_numpy()
    db, dc = _design(ok)
    X = sm.add_constant(np.column_stack([db, dc]))
    model = sm.Logit(y, X)
    res = model.fit(disp=0, cov_type="cluster",
                    cov_kwds={"groups": ok["scenario_id"].to_numpy()})
    alpha, beta = res.params[1], -res.params[2]
    ci = res.conf_int()
    return {"alpha": alpha, "beta": beta,
            "ratio": beta / alpha if alpha != 0 else np.nan,
            "alpha_ci": (ci[1][0], ci[1][1]),
            "beta_ci": (-ci[2][1], -ci[2][0]),
            "n": len(ok), "converged": res.mle_retvals.get("converged", True)}
