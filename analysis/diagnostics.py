"""Sanity checks SC1-SC5 (SPEC §32). A run is invalid unless these pass."""
from __future__ import annotations

import math

import pandas as pd

DOMINANCE_ACCURACY_MIN = 0.95
PARSE_SUCCESS_MIN = 0.99


def sc1_dominance_accuracy(df: pd.DataFrame) -> dict:
    dom = df[(df["family"] == "dominance") & (df["parse_status"] == "ok")]
    # Generator guarantees option 'a' is the dominant option (validator-enforced).
    acc = (dom["choice"] == "a").mean() if len(dom) else float("nan")
    return {"accuracy": float(acc), "n": len(dom),
            "pass": bool(len(dom) and acc >= DOMINANCE_ACCURACY_MIN)}


def sc2_parse_rate(df: pd.DataFrame) -> dict:
    rate = (df["parse_status"] == "ok").mean() if len(df) else float("nan")
    failures = df[df["parse_status"] != "ok"]["trial_id"].tolist()
    return {"rate": float(rate), "n": len(df), "failures": failures,
            "pass": bool(len(df) and rate >= PARSE_SUCCESS_MIN)}


def sc3_position_bias(df: pd.DataFrame) -> dict:
    """Fraction of trials where the DISPLAYED 'A' was chosen. Under the fully
    counterbalanced design this should be ~0.5; report a binomial 95% CI."""
    ok = df[df["parse_status"] == "ok"]
    n = len(ok)
    if n == 0:
        return {"p_displayed_a": float("nan"), "n": 0, "pass": False}
    p = (ok["displayed_choice"] == "A").mean()
    half_width = 1.96 * math.sqrt(p * (1 - p) / n)
    ci = (p - half_width, p + half_width)
    return {"p_displayed_a": float(p), "ci95": ci, "n": n,
            "pass": bool(ci[0] <= 0.5 <= ci[1] or abs(p - 0.5) < 0.10)}


def sc4_template_stability(df: pd.DataFrame, templates: list[str]) -> dict:
    """Direction of the safer-option tendency per template. If one template
    alone drives the effects, results must be labeled prompt-sensitive."""
    ok = df[df["parse_status"] == "ok"]
    per = {t: float(1 - ok[ok["prompt_template_id"] == t]["p_choose_a"].mean())
           for t in templates}
    directions = {t: v > 0.5 for t, v in per.items()}
    return {"p_choose_safer_by_template": per,
            "consistent_direction": len(set(directions.values())) == 1}


def run_all(df: pd.DataFrame, templates: list[str]) -> dict:
    out = {"SC1": sc1_dominance_accuracy(df), "SC2": sc2_parse_rate(df),
           "SC3": sc3_position_bias(df),
           "SC4": sc4_template_stability(df, templates)}
    out["all_pass"] = all(v.get("pass", True) for v in out.values()
                          if isinstance(v, dict))
    return out
