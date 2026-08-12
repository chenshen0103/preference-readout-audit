"""Load trial JSONL into an analysis DataFrame, joined with scenario fields."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scenarios.schema import Scenario


def load_results(path: str | Path, scenarios: list[Scenario]) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append({
            "trial_id": rec["trial_id"],
            "scenario_id": rec["scenario_id"],
            "model_name": rec["model"]["name"],
            "model_revision": rec["model"]["revision"],
            **rec["condition"],
            **rec["payoff"],
            "choice": rec["response"].get("choice"),
            "displayed_choice": rec["response"].get("displayed_choice"),
            "p_choose_a": rec["response"].get("p_choose_schema_a"),
            "confidence": rec["response"].get("confidence"),
            "parse_status": rec["response"].get("parse_status"),
        })
    df = pd.DataFrame(rows)
    sc = pd.DataFrame([{"scenario_id": s.scenario_id, "family": s.family,
                        "delta": s.delta, "set_name": s.set_name}
                       for s in scenarios])
    return df.merge(sc, on="scenario_id", how="left")
