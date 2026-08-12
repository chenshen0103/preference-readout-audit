"""Batch runner with resume (SPEC §43) and machine-readable logging (§44).

Results are an append-only JSONL store keyed by deterministic trial_id.
Existing valid trials are skipped; previous outputs are never overwritten.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path

import yaml

from models import build_adapter
from prompts.registry import load_template
from renderers import get_renderer
from scenarios.schema import Scenario
from schemas.trial import Condition, TrialRecord, compute_trial_id

log = logging.getLogger("pressuretest.runner")

_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def presentation_order_for(scenario_id: str, template_id: str) -> str:
    """Deterministic 50/50 counterbalancing (DR-11)."""
    h = hashlib.sha256(f"{scenario_id}|{template_id}".encode()).hexdigest()
    return "benefit_first" if int(h, 16) % 2 == 0 else "cost_first"


def build_conditions(exp_cfg: dict, scenarios: list[Scenario]) -> list[Condition]:
    conds = []
    for s in scenarios:
        for domain in exp_cfg["domains"]:
            for persona in exp_cfg["personas"]:
                for template_id in exp_cfg["prompt_templates"]:
                    for choice_order in exp_cfg["choice_orders"]:
                        conds.append(Condition(
                            scenario_id=s.scenario_id, domain=domain,
                            persona=persona,
                            elicitation_method=exp_cfg["elicitation_method"],
                            prompt_template_id=template_id,
                            stress_level=s.stress_level,
                            choice_order=choice_order,
                            presentation_order=presentation_order_for(
                                s.scenario_id, template_id)))
    return conds


def load_existing_ids(out_path: Path) -> set[str]:
    ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("response", {}).get("parse_status") == "ok":
                ids.add(rec["trial_id"])
    return ids


def run_batch(exp_cfg: dict, model_cfg: dict, scenarios: list[Scenario],
              personas: dict, out_path: str | Path) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(model_cfg)
    meta = adapter.get_metadata()
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    conditions = build_conditions(exp_cfg, scenarios)
    seed = exp_cfg["generation"]["seed"]
    version = exp_cfg["experiment_version"]
    git_commit = _git_commit()

    existing = load_existing_ids(out_path)
    summary = {"planned": len(conditions), "completed": 0, "skipped": 0,
               "failed": 0, "parse_failures": 0,
               "config_hash": _config_hash(exp_cfg),
               "model_revision": meta["revision"],
               "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    log.info("run start: %s", json.dumps(summary))

    with out_path.open("a", encoding="utf-8") as fh:
        for cond in conditions:
            trial_id = compute_trial_id(version, meta["revision"], cond, seed)
            if trial_id in existing:
                summary["skipped"] += 1
                continue
            s = scenario_by_id[cond.scenario_id]
            try:
                rendered = get_renderer(cond.domain).render(
                    s, choice_order=cond.choice_order,
                    presentation_order=cond.presentation_order)
                template = load_template(cond.prompt_template_id)
                user = template.format(scenario_body=rendered.text)
                system = personas[cond.persona][cond.domain]

                logprobs = adapter.score_choices(system, user)
                displayed = max(logprobs, key=logprobs.get)
                choice = rendered.displayed_to_schema[displayed]  # "a" | "b"
                p = {k: math.exp(v) for k, v in logprobs.items()}
                p_a_schema = (p["A"] if rendered.displayed_to_schema["A"] == "a"
                              else p["B"])
                response = {"raw": json.dumps(logprobs),
                            "choice": choice,
                            "displayed_choice": displayed,
                            "p_choose_schema_a": p_a_schema,
                            "confidence": max(p.values()),
                            "parse_status": "ok"}
            except Exception as e:  # noqa: BLE001 — failures logged, never coerced (T7)
                summary["failed"] += 1
                response = {"raw": "", "choice": None, "confidence": None,
                            "parse_status": f"error: {type(e).__name__}: {e}"}
                log.error(json.dumps({"trial_id": trial_id, "error": str(e)}))

            rec = TrialRecord(
                trial_id=trial_id, experiment_version=version,
                git_commit=git_commit, scenario_id=s.scenario_id,
                model=meta, condition=cond.__dict__,
                payoff={"benefit_a": s.benefit_a, "cost_a": s.cost_a,
                        "benefit_b": s.benefit_b, "cost_b": s.cost_b},
                generation=dict(exp_cfg["generation"]),
                response=response,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
            fh.write(json.dumps(rec.to_dict()) + "\n")
            if response["parse_status"] == "ok":
                summary["completed"] += 1

    summary["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log.info("run end: %s", json.dumps(summary))
    return summary


def load_configs(config_dir: str | Path = None) -> tuple[dict, dict, dict]:
    d = Path(config_dir) if config_dir else _ROOT / "configs"
    exp = yaml.safe_load((d / "experiment.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((d / "models.yaml").read_text(encoding="utf-8"))
    personas = yaml.safe_load((d / "personas.yaml").read_text(encoding="utf-8"))
    return exp, models, personas
