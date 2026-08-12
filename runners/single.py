"""Run a single trial for interactive debugging.

Usage (from repo root):
    python -m runners.single --model mock --domain finance --persona P1
"""
from __future__ import annotations

import argparse
import json

from runners.batch import load_configs, run_batch
from scenarios.generator import pilot_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock")
    ap.add_argument("--domain", default="generic")
    ap.add_argument("--persona", default="P0")
    ap.add_argument("--out", default="data/pilot/debug_single.jsonl")
    args = ap.parse_args()

    exp, models, personas = load_configs()
    exp = dict(exp, domains=[args.domain], personas=[args.persona],
               prompt_templates=exp["prompt_templates"][:1],
               choice_orders=["ab"])
    summary = run_batch(exp, models[args.model], pilot_set()[:1], personas, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
