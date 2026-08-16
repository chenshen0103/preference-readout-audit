#!/usr/bin/env python3
"""One-command reproduction of the paper's core behavioral result:

    Near the elicited indifference boundary, every answer in the 22
    order-discordant paired conditions selected the SECOND-LISTED option
    (44/44).

Needs only the Python standard library. Recomputes the claim from the raw
per-condition records committed in this repository (final-layer logit margins
from single deterministic forward passes of google/gemma-4-31B-it, fp16) —
it does not re-run the model. To regenerate the records themselves from
scratch (GPU required), see "Full regeneration" in README.md.

Every condition was presented twice, with the two option contents exchanged
between the A/B slots ("orig" / "swap"). margin = logit(A) - logit(B) at the
final layer, so the raw answer is "A" if margin > 0 else "B". The option in
slot B is always the second-listed one.

Exit code 0 and "PASS" means every number in the claim checks out.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pair_up(records, key_fn):
    pairs = defaultdict(dict)
    for r in records:
        pairs[key_fn(r)][r["order"]] = r["margin"]
    return [(v["orig"], v["swap"]) for v in pairs.values()
            if "orig" in v and "swap" in v]


def main():
    pairs = []
    boundary = json.load(open(ROOT / "analysis/boundary/meta.json"))
    pairs += pair_up(boundary, lambda m: ("b", m["paraphrase"], m["social_cost"]))
    veto = json.load(open(ROOT / "analysis/veto/meta.json"))
    pairs += pair_up(veto, lambda m: ("v", m["test"], m["framing"],
                                     m["ret"], m["cost"]))
    sweep = json.load(open(ROOT / "analysis/sweep/meta.json"))
    pairs += pair_up([m for m in sweep if m["set"] == "extreme"],
                     lambda m: ("e", m["name"]))
    pairs += pair_up([m for m in sweep if m["set"] == "invest"],
                     lambda m: ("i", m["social_cost"]))
    pilot = list(csv.DictReader(open(ROOT / "analysis/pilot_geo/behavioral_results.csv")))
    pairs += pair_up([{"order": "swap" if r["swap_variant"] == "True" else "orig",
                       "margin": float(r["final_margin"]),
                       "sc": r["social_cost"]} for r in pilot],
                     lambda m: ("p", m["sc"]))

    # A pair agrees on CONTENT when the same content wins in both
    # presentations: semantic margin is m_orig in orig and -m_swap in swap.
    discordant = [(mo, ms) for mo, ms in pairs if (mo > 0) == (ms > 0)]
    concordant_n = len(pairs) - len(discordant)

    # In every discordant answer, which slot won? margin<0 -> answered "B",
    # the second-listed slot (this holds for BOTH presentations of a
    # discordant pair by construction: same sign of raw margin).
    answers = [m for pair in discordant for m in pair]
    second_listed = sum(1 for m in answers if m < 0)
    first_listed = len(answers) - second_listed

    print("Core result check — order-discordant pairs choose the second-listed option")
    print("-" * 74)
    print(f"paired conditions (both presentations present) : {len(pairs)}")
    print(f"  agree on content (genuine preference)        : {concordant_n}")
    print(f"  DISAGREE (order decides the outcome)         : {len(discordant)}")
    print(f"answers within discordant pairs                : {len(answers)}")
    print(f"  chose the SECOND-listed option (raw 'B')     : {second_listed}")
    print(f"  chose the first-listed option (raw 'A')      : {first_listed}")
    m_abs = sorted(abs(m) for m in answers)
    print(f"  |logit gap| range in those answers           : "
          f"{m_abs[0]:.1f} – {m_abs[-1]:.1f}  (confident, not near-zero)")

    # Supporting evidence committed alongside:
    rev = json.load(open(ROOT / "analysis/veto/reversed_listing.json"))
    n2 = sum(x["second_listed"] for x in rev)
    print(f"\nlayout-swap test (Option B block printed FIRST): "
          f"second-listed slot still chosen {n2}/{len(rev)}"
          f"  -> the default follows POSITION, not the letter")
    recs = json.load(open(ROOT / "analysis/validity/records.json"))
    by = defaultdict(dict)
    for r in recs:
        by[(r["pair"], r["labels"])][r["order"]] = r
    lab = defaultdict(int)
    for d in by.values():
        if len(d) == 2 and d["orig"]["semantic_choice"] != d["swap"]["semantic_choice"]:
            lab[(d["orig"]["answer_label"], d["swap"]["answer_label"])] += 1
    n_second = lab.get(("B", "B"), 0) + lab.get(("Y", "Y"), 0)
    n_lab = sum(lab.values())
    print(f"independent everyday-pairs set (A/B and X/Y labels): "
          f"second label chosen {n_second}/{n_lab}"
          f"  -> not specific to the letter 'B'")

    ok = (len(pairs) == 74 and len(discordant) == 22
          and second_listed == 44 and first_listed == 0
          and n2 == 21 and len(rev) == 24 and n_second == 13 and n_lab == 14)
    print("\n" + ("PASS — all numbers match the paper (74 pairs, 22 discordant, "
                  "44/44 second-listed; 21/24 layout; 13/14 labels)."
                  if ok else "FAIL — recomputed numbers deviate from the paper."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
