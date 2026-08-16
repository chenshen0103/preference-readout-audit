"""Lane A: final-layer forced-choice numbers for the frozen prompt set.

Per prompt: restricted softmax over the answer letters (McFadden RUM choice
probability), the unrestricted top-1 token (format-compliance check), and per
scenario x variant a linear-interpolated crossover of the sweep.
IIA check: does adding the dominated third option change the A:B odds?

Outputs: results_lane_a.csv, crossovers_lane_a.csv
"""
import argparse
import csv
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def crossing(xs, ps, level=0.5):
    for i in range(len(ps) - 1):
        if (ps[i] - level) * (ps[i + 1] - level) <= 0 and ps[i] != ps[i + 1]:
            t = (level - ps[i]) / (ps[i + 1] - ps[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-1b-it")
    ap.add_argument("--prompts", default="analysis/jspace/prompts.jsonl")
    ap.add_argument("--out", default="analysis/jspace")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).to("cuda").eval()
    print(f"loaded {args.model}: {model.config.num_hidden_layers} layers")

    # tokenizer sanity: the prompt ends with "Answer:" (no space) — check both
    ids = {}
    for L in "ABC":
        for form in (L, " " + L):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                ids[form] = e[0]
    print("answer-token candidates:", ids)

    rows = []
    for line in open(args.prompts):
        r = json.loads(line)
        enc = tok(r["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1, :].float()
        letters = list(r["letter_map"])
        # per letter, take the better-scoring of bare vs space form
        lets = {}
        for L in letters:
            cands = [ids[f] for f in (L, " " + L) if f in ids]
            best = max(cands, key=lambda t: lg[t].item())
            lets[L] = best
        z = torch.tensor([lg[lets[L]] for L in letters])
        p = torch.softmax(z, 0).numpy()
        top1 = tok.decode([int(lg.argmax())])
        row = dict(condition_id=r["condition_id"], scenario=r["scenario"],
                   variant=r["variant"], sweep=r["sweep"],
                   top1_token=repr(top1),
                   top1_is_expected_letter=top1.strip() in letters)
        for L, pi in zip(letters, p):
            row[f"p_{L}"] = float(pi)
            row[f"logit_{L}"] = float(lg[lets[L]])
            row[f"role_{L}"] = r["letter_map"][L]
        # semantic prob of the 'risky'/'buy'/'fund' role
        target_roles = {"risky", "buy", "fund"}
        row["p_target"] = float(sum(pi for L, pi in zip(letters, p)
                                    if r["letter_map"][L] in target_roles))
        rows.append(row)
        print(f"  {r['condition_id']:<46} p_target={row['p_target']:.3f} "
              f"top1={row['top1_token']}")

    with open(f"{args.out}/results_lane_a.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    # crossovers per scenario x variant
    cross = []
    for scen in sorted({r["scenario"] for r in rows}):
        for var in sorted({r["variant"] for r in rows if r["scenario"] == scen}):
            sub = sorted([r for r in rows if r["scenario"] == scen
                          and r["variant"] == var and r["sweep"] is not None],
                         key=lambda r: r["sweep"])
            if len(sub) < 2:
                continue
            xs = [r["sweep"] for r in sub]
            ps = [r["p_target"] for r in sub]
            th = crossing(xs, ps)
            cross.append({"scenario": scen, "variant": var,
                          "theta_final": th,
                          "p_range": f"{min(ps):.3f}-{max(ps):.3f}",
                          "compliance": np.mean([r["top1_is_expected_letter"]
                                                 for r in sub])})
            print(f"crossover {scen}/{var}: theta={th}  "
                  f"p range {min(ps):.3f}-{max(ps):.3f}")
    with open(f"{args.out}/crossovers_lane_a.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader(); w.writerows(cross)

    # IIA: A:B odds with vs without the dominated C
    print("\nIIA check (finance): odds(risky:safe) base vs with dominated third")
    for r3 in [r for r in rows if r["variant"] == "dominated_third"]:
        base = next(r for r in rows if r["scenario"] == r3["scenario"]
                    and r["variant"] == "base" and r["sweep"] == r3["sweep"])
        def odds(row):
            pr = sum(v for k, v in row.items()
                     if k.startswith("p_") and k != "p_target"
                     and row.get("role_" + k[2:]) == "risky")
            ps = sum(v for k, v in row.items()
                     if k.startswith("p_") and k != "p_target"
                     and row.get("role_" + k[2:]) == "safe")
            return pr / max(ps, 1e-9)
        o2, o3 = odds(base), odds(r3)
        print(f"  sweep={r3['sweep']}: odds2={o2:.3f} odds3={o3:.3f} "
              f"ratio={o3/max(o2,1e-9):.3f}  p_dominated="
              f"{r3.get('p_C', float('nan')):.4f}")


if __name__ == "__main__":
    main()
