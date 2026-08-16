"""Analysis for steps 1-5 of the Gemma 4 readout validity investigation.

Reads analysis/validity/records.json and answers, per layer:
  1. are the candidate label tokens readable at all (vocabulary rank)?
  2. do simple readout corrections change that?
  3. does the signed margin mirror under content swap?
  4. does the margin sign track the SEMANTIC choice, separately from label identity?
  5. what does the behavioral anchor (final-layer margin, this fp16 checkpoint) look like?

No margin is reported at layers where both candidate labels are outside the
readable region; those numbers are not interpretable.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

VARIANTS = ["delta_finalnorm", "delta_raw", "delta_baselinesub"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="analysis/validity/records.json")
    args = ap.parse_args()
    recs = json.load(open(args.records))
    L = len(recs[0]["layers"])
    print(f"{len(recs)} traces, {L} layers\n")

    # ---------------- 5. behavioral anchor (do this first: it defines bins) ----
    print("=" * 78)
    print("5. BEHAVIORAL ANCHOR - final-layer margin from THIS fp16 checkpoint")
    print("=" * 78)
    ab = [r for r in recs if r["labels"] == "AB"]
    fm = np.array([abs(r["final_margin"]) for r in ab])
    qs = np.quantile(fm, [0, .33, .66, 1.0])
    print(f"  |final margin| over {len(ab)} A/B traces:")
    print(f"    min {qs[0]:.2f}   p33 {qs[1]:.2f}   p66 {qs[2]:.2f}   max {qs[3]:.2f}")
    print(f"    near-boundary (<p33): {(fm<qs[1]).sum()}  medium: "
          f"{((fm>=qs[1])&(fm<qs[2])).sum()}  easy (>=p66): {(fm>=qs[2]).sum()}")
    print("  -> difficulty bins now come from the model's own logits, not intuition")

    # ---------------- 1. readability ------------------------------------------
    print()
    print("=" * 78)
    print("1. CANDIDATE-TOKEN READABILITY (rank in vocabulary, 262k tokens)")
    print("=" * 78)
    med_a = [np.median([r["layers"][l]["rank_a"] for r in recs]) for l in range(L)]
    med_b = [np.median([r["layers"][l]["rank_b"] for r in recs]) for l in range(L)]
    frac_read = [np.mean([r["layers"][l]["readable"] for r in recs]) for l in range(L)]
    print(f"{'L':>3} {'med rank A':>11} {'med rank B':>11} {'% readable':>11} "
          f"{'|resid|':>9} {'|normed|':>9} {'logit std':>10}")
    for l in range(L):
        if l % 4 and l < L - 6:
            continue
        rn = np.mean([r["layers"][l]["resid_norm_pre"] for r in recs])
        nn = np.mean([r["layers"][l]["resid_norm_post"] for r in recs])
        ls = np.mean([r["layers"][l]["logit_std"] for r in recs])
        print(f"{l:>3} {med_a[l]:>11.0f} {med_b[l]:>11.0f} {100*frac_read[l]:>10.0f}% "
              f"{rn:>9.1f} {nn:>9.1f} {ls:>10.2f}")
    first_read = next((l for l in range(L) if frac_read[l] > 0.5), None)
    print(f"\n  first layer with >50% of traces readable: "
          f"{'L'+str(first_read) if first_read is not None else 'NEVER'}")
    print(f"  layers readable in >50% of traces: "
          f"{sum(1 for l in range(L) if frac_read[l] > 0.5)}/{L}")

    # ---------------- 3+2. content swap mirroring, per variant ----------------
    print()
    print("=" * 78)
    print("3 + 2. CONTENT-SWAP MIRRORING  (valid readout => delta_swap ~ -delta_orig)")
    print("=" * 78)
    by_key = defaultdict(dict)
    for r in recs:
        by_key[(r["pair"], r["labels"])][r["order"]] = r

    print(f"{'variant':>20} {'all layers':>12} {'readable only':>15} {'final layer':>13}")
    for v in VARIANTS:
        allc, readc, finc = [], [], []
        for k, d in by_key.items():
            if "orig" not in d or "swap" not in d:
                continue
            o = np.array([x[v] for x in d["orig"]["layers"]])
            s = np.array([x[v] for x in d["swap"]["layers"]])
            mask = np.array([x["readable"] for x in d["orig"]["layers"]]) & \
                   np.array([x["readable"] for x in d["swap"]["layers"]])
            if np.std(o) > 0 and np.std(s) > 0:
                allc.append(np.corrcoef(o, -s)[0, 1])
            if mask.sum() > 2 and np.std(o[mask]) > 0 and np.std(s[mask]) > 0:
                readc.append(np.corrcoef(o[mask], -s[mask])[0, 1])
            finc.append(np.sign(o[-1]) != np.sign(s[-1]))
        print(f"{v:>20} {np.mean(allc):>+12.3f} "
              f"{(np.mean(readc) if readc else float('nan')):>+15.3f} "
              f"{100*np.mean(finc):>12.0f}%")
    print("\n  'final layer' column = % of pairs whose final answer LABEL flips on swap")
    print("  (a semantically consistent model should flip the label ~100% of the time)")

    # ---------------- 4. label readability vs semantic tracking ---------------
    print()
    print("=" * 78)
    print("4. SEMANTIC TRACKING vs LABEL IDENTITY")
    print("=" * 78)
    print("  Does sign(margin) at layer L predict the trace's final semantic choice?")
    print(f"{'L':>4} {'acc AB':>9} {'acc XY':>9} {'%readable':>10}")
    for l in list(range(0, L, 6)) + list(range(L - 5, L)):
        row = {}
        for lab in ("AB", "XY"):
            sub = [r for r in recs if r["labels"] == lab]
            hits = []
            for r in sub:
                # positive margin => first label chosen => content_a chosen
                pred_a = r["layers"][l]["delta_finalnorm"] > 0
                chose_a = (r["semantic_choice"] == "x") == (r["order"] == "orig")
                hits.append(pred_a == chose_a)
            row[lab] = np.mean(hits)
        fr = np.mean([r["layers"][l]["readable"] for r in recs])
        print(f"{l:>4} {row['AB']:>9.2f} {row['XY']:>9.2f} {100*fr:>9.0f}%")
    print("\n  0.50 = chance. Values at unreadable layers are not interpretable.")

    # cross-label agreement of the final answer
    agree = []
    per_pair = defaultdict(dict)
    for r in recs:
        per_pair[(r["pair"], r["order"])][r["labels"]] = r["semantic_choice"]
    for k, d in per_pair.items():
        if "AB" in d and "XY" in d:
            agree.append(d["AB"] == d["XY"])
    print(f"  final semantic choice agrees across A/B vs X/Y labels: "
          f"{100*np.mean(agree):.0f}% of {len(agree)} cases")


if __name__ == "__main__":
    main()
