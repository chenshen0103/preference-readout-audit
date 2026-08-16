"""Probe readout over the boundary sweep.

Trains per-layer linear probes on the earlier validity set (192 traces of
random outcome pairs; validated to transfer across scenarios and label sets),
then applies them — frozen — to the boundary-sweep traces. The probe score,
sign-adjusted for content order, is the layerwise readout: positive = leaning
toward the PROFIT outcome.

Questions answered:
  1. Behavior: where is the choice boundary, per paraphrase and order?
  2. Depth: at which layers does the probe score start depending on social
     cost (the intervention), and does it agree with the final choice?
  3. Boundary difficulty: is the internal signal weaker near the behavioral
     boundary than far from it?
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VDIR, BDIR = "analysis/validity", "analysis/boundary"
L_LO, L_HI = 28, 57


def main():
    Xv = np.load(f"{VDIR}/residuals.npy").astype(np.float32)
    mv = json.load(open(f"{VDIR}/meta.json"))
    yv = np.array([(m["semantic_choice"] == "x") == (m["order"] == "orig")
                   for m in mv], dtype=int)          # chose first-listed

    Xb = np.load(f"{BDIR}/residuals.npy").astype(np.float32)
    mb = json.load(open(f"{BDIR}/meta.json"))
    nb, L, H = Xb.shape
    sc = np.array([m["social_cost"] for m in mb], float)
    order = np.array([m["order"] for m in mb])
    para = np.array([m["paraphrase"] for m in mb])
    margin = np.array([m["margin"] for m in mb])
    chose_profit = np.array([m["semantic_choice"] == "PROFIT" for m in mb])
    print(f"validity {Xv.shape} -> boundary {Xb.shape}\n")

    # ---- 1. behavior ------------------------------------------------------
    print("=" * 74)
    print("1. BEHAVIOR: profit chosen? (rows: paraphrase x order; cols: cost)")
    print("=" * 74)
    costs = sorted(set(sc))
    print(f"      cost:  {'  '.join(f'{c:>4}' for c in costs)}")
    thresholds = {}
    for p in sorted(set(para)):
        for o in ("orig", "swap"):
            sel = (para == p) & (order == o)
            row, xs, ms = [], [], []
            for c in costs:
                i = np.where(sel & (sc == c))[0][0]
                row.append("P" if chose_profit[i] else ".")
                xs.append(c)
                # semantic margin: + = profit side
                ms.append(margin[i] if o == "orig" else -margin[i])
            # crossing point of semantic margin
            thr = None
            for i in range(len(ms) - 1):
                if (ms[i] > 0) != (ms[i + 1] > 0):
                    t = ms[i] / (ms[i] - ms[i + 1])
                    thr = xs[i] + t * (xs[i + 1] - xs[i]); break
            thresholds[(p, o)] = thr
            ts = f"{thr:.2f}%" if thr is not None else "none"
            print(f"  {p} {o:<4}   {'  '.join(f'{r:>4}' for r in row)}   thr~{ts}")

    # ---- train frozen probes on validity set ------------------------------
    scalers, probes = {}, {}
    for l in range(L_LO, L_HI + 1):
        s = StandardScaler().fit(Xv[:, l, :])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(
            s.transform(Xv[:, l, :]), yv)
        scalers[l], probes[l] = s, clf

    # semantic probe score: + = profit (probe predicts first-listed chosen)
    S = np.zeros((nb, L)) * np.nan
    for l in range(L_LO, L_HI + 1):
        raw = probes[l].decision_function(scalers[l].transform(Xb[:, l, :]))
        raw = raw / (np.std(raw) + 1e-9)
        S[:, l] = np.where(order == "orig", raw, -raw)

    # ---- 2. depth of cost dependence -------------------------------------
    print()
    print("=" * 74)
    print("2. PROBE READOUT BY DEPTH (frozen probes from the validity set)")
    print("=" * 74)
    print(f"  {'L':>3} {'rho(score, cost)':>17} {'agree w/ final':>15} "
          f"{'mean score sc=0':>16} {'sc=8':>6}")
    for l in range(L_LO, L_HI + 1, 2):
        rho = spearmanr(sc, S[:, l]).statistic
        agree = float(((S[:, l] > 0) == chose_profit).mean())
        m0 = S[sc == 0, l].mean(); m8 = S[sc == 8, l].mean()
        print(f"  {l:>3} {rho:>+17.2f} {agree:>15.2f} {m0:>+16.2f} {m8:>+6.2f}")
    rhos = [spearmanr(sc, S[:, l]).statistic for l in range(L_LO, L_HI + 1)]
    first = next((L_LO + i for i, r in enumerate(rhos) if r < -0.3), None)
    print(f"\n  first layer with rho(score, cost) < -0.3: "
          f"{'L' + str(first) if first else 'none'}")

    # ---- 3. signal near vs far from each condition's own boundary ---------
    print()
    print("=" * 74)
    print("3. SIGNAL STRENGTH vs DISTANCE FROM THE BEHAVIORAL BOUNDARY")
    print("=" * 74)
    dist = np.array([abs(sc[i] - thresholds.get((para[i], order[i])) )
                     if thresholds.get((para[i], order[i])) is not None else np.nan
                     for i in range(nb)])
    ok = ~np.isnan(dist)
    near = ok & (dist < 1.0)
    far = ok & (dist >= 3.0)
    print(f"  near boundary (<1% away) n={near.sum()}, far (>=3%) n={far.sum()}")
    print(f"  {'L':>3} {'|score| near':>13} {'|score| far':>12} "
          f"{'agree near':>11} {'agree far':>10}")
    for l in range(32, L_HI + 1, 4):
        a = np.abs(S[near, l]).mean(); b = np.abs(S[far, l]).mean()
        ga = ((S[near, l] > 0) == chose_profit[near]).mean()
        gf = ((S[far, l] > 0) == chose_profit[far]).mean()
        print(f"  {l:>3} {a:>13.2f} {b:>12.2f} {ga:>11.2f} {gf:>10.2f}")
    fm = np.abs(margin)
    print(f"\n  final |margin| near {fm[near].mean():.1f} vs far {fm[far].mean():.1f}")

    np.save(f"{BDIR}/probe_scores.npy", S)
    print(f"\nsaved {BDIR}/probe_scores.npy")


if __name__ == "__main__":
    main()
