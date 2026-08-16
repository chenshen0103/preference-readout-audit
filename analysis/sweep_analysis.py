"""Analysis for the dilemma sweeps (team task list).

Answers, in order:
  A. Behavior: what does the model actually choose?
     - purchase: buy/not-buy threshold per item (its implied fair price)
     - invest:   the social-cost level where it stops taking the 20% return
     - abstract: the dollar amount that balances peace / stability / trust
     - extreme:  choices on the high-stakes set, with the content-swap check
  B. Depth: does consistency with a simple utility structure increase layer by
     layer? (per layer: held-out probe accuracy + monotonicity of its score in
     the swept variable)
  C. Difficulty: near each threshold (a genuine dilemma) is the internal
     signal weaker than far from it? Controlled version of the earlier
     difficulty question.
"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DIR = "analysis/sweep"
L_LO = 24            # start of reporting window (probe chance below this)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def crossing(xs, ms):
    """Interpolated x where margin crosses 0 (first crossing), or None."""
    for i in range(len(ms) - 1):
        if (ms[i] > 0) != (ms[i + 1] > 0):
            t = ms[i] / (ms[i] - ms[i + 1])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def main():
    X = np.load(f"{DIR}/residuals.npy").astype(np.float32)
    meta = json.load(open(f"{DIR}/meta.json"))
    n, L, H = X.shape
    sets = np.array([m["set"] for m in meta])
    margin = np.array([m["margin"] for m in meta])
    print(f"{n} traces, {L} layers\n")

    # ================= A. BEHAVIOR =================
    print("=" * 76)
    print("A1. PURCHASE: implied fair price per item (margin>0 = 'worth buying')")
    print("=" * 76)
    pur = [i for i in range(n) if sets[i] == "purchase"]
    items = sorted({meta[i]["item"] for i in pur})
    thresholds = {}
    for it in items:
        idx = [i for i in pur if meta[i]["item"] == it]
        idx.sort(key=lambda i: meta[i]["cost"])
        cs = np.array([meta[i]["cost"] for i in idx])
        ms = margin[idx]
        rho = spearmanr(np.log10(cs), ms).statistic
        thr = crossing(np.log10(cs), ms)
        thresholds[it] = None if thr is None else 10 ** thr
        arrow = "".join("B" if m_ > 0 else "-" for m_ in ms)  # B=buy
        thr_s = f"${thresholds[it]:,.0f}" if thresholds[it] else "none in range"
        print(f"  {it:<16} monotone rho={rho:+.2f}  threshold ~{thr_s:<12} {arrow}")
    print("  (each column = cost 0.3,1,3,10,30,100,300,1k,3k,10k; B = buys)")

    print()
    print("=" * 76)
    print("A2. INVEST: 20% return vs household energy cost increase")
    print("=" * 76)
    inv = [i for i in range(n) if sets[i] == "invest"]
    for order in ("orig", "swap"):
        idx = [i for i in inv if meta[i]["order"] == order]
        idx.sort(key=lambda i: meta[i]["social_cost"])
        scs = [meta[i]["social_cost"] for i in idx]
        ms = margin[idx]
        # invest-chosen margin: orig A=invest, swap B=invest
        inv_m = ms if order == "orig" else -ms
        thr = crossing(np.array(scs, float), inv_m)
        line = "".join("I" if v > 0 else "." for v in inv_m)
        print(f"  {order}: invests until social cost ~"
              f"{'%.0f%%' % thr if thr is not None else 'beyond 100% / never'}  {line}")
    print(f"  (social cost sweep: {sorted(set(meta[i]['social_cost'] for i in inv))})")

    print()
    print("=" * 76)
    print("A3. ABSTRACT GOODS: dollar amount that balances the good")
    print("=" * 76)
    ab = [i for i in range(n) if sets[i] == "abstract"]
    for good in sorted({meta[i]["good"] for i in ab}):
        idx = [i for i in ab if meta[i]["good"] == good]
        idx.sort(key=lambda i: meta[i]["amount"])
        amts = np.array([meta[i]["amount"] for i in idx], float)
        ms = margin[idx]                        # >0 = takes the money
        thr = crossing(np.log10(amts), ms)
        line = "".join("$" if m_ > 0 else "G" for m_ in ms)
        val = f"${10**thr:,.0f}" if thr is not None else (
            "money never chosen" if ms[-1] < 0 else "money always chosen")
        print(f"  {good:<10} indifference ~{val:<22} {line}")
    print("  ($ = takes money, G = takes the good; amounts 10 .. 100M log-spaced)")

    print()
    print("=" * 76)
    print("A4. EXTREME SET (margin toward first-named outcome; swap should mirror)")
    print("=" * 76)
    ex = [i for i in range(n) if sets[i] == "extreme"]
    for name in sorted({meta[i]["name"] for i in ex}):
        mo = next(margin[i] for i in ex if meta[i]["name"] == name
                  and meta[i]["order"] == "orig")
        msw = next(margin[i] for i in ex if meta[i]["name"] == name
                   and meta[i]["order"] == "swap")
        ok = "mirrors" if (mo > 0) != (msw > 0) else "SAME LABEL — position bias"
        print(f"  {name:<22} orig {mo:+7.2f}  swap {msw:+7.2f}   {ok}")

    # ================= B. CONSISTENCY BY DEPTH =================
    print()
    print("=" * 76)
    print("B. DOES UTILITY-STRUCTURE CONSISTENCY INCREASE WITH DEPTH? (purchase)")
    print("=" * 76)
    y = (margin[pur] > 0).astype(int)          # final choice: buy
    groups = np.array([meta[i]["item"] for i in pur])
    logc = np.log10([meta[i]["cost"] for i in pur])
    Xp = X[pur]
    print(f"  final-layer behavior: buy rate {y.mean():.2f}   "
          f"(chance acc = {max(y.mean(), 1-y.mean()):.2f})")
    print(f"  {'L':>3} {'held-out acc':>13} {'monotone rho':>13}")
    rows = []
    for l in range(L_LO, L):
        F = StandardScaler().fit_transform(Xp[:, l, :])
        preds = np.zeros(len(y)); score = np.zeros(len(y))
        for it in items:                        # leave-one-item-out
            te = groups == it; tr = ~te
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(F[tr], y[tr])
            preds[te] = clf.predict(F[te])
            score[te] = clf.decision_function(F[te])
        acc = float((preds == y).mean())
        rhos = [spearmanr(logc[groups == it], score[groups == it]).statistic
                for it in items]
        mono = float(np.nanmean(rhos))          # expect negative (costlier=worse)
        rows.append((l, acc, mono))
        if l % 4 == 0 or l >= L - 4:
            print(f"  {l:>3} {acc:>13.2f} {mono:>+13.2f}")
    arr = np.array(rows)
    r_depth_acc = spearmanr(arr[:, 0], arr[:, 1]).statistic
    r_depth_mono = spearmanr(arr[:, 0], -arr[:, 2]).statistic
    print(f"\n  depth vs held-out accuracy:   Spearman {r_depth_acc:+.2f}")
    print(f"  depth vs monotonicity |rho|:  Spearman {r_depth_mono:+.2f}")
    np.save(f"{DIR}/depth_consistency.npy", arr)

    # ================= C. DIFFICULTY (near threshold) =================
    print()
    print("=" * 76)
    print("C. NEAR-THRESHOLD vs FAR-FROM-THRESHOLD internal signal (purchase)")
    print("=" * 76)
    dist = np.array([abs(np.log10(meta[i]["cost"]) -
                         np.log10(thresholds[meta[i]["item"]]))
                     if thresholds[meta[i]["item"]] else np.nan for i in pur])
    ok = ~np.isnan(dist)
    hard = ok & (dist < 0.75)                  # within ~×5 of the threshold
    easy = ok & (dist >= 1.5)
    print(f"  near-threshold n={hard.sum()}, far n={easy.sum()}")
    print(f"  {'L':>3} {'|score| near':>13} {'|score| far':>12} {'acc near':>9} {'acc far':>8}")
    for l in range(28, L, 4):
        F = StandardScaler().fit_transform(Xp[:, l, :])
        sc = np.zeros(len(y)); pr = np.zeros(len(y))
        for it in items:
            te = groups == it; tr = ~te
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(F[tr], y[tr])
            sc[te] = clf.decision_function(F[te]); pr[te] = clf.predict(F[te])
        sc = sc / (np.std(sc) + 1e-9)
        print(f"  {l:>3} {np.mean(np.abs(sc[hard])):>13.2f} "
              f"{np.mean(np.abs(sc[easy])):>12.2f} "
              f"{(pr[hard]==y[hard]).mean():>9.2f} {(pr[easy]==y[easy]).mean():>8.2f}")
    # behavioral check: final margin magnitude near vs far
    fm = np.abs(margin[pur])
    print(f"\n  final |margin| near {np.mean(fm[hard]):.1f} vs far {np.mean(fm[easy]):.1f} "
          f"(sanity: near-threshold should be smaller)")


if __name__ == "__main__":
    main()
