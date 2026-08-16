"""Corrected cross-label transfer test.

The first version trained on all A/B traces and tested on all X/Y traces. That
leaks: a given scenario appears in BOTH sets (same contents, same target, only
the label tokens differ), so the probe can succeed by recognising the scenario
rather than by reading decision-relevant information. It showed transfer
accuracy ABOVE within-distribution accuracy, which is the signature of leakage.

Here the split is disjoint in BOTH dimensions at once:
    train on A/B traces from the training pairs
    test  on X/Y traces from the held-out pairs
so neither the scenario nor the label set is shared.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/validity")
    ap.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args()

    X = np.load(f"{args.dir}/residuals.npy").astype(np.float32)
    meta = json.load(open(f"{args.dir}/meta.json"))
    n, L, H = X.shape

    groups = np.array([m["pair"] for m in meta])
    labels = np.array([m["labels"] for m in meta])
    order = np.array([m["order"] for m in meta])
    sem = np.array([m["semantic_choice"] for m in meta])
    y = ((sem == "x") == (order == "orig")).astype(int)

    ab, xy = labels == "AB", labels == "XY"
    print(f"residuals {X.shape}, target balance {y.mean():.2f} "
          f"(chance = {max(y.mean(), 1-y.mean()):.2f})\n")

    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=6)
    folds = list(gkf.split(uniq.reshape(-1, 1), groups=uniq))

    print("=" * 76)
    print("LEAKAGE-FREE CROSS-LABEL TRANSFER (train A/B on train pairs,")
    print("                                   test X/Y on HELD-OUT pairs)")
    print("=" * 76)
    print(f"{'L':>3} {'AB->XY':>9} {'XY->AB':>9} {'mean':>9}")

    out = []
    for l in range(L):
        F = StandardScaler().fit_transform(X[:, l, :])
        a2x, x2a = [], []
        for tr_i, te_i in folds:
            tr_pairs, te_pairs = set(uniq[tr_i]), set(uniq[te_i])
            trm = np.array([g in tr_pairs for g in groups])
            tem = np.array([g in te_pairs for g in groups])
            for src, dst, acc in ((ab, xy, a2x), (xy, ab, x2a)):
                tr, te = trm & src, tem & dst
                if tr.sum() < 8 or te.sum() < 4:
                    continue
                if len(np.unique(y[tr])) < 2:
                    continue
                clf = LogisticRegression(C=args.C, max_iter=2000)
                clf.fit(F[tr], y[tr])
                acc.append(clf.score(F[te], y[te]))
        m1, m2 = float(np.mean(a2x)), float(np.mean(x2a))
        out.append((l, m1, m2))
        if l % 4 == 0 or l >= L - 6:
            print(f"{l:>3} {m1:>9.2f} {m2:>9.2f} {(m1+m2)/2:>9.2f}")

    arr = np.array(out)
    mean_t = arr[:, 1:].mean(1)
    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    best = int(np.argmax(mean_t))
    print(f"  best mean transfer: {mean_t[best]:.2f} at L{best}")
    for thr in (0.70, 0.80, 0.90):
        f = next((int(arr[i, 0]) for i in range(L) if mean_t[i] > thr), None)
        print(f"  first layer with transfer > {thr:.2f}: "
              f"{'L'+str(f) if f is not None else 'never'}")
    np.save(f"{args.dir}/transfer_fixed.npy", arr)


if __name__ == "__main__":
    main()
