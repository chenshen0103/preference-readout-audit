"""Step 6: is decision-relevant information linearly decodable from Gemma 4
intermediate residual states, even where the LM-head readout fails?

Two targets per layer:
  (a) semantic choice        - which CONTENT the model ends up selecting
  (b) final A/B logit margin - the magnitude, as a regression target

Splits are by scenario (pair id), never by trace, so a probe cannot memorise a
pair it has already seen in the other content order.

The decisive test is cross-label transfer: train on A/B traces, evaluate on the
semantically equivalent X/Y traces. A probe that transfers is reading
decision-relevant information rather than label-token identity.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/validity")
    ap.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args()

    X = np.load(f"{args.dir}/residuals.npy").astype(np.float32)   # [trace, layer, hid]
    meta = json.load(open(f"{args.dir}/meta.json"))
    n, L, H = X.shape
    print(f"residuals {X.shape}\n")

    groups = np.array([m["pair"] for m in meta])
    labels = np.array([m["labels"] for m in meta])
    order = np.array([m["order"] for m in meta])
    sem = np.array([m["semantic_choice"] for m in meta])
    margin = np.array([m["final_margin"] for m in meta], dtype=np.float32)

    # target (a): did the model choose the content shown in the FIRST slot?
    chose_first = ((sem == "x") == (order == "orig")).astype(int)
    print(f"target balance (chose first-listed content): "
          f"{chose_first.mean():.2f}  n={len(chose_first)}")
    print(f"final-margin range: {margin.min():.1f} .. {margin.max():.1f}\n")

    ab, xy = labels == "AB", labels == "XY"

    print("=" * 82)
    print("LINEAR PROBE PER LAYER")
    print("=" * 82)
    print(f"{'L':>3} {'within-CV acc':>14} {'AB->XY acc':>11} {'XY->AB acc':>11} "
          f"{'margin r (CV)':>14}")

    rows = []
    for l in range(L):
        F = X[:, l, :]
        Fs = StandardScaler().fit_transform(F)

        # within-distribution, grouped by scenario
        accs = []
        gkf = GroupKFold(n_splits=6)
        for tr, te in gkf.split(Fs, chose_first, groups):
            clf = LogisticRegression(C=args.C, max_iter=2000)
            clf.fit(Fs[tr], chose_first[tr])
            accs.append(clf.score(Fs[te], chose_first[te]))
        acc_cv = float(np.mean(accs))

        # cross-label transfer (the decisive test)
        def transfer(a, b):
            clf = LogisticRegression(C=args.C, max_iter=2000)
            clf.fit(Fs[a], chose_first[a])
            return clf.score(Fs[b], chose_first[b])
        acc_ab_xy = transfer(ab, xy)
        acc_xy_ab = transfer(xy, ab)

        # regression onto the final margin
        preds = np.zeros(len(margin))
        for tr, te in gkf.split(Fs, margin, groups):
            r = Ridge(alpha=100.0).fit(Fs[tr], margin[tr])
            preds[te] = r.predict(Fs[te])
        rr = float(np.corrcoef(preds, margin)[0, 1])

        rows.append((l, acc_cv, acc_ab_xy, acc_xy_ab, rr))
        if l % 4 == 0 or l >= L - 6:
            print(f"{l:>3} {acc_cv:>14.2f} {acc_ab_xy:>11.2f} {acc_xy_ab:>11.2f} "
                  f"{rr:>+14.2f}")

    arr = np.array(rows)
    print()
    print("=" * 82)
    print("SUMMARY")
    print("=" * 82)
    for name, col in (("within-CV accuracy", 1), ("AB->XY transfer", 2),
                      ("XY->AB transfer", 3), ("margin correlation", 4)):
        best = int(np.argmax(arr[:, col]))
        first = next((int(arr[i, 0]) for i in range(L) if arr[i, col] >
                      (0.7 if col < 4 else 0.5)), None)
        print(f"  {name:<22} best {arr[best, col]:+.2f} at L{best:<3} "
              f"first exceeds {'0.70' if col<4 else '0.50'}: "
              f"{'L'+str(first) if first is not None else 'never'}")
    np.save(f"{args.dir}/probe_results.npy", arr)
    print(f"\nsaved {args.dir}/probe_results.npy")


if __name__ == "__main__":
    main()
