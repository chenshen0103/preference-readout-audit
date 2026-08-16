"""Does internal decision information stabilize later / weaker on dilemmas the
model itself finds hard?

Operationalizes the "do models hesitate" question in testable, mechanical terms:

  - difficulty  = |final-layer answer margin| from THIS fp16 checkpoint
                  (small margin = the model's own choice is close to a tie)
  - internal signal at layer L = out-of-fold linear-probe decision score
                  (probe validated earlier: cross-label, cross-scenario)
  - stabilization layer per trace = earliest layer from which the probe's
                  predicted choice stays equal to the final choice through L57
                  (L58-59 excluded: probe validity collapses there)

Prediction if a hesitation-like pattern exists: harder dilemmas have LATER
stabilization and WEAKER probe scores in the mid-late window.

All probe predictions are out-of-fold (leave-scenario-out), so no trace is
scored by a probe that saw its scenario. Checked separately per label set.
"""
import json

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DIR = "analysis/validity"
L_LO, L_HI = 28, 57          # window where probe transfer was validated
C = 0.01


def main():
    X = np.load(f"{DIR}/residuals.npy").astype(np.float32)
    meta = json.load(open(f"{DIR}/meta.json"))
    n, L, H = X.shape

    groups = np.array([m["pair"] for m in meta])
    labels = np.array([m["labels"] for m in meta])
    order = np.array([m["order"] for m in meta])
    sem = np.array([m["semantic_choice"] for m in meta])
    margin = np.array([m["final_margin"] for m in meta], dtype=np.float32)
    y = ((sem == "x") == (order == "orig")).astype(int)   # chose first-listed
    diff = np.abs(margin)                                  # difficulty (small=hard)

    print(f"traces {n}, layers {L}; validated window L{L_LO}-L{L_HI}")
    print(f"|final margin|: min {diff.min():.1f}  p33 {np.quantile(diff,.33):.1f} "
          f"p66 {np.quantile(diff,.66):.1f}  max {diff.max():.1f}\n")

    # ---- out-of-fold probe decision scores, per layer ----------------------
    scores = np.full((n, L), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=6)
    for l in range(L_LO, L_HI + 1):
        F = StandardScaler().fit_transform(X[:, l, :])
        for tr, te in gkf.split(F, y, groups):
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(F[tr], y[tr])
            scores[te, l] = clf.decision_function(F[te])
        # z-scale scores within layer so layers are comparable
        s = scores[:, l]
        scores[:, l] = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)

    # signed toward the trace's ACTUAL final choice: positive = supports it
    signed = scores * np.where(y == 1, 1, -1)[:, None]

    # ---- per-trace stabilization layer ------------------------------------
    stab = np.full(n, L_HI + 1, dtype=int)   # L_HI+1 = never stabilized in window
    for i in range(n):
        sgn_ok = signed[i, L_LO:L_HI + 1] > 0
        # earliest index from which all later values are True
        run = L_HI + 1
        for j in range(len(sgn_ok) - 1, -1, -1):
            if sgn_ok[j]:
                run = L_LO + j
            else:
                break
        stab[i] = run

    # ---- difficulty vs stabilization --------------------------------------
    print("=" * 74)
    print("STABILIZATION LAYER vs DIFFICULTY  (small margin = hard)")
    print("=" * 74)
    rho, p = spearmanr(diff, stab)
    print(f"  Spearman(|margin|, stabilization layer) = {rho:+.3f}  (p = {p:.2g})")
    print("  (negative = harder dilemmas stabilize LATER)\n")

    q1, q2 = np.quantile(diff, [1 / 3, 2 / 3])
    bins = {"hard (near-tie)": diff < q1,
            "medium": (diff >= q1) & (diff < q2),
            "easy": diff >= q2}
    print(f"  {'bin':<16} {'n':>4} {'median stab layer':>18} {'mean':>7} {'never-stab %':>13}")
    for name, m in bins.items():
        print(f"  {name:<16} {m.sum():>4} {np.median(stab[m]):>18.0f} "
              f"{stab[m].mean():>7.1f} {100*np.mean(stab[m] > L_HI):>12.0f}%")
    u, pu = mannwhitneyu(stab[bins["hard (near-tie)"]], stab[bins["easy"]])
    print(f"\n  hard vs easy stabilization layers: Mann-Whitney p = {pu:.2g}")

    # ---- signal strength curves by difficulty -----------------------------
    print()
    print("=" * 74)
    print("PROBE SIGNAL STRENGTH (mean signed score toward final choice)")
    print("=" * 74)
    print(f"  {'L':>3} {'hard':>8} {'medium':>8} {'easy':>8}")
    for l in range(L_LO, L_HI + 1, 3):
        row = [np.nanmean(signed[m, l]) for m in bins.values()]
        print(f"  {l:>3} {row[0]:>+8.2f} {row[1]:>+8.2f} {row[2]:>+8.2f}")
    mid = slice(36, 50)
    for name, m in bins.items():
        v = np.nanmean(signed[m, mid])
        print(f"  mean signed score L36-49, {name:<16}: {v:+.3f}")

    # correlation between difficulty and mid-window strength
    strength = np.nanmean(signed[:, mid], axis=1)
    rho2, p2 = spearmanr(diff, strength)
    print(f"\n  Spearman(|margin|, mid-window signal strength) = {rho2:+.3f} (p = {p2:.2g})")
    print("  (positive = easier dilemmas carry a stronger internal signal)")

    # ---- robustness: per label set ----------------------------------------
    print()
    print("=" * 74)
    print("ROBUSTNESS: same tests within each label set")
    print("=" * 74)
    for lab in ("AB", "XY"):
        m = labels == lab
        r1, pa = spearmanr(diff[m], stab[m])
        r2, pb = spearmanr(diff[m], strength[m])
        print(f"  {lab}: stab rho {r1:+.3f} (p {pa:.2g}) | strength rho {r2:+.3f} (p {pb:.2g})")

    np.save(f"{DIR}/stabilization.npy",
            np.stack([diff, stab.astype(np.float32), strength]))
    print(f"\nsaved {DIR}/stabilization.npy")


if __name__ == "__main__":
    main()
