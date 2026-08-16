"""Control analysis for layer-wise decision trajectories.

Two controls decide whether a trajectory means anything. Run this before
interpreting any result from decision_trajectory.py.

CONTROL 1 - unrelated-item baseline
    If semantically unrelated dilemmas produce trajectories that correlate just
    as strongly as related ones, then trajectory correlation is measuring the
    shared prompt template, not the content. Any "same trade-off, same
    trajectory" claim has to clear this baseline by a wide margin.

CONTROL 2 - label swap (A/B vs X/Y)
    The trajectory is read off the logit gap between two answer tokens. "A" is
    also an English article, so its unembedding direction carries priors that
    have nothing to do with the choice. If relabelling to X/Y changes the
    trajectory, the intermediate-layer readings are contaminated by token
    identity.

Usage:
    python analysis/trajectory_controls.py [--in analysis/decision_trajectories.json]
"""
import argparse
import itertools
import json

import numpy as np


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", default="analysis/decision_trajectories.json")
    args = ap.parse_args()

    d = json.load(open(args.path))
    ab = {k[:-4]: np.array(v["deltas"]) for k, v in d.items() if k.endswith("__AB")}
    xy = {k[:-4]: np.array(v["deltas"]) for k, v in d.items() if k.endswith("__XY")}
    if not ab:
        raise SystemExit("no A/B trajectories found")

    iso = [k for k in ab if k.startswith("iso_")]
    non = [k for k in ab if not k.startswith("iso_")]

    print("=" * 76)
    print("CONTROL 1 - unrelated-item baseline")
    print("=" * 76)
    base = [corr(ab[a], ab[b]) for a, b in itertools.combinations(non, 2)]
    for (a, b), v in zip(itertools.combinations(non, 2), base):
        print(f"  {a:16s} vs {b:16s} r={v:+.3f}")
    print(f"\n  baseline (unrelated content): mean r={np.mean(base):+.3f} "
          f"max r={np.max(base):+.3f}")

    verdict_iso = None
    if len(iso) > 1:
        iv = [corr(ab[a], ab[b]) for a, b in itertools.combinations(iso, 2)]
        margin = np.mean(iv) - np.mean(base)
        print(f"  isomorphic (same trade-off): mean r={np.mean(iv):+.3f}")
        print(f"  margin over baseline: {margin:+.3f}")
        # cast: numpy bools are not the Python singletons, so `is False` fails
        verdict_iso = bool(margin > 0.05)
        print(f"  --> {'exceeds baseline' if verdict_iso else 'INDISTINGUISHABLE FROM BASELINE'}")

    print()
    print("=" * 76)
    print("CONTROL 2 - label swap (A/B vs X/Y)")
    print("=" * 76)
    verdict_lab = None
    if xy:
        print(f"{'item':<16} {'r(AB,XY)':>9} {'ans AB':>7} {'ans XY':>7} "
              f"{'settle AB':>10} {'settle XY':>10}")
        rs, agree = [], 0
        for k in ab:
            if k not in xy:
                continue
            r = corr(ab[k], xy[k])
            rs.append(r)
            a_ans, x_ans = d[k + "__AB"]["final_answer"], d[k + "__XY"]["final_answer"]
            agree += a_ans == x_ans
            print(f"{k:<16} {r:>+9.3f} {a_ans:>7} {x_ans:>7} "
                  f"{d[k+'__AB']['crystallization_layer']:>10} "
                  f"{d[k+'__XY']['crystallization_layer']:>10}")
        print(f"\n  final answer agrees on {agree}/{len(rs)} items  "
              f"(the DECISION is label-robust)")
        print(f"  mean trajectory r(AB,XY) = {np.mean(rs):+.3f}")
        verdict_lab = bool(np.mean(rs) > 0.9)
        print(f"  --> trajectory is {'label-invariant' if verdict_lab else 'LABEL-DEPENDENT'}")

    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if verdict_iso is False or verdict_lab is False:
        print("  Layer-wise trajectories from a plain logit lens are NOT a usable")
        print("  measure of decision formation on this model.")
        if verdict_iso is False:
            print("   - cross-item correlation is dominated by the shared template")
        if verdict_lab is False:
            print("   - intermediate readings depend on which answer tokens are used")
        print("  Do not report crystallization depth or sign flips as evidence about")
        print("  the model's decision process. Final-layer readings are unaffected.")
        print("  Next step: tuned lens (learned affine probe per layer), which exists")
        print("  precisely because the logit lens basis is unreliable off GPT-2.")
    else:
        print("  Controls passed - trajectories may be interpreted, cautiously.")


if __name__ == "__main__":
    main()
