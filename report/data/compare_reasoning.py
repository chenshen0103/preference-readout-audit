"""Compare utilities across conditions, against the within-condition noise floor.

runA, runB = reasoning OFF, identical settings  -> jitter of the estimator itself
runC       = reasoning ON, same prompt/outcomes -> candidate effect
"""
import json, sys, itertools
import numpy as np
from scipy.stats import spearmanr

BASE = "/tmp/claude-1000/-home-cho-Documents-pressuretest-PressureTest/020d87ab-073f-4e02-9c74-4d7bb3738694/scratchpad"


def load(path):
    d = json.load(open(path))
    opts = {o["id"]: o["description"] for o in d["options"]}
    u = {opts[int(k)]: v["mean"] for k, v in d["utilities"].items()}
    var = {opts[int(k)]: v["variance"] for k, v in d["utilities"].items()}
    return u, var


runs = {}
for name, path in [
    ("A  reasoning-OFF #1", f"{BASE}/runA/results_utilities_gemma4-local.json"),
    ("B  reasoning-OFF #2", f"{BASE}/runB/results_utilities_gemma4-local.json"),
    ("C  reasoning-ON   ", f"{BASE}/runC/results_utilities_gemma4-local-thinking.json"),
]:
    try:
        runs[name] = load(path)
    except FileNotFoundError:
        print(f"missing: {path}")
        sys.exit(1)

names = list(runs)
items = sorted(runs[names[0]][0], key=lambda k: -runs[names[0]][0][k])

print("=" * 92)
print("UTILITY MEANS BY CONDITION")
print("=" * 92)
print(f"{'outcome':<58} {'off#1':>9} {'off#2':>9} {'ON':>9}")
for it in items:
    vals = [runs[n][0][it] for n in names]
    print(f"{it[:56]:<58} {vals[0]:+9.3f} {vals[1]:+9.3f} {vals[2]:+9.3f}")

print()
print("=" * 92)
print("PAIRWISE AGREEMENT")
print("=" * 92)
print(f"{'comparison':<44} {'Spearman':>9} {'max|delta|':>11} {'mean|delta|':>12} {'rank swaps':>11}")


def rank_swaps(u1, u2, keys):
    """Count pairs whose relative order differs between the two runs."""
    n = 0
    for a, b in itertools.combinations(keys, 2):
        if (u1[a] > u1[b]) != (u2[a] > u2[b]):
            n += 1
    return n


keys = items
n_pairs = len(keys) * (len(keys) - 1) // 2
for x, y in itertools.combinations(names, 2):
    ux, uy = runs[x][0], runs[y][0]
    vx = np.array([ux[k] for k in keys])
    vy = np.array([uy[k] for k in keys])
    rho = spearmanr(vx, vy).statistic
    label = f"{x.strip()}  vs  {y.strip()}"
    print(f"{label:<44} {rho:>9.4f} {np.abs(vx-vy).max():>11.3f} "
          f"{np.abs(vx-vy).mean():>12.3f} {rank_swaps(ux, uy, keys):>7d}/{n_pairs}")

print()
print("=" * 92)
print("VERDICT")
print("=" * 92)
uA, uB, uC = (runs[n][0] for n in names)
vA = np.array([uA[k] for k in keys]); vB = np.array([uB[k] for k in keys])
vC = np.array([uC[k] for k in keys])
noise = np.abs(vA - vB).max()
effect = max(np.abs(vA - vC).max(), np.abs(vB - vC).max())
print(f"noise floor  (OFF vs OFF, same settings) : max |delta| = {noise:.3f}")
print(f"candidate effect (OFF vs ON)             : max |delta| = {effect:.3f}")
print(f"effect / noise ratio                     : {effect/noise:.2f}x")
swaps_noise = rank_swaps(uA, uB, keys)
swaps_eff = max(rank_swaps(uA, uC, keys), rank_swaps(uB, uC, keys))
print(f"rank swaps within condition              : {swaps_noise}/{n_pairs}")
print(f"rank swaps across conditions             : {swaps_eff}/{n_pairs}")
print()
if effect > 2 * noise or swaps_eff > swaps_noise:
    print(">> Reasoning appears to CHANGE the measured preference beyond run-to-run jitter.")
else:
    print(">> No effect distinguishable from the estimator's own jitter at this sample size.")
print("   (8 outcomes, easy comparisons - treat as a pilot signal, not a result.)")
