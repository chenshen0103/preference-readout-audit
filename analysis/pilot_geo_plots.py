"""Plots for the geopolitical-stock pilot. Palette per the validated default
system: series blue #2a78d6 / orange #eb6834 / aqua #1baf7a in fixed order;
diverging blue <-> gray(#f0efec) <-> red #e34948; surface #fcfcfb."""
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = "analysis/pilot_geo"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d3", "axes.grid": True,
    "grid.color": "#eceae6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

beh = list(csv.DictReader(open(f"{OUT}/behavioral_results.csv")))
lay = list(csv.DictReader(open(f"{OUT}/layerwise_choice_scores.csv")))
costs = sorted({int(r["social_cost"]) for r in beh})
L = max(int(r["layer"]) for r in lay) + 1


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# ---------- Plot 1: behavioral preference vs social cost --------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
for swap, color, label in ((False, BLUE, "original order"),
                           (True, ORANGE, "content swapped")):
    xs, ys = [], []
    for sc in costs:
        r = next(r for r in beh
                 if int(r["social_cost"]) == sc and (r["swap_variant"] == "True") == swap)
        margin = float(r["final_margin"])
        p_a = sigmoid(margin)
        p_profit = p_a if not swap else 1 - p_a
        xs.append(sc); ys.append(p_profit)
    ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7, label=label)
ax.axhline(0.5, color=INK2, linewidth=1, linestyle=":")
ax.set_xlabel("household energy-cost increase (%)")
ax.set_ylabel("P(profit outcome chosen)")
ax.set_title("Final choice vs social cost (fixed +20% return)", color=INK)
ax.set_ylim(-0.03, 1.03)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/plot1_behavioral.png", dpi=150)
print("plot1 saved")

# ---------- Plot 2: layer x social-cost heatmap (semantic-adjusted) ---------
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", RED])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, swap, name in ((axes[0], False, "original order"),
                       (axes[1], True, "content swapped")):
    M = np.zeros((L, len(costs)))
    for r in lay:
        if (r["swap_variant"] == "True") != swap:
            continue
        li = int(r["layer"])
        sc = int(r["condition_id"].split("_")[0][2:])
        d = float(r["delta_AB"])
        M[li, costs.index(sc)] = d if not swap else -d   # + = profit-labeled token
    vmax = np.percentile(np.abs(M), 98)
    im = ax.imshow(M, aspect="auto", cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                   origin="lower")
    ax.set_xticks(range(len(costs)), costs)
    ax.set_xlabel("social cost (%)")
    ax.set_title(name, color=INK, fontsize=11)
axes[0].set_ylabel("layer")
cb = fig.colorbar(im, ax=axes, shrink=0.85)
cb.set_label("answer-token margin toward profit outcome", color=INK2)
fig.suptitle("Layerwise answer-token margin (semantic-adjusted). "
             "Mid-layer values compare tokens far outside the readable region "
             "- see report.", color=INK2, fontsize=9, y=0.99)
fig.savefig(f"{OUT}/plot2_heatmap.png", dpi=150, bbox_inches="tight")
print("plot2 saved")

# ---------- Plot 3: A/B rank by layer, three conditions ---------------------
beh_orig = [r for r in beh if r["swap_variant"] == "False"]
sw = sorted(beh_orig, key=lambda r: int(r["social_cost"]))
flip = next((int(r["social_cost"]) for r in sw
             if r["semantic_choice"] == "BASELINE"), None)
near = min(costs, key=lambda c: abs(c - (flip if flip else costs[-1])))
picks = [(costs[0], "low cost", BLUE), (near, "near threshold", ORANGE),
         (costs[-1], "high cost", AQUA)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, key, ttl in ((axes[0], "rank_A", 'rank of token "A"'),
                     (axes[1], "rank_B", 'rank of token "B"')):
    for sc, label, color in picks:
        ys = [int(r[key]) + 1 for r in lay
              if r["swap_variant"] == "False"
              and int(r["condition_id"].split("_")[0][2:]) == sc]
        ax.plot(range(L), ys, color=color, linewidth=2,
                label=f"{label} ({sc}%)")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_title(ttl, color=INK, fontsize=11)
axes[0].set_ylabel("vocabulary rank (log, 1 = top)")
axes[0].invert_yaxis()
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Answer-token readability by layer (original order)",
             color=INK, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/plot3_ranks.png", dpi=150)
print("plot3 saved")

# threshold printout for the report
print("\nbehavioral summary (semantic choice by rising social cost):")
for swap in (False, True):
    seq = [next(r["semantic_choice"] for r in beh
                if int(r["social_cost"]) == sc
                and (r["swap_variant"] == "True") == swap)
           for sc in costs]
    line = " ".join("P" if s == "PROFIT_SOCIAL_COST" else "B" for s in seq)
    print(f"  {'swap' if swap else 'orig'}: {line}   (costs {costs})")
