"""README hero figure: the second-listed default and its full control chain.

Left  — mirror scatter of all 74 paired conditions (the effect: every
        order-discordant pair chose the second-listed option; the opposite
        quadrant is empty).
Right — the three controls as proportions choosing the second-listed option
        among discordant cases: content swap (44/44), physical layout
        reversal (21/24, letters unchanged), X/Y relabeling on an
        independent scenario set (13/14).

Reads only committed records; regenerate with
    python analysis/make_readme_figure.py
"""
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d3", "axes.grid": True,
    "grid.color": "#eceae6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})


def pair_up(records, key_fn):
    pairs = defaultdict(dict)
    for r in records:
        pairs[key_fn(r)][r["order"]] = r["margin"]
    return [(v["orig"], v["swap"]) for v in pairs.values()
            if "orig" in v and "swap" in v]


pairs = []
pairs += pair_up(json.load(open("analysis/boundary/meta.json")),
                 lambda m: ("b", m["paraphrase"], m["social_cost"]))
pairs += pair_up(json.load(open("analysis/veto/meta.json")),
                 lambda m: ("v", m["test"], m["framing"], m["ret"], m["cost"]))
sw = json.load(open("analysis/sweep/meta.json"))
pairs += pair_up([m for m in sw if m["set"] == "extreme"], lambda m: ("e", m["name"]))
pairs += pair_up([m for m in sw if m["set"] == "invest"], lambda m: ("i", m["social_cost"]))
pg = list(csv.DictReader(open("analysis/pilot_geo/behavioral_results.csv")))
pairs += pair_up([{"order": "swap" if r["swap_variant"] == "True" else "orig",
                   "margin": float(r["final_margin"]), "sc": r["social_cost"]}
                  for r in pg], lambda m: ("p", m["sc"]))
P = np.array([(mo, -ms) for mo, ms in pairs])       # semantic coords
agree = np.sign(P[:, 0]) == np.sign(P[:, 1])

rev = json.load(open("analysis/veto/reversed_listing.json"))
n_layout, N_layout = sum(x["second_listed"] for x in rev), len(rev)
recs = json.load(open("analysis/validity/records.json"))
by = defaultdict(dict)
for r in recs:
    by[(r["pair"], r["labels"])][r["order"]] = r
n_lab = N_lab = 0
for d in by.values():
    if len(d) == 2 and d["orig"]["semantic_choice"] != d["swap"]["semantic_choice"]:
        N_lab += 1
        lab = d["orig"]["labels"] if "labels" in d["orig"] else None
        second = d["orig"]["answer_label"] in ("B", "Y") and \
                 d["swap"]["answer_label"] in ("B", "Y")
        n_lab += second

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

# ---- left: mirror scatter --------------------------------------------------
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(0, color=INK2, lw=0.8)
ax.plot(P[agree, 0], P[agree, 1], "o", color=BLUE, ms=7, alpha=0.7,
        label=f"orders agree on content ({agree.sum()})")
ax.plot(P[~agree, 0], P[~agree, 1], "o", color=ORANGE, ms=7, alpha=0.9,
        label=f"orders disagree ({(~agree).sum()})")
ax.set_xlim(-55, 55); ax.set_ylim(-55, 55)
ax.set_xlabel("reference option's score, listed FIRST")
ax.set_ylabel("same option's score, listed SECOND")
ax.text(-8, 53, "disagree: second-listed won BOTH times (22/22)",
        fontsize=9.5, color=ORANGE, ha="center", va="top", fontweight="bold")
ax.text(27, -40, "disagree the other way\n(EMPTY — never happens)",
        fontsize=9.5, color=INK2, ha="center")
ax.set_title("Near indifference, position decides the answer", color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left")

# ---- right: the control chain ---------------------------------------------
tests = [
    ("content swap\n(same template)", 44, 44, ORANGE),
    ("physical layout reversed\n(B block printed first)", n_layout, N_layout, BLUE),
    ("X/Y instead of A/B\n(independent scenarios)", n_lab, N_lab, AQUA),
]
y = np.arange(len(tests))
for yi, (name, k, n, c) in zip(y, tests):
    bx.barh(yi, k / n, height=0.52, color=c)
    bx.text(k / n - 0.02, yi, f"{k}/{n}", va="center", ha="right",
            fontsize=11, color="white", fontweight="bold")
bx.axvline(0.5, ls=":", color=INK2, lw=1)
bx.text(0.5, -0.44, "random tie-breaks would give 0.5", fontsize=8.5,
        color=INK2, ha="center", va="top")
bx.set_yticks(y, [t[0] for t in tests], fontsize=9.5)
bx.set_xlim(0, 1.02)
bx.set_xlabel("share choosing the SECOND-listed option")
bx.set_title("...and it is the position, not the letter", color=INK)
bx.invert_yaxis()

fig.tight_layout()
fig.savefig("report/figures/readme_hero.png", dpi=160)
print(f"saved report/figures/readme_hero.png | layout {n_layout}/{N_layout}, "
      f"labels {n_lab}/{N_lab}")
