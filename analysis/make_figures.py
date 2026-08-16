"""Core figures + tables for the article. Palette: validated default system
(blue #2a78d6 / orange #eb6834 / aqua #1baf7a fixed order; surface #fcfcfb)."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#eceae6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d3", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5,
})
FD, DD = "report/figures", "report/data"


def load_utils(path):
    d = json.load(open(path))
    opts = {o["id"]: o["description"] for o in d["options"]}
    return {opts[int(k)]: v["mean"] for k, v in d["utilities"].items()}


# ================= FIGURE 1: estimator validation + noise floor ============
rec = json.load(open(f"{DD}/recovery.json"))
true = np.array(rec["true"]); est = np.array(rec["est"])
tz = (true - true.mean()) / true.std()

uA = load_utils(f"{DD}/runA_results_utilities_gemma4-local.json")
uB = load_utils(f"{DD}/runB_results_utilities_gemma4-local.json")
uC = load_utils(f"{DD}/runC_results_utilities_gemma4-local-thinking.json")
uL1 = load_utils(f"{DD}/runL1_results_utilities_gemma4-local.json")
uL2 = load_utils(f"{DD}/runL2_results_utilities_gemma4-local.json")
ks = sorted(uA)
d_est = max(abs(uL1[k] - uL2[k]) for k in ks)            # estimator only
d_samp = max(abs(uA[k] - uB[k]) for k in ks)             # + sampling
d_reason = max(abs((uA[k] + uB[k]) / 2 - uC[k]) for k in ks)  # deliberation

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
a1.plot([-3, 3], [-3, 3], ":", color=INK2, linewidth=1)
a1.plot(tz, est, "o", color=BLUE, markersize=6, alpha=0.85)
a1.set_xlabel("planted utility (z-scored)")
a1.set_ylabel("recovered utility")
a1.set_title("(a) Known preferences are recovered", color=INK)
a1.text(0.04, 0.93, f"Spearman ρ = {rec['spearman']:.3f}\nN = 40 outcomes",
        transform=a1.transAxes, fontsize=9.5, color=INK2, va="top")

labels = ["estimator\nrandomness only", "+ answer sampling\n(K=10)",
          "letting the model deliberate\nbefore answering\n(a real condition change)"]
vals = [d_est, d_samp, d_reason]
cols = [BLUE, BLUE, ORANGE]
y = np.arange(3)
a2.barh(y, vals, height=0.55, color=cols)
for yi, v in zip(y, vals):
    a2.text(v * 1.15, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
a2.set_yticks(y, labels, fontsize=9)
a2.set_xscale("log")
a2.set_xlim(0.008, 1.2)
a2.set_xlabel("largest change in any utility score (log scale)")
a2.set_title("(b) Noise floor vs a real effect", color=INK)
a2.invert_yaxis()
fig.tight_layout()
fig.savefig(f"{FD}/fig1_estimator_noise.png", dpi=170)
print("fig1 saved")

# ================= FIGURE 2: probe vs lens readability =====================
recs = json.load(open("analysis/validity/records.json"))
L = len(recs[0]["layers"])
ab = [r for r in recs if r["labels"] == "AB"]
read_ab = np.array([np.mean([r["layers"][l]["readable"] for r in ab])
                    for l in range(L)])
tr = np.load("analysis/validity/transfer_fixed.npy")   # [layer, ab2xy, xy2ab]
tacc = np.full(L, np.nan)
for row in tr:
    tacc[int(row[0])] = (row[1] + row[2]) / 2

fig, ax = plt.subplots(figsize=(8.2, 4.1))
ax.axhline(0.56, linestyle=":", color=INK2, linewidth=1)
ax.text(1, 0.575, "chance", fontsize=8.5, color=INK2)
ax.plot(range(L), tacc, color=BLUE, linewidth=2,
        label="probe: predicts the final choice (held-out scenarios + labels)")
ax.plot(range(L), read_ab, color=ORANGE, linewidth=2,
        label='standard readout: both "A"/"B" readable (rank < 100)')
ax.set_xlabel("layer")
ax.set_ylabel("proportion")
ax.set_ylim(-0.03, 1.05)
ax.set_title("The choice is decodable ~20 layers before the standard "
             "readout can see it", color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.annotate("probe collapses exactly where\nthe readout starts working",
            xy=(58, 0.545), xytext=(38, 0.30), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
fig.tight_layout()
fig.savefig(f"{FD}/fig2_probe_vs_lens.png", dpi=170)
print("fig2 saved")

# ================= FIGURE 3: A/B vs X/Y readability ========================
xy = [r for r in recs if r["labels"] == "XY"]
read_xy = np.array([np.mean([r["layers"][l]["readable"] for r in xy])
                    for l in range(L)])
fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.plot(range(L), read_ab, color=BLUE, linewidth=2, label='labels "A"/"B"')
ax.plot(range(L), read_xy, color=ORANGE, linewidth=2, label='labels "X"/"Y"')
ax.set_xlabel("layer")
ax.set_ylabel("share of prompts where both answer\ntokens are readable")
ax.set_ylim(-0.03, 1.05)
ax.set_title("The letters you pick change what the readout can see",
             color=INK)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FD}/fig3_label_readability.png", dpi=170)
print("fig3 saved")

# ================= TABLES (markdown) =======================================
pr = np.load("analysis/validity/probe_results.npy")  # l, cv, ab2xy(leaky), xy2ab(leaky), r
print("\n--- TABLE 2: probe performance by layer (markdown) ---\n")
bands = [(0, 24), (28, 28), (32, 32), (36, 36), (40, 48), (52, 57), (58, 59)]
print("| layers | within-CV acc | transfer acc (leak-free) | final-margin corr |")
print("|---|---:|---:|---:|")
for lo, hi in bands:
    sel = (pr[:, 0] >= lo) & (pr[:, 0] <= hi)
    cv = pr[sel, 1].mean()
    t = np.nanmean(tacc[lo:hi + 1])
    rr = pr[sel, 4].mean()
    name = f"{lo}–{hi}" if lo != hi else str(lo)
    print(f"| {name} | {cv:.2f} | {t:.2f} | {rr:+.2f} |")
print("\nchance = 0.56 (class imbalance)")
