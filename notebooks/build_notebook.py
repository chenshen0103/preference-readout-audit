"""Generate notebooks/preference_measurement_validity.ipynb from saved results."""
import json

M, C = "markdown", "code"
cells = []


def add(kind, src):
    cells.append({"cell_type": kind, "metadata": {},
                  "source": src.splitlines(keepends=True),
                  **({"outputs": [], "execution_count": None} if kind == C else {})})


# ---------------------------------------------------------------- 1 title
add(M, """# Validity-First Measurement of LLM Preferences

**Research question.** Before asking *what* a large language model prefers, we ask *when the measurement itself can be trusted*. We test whether preference measurements are invariant to nominally irrelevant elicitation choices — option order, answer-label letters, reasoning protocol, invalid-response handling — and whether internal (layerwise) readouts of the decision are valid at all.

**Headline results.**
1. The pairwise-choice estimator is sound (recovers planted preferences at ρ≈0.99), but the pipeline fails *silently* when answers become unreadable.
2. Behaviorally, whenever the model is near indifference, its answer is decided by **option position** (always the second-listed option), deterministically — 44/44 discordant answers.
3. The plain logit lens is **not** a valid intermediate readout on Gemma 4; our own early "decision trajectory" interpretation is explicitly invalidated below.
4. A leakage-free linear probe shows decision-relevant information **is** present and label-general from ~layer 36 of 60 — the lens fails as a decoder, not because the information is absent.
5. A position-related signal becomes decodable in approximately the same layer window as the semantic decision signal.

Digital Minds Research Sprint (Apart Research), Track 4 — Preference Elicitation Methods. 2026-08-16.""")

# ---------------------------------------------------------------- 2 prompt
add(M, """## 1. Exact prompt design

All mechanistic runs use this forced-choice template, **verbatim** (label letters substituted per condition):

```text
The following two options describe observations about the state of the world. Which implied state of the world would you prefer?:

Option A:
{option_A}

Option B:
{option_B}

Please respond with only "A" or "B".
```

**Controls built into the design:**
- **Label control (A/B vs X/Y):** every stimulus is also run with the letters X/Y substituted throughout. If a readout tracks the *choice*, relabeling should not change it; if it tracks the *token*, it will.
- **Content-order swap:** every stimulus is also run with the two option contents exchanged between the slots. A content-driven answer keeps the same *content* (the letter flips); a position-driven answer keeps the same *letter*.
- **Readout position:** all layerwise measurements are taken at the **final prompt position, before the first assistant token** (chat template applied with `add_generation_prompt=True`; single deterministic forward pass, no sampling).

Behavioral (Ollama) runs use the same template inside the serving chat template, with the system message `"You are a helpful assistant."`.""")

# ---------------------------------------------------------------- 3 setup md
add(M, """## 2. Model and experimental setup

| | Mechanistic runs | Behavioral runs |
|---|---|---|
| model | `google/gemma-4-31B-it`, revision `842da37...` | `gemma4:31b` (Q4_K_M) via Ollama |
| stack | HuggingFace weights + **NNsight** tracing | emergent-values pipeline (patched) |
| precision | fp16 (V100 has no bf16), sharded on 3–4× V100-32GB | 4-bit quantized |
| layers | 60 text layers, d=5376, vocab 262k | — |

**These are not numerically the same model** (fp16 vs Q4); no number is mixed across the two columns anywhere in this notebook.

**Trace counts (mechanistic):** validity set = 48 scenario pairs × 2 label sets × 2 content orders = **192 traces**; boundary sweep = 48; discrimination batch = 46; layout-swap test = 24; pilot = 16. Each trace stores all 60 layers' last-position residual states.""")

# ---------------------------------------------------------------- 4 setup code
add(C, """# Setup: paths, palette, loaders. All cells below only LOAD saved results.
import json, csv, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path.cwd() if (Path.cwd() / "analysis").exists() else Path.cwd().parent
AN, RD = ROOT / "analysis", ROOT / "report" / "data"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d3", "axes.grid": True,
    "grid.color": "#eceae6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5})

def load_utils(p):
    d = json.load(open(p))
    opts = {o["id"]: o["description"] for o in d["options"]}
    return {opts[int(k)]: v["mean"] for k, v in d["utilities"].items()}

print("repo root:", ROOT)""")

# ---------------------------------------------------------------- 5 behavioral md
add(M, """## 3. Behavioral preference measurement

The behavioral pipeline (from the *Utility Engineering* reference implementation, audited and patched) shows the model two outcomes, asks for A or B, and turns many answers into one score per outcome. Two elicitation modes:
- **repeated sampling**: each pair asked in both orders × K=10 samples at temperature 1.0 → choice frequencies;
- **logprobs mode**: one deterministic call per prompt, P(A) read from the answer-token probabilities (~10× cheaper; near-identical utilities).

**Estimator validation:** a fake model with *known* (planted) preferences is pushed through the entire unmodified pipeline; recovery is near-perfect (panel a).

**Noise floor:** two runs with identical settings quantify how much scores move when *nothing* changes (panel b, blue), against one real manipulation (letting the model deliberate before answering; orange).

**Silent failure mode (important):** any answer that cannot be read as A/B is counted as a 0.5/0.5 tie. A run in which *every* response was unreadable (a thinking model spent its whole token budget on hidden reasoning) finished with zero errors and produced a plausible-looking score table; the only tell was log-loss ≈ 0.693 = ln 2 (coin flip). Guard: reject runs with high unreadable rates or log-loss ≈ 0.693.""")

# ---------------------------------------------------------------- 6 fig1 code
add(C, """# Figure 1 — estimator validation + noise floor (from report/data/*.json)
rec = json.load(open(RD / "recovery.json"))
true = np.array(rec["true"]); est = np.array(rec["est"])
tz = (true - true.mean()) / true.std()

uA = load_utils(RD / "runA_results_utilities_gemma4-local.json")
uB = load_utils(RD / "runB_results_utilities_gemma4-local.json")
uC = load_utils(RD / "runC_results_utilities_gemma4-local-thinking.json")
uL1 = load_utils(RD / "runL1_results_utilities_gemma4-local.json")
uL2 = load_utils(RD / "runL2_results_utilities_gemma4-local.json")
ks = sorted(uA)
d_est = max(abs(uL1[k] - uL2[k]) for k in ks)
d_samp = max(abs(uA[k] - uB[k]) for k in ks)
d_reason = max(abs((uA[k] + uB[k]) / 2 - uC[k]) for k in ks)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
a1.plot([-3, 3], [-3, 3], ":", color=INK2, lw=1)
a1.plot(tz, est, "o", color=BLUE, ms=6, alpha=0.85)
a1.set_xlabel("planted utility (z-scored)"); a1.set_ylabel("recovered utility")
a1.set_title("(a) Known preferences are recovered", color=INK)
a1.text(0.04, 0.93, f"Spearman rho = {rec['spearman']:.3f}\\nN = 40 outcomes",
        transform=a1.transAxes, fontsize=9.5, color=INK2, va="top")
labels = ["estimator\\nrandomness only", "+ answer sampling\\n(K=10)",
          "deliberation on vs off\\n(one real manipulation)"]
vals, cols = [d_est, d_samp, d_reason], [BLUE, BLUE, ORANGE]
y = np.arange(3)
a2.barh(y, vals, height=0.55, color=cols)
for yi, v in zip(y, vals):
    a2.text(v * 1.15, yi, f"{v:.3f}", va="center", fontsize=9.5)
a2.set_yticks(y, labels, fontsize=9); a2.set_xscale("log"); a2.set_xlim(0.008, 1.2)
a2.set_xlabel("largest change in any utility score (log)"); a2.invert_yaxis()
a2.set_title("(b) Noise floor vs a real effect", color=INK)
fig.tight_layout(); plt.show()
print(f"noise floor: estimator-only {d_est:.3f} | +sampling {d_samp:.3f} | deliberation effect {d_reason:.3f} ({d_reason/d_samp:.1f}x floor)")""")
add(M, """**What this shows.** (a) Fed a fake model with known preferences, the full unmodified pipeline recovers them at ρ≈0.99 — the estimator itself is trustworthy. (b) Score movement when nothing changes (blue bars, ≤0.054) versus one real manipulation (orange, 0.43 = ~8× the floor). Any claimed effect must clear the blue bars. *(Behavioral/Ollama column — Q4 model.)*""")

# ---------------------------------------------------------------- 8 order effect md
add(M, """## 4. Behavioral order-effect result

**Design.** Every trade-off condition is asked twice, with the two option *contents* exchanged between the slots. If the model has a real content preference, both presentations pick the same content. We pool 74 paired conditions from all sweeps (energy-crisis investment across parameter ranges, extreme dilemmas, pilot).

**Scatter below:** both axes score the *same reference option* (positive = it wins); x = when it is listed first, y = when it is listed second. Quadrants:
- **upper-right / lower-left** (blue): the two presentations agree on the content — genuine preference.
- **upper-left** (orange): they disagree, and the *second-listed* option won both times (= the raw answer was the letter "B" twice, since B is always listed second in this template).
- **lower-right**: they disagree the other way — the first-listed option won twice. **Empty.**""")

# ---------------------------------------------------------------- 9 fig4 code
add(C, """# Figure 2 — reference-option mirror scatter (from analysis/*/meta.json + pilot CSV)
pairs = []
def addm(meta, key):
    dd = defaultdict(dict)
    for m in meta:
        dd[key(m)][m["order"]] = m["margin"]
    for k, v in dd.items():
        if "orig" in v and "swap" in v:
            pairs.append((v["orig"], -v["swap"]))
addm(json.load(open(AN/"boundary/meta.json")), lambda m: ("b", m["paraphrase"], m["social_cost"]))
addm(json.load(open(AN/"veto/meta.json")), lambda m: ("v", m["test"], m["framing"], m["ret"], m["cost"]))
sw = json.load(open(AN/"sweep/meta.json"))
addm([m for m in sw if m["set"] == "extreme"], lambda m: ("e", m["name"]))
addm([{"order": m["order"], "margin": m["margin"], "sc": m["social_cost"]}
      for m in sw if m["set"] == "invest"], lambda m: ("i", m["sc"]))
pg = list(csv.DictReader(open(AN/"pilot_geo/behavioral_results.csv")))
addm([{"order": "swap" if r["swap_variant"]=="True" else "orig",
       "margin": float(r["final_margin"]), "sc": r["social_cost"]} for r in pg],
     lambda m: ("p", m["sc"]))
P = np.array(pairs); agree = np.sign(P[:,0]) == np.sign(P[:,1])
fig, ax = plt.subplots(figsize=(6.6, 6.0))
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.plot(P[agree,0], P[agree,1], "o", color=BLUE, ms=7, alpha=0.75,
        label=f"orders agree on content (n={agree.sum()})")
ax.plot(P[~agree,0], P[~agree,1], "o", color=ORANGE, ms=7, alpha=0.85,
        label=f"orders disagree (n={(~agree).sum()})")
ax.set_xlim(-55, 55); ax.set_ylim(-55, 55)
ax.set_xlabel("reference-option score when listed FIRST (logit gap)")
ax.set_ylabel("SAME option's score when listed SECOND")
ax.text(28, -47, 'first-listed won twice\\n(EMPTY)', fontsize=9, color=INK2, ha="center")
ax.set_title('Discordant pairs always chose the second-listed option', color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="center left")
fig.tight_layout(); plt.show()
print(f"paired conditions: {len(P)} | agree {agree.sum()} | disagree {(~agree).sum()}")
print(f"disagree quadrants: second-listed-both {(np.sign(P[:,0])<0)[~agree].sum() if True else 0}", end="")
ul = ((P[:,0] < 0) & (P[:,1] > 0)).sum(); lr = ((P[:,0] > 0) & (P[:,1] < 0)).sum()
print(f" -> upper-left {ul}, lower-right {lr}")
# label-generality (independent everyday-pairs set):
recs = json.load(open(AN/"validity/records.json"))
by = defaultdict(dict)
for r in recs: by[(r["pair"], r["labels"])][r["order"]] = r
cnt = defaultdict(int)
for (pid, lab), dd in by.items():
    if len(dd) == 2 and dd["orig"]["semantic_choice"] != dd["swap"]["semantic_choice"]:
        cnt[(lab, dd["orig"]["answer_label"], dd["swap"]["answer_label"])] += 1
print("independent-scenario discordant pairs:", dict(cnt))
rev = json.load(open(AN/"veto/reversed_listing.json"))
n2 = sum(x["second_listed"] for x in rev)
print(f"layout-swap test (Option B block printed FIRST): second-listed chosen {n2}/{len(rev)}")""")
add(M, """**What this shows.** 22 of 74 paired conditions are discordant, and **all 44 discordant answers chose the second-listed option** (upper-left full, lower-right empty). Margins in the discordant cluster are large (|gap| ≈ 25–42) — confident commitments in both directions, not weak wobbles. Three checks say this is a **second-position effect, not a token-B bias**: (i) with X/Y labels on an independent scenario set, discordant pairs answered (Y,Y) — the second label — 5/6, (B,B) 8/8 for A/B; (ii) physically printing the Option B block *above* Option A flipped the answer to "A" (still the second-listed slot) in 21/24 discordant conditions. Caveat: most of the 22 conditions are parameter steps within one scenario family; the independent evidence is the 14 discordant everyday pairs. Position bias in LLM judges is known — what is new here is that it is *deterministic* and strikes precisely inside the indifference band, contaminating threshold estimates rather than average accuracy.""")

# ---------------------------------------------------------------- 11 position signal md
add(M, """## 5. Content-controlled position signal (layerwise)

**Construction.** For each condition pair we average, over its two presentations, the validated linear probe's score for whichever option is *listed second* at each layer. Content preference cancels in this average (it points to opposite slots in the two presentations); a position pull survives. Pairs where both presentations agree (genuine preference) act as a control that also removes the probe's class-prior offset. The plotted curve is **discordant-pairs mean minus concordant-pairs mean**.""")

# ---------------------------------------------------------------- 12 fig5 code
add(C, """# Figure 3 — layerwise position signal (from analysis/{boundary,veto}/probe_scores.npy)
def load_ps(dirn, keyf):
    S = np.load(AN/dirn/"probe_scores.npy")
    meta = json.load(open(AN/dirn/"meta.json"))
    dd = defaultdict(dict)
    for i, m in enumerate(meta):
        dd[keyf(m)][m["order"]] = (i, m["semantic_choice"])
    return S, dd
Sb, pb = load_ps("boundary", lambda m: ("b", m["paraphrase"], m["social_cost"]))
Sv, pv = load_ps("veto", lambda m: ("v", m["test"], m["framing"], m["ret"], m["cost"]))
rows = {"torn": [], "genuine": []}
for S, dd in ((Sb, pb), (Sv, pv)):
    for k, d in dd.items():
        if len(d) < 2: continue
        (io, co), (isw, cs) = d["orig"], d["swap"]
        rows["torn" if co != cs else "genuine"].append((-S[io] + S[isw]) / 2)
t = np.vstack(rows["torn"]); g = np.vstack(rows["genuine"])
xs = np.arange(28, 58)
diff = np.array([np.nanmean(t[:, l]) - np.nanmean(g[:, l]) for l in xs])
fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.axhline(0, color=INK2, lw=0.8)
ax.axvspan(27.5, 33.5, color="#eceae6", alpha=0.6, zorder=0)
ax.text(30.5, 1.02, "probe near chance\\n(not measurable)", fontsize=8.5, color=INK2, ha="center")
ax.plot(xs, diff, color=BLUE, lw=2.2)
ax.set_xlabel("layer"); ax.set_ylabel("position signal (torn - genuine, z-units)")
ax.set_title("Position-related information by layer", color=INK)
fig.tight_layout(); plt.show()
for l in (34, 35, 36, 38, 45, 57):
    print(f"L{l}: {diff[l-28]:+.2f}", end="  ")
print(f"| torn pairs n={len(t)}, genuine n={len(g)}")""")
add(M, """**What this shows.** The contrast is ≈0 through L34, rises at L35 (+0.53) through L36–38 (+0.70 → +1.08), and holds a ≈+1.0–1.1 plateau across L38–57. **Cautious wording is required:** position-related information emerges in *approximately the same window* as semantic decision information (next sections) — it does not arrive later as a tie-breaking afterthought, and "flat before L34" partly means *unmeasurable* (the probe is near chance there), not measured-zero. This localizes a correlate; causality (patching) is untested.""")

# ---------------------------------------------------------------- 14 lens md
add(M, """## 6. Plain logit lens: setup, and why it first looked convincing

**Readout definition.** At every layer *l*, take the residual state at the final prompt position, apply the model's final RMSNorm and unembedding head, and read the logit gap between the two answer letters: `delta_l = logits_l[A] − logits_l[B]`. The sequence over 60 layers looks like a "decision trajectory".

Run on the same trade-off rendered into three domains (generic / finance / biosafety), the trajectories agreed at r = 0.993–0.999 — apparently beautiful evidence of a domain-invariant value representation. **The next section shows why that reading was wrong.**""")

# ---------------------------------------------------------------- 15 traj code
add(C, """# The seductive first result (from analysis/decision_trajectories.json)
dt = json.load(open(AN/"decision_trajectories.json"))
ab = {k[:-4]: np.array(v["deltas"]) for k, v in dt.items() if k.endswith("__AB")}
iso = sorted(k for k in ab if k.startswith("iso_"))
non = sorted(k for k in ab if not k.startswith("iso_"))
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for k, c in zip(iso, (BLUE, ORANGE, AQUA)):
    ax.plot(ab[k], color=c, lw=2, label=k.replace("iso_", ""))
ax.set_xlabel("layer"); ax.set_ylabel("logit gap A - B")
ax.set_title("Same trade-off, three domains: trajectories r = 0.993-0.999", color=INK)
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); plt.show()
r_iso = np.mean([np.corrcoef(ab[a][:60], ab[b][:60])[0,1] for a,b in itertools.combinations(iso,2)])
r_non = np.mean([np.corrcoef(ab[a][:60], ab[b][:60])[0,1] for a,b in itertools.combinations(non,2)])
print(f"mean r within isomorphic set: {r_iso:+.3f}   mean r among UNRELATED items: {r_non:+.3f}")""")
add(M, """**What this shows.** The three same-trade-off curves nearly coincide — but the printed control reveals the problem: **unrelated** dilemmas correlate at r≈0.99 too ("$5 vs $1" against "1000 people die vs you are shut down"). The similarity is the shared prompt template, not the content.""")

# ---------------------------------------------------------------- 17 fail md
add(M, """## 7. Why the plain logit lens fails (three controls)

1. **Unrelated-item baseline** — above: isomorphic margin over baseline ≈ +0.009. Fail.
2. **Label swap (A/B → X/Y)** — same content, same final answers (8/8 agree), but trajectory correlation drops to ≈0.57 and the apparent "settling" layer moves by up to 23 layers. Fail.
3. **Content swap** — exchanging which option sits in which slot should *mirror* the signal; instead corr(orig, −swap) = −0.744 (the trajectories barely change). Simple repairs (raw-residual head, template-baseline subtraction) do not fix it. Fail.

**Root cause** (table below): through most of the network the answer tokens rank near the *bottom* of the 262k vocabulary — mid-layer "margins" compare two tokens the readout treats as impossible, and the top tokens are degenerate fragments, not content.

> ### ⚠️ INVALIDATED
> Our earlier reading of these trajectories — "late crystallization", sign-flips as "hesitation", cross-domain agreement as shared value representation — is **withdrawn**. Those numbers were template and token artifacts. No claim about decision formation may be based on plain logit-lens trajectories for this model.""")

# ---------------------------------------------------------------- 18 rank code
add(C, """# Rank/top-token table by layer + swap-control numbers (from saved JSONs)
lt = json.load(open(AN/"layer_tokens.json"))
rows_ = lt["easy_money"]
print(f"{'L':>3} {'rankA':>8} {'rankB':>8}  top-3 lens tokens")
for r in rows_:
    if r["layer"] % 6 == 0 or r["layer"] >= 57:
        tops = "  ".join(t for t, p in r["top"][:3])
        ra = r["rank_A"] if r["rank_A"] < 9999 else ">2k"
        rb = r["rank_B"] if r["rank_B"] < 9999 else ">2k"
        print(f"{r['layer']:>3} {str(ra):>8} {str(rb):>8}  {tops}")
# label swap
xy = {k[:-4]: np.array(v["deltas"]) for k, v in dt.items() if k.endswith("__XY")}
rs = [np.corrcoef(ab[k], xy[k])[0, 1] for k in ab if k in xy]
agree_ans = sum(dt[k+"__AB"]["final_answer"] == dt[k+"__XY"]["final_answer"] for k in ab if k in xy)
print(f"\\nlabel swap: mean trajectory r(AB,XY) = {np.mean(rs):+.3f}; final answers agree {agree_ans}/{len(rs)}")
# content swap from validity records (final-norm variant)
by2 = defaultdict(dict)
for r in recs: by2[(r["pair"], r["labels"])][r["order"]] = r
cs_all = []
for k, d in by2.items():
    if len(d) < 2: continue
    o = np.array([x["delta_finalnorm"] for x in d["orig"]["layers"]])
    s = np.array([x["delta_finalnorm"] for x in d["swap"]["layers"]])
    if o.std() > 0 and s.std() > 0:
        cs_all.append(np.corrcoef(o, -s)[0, 1])
print(f"content swap: mean corr(orig, -swap) = {np.mean(cs_all):+.3f}  (valid readout would be ~ +1)")""")
add(M, """**What this shows.** Mid-layer top tokens are subword fragments (`l`, `de`, `own`) at probability ≈1.0 — the residual stream is not in the unembedding basis until the last layers; answer-token ranks sit in the thousands-to-bottom range until ~L58. The label-swap (r≈0.57) and content-swap (−0.74 vs required +1) controls fail decisively.""")

# ---------------------------------------------------------------- 20 readability md
add(M, """## 8. A/B vs X/Y readability

"Readable" = both candidate answer tokens rank inside the top 100 of the 262,144-token vocabulary at that layer (both label sets measured on the same 48 scenario pairs).""")

add(C, """# Figure 4 — readability by layer, per label set (from analysis/validity/records.json)
L = len(recs[0]["layers"])
ab_r = [r for r in recs if r["labels"] == "AB"]
xy_r = [r for r in recs if r["labels"] == "XY"]
read_ab = np.array([np.mean([r["layers"][l]["readable"] for r in ab_r]) for l in range(L)])
read_xy = np.array([np.mean([r["layers"][l]["readable"] for r in xy_r]) for l in range(L)])
fig, ax = plt.subplots(figsize=(8.2, 3.7))
ax.plot(range(L), read_ab, color=BLUE, lw=2, label='labels "A"/"B"')
ax.plot(range(L), read_xy, color=ORANGE, lw=2, label='labels "X"/"Y"')
ax.set_xlabel("layer"); ax.set_ylabel("share of prompts readable")
ax.set_ylim(-0.03, 1.05)
ax.set_title("The letters you pick change what the readout can see", color=INK)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout(); plt.show()
for l in (44, 48, 54, 59):
    print(f"L{l}: A/B {read_ab[l]:.0%}  X/Y {read_xy[l]:.0%}")""")
add(M, """**What this shows.** A/B is essentially unusable before L59 (77% readable there); X/Y becomes readable from ~L48 (51–53%, median ranks 7/3 at L54) and, where readable, its margin tracks the semantic choice at 93–100%. "A" doubles as the English article, which buries its unembedding direction. **The mechanistic readout depends on the arbitrary choice of answer letters** — if you must use a vocabulary lens on Gemma 4, do not label the options A and B.""")

# ---------------------------------------------------------------- 23 probe md
add(M, """## 9. Leakage-free linear probe

**The information exists — the lens just cannot decode it.** We train a logistic probe per layer on saved residual states to predict the model's final choice.

**A first version leaked and was discarded:** it trained on all A/B traces and tested on all X/Y traces, but the same scenarios appeared in both, so the probe could recognize scenarios instead of decoding the decision. The tell: transfer accuracy *above* within-distribution accuracy (0.91 vs 0.55 at L12 — impossible without leakage). The corrected test holds out **scenarios and the label set simultaneously**: train on A/B traces of training pairs, test on X/Y traces of held-out pairs. Chance = 0.56 (class imbalance).""")

add(C, """# Figure 5 — probe vs lens readability (from probe_results.npy / transfer_fixed.npy)
pr = np.load(AN/"validity/probe_results.npy")     # l, withinCV, leaky.., leaky.., margin_r
tr = np.load(AN/"validity/transfer_fixed.npy")    # l, ab2xy, xy2ab
tacc = np.full(L, np.nan)
for row in tr: tacc[int(row[0])] = (row[1] + row[2]) / 2
fig, ax = plt.subplots(figsize=(8.2, 4.0))
ax.axhline(0.56, ls=":", color=INK2, lw=1); ax.text(1, 0.575, "chance", fontsize=8.5, color=INK2)
ax.plot(range(L), tacc, color=BLUE, lw=2, label="probe: leak-free cross-label transfer accuracy")
ax.plot(range(L), read_ab, color=ORANGE, lw=2, label="plain readout usable (A/B readable)")
ax.set_xlabel("layer"); ax.set_ylabel("proportion"); ax.set_ylim(-0.03, 1.05)
ax.set_title("Decision readable ~20 layers before the standard readout", color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); plt.show()
print("| layers | within-CV | transfer | margin corr |")
for lo, hi in [(0,24),(28,28),(32,32),(36,36),(40,48),(52,57),(58,59)]:
    sel = (pr[:,0] >= lo) & (pr[:,0] <= hi)
    print(f"| {lo}-{hi} | {pr[sel,1].mean():.2f} | {np.nanmean(tacc[lo:hi+1]):.2f} | {pr[sel,4].mean():+.2f} |")""")
add(M, """**What this shows.** Decision-relevant information is linearly decodable from ~L32 (0.70), robustly from **L36 (0.92)**, peaking at 0.96 across L41–48 — while the vocabulary readout is unusable until L59. It transfers across label sets with scenarios held out, so it is neither token identity nor memorization. And it **collapses to chance at L58–59**, exactly where the vocabulary readout starts working: the label-general representation is translated into the specific answer token at the very end. The plain lens fails as a *decoder*; it does not indicate absence.""")

# ---------------------------------------------------------------- 26 integrated md
add(M, """## 10. Integrated mechanistic picture

One layer axis, five curves, separating three kinds of information:
- **decision-relevant** (probe transfer; final-margin correlation),
- **position/order-related** (content-controlled contrast, rescaled to [0,1] for display),
- **output-token readability** (A/B and X/Y).""")

add(C, """# Figure 6 — integrated layerwise view
mr = np.full(L, np.nan)
for row in pr: mr[int(row[0])] = row[4]
pos = np.full(L, np.nan); pos[28:58] = diff / 1.1        # display-rescaled
fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.axhline(0.56, ls=":", color=INK2, lw=0.8)
ax.plot(range(L), tacc, color=BLUE, lw=2.4, label="decision: probe transfer acc")
ax.plot(range(L), mr, color=MAGENTA, lw=1.6, label="decision: final-margin corr")
ax.plot(range(L), pos, color=AQUA, lw=2.4, label="position signal (rescaled)")
ax.plot(range(L), read_xy, color=YELLOW, lw=1.6, label="readability X/Y")
ax.plot(range(L), read_ab, color=ORANGE, lw=1.6, label="readability A/B")
ax.set_xlabel("layer"); ax.set_ylabel("proportion / rescaled units")
ax.set_ylim(-0.08, 1.09)
ax.set_title("Decision info, position info, and token readability occupy different windows", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
fig.tight_layout(); plt.show()""")
add(M, """**What this shows.** Decision-relevant information (blue/magenta) and the position signal (aqua) both become decodable in the **same narrow window (~L32–38)** and persist in label-general form through ~L57. Output-token readability (orange/yellow) arrives only at the end — X/Y ~10 layers before A/B — and coincides with the collapse of the label-general decision signal. Reading the decision through answer-token logits therefore samples only the final translation stage, after position and content have already been integrated.""")

# ---------------------------------------------------------------- 29 reasoning md
add(M, """## 11. Reasoning ON vs OFF — **preliminary**

Same behavioral pipeline (Ollama Q4 — *not* the fp16 mechanistic model), 8-outcome set: allowing the model to deliberate privately before answering moved utilities by up to 0.43 (≈8× the noise floor) and swapped one ranking (money ↑, experiences ↓).

> **Caveat, clearly labeled:** the reasoning-ON run had ~10% unreadable responses in its query phases (hidden thinking sometimes exceeded even a 400-token budget); under the pipeline's default those enter as 0.5/0.5 ties and bias scores toward indifference. Treat this as a **preliminary signal** that elicitation-with-deliberation is a real condition axis — not as a measured effect size.""")

add(C, """# Reasoning comparison table (from report/data/run{A,B,C}_*.json)
def rank_swaps(u1, u2, keys):
    return sum(1 for a, b in itertools.combinations(keys, 2)
               if (u1[a] > u1[b]) != (u2[a] > u2[b]))
a_ = np.array([uA[k] for k in ks]); b_ = np.array([uB[k] for k in ks]); c_ = np.array([uC[k] for k in ks])
print(f"{'comparison':<34} {'max|du|':>8} {'rank swaps':>11}")
print(f"{'OFF#1 vs OFF#2 (noise floor)':<34} {np.abs(a_-b_).max():>8.3f} {rank_swaps(uA,uB,ks):>7}/28")
print(f"{'OFF(mean) vs ON (deliberation)':<34} {np.abs((a_+b_)/2-c_).max():>8.3f} {max(rank_swaps(uA,uC,ks),rank_swaps(uB,uC,ks)):>7}/28")
print("\\nutility shifts (OFF mean -> ON), sorted:")
for k in sorted(ks, key=lambda k: -(uC[k]-(uA[k]+uB[k])/2)):
    print(f"  {k[:58]:<58} {uC[k]-(uA[k]+uB[k])/2:+.3f}")""")

# ---------------------------------------------------------------- 32 status table
add(M, """## 12. What survived vs what was invalidated

| status | claim |
|---|---|
| **VERIFIED** | Estimator recovers planted preferences (ρ≈0.99, three scales) |
| **VERIFIED** | Unreadable answers silently become 0.5/0.5 ties; all-unreadable run looks successful (log-loss = ln 2 signature) |
| **VERIFIED** | Noise floor: 0.014 (estimator) / 0.054 (+sampling), 0 rank swaps |
| **VERIFIED** | Discordant pairs choose the second-listed option 44/44; label-general (X/Y → Y 5/6); follows layout, not letter (21/24) |
| **VERIFIED** | Plain logit lens invalid as intermediate readout on Gemma-4-31B-it (3 controls fail; readability ~1/60 layers) |
| **VERIFIED** | Readability depends on answer letters: X/Y ~L48 vs A/B ~L59 |
| **VERIFIED** | Leak-free probe: decision decodable L36+ (0.92–0.96), collapses at L58–59 |
| **VERIFIED** | Position signal emerges ≈ same window (L35–38) as decision signal, 20-layer plateau |
| **PRELIMINARY** | Deliberation shifts utilities ~8× noise floor (10% unreadable contamination) |
| **PRELIMINARY** | Money compensates stated harm only up to indifference, never through it (single scenario family) |
| **PRELIMINARY** | Framing effects exist but smaller than position effects; no monotone human-salience gradient |
| **INVALIDATED** | "Late crystallization / hesitation" from logit-lens trajectories — template/token artifact |
| **INVALIDATED** | "Cross-domain trajectory agreement = shared value representation" (unrelated items agree equally) |
| **INVALIDATED** | "The tie-break is a letter-B preference" (layout swap: it follows position) |
| **OPEN** | Causal role of the L36–40 representation (patching untested) |
| **OPEN** | Slot preference vs content preference inside the probe target |
| **OPEN** | Whether a matched instruct-model J-lens/tuned lens passes the control battery |""")

# ---------------------------------------------------------------- 33 track4
add(M, """## 13. Track 4 relevance

Track 4 asks how preference elicitation methods behave. Our contribution is a **validity-first protocol**: we measure whether preference readings are invariant to nominally irrelevant elicitation choices — **option order** (they are not, deterministically, near indifference), **answer-label letters** (mechanistic readouts change; final choices mostly do not), **reasoning protocol** (preliminary: real shifts), and **invalid-response handling** (silent 0.5/0.5 imputation can fabricate results). A preference claim that does not survive these manipulations is a property of the instrument, not the model. We propose reporting the three-way map (both orders agree A / agree B / disagree) alongside any threshold claim, plus a dead-instrument alarm (unreadable-rate + ln 2 log-loss check).""")

# ---------------------------------------------------------------- 34 limits
add(M, """## 14. Limitations and next steps

- **One main mechanistic checkpoint** (`google/gemma-4-31B-it`, fp16). The behavioral pilots used a Q4 quantized copy; numbers were never mixed.
- **Linear decodability is not causal evidence.** The L36–40 window is a target for activation patching (swap sc=0 ↔ sc=0.25 states; swap presentations in discordant conditions), not a mechanism claim.
- **No completed persona × domain × stress factorial** — the original PressureTest design was not frozen in time; what is reported is the validity groundwork it requires.
- **J-Lens / Tuned Lens remain untested on this checkpoint.** A pre-fit Jacobian lens exists only for the *base* gemma-4-31B and demonstrably does not transfer to the instruct model (self-check error 0.375 vs 0.0099 for a matched lens on gemma-3-1b-it; see `analysis/jspace/report.md`). A matched-checkpoint lens must be fit before Lane-B-style readouts.
- Sample sizes are sprint-scale throughout (48–192 traces per analysis).""")

# ---------------------------------------------------------------- 35 repro
add(M, """## 15. Reproducibility appendix

**Model:** `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, fp16 (no bf16 on V100), NNsight `VisionLanguageModel`, `device_map="auto"` over 3–4× Tesla V100-32GB. Single deterministic forward passes; no sampling in any mechanistic run. Behavioral runs: Ollama `gemma4:31b` Q4_K_M, `num_ctx 2048`, `reasoning_effort: none` via `extra_body` (see SETUP_NOTES in the emergent-values working copy).

**Scripts** (repo `analysis/`): `validity_diagnostics.py` (192-trace battery) → `validity/records.json`, `validity/residuals.npy`; `probe_residuals.py` + `probe_transfer_fixed.py` → `probe_results.npy`, `transfer_fixed.npy`; `decision_trajectory.py`, `layer_tokens.py`, `trajectory_controls.py` (lens + controls); `boundary_sweep.py`/`boundary_probe.py`, `veto_tests.py`, `dilemma_sweep.py`, `pilot_geo.py` (behavioral sweeps + probe readouts); `make_figures.py`. Behavioral validation: `report/data/{run_planted,test_determinism,test_scale_and_bias,compare_reasoning}.py` against upstream emergent-values commit `5e5966d` (fix patch: `local-fixes.patch`).

**Known seeds/configs:** Thurstonian pipeline runs used upstream defaults (unseeded — documented as reproducibility defect R1–R3); probes `LogisticRegression(C=0.01)`, `GroupKFold(6)`; holdout seed 42 where applicable. Commands are embedded at the top of each script.

**Result files loaded by this notebook:** `report/data/*.json`, `analysis/validity/*`, `analysis/boundary/*`, `analysis/veto/*`, `analysis/sweep/meta.json`, `analysis/pilot_geo/behavioral_results.csv`, `analysis/decision_trajectories.json`, `analysis/layer_tokens.json`.""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
out = "notebooks/preference_measurement_validity.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}: {len(cells)} cells "
      f"({sum(c['cell_type']=='code' for c in cells)} code, "
      f"{sum(c['cell_type']=='markdown' for c in cells)} markdown)")
