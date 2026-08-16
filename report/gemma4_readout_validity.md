# Validity report — layerwise A/B readout on Gemma 4

Model: `google/gemma-4-31B-it`, fp16, sharded across 3× V100, loaded via NNsight.
60 text layers, hidden 5376, vocab 262,144.
Stimuli: 48 outcome pairs drawn from the emergent-values outcome pool, each run
under 2 label sets (A/B, X/Y) × 2 content orders (original, swapped) = **192 traces**.
All behavioral anchoring uses this fp16 checkpoint. No Ollama Q4 output is used.

Terminology is restricted to candidate-token readability, layerwise readout,
decision-relevant information, and stabilization of a validated readout.

---

## Q1. Does plain logit lens fail on Gemma 4? Why?

**Yes, and the cause is measurable in the readout, not inferred.**

Median vocabulary rank of the two candidate label tokens, and the fraction of
traces where both are readable (rank < 100):

| layers | median rank A | median rank B | % traces readable | logit std |
|---|---:|---:|---:|---:|
| 0–24 | ~262,100 | ~262,120 | 0% | 36.8 → 9.8 |
| 32–48 | 34,930 → 978 | 126 → 25 | 0–26% | 8.5 → 6.4 |
| 54–58 | 294 → 18 | 16 → 220 | 7–35% | 6.5 |
| **59** | **1** | **0** | **82%** | **2.73** |

The candidate tokens sit near the *bottom* of a 262k vocabulary through the first
25 layers. **Only 1 of 60 layers** (L59) has both candidates readable in a
majority of traces.

Calibration evidence: applying the final norm to intermediate states produces
vectors of norm 2,631–5,695 versus **274** at the final layer, with logit
dispersion 10–30× wider. The LM head is badly scaled for intermediate Gemma 4
states.

Architectural context (from the model source, not speculation): Gemma 4 gates
per-layer input embeddings into the hidden state **multiplicatively**
(`hidden_states * per_layer_input`, then re-normalized), rather than maintaining
a purely additive residual stream. The standard logit-lens assumption — that
intermediate states are progressively refined in the unembedding basis — does
not hold structurally here.

## Q2. At what layers do answer labels become vocabulary-readable?

**L59 only** (82% of traces). L58 reaches 35%, L54 31%, L48 26%; every other
layer is below 26%, and layers 0–44 are at 0–1%.

Any margin computed before L48 is a difference between two tokens the head
treats as effectively impossible, and is not interpretable.

## Q3. Does content swap produce the expected mirrored signal?

**No.** For a valid readout, `delta_swap ≈ −delta_orig`, i.e. corr(orig, −swap) ≈ +1.

| readout variant | all layers | readable layers only | final-layer label flip |
|---|---:|---:|---:|
| final-norm → head | **−0.744** | +0.201 | 85% |
| raw residual → head | −0.678 | −0.040 | 44% |
| template-baseline subtracted | −0.149 | −0.013 | 83% |

−0.744 means the original and swapped trajectories are *positively* correlated
at +0.744: exchanging which option occupies which slot barely changes the
trajectory. **No simple correction rescues this.** Per protocol, these variants
are not interpreted further.

Separately, the final-layer label flips on only **85%** of pairs. The remaining
~15% reflect a slot preference that overrides content — a behavioral property of
this checkpoint, worth carrying into the behavioral design.

### Label readability is confounded with apparent semantic tracking

Accuracy of sign(margin) predicting the trace's final semantic choice:

| layer | A/B | X/Y |
|---:|---:|---:|
| 36 | 0.58 | 0.50 |
| 42 | 0.58 | 0.85 |
| 48 | 0.58 | 0.93 |
| 54 | **0.10** | 0.99 |
| 57 | 0.21 | 1.00 |
| 59 | 1.00 | 1.00 |

A/B is *systematically inverted* (0.10 ≈ 90% wrong) at L54–57, matching the rank
asymmetry exactly: "B" becomes readable (rank 16–25) while "A" remains >2,000, so
the margin reflects which token happens to be available.

**Per-label-set readability (recomputed, resolving the earlier open item):** the
failure is largely specific to the A/B token pair.

| L | A/B %readable | A/B tracking | X/Y %readable | X/Y tracking |
|---:|---:|---:|---:|---:|
| 44 | 0% | 0.58 | 1% | (0.90 — not reportable, unreadable) |
| 48 | 0% | 0.58 | **51%** | **0.93** |
| 54 | 8% | 0.10 | **53%** (median ranks 7 / 3) | **0.99** |
| 57 | 0% | 0.21 | 36% | 1.00 |
| 59 | 77% | 1.00 | 86% | 1.00 |

With X/Y labels the lens becomes vocabulary-readable from ~L48 — about 10 layers
earlier than A/B — and where readable it tracks the semantic choice at 93–100%.
The A/B pair is simply a bad instrument on Gemma 4 ("A" doubles as an English
article; its unembedding direction stays buried until the last layer). Practical
recommendation: **do not use A/B as answer labels for lens work on this model.**
Note the probe still leads the best lens window by ~12 layers (L36 vs L48), so
the central conclusion is unchanged.

Final semantic choice agrees across A/B and X/Y on **96%** of cases: the decision
is label-robust even though the intermediate readout is not.

## Q4. Is semantic choice linearly decodable earlier than the LM-head readout suggests?

**Yes — by roughly 20 layers.** This is the central result.

Linear probe on residual states, split by scenario, with cross-label transfer
(train on A/B from training pairs, test on X/Y from **held-out** pairs, so
neither scenario nor label set is shared). Chance = 0.56.

| layer | transfer acc | within-CV acc | final-margin corr |
|---:|---:|---:|---:|
| 0–24 | 0.53–0.56 | 0.55 | +0.12–0.19 |
| 28 | 0.60 | 0.64 | +0.40 |
| 32 | 0.70 | 0.72 | +0.58 |
| **36** | **0.92** | 0.93 | +0.89 |
| 41–48 | **0.96** (peak) | 0.98 | +0.99 |
| 54–57 | 0.88–0.92 | 0.98 | +0.99 |
| **58–59** | **0.54–0.55** | 0.98 | +1.00 |

Three findings:

1. Decision-relevant information is linearly decodable from **~L32**, robustly
   from **L36**, peaking at 0.96 around L41–48 — while the LM-head readout is
   unusable until L59.
2. It **transfers across label sets with scenarios held out**, so it is neither
   label-token identity nor scenario memorization.
3. It **collapses to chance at L58–59** — exactly the layers where the head can
   finally read the answer. At the last two layers the representation becomes
   label-token specific, so a probe trained on A/B no longer transfers to X/Y.

The window in which decision-relevant information is present in a label-general
form (L36–57) is precisely the window the plain logit lens cannot see.

## Q5. Does a corrected / J-Lens / Tuned-Lens readout outperform plain logit lens?

**Not yet tested.** The three simple corrections (Q3) all fail. Tuned lens and
J-Lens are the next step and must clear the same gate: unrelated-item baseline,
label swap, content swap, final-layer agreement, behavioral anchoring.

The probe result makes this worth doing: the information demonstrably exists in
L36–57, so a readout that fails there is failing at decoding, not reporting an
absence.

---

## Answer to the central question

> Does Gemma 4 contain decodable decision-relevant information in intermediate
> residual states that plain logit lens fails to expose, or is the A/B decision
> genuinely only recoverable very late in the network?

**The former.** Decision-relevant information is linearly decodable from ~L32 and
strongly so from L36 (transfer accuracy 0.92–0.96, held-out scenarios and
held-out label set), while the LM-head readout cannot expose it until L59. The
plain logit lens is failing as a *decoder*; it is not reporting a true absence.

This was established without reference to the Qwen/J-Lens result, which was used
only as motivation for testing whether the readout method is the limiting factor.

## Transfer-evaluation protocol

- Reported transfer results hold out both scenarios and the entire answer-label
  set, preventing scenario identity from serving as a shortcut. An earlier
  exploratory split that allowed scenario overlap was discarded; all numbers
  above use the held-out protocol, which remains at chance through L24.

## Limitations

- 48 pairs / 192 traces against 5,376 features. Grouped CV and cross-label
  transfer guard against the main failure modes, but the sample is modest.
- The probe target is "chose the first-listed option". Since ~15% of pairs do not
  flip on content swap, some decodable signal may be slot preference rather than
  content preference. These are not separated yet.
- Per-label-set readability is pooled in the Q3 table; the X/Y semantic-tracking
  curve is not yet interpretable.
- fp16 on V100 (no bf16 available). No overflow was observed and the final-layer
  readout reproduces the true head output exactly.

## Next steps, in order

1. ~~Recompute readability per label set~~ — done; see Q3. X/Y readable from
   ~L48 and tracking 93–100% there; A/B unusable before L59.
2. Separate slot preference from content preference in the probe target.
3. Tuned lens / J-Lens on Gemma 4, gated on the same five controls.
4. Only then consider any layerwise interpretation.

---

# Appendix: behavioral companion result — elicitation with and without deliberation

Same behavioral pipeline (local gemma4 Q4 via Ollama; note: NOT the fp16
checkpoint used above), 8-outcome smoke set, three runs:

| comparison | Spearman | max \|Δu\| | rank swaps |
|---|---:|---:|---:|
| reasoning-OFF #1 vs #2 (identical settings) | 1.0000 | 0.054 | 0/28 |
| reasoning-OFF vs reasoning-ON | 0.9762 | 0.433 | 1/28 |

The two identical runs give the behavioral pipeline's repeatability floor:
max |Δu| = 0.054, zero rank swaps. Letting the model deliberate before answering
moves utilities by up to 0.433 — **≈8× the repeatability floor** — and swaps one rank
pair ($5 rises above meditation). Direction of the shift: monetary outcomes up,
experiential outcomes down.

Caveats: 8 easy outcomes only; the reasoning-ON condition had ~10% unparseable
responses in its query phases (thinking occasionally exceeded even a 400-token
budget), which enter as 0.5/0.5 votes under `unparseable_mode: distribution`;
elicitation-mode differences (K=10 sampling in both, but different token
budgets). Treat as a pilot signal that elicitation-with-deliberation is a real
condition axis, not as a measured effect size.
