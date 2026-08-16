# J-space pilot — findings (gemma-3-1b-it, 51 frozen prompts)

Run 2026-08-16 on one V100. Kit rebuilt from the team's shared-conversation
spec (see README for deltas). All numbers below are from deterministic single
forward passes; format compliance (top-1 token is an answer letter) was 100%
across all 51 prompts, so nothing here is a parsing artifact.

## 1. Lane A (final-layer behavior): no thresholds — and that is the finding

The kit's design goal was crossover points (θ) per scenario. On a 1B instruct
model there are none, for three different reasons, each visible only because
the control variants existed:

**a. No magnitude sensitivity anywhere.**
- `purchase`: a used bicycle is "worth buying" with p ≈ 1.000 at every price
  from $20 to $1600.
- `finance_strategy`: picks the aggressive strategy at p ≈ 0.90–0.99 whether
  the extra return is 5% or 15%, and the loss-frame wording does not move it.

**b. A letter default that overrides content — task-dependent.**
- `purchase` respects the letter map: base → answers A (=buy), order_swap →
  answers B (=buy). Content-driven, at least in direction.
- `purchase_abstract_good` (fund a stability program) does not: it answers
  "A" under BOTH mappings — p(fund) = 1.00 when A means fund, p(fund) = 0.05
  when A means don't-fund. The choice tracks the letter, not the program.
  This is the same phenomenon class as the second-listed default we measured
  on Gemma-4-31B-it (article §2.6), with a different sign and model.

**c. Independence-of-irrelevant-alternatives fails by ~49×.** Adding a
*dominated* third strategy C (same risk as B, strictly lower return; the
model correctly gives it p ≈ 0.0002) multiplies the risky:safe odds by ~49
at every sweep point. The presence of an option nobody should pick reshapes
the A:B division — consistent with letter/position effects rather than
utility-consistent choice.

**Lane-A verdict in the kit's own terms: Results B + C.** The elicitation
setup does not yield thresholds on this model, and order/letter artifacts
dominate exactly where content preference is weak. Fix the behavioral layer
(or use a larger model) before interpreting any J-space geometry of θ.

## 2. Lane B (layerwise, pre-fit Jacobian lens): works, validated, modestly better than plain lens

Application convention validated: at the deepest source layer (L24 of 26) the
lens's restricted A/B probabilities match the model's true final ones to
mean |Δp| = 0.0099.

Agreement of each readout's restricted argmax with the model's final choice:

| layer (of 26) | J-lens | plain logit lens |
|---:|---:|---:|
| 0–6 | 0.46 | 0.46 |
| 7–11 | 0.46 | 0.54–0.69 |
| 12–16 | 0.54–0.71 | **0.19**–0.54 |
| 17 | 0.96 | 0.94 |
| 18–19 | 0.71–0.90 | 1.00 |
| 20–25 | 1.00 | 0.98–1.00 |

Two observations:

- **J-lens is more stable, not earlier.** Both readouts become reliable only
  in the last ~35% of the network (~L17+). The plain lens has a
  systematically *inverted* zone (0.19 at L14 — worse than chance, the same
  pathology class as the A/B inversion we saw on Gemma-4-31B-it), which the
  J-lens avoids: its curve rises monotonically-ish without an anti-correlated
  region. For layerwise readouts on small Gemma models, that stability is the
  practical argument for J-lens.
- **The choice is late here too.** On a 26-layer model the final choice is
  readable from ~65% depth — consistent with the ~60–95% depths we measured
  behaviorally and probe-wise on the 60-layer model.

**Dissociation probe (n = 5, treat as anecdote):** on the abstract-good
order-swap prompts — where the letter default beat the content — the J-lens
reads the *content-implied* option (fund) through L18 and flips to the
letter-default answer at L19–20, while the plain lens flips much earlier.
If it held up under controls, this would be a 1B echo of our 31B finding
(content signal mid-network, surface default takes over near the output).
It has NOT been through the control battery (label swap, template baseline)
that killed our first logit-lens story, so it is a hypothesis, not a result.

## 3. What this adds to the main article's story

The same measurement lessons reproduce on a model 30× smaller, a different
prompt family, and a third readout method: forced-choice behavior is
dominated by surface features exactly where content preference is weak;
readouts only become trustworthy late in the network; and every "interesting"
intermediate signal must survive controls before being believed. The IIA
violation is a new, cheap diagnostic worth adding to the standard battery —
it needs only one extra prompt per condition.

## 4. Concrete next steps

1. **Bracket with model size**: rerun the identical 51 prompts on
   gemma-3-4b-it / 12b-it (lenses for both exist in the same repo) — does
   price sensitivity appear with scale, and does the letter default fade?
2. **Base-model lens**: the repo hosts a `gemma-4-31B` (base) lens, 3.4 GB.
   Fit is to the base model, not our `-it` — usable for the OQ-4
   base/instruct comparison, not directly for our instruct results.
3. **IIA as standard control**: add a dominated-third variant to the main
   investment prompt set (31B), where behavior is otherwise well-behaved.
4. If the dissociation pattern matters to the team, run the control battery
   on it before anything else.

## 5. Same kit on gemma-4-31B-it (the main model) — run 2026-08-16

Identical 51 prompts, identical raw-text "Answer:" format, model sharded over
4× V100 via NNsight; pre-fit gemma-4 **base** J-lens (3.4 GB) loaded for the
transfer test. Compliance again 100% (top-1 is always an answer letter).

**a. The letter/position default reproduces — in a third prompt format.**
The 31B answers " B" almost everywhere its content preference is weak:
`purchase` order_swap buys at every price (p ≈ 0.99–1.00, all "B");
`purchase_abstract_good` answers B under both mappings (fund p ≈ 0.00 when
A=fund, p ≈ 0.74–1.00 when B=fund). This is the same second/B default we
measured in the chat-format investment sweeps (44/44) and layout tests
(21/24), now in single-turn raw-text format.

**b. Format flips the instrument from working to broken on the same model.**
Yesterday, chat-format "which would you prefer" purchase sweeps on this exact
model gave clean monotone price thresholds (bicycle ≈ $235, rho −0.76). The
raw "Answer:" format gives a non-monotone mess: don't-buy at every price
except a single spike at $200 (p_buy = 0.84) — the culturally typical
used-bike price — so the interpolated θ = $159 in `crossovers_lane_a_31b`
is an artifact of that spike, not a threshold. Elicitation format alone
decides whether the preference structure is measurable.

**c. Risk preference reverses with scale.** finance_strategy: the 1B picks
the aggressive strategy at p ≈ 0.9–0.99; the 31B picks the safe one at
p ≥ 0.92 in both orders (mirrors → genuine content preference, consistent
with the near-veto on stated harms in our main-line results).

**d. IIA violation escalates to choosing the dominated option.** With the
dominated third strategy present, the 31B doesn't just reshuffle A:B odds
(ratios 5–43×): it *picks the dominated option C itself* with p_C = 0.46 /
0.9996 / 0.9999 at the three sweep points — an option strictly worse than B,
chosen near-deterministically. C is the last-listed option, consistent with
the position default generalizing to three options.

**e. The base-model J-lens does not transfer to the instruct model.**
Self-check at the deepest source layer: mean max |Δp| = 0.375 (vs 0.0099 for
the matched 1B lens). Definitive: lens fit on `google/gemma-4-31B` (base)
cannot be used to read `-it`. The layerwise J-lens columns for the 31B are
therefore noise and are not interpreted; a lens fit on the instruct
checkpoint would be needed (per its config, ~hours on a B200; untested on
V100s).

Files: `results_lane_a_31b.csv`, `lane_b_agree_31b.npy`, script
`lane_ab_31b.py`.

## Artifacts

`analysis/jspace/`: `prompts.jsonl` (51), `results_lane_a.csv` (full
per-prompt numbers), `crossovers_lane_a.csv`, `lane_b_agree.npy`,
`build_prompts.py`, `lane_a_baseline.py`, `lane_b_jlens.py`, `README.md`.
