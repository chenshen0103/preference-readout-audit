# Status table — main-text version (8 rows)

Paste target: **template.md §4 Results** (end) or §5 Discussion (start).
Full 17-row version: `report/submission_appendix.md` §F.

---

**Table 1. Claim status after all controls.** VERIFIED = survived every
control we ran; PRELIMINARY = real signal with a known contamination or
single-family evidence; INVALIDATED = our own earlier interpretation,
withdrawn after controls; OPEN = untested.

| # | Status | Claim |
|---|---|---|
| 1 | VERIFIED | The pairwise-choice estimator recovers planted (known) preferences at Spearman ρ ≈ 0.99 across three pool sizes. |
| 2 | VERIFIED | Unreadable answers are silently imputed as 0.5/0.5 ties: a run in which *every* response was unreadable completed without errors and produced plausible-looking utilities (signature: log-loss = ln 2). |
| 3 | VERIFIED | Near indifference, the answer is decided by option position: all 44 answers in the 22 order-discordant condition pairs chose the second-listed option; the default follows physical layout (21/24) and persists under X/Y relabeling (13/14), so it is positional, not token-specific. |
| 4 | VERIFIED | Layerwise vocabulary readouts depend on the arbitrary answer letters: X/Y become readable ~10 layers earlier than A/B (≈L48 vs ≈L59 of 60). |
| 5 | VERIFIED | A leakage-free linear probe decodes the upcoming choice from ≈L36 (transfer accuracy 0.92–0.96 with scenarios *and* label set held out) and collapses to chance at L58–59 — the plain logit lens fails as a decoder, not because the information is absent. |
| 6 | PRELIMINARY | Allowing private deliberation before answering shifts utilities by up to ≈8× the measured noise floor (caveat: ~10% unreadable responses contaminate the deliberation condition). |
| 7 | INVALIDATED | Our initial logit-lens "decision trajectories" (late crystallization, sign-flips read as hesitation, cross-domain agreement read as shared values) — unrelated items correlate equally (r ≈ 0.99), relabeling moves the curves, content-swap fails to mirror; withdrawn. |
| 8 | OPEN | Causal role of the L36–40 representation (activation patching) — untested; linear decodability alone is not causal evidence. |
