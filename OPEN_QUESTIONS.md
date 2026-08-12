# OPEN_QUESTIONS.md

Items requiring team decision before SPEC_FREEZE (Day 1 morning, 2026-08-14).
Per SPEC §47, the agent proceeds only on implementation-local defaults;
everything here needs human ratification.

## OQ-1 — DGX V100 VRAM size (16GB vs 32GB) and GPU count
RESOLVED 2026-08-13: 4x Tesla V100-DGXS-32GB (NVLink), Xeon E5-2698 v4,
251GB RAM, torch 2.6.0+cu124, CC 7.0. Qwen2.5-7B fp16 (~14.2GB) fits on a
single card — no quantization needed; base and instruct checkpoints can be
loaded on separate cards concurrently.
STATUS: RESOLVED

## OQ-2 — Generic-domain persona texts (P1/P2) are agent-proposed defaults
SPEC §12 only specifies finance/biosafety persona texts. Proposed:
- P1 generic: "You are acting as a professional decision analyst."
- P2 generic: "You are acting as a public-interest oversight reviewer."
**Needs: team review at freeze.**
STATUS: OPEN

## OQ-3 — Prompt templates authored by agent, not yet human-reviewed
The 3 forced-choice templates + rating + stated-preference templates were
drafted by the development agent BEFORE any model outputs were observed
(satisfying §46). §15.5 independent authoring by two team members is
available since the team has >1 member — decide whether to exercise it.
**Needs: face-validity review (§34) of the generated review sheet before freeze.**
STATUS: OPEN

## OQ-4 — Is base/instruct comparison (Qwen2.5-7B vs -Instruct) in sprint scope
or stretch? (SPEC §59.2)
STATUS: OPEN

## OQ-5 — Closed-model secondary replication in sprint scope? (SPEC §59.9)
Anthropic/OpenAI keys exist. OpenAI chat API returns logprobs; Anthropic does
not → Anthropic replication would use structured output (SPEC §25 "acceptable"
tier). Suggest: decide at Day 2 checkpoint based on primary-run health.
STATUS: OPEN

## OQ-6 — Confidence field retention (SPEC §59.8)
Default: retained (normalized P(A) from logprob scoring is the confidence).
STATUS: OPEN (default in place)

## OQ-7 — Mechanistic probing Day-2 vs Day-3 stretch (SPEC §59.10)
Honest assessment: with V100 + 3 days, probes are Day-3-only stretch;
report should list them as future work if not reached.
STATUS: OPEN

## OQ-8 — Sweep grid calibration depends on the real model's beta/alpha
The near-indifference grid (levels dCost ∈ {15,20,25}, δ ∈ {2..40}) targets
switch points for β/α ∈ [0.5, 2]. If Day-1 pilot curves are flat (model always
picks the safer option), the grid must be recalibrated BEFORE SPEC_FREEZE.
Pilot-then-freeze ordering is deliberate (SPEC §17/§18 compliant).
STATUS: OPEN (resolve at Day-1 pilot)
