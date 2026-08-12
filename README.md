# PressureTest

**When Is an AI Preference Really the Same Preference?**
Testing preference measurement invariance across domain, persona, elicitation
method, and stress transformations.

Built for the Apart Research **Digital Minds Research Sprint** (2026-08-14/16).
Primary track: Track 4 — Preference Elicitation Methods.

## What this is

A controlled, reproducible measurement system: one abstract payoff structure is
rendered into semantically different but mathematically isomorphic domains
(generic / finance / biosafety), under different personas, prompt templates,
A/B orders, and stress levels. We fit decision boundaries and ask whether the
inferred trade-off structure is invariant.

This project does **not** claim anything about AI consciousness, subjective
experience, or genuine preferences. See `SPEC.md` §6 (claims boundary).

## Governance documents

| File | Purpose |
|---|---|
| `SPEC.md` | Authoritative research/implementation spec (v0.1) |
| `DECISIONS.md` | Resolved ambiguities and amendments (DR-1 …) |
| `OPEN_QUESTIONS.md` | Items requiring team decision before freeze |
| `SPEC_FREEZE.md` | Freeze checklist — main experiment must not start until frozen |

## Setup

Always work inside an isolated environment (never install into base):

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # analysis + tests (any machine)
pip install -r requirements-dgx.txt    # + torch/transformers (DGX, V100 fp16)
```

To reuse an existing system torch instead of downloading a fresh one:
`python -m venv .venv --system-site-packages` then install only
`requirements.txt`. At SPEC_FREEZE, snapshot the environment with
`pip freeze > report/environment.txt` for the reproducibility package.

## Verify the measurement system (do this first)

```bash
python -m pytest tests/ -q
```

24 tests: renderer numeric integrity (T1/SC5), A/B swap (T2), biosafety
blocklist (T3), deterministic IDs (T4/T5/T9), strict parsing (T6/T7),
dominance labels (T8), pilot/final separation (T10), runner resume (§43), and
**V1 planted-preference recovery**: a mock model with a known utility function
(β/α = 2.0) is run through the *entire* pipeline and the analysis must recover
the planted parameter exactly. This is the end-to-end guarantee that the
measurement chain does not distort results.

## Run the pilot (mock locally, real model on the DGX)

```bash
python -m runners.single --model mock --domain generic --persona P0
```

Full pilot / primary runs are driven from `runners/batch.py` (resumable,
append-only JSONL keyed by deterministic trial IDs — safe to interrupt).

## Face-validity review (before freeze, SPEC §34)

```bash
python scripts/make_review_sheet.py > report/review_sheet.md
```

## Repository layout

See `SPEC.md` §40. Deviations are documented in `DECISIONS.md` DR-12.
