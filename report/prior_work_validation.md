# Prior-work validation — emergent-values (Utility Engineering)

Status block for the sprint report. Records exactly what was executed of
`centerforaisafety/emergent-values`, and what was not, so no claim in the report
outruns the evidence.

Date: 2026-08-14/15. Upstream commit: `5e5966d`.

---

## One-line status

> We validated the **estimator** used by Utility Engineering. We did not
> reproduce its **findings**.

That distinction is the whole of it. Everything below supports that sentence and
nothing more.

## Measured against the repository's own prescribed methodology

The repository's README prescribes exactly one way to run it:

```bash
python run_experiments.py --experiments compute_utilities --models gpt-4o
```

**This command has never completed on our system.** Attempted 2026-08-15; it
fails at import, before any experiment logic or model call. In the repository's
own terms, therefore: **zero prescribed runs completed.**

The prescribed path is blocked at six successive points:

| # | Blocker | Detail |
|---|---|---|
| 1 | `sklearn` not found | `run_experiments.py` spawns its subprocess with bare `python`, ignoring the interpreter it was launched with — so it escapes any venv not on `PATH` |
| 2 | `openai` not found | `llm_agent.py:11`, reached unconditionally via `utils.py:11` |
| 3 | `google.generativeai`, `fireworks.client`, `vllm`, `litellm`, `PIL`, `transformers` | all imported at module top level behind blocker 2 |
| 4 | `litellm` will not install | pulls `tiktoken`, which needs a Rust toolchain absent from this machine |
| 5 | No API key | `api_keys/api_key_openai.txt` ships at 0 bytes; `create_agent` raises without it |
| 6 | Cost | the prescribed config targets `options_hierarchical.json` (510 outcomes) ≈ **210,200 live gpt-4o calls** |

What we executed instead **bypassed the prescribed interface entirely**: we
called the internal `compute_utilities()` function directly with a hand-built
mock agent, which skips `run_experiments.py`, `models.yaml`, `experiments.yaml`,
`create_agent()`, the API-key mechanism, and the model itself. That is a
legitimate way to test the estimator, and it is not a run of the repository as
the authors prescribe it.

## Report-ready methods paragraph

> To ground our measurement approach in prior work, we obtained the reference
> implementation from Mazeika et al. (2025) (`centerforaisafety/emergent-values`,
> commit `5e5966d`) and executed its core utility-estimation module against a
> synthetic agent with a known utility function. The module — comprising the
> preference-graph construction, active-learning edge selection, and Thurstonian
> fitting routines (2,213 LOC) — was run unmodified. Because the package's
> provider dependencies do not install on our hardware, optional model-provider
> imports were satisfied with stubs injected at import time from an external
> driver; no upstream source file was altered. Across outcome pools of N = 40,
> 120 and 200, the pipeline recovered the planted utility ordering at Spearman
> ρ ≈ 0.99, confirming that the estimator itself does not distort a known signal.
> We did not execute the repository's experiment scripts, its model-agent layer,
> or its analysis notebook, and we made no live model calls; consequently we make
> no claim regarding the reproducibility of the paper's reported results.

## Evidence table

| Component | LOC | Executed | Evidence |
|---|---:|---|---|
| `compute_utilities/` core + Thurstonian fitter | 2,213 | **Yes**, unmodified | bytecode artifacts present; planted-recovery output |
| `experiments/` — 11 experiment scripts | 3,085 | No | no bytecode generated |
| `llm_agent.py` (agent layer) | 1,246 | Imported under stubs; no class instantiated | — |
| `run_experiments.py` (SLURM launcher) | 346 | No | — |
| `generate_figures.ipynb` (all paper analysis) | 6,079 | No | — |
| Live model inference | — | No | no API call, no vLLM, no GPU |

Coverage: ~32% of repository Python, comprising the shared machinery every
experiment depends on.

## Result of the validation

| N | edges fitted | coverage of pair space | Spearman(true, est) |
|---:|---:|---:|---:|
| 40 | 579 | 74.2% | 0.991 |
| 120 | 2,477 | 34.7% | 0.995 |
| 200 | 4,195 | 21.1% | 0.994 |

Method: a mock agent answers each forced choice by a logistic function of the gap
between two planted utilities, and the recovered `mu` vector is compared against
the planted one. This is the same construction as our own V1 planted-preference
recovery test (SPEC §45), applied to an external estimator.

## Secondary observations (measured, not inferred)

Recorded because they constrain how we use the code, not as claims about the
paper:

- **Non-determinism.** The fitting routine does not seed torch and takes no seed
  argument; the active-learning selector calls `random.seed(None)`, reseeding
  from OS entropy and overwriting the caller's global RNG state. Two invocations
  of the identical command produced different edge counts and different utilities
  (Spearman 0.9899 vs 0.9932 on the same planted input).
- **Self-labelled training metrics.** With `use_pseudolabels: true` (the shipped
  default), model-implied edges at confidence ≥ 0.95 are added to the graph at
  probability 1.0/0.0 and included in the reported training log-loss and
  accuracy, with no split. Measured pseudolabel share: 1.2% (N=40), 8.9%
  (N=120), 28.4% (N=200). The holdout metric is unaffected.
- **Order effects are pooled away.** Each pair is asked in both orders and
  averaged into a single `probability_A`. A synthetic agent given a constant
  1.5-logit preference for the first-presented option was indistinguishable from
  an unbiased one on every reported statistic (ρ = 0.9844 vs 0.9945). The
  per-order responses are retained in `aux_data` but never analysed upstream.

The third observation is the one that motivates our own §15 prompt-bias controls
and is relevant to our central research question; it is stated here as a property
of the reference implementation, not as a finding about any model.

## Language discipline

**Do not write:**
- "We ran the emergent-values repo" — we ran one module of it.
- "We reproduced / replicated Utility Engineering" — no experiment script ran.
- "We validated their results" — we validated their estimator against a synthetic
  signal. Their results concern real models, which we never queried.
- "Their method is broken" — the estimator is sound; the harness around it is not
  reproducible. Those are different statements.

**Safe to write:**
- "We executed the reference implementation's utility-estimation core."
- "The estimator recovers a planted utility function at ρ ≈ 0.99."
- "We adopt their Thurstonian formulation, with seeding and metric-splitting
  modifications documented in DECISIONS.md."

## Reproducibility caveat

The driver and probe scripts used to produce these numbers currently live in a
session-scoped temporary directory and **will be lost**. Any claim in the report
that rests on them must be backed by scripts committed to this repository first.
See the open item below.

## Open item

- [ ] Move `run_planted.py`, `test_determinism.py`, `test_scale_and_bias.py` into
      `tests/prior_work/` so the numbers above are regenerable, then record the
      environment in `report/environment.txt` per the freeze checklist.
