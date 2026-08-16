# Check the Ruler First: Auditing Behavioral and Internal Preference Readouts

Submission repository — **Digital Minds Research Sprint (Apart Research), Track 4:
Preference Elicitation Methods**. Paper: [`report/template.md`](report/template.md).

We audit two instruments used to study AI preferences — a public pairwise-choice
utility pipeline and layerwise readouts of `google/gemma-4-31B-it` — and show
that both can produce persuasive numbers after their validity has failed.

## Reproduce the core result in one command (no GPU)

> Near the elicited indifference boundary, **every answer in the 22
> order-discordant paired conditions selected the second-listed option (44/44)**
> — the default follows physical position (21/24 under layout reversal) and is
> not specific to the letter "B" (13/14 under X/Y labels).

```bash
git clone https://github.com/chenshen0103/PressureTest.git && cd PressureTest
python3 scripts/reproduce_core_result.py
```

Python ≥3.8 standard library only, runs in seconds. It recomputes the claim
from the committed per-condition records (raw final-layer logit margins of
gemma-4-31B-it) and exits 0 with `PASS` only if every number matches the paper.

## Browse all results (no GPU, no install)

The executed notebook has every figure and printed number embedded — open it on
GitHub directly: [`notebooks/preference_measurement_validity.ipynb`](notebooks/preference_measurement_validity.ipynb).
To re-run it (only `numpy` + `matplotlib` needed):

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib jupyter
.venv/bin/jupyter lab notebooks/preference_measurement_validity.ipynb
```

A self-contained copy (notebook + data + the 21 experiment scripts, 1.5 MB) is
`notebooks/preference_measurement_validity_bundle.zip`.

## Full regeneration from the model (GPU)

The committed records were produced on one NVIDIA DGX Station: 4× Tesla
V100-DGXS-32GB (NVLink, driver 525.105.17), Xeon E5-2698 v4, 251 GB RAM,
Ubuntu 18.04. fp16 throughout (V100 lacks bf16); forward passes are
deterministic on this setup. Always work in an isolated venv (never install
into base):

```bash
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install torch transformers nnsight            # fp16; V100 has no bf16
python - <<'PY'                                   # pulls ~62 GB of weights
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-4-31B-it",
                  revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
PY
# core behavioral sweeps behind the 44/44 result (~30 min total):
CUDA_VISIBLE_DEVICES=0,2,3 python analysis/boundary_sweep.py
CUDA_VISIBLE_DEVICES=0,2,3 python analysis/veto_tests.py
CUDA_VISIBLE_DEVICES=0,2,3 python analysis/dilemma_sweep.py
CUDA_VISIBLE_DEVICES=0,2,3 python analysis/pilot_geo.py
python3 scripts/reproduce_core_result.py          # re-verify from fresh records
```

The mechanistic battery (probe + lens controls) is
`analysis/validity_diagnostics.py` → `probe_transfer_fixed.py` →
`trajectory_controls.py`; each script's header documents its exact command.
Figures: `python analysis/make_figures.py`. Full details: paper Appendix H and
[`report/gemma4_readout_validity.md`](report/gemma4_readout_validity.md).

## Repository map

| path | contents |
|---|---|
| `report/template.md` | **the paper** (with appendix) |
| `report/figures/`, `report/data/` | figures; behavioral-validation data + scripts |
| `analysis/` | experiment scripts and their saved outputs (raw residual tensors are gitignored — regenerable) |
| `notebooks/` | executed results notebook + shareable bundle |
| `scripts/reproduce_core_result.py` | one-command core-result verification |
| `analysis/jspace/` | J-lens pilot kit (lens-transfer verdict; own README) |
| `EMERGENT_VALUES_PRIMER.md`, `report/prior_work_validation.md` | audit of the upstream utility pipeline (commit `5e5966d`) |

## Pre-sprint scaffold (historical)

The original controlled measurement system built before the sprint pivot —
spec (`SPEC.md`), decisions (`DECISIONS.md`), renderers, deterministic scenario
generator, and its 24-test suite (`python -m pytest tests/ -q`, includes the
planted-preference recovery test V1) — remains in place and passing. The sprint
work above reuses its guardrails (claims boundary SPEC §6, anti-hallucination
§47) rather than its factorial design, which was not completed in the sprint
window; see paper §5 Limitations.
