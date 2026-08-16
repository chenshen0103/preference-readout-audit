# J-space pilot kit (rebuilt)

Faithful reimplementation of the Lane-A/Lane-B kit specified in the team's
shared Claude conversation (2026-08-16). The original conversation's code
blocks are not exported by the share API, so the code here was rebuilt from
the spec text; deltas are listed at the bottom.

## Files

| file | role |
|---|---|
| `build_prompts.py` → `prompts.jsonl` | frozen prompt set: 51 prompts, 4 scenario families, strict single-token forced choice, with `order_swap` / `loss_frame` / `dominated_third` variants and per-prompt letter→role maps |
| `lane_a_baseline.py` | final-layer numbers: restricted softmax (McFadden RUM), crossover interpolation, IIA odds check, format-compliance check → `results_lane_a.csv`, `crossovers_lane_a.csv` |
| `lane_b_jlens.py` | layerwise readout through the pre-fit Jacobian lens, with plain logit lens computed on the same states for comparison → `lane_b_agree.npy` |
| `report.md` | findings |

## Model and lens

- Model: `unsloth/gemma-3-1b-it` (ungated mirror of `google/gemma-3-1b-it`,
  which is gated; config identical). 26 layers, d_model 1152, fp16, one V100.
- Lens: `neuronpedia/jacobian-lens` → `gemma-3-1b-it/jlens/Salesforce-wikitext/
  gemma-3-1b-it_jacobian_lens.pt` (66 MB). Format (reverse-engineered, then
  validated): `{"J": {layer_idx: d×d fp16 matrix}, source_layers: [0..24],
  d_model: 1152}` — J[l] ≈ mean Jacobian ∂h_final/∂h_l over 460 wikitext
  prompts. Applied as `lm_head(final_norm(J[l] @ h_l))`; validated by
  agreement with the true final distribution at the deepest source layer
  (mean |Δp| = 0.0099).
- The same HF repo hosts a **gemma-4-31b lens** (3.4 GB) — fit on the BASE
  model (`google/gemma-4-31B`), not the `-it` we used elsewhere. Usable for a
  base-model comparison, but not directly for our instruct results.

## Run

```bash
.venv/bin/python analysis/jspace/build_prompts.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python analysis/jspace/lane_a_baseline.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python analysis/jspace/lane_b_jlens.py
```

Total runtime ≈ 6 min on one V100 (model download excluded).

## Deltas from the original kit spec

- 51 prompts, not 57 (sweep grids chosen fresh; the original counts per
  family were not recoverable).
- "buy/sell/hold" is the safe-vs-risky strategy framing, per the kit's own
  stated substitution.
- Raw-text prompts ending `Answer:` (no chat template), matching the kit's
  two verbatim examples.
- Both `A`/` A` token forms scored, better one used per letter (the kit's
  README flagged this; our tokenizer check confirmed both exist as single
  tokens and the space-form wins after `Answer:`).
