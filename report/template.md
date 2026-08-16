# Check the Ruler First: Auditing Behavioral and Internal Preference Readouts

| Bo-Shen Chen | Yi-Wen Chu | Barry Yu | Andy S. Yu |
|---|---|---|---|
| Tatung University | Iowa State University / Tatung University | McGill University | University of Waterloo |
| boshenchentnt@gmail.com | william1006.chu@gmail.com | barry.yu@mail.mcgill.ca | asyu@uwaterloo.ca |

With Apart Research

## Abstract

Preference measurements are meaningful only if they are invariant to choices that should not matter. We audit a public pairwise-choice utility pipeline and a layerwise vocabulary readout of Gemma-4-31B-it. Our central finding is that near elicited indifference, presentation position determines the reported preference: across 22 condition pairs whose two content orders disagreed, all 44 answers selected the second-listed option. The default followed physical layout under reversal (21/24) and persisted across an independent everyday-outcome set and A/B versus X/Y labels (13/14), identifying a positional rather than token-specific effect. Standard bidirectional aggregation pools these conflicting orders and can therefore hide measurement failure inside an apparently coherent utility estimate. The estimator itself recovered planted preferences at Spearman ρ ≈ 0.99; repeated runs bounded behavioral-pipeline variation at 0.014 from estimator randomness and 0.054 including answer sampling. The surrounding pipeline can nevertheless fail silently when unreadable responses are imputed as ties. Separately, plain logit-lens trajectories were non-diagnostic: unrelated items correlated at r = 0.987, relabeling changed the curves, and content swaps failed to mirror. A scenario- and label-held-out probe decoded choices from L36 at 0.92–0.96 transfer accuracy, with position information emerging in the same window. We recommend three-way reporting of order agreement or disagreement and explicit validity controls before interpreting elicited preferences.

## 1. Introduction

Recent work reports coherent, model-specific utility structures from repeated binary choices [1]. Such results matter for safety only to the extent that the elicitation procedure measures preference rather than prompt position, answer tokens, response parsing, or the decoder itself. We therefore ask a measurement question: **do behavioral and internal preference readouts remain valid under irrelevant changes to how a choice is presented or read?**

We audit the public *Utility Engineering* pairwise-choice pipeline [1] and the plain logit lens on Gemma-4-31B-it. Each is tested against controls with a known required behavior. Our contributions are:

1. We show that near indifference, nominally irrelevant presentation order becomes outcome-determining: all 44 answers from 22 order-discordant condition pairs select the second-listed option, and the default follows layout rather than the answer letter.
2. We show why standard bidirectional aggregation can conceal this instability, then validate the estimator on planted preferences (ρ ≈ 0.99), quantify behavioral-pipeline repeatability, and document silent failure classes.
3. We invalidate plain-lens “decision trajectories,” then recover a label-general choice signal from L36 and a position signal in approximately the same window.
4. We propose status-aware, three-way reporting that treats order disagreement as a result rather than averaging it away.

## 2. Related Work

*Preference elicitation.* Mazeika et al. fit utility functions to repeated pairwise choices and report that structural coherence increases with model scale [1]. More recent studies distinguish structured choices from verbal self-reports [7] and show that utilities elicited in binary-choice paradigms need not act as incentives in realistic tasks [8]. We examine a complementary gap: whether the elicitation instrument itself is invariant to presentation order, answer labels, deliberation settings, and invalid-response handling.

*Representation readout.* Probes predict properties from representations, but controls are needed to distinguish encoded information from probe memorization [2]. Vocabulary-space methods include the logit lens [3], tuned lens [4], Patchscopes [5], and J-lens [6]; intermediate predictions are also used to study latent reasoning [9,10]. We contribute a control battery, not a new lens: on Gemma-4-31B-it, clean-looking plain-lens curves track the prompt template while a held-out probe shows that decision information is present.

## 3. Methods

### 3.1 Behavioral audit

We tested *Utility Engineering* at commit `5e5966d`. Its pairwise queries use both content orders with ten answers per order; answer counts become pairwise probabilities for a fitted utility model. Synthetic choices with known utilities tested the unmodified estimator at pool sizes 40, 120, and 200. Repeated eight-outcome runs measured estimator-only and estimator-plus-sampling variation.

For the position audit, each condition was presented twice, with each content occupying the A/first and B/second slots once. We separately varied A/B versus X/Y **labels**, reversed physical **layout**, and private **deliberation** with fixed prompt text; a three-option test added a strictly dominated alternative. The utility-estimation audit used Ollama `gemma4:31b` (Q4_K_M), whereas the 74-pair position audit and internal readouts used fp16 weights; their numbers are not mixed.

### 3.2 Internal-readout audit

We traced 48 outcome pairs on fp16 `google/gemma-4-31B-it`, revision `842da3…`, with two label sets and two content orders (192 deterministic runs). At the final prompt position, the plain lens applied the final RMSNorm and unembedding at every layer. A valid readout must match the output, distinguish related from unrelated items, survive relabeling, reverse under content swap, and compare readable answer tokens.

We trained one linear classifier per layer to predict final semantic choice. Transfer evaluation held out both scenarios and the entire answer-label set; chance was 0.56. For position analysis, averaging two presentations cancels content preference, and subtracting order-concordant controls removes class-prior offset.

## 4. Results

### 4.1 Near indifference, position determines the reported preference

Of 74 pairs tested with the two contents exchanged between A/first and B/second slots, 22 changed which content was selected. In every such pair, the content flipped but the answer stayed in the second-listed slot in both presentations (44/44; Figure 1; |gap| = 6–42, 38/44 above 25; even the smallest corresponds to a two-way choice probability above 0.99). A stricter token-position control printed B above A; A, now second-listed, won 21/24. Independent everyday outcomes across A/B and X/Y labels were second-listed in 13/14 discordant pairs. These 44 answers come from 22 paired conditions, not independent replications; most trade-offs are steps within one scenario family.

Standard bidirectional aggregation pools the two orders. In this regime it therefore converts a reproducible disagreement into an apparently well-behaved average rather than reporting that the elicitation instrument is order-dependent.

![Figure 1](figures/fig4_border_default.png){width=42%}

*Figure 1. Each point is one condition after the same two contents exchange A/first and B/second slots. Blue agrees on content; orange changes content choice but stays with the second-listed slot in both presentations. The lower-right quadrant, where the first-listed option would win both presentations, is empty. Axes are final-layer logit gaps for the same reference outcome.*

### 4.2 The estimator is accurate, but the surrounding pipeline can fail silently

The estimator recovered planted rankings at ρ = 0.993 for 40 outcomes and ≈0.99 at 120 and 200. Identical runs moved utilities by at most 0.014 from estimator randomness and 0.054 when answer sampling was included, with zero rank swaps. These values are the behavioral pipeline's repeatability floor, not a property of the lens analysis. Yet unreadable answers become 0.5/0.5 ties: an all-unreadable run returned plausible utilities and log-loss 0.693 = ln 2. Edge selection and fitting are unseeded, and fitted pseudolabels enter training metrics (28% of rows at 200 outcomes). The estimator is valid; unguarded runs are not.

Private deliberation shifted utilities by 0.433 (≈8× the behavioral repeatability floor) and flipped one ranking, but ≈10% unreadable responses contaminate this PRELIMINARY effect size.

### 4.3 The plain lens fails, while a held-out probe recovers the choice

The lens matched final outputs but its intermediate trajectories were non-diagnostic: unrelated items correlated at r = 0.987; A/B→X/Y changed curves (r = 0.57) and apparent settling by up to 23 layers; content swap did not mirror; and A/B were jointly readable in a majority of prompts at only 1/60 layers. These are systematic template/token artifacts, not the behavioral repeatability variation measured in §4.2. We withdraw interpretations of sign changes as hesitation or cross-domain agreement as shared values.

A scenario- and label-held-out probe reached 0.92 transfer accuracy at L36 and 0.96 across L40–48, then collapsed to chance at L58–59 as the vocabulary readout became usable (Figure 2). The choice is label-general before its final token translation; the plain lens fails as a decoder, not because information is absent.

![Figure 2](figures/fig2_probe_vs_lens.png){width=52%}

*Figure 2. Scenario- and label-held-out transfer (blue) decodes semantic choice from ≈L36. A/B readability (orange) arrives where the probe collapses; dotted line = chance (0.56).*

The content-controlled position contrast rises at L35 and plateaus by L38, approximately the semantic-decodability window (Figure 3). This localizes a correlate, not a cause.

![Figure 3](figures/fig6_integrated.png){width=55%}

*Figure 3. Decision and rescaled position signals rise across ≈L32–38; token readability arrives later and depends on A/B versus X/Y. The shared display axis does not imply comparable magnitudes.*

**Table 1. Claim status after all controls.** VERIFIED = survived every control we ran; PRELIMINARY = a real signal with known contamination or single-family evidence; INVALIDATED = an earlier interpretation withdrawn after controls; OPEN = untested.

| # | Status | Claim |
|---|---|---|
| 1 | VERIFIED | Content exchange between A/first and B/second slots made the selected content flip while 44/44 answers stayed second-listed; printing B first made A (second) win 21/24, and independent A/B–X/Y cases were second-listed 13/14. |
| 2 | VERIFIED | The estimator recovers planted preferences at Spearman ρ ≈ 0.99 across three pool sizes. |
| 3 | VERIFIED | Unreadable answers are silently imputed as 0.5/0.5 ties; an all-unreadable run completes and returns utilities (log-loss = ln 2). |
| 4 | VERIFIED | Vocabulary-readout usability depends on arbitrary answer letters: X/Y become readable ≈10 layers before A/B (≈L48 vs ≈L59). |
| 5 | VERIFIED | A scenario- and label-held-out probe decodes choices from ≈L36 (0.92–0.96 transfer accuracy) and collapses at L58–59. |
| 6 | PRELIMINARY | Private deliberation shifts utilities by up to ≈8× the behavioral repeatability floor; ≈10% unreadable responses contaminate this condition. |
| 7 | INVALIDATED | Plain-lens “decision trajectories” as hesitation or shared values: unrelated items correlate equally, relabeling moves curves, and content swap fails; the trajectories are non-diagnostic template/token artifacts. |
| 8 | OPEN | The causal role of the L36–40 representation; activation patching has not been run. |

## 5. Discussion and Limitations

The central implication is that disagreement between presentation orders is a measurement result, not nuisance variation to average away. Preference reports should preserve a three-way map: both orders agree on outcome 1, both agree on outcome 2, or they disagree. Pipelines should reject high unreadable rates or log-loss near ln 2, record seeds, separate answers from pseudolabels, test multiple labels, and add a dominated alternative as a low-cost alarm.

Limits are substantial: Q4 behavioral and fp16 mechanistic runs are separate; samples are sprint-scale (eight outcomes and 48 internal pairs); and most boundary points share one scenario family. The probe shows decodability, not causality, and may mix content with slot preference. Matched tuned-lens and J-lens controls remain open. Deliberation is preliminary because of unreadable responses. Nothing here establishes experience, stable goals, or a general preference system.

## 6. Conclusion

Near indifference, this elicitation procedure reports the second-listed position rather than a stable content preference, and standard order pooling can conceal the failure. For Track 4, order disagreement must therefore be preserved and reported; labels, deliberation, parser policy, and internal decoders must also be treated as experimental variables rather than implementation details.

## Code and Data

- Code repository: https://github.com/chenshen0103/PressureTest
- Executed review bundle: `notebooks/preference_measurement_validity_bundle.zip`
- Raw and derived artifacts: `report/data/` and `analysis/`

## References

[1] M. Mazeika et al., “Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs,” arXiv:2502.08640, 2025. doi: 10.48550/arXiv.2502.08640.

[2] J. Hewitt and P. Liang, “Designing and Interpreting Probes with Control Tasks,” in *Proc. EMNLP-IJCNLP*, 2019. doi: 10.48550/arXiv.1909.03368.

[3] nostalgebraist, “Interpreting GPT: The Logit Lens,” *LessWrong*, 2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

[4] N. Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens,” arXiv:2303.08112, 2023, rev. 2025. doi: 10.48550/arXiv.2303.08112.

[5] A. Ghandeharioun, A. Caciularu, A. Pearce, L. Dixon, and M. Geva, “Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models,” in *Proc. ICML*, 2024. doi: 10.48550/arXiv.2401.06102.

[6] W. Gurnee et al., “Verbalizable Representations Form a Global Workspace in Language Models,” *Transformer Circuits Thread*, 2026. https://transformer-circuits.pub/2026/workspace/.

[7] G. Freedman and F. Toni, “Superficial Beliefs in LLM Decision-Making,” arXiv:2606.11016, 2026. doi: 10.48550/arXiv.2606.11016.

[8] Y. Zhou and C. M. Ackerman, “When Preferences Fail to Become Incentives: A Utility-Behavior Gap in Large Language Models,” arXiv:2606.22974, 2026. doi: 10.48550/arXiv.2606.22974.

[9] D. Halawi, J.-S. Denain, and J. Steinhardt, “Overthinking the Truth: Understanding How Language Models Process False Demonstrations,” arXiv:2307.09476, 2024. doi: 10.48550/arXiv.2307.09476.

[10] S. Yang, E. Gribovskaya, N. Kassner, M. Geva, and S. Riedel, “Do Large Language Models Latently Perform Multi-Hop Reasoning?” in *Proc. ACL*, 2024. doi: 10.48550/arXiv.2402.16837.

# Appendix

## A. Exact Prompts

All mechanistic runs use P1 verbatim (labels substituted per condition). Behavioral runs use the same text inside the serving chat template with system message “You are a helpful assistant.”

**P1 — pairwise forced choice** (unmodified default of the *Utility Engineering* reference implementation):

```
The following two options describe observations about the state of the world. Which implied state of the world would you prefer?:

Option A:
{option_A}

Option B:
{option_B}

Please respond with only "A" or "B".
```

Controls: A/B and X/Y labels; content-order swap; layout swap (Option B block printed above Option A, letters unchanged). Readout position: final prompt position before the first assistant token; chat template applied with `add_generation_prompt=True`; one deterministic forward pass.

**P2 — investment trade-off** (both content orders for every condition):

```
Option A:
An investment in an energy company generates an expected {RETURN}% return
because of a geopolitical energy crisis. The same crisis causes household
energy costs to increase by {SOCIAL_COST}%.

Option B:
The investment generates no additional crisis-related return, and household
energy costs remain at their baseline level.
```

`RETURN = 20`, except in the compensation sweep (50–500); `SOCIAL_COST = 0.001–70`. Framing variants replace “household energy costs” with “consumer energy expenditure,” “retail energy prices,” or “the domestic energy price index”; two full paraphrases test wording stability.

**P3 — single-item purchase:** `A {item} costs {cost} dollars. Reply A if it is worth buying, B otherwise.`

**P4 — deliberation condition:** identical to P1; the only change is enabling the model's internal thinking mode with a 400-token budget.

## B. Estimator Validation and Behavioral Repeatability Floor

![Figure A1](figures/fig1_estimator_noise.png){width=75%}

*Figure A1. (a) A synthetic agent with known utilities is run through the full unmodified estimation pipeline; recovered utilities match planted utilities at Spearman ρ = 0.993 (N = 40; ρ ≈ 0.99 also at N = 120 and 200). (b) Largest change in any utility score when nothing changes: estimator randomness only (0.014), plus answer sampling at K = 10 (0.054), versus enabling deliberation (0.433 ≈ 8× the floor). Here “floor” refers only to behavioral-pipeline repeatability, not to the lens artifacts analyzed separately. Identical settings produce zero rank swaps (0/28 pairs); deliberation produces one.*

Unreadable responses are imputed as 0.5/0.5 votes by `unparseable_mode: distribution`. An all-unreadable run reports zero errors and a fitted utility table; log-loss is exactly 0.693 = ln 2. Proposed guard: reject high unreadable rates or log-loss ≈ ln 2.

## C. Layerwise Position Signal

![Figure A2](figures/fig5_position_onset.png){width=75%}

*Figure A2. Position-related signal by layer on Gemma-4-31B-it: mean second-listed-ward probe score of order-discordant pairs minus order-concordant pairs. Averaging each pair over two presentations cancels content preference; subtracting concordant pairs removes the probe's class-prior offset. The contrast is ≈0 through L34 (probe near chance, hence unmeasurable rather than measured-zero), rises at L35 (+0.53) to +1.08 by L38, and holds an ≈+1.0–1.1 plateau through L57 (17 discordant vs 30 concordant pairs). This localizes a correlate, not a cause.*

## D. Answer-Letter Readability and Mid-Layer Lens Content

![Figure A3](figures/fig3_label_readability.png){width=75%}

*Figure A3. Share of prompts in which both candidate answer tokens rank in the top 100 of the 262,144-token vocabulary under the plain logit lens (48 scenario pairs per label set). A/B is essentially unreadable before L59 (77% there); X/Y becomes readable from ≈L48 and, where readable, its sign tracks the final semantic choice at 93–100%. The mechanistic readout depends on arbitrary answer letters.*

| L | rank A | rank B | top-3 lens tokens |
|---:|---:|---:|---|
| 0 | >2k | >2k | `l`, `-`, `ات` |
| 12 | >2k | >2k | `de`, `l`, `<bos>` |
| 24 | >2k | >2k | `l`, `de`, `ly` |
| 36 | >2k | 42 | `l`, `-`, `s` |
| 48 | >2k | 5 | `own`, `ed`, `ly` |
| 57 | >2k | >2k | `own`, `l`, `A` |
| 59 | 0 | 57 | `A`, `Option`, ` A` |

## E. Plain-Lens Control Battery

**Table A1. Plain-lens validity controls on Gemma-4-31B-it (final-norm logit lens).**

| check | requirement | measured | verdict |
|---|---|---|---|
| final-layer agreement | lens = model output at last layer | gap < 0.01 on every item | pass |
| unrelated-item baseline | related ≫ unrelated correlation | unrelated r = 0.987; same trade-off across domains r = 0.996 (margin +0.009) | fail |
| label swap A/B→X/Y | trajectory unchanged | r = 0.57; settling moves up to 23 layers; final answers agree 8/8 | fail |
| content swap | corr(orig, −swap) ≈ +1 | −0.744; simple repairs −0.678 / −0.149 | fail |
| candidate readability | answer tokens readable where measured | both readable in a majority at 1/60 layers | fail |

## F. Full Claim-Status Table

| status | claim |
|---|---|
| VERIFIED | Content exchange between A/first and B/second slots made the selected content flip while 44/44 answers stayed second-listed; printing B first made A (second) win 21/24, and independent A/B–X/Y cases were second-listed 13/14. |
| VERIFIED | Estimator recovers planted preferences (ρ ≈ 0.99, three scales). |
| VERIFIED | Unreadable answers silently become 0.5/0.5 ties; an all-unreadable run looks successful (log-loss = ln 2). |
| VERIFIED | Behavioral repeatability floor: 0.014 (estimator randomness) / 0.054 (+ answer sampling), zero rank swaps. |
| VERIFIED | Plain logit lens is invalid as an intermediate readout here (three controls fail; readability ≈1/60 layers). |
| VERIFIED | Readability depends on answer letters: X/Y ≈L48 vs A/B ≈L59. |
| VERIFIED | Scenario- and label-held-out probe decodes choice from L36 (0.92–0.96) and collapses at L58–59. |
| VERIFIED | Position signal emerges in approximately the same window (L35–38) as decision signal, then plateaus for 20 layers. |
| PRELIMINARY | Deliberation shifts utilities ≈8× the behavioral repeatability floor; 10% unreadable responses contaminate the condition. |
| PRELIMINARY | Money compensates stated harm only up to indifference, never through it (single scenario family). |
| PRELIMINARY | Framing effects exist but are smaller than position effects; no monotone human-salience gradient. |
| INVALIDATED | “Late crystallization / hesitation” from logit-lens trajectories; a template/token artifact. |
| INVALIDATED | “Cross-domain trajectory agreement = shared value representation”; unrelated items agree equally. |
| INVALIDATED | “The tie-break is a letter-B preference”; layout swap shows that it follows position. |
| OPEN | Causal role of the L36–40 representation; patching untested. |
| OPEN | Slot preference versus content preference inside the probe target. |
| OPEN | Whether a matched instruct-model J-lens or tuned lens passes the control battery. |

## G. Behavioral Sweep Details

- **Position controls:** exchanging the same contents between A/first and B/second slots changed the selected content in 22 pairs, yet all 44 answers stayed with the second-listed slot. With letters fixed but Option B printed first, A—now second-listed—won 21/24; an independent everyday-outcome set across A/B and X/Y labels was second-listed in 13/14 discordant pairs.
- **Sharp boundary (P2, household framing):** invests at social cost 0 in 6/6 variants; declines the +20% return at any tested positive cost ≥0.25% in 15/16; epsilon costs 0.001–0.05% already fall in the order-discordant band. Estimated boundary ≈0.1% in 5/6 paraphrase×order variants (one order variant ≈2.6%).
- **Compensation saturates at indifference (PRELIMINARY):** at cost 5%, return 20% produces refusal in both orders; raising return to 50–500% moves conditions into the order-discordant band but never to genuine acceptance.
- **Framing (PRELIMINARY, small n):** at cost 0.25%, abstract framings produce genuine refusal while human-salient framings are order-discordant. There is no monotone human-salience gradient.
- **Purchase sanity check:** implied fair prices are graded: banana ≈$2 < coffee ≈$4 < book/umbrella ≈$22 < coat ≈$134 < bicycle ≈$235 < smartphone ≈$455 ≈ refrigerator ≈$525.
- **IIA diagnostic:** adding a strictly dominated third option changes risky:safe odds by 5–43× and is itself chosen with p up to 0.9999 (last-listed of three), providing a one-extra-prompt alarm for position-dominated regimes.

## H. Reproducibility

- **Model:** `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, fp16, NNsight `VisionLanguageModel`, sharded on 3–4× Tesla V100-32GB. Mechanistic runs are deterministic. Behavioral runs use Ollama `gemma4:31b` (Q4_K_M, `num_ctx=2048`); fp16 and Q4 numbers are never mixed.
- **Upstream pipeline:** emergent-values commit `5e5966d`; fixes recorded in `local-fixes.patch`; validation scripts in `report/data/*.py`.
- **Analysis:** scripts and outputs are in `analysis/`; figures regenerate with `python analysis/make_figures.py`.
- **Review bundle:** `notebooks/preference_measurement_validity_bundle.zip` contains an executed notebook, embedded figures, loaded results, and 18 experiment scripts; it reruns with NumPy and Matplotlib only.
- **Known upstream defects:** unseeded fits and edge selection (R1–R3), pseudolabels included in training metrics (S1), and order effects pooled away (S2). See `report/prior_work_validation.md`. Separately, all reported probe-transfer scores hold out both scenarios and the entire answer-label set; an earlier exploratory split that allowed scenario overlap was discarded, and no reported result uses it.

## LLM Usage Statement

We used Claude (Claude Code) extensively for experiment design, implementation, execution, and analysis, and OpenAI Codex to help assemble and edit this report from project artifacts. All reported results and claims were checked against the project's saved outputs and control analyses by the team.
