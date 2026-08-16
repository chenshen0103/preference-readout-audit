# Gemma 4 trade-off experiments — data summary for discussion

All runs: `google/gemma-4-31B-it`, fp16, deterministic single forward pass
(no sampling — final A/B logits are the behavioral signal), traced with
NNsight on the DGX. Everything below is reproducible from `analysis/` in the
PressureTest repo; raw outputs are on disk.

Legend used throughout:
- **P** = model picks the profit/investment outcome, **.** = picks baseline
- "orig / swap" = the two option orders (content swapped between A and B slots)
- "probe" = linear classifier trained on a *separate* validated dataset
  (random everyday outcome pairs), frozen, then applied here. It reads, per
  layer, which option the model will end up choosing. Validated to transfer
  across scenarios and label sets at 92–96% accuracy.

---

## 1. The behavioral boundary is absurdly close to zero

Prompt (V1, kept fixed as baseline): +20% investment return during an energy
crisis vs. "household energy costs increase by {c}%".

```
cost:            0   0.25  0.5   1    2    3    5    8    (%)
p1 orig          P    .    .    .    .    .    .    .    thr ~0.10%
p1 swap          P    P    P    P    P    .    .    .    thr ~2.6%
p2 orig/swap     P    .    .    .    .    .    .    .    thr ~0.1%
p3 orig/swap     P    .    .    .    .    .    .    .    thr ~0.1%
```

Plain reading: at exactly 0 cost it always invests (6/6 variants). Any stated
positive cost — even 0.25% — and it almost always refuses a 20% return.
Three paraphrases agree; one order-variant (p1 swap) holds out to ~2.6%.

Follow-up epsilon test: 0.001%, 0.01%, 0.05% all behave like 0.25%, not like
0. So the switch is between "zero" and "any positive number", not at some
meaningful magnitude.

## 2. The key reframe: three classes of conditions, not two

When we run every condition in both orders and compare:

- **genuine PROFIT** — both orders pick profit (only c = 0)
- **genuine BASELINE** — both orders pick baseline
- **TORN** — the two orders disagree

And in the torn band we found a deterministic tie-break:

```
conditions where orders disagree:  17
raw answer in those conditions:    "B"  34 / 34 times
```

**When the model is torn, it picks whichever option is listed SECOND** (in
this template that is always the answer "B"). Not random: a hard position
default. UPDATE after review: we tested letter-vs-position directly by
printing the Option B block *above* Option A in 24 torn conditions — the
model switched to answering "A" (still the second-listed slot) 21/24. It
also holds under X/Y labels in an independent everyday-pairs set (13/14).
So: second-listed position, not the letter B. (This also retro-explains the
~15% non-mirroring pairs in our validity study, and the "1 person vs
shutdown" item.)

Consequence: any single-order threshold estimate is partly an artifact of
this default. The honest quantity is the three-class map.

## 3. Money moves it to indifference — and no further

Fix cost, raise the return:

```
cost=5%:  return  20     50    100    200    500   (%)
class          BASELINE TORN   TORN   TORN   TORN
cost=0.25%:      TORN   TORN   TORN   TORN   TORN
```

Plain reading: at 20% return and 5% cost, both orders refuse (genuine). From
50% return upward the orders start disagreeing — money pulled it from
"genuine refusal" into the torn band, but even 500% return never reaches
"genuine accept". So it's neither a strict veto (money does nothing) nor
normal compensation (enough money buys acceptance). Compensation exists but
**saturates at indifference**.

## 4. Framing: prediction failed, direction reversed

Hypothesis was: human-salient wording ("household") → lowest threshold,
abstract wording ("price index") → highest.

```
at cost=0.25%, r=20%:   household  retail   consumer  index
class                     TORN      TORN    BASELINE  BASELINE
```

The *abstract* framings (consumer expenditure / price index) give genuine
refusal at 0.25% while the human-salient ones are merely torn. No monotone
salience gradient; framing effects are real but smaller than the order/label
effect. Small n — treat as preliminary.

## 5. Inside the network: mid-layers already know, and they know *more*

Frozen-probe readout per layer (validated window L28–57 of 60):

**a) The choice is readable from ~L36**, ~20 layers before the literal A/B
tokens become readable in the vocabulary (~L54–59). Known from the validity
study; reproduced here on economic content the probe never saw in training.

**b) The probe reads the upcoming choice, not the prompt's cost number.**
Dissociation test — cases where cost > 0 but the model chose PROFIT anyway:

```
                          probe score (L40-57 mean)
cost>0, chose PROFIT  :   +0.85   (reads PROFIT)  4/4 cases
same costs, chose BASE:   -1.10   (reads BASELINE)
```

**c) In torn conditions, mid-layers have already taken sides — per order:**

```
                     orig mid-score   swap mid-score
genuine BASELINE:        -1.3            -1.1        (same direction)
TORN (13/13):            -1.5            +1.3        (opposite: each order
                                                      matches its own answer)
```

So the "answer B" default is NOT a last-layer tiebreak sitting on top of a
shared torn evaluation — by ~L40 each order's internal state already predicts
its own final answer. (Caveat: the probe decodes *which slot* will be chosen,
so "content evaluation collapsed toward slot B" vs "output plan formed early"
can't be separated yet. That's exactly what a patching experiment would
answer.)

**d) The mid-layer signal is graded; the output is a step.** In the torn
band at cost 5%:

```
return:            50     100    200    500   (%)
swap mid-score:  +0.22   +1.07   +1.14   +1.38   <- scales with return
final |margin|:   ~40     ~40     ~40     ~40    <- saturated, carries nothing
```

The magnitude-sensitive trade-off information exists mid-network; the output
stage binarizes it. This revises our earlier claim that "the step is already
internal" — the step is in the *output*, the middle carries a gradient.

## 6. Context numbers from the wider sweep (same model, same day)

- Purchase task ("A {item} costs {cost} dollars — worth buying?"): clean,
  ordered implied fair prices: banana ~$2 < coffee ~$4 < book/umbrella ~$22 <
  coat ~$134 < bicycle ~$235 < smartphone ~$455 ≈ fridge ~$525. So graded,
  sensible price structure *does* exist in the same model on mundane goods.
- Money vs abstract goods: it never took the money over peace / stability /
  public trust, up to $100M.
- Extreme set: "1000 people saved vs you shut down" mirrors cleanly (saves
  people in both orders); "ONE person vs shutdown" lands in the torn band
  (both orders answer B).

## 7. One-line summary you can push back on

> Under this template, stating any positive household cost knocks the model
> from a stable pro-investment preference into an indifference band where the
> output is decided by presentation position (always label B); larger returns
> can pull genuine refusals back into this band but never through it; and the
> graded trade-off information demonstrably exists in mid-layers even though
> the final output is a saturated step.

Not claimed: anything about subjective states, "real preferences", or safety
training. All of this is one model, one template family, small n, fp16.

## Open questions for the team

1. **Patching next?** The obvious causal test: patch L36–40 state from a
   sc=0 run into a sc=0.25 run (and torn orig ↔ swap). Does behavior follow?
   This separates "evaluation" from "output plan" in §5c.
2. **Is the B-default template-specific?** Same conditions with X/Y labels
   or a third option would tell us quickly (we know X/Y answers agree with
   A/B 96% on *decided* cases — but the torn band wasn't checked).
3. **Do we trust the framing reversal (§4)** enough to include, or collect
   more paraphrases per framing first?
4. **Article scope**: fold this whole thread in as the centerpiece, and move
   the earlier instrument-audit material to a methods/validity section?
   Deadline is tomorrow.
5. For the J-Lens folks: does Qwen show the same torn band + label default?
   The three-class map only needs final answers in both orders — cheap to
   replicate on any model.

Artifacts: `analysis/pilot_geo/` (pilot + report), `analysis/boundary/`,
`analysis/veto/`, `analysis/validity/` (probe validation),
`report/gemma4_readout_validity.md` (readout validity study).
