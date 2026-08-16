"""Lane A + Lane B for gemma-4-31B-it on the frozen 51-prompt set.

Lane A: identical scoring to lane_a_baseline.py (restricted softmax over the
answer letters, crossovers, IIA odds, compliance).

Lane B: layerwise readouts on the same forward passes —
  - plain logit lens: lm_head(final_norm(h_l))
  - Jacobian lens transfer test: the only available gemma-4 lens was fit on
    the BASE model (google/gemma-4-31B). We apply it to the instruct model
    anyway and let the deepest-layer self-check quantify the mismatch. If the
    self-check fails, that IS the result (base-fit lens does not transfer to
    the instruct model), and only the plain-lens column is interpreted.

Model on GPUs 0,2,3 (NNsight, fp16, as in all previous 31B runs); lens + a
copy of final_norm/lm_head weights on GPU 1 for the readout arithmetic.
"""
import csv
import json

import numpy as np
import torch

LENS_PT = ("/tmp/claude-1000/-home-cho-Documents-pressuretest-PressureTest/"
           "020d87ab-073f-4e02-9c74-4d7bb3738694/scratchpad/g4_31b_jlens.pt")
OUT = "analysis/jspace"


def main():
    from nnsight import VisionLanguageModel
    from transformers import AutoTokenizer

    m = VisionLanguageModel("google/gemma-4-31B-it", device_map="auto",
                            torch_dtype=torch.float16, dispatch=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    lm = m.model.language_model
    layers = lm.layers
    L = len(layers)

    rd = "cuda:1"                     # readout device
    normw = lm.norm.weight.detach().to(rd).float()
    headw = m.lm_head.weight.detach().to(rd)          # [vocab, d] fp16
    eps = 1e-6

    def read(h):                       # h: [d] float32 on rd
        x = h * torch.rsqrt(h.pow(2).mean() + eps)
        x = x * normw
        # fp16 matvec against the head (a fp32 copy would cost 5.25 GB per call)
        return (headw @ x.to(headw.dtype)).float()

    lens = torch.load(LENS_PT, map_location="cpu", weights_only=False)
    J = {l: v for l, v in lens["J"].items()}
    src = sorted(J)
    print(f"model {L} layers | lens source layers {src[0]}..{src[-1]} "
          f"d={lens.get('d_model')}")
    Jg = {l: J[l].to(rd) for l in src}                # 3.4GB fp16 on GPU1

    ids = {}
    for Lt in "ABC":
        for form in (Lt, " " + Lt):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                ids[form] = e[0]

    def restricted(lg, letters):
        lets = {Lt: max([ids[f] for f in (Lt, " " + Lt) if f in ids],
                        key=lambda t: lg[t].item()) for Lt in letters}
        z = torch.tensor([lg[lets[Lt]] for Lt in letters])
        return torch.softmax(z, 0).numpy()

    prompts = [json.loads(x) for x in open(f"{OUT}/prompts.jsonl")]
    rows, agree_j, agree_p, selfcheck = [], [[] for _ in range(L)], [[] for _ in range(L)], []
    diss_j, diss_p = [[] for _ in range(L)], [[] for _ in range(L)]

    for r in prompts:
        text = r["prompt"]             # raw text, no chat template (kit spec)
        saved = []
        with m.trace(text):
            for layer in layers:
                saved.append(layer.output.save())
            true_logits = m.lm_head.output.save()
        hs = []
        for sv in saved:
            h = sv[0] if isinstance(sv, (tuple, list)) else sv
            hs.append((h[0, -1, :] if h.dim() == 3 else h[-1, :]).detach()
                      .to(rd).float())
        tl = true_logits
        tl = tl[0] if isinstance(tl, (tuple, list)) else tl
        tl = (tl[0, -1, :] if tl.dim() == 3 else tl[-1, :]).float().cpu()

        letters = list(r["letter_map"])
        p_true = restricted(tl, letters)
        final_choice = letters[int(np.argmax(p_true))]
        top1 = tok.decode([int(tl.argmax())])
        row = dict(condition_id=r["condition_id"], scenario=r["scenario"],
                   variant=r["variant"], sweep=r["sweep"],
                   top1_token=repr(top1),
                   top1_is_expected_letter=top1.strip() in letters)
        target_roles = {"risky", "buy", "fund"}
        for Lt, pi in zip(letters, p_true):
            row[f"p_{Lt}"] = float(pi)
            row[f"role_{Lt}"] = r["letter_map"][Lt]
        row["p_target"] = float(sum(pi for Lt, pi in zip(letters, p_true)
                                    if r["letter_map"][Lt] in target_roles))
        rows.append(row)

        if r["variant"] != "dominated_third":
            is_diss = (r["scenario"] == "purchase_abstract_good"
                       and r["variant"] == "order_swap")
            with torch.no_grad():
                for l in range(L):
                    lg_p = read(hs[l]).cpu()
                    pj_ok = l in Jg
                    if pj_ok:
                        lg_j = read((Jg[l].float() @ hs[l])).cpu()
                    pp = restricted(lg_p, letters)
                    agree_p[l].append(letters[int(np.argmax(pp))] == final_choice)
                    if pj_ok:
                        pj = restricted(lg_j, letters)
                        agree_j[l].append(letters[int(np.argmax(pj))] == final_choice)
                        if l == src[-1]:
                            selfcheck.append(float(np.abs(pj - p_true).max()))
                        if is_diss:
                            fund = next(k for k, v in r["letter_map"].items() if v == "fund")
                            diss_j[l].append(letters[int(np.argmax(pj))] == fund)
                    if is_diss:
                        fund = next(k for k, v in r["letter_map"].items() if v == "fund")
                        diss_p[l].append(letters[int(np.argmax(pp))] == fund)
        print(f"  {r['condition_id']:<46} p_target={row['p_target']:.3f} "
              f"top1={row['top1_token']}")

    with open(f"{OUT}/results_lane_a_31b.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r_ in rows for k in r_}))
        w.writeheader(); w.writerows(rows)

    def crossing(xs, ps):
        for i in range(len(ps) - 1):
            if (ps[i] - .5) * (ps[i + 1] - .5) <= 0 and ps[i] != ps[i + 1]:
                t = (.5 - ps[i]) / (ps[i + 1] - ps[i])
                return xs[i] + t * (xs[i + 1] - xs[i])
        return None

    print("\n=== crossovers (31B) ===")
    for scen in sorted({r["scenario"] for r in rows}):
        for var in sorted({r["variant"] for r in rows if r["scenario"] == scen}):
            sub = sorted([r for r in rows if r["scenario"] == scen
                          and r["variant"] == var and r["sweep"] is not None],
                         key=lambda r: r["sweep"])
            if len(sub) < 2:
                continue
            ps = [r["p_target"] for r in sub]
            th = crossing([r["sweep"] for r in sub], ps)
            print(f"  {scen}/{var}: theta={th}  p {min(ps):.3f}-{max(ps):.3f}")

    print("\n=== IIA (31B) ===")
    for r3 in [r for r in rows if r["variant"] == "dominated_third"]:
        base = next(r for r in rows if r["scenario"] == r3["scenario"]
                    and r["variant"] == "base" and r["sweep"] == r3["sweep"])
        def odds(row):
            pr = sum(v for k, v in row.items() if k.startswith("p_")
                     and k != "p_target" and row.get("role_" + k[2:]) == "risky")
            ps_ = sum(v for k, v in row.items() if k.startswith("p_")
                      and k != "p_target" and row.get("role_" + k[2:]) == "safe")
            return pr / max(ps_, 1e-9)
        print(f"  sweep={r3['sweep']}: odds2={odds(base):.3f} odds3={odds(r3):.3f} "
              f"ratio={odds(r3)/max(odds(base),1e-9):.3f} p_C={r3.get('p_C'):.4f}")

    print(f"\nlens self-check at L{src[-1]} (BASE lens on IT model): "
          f"mean max|dp| = {np.mean(selfcheck):.4f}  "
          f"({'TRANSFERS' if np.mean(selfcheck) < 0.05 else 'DOES NOT TRANSFER'})")
    print(f"\n{'L':>3} {'J-lens agree':>13} {'plain agree':>12} "
          f"{'Jdiss->content':>15} {'Pdiss->content':>15}")
    for l in range(0, L, 2):
        aj = np.mean(agree_j[l]) if agree_j[l] else float("nan")
        dj = np.mean(diss_j[l]) if diss_j[l] else float("nan")
        dp = np.mean(diss_p[l]) if diss_p[l] else float("nan")
        print(f"{l:>3} {aj:>13.2f} {np.mean(agree_p[l]):>12.2f} "
              f"{dj:>15.2f} {dp:>15.2f}")
    np.save(f"{OUT}/lane_b_agree_31b.npy",
            np.array([[np.mean(agree_j[l]) if agree_j[l] else np.nan,
                       np.mean(agree_p[l])] for l in range(L)]))


if __name__ == "__main__":
    main()
