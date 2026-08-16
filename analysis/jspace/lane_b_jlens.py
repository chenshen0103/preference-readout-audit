"""Lane B: layerwise readout of the frozen prompt set through the pre-fit
Jacobian lens (Anthropic jlens, fit by Neuronpedia on wikitext).

Lens format: J[l] is a d_model x d_model matrix approximating
d h_final / d h_l averaged over prompts. Readout convention (validated
empirically at the deepest source layer before use):

    h_hat_final(l) = J[l] @ h_l          # linear approx to the final state
    logits(l)      = lm_head(final_norm(h_hat_final(l)))

Also computes the PLAIN logit lens (final_norm(h_l) -> lm_head) on the same
states, so the two readouts can be compared per layer on identical data.

Per layer we report, over the 51-prompt set:
  - agreement of the restricted A/B argmax with the model's final choice
  - same, for the plain lens
  - dissociation set (letter default vs content): which side mid-layers read
"""
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LENS_PT = ("/tmp/claude-1000/-home-cho-Documents-pressuretest-PressureTest/"
           "020d87ab-073f-4e02-9c74-4d7bb3738694/scratchpad/g3_1b_jlens.pt")
MODEL = "unsloth/gemma-3-1b-it"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to("cuda").eval()
    lens = torch.load(LENS_PT, map_location="cuda", weights_only=False)
    J = {l: m.float() for l, m in lens["J"].items()}
    L = model.config.num_hidden_layers
    norm = model.model.norm
    head = model.lm_head
    print(f"{L} layers, lens covers source layers {sorted(J)[:3]}..{max(J)}")

    ids = {}
    for Lt in "AB":
        for form in (Lt, " " + Lt):
            e = tok.encode(form, add_special_tokens=False)
            if len(e) == 1:
                ids[form] = e[0]

    prompts = [json.loads(x) for x in open("analysis/jspace/prompts.jsonl")
               if json.loads(x)["variant"] != "dominated_third"]
    print(f"{len(prompts)} two-option prompts")

    def restricted(lg, letters):
        lets = {}
        for Lt in letters:
            cands = [ids[f] for f in (Lt, " " + Lt) if f in ids]
            lets[Lt] = max(cands, key=lambda t: lg[t].item())
        z = torch.tensor([lg[lets[Lt]] for Lt in letters])
        return torch.softmax(z, 0).numpy()

    per_layer_hits_j, per_layer_hits_p = [[] for _ in range(L)], [[] for _ in range(L)]
    diss_j, diss_p = [[] for _ in range(L)], [[] for _ in range(L)]
    val_gap = []

    for r in prompts:
        enc = tok(r["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            true_lg = out.logits[0, -1, :].float()
        letters = list(r["letter_map"])
        p_true = restricted(true_lg, letters)
        final_choice = letters[int(np.argmax(p_true))]
        # dissociation set: final answer's ROLE differs between base and its
        # own letter -> use abstract_good order_swap where letter-A default won
        is_diss = (r["scenario"] == "purchase_abstract_good"
                   and r["variant"] == "order_swap")

        for l in range(L):
            h = out.hidden_states[l + 1][0, -1, :].float()   # after block l
            with torch.no_grad():
                lg_j = head(norm((J[l] @ h if l in J else h)
                                 .unsqueeze(0).to(norm.weight.dtype)))[0].float()
                lg_p = head(norm(h.unsqueeze(0).to(norm.weight.dtype)))[0].float()
            pj = restricted(lg_j, letters)
            pp = restricted(lg_p, letters)
            per_layer_hits_j[l].append(letters[int(np.argmax(pj))] == final_choice)
            per_layer_hits_p[l].append(letters[int(np.argmax(pp))] == final_choice)
            if is_diss:
                # does the readout back the CONTENT-implied option (fund)?
                fund_letter = next(Lt for Lt, role in r["letter_map"].items()
                                   if role == "fund")
                diss_j[l].append(letters[int(np.argmax(pj))] == fund_letter)
                diss_p[l].append(letters[int(np.argmax(pp))] == fund_letter)
            if l == max(J):
                val_gap.append(float(np.abs(pj - p_true).max()))

    print(f"\nlens self-check at deepest source layer L{max(J)}: "
          f"mean |p_jlens - p_true| = {np.mean(val_gap):.4f} "
          f"(small = our application convention is right)")
    print(f"\n{'L':>3} {'J-lens agree':>13} {'plain agree':>12} "
          f"{'J diss->content':>16} {'plain diss->content':>20}")
    for l in range(L):
        dj = np.mean(diss_j[l]) if diss_j[l] else float("nan")
        dp = np.mean(diss_p[l]) if diss_p[l] else float("nan")
        print(f"{l:>3} {np.mean(per_layer_hits_j[l]):>13.2f} "
              f"{np.mean(per_layer_hits_p[l]):>12.2f} {dj:>16.2f} {dp:>20.2f}")

    np.save("analysis/jspace/lane_b_agree.npy",
            np.array([[np.mean(per_layer_hits_j[l]),
                       np.mean(per_layer_hits_p[l])] for l in range(L)]))


if __name__ == "__main__":
    main()
