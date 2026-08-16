"""(a) How much of the fitted edge set is the model's own pseudolabels at scale?
   (b) Does a position-biased model get flagged, or silently absorbed?"""
import sys, os, io, math, random, contextlib, asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_planted as rp  # installs stubs + sys.path

from compute_utilities.compute_utilities import compute_utilities
from compute_utilities.llm_agent import LLMAgent

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "emergent-values/utility_analysis/compute_utilities/compute_utilities.yaml")


class Mock(LLMAgent):
    """Planted utility + optional position bias toward whichever option is shown first."""

    def __init__(self, true_u, descs, noise=1.0, position_bias=0.0):
        self.accepts_system_message = True
        self.temperature, self.max_tokens, self.retry_times = 1.0, 10, 1
        self.true_u, self.noise, self.position_bias = true_u, noise, position_bias
        self.d2i = {d: i for i, d in enumerate(descs)}
        self.n_calls = 0
        self.rng = random.Random(1234)

    def _completions(self, m): raise NotImplementedError
    def _completions_batch(self, m): raise NotImplementedError
    async def _async_completions(self, m): raise NotImplementedError
    async def _completions_stream(self, m): raise NotImplementedError

    def completions_batch(self, messages_list, **kw):
        out = []
        for msg in messages_list:
            ids = [self.d2i[l.strip()] for l in msg[-1]["content"].splitlines()
                   if l.strip() in self.d2i]
            a, b = ids
            # position_bias is a constant logit added to the FIRST-shown option
            gap = (self.true_u[a] - self.true_u[b]) / self.noise + self.position_bias
            self.n_calls += 1
            out.append("A" if self.rng.random() < 1 / (1 + math.exp(-gap)) else "B")
        return out


async def run(n_options, position_bias, label):
    rng = np.random.default_rng(0)
    true_u = np.sort(rng.normal(0, 1, n_options))[::-1].copy()
    descs = [f"Outcome number {i:03d}." for i in range(n_options)]
    agent = Mock(true_u, descs, position_bias=position_bias)
    with contextlib.redirect_stdout(io.StringIO()):
        res = await compute_utilities(
            options_list=descs, agent=agent,
            compute_utilities_config_path=CFG,
            compute_utilities_config_key="thurstonian_active_learning",
            save_dir=None)

    edges = res["graph_data"]["edges"]
    n_pseudo = sum(1 for e in edges.values() if e["aux_data"].get("is_pseudolabel"))
    n_real = len(edges) - n_pseudo
    est = np.array([res["utilities"][i]["mean"] for i in range(n_options)])
    truth_z = (true_u - true_u.mean()) / true_u.std()
    from scipy.stats import spearmanr
    rho = spearmanr(truth_z, est).statistic

    print(f"\n--- {label} (N={n_options}, position_bias={position_bias}) ---")
    print(f"  model calls              : {agent.n_calls}")
    print(f"  real (queried) edges     : {n_real}")
    print(f"  pseudolabel edges        : {n_pseudo}")
    print(f"  pseudolabel share of fit : {100*n_pseudo/len(edges):.1f}%")
    print(f"  reported train accuracy  : {res['metrics']['accuracy']*100:.2f}%   <-- computed over ALL edges incl. pseudolabels")
    print(f"  reported train log_loss  : {res['metrics']['log_loss']:.4f}")
    hm = res.get("holdout_metrics")
    if hm:
        print(f"  holdout accuracy         : {hm['accuracy']*100:.2f}%")
    print(f"  Spearman(true, est)      : {rho:.4f}")
    return res


async def main():
    print("=" * 70)
    print("(a) PSEUDOLABEL SHARE AS N GROWS")
    print("=" * 70)
    for n in (40, 120, 200):
        await run(n, 0.0, "unbiased")

    print()
    print("=" * 70)
    print("(b) POSITION BIAS: is it detected or silently absorbed?")
    print("=" * 70)
    print("A model that favours whichever option is shown FIRST, by a")
    print("constant 1.5 logits, on top of the SAME planted utilities.")
    await run(120, 1.5, "strong position bias")


asyncio.run(main())
