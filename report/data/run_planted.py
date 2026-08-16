"""Run emergent-values' compute_utilities end-to-end against a mock model with a
KNOWN utility function, and check whether the pipeline recovers it.

Heavy optional deps (vllm, google-generativeai, fireworks, litellm, PIL, ...) are
stubbed out so the core numerics can be exercised without a GPU box.
"""
import sys, types, os, asyncio, json, math, random
import numpy as np

# ---- stub the unavailable heavy imports in llm_agent.py -------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()

_stub("openai", AsyncOpenAI=_Any, OpenAI=_Any)
sys.modules["openai"].__spec__ = None
_g = _stub("google")
_gg = _stub("google.generativeai", configure=_Any, GenerativeModel=_Any)
_stub("google.generativeai.types", HarmCategory=_Any, HarmBlockThreshold=_Any)
_g.generativeai = _gg
_stub("imghdr", what=lambda *a, **k: "png")
_stub("anthropic", Anthropic=_Any, AsyncAnthropic=_Any)
_stub("dotenv", load_dotenv=lambda *a, **k: None)
_stub("fireworks")
_stub("fireworks.client", Fireworks=_Any, AsyncFireworks=_Any)
sys.modules["fireworks"].client = sys.modules["fireworks.client"]
_stub("huggingface_hub", login=lambda *a, **k: None)
_stub("PIL", Image=_Any)
_stub("vllm", LLM=_Any, SamplingParams=_Any)
_stub("litellm", acompletion=_Any)
_stub("transformers", AutoTokenizer=_Any, AutoModelForCausalLM=_Any, AutoProcessor=_Any)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "emergent-values", "utility_analysis"))

from compute_utilities.compute_utilities import compute_utilities  # noqa: E402
from compute_utilities.llm_agent import LLMAgent  # noqa: E402

# ---- mock model with a planted utility function --------------------------
N_OPTIONS = int(os.environ.get("N_OPTIONS", 40))
NOISE = float(os.environ.get("NOISE", 1.0))  # logistic temperature on utility gap
SEED = 0

rng_true = np.random.default_rng(SEED)
TRUE_U = np.sort(rng_true.normal(0, 1, N_OPTIONS))[::-1].copy()  # option i has utility TRUE_U[i]
OPTIONS = [f"Outcome number {i:03d}." for i in range(N_OPTIONS)]
DESC_TO_ID = {d: i for i, d in enumerate(OPTIONS)}


class PlantedAgent(LLMAgent):
    """Answers A/B according to a logistic model on the planted utility gap."""

    def __init__(self):
        self.accepts_system_message = True
        self.temperature = 1.0
        self.max_tokens = 10
        self.retry_times = 1
        self.n_calls = 0
        self.rng = random.Random(1234)

    # --- required abstract methods
    def _completions(self, messages): raise NotImplementedError
    def _completions_batch(self, messages): raise NotImplementedError
    async def _async_completions(self, messages): raise NotImplementedError
    async def _completions_stream(self, messages): raise NotImplementedError

    def _parse_prompt(self, prompt):
        """Recover which option ids are in slot A and slot B."""
        ids = []
        for line in prompt.splitlines():
            line = line.strip()
            if line in DESC_TO_ID:
                ids.append(DESC_TO_ID[line])
        assert len(ids) == 2, f"could not parse prompt:\n{prompt}"
        return ids[0], ids[1]

    def completions_batch(self, messages_list, **kw):
        out = []
        for msg in messages_list:
            prompt = msg[-1]["content"]
            a, b = self._parse_prompt(prompt)
            gap = (TRUE_U[a] - TRUE_U[b]) / NOISE
            p_a = 1.0 / (1.0 + math.exp(-gap))
            self.n_calls += 1
            out.append("A" if self.rng.random() < p_a else "B")
        return out


async def main():
    agent = PlantedAgent()
    results = await compute_utilities(
        options_list=OPTIONS,
        agent=agent,
        compute_utilities_config_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "emergent-values/utility_analysis/compute_utilities/compute_utilities.yaml"),
        compute_utilities_config_key="thurstonian_active_learning",
        save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "planted_out"),
        save_suffix="planted",
    )
    est = np.array([results["utilities"][i]["mean"] for i in range(N_OPTIONS)])
    var = np.array([results["utilities"][i]["variance"] for i in range(N_OPTIONS)])

    from scipy.stats import spearmanr, pearsonr
    truth_z = (TRUE_U - TRUE_U.mean()) / TRUE_U.std()
    rho = spearmanr(truth_z, est).statistic
    r = pearsonr(truth_z, est).statistic
    rmse = float(np.sqrt(np.mean((truth_z - est) ** 2)))

    print("\n" + "=" * 62)
    print("PLANTED-UTILITY RECOVERY")
    print("=" * 62)
    print(f"options            : {N_OPTIONS}")
    print(f"model API calls    : {agent.n_calls}")
    print(f"edges in graph     : {len(results['graph_data']['edges'])}")
    print(f"train log_loss     : {results['metrics']['log_loss']:.4f}")
    print(f"train accuracy     : {results['metrics']['accuracy']*100:.2f}%")
    hm = results.get("holdout_metrics")
    if hm:
        print(f"holdout log_loss   : {hm['log_loss']:.4f}")
        print(f"holdout accuracy   : {hm['accuracy']*100:.2f}%")
    print(f"Spearman(true,est) : {rho:.4f}")
    print(f"Pearson (true,est) : {r:.4f}")
    print(f"RMSE vs true z     : {rmse:.4f}")
    print(f"est variance range : [{var.min():.4g}, {var.max():.4g}]")
    json.dump({"true": TRUE_U.tolist(), "est": est.tolist(), "var": var.tolist(),
               "spearman": float(rho), "pearson": float(r), "rmse": rmse,
               "n_calls": agent.n_calls},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "planted_out", "recovery.json"), "w"), indent=2)


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "planted_out"), exist_ok=True)
    asyncio.run(main())
