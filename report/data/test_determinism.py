"""Probe reproducibility of the emergent-values Thurstonian fit + active learning."""
import sys, os, types, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_planted  # noqa: F401  (installs the import stubs + path)

from compute_utilities.compute_utilities import PreferenceGraph
from compute_utilities.utility_models.thurstonian.utils import fit_thurstonian_model
from compute_utilities.utility_models.thurstonian.thurstonian_active_learning import (
    generate_additional_pairs,
)

# ---------- 1. Is fit_thurstonian_model deterministic on a FIXED graph? ----
N = 30
rng = np.random.default_rng(0)
true_u = rng.normal(0, 1, N)
options = [{"id": i, "description": f"opt{i}"} for i in range(N)]


def build_graph():
    g = PreferenceGraph(options, holdout_fraction=0.0, seed=42)
    data = []
    for i in range(N):
        for j in range(i + 1, N):
            p = 1 / (1 + np.exp(-(true_u[i] - true_u[j])))
            data.append({"option_A": options[i], "option_B": options[j],
                         "probability_A": float(p), "aux_data": {}})
    g.add_edges(data)
    return g


print("=" * 66)
print("1. fit_thurstonian_model on an IDENTICAL graph, 3 repeats")
print("=" * 66)
fits = []
for t in range(3):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        u, ll, acc = fit_thurstonian_model(build_graph(), num_epochs=1000, learning_rate=0.01)
    means = np.array([u[i]["mean"] for i in range(N)])
    variances = np.array([u[i]["variance"] for i in range(N)])
    fits.append((means, variances, ll))
    print(f"  run {t}: log_loss={ll:.6f}  mu[0]={means[0]:+.6f}  var[0]={variances[0]:.6f}")

d_mu = np.abs(fits[0][0] - fits[1][0]).max()
d_var = np.abs(fits[0][1] - fits[1][1]).max()
print(f"  max |Δmu| across runs  : {d_mu:.6e}")
print(f"  max |Δvar| across runs : {d_var:.6e}")
print(f"  VERDICT: {'DETERMINISTIC' if d_mu < 1e-9 else 'NON-DETERMINISTIC (no torch seed)'}")

# ---------- 2. Does generate_additional_pairs clobber global RNG state? ----
print()
print("=" * 66)
print("2. Global RNG hygiene in generate_additional_pairs(seed=None)")
print("=" * 66)
utilities = {i: {"mean": float(true_u[i]), "variance": 1.0} for i in range(N)}
avail = {(i, j) for i in range(N) for j in range(i + 1, N)}

random.seed(12345)
np.random.seed(12345)
before_py = random.random()
before_np = float(np.random.rand())

random.seed(12345)
np.random.seed(12345)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    generate_additional_pairs(utilities, set(), avail, 20, 10.0, 20.0, seed=None)
after_py = random.random()
after_np = float(np.random.rand())

print(f"  caller's next random.random() without the call : {before_py:.6f}")
print(f"  caller's next random.random() after   the call : {after_py:.6f}")
print(f"  caller's next np.random.rand() without / after : {before_np:.6f} / {after_np:.6f}")
print(f"  VERDICT: {'RNG PRESERVED' if before_py == after_py else 'CALLER RNG CLOBBERED (random.seed(None) reseeds from OS entropy)'}")

# ---------- 3. Are the selected active-learning pairs stable? -------------
print()
print("=" * 66)
print("3. Active-learning edge selection with default seed=None, 2 repeats")
print("=" * 66)
sels = []
for t in range(2):
    with contextlib.redirect_stdout(io.StringIO()):
        sel = generate_additional_pairs(utilities, set(), avail, 20, 10.0, 20.0, seed=None)
    sels.append(set(sel))
    print(f"  run {t}: {len(sel)} pairs, first 3 = {sorted(sel)[:3]}")
overlap = len(sels[0] & sels[1])
print(f"  overlap between runs: {overlap}/{len(sels[0])}")
print(f"  VERDICT: {'STABLE' if sels[0] == sels[1] else 'UNSTABLE — same config gives a different experiment'}")
