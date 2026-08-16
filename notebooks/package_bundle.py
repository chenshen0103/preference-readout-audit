"""Package the validity notebook as a self-contained shareable bundle.

- Executes every code cell headlessly, embedding stdout and PNG figures into
  the .ipynb outputs, so the notebook is fully readable WITHOUT running it.
- Copies every data file the notebook loads into the bundle, preserving the
  repo-relative layout, so it is also fully RE-RUNNABLE from the bundle root
  with only numpy + matplotlib installed.
"""
import base64
import io
import json
import shutil
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "notebooks" / "preference_measurement_validity_bundle"
DATA_FILES = [
    "analysis/validity/records.json",
    "analysis/validity/probe_results.npy",
    "analysis/validity/transfer_fixed.npy",
    "analysis/boundary/meta.json",
    "analysis/boundary/probe_scores.npy",
    "analysis/veto/meta.json",
    "analysis/veto/probe_scores.npy",
    "analysis/veto/reversed_listing.json",
    "analysis/veto/critical_stability_results.json",
    "analysis/sweep/meta.json",
    "analysis/pilot_geo/behavioral_results.csv",
    "analysis/decision_trajectories.json",
    "analysis/layer_tokens.json",
    "report/data/recovery.json",
    "report/data/runA_results_utilities_gemma4-local.json",
    "report/data/runB_results_utilities_gemma4-local.json",
    "report/data/runC_results_utilities_gemma4-local-thinking.json",
    "report/data/runL1_results_utilities_gemma4-local.json",
    "report/data/runL2_results_utilities_gemma4-local.json",
]


SCRIPT_FILES = [
    # mechanistic experiment scripts (produce the data files above; need GPU+model)
    "analysis/validity_diagnostics.py",
    "analysis/probe_residuals.py",
    "analysis/probe_transfer_fixed.py",
    "analysis/decision_trajectory.py",
    "analysis/trajectory_controls.py",
    "analysis/layer_tokens.py",
    "analysis/boundary_sweep.py",
    "analysis/boundary_probe.py",
    "analysis/veto_tests.py",
    "analysis/layout_swap_test.py",
    "analysis/veto_probe_scores.py",
    "analysis/critical_stability_test.py",
    "analysis/dilemma_sweep.py",
    "analysis/sweep_analysis.py",
    "analysis/pilot_geo.py",
    "analysis/pilot_geo_plots.py",
    "analysis/make_figures.py",
    # behavioral-pipeline validation scripts (run against emergent-values @5e5966d)
    "report/data/run_planted.py",
    "report/data/test_determinism.py",
    "report/data/test_scale_and_bias.py",
    "report/data/compare_reasoning.py",
]

def execute_and_embed(nb):
    """Run code cells in order; attach stream + image outputs."""
    g = {}
    import os
    os.chdir(REPO)                       # data paths resolve against repo
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        outputs, figs = [], []

        def show(*a, **k):
            for n in plt.get_fignums():
                buf = io.BytesIO()
                plt.figure(n).savefig(buf, format="png", dpi=140,
                                      bbox_inches="tight")
                figs.append(base64.b64encode(buf.getvalue()).decode())
            plt.close("all")

        plt.show = show
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(src, f"cell{i}", "exec"), g)
        for f in figs:
            outputs.append({"output_type": "display_data",
                            "data": {"image/png": f}, "metadata": {}})
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": buf.getvalue().splitlines(keepends=True)})
        c["outputs"] = outputs
        c["execution_count"] = i
        print(f"  cell {i}: {len(figs)} figure(s), "
              f"{len(buf.getvalue())} chars stdout")
    return nb


def main():
    nb = json.load(open(REPO / "notebooks/preference_measurement_validity.ipynb"))
    nb = execute_and_embed(nb)

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    for rel in DATA_FILES + SCRIPT_FILES:
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    json.dump(nb, open(BUNDLE / "preference_measurement_validity.ipynb", "w"),
              indent=1)
    # repo copy also gets the embedded outputs (renders on GitHub/VSCode)
    json.dump(nb, open(REPO / "notebooks/preference_measurement_validity.ipynb",
                       "w"), indent=1)
    (BUNDLE / "README.txt").write_text(
        "Preference Measurement Validity - notebook bundle\n"
        "=================================================\n\n"
        "Just READ:  open preference_measurement_validity.ipynb - all figures\n"
        "and printed numbers are already embedded; no execution needed.\n\n"
        "RE-RUN (optional):\n"
        "  pip install numpy matplotlib jupyter\n"
        "  jupyter lab preference_measurement_validity.ipynb\n"
        "Run from this folder (the notebook resolves data paths relative to\n"
        "its working directory: ./analysis and ./report/data).\n\n"
        "Data files are exactly the saved outputs of the analyses described\n"
        "inside; nothing here re-runs model inference.\n\n"
        "EXPERIMENT SCRIPTS are included for the record (analysis/*.py and\n"
        "report/data/*.py): they are what PRODUCED the data files. Re-running\n"
        "them needs the full setup (4x V100, google/gemma-4-31B-it fp16 via\n"
        "NNsight; behavioral scripts additionally need the patched\n"
        "emergent-values working copy at upstream commit 5e5966d). The\n"
        "notebook itself never calls them.\n")

    zpath = REPO / "notebooks/preference_measurement_validity_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(BUNDLE.parent))
    print(f"\nbundle: {BUNDLE}")
    print(f"zip:    {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
