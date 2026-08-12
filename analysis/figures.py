"""Publication figures (SPEC §48-§49). All plots carry model name, n, and
per-point raw counts where practical."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def primary_figure(df: pd.DataFrame, out_path: str | Path,
                   domains: list[str], personas: list[str] | None = None,
                   model_name: str = "") -> Path:
    """Decision curves: x = delta (cost of selecting the lower-risk option),
    y = P(choose lower-risk option), one line per domain, one panel per persona."""
    personas = personas or sorted(df["persona"].dropna().unique())
    ni = df[(df["family"].isin(["near_indifference"])) &
            (df["parse_status"] == "ok")].copy()
    ni["p_safer"] = 1 - ni["p_choose_a"]

    fig, axes = plt.subplots(1, len(personas), figsize=(4.2 * len(personas), 3.6),
                             sharey=True, squeeze=False)
    for ax, persona in zip(axes[0], personas):
        sub = ni[ni["persona"] == persona]
        for domain in domains:
            g = (sub[sub["domain"] == domain]
                 .groupby("delta")["p_safer"].agg(["mean", "count"]).reset_index())
            ax.plot(g["delta"], g["mean"], marker="o", label=domain)
            for _, row in g.iterrows():
                ax.annotate(f"n={int(row['count'])}", (row["delta"], row["mean"]),
                            fontsize=6, alpha=0.6,
                            textcoords="offset points", xytext=(3, 3))
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_title(f"persona {persona}")
        ax.set_xlabel("cost of selecting lower-risk option (delta)")
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("P(select lower-risk option)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Decision curves by domain — {model_name} "
                 f"(N={len(ni)} trials)", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
