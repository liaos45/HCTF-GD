"""Journal-size fixed-checkpoint N×F×C Validation effect plot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE
PRIMARY = BASE / "data/factorial_bootstrap_primary.csv"
RUN = BASE / "data/factorial_run_effects.csv"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "figure.facecolor": "white", "axes.facecolor": "white",
})


def main() -> None:
    summary = pd.read_csv(PRIMARY).set_index("contrast")
    run = pd.read_csv(RUN)
    names = ("N_main", "F_main", "C_main", "N_F_interaction", "N_C_interaction",
             "F_C_interaction", "N_F_C_interaction")
    labels = ("Node residual", "Forest messages", "Community context", "Node × forest",
              "Node × context", "Forest × context", "Three-way interaction")
    rows = summary.loc[list(names)]
    center = rows.equal_domain_mean_effect_pct.to_numpy(float)
    lo = rows.bootstrap_95_low_pct.to_numpy(float)
    hi = rows.bootstrap_95_high_pct.to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.95), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.7, 1]})  # 160 mm final display
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color="#667085", linewidth=0.8)
    ax.errorbar(center, y, xerr=np.vstack((center-lo, hi-center)), fmt="o",
                color="#8470A7", ecolor="#8470A7", elinewidth=1.1,
                markersize=4, capsize=2.5, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_title("a  Marginal and interaction effects", loc="left", fontweight="bold")
    ax.set_xlabel("Pair-normalized Stress effect (%)")
    extent = max(abs(lo.min()), abs(hi.max()))
    ax.set_xlim(min(lo.min()-0.12*extent, -0.005), max(hi.max()+0.12*extent, 0.005))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    ax.tick_params(axis="both", length=2.5, width=0.7)

    ax = axes[1]
    subset = run[(run.metric == "graph_stress") & (run.contrast == "111_vs_000")]
    domains = ("enzymes", "fb15k237", "sbm")
    palette = {"enzymes": "#507E9A", "fb15k237": "#8470A7", "sbm": "#C18A52"}
    for index, domain in enumerate(domains):
        y0 = len(domains)-1-index
        vals = subset[subset.domain == domain].sort_values("seed").effect_pct.to_numpy(float)
        if len(vals) != 3:
            raise RuntimeError(f"expected 3 seed effects: {domain}")
        ax.scatter(vals, y0 + np.array([-0.1, 0, 0.1]), s=16, facecolors="white",
                   edgecolors=palette[domain], linewidths=0.8, zorder=3)
        ax.scatter([vals.mean()], [y0], marker="D", s=23, color=palette[domain], zorder=4)
    ax.axvline(0, color="#667085", linewidth=0.8)
    ax.set_yticks([2, 1, 0], ["ENZYMES", "FB15K237", "SBM"])
    ax.set_title("b  All-on versus all-off", loc="left", fontweight="bold")
    ax.set_xlabel("Pair-normalized Stress gain (%)")
    xmin, xmax = subset.effect_pct.min(), subset.effect_pct.max()
    margin = 0.12*max(xmax-xmin, 0.002)
    ax.set_xlim(xmin-margin, xmax+margin)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    ax.tick_params(axis="both", length=2.5, width=0.7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "factorial_validation_effects.pdf", bbox_inches="tight")
    fig.savefig(OUT / "factorial_validation_effects.svg", bbox_inches="tight")
    fig.savefig(OUT / "factorial_validation_effects.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    (OUT / "factorial_validation_effects_sources.json").write_text(json.dumps({
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (PRIMARY, RUN)
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
