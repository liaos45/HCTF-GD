"""Evaluate seven visual endpoints that never enter the SafeBlend decision."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from analyze_hctf_adaptive_results import (
    HOLDOUT_METRICS, hierarchical_bootstrap, paired_tests, signed_gain,
)

ROOT = Path(__file__).resolve().parents[1]
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":8.,"axes.titlesize":9.,"axes.labelsize":8.,"xtick.labelsize":7.5,
    "ytick.labelsize":7.5,"legend.fontsize":7.5,"svg.fonttype":"none","pdf.fonttype":42})
DISPLAY = {
    "angular_resolution_deg": "Angular resolution",
    "angular_resolution_ratio": "Normalized angular resolution",
    "crossing_angle_deg": "Crossing angle",
    "crossing_angle_p05_deg": "Crossing angle (5th percentile)",
    "node_edge_clearance": "Node–edge clearance",
    "node_edge_clearance_p05": "Clearance (5th percentile)",
    "neighborhood_preservation_k5": "Neighborhood preservation",
}


def collect(root: Path):
    frames, selections = {}, []
    for path in sorted(root.glob("*/*_eval.csv")):
        identity = re.fullmatch(r"(.+)_(full_nodrop|node_only_nodrop)_s(\d+)", path.parent.name)
        if identity is None:
            continue
        dataset, variant, seed = identity.group(1), identity.group(2), int(identity.group(3))
        if "paired" in root.name and not (root / f"PAIRED_{dataset}_s{seed}.json").exists():
            continue
        frame = pd.read_csv(path)
        required = set(HOLDOUT_METRICS + ["method", "graph_id", "batch_id"])
        if not required.issubset(frame):
            raise ValueError(f"Missing holdout columns: {path}")
        for method, label in [("Current-Model-Baseline", "baseline"), ("HCTF-SafeBlend", "safe")]:
            part = frame.loc[frame.method == method].copy()
            if part.empty:
                raise ValueError(f"Missing {method}: {path}")
            frames[(dataset, variant, seed, label)] = part.rename(columns={"graph_id": "local_graph_id"})
        selection = pd.read_csv(path.with_name(f"{dataset}_selection.csv"))
        selection["variant"], selection["seed"] = variant, seed
        selections.append(selection)
    return frames, pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def matched(frames, comparison):
    pairs, rows = {}, []
    for dataset, variant, seed, label in sorted(frames):
        if label != "safe":
            continue
        if comparison == "safe_full_vs_safe_node":
            if variant != "full_nodrop":
                continue
            reference_key = (dataset, "node_only_nodrop", seed, "safe")
            if reference_key in frames:
                reference_baseline = frames[(dataset, "node_only_nodrop", seed, "baseline")].sort_values("local_graph_id")
                full_baseline = frames[(dataset, variant, seed, "baseline")].sort_values("local_graph_id")
                if not np.array_equal(reference_baseline[HOLDOUT_METRICS].to_numpy(), full_baseline[HOLDOUT_METRICS].to_numpy()):
                    raise ValueError(f"Unmatched baselines for {dataset} s{seed}; use shared-input paired evaluator.")
        else:
            if variant != comparison:
                continue
            reference_key = (dataset, variant, seed, "baseline")
        if reference_key not in frames:
            continue
        pair = frames[reference_key].merge(frames[(dataset, variant, seed, "safe")],
                                          on=["batch_id", "local_graph_id"],
                                          suffixes=("_node", "_full"), validate="one_to_one")
        pairs[(dataset, seed)] = pair
        for metric in HOLDOUT_METRICS:
            reference, trial = pair[f"{metric}_node"].to_numpy(float), pair[f"{metric}_full"].to_numpy(float)
            rows.append({"dataset": dataset, "seed": seed, "metric": metric,
                         "graphs": len(pair), "reference_mean": reference.mean(), "safe_mean": trial.mean(),
                         "gain_pct": float(signed_gain([reference.mean()], [trial.mean()], metric)[0]),
                         "graphs_improved_fraction": float(np.mean(trial > reference))})
    return pairs, pd.DataFrame(rows)


def plot(data, output):
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    for offset, comparison, color, label in [(-.13, "full_nodrop", "#7A5195", "Safe full vs backbone"),
                                             (.13, "node_only_nodrop", "#4C78A8", "Safe node-only vs backbone")]:
        part = data[(data.dataset == "ALL") & (data.comparison == comparison)].set_index("metric").loc[HOLDOUT_METRICS]
        y = np.arange(len(part)) + offset
        # Bootstrap percentile CIs are shown as segments; not forced to contain the estimate.
        ax.errorbar((part.ci_low_pct+part.ci_high_pct)/2,y,
            xerr=(part.ci_high_pct-part.ci_low_pct)/2,fmt="none",ecolor=color,linewidth=1.3)
        ax.scatter(part.mean_gain_pct, y, color=color, s=16, label=label, zorder=3)
    ax.axvline(0, color="#555555", linewidth=.8, linestyle="--")
    ax.set_yticks(np.arange(len(HOLDOUT_METRICS)), [DISPLAY[m] for m in HOLDOUT_METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Gain on unselected visual endpoints (%)")
    ax.set_title("Independent of selection rule; crossed seed/graph bootstrap (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2, linewidth=.5)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/hctf_safe_blend_paired_holdout")
    parser.add_argument("--output-dir", default="results/hctf_safe_holdout_analysis")
    parser.add_argument("--expected-runs", type=int, default=80)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    frames, selections = collect(ROOT / args.input_dir)
    count = len(frames) // 2
    if count != args.expected_runs and not args.allow_partial:
        raise RuntimeError(f"Expected {args.expected_runs} complete audits, found {count}")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    effects, intervals, tests = [], [], []
    for comparison in ["full_nodrop", "node_only_nodrop", "safe_full_vs_safe_node"]:
        pairs, effect = matched(frames, comparison)
        if not pairs:
            continue
        interval = hierarchical_bootstrap(pairs, args.bootstrap_repeats, 20260916, HOLDOUT_METRICS)
        test = paired_tests(effect)
        for frame in [effect, interval, test]:
            frame["comparison"] = comparison
        effects.append(effect); intervals.append(interval); tests.append(test)
    effects, intervals, tests = pd.concat(effects), pd.concat(intervals), pd.concat(tests)
    effects.to_csv(output / "paired_seed_effects.csv", index=False)
    intervals.to_csv(output / "crossed_bootstrap_ci.csv", index=False)
    tests.to_csv(output / "paired_wilcoxon_holm.csv", index=False)
    selections.to_csv(output / "audit_selections.csv", index=False)
    plot(intervals, output / "safe_holdout_forest")
    contract = {"claim": "Accepted layouts need external-to-selection evaluation; improvement is not guaranteed.",
                "endpoints": HOLDOUT_METRICS, "selection_endpoints_overlap": [],
                "bootstrap": "Crossed seeds and common graph identities; ALL equally weights fixed datasets.",
                "limitations": ["Metrics are not a human usability study", "Same benchmark pools informed method design",
                                "No-crossing angle is 90 degrees; no-angle constraints are 180 degrees and ratio one",
                                "Percentage gains use denominator floors; source CSV retains absolute means"],
                "audits": count}
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    selected = intervals[intervals.dataset == "ALL"]
    (output / "REPORT.md").write_text("# Unselected visual-endpoint audit\n\n" +
        f"Completed audits: {count}. None of these seven metrics entered acceptance.\n\n" +
        "These are independent endpoints, not independent data or a human task study.\n\n```text\n" +
        selected.to_string(index=False) + "\n```\n", encoding="utf-8")
    print(selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
