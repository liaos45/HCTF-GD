"""Frozen-protocol matched evaluation; existing test pools informed model design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CORE_METRICS = [
    "graph_stress",
    "edge_crossing",
    "edge_crossing_norm",
    "edge_length_variance",
    "edge_length_cv",
    "node_occlusion_rate",
    "community_compactness",
    "community_separation",
    "community_sep_comp_ratio",
    "community_silhouette",
]
HOLDOUT_METRICS = [
    "angular_resolution_deg",
    "angular_resolution_ratio",
    "crossing_angle_deg",
    "crossing_angle_p05_deg",
    "node_edge_clearance",
    "node_edge_clearance_p05",
    "neighborhood_preservation_k5",
]
METRICS = CORE_METRICS + HOLDOUT_METRICS
LOWER_IS_BETTER = set(CORE_METRICS[:7])
FLOORS = {
    "graph_stress": 0.01,
    "edge_crossing": 1.0,
    "edge_crossing_norm": 0.01,
    "edge_length_variance": 0.01,
    "edge_length_cv": 0.05,
    "node_occlusion_rate": 0.001,
    "community_compactness": 0.05,
    "community_separation": 0.05,
    "community_sep_comp_ratio": 0.05,
    "community_silhouette": 0.05,
    **{metric: 0.05 for metric in HOLDOUT_METRICS},
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def signed_gain(reference, trial, metric):
    reference = np.asarray(reference, dtype=float)
    trial = np.asarray(trial, dtype=float)
    difference = reference - trial if metric in LOWER_IS_BETTER else trial - reference
    return 100.0 * difference / np.maximum(np.abs(reference), FLOORS[metric])


def load_runs(
    output_dir: Path, allowed_variants: set[str]
) -> dict[tuple[str, str, int], pd.DataFrame]:
    statuses = []
    for path in sorted(output_dir.glob("run_status_*.csv")):
        statuses.append(pd.read_csv(path))
    if not statuses:
        raise FileNotFoundError(f"no run status files under {output_dir}")
    status = pd.concat(statuses, ignore_index=True)
    status = status[status["status"].isin(["SUCCESS", "SKIPPED_EXISTING"])]
    status = status.drop_duplicates(["dataset", "variant", "seed"], keep="last")
    frames = {}
    for row in status.itertuples(index=False):
        path = ROOT / row.eval_csv
        if not path.exists() or row.variant not in allowed_variants:
            continue
        frame = pd.read_csv(path)
        missing = set(METRICS + ["batch_id", "local_graph_id"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} misses {sorted(missing)}")
        frames[(str(row.dataset), str(row.variant), int(row.seed))] = frame
    return frames


def pair_runs(frames, full_variant: str, node_variant: str):
    pairs = {}
    seed_rows = []
    for dataset, _, seed in sorted(frames):
        full_key = (dataset, full_variant, seed)
        node_key = (dataset, node_variant, seed)
        if full_key not in frames or node_key not in frames or (dataset, seed) in pairs:
            continue
        paired = frames[node_key].merge(
            frames[full_key],
            on=["batch_id", "local_graph_id"],
            suffixes=("_node", "_full"),
            validate="one_to_one",
        )
        pairs[(dataset, seed)] = paired
        for metric in METRICS:
            reference = paired[f"{metric}_node"].to_numpy(float)
            trial = paired[f"{metric}_full"].to_numpy(float)
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "metric": metric,
                    "graphs": len(paired),
                    "node_only_mean": float(reference.mean()),
                    "adaptive_full_mean": float(trial.mean()),
                    "gain_pct": float(
                        signed_gain([reference.mean()], [trial.mean()], metric)[0]
                    ),
                    "graphs_improved_fraction": float(
                        np.mean(signed_gain(reference, trial, metric) > 0.0)
                    ),
                }
            )
    return pairs, pd.DataFrame(seed_rows)


def hierarchical_bootstrap(pairs, repeats: int, random_seed: int, metrics=None):
    # Crossed seed/graph bootstrap: the same graphs recur across seeds.
    # Resolve ORIGINAL graph IDs: split-local ordinals are not identities.
    # Different seeds use overlapping, not identical, test subsets.
    # ALL is a fixed-benchmark, equally weighted dataset average, not an
    # estimate of an unspecified population of datasets.
    rng = np.random.Generator(np.random.PCG64(random_seed))
    rows = []
    datasets = sorted({dataset for dataset, _ in pairs})
    metadata = None
    clustered = {}
    for (dataset, seed), frame in pairs.items():
        ordered = frame.sort_values(["batch_id", "local_graph_id"]).copy()
        if "_graph_cluster_id" not in ordered:
            if metadata is None:
                metadata = pd.read_csv(ROOT / "results/hctf_graph_metadata.csv")
            graph_ids = metadata[(metadata.dataset == dataset) & (metadata.seed == seed)].sort_values("test_graph_id")
            if len(graph_ids) != len(ordered):
                raise ValueError(f"Metadata graph count mismatch: {dataset}, seed {seed}")
            for suffix in ["node", "full"]:
                expected = ordered[[f"num_nodes_{suffix}", f"num_edges_{suffix}"]].to_numpy()
                if not np.array_equal(expected, graph_ids[["num_nodes", "num_edges"]].to_numpy()):
                    raise ValueError(f"Metadata graph order mismatch: {dataset}, seed {seed}, {suffix}")
            ordered["_graph_cluster_id"] = graph_ids.original_graph_index.to_numpy()
        if ordered._graph_cluster_id.duplicated().any():
            raise ValueError(f"Duplicate original graph identity: {dataset}, seed {seed}")
        clustered[(dataset, seed)] = ordered
    for metric in (METRICS if metrics is None else metrics):
        dataset_samples = []
        dataset_observed = []
        for dataset_scope in datasets:
            scoped = [(seed, frame) for (dataset, seed), frame in sorted(clustered.items()) if dataset == dataset_scope]
            unique_ids = sorted({value for _, frame in scoped for value in frame._graph_cluster_id})
            identity_index = {value: index for index, value in enumerate(unique_ids)}
            cluster_indices = [np.asarray([identity_index[value] for value in frame._graph_cluster_id]) for _, frame in scoped]
            reference = [frame[f"{metric}_node"].to_numpy(float) for _, frame in scoped]
            trial = [frame[f"{metric}_full"].to_numpy(float) for _, frame in scoped]
            observed = signed_gain([value.mean() for value in reference], [value.mean() for value in trial], metric)
            samples = np.empty(repeats, dtype=float)
            for repeat in range(repeats):
                seed_ids = rng.integers(0, len(scoped), size=len(scoped))
                weights = rng.multinomial(len(unique_ids), np.full(len(unique_ids), 1./len(unique_ids)))
                while any(not weights[cluster_indices[seed_id]].sum() for seed_id in seed_ids):
                    weights = rng.multinomial(len(unique_ids), np.full(len(unique_ids), 1./len(unique_ids)))
                estimates = []
                for seed_id in seed_ids:
                    run_weights = weights[cluster_indices[seed_id]]
                    reference_mean = np.average(reference[seed_id], weights=run_weights)
                    trial_mean = np.average(trial[seed_id], weights=run_weights)
                    estimates.append(signed_gain([reference_mean], [trial_mean], metric)[0])
                samples[repeat] = float(np.mean(estimates))
            dataset_samples.append(samples)
            dataset_observed.append(float(np.mean(observed)))
            rows.append(
                {
                    "dataset": dataset_scope,
                    "metric": metric,
                    "runs": len(scoped),
                    "seeds": len(scoped),
                    "mean_gain_pct": float(np.mean(observed)),
                    "run_std_pct": float(np.std(observed, ddof=1)) if len(observed) > 1 else np.nan,
                    "ci_low_pct": float(np.quantile(samples, 0.025)),
                    "ci_high_pct": float(np.quantile(samples, 0.975)),
                }
            )
        all_samples = np.stack(dataset_samples).mean(axis=0)
        rows.append({"dataset": "ALL", "metric": metric, "runs": len(pairs),
                     "seeds": len({seed for _, seed in pairs}),
                     "mean_gain_pct": float(np.mean(dataset_observed)),
                     "run_std_pct": float(np.std(dataset_observed, ddof=1)) if len(datasets) > 1 else np.nan,
                     "ci_low_pct": float(np.quantile(all_samples, .025)),
                     "ci_high_pct": float(np.quantile(all_samples, .975))})
    return pd.DataFrame(rows)


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * float(values[index])
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_tests(seed_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in seed_effects.groupby("metric", sort=False):
        values = group["gain_pct"].to_numpy(float)
        nonzero = values[np.abs(values) > 1e-12]
        statistic, p_value = (np.nan, 1.0)
        if len(nonzero):
            statistic, p_value = wilcoxon(
                nonzero, alternative="greater", zero_method="wilcox", method="auto"
            )
        rows.append(
            {
                "metric": metric,
                "dataset_seed_pairs": len(values),
                "median_gain_pct": float(np.median(values)),
                "positive_fraction": float(np.mean(values > 0.0)),
                "wilcoxon_statistic": float(statistic),
                "p_one_sided": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_one_sided"].tolist())
    return result


def gate_history_summary(frames) -> pd.DataFrame:
    rows = []
    for dataset, variant, seed in sorted(frames):
        candidates = sorted(
            (ROOT / "logs").glob(f"*_{dataset}_{variant}_s{seed}_history.csv"),
            key=lambda item: item.stat().st_mtime,
        )
        if not candidates:
            continue
        path = candidates[-1]
        history = pd.read_csv(path)
        if "hctf_structure_gate" not in history or history.empty:
            continue
        best_index = history["val_score"].astype(float).idxmin()
        best = history.loc[best_index]
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "epochs": len(history),
                "best_epoch": int(best["epoch"]),
                "best_val_score": float(best["val_score"]),
                "best_structure_gate": float(best["hctf_structure_gate"]),
                "final_structure_gate": float(history.iloc[-1]["hctf_structure_gate"]),
                "minimum_observed_gate": float(history["hctf_structure_gate_min"].min()),
                "maximum_observed_gate": float(history["hctf_structure_gate_max"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_forest(bootstrap: pd.DataFrame, output: Path) -> None:
    selected_metrics = ["graph_stress", *HOLDOUT_METRICS]
    display = {
        "graph_stress": "Stress",
        "angular_resolution_deg": "Angular resolution",
        "angular_resolution_ratio": "Normalized angular resolution",
        "crossing_angle_deg": "Crossing angle",
        "crossing_angle_p05_deg": "Crossing angle (5th percentile)",
        "node_edge_clearance": "Node–edge clearance",
        "node_edge_clearance_p05": "Node–edge clearance (5th percentile)",
        "neighborhood_preservation_k5": "Neighborhood preservation",
    }
    data = bootstrap[
        (bootstrap["dataset"] == "ALL") & bootstrap["metric"].isin(selected_metrics)
    ].copy()
    data["order"] = data["metric"].map({metric: index for index, metric in enumerate(selected_metrics)})
    data = data.sort_values("order", ascending=False)
    figure, axis = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    y = np.arange(len(data))
    axis.errorbar(
        (data["ci_low_pct"] + data["ci_high_pct"])/2,
        y,
        xerr=(data["ci_high_pct"]-data["ci_low_pct"])/2,
        fmt="none",
        color="#7A5195",
        ecolor="#7A5195",
        capsize=2.5,
        linewidth=1.1,
        markersize=4.0,
    )
    axis.scatter(data["mean_gain_pct"],y,color="#7A5195",s=16,zorder=3)
    axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, [display[metric] for metric in data["metric"]])
    axis.set_xlabel("Gain of adaptive full over node-only (%)")
    axis.set_title("Crossed seed/graph bootstrap (95% CI)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def write_latex(bootstrap: pd.DataFrame, path: Path) -> None:
    data = bootstrap[
        (bootstrap["dataset"] != "ALL") & (bootstrap["metric"] == "graph_stress")
    ].sort_values("dataset")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & Gain (\%) & 95\% CI & Seeds \\",
        r"\midrule",
    ]
    for row in data.itertuples(index=False):
        latex_dataset = str(row.dataset).replace("_", r"\_")
        lines.append(
            f"{latex_dataset} & {row.mean_gain_pct:+.3f} & "
            f"[{row.ci_low_pct:+.3f}, {row.ci_high_pct:+.3f}] & {int(row.seeds)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/hctf_adaptive_final")
    parser.add_argument("--output-dir", default="results/hctf_adaptive_final_analysis")
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--full-variant", default="adaptive_supervised_full")
    parser.add_argument("--node-variant", default="adaptive_supervised_node_only")
    parser.add_argument("--expected-pairs", type=int, default=9)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_runs(input_dir, {args.full_variant, args.node_variant})
    pairs, seed_effects = pair_runs(frames, args.full_variant, args.node_variant)
    if not pairs:
        raise RuntimeError("no completed matched adaptive-full/node-only pairs")
    if len(pairs) != args.expected_pairs and not args.allow_partial:
        raise RuntimeError(f"Expected {args.expected_pairs} matched pairs, found {len(pairs)}; refusing final analysis.")
    bootstrap = hierarchical_bootstrap(pairs, args.bootstrap_repeats, args.bootstrap_seed)
    tests = paired_tests(seed_effects)
    gates = gate_history_summary(frames)
    seed_effects.to_csv(output_dir / "paired_seed_effects.csv", index=False)
    bootstrap.to_csv(output_dir / "hierarchical_bootstrap_ci.csv", index=False)
    tests.to_csv(output_dir / "paired_wilcoxon_holm.csv", index=False)
    gates.to_csv(output_dir / "gate_history_summary.csv", index=False)
    write_latex(bootstrap, output_dir / "adaptive_stress_table.tex")
    plot_forest(bootstrap, output_dir / "adaptive_holdout_forest")

    overall_stress = bootstrap[
        (bootstrap.dataset == "ALL") & (bootstrap.metric == "graph_stress")
    ].iloc[0]
    dataset_stress = bootstrap[
        (bootstrap.dataset != "ALL") & (bootstrap.metric == "graph_stress")
    ]
    holdout = bootstrap[
        (bootstrap.dataset == "ALL") & bootstrap.metric.isin(HOLDOUT_METRICS)
    ]
    promoted = bool(
        overall_stress.ci_low_pct > 0.0
        and int((dataset_stress.ci_low_pct > 0.0).sum()) >= 2
        and int((holdout.ci_high_pct < 0.0).sum()) == 0
    )
    decision = {
        "promote_adaptive_full_to_primary": promoted,
        "criteria": {
            "overall_stress_ci_excludes_zero": bool(overall_stress.ci_low_pct > 0.0),
            "datasets_with_positive_stress_ci": int((dataset_stress.ci_low_pct > 0.0).sum()),
            "holdout_metrics_with_negative_ci": int((holdout.ci_high_pct < 0.0).sum()),
        },
        "fallback": "Retain node-only as the primary architecture and describe the structural branch as optional.",
    }
    (output_dir / "DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = [
        "# Adaptive HCTF frozen-protocol matched evaluation",
        "",
        f"Matched dataset/seed runs: {len(pairs)}.",
        f"Overall stress gain: {overall_stress.mean_gain_pct:+.4f}% "
        f"[95% CI {overall_stress.ci_low_pct:+.4f}, {overall_stress.ci_high_pct:+.4f}].",
        f"Promotion decision: {'PRIMARY' if promoted else 'OPTIONAL / NOT PRIMARY'}.",
        "",
        "The seven hold-out visual metrics were not used by SafeBlend selection.",
        "The automatic decision rule was fixed in code before inspecting final results.",
        "Existing test pools informed architectural development; this is not a blinded external validation.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
