"""Create submission-level tables from learned, classical, robustness, and audit runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
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
LOWER = set(METRICS[:7])
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
}
VARIANT_NAMES = {
    "baseline": "Community-aware backbone",
    "full_nodrop": "Full HCTF",
    "node_only_nodrop": "Node-only control",
}


def signed_gain(reference, trial, metric):
    reference = np.asarray(reference, dtype=float)
    trial = np.asarray(trial, dtype=float)
    scale = np.maximum(np.abs(reference), FLOORS[metric])
    difference = reference - trial if metric in LOWER else trial - reference
    return 100.0 * difference / scale


def load_learned(path):
    status_path = path / "run_status_main.csv"
    if not status_path.exists():
        return pd.DataFrame()
    status = pd.read_csv(status_path)
    status = status[status.status.isin(["SUCCESS", "SKIPPED_EXISTING"])]
    frames = []
    for row in status.itertuples(index=False):
        csv_path = ROOT / row.eval_csv
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["dataset"] = row.dataset
        frame["seed"] = int(row.seed)
        frame["variant"] = row.variant
        frame["display_method"] = VARIANT_NAMES.get(row.variant, row.variant)
        frame["test_graph_id"] = np.arange(len(frame), dtype=int)
        counts = frame.groupby("batch_id")["batch_id"].transform("size")
        frame["inference_seconds_per_graph"] = frame["batch_infer_time"] / counts
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_classical(path):
    frames = []
    for csv_path in sorted(path.glob("*_all_s*.csv")):
        frame = pd.read_csv(csv_path)
        frame["variant"] = frame["method"].str.lower()
        frame["display_method"] = frame["method"]
        frame["inference_seconds_per_graph"] = frame["layout_time"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_original(path):
    frames = []
    for csv_path in sorted(path.glob("original_deepgd_*_s*.csv")):
        frame = pd.read_csv(csv_path)
        frame["variant"] = "original_deepgd"
        frame["display_method"] = "Original DeepGD"
        frame["test_graph_id"] = frame["graph_id"].astype(int)
        counts = frame.groupby("batch_id")["batch_id"].transform("size")
        frame["inference_seconds_per_graph"] = frame["batch_infer_time"] / counts
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def method_summary(frame):
    if frame.empty:
        return pd.DataFrame()
    columns = [metric for metric in METRICS if metric in frame]
    extra = [column for column in ["inference_seconds_per_graph", "peak_cuda_memory_mb"] if column in frame]
    return (
        frame.groupby(["dataset", "display_method", "seed"], as_index=False)[columns + extra]
        .mean()
        .sort_values(["dataset", "display_method", "seed"])
    )


def paired_effects(all_frame, repeats, bootstrap_seed):
    if all_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    reference = all_frame[all_frame.variant == "baseline"]
    rows = []
    paired = {}
    keys = ["dataset", "seed", "test_graph_id"]
    for (dataset, seed), reference_group in reference.groupby(["dataset", "seed"]):
        candidates = all_frame[
            (all_frame.dataset == dataset)
            & (all_frame.seed == seed)
            & (all_frame.variant != "baseline")
        ]
        for variant, trial in candidates.groupby("variant"):
            merged = reference_group.merge(trial, on=keys, suffixes=("_reference", "_trial"), validate="one_to_one")
            if merged.empty:
                continue
            paired[(dataset, variant, int(seed))] = merged
            for metric in METRICS:
                if f"{metric}_reference" not in merged or f"{metric}_trial" not in merged:
                    continue
                r = merged[f"{metric}_reference"].to_numpy(float)
                t = merged[f"{metric}_trial"].to_numpy(float)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": variant,
                        "display_method": str(merged["display_method_trial"].iloc[0]),
                        "metric": metric,
                        "graphs": len(merged),
                        "reference_mean": float(r.mean()),
                        "trial_mean": float(t.mean()),
                        "mean_gain_pct": float(signed_gain([r.mean()], [t.mean()], metric)[0]),
                        "graphs_improved_fraction": float(np.mean(signed_gain(r, t, metric) > 0)),
                    }
                )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_rows = []
    groups = {}
    for (dataset, variant, seed), frame in paired.items():
        groups.setdefault((dataset, variant), []).append((seed, frame))
    for (dataset, variant), seed_frames in sorted(groups.items()):
        display_method = str(seed_frames[0][1]["display_method_trial"].iloc[0])
        for metric in METRICS:
            ref_col = f"{metric}_reference"
            trial_col = f"{metric}_trial"
            if ref_col not in seed_frames[0][1] or trial_col not in seed_frames[0][1]:
                continue
            observed = []
            for _, frame in seed_frames:
                r = frame[ref_col].to_numpy(float)
                t = frame[trial_col].to_numpy(float)
                observed.append(float(signed_gain([r.mean()], [t.mean()], metric)[0]))
            samples = np.empty(repeats)
            for repeat in range(repeats):
                sampled_seed_ids = rng.integers(0, len(seed_frames), size=len(seed_frames))
                estimates = []
                for seed_id in sampled_seed_ids:
                    _, frame = seed_frames[int(seed_id)]
                    indices = rng.integers(0, len(frame), size=len(frame))
                    r = frame[ref_col].to_numpy(float)[indices]
                    t = frame[trial_col].to_numpy(float)[indices]
                    estimates.append(float(signed_gain([r.mean()], [t.mean()], metric)[0]))
                samples[repeat] = float(np.mean(estimates))
            bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "display_method": display_method,
                    "metric": metric,
                    "seeds": len(seed_frames),
                    "mean_gain_pct": float(np.mean(observed)),
                    "seed_std_pct": float(np.std(observed, ddof=1)) if len(observed) > 1 else np.nan,
                    "ci_low_pct": float(np.quantile(samples, 0.025)),
                    "ci_high_pct": float(np.quantile(samples, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def size_stratified_effects(learned):
    if learned.empty:
        return pd.DataFrame()
    reference = learned[learned.variant == "baseline"].copy()
    labels = ["Small", "Medium", "Large"]
    bin_map = []
    for (dataset, seed), group in reference.groupby(["dataset", "seed"]):
        ranked = group["num_nodes"].rank(method="first")
        bins = pd.qcut(ranked, 3, labels=labels)
        mapping = group[["dataset", "seed", "test_graph_id"]].copy()
        mapping["size_bin"] = bins.astype(str).to_numpy()
        bin_map.append(mapping)
    bin_map = pd.concat(bin_map, ignore_index=True)
    enriched = learned.merge(bin_map, on=["dataset", "seed", "test_graph_id"], validate="many_to_one")
    rows = []
    for (dataset, seed, variant, size_bin), trial in enriched[enriched.variant != "baseline"].groupby(
        ["dataset", "seed", "variant", "size_bin"]
    ):
        base = enriched[
            (enriched.dataset == dataset)
            & (enriched.seed == seed)
            & (enriched.variant == "baseline")
            & (enriched.size_bin == size_bin)
        ]
        merged = base.merge(trial, on=["dataset", "seed", "test_graph_id", "size_bin"], suffixes=("_reference", "_trial"))
        for metric in METRICS:
            r = merged[f"{metric}_reference"].to_numpy(float)
            t = merged[f"{metric}_trial"].to_numpy(float)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "size_bin": size_bin,
                    "metric": metric,
                    "graphs": len(merged),
                    "mean_nodes": float(merged["num_nodes_reference"].mean()),
                    "mean_gain_pct": float(signed_gain([r.mean()], [t.mean()], metric)[0]),
                }
            )
    return pd.DataFrame(rows)


def partition_summary(path):
    frames = []
    for csv_path in sorted(path.glob("*_full_nodrop_s*.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    group_columns = ["dataset", "seed", "partition_source", "noise_fraction", "reference_partition"]
    columns = ["partition_nmi", "partition_ari", "partition_modularity", *[metric if metric != "graph_stress" else "batch_stress" for metric in METRICS]]
    summary = frame.groupby(group_columns, as_index=False)[columns].mean()
    summary = summary.rename(columns={"batch_stress": "graph_stress"})
    rows = []
    for (dataset, seed), group in frame.groupby(["dataset", "seed"]):
        reference = group[(group.partition_source == "greedy") & (group.noise_fraction == 0.0)]
        for (source, noise), trial in group.groupby(["partition_source", "noise_fraction"]):
            merged = reference.merge(trial, on=["dataset", "seed", "graph_id"], suffixes=("_reference", "_trial"))
            if merged.empty:
                continue
            for metric in METRICS:
                source_metric = "batch_stress" if metric == "graph_stress" else metric
                r = merged[f"{source_metric}_reference"].to_numpy(float)
                t = merged[f"{source_metric}_trial"].to_numpy(float)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "partition_source": source,
                        "noise_fraction": noise,
                        "metric": metric,
                        "mean_gain_vs_greedy_pct": float(signed_gain([r.mean()], [t.mean()], metric)[0]),
                    }
                )
    return summary, pd.DataFrame(rows)


def community_strength_effects(learned, metadata):
    if learned.empty or metadata.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["dataset", "seed", "test_graph_id"]
    reference = learned[learned.variant == "baseline"]
    rows = []
    correlations = []
    for (dataset, seed, variant), trial in learned[learned.variant != "baseline"].groupby(
        ["dataset", "seed", "variant"]
    ):
        base = reference[(reference.dataset == dataset) & (reference.seed == seed)]
        merged = base.merge(trial, on=keys, suffixes=("_reference", "_trial"))
        merged = merged.merge(metadata, on=keys, validate="one_to_one")
        if merged.empty:
            continue
        graph_gain = signed_gain(
            merged["graph_stress_reference"], merged["graph_stress_trial"], "graph_stress"
        )
        rho, p_value = spearmanr(merged["greedy_modularity"], graph_gain)
        correlations.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "graphs": len(merged),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
            }
        )
        ranked = merged["greedy_modularity"].rank(method="first")
        merged["community_strength_bin"] = pd.qcut(
            ranked, 3, labels=["Weak", "Medium", "Strong"]
        ).astype(str)
        merged["graph_gain_pct"] = graph_gain
        for strength, group in merged.groupby("community_strength_bin"):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "community_strength_bin": strength,
                    "graphs": len(group),
                    "mean_modularity": float(group["greedy_modularity"].mean()),
                    "mean_graph_stress_gain_pct": float(group["graph_gain_pct"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def write_report(output_dir, learned, all_frame, bootstrap, partition):
    lines = [
        "# HCTF submission experiment status",
        "",
        f"Completed learned evaluation rows: {len(learned):,}",
        f"Completed unified layout rows: {len(all_frame):,}",
        "",
    ]
    stress = bootstrap[bootstrap.metric == "graph_stress"] if not bootstrap.empty else bootstrap
    if not stress.empty:
        lines.extend(["## Paired graph-stress effects versus community-aware backbone", ""])
        for row in stress.sort_values(["display_method", "dataset"]).itertuples(index=False):
            lines.append(
                f"- {row.display_method}, {row.dataset}: {row.mean_gain_pct:+.3f}% "
                f"[95% CI {row.ci_low_pct:+.3f}, {row.ci_high_pct:+.3f}], n={row.seeds} seeds."
            )
    if not partition.empty:
        lines.extend(["", "## Partition robustness coverage", ""])
        covered = partition.groupby("dataset").size()
        for dataset, count in covered.items():
            lines.append(f"- {dataset}: {count} seed/partition settings summarized.")
    (output_dir / "submission_results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-dir", type=Path, default=Path("results/hctf_submission_main"))
    parser.add_argument("--classical-dir", type=Path, default=Path("results/hctf_classical_baselines"))
    parser.add_argument("--original-dir", type=Path, default=Path("results/original_deepgd_baselines"))
    parser.add_argument("--partition-dir", type=Path, default=Path("results/hctf_partition_robustness"))
    parser.add_argument("--graph-metadata", type=Path, default=Path("results/hctf_graph_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/submission_generated"))
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    args = parser.parse_args()

    resolve = lambda path: path if path.is_absolute() else ROOT / path
    learned = load_learned(resolve(args.learned_dir))
    classical = load_classical(resolve(args.classical_dir))
    original = load_original(resolve(args.original_dir))
    frames = [frame for frame in [learned, classical, original] if not frame.empty]
    all_frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    summary = method_summary(all_frame)
    seed_effects, bootstrap = paired_effects(all_frame, args.bootstrap_repeats, args.bootstrap_seed)
    size_effects = size_stratified_effects(learned)
    partition, partition_effects = partition_summary(resolve(args.partition_dir))
    metadata_path = resolve(args.graph_metadata)
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    strength_effects, strength_correlations = community_strength_effects(learned, metadata)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "unified_method_seed_summary.csv", index=False)
    seed_effects.to_csv(output_dir / "paired_method_seed_effects.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_method_bootstrap_ci.csv", index=False)
    size_effects.to_csv(output_dir / "size_stratified_effects.csv", index=False)
    partition.to_csv(output_dir / "partition_robustness_summary.csv", index=False)
    partition_effects.to_csv(output_dir / "partition_robustness_effects.csv", index=False)
    strength_effects.to_csv(output_dir / "community_strength_effects.csv", index=False)
    strength_correlations.to_csv(output_dir / "community_strength_correlations.csv", index=False)
    write_report(output_dir, learned, all_frame, bootstrap, partition)
    print(
        f"[DONE] learned_rows={len(learned)}, all_rows={len(all_frame)}, "
        f"bootstrap_rows={len(bootstrap)}, partition_rows={len(partition)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
