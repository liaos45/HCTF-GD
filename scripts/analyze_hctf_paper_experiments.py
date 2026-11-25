"""Aggregate HCTF paper experiments with seed-aware paired statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "hctf_paper_experiments"

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
LOWER_IS_BETTER = set(METRICS[:7])
SCORE_FLOORS = {
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

STRUCTURAL_CONTRASTS = [
    ("no_tree", "full", "Tree-message effect"),
    ("single_tree", "full", "Complementary-tree effect"),
    ("no_relation", "full", "Relation-conditioning effect"),
    ("no_direction", "full", "Direction-conditioning effect"),
    ("no_prototype", "full", "Prototype-context effect"),
    ("node_only", "full", "Structured-field effect"),
    ("node_only_nodrop", "full_nodrop", "Structured-field effect (no dropout)"),
    ("no_tree_nodrop", "full_nodrop", "Tree-message effect (no dropout)"),
    ("single_tree_nodrop", "full_nodrop", "Complementary-tree effect (no dropout)"),
    ("no_relation_nodrop", "full_nodrop", "Relation-conditioning effect (no dropout)"),
    ("no_direction_nodrop", "full_nodrop", "Direction-conditioning effect (no dropout)"),
    ("no_prototype_nodrop", "full_nodrop", "Prototype-context effect (no dropout)"),
    ("node_only_frozen", "full_frozen", "Structured-field effect (frozen backbone)"),
    ("frozen_reference", "full_frozen", "Full branch effect (frozen backbone)"),
    ("frozen_reference", "node_only_frozen", "Node-only branch effect (frozen backbone)"),
    ("no_caid_target_nodrop", "full_nodrop", "CAID target-loss effect (descriptors retained)"),
    ("no_macpsg_nodrop", "full_nodrop", "MA-CPSG auxiliary-loss effect"),
    (
        "no_caid_macpsg_nodrop",
        "full_nodrop",
        "Joint CAID plus MA-CPSG objective effect",
    ),
]


def signed_difference(baseline: np.ndarray, trial: np.ndarray, metric: str) -> np.ndarray:
    if metric in LOWER_IS_BETTER:
        return baseline - trial
    return trial - baseline


def floor_normalized_gain(
    baseline: np.ndarray, trial: np.ndarray, metric: str
) -> np.ndarray:
    scale = np.maximum(np.abs(baseline), SCORE_FLOORS[metric])
    return 100.0 * signed_difference(baseline, trial, metric) / scale


def load_status(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(output_dir.glob("run_status_*.csv")):
        frame = pd.read_csv(path)
        frame["status_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no run status files found under {output_dir}")
    status = pd.concat(frames, ignore_index=True)
    status = status[status["status"].isin(["SUCCESS", "SKIPPED_EXISTING"])]
    status = status.drop_duplicates(["dataset", "variant", "seed"], keep="last")
    return status


def load_runs(status: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    summaries = []
    frames: dict[tuple, pd.DataFrame] = {}
    for row in status.itertuples(index=False):
        path = ROOT / row.eval_csv
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        missing = [metric for metric in METRICS if metric not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")
        if frame.duplicated(["batch_id", "local_graph_id"]).any():
            raise ValueError(f"duplicate graph keys in {path}")
        key = (str(row.dataset), str(row.variant), int(row.seed))
        frames[key] = frame
        summary = {
            "dataset": row.dataset,
            "variant": row.variant,
            "seed": int(row.seed),
            "graphs": len(frame),
            "train_eval_seconds": float(row.elapsed_seconds),
        }
        checkpoint_path = ROOT / row.checkpoint
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state = checkpoint["deepgd_state_dict"]
            non_parameter_suffixes = (
                "running_mean",
                "running_var",
                "num_batches_tracked",
            )
            summary["parameter_count"] = int(
                sum(
                    value.numel()
                    for name, value in state.items()
                    if not name.endswith(non_parameter_suffixes)
                )
            )
            summary["hctf_parameter_count"] = int(
                sum(
                    value.numel()
                    for name, value in state.items()
                    if name.startswith("hctf_")
                    and not name.endswith(non_parameter_suffixes)
                )
            )
            summary["best_epoch"] = int(checkpoint.get("epoch", -1))
            summary["best_val"] = float(checkpoint.get("best_val", np.nan))
            mix_entries = [
                value
                for name, value in state.items()
                if name.endswith("hctf_block.mix_logit")
            ]
            if mix_entries:
                mix_logit = float(mix_entries[0].detach().cpu().item())
                checkpoint_args = checkpoint.get("args", {})
                if isinstance(checkpoint_args, dict):
                    hctf_scale = float(checkpoint_args.get("hctf_scale", 0.20))
                else:
                    hctf_scale = float(getattr(checkpoint_args, "hctf_scale", 0.20))
                summary["hctf_mix_logit"] = mix_logit
                summary["hctf_scale"] = hctf_scale
                summary["hctf_effective_mix"] = hctf_scale / (1.0 + np.exp(-mix_logit))
        summary.update({metric: float(frame[metric].mean()) for metric in METRICS})
        batch_times = frame.groupby("batch_id", sort=False)["batch_infer_time"].first()
        summary["inference_seconds"] = float(batch_times.sum())
        summary["milliseconds_per_graph"] = 1000.0 * float(batch_times.sum()) / len(frame)
        if "peak_cuda_memory_mb" in frame.columns:
            summary["peak_cuda_memory_mb"] = float(frame["peak_cuda_memory_mb"].max())
        summaries.append(summary)
    return pd.DataFrame(summaries), frames


def paired_effects(frames: dict[tuple, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_frames: dict[tuple, pd.DataFrame] = {}
    for dataset, variant, seed in sorted(frames):
        if variant == "baseline":
            continue
        baseline_key = (dataset, "baseline", seed)
        if baseline_key not in frames:
            continue
        base = frames[baseline_key]
        trial = frames[(dataset, variant, seed)]
        merged = base.merge(
            trial,
            on=["batch_id", "local_graph_id"],
            suffixes=("_baseline", "_trial"),
            validate="one_to_one",
        )
        pair_frames[(dataset, variant, seed)] = merged
        for metric in METRICS:
            b = merged[f"{metric}_baseline"].to_numpy(float)
            t = merged[f"{metric}_trial"].to_numpy(float)
            raw = signed_difference(b, t, metric)
            gain = floor_normalized_gain(b, t, metric)
            dataset_gain = float(
                floor_normalized_gain(
                    np.asarray([np.mean(b)]), np.asarray([np.mean(t)]), metric
                )[0]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "metric": metric,
                    "graphs": len(merged),
                    "baseline_mean": float(np.mean(b)),
                    "trial_mean": float(np.mean(t)),
                    "signed_difference_mean": float(np.mean(raw)),
                    "dataset_mean_gain_pct": dataset_gain,
                    "per_graph_gain_mean_pct": float(np.mean(gain)),
                    "graphs_improved_fraction": float(np.mean(raw > 0.0)),
                }
            )
    return pd.DataFrame(rows), pair_frames


def hierarchical_bootstrap(
    pair_frames: dict[tuple, pd.DataFrame], repeats: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups: dict[tuple[str, str], list[tuple[int, pd.DataFrame]]] = {}
    for (dataset, variant, run_seed), frame in pair_frames.items():
        groups.setdefault((dataset, variant), []).append((run_seed, frame))
    for (dataset, variant), seed_frames in sorted(groups.items()):
        for metric in METRICS:
            estimates = np.empty(repeats, dtype=float)
            for repeat in range(repeats):
                sampled_seed_ids = rng.integers(0, len(seed_frames), size=len(seed_frames))
                seed_means = []
                for seed_id in sampled_seed_ids:
                    _, frame = seed_frames[int(seed_id)]
                    sample_ids = rng.integers(0, len(frame), size=len(frame))
                    b = frame[f"{metric}_baseline"].to_numpy(float)[sample_ids]
                    t = frame[f"{metric}_trial"].to_numpy(float)[sample_ids]
                    seed_means.append(
                        float(
                            floor_normalized_gain(
                                np.asarray([np.mean(b)]),
                                np.asarray([np.mean(t)]),
                                metric,
                            )[0]
                        )
                    )
                estimates[repeat] = float(np.mean(seed_means))
            observed = []
            for _, frame in seed_frames:
                b = frame[f"{metric}_baseline"].to_numpy(float)
                t = frame[f"{metric}_trial"].to_numpy(float)
                observed.append(
                    float(
                        floor_normalized_gain(
                            np.asarray([np.mean(b)]),
                            np.asarray([np.mean(t)]),
                            metric,
                        )[0]
                    )
                )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "seeds": len(seed_frames),
                    "mean_gain_pct": float(np.mean(observed)),
                    "seed_std_pct": float(np.std(observed, ddof=1))
                    if len(observed) > 1
                    else np.nan,
                    "hierarchical_bootstrap_ci_low_pct": float(
                        np.quantile(estimates, 0.025)
                    ),
                    "hierarchical_bootstrap_ci_high_pct": float(
                        np.quantile(estimates, 0.975)
                    ),
                }
            )
    return pd.DataFrame(rows)


def structural_contrasts(
    frames: dict[tuple, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    """Pair full fields against matched ablations rather than only the baseline."""
    rows = []
    pair_frames: dict[tuple, pd.DataFrame] = {}
    datasets = sorted({key[0] for key in frames})
    seeds = sorted({key[2] for key in frames})
    for reference_variant, trial_variant, contrast in STRUCTURAL_CONTRASTS:
        for dataset in datasets:
            for seed in seeds:
                reference_key = (dataset, reference_variant, seed)
                trial_key = (dataset, trial_variant, seed)
                if reference_key not in frames or trial_key not in frames:
                    continue
                reference = frames[reference_key]
                trial = frames[trial_key]
                merged = reference.merge(
                    trial,
                    on=["batch_id", "local_graph_id"],
                    suffixes=("_reference", "_trial"),
                    validate="one_to_one",
                )
                key = (dataset, reference_variant, trial_variant, seed)
                pair_frames[key] = merged
                for metric in METRICS:
                    reference_values = merged[f"{metric}_reference"].to_numpy(float)
                    trial_values = merged[f"{metric}_trial"].to_numpy(float)
                    raw = signed_difference(reference_values, trial_values, metric)
                    dataset_gain = float(
                        floor_normalized_gain(
                            np.asarray([np.mean(reference_values)]),
                            np.asarray([np.mean(trial_values)]),
                            metric,
                        )[0]
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "reference_variant": reference_variant,
                            "trial_variant": trial_variant,
                            "contrast": contrast,
                            "seed": seed,
                            "metric": metric,
                            "graphs": len(merged),
                            "reference_mean": float(np.mean(reference_values)),
                            "trial_mean": float(np.mean(trial_values)),
                            "signed_difference_mean": float(np.mean(raw)),
                            "dataset_mean_gain_pct": dataset_gain,
                            "graphs_improved_fraction": float(np.mean(raw > 0.0)),
                        }
                    )
    return pd.DataFrame(rows), pair_frames


def hierarchical_bootstrap_contrasts(
    pair_frames: dict[tuple, pd.DataFrame], repeats: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups: dict[tuple[str, str, str], list[tuple[int, pd.DataFrame]]] = {}
    for (dataset, reference_variant, trial_variant, run_seed), frame in pair_frames.items():
        groups.setdefault((dataset, reference_variant, trial_variant), []).append(
            (run_seed, frame)
        )
    contrast_lookup = {
        (reference, trial): name
        for reference, trial, name in STRUCTURAL_CONTRASTS
    }
    for (dataset, reference_variant, trial_variant), seed_frames in sorted(groups.items()):
        for metric in METRICS:
            estimates = np.empty(repeats, dtype=float)
            for repeat in range(repeats):
                sampled_seed_ids = rng.integers(0, len(seed_frames), size=len(seed_frames))
                seed_means = []
                for seed_id in sampled_seed_ids:
                    _, frame = seed_frames[int(seed_id)]
                    sample_ids = rng.integers(0, len(frame), size=len(frame))
                    reference_values = frame[f"{metric}_reference"].to_numpy(float)[sample_ids]
                    trial_values = frame[f"{metric}_trial"].to_numpy(float)[sample_ids]
                    seed_means.append(
                        float(
                            floor_normalized_gain(
                                np.asarray([np.mean(reference_values)]),
                                np.asarray([np.mean(trial_values)]),
                                metric,
                            )[0]
                        )
                    )
                estimates[repeat] = float(np.mean(seed_means))
            observed = []
            for _, frame in seed_frames:
                reference_values = frame[f"{metric}_reference"].to_numpy(float)
                trial_values = frame[f"{metric}_trial"].to_numpy(float)
                observed.append(
                    float(
                        floor_normalized_gain(
                            np.asarray([np.mean(reference_values)]),
                            np.asarray([np.mean(trial_values)]),
                            metric,
                        )[0]
                    )
                )
            rows.append(
                {
                    "dataset": dataset,
                    "reference_variant": reference_variant,
                    "trial_variant": trial_variant,
                    "contrast": contrast_lookup[(reference_variant, trial_variant)],
                    "metric": metric,
                    "seeds": len(seed_frames),
                    "mean_gain_pct": float(np.mean(observed)),
                    "seed_std_pct": float(np.std(observed, ddof=1))
                    if len(observed) > 1
                    else np.nan,
                    "hierarchical_bootstrap_ci_low_pct": float(
                        np.quantile(estimates, 0.025)
                    ),
                    "hierarchical_bootstrap_ci_high_pct": float(
                        np.quantile(estimates, 0.975)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    status = load_status(output_dir)
    run_summary, frames = load_runs(status)
    effects, pair_frames = paired_effects(frames)
    bootstrap = hierarchical_bootstrap(
        pair_frames, args.bootstrap_repeats, args.bootstrap_seed
    )
    contrasts, contrast_pair_frames = structural_contrasts(frames)
    contrast_bootstrap = hierarchical_bootstrap_contrasts(
        contrast_pair_frames, args.bootstrap_repeats, args.bootstrap_seed + 1
    )
    run_summary.to_csv(output_dir / "run_metric_summary.csv", index=False)
    effects.to_csv(output_dir / "paired_seed_effects.csv", index=False)
    bootstrap.to_csv(output_dir / "hierarchical_bootstrap_ci.csv", index=False)
    contrasts.to_csv(output_dir / "structural_contrast_seed_effects.csv", index=False)
    contrast_bootstrap.to_csv(
        output_dir / "structural_contrast_bootstrap_ci.csv", index=False
    )
    print(
        f"[DONE] runs={len(run_summary)}, paired rows={len(effects)}, "
        f"bootstrap rows={len(bootstrap)}, contrast rows={len(contrasts)}, "
        f"contrast bootstrap rows={len(contrast_bootstrap)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
