"""Analyze CAID/MA-CPSG loss ablations against the matched full reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_hctf_paper_experiments as analysis  # noqa: E402


ABLATIONS = (
    "no_caid_target_nodrop",
    "no_macpsg_nodrop",
    "no_caid_macpsg_nodrop",
)


def equal_domain_bootstrap(
    pair_frames: dict[tuple, pd.DataFrame], repeats: int, seed: int
) -> pd.DataFrame:
    """Bootstrap seeds/graphs within each domain, then average domains equally."""
    rng = np.random.default_rng(seed)
    rows = []
    contrast_lookup = {
        (reference, trial): name
        for reference, trial, name in analysis.STRUCTURAL_CONTRASTS
    }
    comparisons = sorted({(key[1], key[2]) for key in pair_frames})
    for reference_variant, trial_variant in comparisons:
        by_dataset: dict[str, list[tuple[int, pd.DataFrame]]] = {}
        for (dataset, reference, trial, run_seed), frame in pair_frames.items():
            if (reference, trial) == (reference_variant, trial_variant):
                by_dataset.setdefault(dataset, []).append((run_seed, frame))
        for dataset in by_dataset:
            by_dataset[dataset].sort(key=lambda item: item[0])
        if sorted(by_dataset) != ["enzymes", "fb15k237", "sbm"]:
            raise RuntimeError(
                "Equal-domain objective summary requires all three domains: "
                f"{reference_variant} -> {trial_variant} has {sorted(by_dataset)}"
            )
        for metric in analysis.METRICS:
            observed_domain_means = []
            for seed_frames in by_dataset.values():
                seed_gains = []
                for _, frame in seed_frames:
                    reference_values = frame[f"{metric}_reference"].to_numpy(float)
                    trial_values = frame[f"{metric}_trial"].to_numpy(float)
                    seed_gains.append(
                        float(
                            analysis.floor_normalized_gain(
                                np.asarray([np.mean(reference_values)]),
                                np.asarray([np.mean(trial_values)]),
                                metric,
                            )[0]
                        )
                    )
                observed_domain_means.append(float(np.mean(seed_gains)))

            estimates = np.empty(repeats, dtype=float)
            for repeat in range(repeats):
                sampled_domains = []
                for seed_frames in by_dataset.values():
                    seed_ids = rng.integers(0, len(seed_frames), size=len(seed_frames))
                    seed_gains = []
                    for seed_id in seed_ids:
                        _, frame = seed_frames[int(seed_id)]
                        sample_ids = rng.integers(0, len(frame), size=len(frame))
                        reference_values = frame[f"{metric}_reference"].to_numpy(float)[
                            sample_ids
                        ]
                        trial_values = frame[f"{metric}_trial"].to_numpy(float)[
                            sample_ids
                        ]
                        seed_gains.append(
                            float(
                                analysis.floor_normalized_gain(
                                    np.asarray([np.mean(reference_values)]),
                                    np.asarray([np.mean(trial_values)]),
                                    metric,
                                )[0]
                            )
                        )
                    sampled_domains.append(float(np.mean(seed_gains)))
                estimates[repeat] = float(np.mean(sampled_domains))

            rows.append(
                {
                    "dataset": "ALL",
                    "reference_variant": reference_variant,
                    "trial_variant": trial_variant,
                    "contrast": contrast_lookup[(reference_variant, trial_variant)],
                    "metric": metric,
                    "seeds": 3,
                    "mean_gain_pct": float(np.mean(observed_domain_means)),
                    "seed_std_pct": float(np.std(observed_domain_means, ddof=1)),
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
    parser.add_argument(
        "--input-dir", default="results/hctf_objective_ablations"
    )
    parser.add_argument(
        "--reference-dir", default="results/hctf_submission_extensions/ablation"
    )
    parser.add_argument(
        "--output-dir", default="results/hctf_objective_ablations_analysis"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()

    objective_status = analysis.load_status(ROOT / args.input_dir)
    reference_status = analysis.load_status(ROOT / args.reference_dir)
    objective_status = objective_status[objective_status.variant.isin(ABLATIONS)]
    reference_status = reference_status[reference_status.variant == "full_nodrop"]
    status = pd.concat([objective_status, reference_status], ignore_index=True)
    status = status.drop_duplicates(["dataset", "variant", "seed"], keep="last")
    run_summary, frames = analysis.load_runs(status)

    expected = {
        (dataset, variant, seed)
        for dataset in ("enzymes", "fb15k237", "sbm")
        for variant in ("full_nodrop", *ABLATIONS)
        for seed in (17, 42, 73)
    }
    missing = sorted(expected - set(frames))
    if missing:
        raise RuntimeError(f"Formal objective ablation is incomplete: {missing}")

    contrasts, paired = analysis.structural_contrasts(frames)
    contrasts = contrasts[
        contrasts.reference_variant.isin(ABLATIONS)
        & (contrasts.trial_variant == "full_nodrop")
    ]
    paired = {
        key: value
        for key, value in paired.items()
        if key[1] in ABLATIONS and key[2] == "full_nodrop"
    }
    bootstrap = analysis.hierarchical_bootstrap_contrasts(
        paired, args.bootstrap_repeats, 20261009
    )
    bootstrap = pd.concat(
        [
            bootstrap,
            equal_domain_bootstrap(
                paired, args.bootstrap_repeats, 20261010
            ),
        ],
        ignore_index=True,
    )

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(output / "run_summary.csv", index=False)
    contrasts.to_csv(output / "objective_contrast_seed_effects.csv", index=False)
    bootstrap.to_csv(output / "objective_contrast_bootstrap_ci.csv", index=False)
    marker = {
        "complete": True,
        "datasets": ["enzymes", "fb15k237", "sbm"],
        "seeds": [17, 42, 73],
        "reference": "full_nodrop",
        "ablations": list(ABLATIONS),
        "paired_runs": len(paired),
        "caid_ablation_scope": (
            "CAID target loss removed while conditional descriptors are retained"
        ),
        "intervals": (
            "Endpoint-wise hierarchical graph/seed bootstrap; exploratory rather "
            "than simultaneous family-wise intervals"
        ),
    }
    (output / "COMPLETE.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# HCTF objective ablation\n\n"
        "The full no-dropout reference is paired with loss-level removals of the "
        "CAID adjusted-distance target, all MA-CPSG terms, and both objective "
        "families. CAID conditional input descriptors remain present, so the CAID "
        "contrast isolates the training target rather than changing model input.\n\n"
        "All datasets use three seeds and the same 12-epoch, 16-train-batch, "
        "6-validation-batch budget. Positive gains denote improvement in the "
        "predeclared direction.\n",
        encoding="utf-8",
    )
    print(f"[OBJECTIVE ANALYSIS] {len(paired)} paired runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
