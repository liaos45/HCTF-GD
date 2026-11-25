"""Paired, seed/graph-clustered descriptive Validation analysis for A/B/C/D."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research_v2/graph_local"
OUT = BASE / "metrics"
DOMAINS = ("enzymes", "fb15k237", "sbm")
SEEDS = (17, 42, 73)
VARIANTS = ("A_both", "B_caid_only", "C_macpsg_only", "D_neither")
METRICS = ("graph_stress", "edge_crossing", "edge_length_cv", "node_occlusion_rate",
           "community_compactness", "community_separation", "community_silhouette",
           "community_sep_comp_ratio")
HIGHER = {"community_separation", "community_silhouette", "community_sep_comp_ratio"}
STATS_LOCK = json.loads((ROOT / "research_v2/formal_test/STATISTICS_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
FLOORS = STATS_LOCK["reference_floors"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, dict]:
    frames, hashes = [], {}
    for domain in DOMAINS:
        for seed in SEEDS:
            cohort = BASE / "validation_graphwise" / f"{domain}_s{seed}"
            protocol = json.loads((cohort / "PROTOCOL.json").read_text(encoding="utf-8"))
            if protocol["status"] != "VALIDATION_GRAPHWISE_COMPLETE" or protocol["split"] != "validation":
                raise RuntimeError(f"cohort incomplete: {cohort}")
            csv_path = cohort / "graphwise_validation.csv"
            npz_path = cohort / "coordinates.npz"
            if sha(csv_path) != protocol["graphwise_csv_sha256"] or sha(npz_path) != protocol["coordinates_npz_sha256"]:
                raise RuntimeError(f"cohort artifact SHA mismatch: {cohort}")
            frame = pd.read_csv(csv_path)
            if len(frame) != 4 * protocol["graphs"]:
                raise RuntimeError(f"row count mismatch: {cohort}")
            if set(frame.variant) != set(VARIANTS) or frame.graph_id.nunique() != protocol["graphs"]:
                raise RuntimeError(f"variant/graph identity mismatch: {cohort}")
            if frame.groupby("graph_id").variant.nunique().ne(4).any():
                raise RuntimeError(f"unpaired graph identities: {cohort}")
            for field in METRICS:
                if not np.isfinite(frame[field]).all():
                    raise RuntimeError(f"nonfinite {field}: {cohort}")
            frames.append(frame)
            hashes[str(csv_path)] = sha(csv_path)
    return pd.concat(frames, ignore_index=True), hashes


def gain(reference: np.ndarray, trial: np.ndarray, metric: str) -> np.ndarray:
    sign = -1 if metric in HIGHER else 1
    return 100 * sign * (reference-trial) / np.maximum(np.abs(reference), FLOORS[metric])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, hashes = load()
    run_rows = []
    primary_matrices = {}
    graph_gain_count = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            part = data[(data.domain == domain) & (data.seed == seed)]
            for metric in METRICS:
                pivot = part.pivot(index="graph_id", columns="variant", values=metric)[list(VARIANTS)]
                values = pivot.mean().to_numpy(float)
                a, b, c, d = values
                contrasts = {
                    "A_vs_B": float(gain(np.array(b), np.array(a), metric)),
                    "A_vs_C": float(gain(np.array(c), np.array(a), metric)),
                    "A_vs_D": float(gain(np.array(d), np.array(a), metric)),
                }
                if metric == "graph_stress":
                    norm = max(abs(d), FLOORS[metric])
                    contrasts.update({
                        "CAID_main": 100 * ((d-b)+(c-a))/(2*norm),
                        "MACPSG_main": 100 * ((d-c)+(b-a))/(2*norm),
                        "interaction": 100 * (b+c-d-a)/norm,
                    })
                    primary_matrices[(domain, seed)] = pivot.to_numpy(float)
                    graph_gain_count[(domain, seed)] = {
                        other: (int((pivot[other] > pivot["A_both"]).sum()),
                                int((pivot[other] < pivot["A_both"]).sum()))
                        for other in VARIANTS[1:]
                    }
                for contrast, value in contrasts.items():
                    run_rows.append({"domain": domain, "seed": seed, "metric": metric,
                                     "contrast": contrast, "gain_pct": value,
                                     "graphs": len(pivot)})
    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(OUT / "graphwise_validation_run_effects.csv", index=False)
    domain_frame = (run_frame.groupby(["domain", "metric", "contrast"], as_index=False)
                    .agg(mean_gain_pct=("gain_pct", "mean"),
                         median_seed_gain_pct=("gain_pct", "median"),
                         positive_seeds=("gain_pct", lambda x: int((x > 0).sum())),
                         negative_seeds=("gain_pct", lambda x: int((x < 0).sum()))))
    domain_frame.to_csv(OUT / "graphwise_validation_domain_effects.csv", index=False)
    graph_counts = []
    for (domain, seed), contrasts in graph_gain_count.items():
        for reference, (positive, negative) in contrasts.items():
            graph_counts.append({"domain": domain, "seed": seed,
                                 "contrast": f"A_vs_{reference[0]}",
                                 "positive_graphs": positive, "negative_graphs": negative,
                                 "ties": int(primary_matrices[(domain, seed)].shape[0] - positive - negative)})
    pd.DataFrame(graph_counts).to_csv(OUT / "graphwise_validation_graph_signs.csv", index=False)

    # The same sampled seed IDs are used across all domains. Each selected
    # occurrence gets its own paired within-graph resample.
    replicates = STATS_LOCK["bootstrap"]["replicates"]
    rng = np.random.default_rng(STATS_LOCK["bootstrap"]["random_seed"])
    seed_draw = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)))
    boot = np.empty((replicates, len(DOMAINS), len(SEEDS), len(VARIANTS)), dtype=float)
    for domain_index, domain in enumerate(DOMAINS):
        for slot in range(len(SEEDS)):
            for seed_index, seed in enumerate(SEEDS):
                selected = np.flatnonzero(seed_draw[:, slot] == seed_index)
                if not len(selected):
                    continue
                matrix = primary_matrices[(domain, seed)]
                indices = rng.integers(0, len(matrix), size=(len(selected), len(matrix)))
                boot[selected, domain_index, slot] = matrix[indices].mean(axis=1)
    a, b, c, d = [boot[..., index] for index in range(4)]
    denom = np.maximum(np.abs(d), FLOORS["graph_stress"])
    estimates = {
        "A_vs_B": gain(b, a, "graph_stress"),
        "A_vs_C": gain(c, a, "graph_stress"),
        "A_vs_D": gain(d, a, "graph_stress"),
        "CAID_main": 100*((d-b)+(c-a))/(2*denom),
        "MACPSG_main": 100*((d-c)+(b-a))/(2*denom),
        "interaction": 100*(b+c-d-a)/denom,
    }
    summaries = []
    for contrast, distribution in estimates.items():
        estimate = run_frame[(run_frame.metric == "graph_stress") &
                             (run_frame.contrast == contrast)].gain_pct.mean()
        aggregate = distribution.mean(axis=(1,2))
        lower, upper = np.percentile(aggregate, [2.5, 97.5])
        summaries.append({"metric": "graph_stress", "contrast": contrast,
                          "equal_domain_mean_gain_pct": estimate,
                          "bootstrap_95_low_pct": float(lower),
                          "bootstrap_95_high_pct": float(upper),
                          "bootstrap_replicates": replicates,
                          "statistical_scope": "Validation development; checkpoint selected on same split"})
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / "graphwise_validation_bootstrap_primary.csv", index=False)
    (OUT / "graphwise_validation_input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    descriptive = (run_frame.groupby(["metric", "contrast"], as_index=False)
                   .agg(mean_gain_pct=("gain_pct", "mean"),
                        median_gain_pct=("gain_pct", "median"),
                        sd_gain_pct=("gain_pct", "std"),
                        positive_runs=("gain_pct", lambda x: int((x > 0).sum())),
                        negative_runs=("gain_pct", lambda x: int((x < 0).sum()))))
    descriptive.to_csv(OUT / "graphwise_validation_descriptive.csv", index=False)
    lines = [
        "# GraphLocal-v1 graphwise Validation analysis",
        "",
        "All 9 cohorts × 4 variants are present with paired graph IDs; 36 best checkpoints were independently evaluated. "
        "The stored coordinates and all eight metric families are in `validation_graphwise/`.",
        "",
        "**Evidence level:** developmental Validation, reused for checkpoint selection. "
        "The following bootstrap distributions describe graph/seed sensitivity but are not independent Test confidence intervals.",
        "",
        "| Pair-normalized Stress contrast | Equal-domain gain (%) | Descriptive 95% bootstrap interval (%) |",
        "|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['contrast']} | {row['equal_domain_mean_gain_pct']:+.6f} | "
                     f"[{row['bootstrap_95_low_pct']:+.6f}, {row['bootstrap_95_high_pct']:+.6f}] |")
    lines += ["", "Positive gain favors the named full A or the corresponding positive factorial effect. "
              "The estimator uses each run's metric mean before forming its relative ratio, then weights seeds and domains equally. "
              "The seed draw is common across domains, and graph identities are paired inside each sampled seed occurrence.",
              "", "## Domain/seed and secondary endpoints", "",
              "All domain/seed effects for the eight metrics are in `metrics/graphwise_validation_run_effects.csv`; "
              "none were filtered by sign. Secondary values are descriptive and have not been adjusted for multiplicity. "
              "Main CAID/MA-CPSG effects and their interaction are computed from the four settings within each matched domain/seed, not from an independently retrained legacy control.",
              "Domain means, run-level descriptive summaries, and per-graph Stress sign counts are saved separately in `metrics/`.",
              "", "## Integrity", "",
              "Input CSV SHA-256 values: `metrics/graphwise_validation_input_hashes.json`. "
              "No reserved Test CSV, checkpoint change, additional model fitting, or post-hoc metric selection occurred.", ""]
    (BASE / "graphwise_validation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
