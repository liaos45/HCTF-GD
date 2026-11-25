"""Paired 2x2x2 fixed-checkpoint Validation analysis; no Test access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_v2/factorial_ablation"
OUT = BASE / "metrics"
DOMAINS = ("enzymes", "fb15k237", "sbm")
SEEDS = (17, 42, 73)
COMBOS = tuple(f"{n}{f}{c}" for n in (0, 1) for f in (0, 1) for c in (0, 1))
METRICS = ("graph_stress", "edge_crossing", "edge_length_cv", "node_occlusion_rate",
           "community_compactness", "community_separation", "community_silhouette",
           "community_sep_comp_ratio")
HIGHER = {"community_separation", "community_silhouette", "community_sep_comp_ratio"}
LOCK = json.loads((ROOT / "research_v2/formal_test/STATISTICS_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
FLOORS = LOCK["reference_floors"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, dict]:
    frames, hashes = [], {}
    for domain in DOMAINS:
        for seed in SEEDS:
            cohort = BASE / "validation_graphwise_deterministic_v2" / f"{domain}_s{seed}"
            protocol = json.loads((cohort / "PROTOCOL.json").read_text(encoding="utf-8"))
            if (protocol["status"] != "VALIDATION_FACTORIAL_COMPLETE" or protocol["test_evaluation"]
                    or not protocol["deterministic_cuda"]):
                raise RuntimeError(f"invalid cohort protocol: {cohort}")
            if protocol["max_all_on_coordinate_abs_diff"] > 1e-6 or protocol["max_all_on_hidden_abs_diff"] > 1e-6:
                raise RuntimeError(f"all-on drift: {cohort}")
            for filename, field in (("graphwise_factorial_validation.csv", "graphwise_csv_sha256"),
                                    ("all_on_equivalence.csv", "equivalence_csv_sha256"),
                                    ("coordinates.npz", "coordinates_npz_sha256")):
                path = cohort / filename
                if sha(path) != protocol[field]:
                    raise RuntimeError(f"artifact SHA mismatch: {path}")
                hashes[str(path)] = protocol[field]
            frame = pd.read_csv(cohort / "graphwise_factorial_validation.csv")
            if len(frame) != 8 * protocol["graphs"] or set(frame.combination.astype(str).str.zfill(3)) != set(COMBOS):
                raise RuntimeError(f"missing combinations: {cohort}")
            frame["combination"] = frame.combination.astype(str).str.zfill(3)
            if frame.graph_id.nunique() != protocol["graphs"] or frame.groupby("graph_id").combination.nunique().ne(8).any():
                raise RuntimeError(f"unpaired graph IDs: {cohort}")
            for metric in METRICS:
                if not np.isfinite(frame[metric]).all():
                    raise RuntimeError(f"nonfinite {metric}: {cohort}")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), hashes


def effects(values: np.ndarray, metric: str) -> dict[str, np.ndarray]:
    """Last axis follows COMBOS. All values are percent, positive favors enabled factors."""
    idx = {combo: i for i, combo in enumerate(COMBOS)}
    sign = -1.0 if metric in HIGHER else 1.0  # lower is better: off minus on
    denom = np.maximum(np.abs(values[..., idx["000"]]), FLOORS[metric])
    result = {}
    for label, off in (("111_vs_110", "110"), ("111_vs_101", "101"),
                       ("111_vs_011", "011"), ("111_vs_000", "000")):
        ref = values[..., idx[off]]
        result[label] = 100 * sign * (ref - values[..., idx["111"]]) / np.maximum(np.abs(ref), FLOORS[metric])
    # Marginal differences average over the other factor states.
    for factor, axis in (("N", 0), ("F", 1), ("C", 2)):
        deltas = []
        for combo in COMBOS:
            if combo[axis] == "0":
                on = combo[:axis] + "1" + combo[axis+1:]
                deltas.append(values[..., idx[combo]] - values[..., idx[on]])
        result[f"{factor}_main"] = 100 * sign * np.mean(deltas, axis=0) / denom
    # Difference of marginal effects: positive means the factors reinforce each other.
    for first, second, a, b in (("N_F", "N", 0, 1), ("N_C", "N", 0, 2), ("F_C", "F", 1, 2)):
        deltas = []
        for combo in COMBOS:
            if combo[a] == "0" and combo[b] == "0":
                ao = combo[:a] + "1" + combo[a+1:]
                bo = combo[:b] + "1" + combo[b+1:]
                both = list(combo); both[a] = both[b] = "1"; both = "".join(both)
                deltas.append(values[..., idx[ao]] + values[..., idx[bo]]
                              - values[..., idx[combo]] - values[..., idx[both]])
        result[f"{first}_interaction"] = 100 * sign * np.mean(deltas, axis=0) / denom
    # Three-factor interaction: N×F interaction at C=1 minus its value at C=0.
    t = (values[..., idx["101"]] + values[..., idx["011"]] - values[..., idx["001"]] - values[..., idx["111"]]
         - values[..., idx["100"]] - values[..., idx["010"]] + values[..., idx["000"]] + values[..., idx["110"]])
    result["N_F_C_interaction"] = 100 * sign * t / denom
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, hashes = load()
    rows, matrices = [], {}
    for domain in DOMAINS:
        for seed in SEEDS:
            frame = data[(data.domain == domain) & (data.seed == seed)]
            for metric in METRICS:
                matrix = frame.pivot(index="graph_id", columns="combination", values=metric)[list(COMBOS)].to_numpy(float)
                if metric == "graph_stress":
                    matrices[(domain, seed)] = matrix
                for name, value in effects(matrix.mean(axis=0), metric).items():
                    rows.append({"domain": domain, "seed": seed, "metric": metric,
                                 "contrast": name, "effect_pct": float(value), "graphs": len(matrix)})
    run = pd.DataFrame(rows)
    run.to_csv(OUT / "factorial_run_effects.csv", index=False)
    run.groupby(["domain", "metric", "contrast"], as_index=False).effect_pct.mean().to_csv(
        OUT / "factorial_domain_effects.csv", index=False)
    descriptive = (run.groupby(["metric", "contrast"], as_index=False)
                   .agg(equal_domain_mean_pct=("effect_pct", "mean"),
                        median_run_pct=("effect_pct", "median"), sd_run_pct=("effect_pct", "std"),
                        positive_runs=("effect_pct", lambda x: int((x > 0).sum())),
                        negative_runs=("effect_pct", lambda x: int((x < 0).sum()))))
    descriptive.to_csv(OUT / "factorial_descriptive.csv", index=False)

    replicates = LOCK["bootstrap"]["replicates"]
    rng = np.random.default_rng(LOCK["bootstrap"]["random_seed"])
    seed_draw = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)))
    boot = np.empty((replicates, len(DOMAINS), len(SEEDS), len(COMBOS)), dtype=float)
    for di, domain in enumerate(DOMAINS):
        for slot in range(len(SEEDS)):
            for si, seed in enumerate(SEEDS):
                selected = np.flatnonzero(seed_draw[:, slot] == si)
                if not len(selected):
                    continue
                matrix = matrices[(domain, seed)]
                graph_draw = rng.integers(0, len(matrix), size=(len(selected), len(matrix)))
                boot[selected, di, slot] = matrix[graph_draw].mean(axis=1)
    distributions = effects(boot, "graph_stress")
    primary_rows = []
    for name, values in distributions.items():
        estimate = float(run[(run.metric == "graph_stress") & (run.contrast == name)].effect_pct.mean())
        distribution = values.mean(axis=(1, 2))
        lo, hi = np.percentile(distribution, [2.5, 97.5])
        primary_rows.append({"contrast": name, "equal_domain_mean_effect_pct": estimate,
                             "bootstrap_95_low_pct": float(lo), "bootstrap_95_high_pct": float(hi),
                             "replicates": replicates,
                             "scope": "Validation, selected checkpoints; descriptive interval"})
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(OUT / "factorial_bootstrap_primary.csv", index=False)
    (OUT / "factorial_input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    lines = ["# Fixed-checkpoint N×F×C Validation analysis", "",
             "Nine cohorts, all eight combinations and paired graph identities verified. All-on replay passes the registered 1e-6 coordinate and hidden-state tolerances.", "",
             "The observed effects are conditional inference interventions at historical strict gated checkpoints. "
             "They are developmental Validation evidence, because these checkpoints were selected on Validation. "
             "They do not identify CAID or MA-CPSG training effects. The C switch disables the explicit prototype/context input; upstream community features and forest relation descriptors remain.", "",
             "| Stress contrast | Domain-equal effect (%) | Descriptive 95% bootstrap interval (%) |",
             "|---|---:|---:|"]
    for row in primary_rows:
        lines.append(f"| {row['contrast']} | {row['equal_domain_mean_effect_pct']:+.6f} | "
                     f"[{row['bootstrap_95_low_pct']:+.6f}, {row['bootstrap_95_high_pct']:+.6f}] |")
    lines += ["", "Positive values favor the enabled component or Full. Main effects average over the other factor states. "
              "Two-factor interaction measures the change in one factor's marginal gain when the other is enabled. "
              "All factorial effects use the 000 run mean as a fixed normalization reference; direct 111 contrasts use their named comparator as reference.",
              "", "Domain, seed, secondary-metric, and sign summaries are preserved in `metrics/`. No negative contrasts were removed.",
              "The bootstrap resamples shared seed indices and paired original graph identities, then averages domains equally. "
              "Intervals are descriptive, endpoint-wise, and do not constitute independent Test inference.", ""]
    (BASE / "FINAL_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
