"""Crossed graph/seed bootstrap for matched HCTF mechanism counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_hctf_adaptive_results import CORE_METRICS, hierarchical_bootstrap
from evaluate_hctf_mechanistic_counterfactuals import SOURCES


ROOT = Path(__file__).resolve().parents[1]
# The mechanism evaluator intentionally records only the ten metrics that enter
# SafeBlend.  The seven held-out perceptual endpoints belong to the separate
# post-selection audit and must not be silently required here.
METRICS = CORE_METRICS
COUNTERFACTUALS = (
    "same_weight_gate_zero",
    "same_weight_gate_one",
    "top_down_only",
    "bottom_up_only",
)


def load_frames(root: Path, mode: str, datasets: list[str], seeds: list[int]):
    frames = {}
    missing = []
    expected = {("baseline", "baseline")} | {
        (source, stage) for source in SOURCES for stage in ("raw", "safe")
    }
    for dataset in datasets:
        for seed in seeds:
            folder = root / f"{mode}_{dataset}_s{seed}"
            marker_path = folder / "COMPLETE.json"
            if not marker_path.exists():
                missing.append(f"{dataset}/s{seed}")
                continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker["partial"] or not marker["all_sources_complete"]:
                raise RuntimeError(f"Partial group cannot enter formal analysis: {folder}")
            frame = pd.read_csv(folder / "observations.csv")
            if frame.duplicated(["graph_name", "source", "stage"]).any():
                raise RuntimeError(f"Duplicate source rows: {folder}")
            for _, group in frame.groupby("graph_name"):
                if set(zip(group.source, group.stage)) != expected:
                    raise RuntimeError(f"Incomplete source rows: {folder}")
            if not np.isfinite(frame[METRICS].to_numpy()).all():
                raise RuntimeError(f"Non-finite metrics: {folder}")
            frames[(dataset, seed)] = frame
    if missing:
        raise RuntimeError(f"Formal mechanism analysis requires every group: {missing}")
    return frames


def paired(frames, reference: tuple[str, str], trial: tuple[str, str]):
    pairs = {}
    for key, frame in frames.items():
        left = frame[(frame.source == reference[0]) & (frame.stage == reference[1])]
        right = frame[(frame.source == trial[0]) & (frame.stage == trial[1])]
        joined = left.merge(
            right,
            on=["graph_name", "batch_id", "local_graph_id"],
            suffixes=("_node", "_full"),
            validate="one_to_one",
        )
        if len(joined) != len(left) or len(joined) != len(right):
            raise RuntimeError(f"Unpaired graph rows: {key}")
        joined["_graph_cluster_id"] = joined.graph_name.astype(str)
        pairs[key] = joined
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="results/hctf_mechanistic_counterfactuals"
    )
    parser.add_argument(
        "--output-dir", default="results/hctf_mechanistic_counterfactuals_analysis"
    )
    parser.add_argument("--mode", choices=["strict", "clean-id"], default="strict")
    parser.add_argument(
        "--datasets", nargs="+", default=["enzymes", "fb15k237", "sbm"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()

    frames = load_frames(
        ROOT / args.input_dir, args.mode, args.datasets, args.seeds
    )
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    observations = pd.concat(frames.values(), ignore_index=True)
    observations.groupby(["dataset", "seed", "source", "stage"])[
        METRICS + ["generation_seconds", "audit_seconds"]
    ].mean().reset_index().to_csv(output / "absolute_seed_means.csv", index=False)
    observations[observations.stage == "safe"].assign(
        accepted=lambda frame: frame.alpha > 0
    ).groupby(["dataset", "seed", "source"]).agg(
        graphs=("graph_name", "size"),
        accepted=("accepted", "sum"),
        acceptance_rate=("accepted", "mean"),
    ).reset_index().to_csv(output / "acceptance.csv", index=False)

    tables = []
    for stage in ("raw", "safe"):
        for reference in COUNTERFACTUALS:
            comparison = f"full_{stage}_vs_{reference}_{stage}"
            table = hierarchical_bootstrap(
                paired(frames, (reference, stage), ("full", stage)),
                args.bootstrap_repeats,
                20261009,
                metrics=METRICS,
            )
            table["comparison"] = comparison
            table["counterfactual_type"] = "same_weight_inference"
            tables.append(table)
            print(f"[MECHANISM CI] {comparison}", flush=True)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output / "crossed_bootstrap_ci.csv", index=False)
    complete = {
        "complete": True,
        "mode": args.mode,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "paired_groups": len(frames),
        "counterfactuals": list(COUNTERFACTUALS),
        "intervals": (
            "Endpoint-wise crossed graph/seed bootstrap; exploratory rather than "
            "simultaneous family-wise intervals"
        ),
    }
    (output / "COMPLETE.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# HCTF matched mechanism counterfactuals\n\n"
        "A single frozen full checkpoint is compared with gate zero, gate one, "
        "top-down-only and bottom-up-only inference. No learned parameter changes.\n\n"
        "Raw and SafeBlend-selected results are kept separate. Positive gains use "
        "the predeclared metric direction. Intervals are endpoint-wise and should "
        "not be described as multiplicity-controlled family-wise claims.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
