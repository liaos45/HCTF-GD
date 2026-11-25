"""Matched counterfactual audit of HCTF bidirectionality and structure gating.

The evaluator changes no learned parameter.  A single frozen full checkpoint is
run normally, with the structural gate forced to zero or one, and with either
the top-down or bottom-up forest messages removed.  The exact matched backbone
layout is evaluated in the same process and remains the SafeBlend fallback.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import evaluate_hctf_safe_blend as evaluator
from hctf_fast_audit import install_fast_audit
from hctf_q1_controls import choose_safe, graph_from_edges


install_fast_audit(evaluator)
ROOT, train = evaluator.ROOT, evaluator.train
SOURCES = (
    "full",
    "same_weight_gate_zero",
    "same_weight_gate_one",
    "top_down_only",
    "bottom_up_only",
)


def checkpoint_paths(dataset: str, seed: int, mode: str) -> dict[str, Path]:
    directory, prefix = (
        ("hctf_strict_size_ood", "strictsize")
        if mode == "strict"
        else ("hctf_clean_id", "cleanid")
    )
    folder = ROOT / "checkpoints" / directory
    return {
        "baseline": folder / f"{prefix}_{dataset}_baseline_s{seed}_best.pt",
        "full": folder
        / f"{prefix}_{dataset}_adaptive_supervised_full_s{seed}_best.pt",
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "mode": args.mode,
        "sources": list(SOURCES),
        "alphas": list(__import__("hctf_q1_controls").ALPHAS),
        "selection_metrics": list(evaluator.METRICS),
        "max_graphs": args.max_graphs,
        "partial": args.max_graphs > 0,
        "same_weight_counterfactuals": True,
        "checkpoint_sha256": {name: file_hash(path) for name, path in paths.items()},
        "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in [
                Path(__file__).resolve(),
                ROOT / "deepgd" / "model" / "generator" / "hctf.py",
                ROOT / "deepgd" / "model" / "generator" / "generator.py",
                ROOT / "scripts" / "hctf_q1_controls.py",
                ROOT / "scripts" / "hctf_fast_audit.py",
            ]
        },
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    paths = checkpoint_paths(args.dataset, args.seed, args.mode)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    audit_dir = ROOT / "results" / (
        "hctf_strict_size_ood" if args.mode == "strict" else "hctf_clean_id"
    )
    marker_path = audit_dir / f"COMPLETE_{args.dataset}_s{args.seed}.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["partial"] or not marker["frozen_common_backbone_state_verified"]:
        raise RuntimeError("A full, common-backbone checkpoint audit is required")

    output = ROOT / args.output_dir / f"{args.mode}_{args.dataset}_s{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    protocol = contract(args, paths)
    protocol_path = output / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol or source hashes changed; use a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.skip_completed and (output / "COMPLETE.json").exists():
        print(f"[SKIP] {output}", flush=True)
        return 0

    device = train.get_device()
    baseline_args, baseline_model = evaluator.load_model(paths["baseline"], device)
    full_args, full_model = evaluator.load_model(paths["full"], device)
    baseline_args._hctf_teacher_enabled = False
    full_args._hctf_teacher_enabled = False
    train.set_random_seed(args.seed)
    data_list = train.load_graph_dataset(baseline_args)[2]
    metadata = pd.read_csv(
        audit_dir / f"graph_metadata_{args.dataset}_s{args.seed}.csv"
    ).sort_values("test_graph_id")
    if [str(data.name) for data in data_list] != metadata.graph_name.astype(str).tolist():
        raise RuntimeError("Evaluation graph identity/order differs from the checkpoint audit")
    if args.max_graphs > 0:
        data_list = data_list[: args.max_graphs]
    loader = train.make_loader(data_list, int(baseline_args.batch_size), shuffle=False)

    rows: list[dict] = []
    graph_id = 0
    for batch_id, batch in enumerate(loader):
        batch = batch.to(device)
        positions: dict[str, torch.Tensor] = {}
        forward_seconds: dict[str, float] = {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        _, positions["baseline"], _, _ = train.forward_model(
            batch, baseline_model, None, evaluator.Stress(), baseline_args
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_seconds["baseline"] = time.perf_counter() - started

        counterfactuals = {
            "full": (None, None),
            "same_weight_gate_zero": (0.0, None),
            "same_weight_gate_one": (1.0, None),
            "top_down_only": (None, "top_down"),
            "bottom_up_only": (None, "bottom_up"),
        }
        try:
            for source, (gate_override, flow_override) in counterfactuals.items():
                full_model.set_hctf_structure_gate_override(gate_override)
                full_model.set_hctf_tree_flow_override(flow_override)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                _, positions[source], _, _ = train.forward_model(
                    batch, full_model, None, evaluator.Stress(), full_args
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_seconds[source] = time.perf_counter() - started
        finally:
            full_model.set_hctf_structure_gate_override(None)
            full_model.set_hctf_tree_flow_override(None)

        records = train.mtpf_build_local_graph_records(batch)
        for record in records:
            nodes = record["global_nodes_device"]
            name = str(data_list[graph_id].name)
            stable_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 2147483647
            if args.dataset == "sbm":
                labels = train._community_labels_from_graph(
                    data_list[graph_id], "ground_truth", stable_seed
                )
                reference = "planted_sbm"
            else:
                labels = evaluator.detect_communities(
                    record["edge_index"], record["num_nodes"]
                )
                reference = "fixed_greedy_proxy"

            baseline, baseline_metrics = evaluator.measure_layout(
                positions["baseline"][nodes], record, labels, include_holdout=False
            )
            baseline = baseline.cpu()
            baseline_metrics["graph_stress"] = baseline_metrics.pop("batch_stress") / (
                record["num_nodes"] * (record["num_nodes"] - 1)
            )
            audit_baseline = {
                **baseline_metrics,
                "batch_stress": baseline_metrics["graph_stress"]
                * record["num_nodes"]
                * (record["num_nodes"] - 1),
            }
            common = {
                "dataset": args.dataset,
                "seed": args.seed,
                "mode": args.mode,
                "graph_name": name,
                "batch_id": batch_id,
                "local_graph_id": graph_id,
                "num_nodes": record["num_nodes"],
                "num_edges": int(graph_from_edges(record["num_nodes"], record["edge_index"]).number_of_edges()),
                "reference_partition": reference,
                "timing_isolated": False,
            }
            rows.append(
                {
                    **common,
                    **baseline_metrics,
                    "source": "baseline",
                    "stage": "baseline",
                    "alpha": 0.0,
                    "generation_seconds": forward_seconds["baseline"] / len(records),
                    "audit_seconds": 0.0,
                }
            )
            for source in SOURCES:
                started = time.perf_counter()
                raw, raw_metrics = evaluator.measure_layout(
                    positions[source][nodes], record, labels, include_holdout=False
                )
                chosen, chosen_metrics, alpha = choose_safe(
                    baseline,
                    audit_baseline,
                    raw,
                    record,
                    labels,
                    evaluator,
                )
                audit_seconds = time.perf_counter() - started
                for stage, metrics in (("raw", raw_metrics), ("safe", chosen_metrics)):
                    export = dict(metrics)
                    export["graph_stress"] = export.pop("batch_stress") / (
                        record["num_nodes"] * (record["num_nodes"] - 1)
                    )
                    rows.append(
                        {
                            **common,
                            **export,
                            "source": source,
                            "stage": stage,
                            "alpha": alpha if stage == "safe" else 1.0,
                            "generation_seconds": forward_seconds[source] / len(records),
                            "audit_seconds": audit_seconds,
                        }
                    )
            graph_id += 1
            if graph_id % 10 == 0:
                print(
                    f"[MECHANISM] {args.mode} {args.dataset} s{args.seed} "
                    f"{graph_id}/{len(data_list)}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "observations.csv", index=False)
    complete = {
        **protocol,
        "graphs": graph_id,
        "rows": len(frame),
        "all_sources_complete": graph_id == len(data_list),
        "reference_partition": reference,
    }
    (output / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(f"[MECHANISM COMPLETE] {output} {graph_id} graphs", flush=True)
    del baseline_model, full_model, data_list, batch, positions
    gc.collect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["strict", "clean-id"], default="strict")
    parser.add_argument("--max-graphs", type=int, default=-1)
    parser.add_argument(
        "--output-dir", default="results/hctf_mechanistic_counterfactuals"
    )
    parser.add_argument("--skip-completed", action="store_true")
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
