"""Locked, raw-coordinate, graphwise Validation audit for GraphLocal-v1.

This evaluates existing best checkpoints only. It never runs training, the
reserved Test split, SafeBlend, or post-hoc layout optimization. The historical
loader materializes graph inputs before splitting; only the Validation list is
passed to a model or metric here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "research_v2/graph_local/source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
import train_graphlocal_v1 as train  # noqa: E402
from evaluate_extra_metrics import (  # noqa: E402
    community_metrics,
    edge_crossing_count,
    edge_length_variance,
    node_occlusion_rate,
    unique_undirected_edges,
)

VARIANTS = ("A_both", "B_caid_only", "C_macpsg_only", "D_neither")
BASE = ROOT / "research_v2/graph_local"
OUTPUT = BASE / "validation_graphwise"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_from_checkpoint(path: Path, device: str):
    checkpoint = train.torch_load_trusted(str(path), map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    if not getattr(model_args, "graph_local_community_objectives", False):
        raise RuntimeError(f"not a graph-local checkpoint: {path}")
    if getattr(model_args, "split_mode", None) != "size_ood":
        raise RuntimeError(f"not a size-OOD checkpoint: {path}")
    model_args.mode = "eval"
    model_args.current_epoch = int(checkpoint.get("epoch", 0))
    model_args.stress_graph_balance_power = 0.0  # training-only
    model = train.build_deepgd_model(model_args, device)
    train.load_checkpoint(str(path), model, None, device, strict=True)
    model.eval()
    return model_args, model


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def evaluate(domain: str, seed: int, device: str) -> None:
    name = f"{domain}_s{seed}"
    out = OUTPUT / name
    final_protocol = out / "PROTOCOL.json"
    if final_protocol.exists():
        raise RuntimeError(f"completed cohort exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    models = {}
    for variant in VARIANTS:
        run = f"HCTF-GD-GraphLocal-v1_{name}_{variant}"
        status_path = BASE / "logs" / f"{run}_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] != "VALIDATION_COMPLETE" or status["test_csv_exists"]:
            raise RuntimeError(f"invalid cohort status: {run}")
        path = BASE / "checkpoints" / f"{run}_best.pt"
        digest = sha256(path)
        if digest != status["best_checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {run}")
        paths[variant] = {"path": str(path), "sha256": digest}
        models[variant] = model_from_checkpoint(path, device)

    first_args = models[VARIANTS[0]][0]
    if first_args.dataset.lower() != domain or int(first_args.seed) != seed:
        raise RuntimeError("checkpoint domain/seed mismatch")
    for variant in VARIANTS[1:]:
        args = models[variant][0]
        if args.dataset != first_args.dataset or int(args.seed) != seed:
            raise RuntimeError("four checkpoint inputs are not matched")
    train.set_random_seed(seed)
    _, validation, test_reserved = train.load_graph_dataset(first_args)
    del test_reserved  # no reserved graph enters a loader or a metric
    manifest = BASE / "manifests" / f"{name}_size_ood.csv"
    expected = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            if item["split"] == "validation":
                expected[item["graph_id"]] = item
    observed = [str(data.name) for data in validation]
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise RuntimeError("Validation identities differ from locked metadata manifest")
    loader = train.make_loader(validation, 1, shuffle=False)
    stress_metric = train.Stress()
    rows = []
    coordinates = {}
    with torch.no_grad():
        batches = iter(loader)
        for graph_index in range(len(loader)):
            # Dataset preprocessing and forward may consume random numbers.
            train.set_random_seed(seed + 1009 * graph_index + 1)
            batch = next(batches).to(device)
            record = train.mtpf_build_local_graph_records(batch)
            if len(record) != 1:
                raise RuntimeError("batch size 1 did not yield one local graph")
            record = record[0]
            graph_id = str(validation[graph_index].name)
            meta = expected[graph_id]
            n = record["num_nodes"]
            edges = record["edge_index"]
            edge_count = len(unique_undirected_edges(edges))
            if n != int(meta["node_count"]) or edge_count != int(meta["edge_count_undirected"]):
                raise RuntimeError(f"graph topology metadata mismatch: {graph_id}")
            labels = batch.community_label[record["global_nodes_device"]].detach().cpu().numpy()
            for variant in VARIANTS:
                model_args, model = models[variant]
                train.set_random_seed(seed + 1009 * graph_index + 2)
                if device == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                _, _, raw_pos, _ = train.forward_model(batch, model, None, stress_metric, model_args)
                if device == "cuda":
                    torch.cuda.synchronize()
                runtime = time.perf_counter() - started
                xy = raw_pos[record["global_nodes_device"]].detach().cpu().float()
                if xy.shape != (n, 2) or not torch.isfinite(xy).all():
                    raise RuntimeError(f"invalid coordinates: {graph_id}, {variant}")
                crossing, crossing_norm = edge_crossing_count(xy, edges)
                _, edge_cv = edge_length_variance(xy, edges)
                occlusion, _ = node_occlusion_rate(xy, edge_index=edges)
                community = community_metrics(xy, labels)
                stress_sum = train.local_deepgd_stress(xy, record["perm_index"], record["apsp_attr"])
                stress = stress_sum / max(1, n * (n - 1))
                key = f"{variant}__g{graph_index:04d}"
                coordinates[key] = xy.numpy().copy()
                row = {
                    "domain": domain, "seed": seed, "graph_index": graph_index,
                    "graph_id": graph_id, "variant": variant, "num_nodes": n,
                    "num_edges": edge_count, "coordinate_key": key,
                    "graph_stress": stress, "stress_pair_sum": stress_sum,
                    "edge_crossing": crossing, "edge_crossing_norm": crossing_norm,
                    "edge_length_cv": edge_cv, "node_occlusion_rate": occlusion,
                    "community_compactness": community["community_compactness"],
                    "community_separation": community["community_separation"],
                    "community_silhouette": community["community_silhouette"],
                    "community_sep_comp_ratio": community["community_sep_comp_ratio"],
                    "forward_seconds": runtime,
                    "checkpoint_sha256": paths[variant]["sha256"],
                }
                for metric in ("graph_stress", "edge_length_cv", "node_occlusion_rate",
                               "community_compactness", "community_separation",
                               "community_silhouette", "community_sep_comp_ratio"):
                    if not math.isfinite(float(row[metric])):
                        raise RuntimeError(f"nonfinite {metric}: {graph_id}, {variant}")
                rows.append(row)
            if (graph_index + 1) % 25 == 0 or graph_index + 1 == len(validation):
                print(f"[{name}] Validation graphs {graph_index + 1}/{len(validation)}", flush=True)

    csv_path = out / "graphwise_validation.csv"
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(csv_path)
    npz_path = out / "coordinates.npz"
    tmp_npz = out / "coordinates.npz.tmp"
    with tmp_npz.open("wb") as stream:
        np.savez_compressed(stream, **coordinates)
    tmp_npz.replace(npz_path)
    atomic_json(final_protocol, {
        "status": "VALIDATION_GRAPHWISE_COMPLETE", "domain": domain, "seed": seed,
        "model_id": "HCTF-GD-GraphLocal-v1", "split": "validation",
        "graphs": len(validation), "rows": len(rows), "variants": list(VARIANTS),
        "checkpoint_paths_and_hashes": paths,
        "manifest_sha256": sha256(manifest),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "graphwise_csv_sha256": sha256(csv_path), "coordinates_npz_sha256": sha256(npz_path),
        "primary_metric": "stored-forward pair-normalized stress: local DeepGD pair-sum / n(n-1)",
        "runtime_scope": "one graph forward_model, GPU synchronized; excludes dataset construction and CPU metric audits",
        "community_labels": "checkpoint dataset's graph-local supplied labels",
        "selection_note": "Validation was used to select these checkpoints; results are developmental, not independent Test evidence.",
        "reserved_test_evaluation": False,
        "device": device,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("enzymes", "fb15k237", "sbm"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 73), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    evaluate(args.domain, args.seed, args.device)


if __name__ == "__main__":
    main()
