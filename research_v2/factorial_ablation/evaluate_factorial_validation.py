"""Fixed-checkpoint N×F×C HCTF interventions on Validation graphs only.

N disables the node-residual network output; F disables forest messages; C
disables the explicit community-prototype context vector. All graph and node
gate outputs are recorded on the unmodified Full pass and replayed at the same
eligible insertion in every variant. This measures conditional inference
effects at one historical strict gated checkpoint, not retraining effects.
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

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "results/hctf_strict_size_ood/source_snapshot_strict"
sys.path.insert(0, str(SNAPSHOT))
import train_cser_spectral_mog_end2end as train  # noqa: E402
from evaluate_extra_metrics import (  # noqa: E402
    community_metrics, edge_crossing_count, edge_length_variance,
    node_occlusion_rate, unique_undirected_edges,
)

BASE = ROOT / "research_v2/factorial_ablation"
TOL_ABS = 1e-6
TOL_REL = 1e-6
COMBINATIONS = tuple(f"{n}{f}{c}" for n in (0, 1) for f in (0, 1) for c in (0, 1))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def load_model(checkpoint_path: Path, device: str):
    checkpoint = train.torch_load_trusted(str(checkpoint_path), map_location="cpu")
    args = argparse.Namespace(**checkpoint["args"])
    if (not args.use_hctf or not args.hctf_adaptive_structure_gate or args.hctf_graph_gate
            or args.hctf_version != "v1" or args.hctf_num_trees != 3
            or args.hctf_coordinate_residual or args.split_mode != "size_ood"):
        raise RuntimeError("checkpoint is not the registered three-forest gated strict model")
    args.mode = "eval"
    args.current_epoch = int(checkpoint.get("epoch", 0))
    args._hctf_teacher_enabled = False
    model = train.build_deepgd_model(args, device)
    train.load_checkpoint(str(checkpoint_path), model, None, device, strict=True)
    model.eval()
    block = model.hctf_block
    if block is None or block.node_update_net is None or block.structure_gate is None:
        raise RuntimeError("missing gated HCTF intervention modules")
    if any(module.training for module in model.modules()):
        raise RuntimeError("dropout or BatchNorm remains in training mode")
    expected_calls = sum(
        index % max(int(model.params.hctf_block_stride), 1) == 0
        for index in range(len(model.block_list) - 1)
    )
    if model.params.hctf_late_only:
        expected_calls = 1
    if expected_calls < 1:
        raise RuntimeError("no eligible HCTF insertion")
    return args, model, expected_calls


def raw_forward(batch, args, model, stress_metric, seed: int, graph_index: int,
                device: str, combo: str | None, gate_record=None):
    """Return coordinates, gate tensors and intermediate HCTF outputs."""
    block = model.hctf_block
    original_tree = block.tree_messages_enabled
    original_context = block.prototype_context_enabled
    handles = []
    structure_values = []
    adaptive_values = []
    block_values = []
    counters = {"structure": 0, "adaptive": 0}

    def capture_structure(_module, _inputs, output):
        structure_values.append(output.detach().clone())

    def capture_adaptive(_module, _inputs, output):
        adaptive_values.append(output.detach().clone())

    def replay(name, values):
        def callback(_module, _inputs, output):
            index = counters[name]
            counters[name] += 1
            if index >= len(values) or output.shape != values[index].shape:
                raise RuntimeError(f"{name} gate replay count/shape mismatch")
            return values[index]
        return callback

    def capture_block(_module, _inputs, output):
        block_values.append(output.detach().clone())

    try:
        if gate_record is None:
            handles.append(block.structure_gate.register_forward_hook(capture_structure))
            handles.append(block.adaptive_gate.register_forward_hook(capture_adaptive))
        else:
            handles.append(block.structure_gate.register_forward_hook(replay("structure", gate_record["structure"])))
            handles.append(block.adaptive_gate.register_forward_hook(replay("adaptive", gate_record["adaptive"])))
        handles.append(block.register_forward_hook(capture_block))
        if combo is not None:
            block.tree_messages_enabled = combo[1] == "1"
            block.prototype_context_enabled = combo[2] == "1"
            if combo[0] == "0":
                handles.append(block.node_update_net.register_forward_hook(
                    lambda _module, _inputs, output: torch.zeros_like(output)
                ))
        train.set_random_seed(seed + 1009 * graph_index + 2)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            _, _, coordinates, _ = train.forward_model(batch, model, None, stress_metric, args)
        if device == "cuda":
            torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        if gate_record is not None:
            if counters["structure"] != len(gate_record["structure"]) or counters["adaptive"] != len(gate_record["adaptive"]):
                raise RuntimeError("not all recorded gate values were replayed")
        return coordinates.detach(), {
            "structure": structure_values, "adaptive": adaptive_values,
            "hctf": block_values,
        }, runtime
    finally:
        for handle in handles:
            handle.remove()
        block.tree_messages_enabled = original_tree
        block.prototype_context_enabled = original_context


def evaluate(domain: str, seed: int, device: str, limit: int | None) -> None:
    checkpoint = ROOT / "checkpoints/hctf_strict_size_ood" / f"strictsize_{domain}_adaptive_supervised_full_s{seed}_best.pt"
    checkpoint_hash = sha(checkpoint)
    args, model, expected_calls = load_model(checkpoint, device)
    if args.dataset.lower() != domain or int(args.seed) != seed:
        raise RuntimeError("checkpoint dataset/seed mismatch")
    out = BASE / ("smoke" if limit else "validation_graphwise") / f"{domain}_s{seed}"
    if (out / "PROTOCOL.json").exists():
        raise RuntimeError(f"completed factorial cohort exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    train.set_random_seed(seed)
    _, validation, test_reserved = train.load_graph_dataset(args)
    del test_reserved
    manifest = ROOT / "research_v2/graph_local/manifests" / f"{domain}_s{seed}_size_ood.csv"
    expected_names = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            if item["split"] == "validation":
                expected_names[item["graph_id"]] = item
    names = [str(data.name) for data in validation]
    if set(names) != set(expected_names) or len(names) != len(expected_names):
        raise RuntimeError("Validation graph IDs do not match the locked manifest")
    if limit is not None:
        validation = validation[:limit]
    loader = train.make_loader(validation, 1, shuffle=False)
    metric = train.Stress()
    rows, equivalence, coords = [], [], {}
    with torch.no_grad():
        iterator = iter(loader)
        for graph_index in range(len(loader)):
            train.set_random_seed(seed + 1009 * graph_index + 1)
            batch = next(iterator).to(device)
            record = train.mtpf_build_local_graph_records(batch)
            if len(record) != 1:
                raise RuntimeError("expected one graph per batch")
            record = record[0]
            graph_id = str(validation[graph_index].name)
            n = record["num_nodes"]
            edges = record["edge_index"]
            num_edges = len(unique_undirected_edges(edges))
            meta = expected_names[graph_id]
            if n != int(meta["node_count"]) or num_edges != int(meta["edge_count_undirected"]):
                raise RuntimeError(f"topology mismatch: {graph_id}")
            labels = batch.community_label[record["global_nodes_device"]].detach().cpu().numpy()
            raw, gates, native_time = raw_forward(batch, args, model, metric, seed, graph_index, device, None)
            if not (len(gates["structure"]) == len(gates["adaptive"]) == len(gates["hctf"]) == expected_calls):
                raise RuntimeError("Full HCTF insertion count mismatch")
            replay, replay_gates, replay_time = raw_forward(batch, args, model, metric, seed,
                                                             graph_index, device, "111", gates)
            if len(replay_gates["hctf"]) != expected_calls:
                raise RuntimeError("replayed HCTF insertion count mismatch")
            max_coord_diff = float((raw - replay).abs().max().item())
            max_hidden_diff = max(float((left - right).abs().max().item())
                                  for left, right in zip(gates["hctf"], replay_gates["hctf"]))
            if not torch.allclose(raw, replay, atol=TOL_ABS, rtol=TOL_REL) or any(
                not torch.allclose(left, right, atol=TOL_ABS, rtol=TOL_REL)
                for left, right in zip(gates["hctf"], replay_gates["hctf"])
            ):
                raise RuntimeError(f"all-on numerical equivalence failed: {graph_id}")
            equivalence.append({
                "domain": domain, "seed": seed, "graph_id": graph_id,
                "hctf_insertions": expected_calls, "max_coordinate_abs_diff": max_coord_diff,
                "max_hctf_hidden_abs_diff": max_hidden_diff,
                "native_forward_seconds": native_time, "replay_forward_seconds": replay_time,
            })
            for combo in COMBINATIONS:
                if combo == "111":
                    raw_pos, runtime = replay, replay_time
                else:
                    raw_pos, result, runtime = raw_forward(batch, args, model, metric,
                                                           seed, graph_index, device, combo, gates)
                    if len(result["hctf"]) != expected_calls:
                        raise RuntimeError(f"{combo} insertion count mismatch")
                xy = raw_pos[record["global_nodes_device"]].detach().cpu().float()
                if xy.shape != (n, 2) or not torch.isfinite(xy).all():
                    raise RuntimeError(f"invalid coordinate array: {graph_id} {combo}")
                crossing, crossing_norm = edge_crossing_count(xy, edges)
                _, edge_cv = edge_length_variance(xy, edges)
                occlusion, _ = node_occlusion_rate(xy, edge_index=edges)
                community = community_metrics(xy, labels)
                stress_sum = train.local_deepgd_stress(xy, record["perm_index"], record["apsp_attr"])
                key = f"{combo}__g{graph_index:04d}"
                coords[key] = xy.numpy().copy()
                row = {
                    "domain": domain, "seed": seed, "graph_index": graph_index,
                    "graph_id": graph_id, "combination": combo,
                    "node_residual_on": int(combo[0]), "forest_on": int(combo[1]),
                    "prototype_context_on": int(combo[2]),
                    "num_nodes": n, "num_edges": num_edges, "coordinate_key": key,
                    "graph_stress": stress_sum / max(1, n * (n-1)),
                    "stress_pair_sum": stress_sum,
                    "edge_crossing": crossing, "edge_crossing_norm": crossing_norm,
                    "edge_length_cv": edge_cv, "node_occlusion_rate": occlusion,
                    "community_compactness": community["community_compactness"],
                    "community_separation": community["community_separation"],
                    "community_silhouette": community["community_silhouette"],
                    "community_sep_comp_ratio": community["community_sep_comp_ratio"],
                    "forward_seconds_with_hooks": runtime,
                    "checkpoint_sha256": checkpoint_hash,
                }
                for field in ("graph_stress", "edge_length_cv", "node_occlusion_rate",
                              "community_compactness", "community_separation",
                              "community_silhouette", "community_sep_comp_ratio"):
                    if not math.isfinite(float(row[field])):
                        raise RuntimeError(f"nonfinite {field}: {graph_id}, {combo}")
                rows.append(row)
            if (graph_index + 1) % 25 == 0 or graph_index + 1 == len(loader):
                print(f"[{domain} s{seed}] factorial Validation {graph_index + 1}/{len(loader)}", flush=True)

    for filename, values in (("graphwise_factorial_validation.csv", rows),
                             ("all_on_equivalence.csv", equivalence)):
        target = out / filename
        temp = target.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        temp.replace(target)
    tmp = out / "coordinates.npz.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **coords)
    tmp.replace(out / "coordinates.npz")
    atomic_json(out / "PROTOCOL.json", {
        "status": "SMOKE_ONLY" if limit else "VALIDATION_FACTORIAL_COMPLETE",
        "domain": domain, "seed": seed, "graphs": len(validation),
        "combinations": list(COMBINATIONS), "rows": len(rows),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "source_hctf_sha256": sha(SNAPSHOT / "deepgd/model/generator/hctf.py"),
        "evaluation_script_sha256": sha(Path(__file__)),
        "manifest_sha256": sha(manifest), "all_on_abs_tolerance": TOL_ABS,
        "all_on_rel_tolerance": TOL_REL, "expected_hctf_insertions": expected_calls,
        "max_all_on_coordinate_abs_diff": max(row["max_coordinate_abs_diff"] for row in equivalence),
        "max_all_on_hidden_abs_diff": max(row["max_hctf_hidden_abs_diff"] for row in equivalence),
        "graphwise_csv_sha256": sha(out / "graphwise_factorial_validation.csv"),
        "equivalence_csv_sha256": sha(out / "all_on_equivalence.csv"),
        "coordinates_npz_sha256": sha(out / "coordinates.npz"),
        "gate_control": "replay unmodified Full structure_gate and adaptive_gate outputs per insertion",
        "community_factor_scope": "explicit prototype/context vector; forest relation descriptors and upstream community features retained",
        "metric_scope": "Validation, selected checkpoint; pair-normalized stress is pair-sum / n(n-1)",
        "test_evaluation": False, "device": device,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("enzymes", "fb15k237", "sbm"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 73), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-one-graph", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    evaluate(args.domain, args.seed, args.device, 1 if args.smoke_one_graph else None)


if __name__ == "__main__":
    main()
