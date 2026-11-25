"""Shared-input full/node SafeBlend controls with unselected visual endpoints.

Baseline inference is performed once. Each candidate has its own line-search
and fallback; candidates never compete against each other during acceptance.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch
import evaluate_hctf_safe_blend as evaluator
from hctf_fast_audit import install_fast_audit

install_fast_audit(evaluator)

from evaluate_hctf_safe_blend import (
    ROOT, METRICS, HOLDOUT_METRICS, train, Stress, load_model, measure_layout,
    detect_communities, improvement, is_componentwise_safe,
    objective_metric_weights, row_with_metadata, unique_undirected_edges,
    independent_visual_metrics,
)

DATASETS = ["sbm", "fb15k237", "imdb_binary", "enzymes", "proteins", "wn18rr", "north", "rome"]
VARIANTS = ["full_nodrop", "node_only_nodrop"]


@torch.no_grad()
def evaluate(args):
    checkpoint_dir = ROOT / args.checkpoint_dir
    device = train.get_device()
    dataset, seed = args.datasets[0], args.seeds[0]
    baseline_path = checkpoint_dir / f"submission_{dataset}_baseline_s{seed}_best.pt"
    baseline_args, baseline = load_model(baseline_path, device)
    candidates = []
    for variant in VARIANTS:
        path = checkpoint_dir / f"submission_{dataset}_{variant}_s{seed}_best.pt"
        candidate_args, model = load_model(path, device)
        candidates.append((variant, candidate_args, model))
    train.set_random_seed(int(baseline_args.seed))
    _, _, test_data = train.load_graph_dataset(baseline_args)
    loader = train.make_loader(test_data, int(baseline_args.batch_size), shuffle=False)
    alphas = [0.0625, .125, .25, .5, .75, 1.]
    weights = objective_metric_weights("")
    rows = {variant: [] for variant in VARIANTS}
    selections = {variant: [] for variant in VARIANTS}
    stress = Stress()
    graph_id = 0
    for batch_id, batch in enumerate(loader):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break
        batch = batch.to(device)
        _, seed_batch, _, _ = train.forward_model(batch, baseline, None, stress, baseline_args)
        proposals = []
        for variant, candidate_args, model in candidates:
            _, proposal, _, _ = train.forward_model(batch, model, None, stress, candidate_args)
            proposals.append((variant, proposal))
        for record in train.mtpf_build_local_graph_records(batch):
            shared_started = time.perf_counter()
            nodes = record["global_nodes_device"]
            labels = detect_communities(record["edge_index"], record["num_nodes"])
            seed_pos, seed_metrics = measure_layout(seed_batch[nodes], record, labels, include_holdout=False)
            shared_seconds = time.perf_counter() - shared_started
            for variant, proposal_batch in proposals:
                started = time.perf_counter()
                candidate, candidate_metrics = measure_layout(proposal_batch[nodes], record, labels, include_holdout=False)
                aligned = train.mtpf_procrustes_align(candidate, seed_pos)
                best_pos, best_metrics, best_score, best_alpha = seed_pos, seed_metrics, 0., 0.
                for alpha in alphas:
                    trial_pos, metrics = measure_layout(seed_pos + alpha * (aligned - seed_pos), record, labels, include_holdout=False)
                    if not is_componentwise_safe(seed_metrics, metrics, relative_tolerance=0.):
                        continue
                    gains = [improvement(seed_metrics, metrics, metric) for metric in METRICS]
                    score = sum(weights[metric] * gain for metric, gain in zip(METRICS, gains))
                    if any(gain > 1e-10 for gain in gains) and (
                        score > best_score + 1e-12 or (abs(score-best_score) <= 1e-12 and alpha > best_alpha)
                    ):
                        best_pos, best_metrics, best_score, best_alpha = trial_pos, metrics, score, alpha
                # Full cost when used alone: shared baseline measurement + this candidate's audit.
                seconds = shared_seconds + time.perf_counter() - started
                accepted = best_alpha > 0.
                # Evaluate unselected endpoints only after audit timing and decision.
                # Copy dicts: fallback references the same baseline metrics object.
                seed_export = {**seed_metrics, **independent_visual_metrics(seed_pos.cpu(), record["edge_index"])}
                candidate_export = {**candidate_metrics, **independent_visual_metrics(candidate.cpu(), record["edge_index"])}
                best_export = seed_export if not accepted else {**best_metrics,
                    **independent_visual_metrics(best_pos.cpu(), record["edge_index"])}
                for metrics, method, alpha in [
                    (candidate_export, f"HCTF-{variant}", 1.),
                    (seed_export, "Current-Model-Baseline", 0.),
                    (best_export, "HCTF-SafeBlend", best_alpha),
                ]:
                    rows[variant].append(row_with_metadata(metrics, method, graph_id, batch_id,
                        selected_alpha=alpha, selected_source=variant if alpha > 0 else "baseline",
                        accepted=accepted if method == "HCTF-SafeBlend" else False, audit_seconds=seconds))
                selections[variant].append({"dataset": dataset, "graph_id": graph_id, "batch_id": batch_id,
                    "accepted": accepted, "selected_source": variant if accepted else "baseline",
                    "selected_alpha": best_alpha, "score": best_score, "num_nodes": int(record["num_nodes"]),
                    "num_edges": len(unique_undirected_edges(record["edge_index"])), "audit_seconds": seconds})
            graph_id += 1
    for variant in VARIANTS:
        directory = ROOT / args.output_root / f"{dataset}_{variant}_s{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        frame, selection = pd.DataFrame(rows[variant]), pd.DataFrame(selections[variant])
        frame.to_csv(directory / f"{dataset}_eval.csv", index=False)
        selection.to_csv(directory / f"{dataset}_selection.csv", index=False)
        summary = {"dataset": dataset, "candidate": variant, "seed": seed, "graphs": graph_id,
                   "accepted_graphs": int(selection.accepted.sum()), "acceptance_rate": float(selection.accepted.mean()),
                   "mean_audit_ms_per_graph": float(selection.audit_seconds.mean()*1000), "shared_input": True,
                   "holdout_outside_audit_timer": True}
        pd.DataFrame([summary]).to_csv(directory / "summary.csv", index=False)
    # Commit marker only after both controls are completely written.
    marker = ROOT / args.output_root / f"PAIRED_{dataset}_s{seed}.json"
    marker.write_text(json.dumps({"dataset": dataset, "seed": seed, "graphs": graph_id,
        "variants": VARIANTS, "selection_metrics": METRICS, "unselected_metrics": HOLDOUT_METRICS,
        "same_baseline_inference": True}, indent=2), encoding="utf-8")
    print(f"[SUCCESS] shared-input {dataset} s{seed}: {graph_id} graphs", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17,42,73,101,137])
    parser.add_argument("--checkpoint-dir", default="checkpoints/hctf_submission_main")
    parser.add_argument("--output-root", default="results/hctf_safe_blend_paired_holdout")
    parser.add_argument("--log-root", default="logs/hctf_safe_blend_paired_holdout")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-test-batches", type=int, default=-1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    if args.single:
        evaluate(args)
        return 0
    output, logs = ROOT / args.output_root, ROOT / args.log_root
    output.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    def run(dataset, seed):
        marker = output / f"PAIRED_{dataset}_s{seed}.json"
        if args.skip_completed and marker.exists():
            return {"dataset": dataset, "seed": seed, "status": "SKIPPED_EXISTING", "return_code": 0}
        command = [sys.executable, str(Path(__file__).resolve()), "--single", "--datasets", dataset,
            "--seeds", str(seed), "--checkpoint-dir", args.checkpoint_dir, "--output-root", args.output_root,
            "--max-test-batches", str(args.max_test_batches)]
        print(f"[RUN] shared-input {dataset} s{seed}", flush=True)
        with (logs / f"paired_{dataset}_s{seed}.log").open("wb") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        status = "SUCCESS" if completed.returncode == 0 and marker.exists() else "FAILED"
        print(f"[{status}] {dataset} s{seed}", flush=True)
        return {"dataset": dataset, "seed": seed, "status": status, "return_code": completed.returncode}
    statuses = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        jobs = [pool.submit(run, dataset, seed) for dataset in args.datasets for seed in args.seeds]
        for job in as_completed(jobs):
            statuses.append(job.result())
            pd.DataFrame(statuses).to_csv(output / "run_status_paired.csv", index=False)
    return int(any(row["return_code"] != 0 for row in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
