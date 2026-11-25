"""Isolated raw-graph-to-layout scaling, including APSP/MDS and exact audit.

Synthetic SBM graphs are used solely to measure runtime, never as accuracy
results. Model construction and disk I/O are excluded (persistent service).
"""
from __future__ import annotations
import argparse
import json
import platform
import time
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import evaluate_hctf_safe_blend as evaluator
from hctf_fast_audit import install_fast_audit

install_fast_audit(evaluator)

from evaluate_hctf_safe_blend import (ROOT, train, Stress, load_model, measure_layout,
    METRICS, improvement, is_componentwise_safe, objective_metric_weights, detect_communities,
    namespace_from_checkpoint)
from deepgd.data import GraphDrawingData
from run_hctf_all8_pilot import DATASETS as DATASET_CONFIGURATION


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def graph_for_size(size, seed):
    sizes = [size // 4] * 4
    for i in range(size % 4): sizes[i] += 1
    within = min(.9, (6.0-.75) / max(np.mean(sizes)-1, 1))
    probabilities = [[within if i==j else 1./size for j in range(4)] for i in range(4)]
    graph = nx.stochastic_block_model(sizes, probabilities, seed=seed)
    components = [min(component) for component in nx.connected_components(graph)]
    for a,b in zip(components,components[1:]): graph.add_edge(a,b)
    graph.graph.update(name=f"runtime_sbm_{size}_{seed}",dataset_name="runtime_sbm")
    return graph


@torch.no_grad()
def benchmark(args):
    device = train.get_device()
    models = {}
    for variant in ["full_nodrop", "node_only_nodrop"]:
        if args.configuration=="adaptive":
            name="adaptive_supervised_full" if variant=="full_nodrop" else "adaptive_supervised_node_only"
            path=ROOT/args.checkpoint_dir/f"adaptivefinal_{args.dataset}_{name}_s{args.seed}_best.pt"
        else:
            path = ROOT / args.checkpoint_dir / f"submission_{args.dataset}_{variant}_s{args.seed}_best.pt"
        model_args, model = load_model(path, device)
        model_args._hctf_teacher_enabled = False
        models[variant] = (model_args, model)
    if args.configuration=="adaptive":
        path=ROOT/args.checkpoint_dir/f"adaptivefinal_{args.dataset}_adaptive_supervised_full_s{args.seed}_best.pt"
        baseline_args=namespace_from_checkpoint(path)
        baseline_args.use_hctf=False
        baseline=train.build_deepgd_model(baseline_args,device)
        train.load_checkpoint(str(ROOT/"checkpoints"/DATASET_CONFIGURATION[args.dataset]["checkpoint"]),baseline,None,device)
        baseline.eval()
    else:
        baseline_args,baseline=load_model(ROOT/args.checkpoint_dir/f"submission_{args.dataset}_baseline_s{args.seed}_best.pt",device)
    baseline_args._hctf_teacher_enabled=False
    models={"baseline":(baseline_args,baseline),**models}
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    weights = objective_metric_weights("")
    stress = Stress()
    rows = []
    for size in args.sizes:
        for graph_seed in args.graph_seeds:
            graph = graph_for_size(size, graph_seed)
            # Rotated model order reduces order/thermal bias.
            for repeat in range(-1, args.repeats):
                variants = list(models)
                rotation = (repeat+1) % len(variants)
                variants = variants[rotation:] + variants[:rotation]
                for variant in variants:
                    model_args, model = models[variant]
                    train.set_random_seed(graph_seed)
                    synchronize()
                    resident_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    data = GraphDrawingData.new(graph.copy())
                    data.pos = torch.tensor(train.classical_mds_layout(data), dtype=torch.float32)
                    train.attach_community_labels([data], model_args)
                    batch = pyg.data.Batch.from_data_list([data]).to(device)
                    synchronize(); preprocessed = time.perf_counter()
                    _, proposal, _, _ = train.forward_model(batch, model, None, stress, model_args)
                    synchronize(); inferred = time.perf_counter()
                    # Audited deployment needs a separate baseline forward.
                    baseline_forward_seconds = 0.
                    if variant != "baseline":
                        baseline_args, baseline = models["baseline"]
                        _, seed_batch, _, _ = train.forward_model(batch, baseline, None, stress, baseline_args)
                        synchronize(); baseline_forward_seconds = time.perf_counter()-inferred
                    else:
                        seed_batch = proposal
                    audit_started = time.perf_counter()
                    record = train.mtpf_build_local_graph_records(batch)[0]
                    labels = detect_communities(record["edge_index"],record["num_nodes"])
                    seed_pos, seed_metrics = measure_layout(seed_batch,record,labels,include_holdout=False)
                    best_alpha, best_score = 0., 0.
                    if variant != "baseline":
                        candidate, _ = measure_layout(proposal,record,labels,include_holdout=False)
                        aligned = train.mtpf_procrustes_align(candidate,seed_pos)
                        for alpha in [.0625,.125,.25,.5,.75,1.]:
                            _, metrics = measure_layout(seed_pos+alpha*(aligned-seed_pos),record,labels,include_holdout=False)
                            if not is_componentwise_safe(seed_metrics,metrics,relative_tolerance=0.): continue
                            gains = [improvement(seed_metrics,metrics,m) for m in METRICS]
                            score = sum(weights[m]*g for m,g in zip(METRICS,gains))
                            if any(g>1e-10 for g in gains) and score>best_score+1e-12:
                                best_score,best_alpha=score,alpha
                    synchronize(); finished=time.perf_counter()
                    if repeat >= 0:
                        rows.append({"num_nodes":size,"num_edges":graph.number_of_edges(),"graph_seed":graph_seed,
                            "repeat":repeat,"variant":variant,"preprocess_ms":1000*(preprocessed-started),
                            "proposal_forward_ms":1000*(inferred-preprocessed),
                            "baseline_forward_ms":1000*baseline_forward_seconds,
                            "audit_ms":1000*(finished-audit_started),"total_ms":1000*(finished-started),
                            "selected_alpha":best_alpha,"peak_cuda_memory_mb":
                            torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.,
                            "resident_cuda_memory_mb":resident_memory/2**20,
                            "incremental_peak_cuda_memory_mb":
                            (torch.cuda.max_memory_allocated()-resident_memory)/2**20 if torch.cuda.is_available() else 0.})
                        pd.DataFrame(rows).to_csv(output/"end_to_end_observations.csv",index=False)
                    # No previous batch/layout survives into the next peak.
                    # The common three persistent models remain resident.
                    del data, batch, proposal, seed_batch, record, seed_pos, seed_metrics, _
                    if variant != "baseline": del candidate, aligned, metrics
                print(f"[RUNTIME] nodes={size}, seed={graph_seed}, repeat={repeat}",flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["variant","num_nodes"])[["preprocess_ms","proposal_forward_ms","baseline_forward_ms","audit_ms","total_ms","peak_cuda_memory_mb"]].agg(["median","min","max"])
    summary.columns=[f"{metric}_{stat}" for metric,stat in summary.columns]
    summary.reset_index().to_csv(output/"end_to_end_summary.csv",index=False)
    contract = {"complete":True,"sizes":args.sizes,"graph_seeds":args.graph_seeds,"repeats":args.repeats,
        "configuration":args.configuration,"checkpoint_dataset":args.dataset,"checkpoint_seed":args.seed,
        "warmup_repeats_per_graph":1,"python":platform.python_version(),"torch":torch.__version__,
        "device":torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "accuracy_evidence":False,"input":"Fresh connected SBM; target mean degree approximately 6",
        "included":["Fresh graph transforms and APSP","Classical MDS and community detection","Feature construction and model forward","Separate baseline forward for audited proposals","Six-point exact audit"],
        "excluded":["Model initialization/loading","Disk I/O","Unselected visual endpoints"],
        "initial_layout":"Fresh classical MDS, shared by learned controls",
        "memory_scope":"CUDA allocated peak including all three persistent models; resident and incremental peaks also recorded. CPU RAM and allocator caches not measured.",
        "warning":"Run with no other training/evaluation processes for interpretable timing comparisons."}
    (output/"COMPLETE.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sizes",nargs="+",type=int,default=[16,32,64,128,256,512])
    parser.add_argument("--graph-seeds",nargs="+",type=int,default=[901,902])
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--configuration",choices=["adaptive","legacy"],default="adaptive")
    parser.add_argument("--dataset",default="enzymes")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--checkpoint-dir",default="checkpoints/hctf_adaptive_final")
    parser.add_argument("--output-dir",default="results/hctf_end_to_end")
    args=parser.parse_args()
    benchmark(args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
