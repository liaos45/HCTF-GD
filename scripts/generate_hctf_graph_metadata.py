"""Generate split-aligned graph-size and community-strength metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_cser_spectral_mog_end2end as train  # noqa: E402


DATASETS = ["sbm", "enzymes", "proteins", "wn18rr", "fb15k237", "rome", "north", "imdb_binary"]
INDEX_PATHS = {name: ROOT / "assets" / f"{name}_index.txt" for name in DATASETS}


def graph_metadata(data, dataset, graph_index):
    graph = nx.Graph(data.G)
    graph.add_nodes_from(range(int(data.num_nodes)))
    seed = train._stable_graph_seed(getattr(data, "name", f"graph_{graph_index}"), 42)
    greedy = train._community_labels_from_graph(data, "greedy", seed)
    groups = [set(np.where(greedy == label)[0].tolist()) for label in np.unique(greedy)]
    modularity = (
        float(nx.community.modularity(graph, groups, weight=None))
        if graph.number_of_edges() and len(groups) > 1
        else 0.0
    )
    edges = [(int(u), int(v)) for u, v in graph.edges() if int(u) != int(v)]
    inter_edges = sum(int(greedy[u] != greedy[v]) for u, v in edges)
    boundary_nodes = set()
    for u, v in edges:
        if greedy[u] != greedy[v]:
            boundary_nodes.update([u, v])
    ground_truth_nmi = np.nan
    if dataset == "sbm":
        truth = train._community_labels_from_graph(data, "ground_truth", seed)
        ground_truth_nmi = float(normalized_mutual_info_score(truth, greedy))
    return {
        "dataset": dataset,
        "original_graph_index": graph_index,
        "graph_name": getattr(data, "name", f"graph_{graph_index}"),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(graph.number_of_edges()),
        "edge_density": float(nx.density(graph)),
        "greedy_communities": int(np.unique(greedy).size),
        "greedy_modularity": modularity,
        "inter_community_edge_fraction": inter_edges / max(len(edges), 1),
        "boundary_node_fraction": len(boundary_nodes) / max(int(data.num_nodes), 1),
        "greedy_ground_truth_nmi": ground_truth_nmi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73, 101, 137])
    parser.add_argument("--output", type=Path, default=Path("results/hctf_graph_metadata.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        dataset_cls = train.dataset_class_from_name(dataset)
        index = pd.read_csv(INDEX_PATHS[dataset], header=None)[0]
        datalist = list(dataset_cls(index=index))
        metadata = [graph_metadata(data, dataset, index) for index, data in enumerate(datalist)]
        name_to_index = {str(getattr(data, "name", index)): index for index, data in enumerate(datalist)}
        for seed in args.seeds:
            split_args = argparse.Namespace(dataset=dataset, seed=seed, split_mode="random")
            _, _, test_data = train.split_graph_dataset(datalist, split_args)
            for test_graph_id, data in enumerate(test_data):
                original_index = name_to_index[str(getattr(data, "name", ""))]
                rows.append(
                    {
                        **metadata[original_index],
                        "seed": seed,
                        "test_graph_id": test_graph_id,
                    }
                )
        print(f"[DONE] {dataset}: graphs={len(datalist)}", flush=True)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {len(rows)} split-aligned rows to {output}", flush=True)


if __name__ == "__main__":
    main()

