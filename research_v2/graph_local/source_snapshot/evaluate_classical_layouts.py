# evaluate_classical_layouts.py
# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import networkx as nx
from sklearn.manifold import smacof
from tqdm import tqdm

from deepgd.datasets import (
    ENZYMESDataset,
    FB15k237SampledDataset,
    IMDBBinaryDataset,
    NorthDataset,
    PROTEINSDataset,
    RomeDataset,
    SBMDataset,
    WN18RRSampledDataset,
)
from deepgd.metrics import Stress

from evaluate_extra_metrics import (
    rescale_by_stress,
    evaluate_single_graph,
    split_batch_to_graphs,
)


def pyg_data_to_networkx(data):
    """
    将 PyG Data 转为 networkx Graph。
    """
    G = nx.Graph()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))

    edge_index = data.edge_index.long().cpu().numpy()
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u != v:
            G.add_edge(u, v)

    return G


def nx_pos_to_tensor(pos_dict, num_nodes):
    """
    将 networkx layout 输出的 dict 转为 torch.Tensor [N, 2]
    """
    pos = np.zeros((num_nodes, 2), dtype=np.float32)

    for i in range(num_nodes):
        if i in pos_dict:
            pos[i] = np.array(pos_dict[i], dtype=np.float32)
        else:
            pos[i] = np.random.randn(2).astype(np.float32)

    return torch.tensor(pos, dtype=torch.float32)


def run_fr_layout(G, seed=42):
    """
    Fruchterman-Reingold 布局。
    """
    return nx.spring_layout(G, seed=seed, iterations=100)


def run_kk_layout(G):
    """
    Kamada-Kawai 布局。
    """
    return nx.kamada_kawai_layout(G)


def run_forceatlas2_layout(G, seed=42, iterations=100):
    """NetworkX ForceAtlas2 with an explicit seed and iteration budget."""
    return nx.forceatlas2_layout(G, seed=seed, max_iter=int(iterations))


def run_smacof_layout(G, seed=42, max_iter=150):
    """Metric stress majorization on all-pairs shortest-path distances."""
    nodes = list(range(G.number_of_nodes()))
    if len(nodes) <= 1:
        return {node: np.zeros(2, dtype=np.float32) for node in nodes}
    distances = np.asarray(nx.floyd_warshall_numpy(G, nodelist=nodes), dtype=np.float64)
    finite = distances[np.isfinite(distances)]
    replacement = (float(finite.max()) + 1.0) if finite.size else 1.0
    distances[~np.isfinite(distances)] = replacement
    positions, _, _ = smacof(
        distances,
        metric=True,
        n_components=2,
        n_init=1,
        max_iter=int(max_iter),
        eps=1e-4,
        random_state=int(seed),
        return_n_iter=True,
        normalized_stress=False,
    )
    return {node: positions[index] for index, node in enumerate(nodes)}


def run_pmds_layout(data):
    """
    使用 DeepGD assets/layouts/pmds.npy 中已有的 PivotMDS 坐标。
    这里在主流程里直接赋值，不在这里计算。
    """
    return data.pos


def dataset_class_from_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "rome":
        return RomeDataset
    if dataset_name == "north":
        return NorthDataset
    if dataset_name == "sbm":
        return SBMDataset
    if dataset_name == "fb15k237":
        return FB15k237SampledDataset
    if dataset_name == "wn18rr":
        return WN18RRSampledDataset
    if dataset_name == "imdb_binary":
        return IMDBBinaryDataset
    if dataset_name == "proteins":
        return PROTEINSDataset
    if dataset_name == "enzymes":
        return ENZYMESDataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def split_graph_dataset(datalist, dataset_name, seed):
    if dataset_name.lower() == "rome" and len(datalist) >= 11000:
        return datalist[10000:11000]

    indices = np.arange(len(datalist))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n = len(indices)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n)) if n >= 3 else 0
    test_idx = indices[n_train + n_val:]
    if len(test_idx) == 0 and n_val > 0:
        test_idx = indices[n_train + n_val - 1:n_train + n_val]

    return [datalist[int(i)] for i in test_idx]


def evaluate_batch_layout(batch, pos, stress_metric, method_name, batch_id, infer_time):
    """
    对一个 batch 的布局进行 stress + 额外指标评价。
    """
    batch = batch.cpu()
    pos = pos.cpu()

    edge_index_apsp = batch.perm_index.long()
    batch_index = batch.batch.long()

    pos = rescale_by_stress(
        pos=pos,
        apsp=batch.apsp_attr,
        edge_index=edge_index_apsp,
        batch_index=batch_index,
    )

    stress_value = stress_metric(
        pos,
        edge_index_apsp,
        batch.apsp_attr,
        batch_index,
    ).item()

    # Keep the historical batch aggregate for backwards compatibility, but
    # record a true per-graph stress value for paired tests and graph-level
    # summaries. Copying ``stress_value`` to every graph biases variable-size
    # batches and creates pseudoreplication.
    from deepgd.paretofield.losses import graphwise_stress

    per_graph_stress = graphwise_stress(
        pos,
        edge_index_apsp,
        batch.apsp_attr,
        batch_index,
    ).detach().cpu()

    rows = []
    graphs = split_batch_to_graphs(
        pos=pos,
        batch_edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
    )

    for local_id, (local_pos, local_edge_index) in enumerate(graphs):
        row = evaluate_single_graph(local_pos, local_edge_index)
        row["method"] = method_name
        row["batch_id"] = batch_id
        row["local_graph_id"] = local_id
        row["batch_stress"] = float(stress_value)
        row["graph_stress"] = float(per_graph_stress[local_id])
        row["batch_infer_time"] = float(infer_time)
        rows.append(row)

    return rows, stress_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="rome",
        choices=[
            "rome",
            "north",
            "sbm",
            "fb15k237",
            "wn18rr",
            "imdb_binary",
            "proteins",
            "enzymes",
        ],
    )
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-graphs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["fr", "kk", "pmds", "forceatlas2", "smacof", "all"],
    )
    parser.add_argument("--fa2-iterations", type=int, default=100)
    parser.add_argument("--smacof-max-iter", type=int, default=150)
    parser.add_argument("--output-csv", type=str, default="classical_layout_metrics.csv")
    args = parser.parse_args()

    if args.dataset == "north":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/north_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/north_pmds.npy"
    elif args.dataset == "sbm":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/sbm_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/sbm_pmds.npy"
    elif args.dataset == "fb15k237":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/fb15k237_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/fb15k237_pmds.npy"
    elif args.dataset == "wn18rr":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/wn18rr_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/wn18rr_pmds.npy"
    elif args.dataset == "imdb_binary":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/imdb_binary_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/imdb_binary_pmds.npy"
    elif args.dataset == "proteins":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/proteins_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/proteins_pmds.npy"
    elif args.dataset == "enzymes":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/enzymes_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/enzymes_pmds.npy"

    print(f"[INFO] loading {args.dataset} dataset...")
    dataset_cls = dataset_class_from_name(args.dataset)
    dataset = dataset_cls(
        index=pd.read_csv(args.index_file, header=None)[0],
    )

    layouts = np.load(args.layouts_file, allow_pickle=True)

    datalist = list(dataset)

    for i, data in enumerate(datalist):
        data.pos = torch.tensor(layouts[i]).float()

    test_list = split_graph_dataset(datalist, args.dataset, args.seed)

    if args.max_test_graphs > 0:
        test_list = test_list[:args.max_test_graphs]

    print(f"[INFO] test graphs = {len(test_list)}")

    stress_metric = Stress()
    all_rows = []
    stress_records = {}

    methods = []
    if args.method == "all":
        methods = ["fr", "kk", "pmds", "forceatlas2", "smacof"]
    else:
        methods = [args.method]

    for method in methods:
        print(f"\n========== Running {method.upper()} ==========")

        method_rows = []
        method_stress_values = []
        layout_times = []

        generated_data_list = []

        for graph_id, data in enumerate(tqdm(test_list, desc=f"Generate {method} layouts")):
            G = pyg_data_to_networkx(data)

            start_time = time.time()

            if method == "fr":
                pos_dict = run_fr_layout(G, seed=42)
                pos = nx_pos_to_tensor(pos_dict, data.num_nodes)

            elif method == "kk":
                pos_dict = run_kk_layout(G)
                pos = nx_pos_to_tensor(pos_dict, data.num_nodes)

            elif method == "pmds":
                pos = data.pos.clone().float()

            elif method == "forceatlas2":
                pos_dict = run_forceatlas2_layout(
                    G,
                    seed=args.seed,
                    iterations=args.fa2_iterations,
                )
                pos = nx_pos_to_tensor(pos_dict, data.num_nodes)

            elif method == "smacof":
                pos_dict = run_smacof_layout(
                    G,
                    seed=args.seed,
                    max_iter=args.smacof_max_iter,
                )
                pos = nx_pos_to_tensor(pos_dict, data.num_nodes)

            else:
                raise ValueError(f"Unknown method: {method}")

            infer_time = time.time() - start_time
            layout_times.append(float(infer_time))

            new_data = data.clone()
            new_data.pos = pos
            new_data.layout_time = infer_time
            new_data.graph_id_for_eval = graph_id
            generated_data_list.append(new_data)

        loader = pyg.loader.DataLoader(
            generated_data_list,
            batch_size=args.batch_size,
            shuffle=False,
        )

        graph_offset = 0
        for batch_id, batch in enumerate(tqdm(loader, desc=f"Evaluate {method}")):
            infer_times = []
            if hasattr(batch, "layout_time"):
                # PyG 会把 layout_time 聚合成 tensor/list，这里不用严格依赖它
                pass

            rows, stress_value = evaluate_batch_layout(
                batch=batch,
                pos=batch.pos,
                stress_metric=stress_metric,
                method_name=method.upper(),
                batch_id=batch_id,
                infer_time=0.0,
            )

            for local_id, row in enumerate(rows):
                test_graph_id = graph_offset + local_id
                row["dataset"] = args.dataset
                row["seed"] = int(args.seed)
                row["test_graph_id"] = int(test_graph_id)
                row["layout_time"] = float(layout_times[test_graph_id])
            graph_offset += len(rows)

            method_rows.extend(rows)
            method_stress_values.append(stress_value)

        all_rows.extend(method_rows)
        stress_records[method.upper()] = float(np.mean(method_stress_values))

        df_method = pd.DataFrame(method_rows)

        print(f"\n[{method.upper()}] Mean Stress: {np.mean(method_stress_values):.6f}")
        print(f"[{method.upper()}] Edge Crossing: {df_method['edge_crossing'].mean():.6f}")
        print(f"[{method.upper()}] Edge Length CV: {df_method['edge_length_cv'].mean():.6f}")
        print(f"[{method.upper()}] Node Occlusion Rate: {df_method['node_occlusion_rate'].mean():.6f}")
        print(f"[{method.upper()}] Community Compactness: {df_method['community_compactness'].mean():.6f}")
        print(f"[{method.upper()}] Community Separation: {df_method['community_separation'].mean():.6f}")
        print(f"[{method.upper()}] Sep/Comp Ratio: {df_method['community_sep_comp_ratio'].mean():.6f}")
        print(f"[{method.upper()}] Silhouette: {df_method['community_silhouette'].mean():.6f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n========== Classical Layout Summary ==========")

    metric_cols = [
        "graph_stress",
        "batch_stress",
        "edge_crossing",
        "edge_crossing_norm",
        "edge_length_variance",
        "edge_length_cv",
        "node_occlusion_rate",
        "node_occlusion_pairs",
        "community_compactness",
        "community_separation",
        "community_sep_comp_ratio",
        "community_silhouette",
        "num_communities",
        "layout_time",
    ]

    summary = df.groupby("method")[metric_cols].mean()
    print(summary)

    print(f"\n[INFO] saved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
