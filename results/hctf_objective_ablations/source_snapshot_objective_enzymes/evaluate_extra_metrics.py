# evaluate_extra_metrics.py
# -*- coding: utf-8 -*-

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from deepgd.model import Generator
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
from deepgd.paretofield.losses import graphwise_stress


# =========================
# 1. DeepGD 原始辅助函数
# =========================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_edge_features(all_pair_shortest_path):
    return torch.cat([
        all_pair_shortest_path[:, None],
        1 / all_pair_shortest_path[:, None].square()
    ], dim=-1)


def rescale_by_stress(pos, apsp, edge_index, batch_index):
    edge_index = edge_index.long()
    batch_index = batch_index.long()

    src_pos, dst_pos = pos[edge_index[0]], pos[edge_index[1]]
    dist = (dst_pos - src_pos).norm(dim=1)

    eps = 1e-9
    u_over_d = dist / (apsp + eps)

    scatterd_u_over_d_2 = pyg.utils.scatter(
        u_over_d ** 2,
        batch_index[edge_index[0]],
        reduce="sum"
    )
    scatterd_u_over_d = pyg.utils.scatter(
        u_over_d,
        batch_index[edge_index[0]],
        reduce="sum"
    )

    scale = scatterd_u_over_d_2 / (scatterd_u_over_d + eps)
    return pos / (scale[batch_index][:, None] + eps)


def generate_init_pos(batch):
    pos = rescale_by_stress(
        pos=batch.pos,
        apsp=batch.apsp_attr,
        edge_index=batch.perm_index.long(),
        batch_index=batch.batch.long(),
    )
    return pos


# =========================
# 2. 几何指标函数
# =========================

def ccw(a, b, c):
    """判断三个点的方向。"""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    """判断线段 ab 和 cd 是否相交。"""
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def unique_undirected_edges(edge_index):
    """
    去除无向图重复边和自环。
    输入 edge_index: shape [2, E]
    返回 [(u, v), ...]，其中 u < v
    """
    edges = set()
    edge_index = edge_index.long().cpu().numpy()

    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        edges.add((a, b))

    return list(edges)


def edge_crossing_count(pos, edge_index):
    """
    计算边交叉数。
    注意：复杂度 O(E^2)，Rome 小图可以接受。
    """
    pos_np = pos.detach().cpu().numpy()
    edges = unique_undirected_edges(edge_index)

    crossing = 0
    checked = 0

    for i in range(len(edges)):
        u1, v1 = edges[i]
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]

            # 共享端点的边不算交叉
            if len({u1, v1, u2, v2}) < 4:
                continue

            checked += 1

            a = pos_np[u1]
            b = pos_np[v1]
            c = pos_np[u2]
            d = pos_np[v2]

            if segments_intersect(a, b, c, d):
                crossing += 1

    normalized = crossing / checked if checked > 0 else 0.0
    return crossing, normalized


def edge_length_variance(pos, edge_index):
    """
    计算实际边长度的方差和变异系数。
    """
    edge_index = edge_index.long()
    edges = unique_undirected_edges(edge_index)

    if len(edges) == 0:
        return 0.0, 0.0

    lengths = []
    for u, v in edges:
        length = torch.norm(pos[u] - pos[v]).item()
        lengths.append(length)

    lengths = np.array(lengths)
    var = float(np.var(lengths))
    cv = float(np.std(lengths) / (np.mean(lengths) + 1e-9))
    return var, cv


def node_occlusion_rate(pos, edge_index=None, ratio=0.03):
    """
    计算节点遮挡率。
    思路：
    1. 如果有边，用中位数边长作为尺度；
    2. threshold = ratio * median_edge_length；
    3. 统计距离小于 threshold 的节点对比例。
    """
    pos_np = pos.detach().cpu().numpy()
    n = pos_np.shape[0]

    if n <= 1:
        return 0.0, 0

    if edge_index is not None:
        edges = unique_undirected_edges(edge_index)
        edge_lengths = []
        for u, v in edges:
            edge_lengths.append(np.linalg.norm(pos_np[u] - pos_np[v]))
        if len(edge_lengths) > 0:
            scale = float(np.median(edge_lengths))
        else:
            scale = float(np.std(pos_np))
    else:
        scale = float(np.std(pos_np))

    threshold = max(ratio * scale, 1e-9)

    overlap_pairs = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos_np[i] - pos_np[j])
            if dist < threshold:
                overlap_pairs += 1

    rate = overlap_pairs / total_pairs if total_pairs > 0 else 0.0
    return rate, overlap_pairs


def angular_resolution(pos, edge_index):
    """Return mean minimum incident angle and its degree-normalized ratio.

    The ratio compares a node's smallest incident angle with the ideal
    ``2*pi/degree`` separation and is clipped to one.  Isolated and degree-one
    nodes do not constrain angular resolution and are therefore omitted.
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    edges = unique_undirected_edges(edge_index)
    incident = defaultdict(list)
    for source, target in edges:
        incident[source].append(target)
        incident[target].append(source)

    minimum_angles = []
    normalized = []
    for node, neighbours in incident.items():
        if len(neighbours) < 2:
            continue
        vectors = pos_np[np.asarray(neighbours)] - pos_np[node]
        valid = np.linalg.norm(vectors, axis=1) > 1e-12
        vectors = vectors[valid]
        if len(vectors) < 2:
            continue
        angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        minimum = float(np.min(gaps))
        ideal = 2.0 * np.pi / float(len(angles))
        minimum_angles.append(np.degrees(minimum))
        normalized.append(min(minimum / max(ideal, 1e-12), 1.0))

    if not minimum_angles:
        return 180.0, 1.0
    return float(np.mean(minimum_angles)), float(np.mean(normalized))


def crossing_angle_resolution(pos, edge_index):
    """Return mean and fifth-percentile acute angles of proper crossings.

    A crossing-free layout receives 90 degrees, the best possible value.  The
    metric is deliberately separate from crossing count: among layouts that
    still contain crossings, near-perpendicular intersections are easier to
    visually disambiguate than shallow intersections.
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    edges = unique_undirected_edges(edge_index)
    angles = []
    for first in range(len(edges)):
        u1, v1 = edges[first]
        for second in range(first + 1, len(edges)):
            u2, v2 = edges[second]
            if len({u1, v1, u2, v2}) < 4:
                continue
            a, b = pos_np[u1], pos_np[v1]
            c, d = pos_np[u2], pos_np[v2]
            if not segments_intersect(a, b, c, d):
                continue
            first_vector = b - a
            second_vector = d - c
            denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
            if denominator <= 1e-12:
                continue
            cosine = float(np.clip(abs(np.dot(first_vector, second_vector)) / denominator, 0.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
    if not angles:
        return 90.0, 90.0
    return float(np.mean(angles)), float(np.percentile(angles, 5.0))


def _point_segment_distance(point, start, end):
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    projection = start + fraction * segment
    return float(np.linalg.norm(point - projection))


def node_edge_clearance(pos, edge_index):
    """Return normalized mean and fifth-percentile node--edge clearance.

    Only non-incident node/edge pairs are considered.  Distances are divided
    by the median drawn edge length, making the result invariant to uniform
    scaling of a layout.  Larger values are better.
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    edges = unique_undirected_edges(edge_index)
    if not edges:
        return 1.0, 1.0
    lengths = np.asarray([np.linalg.norm(pos_np[u] - pos_np[v]) for u, v in edges])
    scale = max(float(np.median(lengths)), 1e-12)
    per_node = []
    for node in range(len(pos_np)):
        distances = [
            _point_segment_distance(pos_np[node], pos_np[source], pos_np[target])
            for source, target in edges
            if node not in (source, target)
        ]
        if distances:
            per_node.append(min(distances) / scale)
    if not per_node:
        return 1.0, 1.0
    values = np.asarray(per_node, dtype=np.float64)
    return float(np.mean(values)), float(np.percentile(values, 5.0))


def neighborhood_preservation(pos, edge_index, k=5):
    """Tie-aware precision of geometric k-neighbours in geodesic neighbourhoods.

    All nodes tied at the geodesic k-th distance are equally eligible.  Ties at
    the geometric k-th distance receive fractional membership, so node IDs do
    not decide the score on symmetric graphs.
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    num_nodes = len(pos_np)
    if num_nodes <= 1:
        return 1.0
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(unique_undirected_edges(edge_index))
    euclidean = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
    np.fill_diagonal(euclidean, np.inf)
    scores = []
    for node in range(num_nodes):
        hop_lengths = nx.single_source_shortest_path_length(graph, node)
        reachable = [(distance, other) for other, distance in hop_lengths.items() if other != node]
        if not reachable:
            continue
        reachable.sort(key=lambda item: item[0])
        count = min(int(k), len(reachable), num_nodes - 1)
        cutoff = reachable[count - 1][0]
        eligible = np.zeros(num_nodes, dtype=bool)
        for distance, other in reachable:
            eligible[other] = distance <= cutoff
        geometric_cutoff = np.partition(euclidean[node], count - 1)[count - 1]
        geometric_ties = np.isclose(
            euclidean[node], geometric_cutoff, rtol=1e-10, atol=1e-12
        )
        strictly_closer = (euclidean[node] < geometric_cutoff) & ~geometric_ties
        weights = strictly_closer.astype(float)
        remaining = count - int(strictly_closer.sum())
        weights[geometric_ties] = remaining / max(int(geometric_ties.sum()), 1)
        scores.append(float(weights[eligible].sum()) / float(count))
    return float(np.mean(scores)) if scores else 1.0


def independent_visual_metrics(pos, edge_index):
    """Metrics reserved for post-selection validation, never safe selection."""
    angular_mean, angular_ratio = angular_resolution(pos, edge_index)
    crossing_mean, crossing_p05 = crossing_angle_resolution(pos, edge_index)
    clearance_mean, clearance_p05 = node_edge_clearance(pos, edge_index)
    return {
        "angular_resolution_deg": angular_mean,
        "angular_resolution_ratio": angular_ratio,
        "crossing_angle_deg": crossing_mean,
        "crossing_angle_p05_deg": crossing_p05,
        "node_edge_clearance": clearance_mean,
        "node_edge_clearance_p05": clearance_p05,
        "neighborhood_preservation_k5": neighborhood_preservation(pos, edge_index, k=5),
    }


# =========================
# 3. 社区可读性指标
# =========================

def detect_communities(edge_index, num_nodes):
    """
    使用 NetworkX greedy modularity 做社区检测。
    如果图太小或没有边，则全部归为一个社区。
    """
    edges = unique_undirected_edges(edge_index)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    if G.number_of_edges() == 0 or num_nodes <= 2:
        return np.zeros(num_nodes, dtype=int)

    communities = list(nx.community.greedy_modularity_communities(G))

    labels = np.zeros(num_nodes, dtype=int)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid

    return labels


def community_metrics(pos, labels):
    """
    计算社区内聚度、社区分离度、社区分离/内聚比、轮廓系数。
    compactness 越小越好；
    separation 越大越好；
    sep_comp_ratio 越大越好；
    silhouette 越大越好。
    """
    pos_np = pos.detach().cpu().numpy()
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1:
        return {
            "community_compactness": 0.0,
            "community_separation": 0.0,
            "community_sep_comp_ratio": 0.0,
            "community_silhouette": 0.0,
            "num_communities": int(len(unique_labels)),
        }

    centroids = []
    compactness_values = []

    for lab in unique_labels:
        points = pos_np[labels == lab]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        centroids.append(centroid)

        dists = np.linalg.norm(points - centroid, axis=1)
        compactness_values.append(float(np.mean(dists)))

    compactness = float(np.mean(compactness_values)) if compactness_values else 0.0

    centroids = np.array(centroids)
    sep_values = []

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            sep_values.append(np.linalg.norm(centroids[i] - centroids[j]))

    separation = float(np.mean(sep_values)) if sep_values else 0.0
    ratio = separation / (compactness + 1e-9)

    try:
        if len(unique_labels) >= 2 and len(unique_labels) < len(pos_np):
            sil = float(silhouette_score(pos_np, labels))
        else:
            sil = 0.0
    except Exception:
        sil = 0.0

    return {
        "community_compactness": compactness,
        "community_separation": separation,
        "community_sep_comp_ratio": ratio,
        "community_silhouette": sil,
        "num_communities": int(len(unique_labels)),
    }


# =========================
# 4. 单图评价
# =========================

def evaluate_single_graph(pos, edge_index):
    """
    对一张图的二维布局计算额外指标。
    """
    num_nodes = pos.shape[0]

    crossing, crossing_norm = edge_crossing_count(pos, edge_index)
    edge_var, edge_cv = edge_length_variance(pos, edge_index)
    occ_rate, occ_pairs = node_occlusion_rate(pos, edge_index=edge_index)

    labels = detect_communities(edge_index, num_nodes)
    comm = community_metrics(pos, labels)

    result = {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(unique_undirected_edges(edge_index))),
        "edge_crossing": float(crossing),
        "edge_crossing_norm": float(crossing_norm),
        "edge_length_variance": float(edge_var),
        "edge_length_cv": float(edge_cv),
        "node_occlusion_rate": float(occ_rate),
        "node_occlusion_pairs": float(occ_pairs),
    }

    result.update(comm)
    result.update(independent_visual_metrics(pos, edge_index))
    return result


def split_batch_to_graphs(pos, batch_edge_index, batch_vector):
    """
    将 PyG batch 拆成单图。
    返回每张图的局部 pos 和局部 edge_index。
    """
    batch_vector = batch_vector.long().detach().cpu()
    edge_index = batch_edge_index.long().detach().cpu()

    graph_ids = batch_vector.unique(sorted=True).tolist()
    graphs = []

    for gid in graph_ids:
        node_mask = batch_vector == gid
        global_nodes = torch.where(node_mask)[0]
        node_set = set(global_nodes.tolist())

        if len(global_nodes) == 0:
            continue

        # 全局节点编号 -> 局部节点编号
        mapping = {int(g): i for i, g in enumerate(global_nodes.tolist())}

        # 取当前图内部边
        local_edges = []
        for u, v in edge_index.t().tolist():
            if u in node_set and v in node_set:
                local_edges.append([mapping[u], mapping[v]])

        if len(local_edges) == 0:
            local_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()

        local_pos = pos[global_nodes.to(pos.device)].detach().cpu()

        graphs.append((local_pos, local_edge_index))

    return graphs


# =========================
# 5. 主流程
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument(
        "--dataset",
        choices=["rome", "north", "sbm", "fb15k237", "wn18rr", "imdb_binary", "proteins", "enzymes"],
        default="rome",
    )
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-csv", type=str, default="deepgd_extra_metrics.csv")
    parser.add_argument("--max-test-batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    dataset_defaults = {
        "rome": (RomeDataset, "assets/rome_index.txt", "assets/layouts/pmds.npy"),
        "north": (NorthDataset, "assets/north_index.txt", "assets/layouts/north_pmds.npy"),
        "sbm": (SBMDataset, "assets/sbm_index.txt", "assets/layouts/sbm_pmds.npy"),
        "fb15k237": (FB15k237SampledDataset, "assets/fb15k237_index.txt", "assets/layouts/fb15k237_pmds.npy"),
        "wn18rr": (WN18RRSampledDataset, "assets/wn18rr_index.txt", "assets/layouts/wn18rr_pmds.npy"),
        "imdb_binary": (IMDBBinaryDataset, "assets/imdb_binary_index.txt", "assets/layouts/imdb_binary_pmds.npy"),
        "proteins": (PROTEINSDataset, "assets/proteins_index.txt", "assets/layouts/proteins_pmds.npy"),
        "enzymes": (ENZYMESDataset, "assets/enzymes_index.txt", "assets/layouts/enzymes_pmds.npy"),
    }
    dataset_cls, default_index, default_layouts = dataset_defaults[args.dataset]
    if args.index_file == "assets/rome_index.txt" and args.dataset != "rome":
        args.index_file = default_index
    if args.layouts_file == "assets/layouts/pmds.npy" and args.dataset != "rome":
        args.layouts_file = default_layouts

    device = get_device() if args.device == "auto" else args.device
    print(f"[INFO] device = {device}")

    dataset = dataset_cls(
        index=pd.read_csv(args.index_file, header=None)[0],
    )

    layouts = np.load(args.layouts_file, allow_pickle=True)

    params = Generator.Params(
        num_blocks=11,
        block_depth=3,
        block_width=8,
        block_output_dim=8,
        edge_net_depth=2,
        edge_net_width=16,
        edge_attr_dim=2,
        node_attr_dim=2,
    )

    model = Generator(params=params).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"[INFO] loaded model: {args.model_path}")

    datalist = list(dataset)
    for i, data in enumerate(datalist):
        data.pos = torch.tensor(layouts[i]).float()

    if args.dataset == "rome" and len(datalist) >= 11000:
        test_data = datalist[10000:11000]
    else:
        indices = np.arange(len(datalist))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
        split = int(0.8 * len(indices)) + int(0.1 * len(indices))
        test_data = [datalist[int(index)] for index in indices[split:]]

    test_loader = pyg.loader.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False
    )

    stress_metric = Stress()

    all_rows = []
    stress_values = []
    infer_times = []

    graph_counter = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader, desc="Evaluate test set")):
            if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
                break

            batch = batch.to(device)
            batch.perm_index = batch.perm_index.long()
            batch.batch = batch.batch.long()

            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
                if batch_id == 0:
                    torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()

            pred = model(
                init_pos=generate_init_pos(batch),
                edge_index=batch.perm_index.long(),
                edge_attr=get_edge_features(batch.apsp_attr),
                batch_index=batch.batch.long(),
            )

            pos = rescale_by_stress(
                pred,
                batch.apsp_attr,
                batch.perm_index.long(),
                batch.batch.long()
            )

            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            infer_times.append(elapsed)

            stress_value = stress_metric(
                pos,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long()
            ).item()
            stress_values.append(stress_value)
            per_graph_stress = graphwise_stress(
                pos,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long(),
            ).detach().cpu()

            # 注意：边交叉、遮挡、社区等指标应该基于真实边 batch.edge_index，
            # 不是基于 all-pairs 的 batch.perm_index。
            graphs = split_batch_to_graphs(
                pos=pos,
                batch_edge_index=batch.edge_index.long(),
                batch_vector=batch.batch.long()
            )

            for local_id, (local_pos, local_edge_index) in enumerate(graphs):
                row = evaluate_single_graph(local_pos, local_edge_index)
                row["graph_id"] = graph_counter
                row["batch_id"] = batch_id
                row["batch_stress"] = stress_value
                row["graph_stress"] = float(per_graph_stress[local_id])
                row["batch_infer_time"] = elapsed
                row["dataset"] = args.dataset
                row["seed"] = int(args.seed)
                row["method"] = "Original DeepGD"
                row["peak_cuda_memory_mb"] = (
                    float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                    if torch.device(device).type == "cuda"
                    else 0.0
                )
                all_rows.append(row)
                graph_counter += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n========== DeepGD Baseline Extra Metrics ==========")
    print(f"Test graphs: {len(df)}")
    print(f"Mean Stress: {np.mean(stress_values):.6f}")
    print(f"Mean batch inference time: {np.mean(infer_times):.6f} s")

    metric_cols = [
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
        "angular_resolution_deg",
        "angular_resolution_ratio",
        "crossing_angle_deg",
        "crossing_angle_p05_deg",
        "node_edge_clearance",
        "node_edge_clearance_p05",
        "neighborhood_preservation_k5",
    ]

    print("\n========== Mean Extra Metrics ==========")
    for col in metric_cols:
        if col in df.columns:
            print(f"{col}: {df[col].mean():.6f}")

    print(f"\n[INFO] Saved per-graph metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
