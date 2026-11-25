# sc_deepgd_refine.py
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from tqdm import tqdm

from deepgd.model import Generator
from deepgd.datasets import RomeDataset
from deepgd.metrics import Stress

from evaluate_extra_metrics import (
    get_device,
    get_edge_features,
    rescale_by_stress,
    generate_init_pos,
    split_batch_to_graphs,
    evaluate_single_graph,
    detect_communities,
)


# =========================================================
# 1. 构建 batch 内每个节点的社区标签
# =========================================================

def build_batch_community_labels(edge_index, batch_vector, num_nodes):
    """
    给 batch 中每个节点构建社区标签。
    每张图单独做社区检测，不跨图混合社区。
    """
    edge_index = edge_index.long().detach().cpu()
    batch_vector_cpu = batch_vector.long().detach().cpu()

    labels_global = torch.zeros(num_nodes, dtype=torch.long)
    graph_ids = batch_vector_cpu.unique(sorted=True).tolist()

    offset = 0

    for gid in graph_ids:
        node_mask = batch_vector_cpu == gid
        global_nodes = torch.where(node_mask)[0]
        node_set = set(global_nodes.tolist())

        mapping = {int(g): i for i, g in enumerate(global_nodes.tolist())}

        local_edges = []
        for u, v in edge_index.t().tolist():
            if u in node_set and v in node_set:
                local_edges.append([mapping[u], mapping[v]])

        if len(local_edges) == 0:
            local_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()

        local_labels = detect_communities(local_edge_index, len(global_nodes))
        local_labels = torch.tensor(local_labels, dtype=torch.long) + offset

        labels_global[global_nodes] = local_labels
        offset = int(labels_global.max().item()) + 1

    return labels_global.to(batch_vector.device)


# =========================================================
# 2. 社区中心级约束损失：Intra + Inter
# =========================================================

def community_refine_loss(pos, labels, batch_vector, margin=3.0):
    """
    社区中心级约束：
    1. intra：同一社区内部节点更紧凑；
    2. inter：不同社区中心保持一定距离。
    """
    device = pos.device
    labels = labels.long()
    batch_vector = batch_vector.long()

    intra_losses = []
    inter_losses = []

    for gid in batch_vector.unique(sorted=True):
        node_mask = batch_vector == gid
        pos_g = pos[node_mask]
        labels_g = labels[node_mask]

        unique_labels = labels_g.unique(sorted=True)

        if pos_g.shape[0] <= 1 or len(unique_labels) <= 1:
            continue

        centroids = []

        for lab in unique_labels:
            comm_pos = pos_g[labels_g == lab]
            if comm_pos.shape[0] == 0:
                continue

            centroid = comm_pos.mean(dim=0)
            centroids.append(centroid)

            intra_losses.append(((comm_pos - centroid) ** 2).sum(dim=1).mean())

        if len(centroids) >= 2:
            centroids = torch.stack(centroids, dim=0)
            dists = torch.pdist(centroids, p=2)
            inter_losses.append(torch.relu(margin - dists).pow(2).mean())

    if len(intra_losses) > 0:
        intra_loss = torch.stack(intra_losses).mean()
    else:
        intra_loss = torch.tensor(0.0, device=device)

    if len(inter_losses) > 0:
        inter_loss = torch.stack(inter_losses).mean()
    else:
        inter_loss = torch.tensor(0.0, device=device)

    return intra_loss, inter_loss


# =========================================================
# 3. 新增：节点级社区对比损失
# =========================================================

def community_pair_contrast_loss(
    pos,
    labels,
    batch_vector,
    same_weight=1.0,
    diff_weight=1.0,
    pair_margin=2.0,
    max_pairs=4096,
):
    """
    节点级社区对比损失：
    1. 同社区节点对距离尽量小；
    2. 不同社区节点对距离至少大于 pair_margin。

    该损失比社区中心级约束更直接，目标是提升：
    - community_silhouette
    - community_sep_comp_ratio
    - community_separation
    """
    device = pos.device
    labels = labels.long()
    batch_vector = batch_vector.long()

    losses = []

    for gid in batch_vector.unique(sorted=True):
        node_mask = batch_vector == gid
        pos_g = pos[node_mask]
        labels_g = labels[node_mask]

        n = pos_g.shape[0]

        if n <= 2 or labels_g.unique().numel() <= 1:
            continue

        dist_mat = torch.cdist(pos_g, pos_g, p=2)

        upper_mask = torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=device),
            diagonal=1
        )

        same_mask = (labels_g[:, None] == labels_g[None, :]) & upper_mask
        diff_mask = (labels_g[:, None] != labels_g[None, :]) & upper_mask

        same_dist = dist_mat[same_mask]
        diff_dist = dist_mat[diff_mask]

        if same_dist.numel() > max_pairs:
            idx = torch.randperm(same_dist.numel(), device=device)[:max_pairs]
            same_dist = same_dist[idx]

        if diff_dist.numel() > max_pairs:
            idx = torch.randperm(diff_dist.numel(), device=device)[:max_pairs]
            diff_dist = diff_dist[idx]

        loss = torch.tensor(0.0, device=device)

        if same_dist.numel() > 0:
            loss = loss + same_weight * same_dist.pow(2).mean()

        if diff_dist.numel() > 0:
            loss = loss + diff_weight * torch.relu(pair_margin - diff_dist).pow(2).mean()

        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


# =========================================================
# 4. 对 DeepGD 输出坐标进行测试阶段细化
# =========================================================

def refine_positions(
    pos,
    batch,
    stress_metric,
    steps=30,
    lr=0.005,
    lambda_stress=1.0,
    lambda_intra=0.1,
    lambda_inter=0.05,
    lambda_pair=0.0,
    lambda_anchor=0.1,
    margin=3.0,
    pair_margin=2.0,
):
    """
    对 DeepGD 输出坐标进行测试阶段细化。

    总损失：
    L = lambda_stress * L_stress
      + lambda_intra  * L_intra
      + lambda_inter  * L_inter
      + lambda_pair   * L_pair_contrast
      + lambda_anchor * L_anchor
    """
    edge_index_apsp = batch.perm_index.long()
    batch_index = batch.batch.long()

    labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
    )

    pos0 = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        stress_loss = stress_metric(
            pos_var,
            edge_index_apsp,
            batch.apsp_attr,
            batch_index,
        )

        intra_loss, inter_loss = community_refine_loss(
            pos_var,
            labels,
            batch_index,
            margin=margin,
        )

        if lambda_pair > 0:
            pair_loss = community_pair_contrast_loss(
                pos_var,
                labels,
                batch_index,
                pair_margin=pair_margin,
            )
        else:
            pair_loss = torch.tensor(0.0, device=pos_var.device)

        anchor_loss = ((pos_var - pos0) ** 2).sum(dim=1).mean()

        loss = (
            lambda_stress * stress_loss
            + lambda_intra * intra_loss
            + lambda_inter * inter_loss
            + lambda_pair * pair_loss
            + lambda_anchor * anchor_loss
        )

        loss.backward()
        optimizer.step()

    refined_pos = rescale_by_stress(
        pos_var.detach(),
        batch.apsp_attr,
        edge_index_apsp,
        batch_index,
    )

    return refined_pos


# =========================================================
# 5. 评价单个 batch 的布局结果
# =========================================================

def evaluate_pos_by_graph(pos, batch, method_name, batch_id, batch_stress, infer_time):
    """
    把 batch 拆成单图，然后计算额外指标。
    """
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
        row["batch_stress"] = float(batch_stress)
        row["batch_infer_time"] = float(infer_time)
        rows.append(row)

    return rows


# =========================================================
# 6. 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="sc_deepgd_refine_metrics.csv")

    parser.add_argument("--refine-steps", type=int, default=30)
    parser.add_argument("--refine-lr", type=float, default=0.005)

    parser.add_argument("--lambda-stress", type=float, default=1.0)
    parser.add_argument("--lambda-intra", type=float, default=0.1)
    parser.add_argument("--lambda-inter", type=float, default=0.05)
    parser.add_argument("--lambda-pair", type=float, default=0.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)

    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--pair-margin", type=float, default=2.0)

    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] device = {device}")

    print("[INFO] loading Rome dataset...")
    dataset = RomeDataset(
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

    test_loader = pyg.loader.DataLoader(
        datalist[10000:11000],
        batch_size=args.batch_size,
        shuffle=False,
    )

    stress_metric = Stress()
    all_rows = []

    baseline_stress_values = []
    refine_stress_values = []

    for batch_id, batch in enumerate(tqdm(test_loader, desc="SC-DeepGD refine")):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)
        batch.perm_index = batch.perm_index.long()
        batch.batch = batch.batch.long()

        # -------------------------------------------------
        # 1. DeepGD baseline 布局
        # -------------------------------------------------
        start_time = time.time()

        with torch.no_grad():
            pred = model(
                init_pos=generate_init_pos(batch),
                edge_index=batch.perm_index.long(),
                edge_attr=get_edge_features(batch.apsp_attr),
                batch_index=batch.batch.long(),
            )

            base_pos = rescale_by_stress(
                pred,
                batch.apsp_attr,
                batch.perm_index.long(),
                batch.batch.long(),
            )

            base_stress = stress_metric(
                base_pos,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long(),
            ).item()

        infer_time = time.time() - start_time
        baseline_stress_values.append(base_stress)

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=base_pos,
                batch=batch,
                method_name="DeepGD",
                batch_id=batch_id,
                batch_stress=base_stress,
                infer_time=infer_time,
            )
        )

        # -------------------------------------------------
        # 2. SC-DeepGD-Refine / Contrast 细化
        # -------------------------------------------------
        refine_start = time.time()

        refined_pos = refine_positions(
            pos=base_pos,
            batch=batch,
            stress_metric=stress_metric,
            steps=args.refine_steps,
            lr=args.refine_lr,
            lambda_stress=args.lambda_stress,
            lambda_intra=args.lambda_intra,
            lambda_inter=args.lambda_inter,
            lambda_pair=args.lambda_pair,
            lambda_anchor=args.lambda_anchor,
            margin=args.margin,
            pair_margin=args.pair_margin,
        )

        refine_time = time.time() - refine_start

        with torch.no_grad():
            refine_stress = stress_metric(
                refined_pos,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long(),
            ).item()

        refine_stress_values.append(refine_stress)

        if args.lambda_pair > 0:
            method_name = "SC-DeepGD-Contrast"
        else:
            method_name = "SC-DeepGD-Refine"

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=refined_pos,
                batch=batch,
                method_name=method_name,
                batch_id=batch_id,
                batch_stress=refine_stress,
                infer_time=refine_time,
            )
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n========== Stress Summary ==========")
    print(f"DeepGD Mean Stress: {np.mean(baseline_stress_values):.6f}")
    print(f"SC-Refine Mean Stress: {np.mean(refine_stress_values):.6f}")

    print("\n========== Mean Metrics by Method ==========")

    metric_cols = [
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
    ]

    summary = df.groupby("method")[metric_cols].mean()
    print(summary)

    print(f"\n[INFO] saved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()