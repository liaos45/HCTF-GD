# sc_deepgd_gpslite.py
# -*- coding: utf-8 -*-

"""
GPSLite-SC-DeepGD

在 DeepGD 输出布局的基础上，引入 Community-GPS Lite 全局社区感知模块。

核心思想：
1. DeepGD 生成初始布局；
2. 检测拓扑社区；
3. 将每个社区聚合为 community token；
4. 在 community token 之间做轻量级 Multi-head Attention；
5. 将社区全局信息广播回节点；
6. 预测节点坐标修正量 delta_pos；
7. 联合 stress、社区内聚、社区分离、节点级社区对比损失进行测试阶段细化。

运行示例：
python sc_deepgd_gpslite.py --max-test-batches 2 --refine-steps 80 --refine-lr 0.003 --lambda-stress 0.8 --lambda-intra 0.5 --lambda-inter 0.3 --lambda-pair 0.05 --lambda-anchor 0.15 --margin 4.5 --pair-margin 3.0 --use-gps-lite --gps-lr 0.001 --gps-hidden 64 --gps-heads 4 --gps-delta-scale 0.2 --output-csv gpslite_test_2batch.csv
"""

import argparse
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# 0. 随机种子
# =========================================================

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# 2. 社区中心级约束损失
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
# 3. 节点级社区对比损失
# =========================================================

def community_pair_contrast_loss(
    pos,
    labels,
    batch_vector,
    pair_margin=3.0,
    max_pairs=4096,
):
    """
    节点级社区对比损失：
    1. 同社区节点对距离尽量小；
    2. 不同社区节点对距离至少大于 pair_margin。
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
            loss = loss + same_dist.pow(2).mean()

        if diff_dist.numel() > 0:
            loss = loss + torch.relu(pair_margin - diff_dist).pow(2).mean()

        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


# =========================================================
# 4. Community-GPS Lite 模块
# =========================================================

class CommunityGPSLiteRefiner(nn.Module):
    """
    Community-GPS Lite 全局社区感知坐标细化模块。

    输入节点特征：
    1. x 坐标
    2. y 坐标
    3. 节点度
    4. 到社区中心距离
    5. 社区规模
    6. 到图中心距离

    模块结构：
    node encoder
        ↓
    community pooling
        ↓
    community token attention
        ↓
    broadcast back to nodes
        ↓
    delta head 输出坐标修正量
    """

    def __init__(
        self,
        node_in_dim=6,
        hidden_dim=64,
        heads=4,
        delta_scale=0.2,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.delta_scale = delta_scale

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.comm_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.comm_norm = nn.LayerNorm(hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def build_node_features(self, pos, labels, batch_vector, edge_index):
        """
        构建节点特征。
        """
        device = pos.device
        n = pos.shape[0]

        edge_index = edge_index.long()
        batch_vector = batch_vector.long()
        labels = labels.long()

        deg = torch.zeros(n, device=device)

        if edge_index.numel() > 0:
            ones = torch.ones(edge_index.shape[1], device=device)
            deg.scatter_add_(0, edge_index[0], ones)
            deg.scatter_add_(0, edge_index[1], ones)

        deg = torch.log1p(deg)
        deg = deg / (deg.max() + 1e-6)

        dist_to_comm = torch.zeros(n, device=device)
        comm_size_feat = torch.zeros(n, device=device)
        dist_to_graph_center = torch.zeros(n, device=device)

        for gid in batch_vector.unique(sorted=True):
            graph_mask = batch_vector == gid
            pos_g = pos[graph_mask]
            labels_g = labels[graph_mask]
            global_nodes = torch.where(graph_mask)[0]

            if pos_g.shape[0] == 0:
                continue

            graph_center = pos_g.mean(dim=0)

            graph_dist = torch.norm(pos_g - graph_center, dim=1)
            graph_dist = graph_dist / (graph_dist.max() + 1e-6)

            dist_to_graph_center[global_nodes] = graph_dist

            for lab in labels_g.unique(sorted=True):
                local_mask = labels_g == lab
                local_nodes = global_nodes[local_mask]
                comm_pos = pos[local_nodes]

                if comm_pos.shape[0] == 0:
                    continue

                centroid = comm_pos.mean(dim=0)
                d = torch.norm(comm_pos - centroid, dim=1)
                d = d / (d.max() + 1e-6)

                size_value = float(comm_pos.shape[0]) / float(pos_g.shape[0] + 1e-6)

                dist_to_comm[local_nodes] = d
                comm_size_feat[local_nodes] = size_value

        feats = torch.stack(
            [
                pos[:, 0],
                pos[:, 1],
                deg,
                dist_to_comm,
                comm_size_feat,
                dist_to_graph_center,
            ],
            dim=-1
        )

        feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        feats = torch.clamp(feats, -10.0, 10.0)

        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True, unbiased=False) + 1e-6
        feats = (feats - mean) / std
        feats = torch.clamp(feats, -3.0, 3.0)

        return feats

    def forward(self, pos, labels, batch_vector, edge_index):
        """
        输出 delta_pos: [N, 2]
        """
        device = pos.device
        n = pos.shape[0]

        node_feats = self.build_node_features(pos, labels, batch_vector, edge_index)
        node_repr = self.node_encoder(node_feats)

        comm_context_all = torch.zeros_like(node_repr)

        for gid in batch_vector.unique(sorted=True):
            graph_mask = batch_vector == gid
            global_nodes = torch.where(graph_mask)[0]

            labels_g = labels[graph_mask]
            node_repr_g = node_repr[graph_mask]

            unique_labels = labels_g.unique(sorted=True)

            if node_repr_g.shape[0] == 0:
                continue

            comm_tokens = []
            comm_node_indices = []

            for lab in unique_labels:
                local_mask = labels_g == lab
                local_nodes = global_nodes[local_mask]
                comm_node_indices.append(local_nodes)

                token = node_repr[local_nodes].mean(dim=0)
                comm_tokens.append(token)

            if len(comm_tokens) == 0:
                continue

            comm_tokens = torch.stack(comm_tokens, dim=0)  # [C, H]

            # MultiheadAttention 输入 [B, C, H]
            comm_input = comm_tokens.unsqueeze(0)

            attn_out, _ = self.comm_attn(
                comm_input,
                comm_input,
                comm_input,
                need_weights=False,
            )

            attn_out = self.comm_norm(attn_out.squeeze(0) + comm_tokens)

            for idx, local_nodes in enumerate(comm_node_indices):
                comm_context_all[local_nodes] = attn_out[idx].unsqueeze(0).expand(
                    local_nodes.shape[0],
                    self.hidden_dim
                )

        fused = torch.cat([node_repr, comm_context_all], dim=-1)
        fused = self.fuse(fused)

        delta = self.delta_head(fused)
        delta = self.delta_scale * torch.tanh(delta)

        if torch.isnan(delta).any():
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta


# =========================================================
# 5. GPS-Lite 坐标细化
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
    lambda_gps_reg=0.01,
    margin=3.0,
    pair_margin=2.0,
    use_gps_lite=False,
    gps_lr=0.001,
    gps_hidden=64,
    gps_heads=4,
    gps_delta_scale=0.2,
    max_pairs=4096,
):
    """
    对 DeepGD 输出坐标进行测试阶段细化。

    如果 use_gps_lite=True：
        使用 CommunityGPSLiteRefiner 预测 delta_pos。

    如果 use_gps_lite=False：
        退化为普通坐标优化。
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

    gps_refiner = None

    if use_gps_lite:
        gps_refiner = CommunityGPSLiteRefiner(
            node_in_dim=6,
            hidden_dim=gps_hidden,
            heads=gps_heads,
            delta_scale=gps_delta_scale,
        ).to(pos.device)

        optimizer = torch.optim.Adam(
            [
                {"params": [pos_var], "lr": lr},
                {"params": gps_refiner.parameters(), "lr": gps_lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        if use_gps_lite and gps_refiner is not None:
            delta = gps_refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
            )
            current_pos = pos_var + delta
            gps_reg_loss = delta.pow(2).sum(dim=1).mean()
        else:
            current_pos = pos_var
            gps_reg_loss = torch.tensor(0.0, device=pos_var.device)

        stress_loss = stress_metric(
            current_pos,
            edge_index_apsp,
            batch.apsp_attr,
            batch_index,
        )

        intra_loss, inter_loss = community_refine_loss(
            current_pos,
            labels,
            batch_index,
            margin=margin,
        )

        if lambda_pair > 0:
            pair_loss = community_pair_contrast_loss(
                current_pos,
                labels,
                batch_index,
                pair_margin=pair_margin,
                max_pairs=max_pairs,
            )
        else:
            pair_loss = torch.tensor(0.0, device=pos_var.device)

        anchor_loss = ((current_pos - pos0) ** 2).sum(dim=1).mean()

        loss = (
            lambda_stress * stress_loss
            + lambda_intra * intra_loss
            + lambda_inter * inter_loss
            + lambda_pair * pair_loss
            + lambda_anchor * anchor_loss
            + lambda_gps_reg * gps_reg_loss
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if use_gps_lite and gps_refiner is not None:
            delta = gps_refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
            )
            final_pos = pos_var + delta
        else:
            final_pos = pos_var

        refined_pos = rescale_by_stress(
            final_pos.detach(),
            batch.apsp_attr,
            edge_index_apsp,
            batch_index,
        )

    return refined_pos


# =========================================================
# 6. 评价 batch 布局
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
# 7. 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="gpslite_sc_deepgd_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--refine-steps", type=int, default=30)
    parser.add_argument("--refine-lr", type=float, default=0.005)

    parser.add_argument("--lambda-stress", type=float, default=1.0)
    parser.add_argument("--lambda-intra", type=float, default=0.1)
    parser.add_argument("--lambda-inter", type=float, default=0.05)
    parser.add_argument("--lambda-pair", type=float, default=0.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--lambda-gps-reg", type=float, default=0.01)

    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--pair-margin", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-gps-lite", action="store_true")
    parser.add_argument("--gps-lr", type=float, default=0.001)
    parser.add_argument("--gps-hidden", type=int, default=64)
    parser.add_argument("--gps-heads", type=int, default=4)
    parser.add_argument("--gps-delta-scale", type=float, default=0.2)

    args = parser.parse_args()

    set_random_seed(args.seed)

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

    if args.use_gps_lite:
        refined_method_name = "GPSLite-SC-DeepGD"
    elif args.lambda_pair > 0:
        refined_method_name = "SC-DeepGD-Contrast"
    else:
        refined_method_name = "SC-DeepGD-Refine"

    print(f"[INFO] refined method = {refined_method_name}")

    for batch_id, batch in enumerate(tqdm(test_loader, desc="GPSLite-SC-DeepGD refine")):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)
        batch.perm_index = batch.perm_index.long()
        batch.batch = batch.batch.long()

        # -------------------------------------------------
        # 1. DeepGD baseline
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
        # 2. GPSLite-SC-DeepGD refinement
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
            lambda_gps_reg=args.lambda_gps_reg,
            margin=args.margin,
            pair_margin=args.pair_margin,
            use_gps_lite=args.use_gps_lite,
            gps_lr=args.gps_lr,
            gps_hidden=args.gps_hidden,
            gps_heads=args.gps_heads,
            gps_delta_scale=args.gps_delta_scale,
            max_pairs=args.max_pairs,
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

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=refined_pos,
                batch=batch,
                method_name=refined_method_name,
                batch_id=batch_id,
                batch_stress=refine_stress,
                infer_time=refine_time,
            )
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n========== Stress Summary ==========")
    print(f"DeepGD Mean Stress: {np.mean(baseline_stress_values):.6f}")
    print(f"{refined_method_name} Mean Stress: {np.mean(refine_stress_values):.6f}")

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