# sc_deepgd_mog_moe_gps_ecr.py
# -*- coding: utf-8 -*-

"""
MoG-MoE-GPS-ECR-DeepGD

基于 MoG 思想的多视图多专家布局细化方法。

相较于 MoE-GPS-ECR：
1. GPS Expert：社区级全局感知，直接预测 delta_gps；
2. Original-ECR Expert：基于原始边 edge_index 的坐标方向更新；
3. Intra-KNN-ECR Expert：基于同社区 KNN 边的坐标方向更新；
4. Inter-Skeleton-ECR Expert：基于跨社区骨架边的坐标方向更新；
5. Router：为每个节点自适应融合四类 expert 的 delta_pos。

注意：
这是测试阶段 layout refinement 方法，不重新训练 DeepGD 主干。
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
)

from sc_deepgd_moe_gps_ecr import (
    set_random_seed,
    build_batch_community_labels,
    community_refine_loss,
    community_pair_contrast_loss,
    evaluate_pos_by_graph,
    get_split_data,
)


# =========================================================
# 1. 构建多视图边
# =========================================================

def build_intra_community_knn_edges(pos, labels, batch_vector, k=3):
    """
    同社区 KNN 视图：
    在每个图、每个社区内部，根据当前坐标选择 K 个近邻。
    作用：增强社区内部紧凑性。
    """
    device = pos.device
    labels = labels.long()
    batch_vector = batch_vector.long()

    edge_list = []

    with torch.no_grad():
        pos_detach = pos.detach()

        for gid in batch_vector.unique(sorted=True):
            graph_mask = batch_vector == gid
            global_nodes = torch.where(graph_mask)[0]
            labels_g = labels[graph_mask]

            for lab in labels_g.unique(sorted=True):
                comm_mask = labels_g == lab
                comm_nodes = global_nodes[comm_mask]

                n = comm_nodes.numel()
                if n <= 1:
                    continue

                kk = min(k, n - 1)

                comm_pos = pos_detach[comm_nodes]
                dist = torch.cdist(comm_pos, comm_pos, p=2)
                dist.fill_diagonal_(float("inf"))

                nbr = torch.topk(dist, kk, largest=False, dim=1).indices

                src = comm_nodes.repeat_interleave(kk)
                dst = comm_nodes[nbr.reshape(-1)]

                edge_list.append(torch.stack([src, dst], dim=0))

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat(edge_list, dim=1).long().to(device)
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def build_inter_community_skeleton_edges(pos, labels, batch_vector, inter_k=2):
    """
    跨社区骨架视图：
    每个社区连接最近的若干个其它社区。
    具体做法：
    1. 计算每个社区 centroid；
    2. 每个社区选择最近 inter_k 个社区；
    3. 用两个社区中最靠近对方 centroid 的节点建立双向边。

    作用：
    保持社区之间的骨架联系，避免社区完全断裂。
    """
    device = pos.device
    labels = labels.long()
    batch_vector = batch_vector.long()

    edge_list = []

    with torch.no_grad():
        pos_detach = pos.detach()

        for gid in batch_vector.unique(sorted=True):
            graph_mask = batch_vector == gid
            global_nodes = torch.where(graph_mask)[0]
            labels_g = labels[graph_mask]
            unique_labels = labels_g.unique(sorted=True)

            if unique_labels.numel() <= 1:
                continue

            comm_nodes_list = []
            centroids = []

            for lab in unique_labels:
                local_mask = labels_g == lab
                comm_nodes = global_nodes[local_mask]

                if comm_nodes.numel() == 0:
                    continue

                comm_nodes_list.append(comm_nodes)
                centroids.append(pos_detach[comm_nodes].mean(dim=0))

            c = len(centroids)
            if c <= 1:
                continue

            centroids = torch.stack(centroids, dim=0)
            cdists = torch.cdist(centroids, centroids, p=2)
            cdists.fill_diagonal_(float("inf"))

            kk = min(inter_k, c - 1)
            nbr_comms = torch.topk(cdists, kk, largest=False, dim=1).indices

            for ci in range(c):
                nodes_i = comm_nodes_list[ci]
                centroid_i = centroids[ci]

                for cj_tensor in nbr_comms[ci]:
                    cj = int(cj_tensor.item())
                    nodes_j = comm_nodes_list[cj]
                    centroid_j = centroids[cj]

                    # i 社区中最靠近 j centroid 的边界代表节点
                    dist_i = torch.norm(pos_detach[nodes_i] - centroid_j, dim=1)
                    node_i = nodes_i[torch.argmin(dist_i)]

                    # j 社区中最靠近 i centroid 的边界代表节点
                    dist_j = torch.norm(pos_detach[nodes_j] - centroid_i, dim=1)
                    node_j = nodes_j[torch.argmin(dist_j)]

                    edge_list.append(
                        torch.tensor(
                            [[int(node_i.item()), int(node_j.item())],
                             [int(node_j.item()), int(node_i.item())]],
                            dtype=torch.long,
                            device=device,
                        )
                    )

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat(edge_list, dim=1).long().to(device)
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


# =========================================================
# 2. MoG-MoE-GPS-ECR Refiner
# =========================================================

class MoGMoEGPSECRRefiner(nn.Module):
    """
    多视图多专家坐标细化网络。

    4 个 expert：
    1. GPS Expert
    2. Original Edge ECR Expert
    3. Intra-community KNN ECR Expert
    4. Inter-community Skeleton ECR Expert
    """

    def __init__(
        self,
        node_in_dim=6,
        hidden_dim=64,
        heads=4,
        gps_delta_scale=0.20,
        ori_delta_scale=0.10,
        intra_delta_scale=0.10,
        inter_delta_scale=0.08,
        router_temperature=1.0,
        intra_knn_k=3,
        inter_comm_k=2,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gps_delta_scale = gps_delta_scale
        self.ori_delta_scale = ori_delta_scale
        self.intra_delta_scale = intra_delta_scale
        self.inter_delta_scale = inter_delta_scale
        self.router_temperature = router_temperature

        self.intra_knn_k = intra_knn_k
        self.inter_comm_k = inter_comm_k

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

        # GPS Expert
        self.gps_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        # ECR Expert edge message input:
        # h_i, h_j, dist, same_comm, deg_i, deg_j, deg_diff
        edge_in_dim = hidden_dim * 2 + 5

        self.ori_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)
        self.intra_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)
        self.inter_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)

        # Router 输出 4 个 expert 权重
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _make_edge_mlp(self, edge_in_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def build_node_features(self, pos, labels, batch_vector, edge_index):
        device = pos.device
        n = pos.shape[0]

        edge_index = edge_index.long()
        labels = labels.long()
        batch_vector = batch_vector.long()

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
            dim=-1,
        )

        feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        feats = torch.clamp(feats, -10.0, 10.0)

        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True, unbiased=False) + 1e-6
        feats = (feats - mean) / std
        feats = torch.clamp(feats, -3.0, 3.0)

        aux_feats = torch.stack(
            [
                deg,
                dist_to_comm,
                comm_size_feat,
                dist_to_graph_center,
            ],
            dim=-1,
        )

        aux_feats = torch.nan_to_num(aux_feats, nan=0.0, posinf=1.0, neginf=-1.0)
        aux_feats = torch.clamp(aux_feats, 0.0, 1.0)

        return feats, deg, aux_feats

    def build_community_context(self, node_repr, labels, batch_vector):
        comm_context_all = torch.zeros_like(node_repr)

        for gid in batch_vector.unique(sorted=True):
            graph_mask = batch_vector == gid
            global_nodes = torch.where(graph_mask)[0]
            labels_g = labels[graph_mask]
            unique_labels = labels_g.unique(sorted=True)

            if global_nodes.numel() == 0:
                continue

            comm_tokens = []
            comm_node_indices = []

            for lab in unique_labels:
                local_mask = labels_g == lab
                local_nodes = global_nodes[local_mask]

                if local_nodes.numel() == 0:
                    continue

                token = node_repr[local_nodes].mean(dim=0)
                comm_tokens.append(token)
                comm_node_indices.append(local_nodes)

            if len(comm_tokens) == 0:
                continue

            comm_tokens = torch.stack(comm_tokens, dim=0)
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
                    self.hidden_dim,
                )

        return comm_context_all

    def gps_expert(self, node_context):
        delta = self.gps_delta_head(node_context)
        delta = self.gps_delta_scale * torch.tanh(delta)
        return delta

    def ecr_expert(self, pos, node_context, labels, edge_index, deg, edge_mlp, delta_scale):
        device = pos.device
        n = pos.shape[0]

        if edge_index.numel() == 0:
            return torch.zeros_like(pos)

        row = edge_index[0].long()
        col = edge_index[1].long()

        rel = pos[col] - pos[row]
        dist = torch.norm(rel, dim=1, keepdim=True) + 1e-6
        direction = rel / dist

        dist_norm = dist / (dist.mean() + 1e-6)
        dist_norm = torch.clamp(dist_norm, 0.0, 5.0)

        same_comm = (labels[row] == labels[col]).float().unsqueeze(-1)

        deg_i = deg[row].unsqueeze(-1)
        deg_j = deg[col].unsqueeze(-1)
        deg_diff = torch.abs(deg_i - deg_j)

        edge_feat = torch.cat(
            [
                node_context[row],
                node_context[col],
                dist_norm,
                same_comm,
                deg_i,
                deg_j,
                deg_diff,
            ],
            dim=-1,
        )

        scalar = edge_mlp(edge_feat)
        scalar = torch.tanh(scalar)

        msg = scalar * direction

        delta = torch.zeros_like(pos)
        delta.index_add_(0, row, msg)

        count = torch.zeros((n, 1), device=device)
        count.index_add_(0, row, torch.ones((row.shape[0], 1), device=device))

        delta = delta / (count + 1e-6)
        delta = delta_scale * torch.tanh(delta)

        return delta

    def forward(self, pos, labels, batch_vector, edge_index):
        labels = labels.long()
        batch_vector = batch_vector.long()
        edge_index = edge_index.long()

        node_feats, deg, aux_feats = self.build_node_features(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
            edge_index=edge_index,
        )

        node_repr = self.node_encoder(node_feats)

        comm_context = self.build_community_context(
            node_repr=node_repr,
            labels=labels,
            batch_vector=batch_vector,
        )

        node_context = torch.cat([node_repr, comm_context], dim=-1)
        node_context = self.fuse(node_context)

        # Expert 1: GPS
        delta_gps = self.gps_expert(node_context)

        # View 1: original edge
        ori_edge_index = edge_index

        # View 2: intra-community KNN
        intra_edge_index = build_intra_community_knn_edges(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
            k=self.intra_knn_k,
        )

        # View 3: inter-community skeleton
        inter_edge_index = build_inter_community_skeleton_edges(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
            inter_k=self.inter_comm_k,
        )

        # Expert 2: original ECR
        delta_ori = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=ori_edge_index,
            deg=deg,
            edge_mlp=self.ori_edge_mlp,
            delta_scale=self.ori_delta_scale,
        )

        # Expert 3: intra-community KNN ECR
        delta_intra = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=intra_edge_index,
            deg=deg,
            edge_mlp=self.intra_edge_mlp,
            delta_scale=self.intra_delta_scale,
        )

        # Expert 4: inter-community skeleton ECR
        delta_inter = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=inter_edge_index,
            deg=deg,
            edge_mlp=self.inter_edge_mlp,
            delta_scale=self.inter_delta_scale,
        )

        router_input = torch.cat([node_context, aux_feats], dim=-1)
        logits = self.router(router_input)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)

        w_gps = weights[:, 0:1]
        w_ori = weights[:, 1:2]
        w_intra = weights[:, 2:3]
        w_inter = weights[:, 3:4]

        delta = (
            w_gps * delta_gps
            + w_ori * delta_ori
            + w_intra * delta_intra
            + w_inter * delta_inter
        )

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta, weights


# =========================================================
# 3. refinement
# =========================================================

def refine_positions_mog_moe_gps_ecr(
    pos,
    batch,
    stress_metric,
    steps=80,
    lr=0.003,

    lambda_stress=0.8,
    lambda_intra=0.55,
    lambda_inter=0.35,
    lambda_pair=0.05,
    lambda_anchor=0.15,

    lambda_mog_reg=0.01,
    lambda_balance_reg=0.001,

    margin=4.8,
    pair_margin=3.0,

    use_mog=True,
    mog_lr=0.001,
    mog_hidden=64,
    mog_heads=4,

    gps_delta_scale=0.20,
    ori_delta_scale=0.10,
    intra_delta_scale=0.10,
    inter_delta_scale=0.08,

    router_temperature=1.0,
    intra_knn_k=3,
    inter_comm_k=2,

    max_pairs=4096,
):
    edge_index_apsp = batch.perm_index.long()
    batch_index = batch.batch.long()

    labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
    )

    pos0 = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)

    if use_mog:
        mog_refiner = MoGMoEGPSECRRefiner(
            node_in_dim=6,
            hidden_dim=mog_hidden,
            heads=mog_heads,
            gps_delta_scale=gps_delta_scale,
            ori_delta_scale=ori_delta_scale,
            intra_delta_scale=intra_delta_scale,
            inter_delta_scale=inter_delta_scale,
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
        ).to(pos.device)

        optimizer = torch.optim.Adam(
            [
                {"params": [pos_var], "lr": lr},
                {"params": mog_refiner.parameters(), "lr": mog_lr},
            ]
        )
    else:
        mog_refiner = None
        optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        if use_mog and mog_refiner is not None:
            delta, weights = mog_refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
            )

            current_pos = pos_var + delta

            mog_reg_loss = delta.pow(2).sum(dim=1).mean()

            # balance loss，避免 router 只使用一个 expert
            mean_w = weights.mean(dim=0)
            target = torch.ones_like(mean_w) / mean_w.numel()
            balance_reg_loss = ((mean_w - target) ** 2).sum()
        else:
            current_pos = pos_var
            mog_reg_loss = torch.tensor(0.0, device=pos_var.device)
            balance_reg_loss = torch.tensor(0.0, device=pos_var.device)

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
            + lambda_mog_reg * mog_reg_loss
            + lambda_balance_reg * balance_reg_loss
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if use_mog and mog_refiner is not None:
            delta, weights = mog_refiner(
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
# 4. main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="mog_moe_gps_ecr_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.003)

    parser.add_argument("--lambda-stress", type=float, default=0.8)
    parser.add_argument("--lambda-intra", type=float, default=0.55)
    parser.add_argument("--lambda-inter", type=float, default=0.35)
    parser.add_argument("--lambda-pair", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.15)

    parser.add_argument("--lambda-mog-reg", type=float, default=0.01)
    parser.add_argument("--lambda-balance-reg", type=float, default=0.001)

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-mog", action="store_true")
    parser.add_argument("--mog-lr", type=float, default=0.001)
    parser.add_argument("--mog-hidden", type=int, default=64)
    parser.add_argument("--mog-heads", type=int, default=4)

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.08)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)

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

    split_data = get_split_data(datalist, args.split)

    print(f"[INFO] split = {args.split}, graphs = {len(split_data)}")

    test_loader = pyg.loader.DataLoader(
        split_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    stress_metric = Stress()

    all_rows = []
    baseline_stress_values = []
    refine_stress_values = []

    if args.use_mog:
        refined_method_name = "MoG-MoE-GPS-ECR-DeepGD"
    elif args.lambda_pair > 0:
        refined_method_name = "SC-DeepGD-Contrast"
    else:
        refined_method_name = "SC-DeepGD-Refine"

    print(f"[INFO] refined method = {refined_method_name}")

    for batch_id, batch in enumerate(tqdm(test_loader, desc=f"{refined_method_name} refine")):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)
        batch.perm_index = batch.perm_index.long()
        batch.batch = batch.batch.long()

        # DeepGD baseline
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

        # MoG-MoE-GPS-ECR refinement
        refine_start = time.time()

        refined_pos = refine_positions_mog_moe_gps_ecr(
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

            lambda_mog_reg=args.lambda_mog_reg,
            lambda_balance_reg=args.lambda_balance_reg,

            margin=args.margin,
            pair_margin=args.pair_margin,

            use_mog=args.use_mog,
            mog_lr=args.mog_lr,
            mog_hidden=args.mog_hidden,
            mog_heads=args.mog_heads,

            gps_delta_scale=args.gps_delta_scale,
            ori_delta_scale=args.ori_delta_scale,
            intra_delta_scale=args.intra_delta_scale,
            inter_delta_scale=args.inter_delta_scale,

            router_temperature=args.router_temperature,
            intra_knn_k=args.intra_knn_k,
            inter_comm_k=args.inter_comm_k,

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