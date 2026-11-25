# sc_deepgd_xcross_spectral_mog_moe_gps_ecr.py
# -*- coding: utf-8 -*-

"""
XCross-Spectral-MoG-MoE-GPS-ECR-DeepGD

在 Spectral-MoG-MoE-GPS-ECR 基础上继续加入：
1. XCross-ECR Expert：边交叉感知坐标修正专家；
2. Soft Crossing Loss：可微近似边交叉惩罚项。

专家组成：
1. GPS Expert：社区级全局感知；
2. Original Edge ECR Expert：原始边坐标细化；
3. Intra-community KNN ECR Expert：社区内 KNN 视图；
4. Inter-community Skeleton ECR Expert：跨社区骨架视图；
5. Spectral-KNN ECR Expert：Laplacian 谱空间近邻视图；
6. XCross-ECR Expert：潜在交叉边对的几何排斥视图。

方法定位：
这是基于预训练 DeepGD 初始布局的 test-time refinement 方法，
不重新训练 DeepGD 主干。
"""

import argparse
import time
import random
import math

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
)

from sc_deepgd_mog_moe_gps_ecr import (
    build_intra_community_knn_edges,
    build_inter_community_skeleton_edges,
)

from sc_deepgd_spectral_mog_moe_gps_ecr import (
    build_spectral_knn_edges,
    SpectralMoGMoEGPSECRRefiner,
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
# 1. Crossing candidate view
# =========================================================

def _orientation_2d(a, b, c):
    """
    CPU / Python 几何方向判断。
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4):
    """
    判断两条线段是否相交。
    这里用于候选边对筛选，不参与反向传播。
    """
    o1 = _orientation_2d(p1, p2, p3)
    o2 = _orientation_2d(p1, p2, p4)
    o3 = _orientation_2d(p3, p4, p1)
    o4 = _orientation_2d(p3, p4, p2)

    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def build_crossing_candidate_pairs(
    pos,
    edge_index,
    batch_vector,
    max_cross_pairs=4096,
    bbox_padding=0.05,
):
    """
    构造潜在交叉边对。

    输出：
        pairs: LongTensor [P, 4]
        每一行是 [a, b, c, d]，代表边 (a,b) 和边 (c,d)。

    说明：
    1. 只在每张图内部找边对；
    2. 两条边不能共享端点；
    3. 优先保留真实几何相交的边对；
    4. 如果实际交叉边对过少，也会保留包围盒接近的候选边对。
    """
    device = pos.device

    pos_cpu = pos.detach().cpu()
    edge_index_cpu = edge_index.long().detach().cpu()
    batch_cpu = batch_vector.long().detach().cpu()

    all_pairs = []
    fallback_pairs = []

    graph_ids = batch_cpu.unique(sorted=True).tolist()

    for gid in graph_ids:
        node_mask = batch_cpu == gid
        global_nodes = torch.where(node_mask)[0]
        node_set = set(global_nodes.tolist())

        # 构造无向边列表，去重
        edge_set = set()
        for u, v in edge_index_cpu.t().tolist():
            if u == v:
                continue
            if u in node_set and v in node_set:
                a, b = sorted((int(u), int(v)))
                edge_set.add((a, b))

        edges = list(edge_set)
        m = len(edges)

        if m <= 1:
            continue

        # Rome 图一般不大，直接双循环足够。
        # 为了避免极端情况太慢，限制每张图候选边对数量。
        local_true_pairs = []
        local_fallback_pairs = []

        for i in range(m):
            a, b = edges[i]
            pa = pos_cpu[a].numpy()
            pb = pos_cpu[b].numpy()

            min_ab_x = min(pa[0], pb[0]) - bbox_padding
            max_ab_x = max(pa[0], pb[0]) + bbox_padding
            min_ab_y = min(pa[1], pb[1]) - bbox_padding
            max_ab_y = max(pa[1], pb[1]) + bbox_padding

            for j in range(i + 1, m):
                c, d = edges[j]

                # 共享端点不算边交叉
                if len({a, b, c, d}) < 4:
                    continue

                pc = pos_cpu[c].numpy()
                pd = pos_cpu[d].numpy()

                min_cd_x = min(pc[0], pd[0]) - bbox_padding
                max_cd_x = max(pc[0], pd[0]) + bbox_padding
                min_cd_y = min(pc[1], pd[1]) - bbox_padding
                max_cd_y = max(pc[1], pd[1]) + bbox_padding

                # 包围盒不重叠，基本不可能交叉
                if max_ab_x < min_cd_x or max_cd_x < min_ab_x:
                    continue
                if max_ab_y < min_cd_y or max_cd_y < min_ab_y:
                    continue

                if _segments_intersect(pa, pb, pc, pd):
                    local_true_pairs.append([a, b, c, d])
                else:
                    local_fallback_pairs.append([a, b, c, d])

        all_pairs.extend(local_true_pairs)
        fallback_pairs.extend(local_fallback_pairs)

        if len(all_pairs) >= max_cross_pairs:
            break

    if len(all_pairs) < max_cross_pairs and len(fallback_pairs) > 0:
        random.shuffle(fallback_pairs)
        need = max_cross_pairs - len(all_pairs)
        all_pairs.extend(fallback_pairs[:need])

    if len(all_pairs) == 0:
        return torch.empty((0, 4), dtype=torch.long, device=device)

    if len(all_pairs) > max_cross_pairs:
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_cross_pairs]

    pairs = torch.tensor(all_pairs, dtype=torch.long, device=device)
    return pairs


# =========================================================
# 2. Crossing-aware expert and soft crossing loss
# =========================================================

def crossing_repulsion_delta(
    pos,
    crossing_pairs,
    delta_scale=0.04,
):
    """
    XCross-ECR Expert 的几何排斥坐标更新。

    对候选交叉边对 (a,b) 和 (c,d)：
    1. 计算两条边的中点；
    2. 让两条边沿中点差方向轻微分离；
    3. 得到 delta_cross。
    """
    device = pos.device
    n = pos.shape[0]

    if crossing_pairs.numel() == 0:
        return torch.zeros_like(pos)

    a = crossing_pairs[:, 0].long()
    b = crossing_pairs[:, 1].long()
    c = crossing_pairs[:, 2].long()
    d = crossing_pairs[:, 3].long()

    mid_ab = 0.5 * (pos[a] + pos[b])
    mid_cd = 0.5 * (pos[c] + pos[d])

    direction = mid_ab - mid_cd
    norm = torch.norm(direction, dim=1, keepdim=True) + 1e-6
    direction = direction / norm

    # 避免过大更新
    direction = torch.tanh(direction)

    delta = torch.zeros_like(pos)
    count = torch.zeros((n, 1), device=device)

    # 边 ab 往 +direction 方向移动
    delta.index_add_(0, a, direction)
    delta.index_add_(0, b, direction)

    # 边 cd 往 -direction 方向移动
    delta.index_add_(0, c, -direction)
    delta.index_add_(0, d, -direction)

    ones = torch.ones((crossing_pairs.shape[0], 1), device=device)

    count.index_add_(0, a, ones)
    count.index_add_(0, b, ones)
    count.index_add_(0, c, ones)
    count.index_add_(0, d, ones)

    delta = delta / (count + 1e-6)
    delta = delta_scale * torch.tanh(delta)

    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    return delta


def cross2d(u, v):
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def soft_crossing_loss(
    pos,
    crossing_pairs,
    beta=5.0,
):
    """
    可微近似边交叉损失。

    对候选边对 (a,b) 和 (c,d)：
    线段相交通常满足：
        orient(a,b,c) * orient(a,b,d) < 0
        orient(c,d,a) * orient(c,d,b) < 0

    用 sigmoid 做 soft relaxation。
    """
    if crossing_pairs.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    a = crossing_pairs[:, 0].long()
    b = crossing_pairs[:, 1].long()
    c = crossing_pairs[:, 2].long()
    d = crossing_pairs[:, 3].long()

    xa = pos[a]
    xb = pos[b]
    xc = pos[c]
    xd = pos[d]

    ab = xb - xa
    cd = xd - xc

    # 归一化方向，减少尺度影响
    ab_norm = torch.norm(ab, dim=1, keepdim=True) + 1e-6
    cd_norm = torch.norm(cd, dim=1, keepdim=True) + 1e-6

    ab_u = ab / ab_norm
    cd_u = cd / cd_norm

    o1 = cross2d(ab_u, xc - xa)
    o2 = cross2d(ab_u, xd - xa)
    o3 = cross2d(cd_u, xa - xc)
    o4 = cross2d(cd_u, xb - xc)

    # 如果 o1*o2 < 0 且 o3*o4 < 0，则倾向交叉
    s1 = torch.sigmoid(-beta * o1 * o2)
    s2 = torch.sigmoid(-beta * o3 * o4)

    loss = (s1 * s2).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss


# =========================================================
# 3. XCross-Spectral-MoG-MoE-GPS-ECR Refiner
# =========================================================

class XCrossSpectralMoGMoEGPSECRRefiner(SpectralMoGMoEGPSECRRefiner):
    """
    在 Spectral-MoG-MoE-GPS-ECR 上新增 XCross Expert。

    Expert 1: GPS
    Expert 2: Original Edge ECR
    Expert 3: Intra-community KNN ECR
    Expert 4: Inter-community Skeleton ECR
    Expert 5: Spectral-KNN ECR
    Expert 6: XCross-ECR
    """

    def __init__(
        self,
        node_in_dim=6,
        hidden_dim=64,
        heads=4,
        gps_delta_scale=0.20,
        ori_delta_scale=0.10,
        intra_delta_scale=0.10,
        inter_delta_scale=0.04,
        spectral_delta_scale=0.04,
        cross_delta_scale=0.04,
        router_temperature=1.0,
        intra_knn_k=3,
        inter_comm_k=2,
        dropout=0.0,
    ):
        super().__init__(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            gps_delta_scale=gps_delta_scale,
            ori_delta_scale=ori_delta_scale,
            intra_delta_scale=intra_delta_scale,
            inter_delta_scale=inter_delta_scale,
            spectral_delta_scale=spectral_delta_scale,
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
            dropout=dropout,
        )

        self.cross_delta_scale = cross_delta_scale

        # 原来 Spectral 版 router 输出 5 个专家权重，现在改成 6 个
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        pos,
        labels,
        batch_vector,
        edge_index,
        spectral_edge_index,
        crossing_pairs,
    ):
        labels = labels.long()
        batch_vector = batch_vector.long()
        edge_index = edge_index.long()
        spectral_edge_index = spectral_edge_index.long()

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

        # Expert 2: Original Edge ECR
        delta_ori = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=edge_index,
            deg=deg,
            edge_mlp=self.ori_edge_mlp,
            delta_scale=self.ori_delta_scale,
        )

        # Expert 3: Intra-community KNN ECR
        intra_edge_index = build_intra_community_knn_edges(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
            k=self.intra_knn_k,
        )

        delta_intra = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=intra_edge_index,
            deg=deg,
            edge_mlp=self.intra_edge_mlp,
            delta_scale=self.intra_delta_scale,
        )

        # Expert 4: Inter-community Skeleton ECR
        inter_edge_index = build_inter_community_skeleton_edges(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
            inter_k=self.inter_comm_k,
        )

        delta_inter = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=inter_edge_index,
            deg=deg,
            edge_mlp=self.inter_edge_mlp,
            delta_scale=self.inter_delta_scale,
        )

        # Expert 5: Spectral-KNN ECR
        delta_spectral = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=spectral_edge_index,
            deg=deg,
            edge_mlp=self.spectral_edge_mlp,
            delta_scale=self.spectral_delta_scale,
        )

        # Expert 6: XCross-ECR
        delta_cross = crossing_repulsion_delta(
            pos=pos,
            crossing_pairs=crossing_pairs,
            delta_scale=self.cross_delta_scale,
        )

        router_input = torch.cat([node_context, aux_feats], dim=-1)
        logits = self.router(router_input)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)

        w_gps = weights[:, 0:1]
        w_ori = weights[:, 1:2]
        w_intra = weights[:, 2:3]
        w_inter = weights[:, 3:4]
        w_spectral = weights[:, 4:5]
        w_cross = weights[:, 5:6]

        delta = (
            w_gps * delta_gps
            + w_ori * delta_ori
            + w_intra * delta_intra
            + w_inter * delta_inter
            + w_spectral * delta_spectral
            + w_cross * delta_cross
        )

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta, weights


# =========================================================
# 4. Refinement
# =========================================================

def refine_positions_xcross_spectral_mog_moe_gps_ecr(
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

    lambda_spectral_reg=0.01,
    lambda_balance_reg=0.001,
    lambda_cross=0.01,

    margin=4.8,
    pair_margin=3.0,

    use_xcross_spectral_mog=True,
    xcross_lr=0.001,
    xcross_hidden=64,
    xcross_heads=4,

    gps_delta_scale=0.20,
    ori_delta_scale=0.10,
    intra_delta_scale=0.10,
    inter_delta_scale=0.04,
    spectral_delta_scale=0.04,
    cross_delta_scale=0.04,

    router_temperature=1.0,
    intra_knn_k=3,
    inter_comm_k=2,

    spectral_dim=4,
    spectral_knn_k=3,

    max_pairs=4096,
    max_cross_pairs=4096,
    beta_cross=5.0,
):
    edge_index_apsp = batch.perm_index.long()
    batch_index = batch.batch.long()

    labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
    )

    # Spectral-KNN 只依赖拓扑，每个 batch 构造一次
    spectral_edge_index = build_spectral_knn_edges(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
        spectral_dim=spectral_dim,
        spectral_knn_k=spectral_knn_k,
    )

    # Crossing candidates 根据 DeepGD 初始布局构造一次
    crossing_pairs = build_crossing_candidate_pairs(
        pos=pos,
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        max_cross_pairs=max_cross_pairs,
    )

    pos0 = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)

    if use_xcross_spectral_mog:
        refiner = XCrossSpectralMoGMoEGPSECRRefiner(
            node_in_dim=6,
            hidden_dim=xcross_hidden,
            heads=xcross_heads,
            gps_delta_scale=gps_delta_scale,
            ori_delta_scale=ori_delta_scale,
            intra_delta_scale=intra_delta_scale,
            inter_delta_scale=inter_delta_scale,
            spectral_delta_scale=spectral_delta_scale,
            cross_delta_scale=cross_delta_scale,
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
        ).to(pos.device)

        optimizer = torch.optim.Adam(
            [
                {"params": [pos_var], "lr": lr},
                {"params": refiner.parameters(), "lr": xcross_lr},
            ]
        )
    else:
        refiner = None
        optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        if use_xcross_spectral_mog and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                spectral_edge_index=spectral_edge_index,
                crossing_pairs=crossing_pairs,
            )

            current_pos = pos_var + delta

            reg_loss = delta.pow(2).sum(dim=1).mean()

            mean_w = weights.mean(dim=0)
            target = torch.ones_like(mean_w) / mean_w.numel()
            balance_reg_loss = ((mean_w - target) ** 2).sum()

            cross_loss = soft_crossing_loss(
                pos=current_pos,
                crossing_pairs=crossing_pairs,
                beta=beta_cross,
            )
        else:
            current_pos = pos_var
            reg_loss = torch.tensor(0.0, device=pos_var.device)
            balance_reg_loss = torch.tensor(0.0, device=pos_var.device)
            cross_loss = torch.tensor(0.0, device=pos_var.device)

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
            + lambda_spectral_reg * reg_loss
            + lambda_balance_reg * balance_reg_loss
            + lambda_cross * cross_loss
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if use_xcross_spectral_mog and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                spectral_edge_index=spectral_edge_index,
                crossing_pairs=crossing_pairs,
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
# 5. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="xcross_spectral_mog_moe_gps_ecr_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.003)

    parser.add_argument("--lambda-stress", type=float, default=0.8)
    parser.add_argument("--lambda-intra", type=float, default=0.55)
    parser.add_argument("--lambda-inter", type=float, default=0.35)
    parser.add_argument("--lambda-pair", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.15)

    parser.add_argument("--lambda-spectral-reg", type=float, default=0.01)
    parser.add_argument("--lambda-balance-reg", type=float, default=0.001)
    parser.add_argument("--lambda-cross", type=float, default=0.01)

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-xcross-spectral-mog", action="store_true")
    parser.add_argument("--xcross-lr", type=float, default=0.001)
    parser.add_argument("--xcross-hidden", type=int, default=64)
    parser.add_argument("--xcross-heads", type=int, default=4)

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.04)
    parser.add_argument("--spectral-delta-scale", type=float, default=0.04)
    parser.add_argument("--cross-delta-scale", type=float, default=0.04)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)

    parser.add_argument("--spectral-dim", type=int, default=4)
    parser.add_argument("--spectral-knn-k", type=int, default=3)

    parser.add_argument("--max-cross-pairs", type=int, default=4096)
    parser.add_argument("--beta-cross", type=float, default=5.0)

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

    if args.use_xcross_spectral_mog:
        refined_method_name = "XCross-Spectral-MoG-MoE-GPS-ECR-DeepGD"
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
        # 2. XCross-Spectral-MoG-MoE-GPS-ECR refinement
        # -------------------------------------------------
        refine_start = time.time()

        refined_pos = refine_positions_xcross_spectral_mog_moe_gps_ecr(
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

            lambda_spectral_reg=args.lambda_spectral_reg,
            lambda_balance_reg=args.lambda_balance_reg,
            lambda_cross=args.lambda_cross,

            margin=args.margin,
            pair_margin=args.pair_margin,

            use_xcross_spectral_mog=args.use_xcross_spectral_mog,
            xcross_lr=args.xcross_lr,
            xcross_hidden=args.xcross_hidden,
            xcross_heads=args.xcross_heads,

            gps_delta_scale=args.gps_delta_scale,
            ori_delta_scale=args.ori_delta_scale,
            intra_delta_scale=args.intra_delta_scale,
            inter_delta_scale=args.inter_delta_scale,
            spectral_delta_scale=args.spectral_delta_scale,
            cross_delta_scale=args.cross_delta_scale,

            router_temperature=args.router_temperature,
            intra_knn_k=args.intra_knn_k,
            inter_comm_k=args.inter_comm_k,

            spectral_dim=args.spectral_dim,
            spectral_knn_k=args.spectral_knn_k,

            max_pairs=args.max_pairs,
            max_cross_pairs=args.max_cross_pairs,
            beta_cross=args.beta_cross,
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