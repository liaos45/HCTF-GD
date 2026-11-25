# sc_deepgd_multiband_spectral_mog_moe_gps_ecr.py
# -*- coding: utf-8 -*-

"""
MB-Spectral-MoG-MoE-GPS-ECR-DeepGD

在 Spectral-MoG-MoE-GPS-ECR 基础上，将单一 Spectral-KNN Expert
拆分为多频段谱专家：

1. Low-band Spectral Expert：低频谱结构，全局骨架保持
2. Mid-band Spectral Expert：中频谱结构，社区边界与簇结构
3. High-band Spectral Expert：高频谱结构，局部细节与扰动

最终专家：
1. GPS Expert
2. Original Edge ECR Expert
3. Intra-community KNN ECR Expert
4. Inter-community Skeleton ECR Expert
5. Low-band Spectral ECR Expert
6. Mid-band Spectral ECR Expert
7. High-band Spectral ECR Expert

方法定位：
这是基于预训练 DeepGD 初始布局的 test-time refinement 方法，
不重新训练 DeepGD 主干。
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
)

from sc_deepgd_mog_moe_gps_ecr import (
    build_intra_community_knn_edges,
    build_inter_community_skeleton_edges,
    MoGMoEGPSECRRefiner,
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
# 1. Multi-band Spectral-KNN View
# =========================================================

def _knn_edges_from_embedding(
    U,
    global_nodes,
    knn_k=3,
    device=None,
):
    """
    根据谱嵌入 U 构造 KNN 边。
    U: [n, d]
    global_nodes: [n]
    """
    if device is None:
        device = U.device

    n = U.shape[0]

    if n <= 1:
        return None

    kk = min(knn_k, n - 1)

    if kk <= 0:
        return None

    U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

    dist = torch.cdist(U, U, p=2)
    dist.fill_diagonal_(float("inf"))

    nbr = torch.topk(dist, kk, largest=False, dim=1).indices

    src_local = torch.arange(n).repeat_interleave(kk)
    dst_local = nbr.reshape(-1)

    global_nodes = global_nodes.long()

    src = global_nodes[src_local]
    dst = global_nodes[dst_local]

    edges_1 = torch.stack([src, dst], dim=0)
    edges_2 = torch.stack([dst, src], dim=0)

    return torch.cat([edges_1, edges_2], dim=1).long().to(device)


def build_multiband_spectral_knn_edges(
    edge_index,
    batch_vector,
    num_nodes,
    low_dim=4,
    mid_dim=4,
    high_dim=4,
    spectral_knn_k=3,
):
    """
    构造三类谱空间 KNN 视图：

    1. low_edge_index:
       使用低频 Laplacian eigenvectors。
       主要表达全局拓扑骨架。

    2. mid_edge_index:
       使用中频 Laplacian eigenvectors。
       主要表达社区边界与簇结构。

    3. high_edge_index:
       使用高频 Laplacian eigenvectors。
       主要表达局部结构细节。

    注意：
    Rome 图规模不大，部分图的可用特征向量数量有限。
    因此代码会自动做 fallback，保证每个 band 尽量可用。
    """
    device = batch_vector.device

    edge_index_cpu = edge_index.long().detach().cpu()
    batch_cpu = batch_vector.long().detach().cpu()

    low_edges_all = []
    mid_edges_all = []
    high_edges_all = []

    with torch.no_grad():
        graph_ids = batch_cpu.unique(sorted=True).tolist()

        for gid in graph_ids:
            node_mask = batch_cpu == gid
            global_nodes = torch.where(node_mask)[0]
            n = global_nodes.numel()

            if n <= 1:
                continue

            node_set = set(global_nodes.tolist())
            mapping = {int(g): i for i, g in enumerate(global_nodes.tolist())}

            A = torch.zeros((n, n), dtype=torch.float32)

            for u, v in edge_index_cpu.t().tolist():
                if u in node_set and v in node_set:
                    lu = mapping[int(u)]
                    lv = mapping[int(v)]
                    if lu != lv:
                        A[lu, lv] = 1.0
                        A[lv, lu] = 1.0

            if A.sum() <= 0:
                continue

            deg = A.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)

            I = torch.eye(n, dtype=torch.float32)
            L = I - D_inv_sqrt @ A @ D_inv_sqrt
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                eigvals, eigvecs = torch.linalg.eigh(L)
            except RuntimeError:
                eigvecs = torch.arange(n, dtype=torch.float32).view(n, 1)

            # 去掉第一个平凡特征向量
            if eigvecs.shape[1] > 1:
                U_all = eigvecs[:, 1:]
            else:
                U_all = eigvecs[:, :1]

            U_all = torch.nan_to_num(U_all, nan=0.0, posinf=0.0, neginf=0.0)

            m = U_all.shape[1]

            if m <= 0:
                continue

            # ---------- Low band ----------
            low_end = min(low_dim, m)
            U_low = U_all[:, :low_end]

            # ---------- Mid band ----------
            mid_start = min(low_end, m)
            mid_end = min(mid_start + mid_dim, m)

            if mid_end > mid_start:
                U_mid = U_all[:, mid_start:mid_end]
            else:
                # fallback：如果中频维度不足，使用 low band
                U_mid = U_low

            # ---------- High band ----------
            high_use_dim = min(high_dim, m)
            if high_use_dim > 0:
                U_high = U_all[:, -high_use_dim:]
            else:
                U_high = U_low

            low_edges = _knn_edges_from_embedding(
                U=U_low,
                global_nodes=global_nodes,
                knn_k=spectral_knn_k,
                device=device,
            )

            mid_edges = _knn_edges_from_embedding(
                U=U_mid,
                global_nodes=global_nodes,
                knn_k=spectral_knn_k,
                device=device,
            )

            high_edges = _knn_edges_from_embedding(
                U=U_high,
                global_nodes=global_nodes,
                knn_k=spectral_knn_k,
                device=device,
            )

            if low_edges is not None:
                low_edges_all.append(low_edges)

            if mid_edges is not None:
                mid_edges_all.append(mid_edges)

            if high_edges is not None:
                high_edges_all.append(high_edges)

    def merge_edges(edge_list):
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        edge_index_new = torch.cat(edge_list, dim=1).long().to(device)
        edge_index_new = torch.unique(edge_index_new, dim=1)
        return edge_index_new

    low_edge_index = merge_edges(low_edges_all)
    mid_edge_index = merge_edges(mid_edges_all)
    high_edge_index = merge_edges(high_edges_all)

    return low_edge_index, mid_edge_index, high_edge_index


# =========================================================
# 2. MB-Spectral-MoG-MoE-GPS-ECR Refiner
# =========================================================

class MultiBandSpectralMoGMoEGPSECRRefiner(MoGMoEGPSECRRefiner):
    """
    在 MoG-MoE-GPS-ECR 上加入多频段谱专家。

    Expert 1: GPS Expert
    Expert 2: Original Edge ECR Expert
    Expert 3: Intra-community KNN ECR Expert
    Expert 4: Inter-community Skeleton ECR Expert
    Expert 5: Low-band Spectral ECR Expert
    Expert 6: Mid-band Spectral ECR Expert
    Expert 7: High-band Spectral ECR Expert
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
        low_delta_scale=0.04,
        mid_delta_scale=0.04,
        high_delta_scale=0.02,
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
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
            dropout=dropout,
        )

        self.low_delta_scale = low_delta_scale
        self.mid_delta_scale = mid_delta_scale
        self.high_delta_scale = high_delta_scale

        edge_in_dim = hidden_dim * 2 + 5

        self.low_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)
        self.mid_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)
        self.high_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)

        # 原 MoG router 是 4 个专家，现在变成 7 个专家
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(
        self,
        pos,
        labels,
        batch_vector,
        edge_index,
        low_edge_index,
        mid_edge_index,
        high_edge_index,
    ):
        labels = labels.long()
        batch_vector = batch_vector.long()
        edge_index = edge_index.long()
        low_edge_index = low_edge_index.long()
        mid_edge_index = mid_edge_index.long()
        high_edge_index = high_edge_index.long()

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

        # Expert 5: Low-band Spectral ECR
        delta_low = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=low_edge_index,
            deg=deg,
            edge_mlp=self.low_edge_mlp,
            delta_scale=self.low_delta_scale,
        )

        # Expert 6: Mid-band Spectral ECR
        delta_mid = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=mid_edge_index,
            deg=deg,
            edge_mlp=self.mid_edge_mlp,
            delta_scale=self.mid_delta_scale,
        )

        # Expert 7: High-band Spectral ECR
        delta_high = self.ecr_expert(
            pos=pos,
            node_context=node_context,
            labels=labels,
            edge_index=high_edge_index,
            deg=deg,
            edge_mlp=self.high_edge_mlp,
            delta_scale=self.high_delta_scale,
        )

        router_input = torch.cat([node_context, aux_feats], dim=-1)
        logits = self.router(router_input)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)

        w_gps = weights[:, 0:1]
        w_ori = weights[:, 1:2]
        w_intra = weights[:, 2:3]
        w_inter = weights[:, 3:4]
        w_low = weights[:, 4:5]
        w_mid = weights[:, 5:6]
        w_high = weights[:, 6:7]

        delta = (
            w_gps * delta_gps
            + w_ori * delta_ori
            + w_intra * delta_intra
            + w_inter * delta_inter
            + w_low * delta_low
            + w_mid * delta_mid
            + w_high * delta_high
        )

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta, weights


# =========================================================
# 3. Multi-band refinement
# =========================================================

def refine_positions_multiband_spectral_mog_moe_gps_ecr(
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

    lambda_mb_reg=0.01,
    lambda_balance_reg=0.001,

    margin=4.8,
    pair_margin=3.0,

    use_multiband=True,
    mb_lr=0.001,
    mb_hidden=64,
    mb_heads=4,

    gps_delta_scale=0.20,
    ori_delta_scale=0.10,
    intra_delta_scale=0.10,
    inter_delta_scale=0.04,
    low_delta_scale=0.04,
    mid_delta_scale=0.04,
    high_delta_scale=0.02,

    router_temperature=1.0,
    intra_knn_k=3,
    inter_comm_k=2,

    low_dim=4,
    mid_dim=4,
    high_dim=4,
    spectral_knn_k=3,

    max_pairs=4096,
):
    edge_index_apsp = batch.perm_index.long()
    batch_index = batch.batch.long()

    labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
    )

    # 多频段谱 KNN 只依赖图拓扑，每个 batch 构造一次
    low_edge_index, mid_edge_index, high_edge_index = build_multiband_spectral_knn_edges(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
        low_dim=low_dim,
        mid_dim=mid_dim,
        high_dim=high_dim,
        spectral_knn_k=spectral_knn_k,
    )

    pos0 = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)

    if use_multiband:
        refiner = MultiBandSpectralMoGMoEGPSECRRefiner(
            node_in_dim=6,
            hidden_dim=mb_hidden,
            heads=mb_heads,
            gps_delta_scale=gps_delta_scale,
            ori_delta_scale=ori_delta_scale,
            intra_delta_scale=intra_delta_scale,
            inter_delta_scale=inter_delta_scale,
            low_delta_scale=low_delta_scale,
            mid_delta_scale=mid_delta_scale,
            high_delta_scale=high_delta_scale,
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
        ).to(pos.device)

        optimizer = torch.optim.Adam(
            [
                {"params": [pos_var], "lr": lr},
                {"params": refiner.parameters(), "lr": mb_lr},
            ]
        )
    else:
        refiner = None
        optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        if use_multiband and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                low_edge_index=low_edge_index,
                mid_edge_index=mid_edge_index,
                high_edge_index=high_edge_index,
            )

            current_pos = pos_var + delta

            mb_reg_loss = delta.pow(2).sum(dim=1).mean()

            # 7 个专家的 balance loss，防止 router 只选择一个专家
            mean_w = weights.mean(dim=0)
            target = torch.ones_like(mean_w) / mean_w.numel()
            balance_reg_loss = ((mean_w - target) ** 2).sum()

        else:
            current_pos = pos_var
            mb_reg_loss = torch.tensor(0.0, device=pos_var.device)
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
            + lambda_mb_reg * mb_reg_loss
            + lambda_balance_reg * balance_reg_loss
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if use_multiband and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                low_edge_index=low_edge_index,
                mid_edge_index=mid_edge_index,
                high_edge_index=high_edge_index,
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
# 4. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="mb_spectral_mog_moe_gps_ecr_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.003)

    parser.add_argument("--lambda-stress", type=float, default=0.8)
    parser.add_argument("--lambda-intra", type=float, default=0.55)
    parser.add_argument("--lambda-inter", type=float, default=0.35)
    parser.add_argument("--lambda-pair", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.15)

    parser.add_argument("--lambda-mb-reg", type=float, default=0.01)
    parser.add_argument("--lambda-balance-reg", type=float, default=0.001)

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-multiband", action="store_true")
    parser.add_argument("--mb-lr", type=float, default=0.001)
    parser.add_argument("--mb-hidden", type=int, default=64)
    parser.add_argument("--mb-heads", type=int, default=4)

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.04)

    parser.add_argument("--low-delta-scale", type=float, default=0.04)
    parser.add_argument("--mid-delta-scale", type=float, default=0.04)
    parser.add_argument("--high-delta-scale", type=float, default=0.02)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)

    parser.add_argument("--low-dim", type=int, default=4)
    parser.add_argument("--mid-dim", type=int, default=4)
    parser.add_argument("--high-dim", type=int, default=4)
    parser.add_argument("--spectral-knn-k", type=int, default=3)

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

    if args.use_multiband:
        refined_method_name = "MB-Spectral-MoG-MoE-GPS-ECR-DeepGD"
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
        # 2. MB-Spectral-MoG-MoE-GPS-ECR refinement
        # -------------------------------------------------
        refine_start = time.time()

        refined_pos = refine_positions_multiband_spectral_mog_moe_gps_ecr(
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

            lambda_mb_reg=args.lambda_mb_reg,
            lambda_balance_reg=args.lambda_balance_reg,

            margin=args.margin,
            pair_margin=args.pair_margin,

            use_multiband=args.use_multiband,
            mb_lr=args.mb_lr,
            mb_hidden=args.mb_hidden,
            mb_heads=args.mb_heads,

            gps_delta_scale=args.gps_delta_scale,
            ori_delta_scale=args.ori_delta_scale,
            intra_delta_scale=args.intra_delta_scale,
            inter_delta_scale=args.inter_delta_scale,

            low_delta_scale=args.low_delta_scale,
            mid_delta_scale=args.mid_delta_scale,
            high_delta_scale=args.high_delta_scale,

            router_temperature=args.router_temperature,
            intra_knn_k=args.intra_knn_k,
            inter_comm_k=args.inter_comm_k,

            low_dim=args.low_dim,
            mid_dim=args.mid_dim,
            high_dim=args.high_dim,
            spectral_knn_k=args.spectral_knn_k,

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