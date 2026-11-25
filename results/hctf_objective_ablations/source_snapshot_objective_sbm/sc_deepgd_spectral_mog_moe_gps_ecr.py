# sc_deepgd_spectral_mog_moe_gps_ecr.py
# -*- coding: utf-8 -*-

"""
Spectral-MoG-MoE-GPS-ECR-DeepGD

在 MoG-MoE-GPS-ECR 基础上新增 Spectral-KNN ECR Expert。

专家组成：
1. GPS Expert：社区级全局感知
2. Original Edge ECR Expert：原始边坐标细化
3. Intra-community KNN ECR Expert：社区内 KNN 视图
4. Inter-community Skeleton ECR Expert：跨社区骨架视图
5. Spectral-KNN ECR Expert：Laplacian 谱空间近邻视图

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
# 1. Spectral-KNN View
# =========================================================

def build_spectral_knn_edges(
    edge_index,
    batch_vector,
    num_nodes,
    spectral_dim=4,
    spectral_knn_k=3,
):
    """
    构造 Spectral-KNN 边视图。

    对每一张图：
    1. 根据原始 edge_index 构建邻接矩阵 A；
    2. 计算归一化 Laplacian: L = I - D^{-1/2} A D^{-1/2}；
    3. 取前 spectral_dim 个非平凡特征向量作为谱嵌入；
    4. 在谱空间中做 KNN；
    5. 得到 spectral edge_index。

    这个视图用于补充全局拓扑结构信息。
    """
    device = batch_vector.device

    edge_index_cpu = edge_index.long().detach().cpu()
    batch_cpu = batch_vector.long().detach().cpu()

    edge_list = []

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

            # 如果没有边，跳过
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
                # 极少数数值异常时，退化为原始坐标编号的一维嵌入
                eigvecs = torch.arange(n, dtype=torch.float32).view(n, 1)

            # 跳过第一个平凡特征向量
            if eigvecs.shape[1] > 1:
                end = min(1 + spectral_dim, eigvecs.shape[1])
                U = eigvecs[:, 1:end]
            else:
                U = eigvecs[:, :1]

            if U.shape[1] == 0:
                continue

            U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

            kk = min(spectral_knn_k, n - 1)
            if kk <= 0:
                continue

            dist = torch.cdist(U, U, p=2)
            dist.fill_diagonal_(float("inf"))

            nbr = torch.topk(dist, kk, largest=False, dim=1).indices

            src_local = torch.arange(n).repeat_interleave(kk)
            dst_local = nbr.reshape(-1)

            global_nodes_long = global_nodes.long()
            src = global_nodes_long[src_local]
            dst = global_nodes_long[dst_local]

            # 双向边
            edges_1 = torch.stack([src, dst], dim=0)
            edges_2 = torch.stack([dst, src], dim=0)

            edge_list.append(edges_1)
            edge_list.append(edges_2)

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    spectral_edge_index = torch.cat(edge_list, dim=1).long().to(device)
    spectral_edge_index = torch.unique(spectral_edge_index, dim=1)

    return spectral_edge_index


# =========================================================
# 2. Spectral-MoG-MoE-GPS-ECR Refiner
# =========================================================

class SpectralMoGMoEGPSECRRefiner(MoGMoEGPSECRRefiner):
    """
    在 MoG-MoE-GPS-ECR 上新增 Spectral-KNN Expert。

    Expert 1: GPS Expert
    Expert 2: Original ECR Expert
    Expert 3: Intra-community KNN ECR Expert
    Expert 4: Inter-community Skeleton ECR Expert
    Expert 5: Spectral-KNN ECR Expert
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
        spectral_delta_scale=0.08,
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

        self.spectral_delta_scale = spectral_delta_scale

        edge_in_dim = hidden_dim * 2 + 5
        self.spectral_edge_mlp = self._make_edge_mlp(edge_in_dim, hidden_dim)

        # 原来 router 输出 4 个专家权重，现在改成 5 个
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, pos, labels, batch_vector, edge_index, spectral_edge_index):
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

        router_input = torch.cat([node_context, aux_feats], dim=-1)
        logits = self.router(router_input)
        weights = torch.softmax(logits / self.router_temperature, dim=-1)

        w_gps = weights[:, 0:1]
        w_ori = weights[:, 1:2]
        w_intra = weights[:, 2:3]
        w_inter = weights[:, 3:4]
        w_spectral = weights[:, 4:5]

        delta = (
            w_gps * delta_gps
            + w_ori * delta_ori
            + w_intra * delta_intra
            + w_inter * delta_inter
            + w_spectral * delta_spectral
        )

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta, weights


# =========================================================
# 3. Spectral refinement
# =========================================================

def refine_positions_spectral_mog_moe_gps_ecr(
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

    margin=4.8,
    pair_margin=3.0,

    use_spectral_mog=True,
    spectral_lr=0.001,
    spectral_hidden=64,
    spectral_heads=4,

    gps_delta_scale=0.20,
    ori_delta_scale=0.10,
    intra_delta_scale=0.10,
    inter_delta_scale=0.04,
    spectral_delta_scale=0.08,

    router_temperature=1.0,
    intra_knn_k=3,
    inter_comm_k=2,

    spectral_dim=4,
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

    # Spectral-KNN 只依赖图拓扑，不依赖坐标，所以每个 batch 只构造一次
    spectral_edge_index = build_spectral_knn_edges(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=pos.shape[0],
        spectral_dim=spectral_dim,
        spectral_knn_k=spectral_knn_k,
    )

    pos0 = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)

    if use_spectral_mog:
        refiner = SpectralMoGMoEGPSECRRefiner(
            node_in_dim=6,
            hidden_dim=spectral_hidden,
            heads=spectral_heads,
            gps_delta_scale=gps_delta_scale,
            ori_delta_scale=ori_delta_scale,
            intra_delta_scale=intra_delta_scale,
            inter_delta_scale=inter_delta_scale,
            spectral_delta_scale=spectral_delta_scale,
            router_temperature=router_temperature,
            intra_knn_k=intra_knn_k,
            inter_comm_k=inter_comm_k,
        ).to(pos.device)

        optimizer = torch.optim.Adam(
            [
                {"params": [pos_var], "lr": lr},
                {"params": refiner.parameters(), "lr": spectral_lr},
            ]
        )
    else:
        refiner = None
        optimizer = torch.optim.Adam([pos_var], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        if use_spectral_mog and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                spectral_edge_index=spectral_edge_index,
            )

            current_pos = pos_var + delta

            spectral_reg_loss = delta.pow(2).sum(dim=1).mean()

            # 5 个专家的 balance loss，防止 router 只选择一个专家
            mean_w = weights.mean(dim=0)
            target = torch.ones_like(mean_w) / mean_w.numel()
            balance_reg_loss = ((mean_w - target) ** 2).sum()

        else:
            current_pos = pos_var
            spectral_reg_loss = torch.tensor(0.0, device=pos_var.device)
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
            + lambda_spectral_reg * spectral_reg_loss
            + lambda_balance_reg * balance_reg_loss
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if use_spectral_mog and refiner is not None:
            delta, weights = refiner(
                pos=pos_var,
                labels=labels,
                batch_vector=batch_index,
                edge_index=batch.edge_index.long(),
                spectral_edge_index=spectral_edge_index,
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
    parser.add_argument("--output-csv", type=str, default="spectral_mog_moe_gps_ecr_metrics.csv")
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

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-spectral-mog", action="store_true")
    parser.add_argument("--spectral-lr", type=float, default=0.001)
    parser.add_argument("--spectral-hidden", type=int, default=64)
    parser.add_argument("--spectral-heads", type=int, default=4)

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.04)
    parser.add_argument("--spectral-delta-scale", type=float, default=0.08)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)

    parser.add_argument("--spectral-dim", type=int, default=4)
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

    if args.use_spectral_mog:
        refined_method_name = "Spectral-MoG-MoE-GPS-ECR-DeepGD"
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
        # 2. Spectral-MoG-MoE-GPS-ECR refinement
        # -------------------------------------------------
        refine_start = time.time()

        refined_pos = refine_positions_spectral_mog_moe_gps_ecr(
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

            margin=args.margin,
            pair_margin=args.pair_margin,

            use_spectral_mog=args.use_spectral_mog,
            spectral_lr=args.spectral_lr,
            spectral_hidden=args.spectral_hidden,
            spectral_heads=args.spectral_heads,

            gps_delta_scale=args.gps_delta_scale,
            ori_delta_scale=args.ori_delta_scale,
            intra_delta_scale=args.intra_delta_scale,
            inter_delta_scale=args.inter_delta_scale,
            spectral_delta_scale=args.spectral_delta_scale,

            router_temperature=args.router_temperature,
            intra_knn_k=args.intra_knn_k,
            inter_comm_k=args.inter_comm_k,

            spectral_dim=args.spectral_dim,
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