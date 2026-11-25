# sc_deepgd_gps_edgekan.py
# -*- coding: utf-8 -*-

"""
EdgeKAN-GPSLite-SC-DeepGD

组合模块：
1. DeepGD 生成初始布局；
2. Community-GPS Lite 通过社区 token attention 建模全局社区关系；
3. Edge-KAN Router 对节点对社区对比损失进行自适应加权；
4. 联合 stress、社区内聚、社区分离、节点级社区对比、anchor、gps_reg、kan_reg 进行测试阶段细化。

注意：
该文件依赖你已经写好的：
- sc_deepgd_gpslite.py
- sc_deepgd_edgekan.py
- evaluate_extra_metrics.py

不要删除原文件。
"""

import argparse
import time
import random

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
)

from sc_deepgd_gpslite import (
    build_batch_community_labels,
    community_refine_loss,
    community_pair_contrast_loss,
    CommunityGPSLiteRefiner,
)

from sc_deepgd_edgekan import (
    EdgeKANRouter,
    community_pair_contrast_loss_edgekan,
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
# 1. GPSLite + EdgeKAN 联合细化
# =========================================================

def refine_positions_gps_edgekan(
    pos,
    batch,
    stress_metric,
    steps=80,
    lr=0.003,

    lambda_stress=0.8,
    lambda_intra=0.5,
    lambda_inter=0.3,
    lambda_pair=0.05,
    lambda_anchor=0.15,

    lambda_gps_reg=0.01,
    kan_reg=0.01,

    margin=4.5,
    pair_margin=3.0,

    use_gps_lite=True,
    gps_lr=0.001,
    gps_hidden=64,
    gps_heads=4,
    gps_delta_scale=0.2,

    use_edge_kan=True,
    kan_lr=0.003,
    kan_hidden=32,
    kan_basis=8,

    max_pairs=4096,
):
    """
    对 DeepGD 输出坐标进行 GPSLite + EdgeKAN 联合测试阶段细化。

    总损失：
    L =
        lambda_stress * L_stress
      + lambda_intra  * L_intra
      + lambda_inter  * L_inter
      + lambda_pair   * L_pair_edgekan
      + lambda_anchor * L_anchor
      + lambda_gps_reg * L_gps_reg
      + kan_reg * L_kan_reg
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
    edgekan_router = None

    params = [
        {"params": [pos_var], "lr": lr}
    ]

    if use_gps_lite:
        gps_refiner = CommunityGPSLiteRefiner(
            node_in_dim=6,
            hidden_dim=gps_hidden,
            heads=gps_heads,
            delta_scale=gps_delta_scale,
        ).to(pos.device)

        params.append(
            {"params": gps_refiner.parameters(), "lr": gps_lr}
        )

    if use_edge_kan and lambda_pair > 0:
        edgekan_router = EdgeKANRouter(
            in_dim=5,
            hidden_dim=kan_hidden,
            num_basis=kan_basis,
        ).to(pos.device)

        params.append(
            {"params": edgekan_router.parameters(), "lr": kan_lr}
        )

    optimizer = torch.optim.Adam(params)

    for _ in range(steps):
        optimizer.zero_grad()

        # 1. GPSLite 预测坐标修正量
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

        # 2. stress 损失
        stress_loss = stress_metric(
            current_pos,
            edge_index_apsp,
            batch.apsp_attr,
            batch_index,
        )

        # 3. 社区中心级损失
        intra_loss, inter_loss = community_refine_loss(
            current_pos,
            labels,
            batch_index,
            margin=margin,
        )

        # 4. 节点级社区对比损失：优先 EdgeKAN 加权
        kan_reg_loss = torch.tensor(0.0, device=pos_var.device)

        if lambda_pair > 0:
            if use_edge_kan and edgekan_router is not None:
                pair_loss, mean_weight = community_pair_contrast_loss_edgekan(
                    pos=current_pos,
                    labels=labels,
                    batch_vector=batch_index,
                    edge_index_real=batch.edge_index.long(),
                    edgekan_router=edgekan_router,
                    pair_margin=pair_margin,
                    max_pairs=max_pairs,
                )

                # 防止 EdgeKAN 权重整体过大或过小
                kan_reg_loss = (mean_weight - 1.0).pow(2)

            else:
                pair_loss = community_pair_contrast_loss(
                    current_pos,
                    labels,
                    batch_index,
                    pair_margin=pair_margin,
                    max_pairs=max_pairs,
                )
        else:
            pair_loss = torch.tensor(0.0, device=pos_var.device)

        # 5. anchor 防止偏离 DeepGD 初始布局太远
        anchor_loss = ((current_pos - pos0) ** 2).sum(dim=1).mean()

        loss = (
            lambda_stress * stress_loss
            + lambda_intra * intra_loss
            + lambda_inter * inter_loss
            + lambda_pair * pair_loss
            + lambda_anchor * anchor_loss
            + lambda_gps_reg * gps_reg_loss
            + kan_reg * kan_reg_loss
        )

        loss.backward()
        optimizer.step()

    # 最终再计算一次 GPSLite delta
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
# 2. 评价 batch
# =========================================================

def evaluate_pos_by_graph(pos, batch, method_name, batch_id, batch_stress, infer_time):
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
# 3. 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="model_stress_only.pt")
    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-batches", type=int, default=2)
    parser.add_argument("--output-csv", type=str, default="gps_edgekan_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--refine-steps", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.003)

    parser.add_argument("--lambda-stress", type=float, default=0.8)
    parser.add_argument("--lambda-intra", type=float, default=0.5)
    parser.add_argument("--lambda-inter", type=float, default=0.3)
    parser.add_argument("--lambda-pair", type=float, default=0.05)
    parser.add_argument("--lambda-anchor", type=float, default=0.15)

    parser.add_argument("--lambda-gps-reg", type=float, default=0.01)
    parser.add_argument("--kan-reg", type=float, default=0.01)

    parser.add_argument("--margin", type=float, default=4.5)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--use-gps-lite", action="store_true")
    parser.add_argument("--gps-lr", type=float, default=0.001)
    parser.add_argument("--gps-hidden", type=int, default=64)
    parser.add_argument("--gps-heads", type=int, default=4)
    parser.add_argument("--gps-delta-scale", type=float, default=0.2)

    parser.add_argument("--use-edge-kan", action="store_true")
    parser.add_argument("--kan-lr", type=float, default=0.003)
    parser.add_argument("--kan-hidden", type=int, default=32)
    parser.add_argument("--kan-basis", type=int, default=8)

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

    if args.use_gps_lite and args.use_edge_kan:
        refined_method_name = "EdgeKAN-GPSLite-SC-DeepGD"
    elif args.use_gps_lite:
        refined_method_name = "GPSLite-SC-DeepGD"
    elif args.use_edge_kan:
        refined_method_name = "EdgeKAN-SC-DeepGD"
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
        # 2. GPSLite + EdgeKAN refinement
        # -------------------------------------------------
        refine_start = time.time()

        refined_pos = refine_positions_gps_edgekan(
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
            kan_reg=args.kan_reg,

            margin=args.margin,
            pair_margin=args.pair_margin,

            use_gps_lite=args.use_gps_lite,
            gps_lr=args.gps_lr,
            gps_hidden=args.gps_hidden,
            gps_heads=args.gps_heads,
            gps_delta_scale=args.gps_delta_scale,

            use_edge_kan=args.use_edge_kan,
            kan_lr=args.kan_lr,
            kan_hidden=args.kan_hidden,
            kan_basis=args.kan_basis,

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