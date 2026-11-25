# train_ssrb_spectral_mog_end2end.py
# -*- coding: utf-8 -*-

"""
Train SSRB-DeepGD + Spectral-MoG-MoE-GPS-ECR Refiner.

两种训练方式：
1. 只训练 SSRB-DeepGD：
   --use-ssrb

2. 训练 SSRB-DeepGD + 你之前的 Spectral-MoG-MoE-GPS-ECR Refiner：
   --use-ssrb --use-refiner
"""

import os
import time
import argparse

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
)

from sc_deepgd_moe_gps_ecr import (
    set_random_seed,
    build_batch_community_labels,
    community_refine_loss,
    community_pair_contrast_loss,
    evaluate_pos_by_graph,
    get_split_data,
)

from sc_deepgd_spectral_mog_moe_gps_ecr import (
    build_spectral_knn_edges,
    SpectralMoGMoEGPSECRRefiner,
)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def build_batch_spectral_pe(edge_index, batch_vector, num_nodes, spectral_dim=4):
    """
    为 SSRB 构造 batch 内每张图的 Laplacian spectral PE。
    输出: [num_nodes, spectral_dim]
    """
    device = batch_vector.device

    edge_index_cpu = edge_index.long().detach().cpu()
    batch_cpu = batch_vector.long().detach().cpu()

    spectral_pe = torch.zeros(
        num_nodes,
        spectral_dim,
        dtype=torch.float32,
        device=device,
    )

    graph_ids = batch_cpu.unique(sorted=True).tolist()

    with torch.no_grad():
        for gid in graph_ids:
            node_mask = batch_cpu == gid
            global_nodes = torch.where(node_mask)[0]
            n = int(global_nodes.numel())

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
                _, eigvecs = torch.linalg.eigh(L)
                if eigvecs.shape[1] > 1:
                    U = eigvecs[:, 1 : 1 + spectral_dim]
                else:
                    U = eigvecs[:, :1]
            except RuntimeError:
                U = torch.zeros((n, 1), dtype=torch.float32)

            if U.shape[1] < spectral_dim:
                pad = torch.zeros((n, spectral_dim - U.shape[1]), dtype=torch.float32)
                U = torch.cat([U, pad], dim=1)

            if U.shape[1] > spectral_dim:
                U = U[:, :spectral_dim]

            # 固定特征向量符号，减少训练震荡
            for k in range(U.shape[1]):
                col = U[:, k]
                idx = torch.argmax(torch.abs(col))
                if col[idx] < 0:
                    U[:, k] = -col

            spectral_pe[global_nodes.to(device)] = U.to(device)

    return spectral_pe


def build_deepgd_model(args, device):
    params = Generator.Params(
        num_blocks=args.num_blocks,
        block_depth=args.block_depth,
        block_width=args.block_width,
        block_output_dim=args.block_output_dim,
        edge_net_depth=args.edge_net_depth,
        edge_net_width=args.edge_net_width,
        edge_attr_dim=2,
        node_attr_dim=2,
        use_ssrb=args.use_ssrb,
        ssrb_start_block=args.ssrb_start_block,
        ssrb_every=args.ssrb_every,
        ssrb_end_block=args.ssrb_end_block,
        ssrb_spectral_dim=args.ssrb_spectral_dim,
        ssrb_hidden_dim=args.ssrb_hidden_dim,
        ssrb_dropout=args.ssrb_dropout,
    )

    return Generator(params=params).to(device)


def build_refiner(args, device):
    if not args.use_refiner:
        return None

    refiner = SpectralMoGMoEGPSECRRefiner(
        node_in_dim=6,
        hidden_dim=args.refiner_hidden,
        heads=args.refiner_heads,
        gps_delta_scale=args.gps_delta_scale,
        ori_delta_scale=args.ori_delta_scale,
        intra_delta_scale=args.intra_delta_scale,
        inter_delta_scale=args.inter_delta_scale,
        spectral_delta_scale=args.spectral_delta_scale,
        router_temperature=args.router_temperature,
        intra_knn_k=args.intra_knn_k,
        inter_comm_k=args.inter_comm_k,
    ).to(device)

    return refiner


def load_rome_dataset(args):
    print("[INFO] loading Rome dataset...")

    index = pd.read_csv(args.index_file, header=None)[0]
    dataset = RomeDataset(index=index)

    layouts = np.load(args.layouts_file, allow_pickle=True)
    datalist = list(dataset)

    for i, data in enumerate(datalist):
        data.pos = torch.tensor(layouts[i]).float()

    train_data = get_split_data(datalist, "train")
    val_data = get_split_data(datalist, "val")
    test_data = get_split_data(datalist, "test")

    print(f"[INFO] train graphs = {len(train_data)}")
    print(f"[INFO] val graphs   = {len(val_data)}")
    print(f"[INFO] test graphs  = {len(test_data)}")

    return train_data, val_data, test_data


def make_loader(data_list, batch_size, shuffle):
    return pyg.loader.DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def forward_model(batch, deepgd_model, refiner, stress_metric, args):
    batch.perm_index = batch.perm_index.long()
    batch.batch = batch.batch.long()

    num_nodes = int(batch.batch.numel())

    community_labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=num_nodes,
    )

    spectral_pe = build_batch_spectral_pe(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=num_nodes,
        spectral_dim=args.ssrb_spectral_dim,
    )

    base_pos_raw = deepgd_model(
        init_pos=generate_init_pos(batch),
        edge_index=batch.perm_index.long(),
        edge_attr=get_edge_features(batch.apsp_attr),
        batch_index=batch.batch.long(),
        community_labels=community_labels,
        spectral_pe=spectral_pe,
    )

    if args.rescale_before_refiner:
        base_pos = rescale_by_stress(
            base_pos_raw,
            batch.apsp_attr,
            batch.perm_index.long(),
            batch.batch.long(),
        )
    else:
        base_pos = base_pos_raw

    if args.use_refiner and refiner is not None:
        spectral_edge_index = build_spectral_knn_edges(
            edge_index=batch.edge_index.long(),
            batch_vector=batch.batch.long(),
            num_nodes=num_nodes,
            spectral_dim=args.refiner_spectral_dim,
            spectral_knn_k=args.spectral_knn_k,
        )

        delta, weights = refiner(
            pos=base_pos,
            labels=community_labels,
            batch_vector=batch.batch.long(),
            edge_index=batch.edge_index.long(),
            spectral_edge_index=spectral_edge_index,
        )

        refined_pos = base_pos + delta
        delta_reg_loss = delta.pow(2).sum(dim=1).mean()

        mean_w = weights.mean(dim=0)
        target = torch.ones_like(mean_w) / mean_w.numel()
        balance_loss = ((mean_w - target) ** 2).sum()
    else:
        refined_pos = base_pos
        delta_reg_loss = torch.tensor(0.0, device=base_pos.device)
        balance_loss = torch.tensor(0.0, device=base_pos.device)

    stress_loss = stress_metric(
        refined_pos,
        batch.perm_index.long(),
        batch.apsp_attr,
        batch.batch.long(),
    )

    intra_loss, inter_loss = community_refine_loss(
        refined_pos,
        community_labels,
        batch.batch.long(),
        margin=args.margin,
    )

    if args.lambda_pair > 0:
        pair_loss = community_pair_contrast_loss(
            refined_pos,
            community_labels,
            batch.batch.long(),
            pair_margin=args.pair_margin,
            max_pairs=args.max_pairs,
        )
    else:
        pair_loss = torch.tensor(0.0, device=base_pos.device)

    anchor_loss = ((refined_pos - base_pos.detach()) ** 2).sum(dim=1).mean()

    total_loss = (
        args.lambda_stress * stress_loss
        + args.lambda_intra * intra_loss
        + args.lambda_inter * inter_loss
        + args.lambda_pair * pair_loss
        + args.lambda_anchor * anchor_loss
        + args.lambda_delta_reg * delta_reg_loss
        + args.lambda_balance * balance_loss
    )

    with torch.no_grad():
        base_stress = stress_metric(
            base_pos,
            batch.perm_index.long(),
            batch.apsp_attr,
            batch.batch.long(),
        )

    loss_dict = {
        "total_loss": total_loss,
        "base_stress": base_stress.detach(),
        "refined_stress": stress_loss.detach(),
        "stress_loss": stress_loss.detach(),
        "intra_loss": intra_loss.detach(),
        "inter_loss": inter_loss.detach(),
        "pair_loss": pair_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "delta_reg_loss": delta_reg_loss.detach(),
        "balance_loss": balance_loss.detach(),
    }

    return total_loss, base_pos, refined_pos, loss_dict


def run_train_epoch(epoch, loader, deepgd_model, refiner, optimizer, stress_metric, device, args):
    deepgd_model.train()
    if refiner is not None:
        refiner.train()

    total_loss_sum = 0.0
    base_stress_sum = 0.0
    refined_stress_sum = 0.0
    count = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch_id, batch in enumerate(pbar):
        if args.max_train_batches > 0 and batch_id >= args.max_train_batches:
            break

        batch = batch.to(device)

        optimizer.zero_grad()

        total_loss, _, _, loss_dict = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        total_loss.backward()

        params = list(deepgd_model.parameters())
        if refiner is not None:
            params += list(refiner.parameters())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

        optimizer.step()

        total_loss_sum += float(loss_dict["total_loss"].item())
        base_stress_sum += float(loss_dict["base_stress"].item())
        refined_stress_sum += float(loss_dict["refined_stress"].item())
        count += 1

        pbar.set_postfix({
            "loss": total_loss_sum / max(count, 1),
            "base_s": base_stress_sum / max(count, 1),
            "ref_s": refined_stress_sum / max(count, 1),
        })

    return {
        "train_loss": total_loss_sum / max(count, 1),
        "train_base_stress": base_stress_sum / max(count, 1),
        "train_refined_stress": refined_stress_sum / max(count, 1),
    }


@torch.no_grad()
def run_validation(epoch, loader, deepgd_model, refiner, stress_metric, device, args):
    deepgd_model.eval()
    if refiner is not None:
        refiner.eval()

    total_loss_sum = 0.0
    base_stress_sum = 0.0
    refined_stress_sum = 0.0
    count = 0

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")

    for batch_id, batch in enumerate(pbar):
        if args.max_val_batches > 0 and batch_id >= args.max_val_batches:
            break

        batch = batch.to(device)

        total_loss, _, _, loss_dict = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        total_loss_sum += float(loss_dict["total_loss"].item())
        base_stress_sum += float(loss_dict["base_stress"].item())
        refined_stress_sum += float(loss_dict["refined_stress"].item())
        count += 1

        pbar.set_postfix({
            "val_loss": total_loss_sum / max(count, 1),
            "base_s": base_stress_sum / max(count, 1),
            "ref_s": refined_stress_sum / max(count, 1),
        })

    return {
        "val_loss": total_loss_sum / max(count, 1),
        "val_base_stress": base_stress_sum / max(count, 1),
        "val_refined_stress": refined_stress_sum / max(count, 1),
    }


@torch.no_grad()
def evaluate_split(loader, deepgd_model, refiner, stress_metric, device, args):
    deepgd_model.eval()
    if refiner is not None:
        refiner.eval()

    all_rows = []
    base_stress_values = []
    refined_stress_values = []

    pbar = tqdm(loader, desc="Evaluate test")

    for batch_id, batch in enumerate(pbar):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)

        start_time = time.time()

        _, base_pos, refined_pos, _ = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        infer_time = time.time() - start_time

        base_pos_eval = rescale_by_stress(
            base_pos,
            batch.apsp_attr,
            batch.perm_index.long(),
            batch.batch.long(),
        )

        refined_pos_eval = rescale_by_stress(
            refined_pos,
            batch.apsp_attr,
            batch.perm_index.long(),
            batch.batch.long(),
        )

        base_stress = stress_metric(
            base_pos_eval,
            batch.perm_index.long(),
            batch.apsp_attr,
            batch.batch.long(),
        ).item()

        refined_stress = stress_metric(
            refined_pos_eval,
            batch.perm_index.long(),
            batch.apsp_attr,
            batch.batch.long(),
        ).item()

        base_stress_values.append(base_stress)
        refined_stress_values.append(refined_stress)

        base_name = "SSRB-DeepGD" if args.use_ssrb else "DeepGD"
        final_name = (
            "SSRB-DeepGD+Spectral-MoG-Refiner"
            if args.use_refiner
            else base_name
        )

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=base_pos_eval,
                batch=batch,
                method_name=base_name,
                batch_id=batch_id,
                batch_stress=base_stress,
                infer_time=infer_time,
            )
        )

        if args.use_refiner:
            all_rows.extend(
                evaluate_pos_by_graph(
                    pos=refined_pos_eval,
                    batch=batch,
                    method_name=final_name,
                    batch_id=batch_id,
                    batch_stress=refined_stress,
                    infer_time=infer_time,
                )
            )

    df = pd.DataFrame(all_rows)

    print("\n========== Evaluation Stress Summary ==========")
    print(f"Base Mean Stress: {np.mean(base_stress_values):.6f}")

    if args.use_refiner:
        print(f"Final Mean Stress: {np.mean(refined_stress_values):.6f}")

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

    print("\n========== Mean Metrics by Method ==========")
    print(df.groupby("method")[metric_cols].mean())

    ensure_dir(os.path.dirname(args.eval_csv))
    df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] saved evaluation csv to: {args.eval_csv}")

    return df


def save_checkpoint(path, epoch, deepgd_model, refiner, optimizer, best_val, args):
    ensure_dir(os.path.dirname(path))

    ckpt = {
        "epoch": epoch,
        "deepgd_state_dict": deepgd_model.state_dict(),
        "refiner_state_dict": refiner.state_dict() if refiner is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val": best_val,
        "args": vars(args),
    }

    torch.save(ckpt, path)
    print(f"[INFO] saved checkpoint to: {path}")


def load_checkpoint(path, deepgd_model, refiner, device, strict=True):
    ckpt = torch.load(path, map_location=device)

    deepgd_model.load_state_dict(ckpt["deepgd_state_dict"], strict=strict)

    if refiner is not None and ckpt.get("refiner_state_dict") is not None:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=strict)

    print(f"[INFO] loaded checkpoint from: {path}")
    print(f"[INFO] checkpoint epoch = {ckpt.get('epoch', 'unknown')}")
    print(f"[INFO] checkpoint best_val = {ckpt.get('best_val', 'unknown')}")

    return ckpt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")

    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="ssrb_deepgd")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--eval-csv", type=str, default="results/ssrb_deepgd_test.csv")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-test-batches", type=int, default=-1)

    parser.add_argument("--deepgd-lr", type=float, default=0.0002)
    parser.add_argument("--refiner-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-blocks", type=int, default=11)
    parser.add_argument("--block-depth", type=int, default=3)
    parser.add_argument("--block-width", type=int, default=8)
    parser.add_argument("--block-output-dim", type=int, default=8)
    parser.add_argument("--edge-net-depth", type=int, default=2)
    parser.add_argument("--edge-net-width", type=int, default=16)

    parser.add_argument("--use-ssrb", action="store_true")
    parser.add_argument("--ssrb-start-block", type=int, default=3)
    parser.add_argument("--ssrb-every", type=int, default=3)
    parser.add_argument("--ssrb-end-block", type=int, default=-1)
    parser.add_argument("--ssrb-spectral-dim", type=int, default=4)
    parser.add_argument("--ssrb-hidden-dim", type=int, default=32)
    parser.add_argument("--ssrb-dropout", type=float, default=0.1)

    parser.add_argument("--use-refiner", action="store_true")
    parser.add_argument("--refiner-hidden", type=int, default=64)
    parser.add_argument("--refiner-heads", type=int, default=4)

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.04)
    parser.add_argument("--spectral-delta-scale", type=float, default=0.04)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)

    parser.add_argument("--refiner-spectral-dim", type=int, default=4)
    parser.add_argument("--spectral-knn-k", type=int, default=3)

    parser.add_argument("--lambda-stress", type=float, default=1.0)
    parser.add_argument("--lambda-intra", type=float, default=0.20)
    parser.add_argument("--lambda-inter", type=float, default=0.15)
    parser.add_argument("--lambda-pair", type=float, default=0.02)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--lambda-delta-reg", type=float, default=0.01)
    parser.add_argument("--lambda-balance", type=float, default=0.001)

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument("--rescale-before-refiner", action="store_true")

    args = parser.parse_args()

    set_random_seed(args.seed)

    device = get_device()
    print(f"[INFO] device = {device}")

    ensure_dir(args.save_dir)
    ensure_dir("results")
    ensure_dir("logs")

    train_data, val_data, test_data = load_rome_dataset(args)

    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    val_loader = make_loader(val_data, args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False)

    stress_metric = Stress()

    deepgd_model = build_deepgd_model(args, device)
    refiner = build_refiner(args, device)

    best_ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    last_ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_last.pt")

    if args.mode == "eval":
        if args.ckpt == "":
            args.ckpt = best_ckpt_path

        load_checkpoint(args.ckpt, deepgd_model, refiner, device)

        evaluate_split(
            loader=test_loader,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            device=device,
            args=args,
        )
        return

    optim_groups = [
        {"params": deepgd_model.parameters(), "lr": args.deepgd_lr},
    ]

    if refiner is not None:
        optim_groups.append(
            {"params": refiner.parameters(), "lr": args.refiner_lr}
        )

    optimizer = torch.optim.Adam(
        optim_groups,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    history = []
    best_val = float("inf")
    best_epoch = -1

    print("\n========== Start Training ==========")
    print(f"[INFO] run name = {args.run_name}")
    print(f"[INFO] use_ssrb = {args.use_ssrb}")
    print(f"[INFO] use_refiner = {args.use_refiner}")
    print(f"[INFO] best checkpoint = {best_ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        train_info = run_train_epoch(
            epoch=epoch,
            loader=train_loader,
            deepgd_model=deepgd_model,
            refiner=refiner,
            optimizer=optimizer,
            stress_metric=stress_metric,
            device=device,
            args=args,
        )

        val_info = run_validation(
            epoch=epoch,
            loader=val_loader,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            device=device,
            args=args,
        )

        val_score = val_info["val_refined_stress"]
        scheduler.step(val_score)

        row = {
            "epoch": epoch,
            **train_info,
            **val_info,
            "val_score": val_score,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_info['train_loss']:.6f} "
            f"train_base_stress={train_info['train_base_stress']:.6f} "
            f"train_refined_stress={train_info['train_refined_stress']:.6f} "
            f"val_loss={val_info['val_loss']:.6f} "
            f"val_base_stress={val_info['val_base_stress']:.6f} "
            f"val_refined_stress={val_info['val_refined_stress']:.6f} "
            f"best_val={best_val:.6f}"
        )

        save_checkpoint(
            path=last_ckpt_path,
            epoch=epoch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            optimizer=optimizer,
            best_val=best_val,
            args=args,
        )

        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch

            save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                deepgd_model=deepgd_model,
                refiner=refiner,
                optimizer=optimizer,
                best_val=best_val,
                args=args,
            )

            print(f"[INFO] new best epoch = {best_epoch}, best val = {best_val:.6f}")

        history_path = os.path.join("logs", f"{args.run_name}_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== Training Finished ==========")
    print(f"[INFO] best epoch = {best_epoch}")
    print(f"[INFO] best val = {best_val:.6f}")
    print(f"[INFO] best checkpoint = {best_ckpt_path}")

    print("\n========== Evaluate Best on Test ==========")
    load_checkpoint(best_ckpt_path, deepgd_model, refiner, device)

    evaluate_split(
        loader=test_loader,
        deepgd_model=deepgd_model,
        refiner=refiner,
        stress_metric=stress_metric,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()