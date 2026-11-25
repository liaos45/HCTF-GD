# train_spectral_mog_end2end_scratch.py
# -*- coding: utf-8 -*-

"""
End-to-End Training for Spectral-MoG-MoE-GPS-ECR-DeepGD

功能：
1. DeepGD 从头训练；
2. Spectral-MoG-MoE-GPS-ECR Refiner 从头训练；
3. DeepGD + Refiner 端到端联合优化；
4. 保存 best validation checkpoint；
5. 支持 test split 评估并导出 CSV。

注意：
这不是 test-time refinement。
这里不会优化 pos_var。
坐标由 DeepGD 和 Refiner 共同预测。
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
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def build_deepgd_model(device):
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
    return model


def build_refiner(args, device):
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


def forward_end2end(batch, deepgd_model, refiner, stress_metric, args):
    batch.perm_index = batch.perm_index.long()
    batch.batch = batch.batch.long()

    # 1. DeepGD 生成初始布局
    base_pos_raw = deepgd_model(
        init_pos=generate_init_pos(batch),
        edge_index=batch.perm_index.long(),
        edge_attr=get_edge_features(batch.apsp_attr),
        batch_index=batch.batch.long(),
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

    # 2. 社区标签
    labels = build_batch_community_labels(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=base_pos.shape[0],
    )

    # 3. Spectral-KNN 视图
    spectral_edge_index = build_spectral_knn_edges(
        edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
        num_nodes=base_pos.shape[0],
        spectral_dim=args.spectral_dim,
        spectral_knn_k=args.spectral_knn_k,
    )

    # 4. Refiner 输出 delta
    delta, weights = refiner(
        pos=base_pos,
        labels=labels,
        batch_vector=batch.batch.long(),
        edge_index=batch.edge_index.long(),
        spectral_edge_index=spectral_edge_index,
    )

    refined_pos = base_pos + delta

    # 5. Loss
    stress_loss = stress_metric(
        refined_pos,
        batch.perm_index.long(),
        batch.apsp_attr,
        batch.batch.long(),
    )

    intra_loss, inter_loss = community_refine_loss(
        refined_pos,
        labels,
        batch.batch.long(),
        margin=args.margin,
    )

    if args.lambda_pair > 0:
        pair_loss = community_pair_contrast_loss(
            refined_pos,
            labels,
            batch.batch.long(),
            pair_margin=args.pair_margin,
            max_pairs=args.max_pairs,
        )
    else:
        pair_loss = torch.tensor(0.0, device=refined_pos.device)

    anchor_loss = ((refined_pos - base_pos.detach()) ** 2).sum(dim=1).mean()
    delta_reg_loss = delta.pow(2).sum(dim=1).mean()

    mean_w = weights.mean(dim=0)
    target = torch.ones_like(mean_w) / mean_w.numel()
    balance_loss = ((mean_w - target) ** 2).sum()

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
        "stress_loss": stress_loss.detach(),
        "intra_loss": intra_loss.detach(),
        "inter_loss": inter_loss.detach(),
        "pair_loss": pair_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "delta_reg_loss": delta_reg_loss.detach(),
        "balance_loss": balance_loss.detach(),
        "base_stress": base_stress.detach(),
        "refined_stress": stress_loss.detach(),
    }

    return total_loss, base_pos, refined_pos, loss_dict


def run_train_epoch(
    epoch,
    loader,
    deepgd_model,
    refiner,
    optimizer,
    stress_metric,
    device,
    args,
):
    deepgd_model.train()
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

        total_loss, base_pos, refined_pos, loss_dict = forward_end2end(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(deepgd_model.parameters()) + list(refiner.parameters()),
                args.grad_clip,
            )

        optimizer.step()

        total_loss_sum += float(loss_dict["total_loss"].detach().item())
        base_stress_sum += float(loss_dict["base_stress"].detach().item())
        refined_stress_sum += float(loss_dict["refined_stress"].detach().item())
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
def run_validation(
    epoch,
    loader,
    deepgd_model,
    refiner,
    stress_metric,
    device,
    args,
):
    deepgd_model.eval()
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

        total_loss, base_pos, refined_pos, loss_dict = forward_end2end(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        total_loss_sum += float(loss_dict["total_loss"].detach().item())
        base_stress_sum += float(loss_dict["base_stress"].detach().item())
        refined_stress_sum += float(loss_dict["refined_stress"].detach().item())
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
def evaluate_split(
    loader,
    deepgd_model,
    refiner,
    stress_metric,
    device,
    args,
    split_name="test",
):
    deepgd_model.eval()
    refiner.eval()

    all_rows = []
    base_stress_values = []
    refined_stress_values = []

    pbar = tqdm(loader, desc=f"Evaluate {split_name}")

    for batch_id, batch in enumerate(pbar):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)
        batch.perm_index = batch.perm_index.long()
        batch.batch = batch.batch.long()

        start_time = time.time()

        total_loss, base_pos, refined_pos, loss_dict = forward_end2end(
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

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=base_pos_eval,
                batch=batch,
                method_name="DeepGD-End2End-Base",
                batch_id=batch_id,
                batch_stress=base_stress,
                infer_time=infer_time,
            )
        )

        all_rows.extend(
            evaluate_pos_by_graph(
                pos=refined_pos_eval,
                batch=batch,
                method_name="Spectral-MoG-End2End",
                batch_id=batch_id,
                batch_stress=refined_stress,
                infer_time=infer_time,
            )
        )

    df = pd.DataFrame(all_rows)

    print("\n========== Evaluation Stress Summary ==========")
    print(f"DeepGD-End2End-Base Mean Stress: {np.mean(base_stress_values):.6f}")
    print(f"Spectral-MoG-End2End Mean Stress: {np.mean(refined_stress_values):.6f}")

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

    if args.eval_csv is not None and args.eval_csv != "":
        ensure_dir(os.path.dirname(args.eval_csv))
        df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] saved evaluation csv to: {args.eval_csv}")

    return df


def save_checkpoint(path, epoch, deepgd_model, refiner, optimizer, best_val, args):
    ensure_dir(os.path.dirname(path))

    ckpt = {
        "epoch": epoch,
        "deepgd_state_dict": deepgd_model.state_dict(),
        "refiner_state_dict": refiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val": best_val,
        "args": vars(args),
    }

    torch.save(ckpt, path)
    print(f"[INFO] saved checkpoint to: {path}")


def load_checkpoint(path, deepgd_model, refiner, device, strict=True):
    ckpt = torch.load(path, map_location=device)

    deepgd_model.load_state_dict(ckpt["deepgd_state_dict"], strict=strict)
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
    parser.add_argument("--run-name", type=str, default="spectral_mog_end2end_scratch")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--eval-csv", type=str, default="results/spectral_mog_end2end_test.csv")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-test-batches", type=int, default=-1)

    parser.add_argument("--deepgd-lr", type=float, default=0.0002)
    parser.add_argument("--refiner-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

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

    parser.add_argument("--spectral-dim", type=int, default=4)
    parser.add_argument("--spectral-knn-k", type=int, default=3)

    parser.add_argument(
        "--rescale-before-refiner",
        action="store_true",
        help="训练时先对 DeepGD 输出做 stress rescale，再输入 Refiner。",
    )

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

    deepgd_model = build_deepgd_model(device)
    refiner = build_refiner(args, device)

    best_ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    last_ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_last.pt")

    if args.mode == "eval":
        if args.ckpt == "":
            args.ckpt = best_ckpt_path

        load_checkpoint(
            path=args.ckpt,
            deepgd_model=deepgd_model,
            refiner=refiner,
            device=device,
        )

        evaluate_split(
            loader=test_loader,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            device=device,
            args=args,
            split_name="test",
        )
        return

    optimizer = torch.optim.Adam(
        [
            {"params": deepgd_model.parameters(), "lr": args.deepgd_lr},
            {"params": refiner.parameters(), "lr": args.refiner_lr},
        ],
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

    print("\n========== Start End-to-End Training ==========")
    print(f"[INFO] run name = {args.run_name}")
    print(f"[INFO] epochs = {args.epochs}")
    print(f"[INFO] batch size = {args.batch_size}")
    print(f"[INFO] deepgd lr = {args.deepgd_lr}")
    print(f"[INFO] refiner lr = {args.refiner_lr}")
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
            f"train_ref_stress={train_info['train_refined_stress']:.6f} "
            f"val_loss={val_info['val_loss']:.6f} "
            f"val_ref_stress={val_info['val_refined_stress']:.6f} "
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

            print(f"[INFO] new best epoch = {best_epoch}, best val refined stress = {best_val:.6f}")

        history_df = pd.DataFrame(history)
        history_path = os.path.join("logs", f"{args.run_name}_history.csv")
        history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    print("\n========== Training Finished ==========")
    print(f"[INFO] best epoch = {best_epoch}")
    print(f"[INFO] best val refined stress = {best_val:.6f}")
    print(f"[INFO] best checkpoint = {best_ckpt_path}")

    print("\n========== Evaluate Best Checkpoint on Test ==========")

    load_checkpoint(
        path=best_ckpt_path,
        deepgd_model=deepgd_model,
        refiner=refiner,
        device=device,
    )

    evaluate_split(
        loader=test_loader,
        deepgd_model=deepgd_model,
        refiner=refiner,
        stress_metric=stress_metric,
        device=device,
        args=args,
        split_name="test",
    )


if __name__ == "__main__":
    main()