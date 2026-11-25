# HCTF-GD-GraphLocal-v1: isolated copy of the historical training program.
# The loss arithmetic is preserved; only fail-fast graph-local validation and
# the required opt-in mode are added below. Never use this file for Legacy runs.
# -*- coding: utf-8 -*-

"""
Train CAER/CSER-DeepGD for community-readable graph layout.

核心目标：
    第二个工作量主打“社区可读性”，所以本脚本不再只围绕 stress 选最优模型。
    新增：
        1. --cser-mode:
            none            原始 DeepGD
            same            CAER-lite: original edge_attr + same_community
            same_spectral   CSER-lite: original edge_attr + same_community + spectral_distance
            full            full CSER: original edge_attr + same + inter + community_center_distance + spectral_distance

        2. --select-by:
            val_loss
            val_refined_stress
            val_base_stress

        3. --lambda-base-aux:
            如果使用 Refiner，则额外约束 base_pos 的社区可读性；
            避免所有社区优化都被 Refiner 承担。

推荐主实验：
    先跑不带 Refiner 的 CAER-lite：

    python train_cser_spectral_mog_end2end.py --mode train --epochs 100 --batch-size 16 ^
    --max-train-batches -1 --max-val-batches -1 --max-test-batches -1 ^
    --use-cser --cser-mode same --select-by val_loss ^
    --lambda-stress 0.7 --lambda-intra 0.6 --lambda-inter 0.4 --lambda-pair 0.05 ^
    --deepgd-lr 0.0002 ^
    --run-name caer_same_comm_100ep ^
    --eval-csv results\\caer_same_comm_100ep_test.csv
"""

import os
import time
import argparse
import gc
import math
import hashlib
import csv
import sys
from pathlib import Path

# During development import archived project modules from the repository.
# In a locked source snapshot, the sibling deepgd package takes precedence.
if not (Path(__file__).resolve().parent / "deepgd").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import networkx as nx
from tqdm import tqdm

from deepgd.model import Generator
from deepgd.model.generator.hctf_gradient_guard import (
    project_step_to_guard_halfspaces,
)
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
from deepgd.model.generator.hctf_conflict import select_deterministic_conflict_pairs
from deepgd.metrics.readability_surrogates import (
    componentwise_readability_dominance_loss,
    readability_surrogate_losses,
)
from deepgd.paretofield import (
    FlingAdapter,
    FlingUnavailableError,
    ParetoFieldRefiner,
    ParetoTeacherArchive,
    apply_conflict_free_gradients,
    crossing_objectives_from_edge_pairs_per_graph,
    crossing_topology_barrier_per_graph,
    crossing_loss_from_edge_pairs_per_graph,
    graphwise_pareto_guard,
    graphwise_pareto_objectives,
    graphwise_regret_loss,
    hard_sampled_crossing_loss,
    hard_sampled_crossing_loss_per_graph,
    reference_directions,
    refine_pareto_set,
    sample_preferences,
    select_metric_aligned_edge_pairs,
    sample_hard_edge_pairs,
    similarity_aligned_mse,
)
from deepgd.paretofield.losses import graphwise_stress

from evaluate_extra_metrics import (
    get_device,
    get_edge_features,
    rescale_by_stress,
    generate_init_pos,
    edge_crossing_count,
    edge_length_variance,
    node_occlusion_rate,
    detect_communities,
    community_metrics,
    evaluate_single_graph,
    split_batch_to_graphs,
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


def effective_cser_mode(args):
    if not args.use_cser:
        return "none"
    return args.cser_mode


def get_cser_mode_extra_dim(mode):
    if mode == "none":
        return 0
    elif mode == "same":
        return 1
    elif mode == "same_spectral":
        return 2
    elif mode == "full":
        return 4

    raise ValueError(f"Unknown cser_mode: {mode}")


def get_cser_extra_dim(args):
    mode = effective_cser_mode(args)

    extra_dim = get_cser_mode_extra_dim(mode)

    if getattr(args, "use_caid", False) and not getattr(args, "no_caid_edge_features", False):
        extra_dim += 2

    if getattr(args, "use_care", False):
        extra_dim += 2

    if getattr(args, "use_topocser", False):
        extra_dim += 4

    return extra_dim


def get_care_edge_indices(args):
    mode = effective_cser_mode(args)
    cser_dim = get_cser_mode_extra_dim(mode)

    same_index = 2 if mode in ["same", "same_spectral", "full"] else -1

    cursor = 2 + cser_dim
    boundary_index = -1
    if getattr(args, "use_caid", False) and not getattr(args, "no_caid_edge_features", False):
        boundary_index = cursor + 1
        cursor += 2

    adj_index = cursor if getattr(args, "use_care", False) else -1
    far_index = cursor + 1 if getattr(args, "use_care", False) else -1

    return {
        "same": same_index,
        "adj": adj_index,
        "far": far_index,
        "boundary": boundary_index,
    }


def _normalize_dense_adjacency(A, eps=1e-6):
    deg = A.sum(dim=1).clamp_min(eps)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]


def _build_adaspec_matrix(
    A,
    labels,
    alpha_s=0.03,
    alpha_f=0.08,
    self_loop_scale=1.0,
    boundary_weight=0.35,
    core_weight=0.10,
    eps=1e-6,
):
    """
    Lightweight AdaSpec-style matrix for layout spectral PE.

    The ICLR 2026 AdaSpec paper learns B in Omega_D(A). Here we use a
    deterministic community-aware B so the module can be used without changing
    the DeepGD parameter shapes or invalidating old loaders.
    """
    n = int(A.size(0))
    I = torch.eye(n, dtype=A.dtype, device=A.device)

    deg = A.sum(dim=1)
    safe_deg = deg.clamp_min(1.0)

    same = labels[:, None] == labels[None, :]
    inter_edges = (A * (~same).to(dtype=A.dtype)).sum(dim=1)
    boundary = torch.where(deg > 0, inter_edges / safe_deg, torch.zeros_like(deg))
    core = 1.0 - boundary

    unique_labels, inv = torch.unique(labels, sorted=True, return_inverse=True)
    _ = unique_labels
    comm_size = torch.bincount(inv, minlength=int(inv.max().item()) + 1).to(dtype=A.dtype)
    node_comm_size = comm_size[inv].clamp_min(1.0)
    comm_size_inv = 1.0 / node_comm_size
    comm_size_inv = comm_size_inv / comm_size_inv.max().clamp_min(eps)

    degree_bias = 1.0 / (safe_deg + 1.0)
    b = (
        float(self_loop_scale) * degree_bias
        + float(boundary_weight) * boundary
        + float(core_weight) * core * comm_size_inv
    ).clamp_min(eps)

    omega_d = _normalize_dense_adjacency(A + torch.diag(b), eps=eps)

    deg_norm = deg / deg.max().clamp_min(eps)
    feature = torch.stack(
        [
            deg_norm,
            boundary,
            core,
            comm_size_inv,
        ],
        dim=1,
    )
    feature = feature / feature.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    feature_affinity = (feature @ feature.t()).clamp_min(0.0)
    same_bonus = same.to(dtype=A.dtype)
    boundary_similarity = 1.0 - torch.abs(boundary[:, None] - boundary[None, :])
    omega_f = A * (
        0.60 * feature_affinity
        + 0.30 * same_bonus
        + 0.10 * boundary_similarity.clamp_min(0.0)
    )
    if omega_f.sum() > 0:
        omega_f = _normalize_dense_adjacency(omega_f, eps=eps)

    omega = omega_d + float(alpha_s) * I + float(alpha_f) * omega_f
    omega = 0.5 * (omega + omega.t())
    return omega


def build_batch_spectral_pe(
    edge_index,
    batch_vector,
    num_nodes,
    spectral_dim=4,
    community_labels=None,
    use_adaspec=False,
    adaspec_alpha_s=0.03,
    adaspec_alpha_f=0.08,
    adaspec_self_loop_scale=1.0,
    adaspec_boundary_weight=0.35,
    adaspec_core_weight=0.10,
):
    """
    为每张图构造 Laplacian spectral PE。
    Rome 图较小，可以在线算；后续大数据集建议预计算缓存。
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

            I = torch.eye(n, dtype=torch.float32)

            if use_adaspec:
                if community_labels is None:
                    labels_g = torch.zeros(n, dtype=torch.long)
                else:
                    labels_g = (
                        community_labels.long()
                        .detach()
                        .cpu()[global_nodes]
                        .contiguous()
                    )
                A_for_spectrum = _build_adaspec_matrix(
                    A=A,
                    labels=labels_g,
                    alpha_s=adaspec_alpha_s,
                    alpha_f=adaspec_alpha_f,
                    self_loop_scale=adaspec_self_loop_scale,
                    boundary_weight=adaspec_boundary_weight,
                    core_weight=adaspec_core_weight,
                )
            else:
                A_for_spectrum = _normalize_dense_adjacency(A)

            L = I - A_for_spectrum
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                _, eigvecs = torch.linalg.eigh(L)
                if eigvecs.shape[1] > 1:
                    U = eigvecs[:, 1: 1 + spectral_dim]
                else:
                    U = eigvecs[:, :1]
            except RuntimeError:
                U = torch.zeros((n, 1), dtype=torch.float32)

            if U.shape[1] < spectral_dim:
                pad = torch.zeros((n, spectral_dim - U.shape[1]), dtype=torch.float32)
                U = torch.cat([U, pad], dim=1)

            if U.shape[1] > spectral_dim:
                U = U[:, :spectral_dim]

            for k in range(U.shape[1]):
                col = U[:, k]
                idx = torch.argmax(torch.abs(col))
                if col[idx] < 0:
                    U[:, k] = -col

            spectral_pe[global_nodes.to(device)] = U.to(device)

    return spectral_pe


def compute_community_centers(pos, labels, batch_vector):
    device = pos.device
    labels = labels.long().to(device)
    batch_vector = batch_vector.long().to(device)

    centers_per_node = torch.zeros_like(pos)

    for gid in torch.unique(batch_vector):
        graph_mask = batch_vector == gid
        if graph_mask.sum() == 0:
            continue

        node_indices = torch.where(graph_mask)[0]
        labels_g = labels[graph_mask]

        for cid in torch.unique(labels_g):
            local_mask = labels_g == cid
            if local_mask.sum() == 0:
                continue

            global_indices = node_indices[local_mask]
            center = pos[global_indices].mean(dim=0, keepdim=True)
            centers_per_node[global_indices] = center

    return centers_per_node


def normalize_feature(x, eps=1e-6):
    if x.numel() == 0:
        return x

    mean = x.mean().detach()
    std = x.std().detach()

    return (x - mean) / (std + eps)


def build_topocser_node_features(
    edge_index,
    community_labels,
    batch_vector,
    eps=1e-6,
):
    """
    LEAP-inspired lightweight topology positional encoding.

    Full LEAP computes local ECT signatures over k-hop neighborhoods. For this
    layout model we use a deterministic, dependency-free proxy that captures
    local topology useful for graph drawing:
        1. normalized log degree
        2. local clustering coefficient
        3. community boundary ratio
        4. local density
    """
    device = community_labels.device
    dtype = torch.float32
    edge_index = edge_index.long().to(device)
    labels = community_labels.long().to(device)
    batch_vector = batch_vector.long().to(device)

    num_nodes = int(labels.numel())
    features = torch.zeros((num_nodes, 4), dtype=dtype, device=device)

    for gid in batch_vector.unique(sorted=True):
        node_mask = batch_vector == gid
        global_nodes = torch.where(node_mask)[0]
        n = int(global_nodes.numel())
        if n == 0:
            continue

        local_index = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        local_index[global_nodes] = torch.arange(n, dtype=torch.long, device=device)

        src_all = edge_index[0]
        dst_all = edge_index[1]
        src = local_index[src_all]
        dst = local_index[dst_all]
        valid = (src >= 0) & (dst >= 0) & (src != dst)
        src = src[valid]
        dst = dst[valid]

        A = torch.zeros((n, n), dtype=dtype, device=device)
        if src.numel() > 0:
            A[src, dst] = 1.0
            A[dst, src] = 1.0
        A.fill_diagonal_(0.0)

        degree = A.sum(dim=1)
        max_degree = degree.max().clamp_min(1.0)
        log_degree = torch.log1p(degree) / torch.log1p(max_degree)

        possible_triangles = degree * (degree - 1.0)
        if n >= 3 and A.sum() > 0:
            closed_triplets = torch.diag(A @ A @ A)
            clustering = closed_triplets / possible_triangles.clamp_min(eps)
            clustering = torch.where(
                possible_triangles > eps,
                clustering,
                torch.zeros_like(clustering),
            )
        else:
            clustering = torch.zeros_like(degree)

        local_labels = labels[global_nodes]
        same = local_labels[:, None] == local_labels[None, :]
        cross_adj = A * (~same).to(dtype=dtype)
        boundary_ratio = cross_adj.sum(dim=1) / degree.clamp_min(eps)
        boundary_ratio = torch.where(
            degree > eps,
            boundary_ratio,
            torch.zeros_like(boundary_ratio),
        )

        local_density = degree / max(float(max(n - 1, 1)), 1.0)

        features[global_nodes] = torch.stack(
            [
                log_degree,
                clustering.clamp(0.0, 1.0),
                boundary_ratio.clamp(0.0, 1.0),
                local_density.clamp(0.0, 1.0),
            ],
            dim=1,
        )

    return features


def build_topocser_edge_attr(
    pair_index,
    graph_edge_index,
    community_labels,
    batch_vector,
    eps=1e-6,
):
    """
    Convert Topo-PE node descriptors into pairwise edge attributes.

    Returned dimensions:
        topo_dist, topo_cosine, boundary_pair, core_pair
    """
    device = community_labels.device
    pair_index = pair_index.long().to(device)
    topo = build_topocser_node_features(
        edge_index=graph_edge_index,
        community_labels=community_labels,
        batch_vector=batch_vector,
        eps=eps,
    )

    src = pair_index[0]
    dst = pair_index[1]
    src_feat = topo[src]
    dst_feat = topo[dst]

    topo_dist = torch.norm(src_feat - dst_feat, p=2, dim=1, keepdim=True)
    topo_dist = normalize_feature(topo_dist)

    topo_cos = torch.nn.functional.cosine_similarity(
        src_feat,
        dst_feat,
        dim=1,
        eps=eps,
    ).unsqueeze(-1)

    src_boundary = src_feat[:, 2:3]
    dst_boundary = dst_feat[:, 2:3]
    boundary_pair = torch.maximum(src_boundary, dst_boundary)
    core_pair = 1.0 - boundary_pair

    return torch.cat(
        [
            topo_dist,
            topo_cos,
            boundary_pair,
            core_pair,
        ],
        dim=-1,
    )


def caid_schedule(args):
    warmup_epochs = max(int(getattr(args, "caid_warmup_epochs", 0)), 0)
    if warmup_epochs <= 0:
        return 1.0

    epoch = int(getattr(args, "current_epoch", getattr(args, "epochs", warmup_epochs)))
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def _require_graph_local_index(index, batch_vector, num_nodes, name):
    """Reject malformed or inter-graph indices before an objective uses them."""
    if index.ndim != 2 or index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, count]")
    if batch_vector.numel() != num_nodes:
        raise ValueError("batch_vector must contain one graph id per node")
    if index.numel() == 0:
        return
    if int(index.min()) < 0 or int(index.max()) >= num_nodes:
        raise ValueError(f"{name} contains an invalid node index")
    if not bool((batch_vector[index[0]] == batch_vector[index[1]]).all()):
        raise ValueError(f"{name} contains a cross-graph pair")


def build_caid_context(
    apsp_attr,
    perm_index,
    edge_index,
    community_labels,
    batch_vector,
    intra_shrink=0.08,
    adj_expand=0.02,
    far_expand=0.10,
    min_ratio=0.90,
    max_ratio=1.12,
    graph_local_communities=False,
    eps=1e-6,
):
    """
    Build CAID target-distance reweighting and optional edge context.

    CAID shortens targets inside the same community and slightly expands
    targets between communities, with stronger contraction for core nodes.
    The original APSP target is still kept by the main stress objective.
    """
    device = apsp_attr.device
    dtype = apsp_attr.dtype

    perm_index = perm_index.long().to(device)
    edge_index = edge_index.long().to(device)
    labels = community_labels.long().to(device)
    batch_vector = batch_vector.long().to(device)

    num_nodes = int(labels.numel())
    if graph_local_communities:
        _require_graph_local_index(perm_index, batch_vector, num_nodes, "APSP pairs")
        _require_graph_local_index(edge_index, batch_vector, num_nodes, "original edges")

    src = perm_index[0]
    dst = perm_index[1]

    same_pair = labels[src] == labels[dst]

    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    cross_degree = torch.zeros(num_nodes, dtype=dtype, device=device)

    if edge_index.numel() > 0:
        real_src = edge_index[0]
        real_dst = edge_index[1]
        valid = (
            (real_src >= 0)
            & (real_src < num_nodes)
            & (real_dst >= 0)
            & (real_dst < num_nodes)
            & (batch_vector[real_src] == batch_vector[real_dst])
        )

        real_src = real_src[valid]
        real_dst = real_dst[valid]

        ones = torch.ones(real_src.numel(), dtype=dtype, device=device)
        degree.index_add_(0, real_src, ones)

        cross_mask = labels[real_src] != labels[real_dst]
        if cross_mask.any():
            cross_degree.index_add_(0, real_src[cross_mask], ones[cross_mask])

    boundary_score = cross_degree / (degree + eps)
    core_score = 1.0 - boundary_score.clamp(0.0, 1.0)
    boundary_pair = torch.maximum(boundary_score[src], boundary_score[dst])

    if graph_local_communities:
        # Community ids are local to each graph and normally restart from zero.
        # Pairing them with the graph id prevents another graph in the mini-batch
        # from changing this graph's community adjacency / hop distances.
        community_keys = torch.stack([batch_vector, labels], dim=1)
        unique_communities, inv_labels = torch.unique(
            community_keys,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        num_comms = int(unique_communities.size(0))
    else:
        # Legacy behaviour retained for exact reproduction of published runs.
        unique_labels, inv_labels = torch.unique(labels, sorted=True, return_inverse=True)
        num_comms = int(unique_labels.numel())

    if num_comms <= 1:
        adjacent_pair = torch.zeros_like(same_pair)
        far_pair = torch.zeros_like(same_pair)
    else:
        inf = torch.tensor(1.0e6, dtype=dtype, device=device)
        comm_hop = torch.full((num_comms, num_comms), inf, dtype=dtype, device=device)
        diag = torch.arange(num_comms, device=device)
        comm_hop[diag, diag] = 0.0

        if edge_index.numel() > 0:
            real_src = edge_index[0]
            real_dst = edge_index[1]
            valid = (
                (real_src >= 0)
                & (real_src < num_nodes)
                & (real_dst >= 0)
                & (real_dst < num_nodes)
                & (batch_vector[real_src] == batch_vector[real_dst])
            )
            real_src = real_src[valid]
            real_dst = real_dst[valid]

            comm_src = inv_labels[real_src]
            comm_dst = inv_labels[real_dst]
            cross_comm = comm_src != comm_dst
            if cross_comm.any():
                comm_hop[comm_src[cross_comm], comm_dst[cross_comm]] = 1.0
                comm_hop[comm_dst[cross_comm], comm_src[cross_comm]] = 1.0

        for k in range(num_comms):
            comm_hop = torch.minimum(
                comm_hop,
                comm_hop[:, k:k + 1] + comm_hop[k:k + 1, :],
            )

        comm_src = inv_labels[src]
        comm_dst = inv_labels[dst]
        pair_hop = comm_hop[comm_src, comm_dst]

        adjacent_pair = (~same_pair) & (pair_hop <= 1.0)
        far_pair = (~same_pair) & (pair_hop > 1.0)

    core_pair = core_score[src] * core_score[dst]
    shrink = float(intra_shrink) * same_pair.to(dtype) * (0.5 + 0.5 * core_pair)
    expand = (
        float(adj_expand) * adjacent_pair.to(dtype)
        + float(far_expand) * far_pair.to(dtype)
    )

    ratio = (1.0 - shrink + expand).clamp(float(min_ratio), float(max_ratio))
    adjusted_apsp = apsp_attr * ratio

    return {
        "adjusted_apsp": adjusted_apsp,
        "ratio": ratio.unsqueeze(-1),
        "boundary_pair": boundary_pair.unsqueeze(-1),
        "same_pair": same_pair.float().unsqueeze(-1),
        "adjacent_pair": adjacent_pair.float().unsqueeze(-1),
        "far_pair": far_pair.float().unsqueeze(-1),
    }


def cpsg_schedule(args):
    warmup_epochs = max(int(getattr(args, "cpsg_warmup_epochs", 0)), 0)
    if warmup_epochs <= 0:
        return 1.0

    epoch = int(getattr(args, "current_epoch", getattr(args, "epochs", warmup_epochs)))
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def cpsg_loss(
    pos,
    edge_index,
    community_labels,
    batch_vector,
    args,
    eps=1e-6,
):
    device = pos.device
    dtype = pos.dtype

    edge_index = edge_index.long().to(device)
    labels = community_labels.long().to(device)
    batch_vector = batch_vector.long().to(device)

    num_nodes = int(pos.size(0))
    graph_local_communities = bool(
        getattr(args, "graph_local_community_objectives", False)
    )
    if graph_local_communities:
        if labels.numel() != num_nodes:
            raise ValueError("community_labels must contain one label per node")
        _require_graph_local_index(edge_index, batch_vector, num_nodes, "original edges")
        # A community is identified by (graph id, local community id).  The
        # unqualified labels are intentionally preserved behind the opt-in
        # flag so historical HCTF checkpoints and results remain reproducible.
        community_keys = torch.stack([batch_vector, labels], dim=1)
        unique_communities, inv = torch.unique(
            community_keys,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        num_comms = int(unique_communities.size(0))
    else:
        unique_labels, inv = torch.unique(labels, sorted=True, return_inverse=True)
        num_comms = int(unique_labels.numel())

    if num_nodes == 0 or num_comms <= 1:
        zero = pos.sum() * 0.0
        return zero, zero, zero

    node_ones = torch.ones(num_nodes, dtype=dtype, device=device)
    counts = torch.zeros(num_comms, dtype=dtype, device=device)
    counts.index_add_(0, inv, node_ones)

    centers = torch.zeros(num_comms, pos.size(1), dtype=dtype, device=device)
    centers.index_add_(0, inv, pos)
    centers = centers / counts.clamp_min(1.0).unsqueeze(-1)

    own_dist = torch.norm(pos - centers[inv], p=2, dim=1)

    radii = torch.zeros(num_comms, dtype=dtype, device=device)
    radii.index_add_(0, inv, own_dist)
    radii = radii / counts.clamp_min(1.0)

    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    cross_degree = torch.zeros(num_nodes, dtype=dtype, device=device)

    comm_adj = torch.zeros(num_comms, num_comms, dtype=torch.bool, device=device)

    if edge_index.numel() > 0:
        src = edge_index[0]
        dst = edge_index[1]
        valid = (
            (src >= 0)
            & (src < num_nodes)
            & (dst >= 0)
            & (dst < num_nodes)
            & (batch_vector[src] == batch_vector[dst])
        )
        src = src[valid]
        dst = dst[valid]

        edge_ones = torch.ones(src.numel(), dtype=dtype, device=device)
        degree.index_add_(0, src, edge_ones)

        cross_mask = labels[src] != labels[dst]
        if cross_mask.any():
            cross_degree.index_add_(0, src[cross_mask], edge_ones[cross_mask])

            comm_src = inv[src[cross_mask]]
            comm_dst = inv[dst[cross_mask]]
            comm_adj[comm_src, comm_dst] = True
            comm_adj[comm_dst, comm_src] = True

    boundary_score = (cross_degree / (degree + eps)).clamp(0.0, 1.0)
    core_weight = (
        float(args.cpsg_boundary_floor)
        + (1.0 - float(args.cpsg_boundary_floor))
        * (1.0 - boundary_score).pow(float(args.cpsg_core_gamma))
    )

    if getattr(args, "use_macpsg", False):
        node_proto_loss = core_weight * own_dist.square()
        comm_proto_sum = torch.zeros(num_comms, dtype=dtype, device=device)
        comm_proto_weight = torch.zeros(num_comms, dtype=dtype, device=device)
        comm_proto_sum.index_add_(0, inv, node_proto_loss)
        comm_proto_weight.index_add_(0, inv, core_weight.clamp_min(eps))
        valid_proto_comm = comm_proto_weight > eps
        proto_loss = (
            comm_proto_sum[valid_proto_comm]
            / comm_proto_weight[valid_proto_comm].clamp_min(eps)
        ).mean()
    else:
        proto_loss = (core_weight * own_dist.square()).mean()

    if graph_local_communities:
        comm_batch = unique_communities[:, 0]
    else:
        comm_batch = torch.zeros(num_comms, dtype=torch.long, device=device)
        comm_batch.index_copy_(0, inv, batch_vector)

    center_dist_to_nodes = torch.cdist(pos, centers)
    same_graph = batch_vector.view(-1, 1) == comm_batch.view(1, -1)
    not_own = inv.view(-1, 1) != torch.arange(num_comms, device=device).view(1, -1)
    valid_other = same_graph & not_own

    inf = torch.tensor(1.0e9, dtype=dtype, device=device)
    nearest_other = center_dist_to_nodes.masked_fill(~valid_other, inf).min(dim=1).values
    valid_node = nearest_other < inf * 0.5

    if valid_node.any():
        sil_margin = torch.relu(
            float(args.cpsg_sil_margin)
            + own_dist[valid_node]
            - nearest_other[valid_node]
        ).square()

        if getattr(args, "use_macpsg", False):
            valid_inv = inv[valid_node]
            valid_weight = core_weight[valid_node].clamp_min(eps)
            comm_sil_sum = torch.zeros(num_comms, dtype=dtype, device=device)
            comm_sil_weight = torch.zeros(num_comms, dtype=dtype, device=device)
            comm_sil_sum.index_add_(0, valid_inv, valid_weight * sil_margin)
            comm_sil_weight.index_add_(0, valid_inv, valid_weight)
            valid_sil_comm = comm_sil_weight > eps
            sil_loss = (
                comm_sil_sum[valid_sil_comm]
                / comm_sil_weight[valid_sil_comm].clamp_min(eps)
            ).mean()
        else:
            sil_loss = (core_weight[valid_node] * sil_margin).mean()
    else:
        sil_loss = pos.sum() * 0.0

    center_dist = torch.cdist(centers, centers)
    same_comm_graph = comm_batch.view(-1, 1) == comm_batch.view(1, -1)
    upper = torch.triu(
        torch.ones(num_comms, num_comms, dtype=torch.bool, device=device),
        diagonal=1,
    )
    pair_mask = upper & same_comm_graph

    if pair_mask.any():
        margin = torch.where(
            comm_adj,
            torch.full_like(center_dist, float(args.cpsg_adj_margin)),
            torch.full_like(center_dist, float(args.cpsg_far_margin)),
        )
        target_dist = radii.view(-1, 1).detach() + radii.view(1, -1).detach() + margin
        sep_violation = torch.relu(target_dist[pair_mask] - center_dist[pair_mask])

        if getattr(args, "use_macpsg", False):
            sep_scale = target_dist[pair_mask].detach().clamp_min(float(args.macpsg_sep_scale_floor))
            sep_term = (sep_violation / sep_scale).square()
        else:
            sep_term = sep_violation.square()

        sep_loss = sep_term.mean()
    else:
        sep_loss = pos.sum() * 0.0

    return proto_loss, sil_loss, sep_loss


def build_cser_edge_attr(
    base_edge_attr,
    edge_index,
    pos,
    community_labels,
    batch_vector,
    spectral_pe=None,
    mode="same",
    use_spectral=True,
):
    """
    构造 CAER/CSER 增强边特征。

    mode:
        none:
            base_edge_attr

        same:
            base_edge_attr + same_community

        same_spectral:
            base_edge_attr + same_community + spectral_distance

        full:
            base_edge_attr + same_community + inter_community
            + community_center_distance + spectral_distance
    """
    if mode == "none":
        return base_edge_attr

    device = base_edge_attr.device
    edge_index = edge_index.long().to(device)
    pos = pos.to(device)
    labels = community_labels.long().to(device)
    batch_vector = batch_vector.long().to(device)

    src = edge_index[0]
    dst = edge_index[1]

    same_comm = (labels[src] == labels[dst]).float().unsqueeze(-1)

    feature_list = [same_comm]

    if mode == "same":
        return torch.cat(
            [base_edge_attr, same_comm.to(dtype=base_edge_attr.dtype)],
            dim=-1,
        )

    if mode == "same_spectral":
        if use_spectral and spectral_pe is not None:
            spectral_pe = spectral_pe.to(device=device, dtype=pos.dtype)
            spectral_dist = torch.norm(
                spectral_pe[src] - spectral_pe[dst],
                p=2,
                dim=1,
                keepdim=True,
            )
            spectral_dist = normalize_feature(spectral_dist)
        else:
            spectral_dist = torch.zeros_like(same_comm)

        return torch.cat(
            [
                base_edge_attr,
                same_comm.to(dtype=base_edge_attr.dtype),
                spectral_dist.to(dtype=base_edge_attr.dtype),
            ],
            dim=-1,
        )

    if mode == "full":
        inter_comm = 1.0 - same_comm

        centers = compute_community_centers(
            pos=pos,
            labels=labels,
            batch_vector=batch_vector,
        )

        comm_center_dist = torch.norm(
            centers[src] - centers[dst],
            p=2,
            dim=1,
            keepdim=True,
        )
        comm_center_dist = normalize_feature(comm_center_dist)

        if use_spectral and spectral_pe is not None:
            spectral_pe = spectral_pe.to(device=device, dtype=pos.dtype)
            spectral_dist = torch.norm(
                spectral_pe[src] - spectral_pe[dst],
                p=2,
                dim=1,
                keepdim=True,
            )
            spectral_dist = normalize_feature(spectral_dist)
        else:
            spectral_dist = torch.zeros_like(same_comm)

        cser_context = torch.cat(
            [
                same_comm,
                inter_comm,
                comm_center_dist,
                spectral_dist,
            ],
            dim=-1,
        )

        return torch.cat(
            [
                base_edge_attr,
                cser_context.to(dtype=base_edge_attr.dtype),
            ],
            dim=-1,
        )

    raise ValueError(f"Unknown cser mode: {mode}")


def build_deepgd_model(args, device):
    extra_dim = get_cser_extra_dim(args)
    edge_attr_dim = 2 + extra_dim
    care_indices = get_care_edge_indices(args)
    hctf_preference = tuple(
        float(value.strip())
        for value in str(
            getattr(args, "hctf_preference", "0.34,0.33,0.33")
        ).split(",")
        if value.strip()
    )

    params = Generator.Params(
        num_blocks=args.num_blocks,
        block_depth=args.block_depth,
        block_width=args.block_width,
        block_output_dim=args.block_output_dim,
        edge_net_depth=args.edge_net_depth,
        edge_net_width=args.edge_net_width,
        edge_attr_dim=edge_attr_dim,
        node_attr_dim=2,
        care_enabled=getattr(args, "use_care", False),
        care_rank=args.care_rank,
        care_scale=args.care_scale,
        care_same_index=care_indices["same"],
        care_adj_index=care_indices["adj"],
        care_far_index=care_indices["far"],
        care_boundary_index=care_indices["boundary"],
        care_use_boundary_gate=not args.no_care_boundary_gate,
        care_boundary_damp=args.care_boundary_damp,
        care_checkpoint=not args.no_care_checkpoint,
        hctf_enabled=getattr(args, "use_hctf", False),
        hctf_version=getattr(args, "hctf_version", "v1"),
        hctf_num_trees=getattr(args, "hctf_num_trees", 3),
        hctf_hidden_dim=getattr(args, "hctf_hidden_dim", 16),
        hctf_dropout=getattr(args, "hctf_dropout", 0.05),
        hctf_scale=getattr(args, "hctf_scale", 0.20),
        hctf_mix_init=getattr(args, "hctf_mix_init", 0.05),
        hctf_block_stride=getattr(args, "hctf_block_stride", 1),
        hctf_late_only=getattr(args, "hctf_late_only", False),
        hctf_stochastic_depth_prob=getattr(args, "hctf_stochastic_depth_prob", 0.0),
        hctf_stochastic_depth_seed=getattr(args, "seed", 42),
        hctf_cache_size=getattr(args, "hctf_cache_size", 4096),
        hctf_graph_gate=getattr(args, "hctf_graph_gate", False),
        hctf_adaptive_structure_gate=getattr(
            args, "hctf_adaptive_structure_gate", False
        ),
        hctf_structure_gate_init=getattr(args, "hctf_structure_gate_init", 0.05),
        hctf_zero_init=getattr(args, "hctf_zero_init", False),
        hctf_coordinate_residual=getattr(args, "hctf_coordinate_residual", False),
        hctf_coordinate_max_step=getattr(args, "hctf_coordinate_max_step", 0.08),
        hctf_conflict_head=getattr(args, "hctf_conflict_head", False),
        hctf_conflict_max_pairs=getattr(args, "hctf_conflict_max_pairs", 1024),
        hctf_conflict_max_step=getattr(args, "hctf_conflict_max_step", 0.04),
        hctf_conflict_deterministic_pairs=getattr(args, "hctf_conflict_deterministic_pairs", False),
        hctf_conflict_graph_context=getattr(args, "hctf_conflict_graph_context", False),
        hctf_graph_size_gate=getattr(args, "hctf_graph_size_gate", False),
        hctf_graph_size_midpoint=getattr(args, "hctf_graph_size_midpoint", 20.0),
        hctf_graph_size_width=getattr(args, "hctf_graph_size_width", 5.0),
        hctf_exact_risk_gate=getattr(args, "hctf_exact_risk_gate", False),
        hctf_tree_messages=not getattr(args, "hctf_disable_tree_messages", False),
        hctf_direction_conditioning=not getattr(
            args, "hctf_disable_direction_conditioning", False
        ),
        hctf_relation_conditioning=not getattr(
            args, "hctf_disable_relation_conditioning", False
        ),
        hctf_prototype_context=not getattr(
            args, "hctf_disable_prototype_context", False
        ),
        hctf_node_only_control=getattr(args, "hctf_node_only_control", False),
        hctf_learn_direction_balance=getattr(
            args, "hctf_learn_direction_balance", False
        ),
        hctf_unsaturated_degree_mean=getattr(
            args, "hctf_unsaturated_degree_mean", False
        ),
        hctf_centered_prototype_context=getattr(
            args, "hctf_centered_prototype_context", False
        ),
        hctf_prototype_graph_mean_scale=getattr(
            args, "hctf_prototype_graph_mean_scale", 1.0
        ),
        hctf_independent_prototype_gate=getattr(
            args, "hctf_independent_prototype_gate", False
        ),
        hctf_route_graph_mean_to_node=getattr(
            args, "hctf_route_graph_mean_to_node", False
        ),
        hctf_relative_tree_messages=getattr(
            args, "hctf_relative_tree_messages", False
        ),
        hctf_depth_scheduled_tree_flow=getattr(
            args, "hctf_depth_scheduled_tree_flow", False
        ),
        hctf_tree_identity_agnostic=getattr(
            args, "hctf_tree_identity_agnostic", False
        ),
        hctf_tree_coverage_relation=getattr(
            args, "hctf_tree_coverage_relation", False
        ),
        hctf_community_first_cover=getattr(
            args, "hctf_community_first_cover", False
        ),
        hctf_node_relative_prototype=getattr(
            args, "hctf_node_relative_prototype", False
        ),
        hctf_stage_residual_adapters=getattr(
            args, "hctf_stage_residual_adapters", False
        ),
        hctf_preference=hctf_preference,
        hctf_v2_tree_attention=not getattr(
            args, "hctf_v2_disable_tree_attention", False
        ),
        hctf_v2_equivariant_coordinate=not getattr(
            args, "hctf_v2_disable_equivariant_coordinate", False
        ),
    )

    model = Generator(params=params).to(device)
    if getattr(args, "use_paretofield", False):
        model.paretofield = ParetoFieldRefiner(
            node_feature_dim=4 + int(args.block_output_dim),
            hidden_dim=args.paretofield_hidden,
            num_objectives=3,
            train_steps=args.paretofield_train_steps,
            inference_steps=args.paretofield_inference_steps,
            step_size=args.paretofield_step_size,
            max_sampled_pairs=args.paretofield_max_pairs,
            adaptive_tolerance=args.paretofield_adaptive_tolerance,
            min_inference_steps=args.paretofield_min_inference_steps,
            use_soft_community=args.paretofield_soft_community,
            use_relation_consensus=args.paretofield_relation_consensus,
            relation_consensus_scale=args.paretofield_relation_consensus_scale,
            use_graph_preference_calibration=(
                args.paretofield_graph_preference_calibration
            ),
            preference_calibration_scale=(
                args.paretofield_preference_calibration_scale
            ),
            use_edge_pair_field=args.paretofield_edge_pair_field,
            edge_pair_max_pairs=args.paretofield_edge_pair_max_pairs,
            edge_pair_oversample_factor=(
                args.paretofield_edge_pair_oversample_factor
            ),
            edge_pair_temperature=args.crossing_temperature,
            edge_pair_scale=args.paretofield_edge_pair_scale,
        ).to(device)
        archive_path = str(getattr(args, "paretofield_teacher_archive", "")).strip()
        if archive_path:
            model.pareto_teacher_archive = ParetoTeacherArchive.load(archive_path)
            print(f"[INFO] loaded Pareto teacher archive: {archive_path}")
        else:
            model.pareto_teacher_archive = None
    return model


def build_paretofield_preferences(args, batch_index, training, dtype):
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    if training and getattr(args, "paretofield_sample_preferences", False):
        return sample_preferences(
            num_graphs=graph_count,
            num_objectives=3,
            alpha=args.paretofield_dirichlet_alpha,
            device=batch_index.device,
            dtype=dtype,
        )
    values = parse_float_list(getattr(args, "paretofield_preference", "0.34,0.33,0.33"))
    if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("--paretofield-preference must contain three non-negative values")
    preference = torch.tensor(values, dtype=dtype, device=batch_index.device)
    preference = preference / preference.sum().clamp_min(1e-8)
    return preference[None, :].expand(graph_count, -1).contiguous()


def get_batch_graph_names(batch, graph_count):
    names = getattr(batch, "name", None)
    if isinstance(names, str):
        names = [names]
    elif names is None:
        names = []
    else:
        names = list(names)
    return [str(names[index]) if index < len(names) else f"graph_{index}" for index in range(graph_count)]


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


def dataset_class_from_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "rome":
        return RomeDataset
    if dataset_name == "north":
        return NorthDataset
    if dataset_name == "sbm":
        return SBMDataset
    if dataset_name == "fb15k237":
        return FB15k237SampledDataset
    if dataset_name == "wn18rr":
        return WN18RRSampledDataset
    if dataset_name == "imdb_binary":
        return IMDBBinaryDataset
    if dataset_name == "proteins":
        return PROTEINSDataset
    if dataset_name == "enzymes":
        return ENZYMESDataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def graph_data_to_nx(data):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    edge_index = data.edge_index.long().detach().cpu().numpy()
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u != v:
            graph.add_edge(u, v)
    return graph


def classical_mds_layout(data):
    graph = graph_data_to_nx(data)
    n = int(data.num_nodes)
    if n <= 1:
        return np.zeros((n, 2), dtype=np.float32)

    dist = np.full((n, n), float(n), dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for src, lengths in nx.all_pairs_shortest_path_length(graph):
        for dst, length in lengths.items():
            dist[int(src), int(dst)] = float(length)

    dist2 = dist ** 2
    center = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * center @ dist2 @ center
    gram = np.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:2]
        eigvals = np.maximum(eigvals[order], 0.0)
        pos = eigvecs[:, order] * np.sqrt(eigvals)[None, :]
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        pos = rng.normal(size=(n, 2))

    if pos.shape[1] < 2:
        pos = np.pad(pos, ((0, 0), (0, 2 - pos.shape[1])))

    pos = pos[:, :2]
    pos = pos - pos.mean(axis=0, keepdims=True)
    scale = np.std(pos)
    if scale > 1e-9:
        pos = pos / scale
    return pos.astype(np.float32)


def load_or_generate_layouts(datalist, layouts_file):
    if os.path.exists(layouts_file):
        layouts = np.load(layouts_file, allow_pickle=True)
        if len(layouts) != len(datalist):
            raise RuntimeError(
                f"Layout file {layouts_file} has {len(layouts)} layouts, "
                f"but dataset has {len(datalist)} graphs."
            )
        return layouts

    print(f"[INFO] layout file not found, generating MDS layouts: {layouts_file}")
    layouts = [classical_mds_layout(data) for data in tqdm(datalist, desc="Generate layouts")]
    ensure_dir(os.path.dirname(layouts_file))
    np.save(layouts_file, np.array(layouts, dtype=object), allow_pickle=True)
    print(f"[INFO] saved generated layouts to: {layouts_file}")
    return layouts


def split_graph_dataset(datalist, args):
    if getattr(args, "split_mode", "random") == "size_ood":
        rng = np.random.default_rng(int(args.seed))
        tie_break = rng.random(len(datalist))
        indices = sorted(
            range(len(datalist)),
            key=lambda index: (int(datalist[index].num_nodes), float(tie_break[index])),
        )
        n = len(indices)
        train_end = max(1, int(0.70 * n))
        val_end = max(train_end + 1, int(0.85 * n)) if n >= 3 else train_end
        val_end = min(val_end, n - 1) if n >= 2 else n
        return (
            [datalist[index] for index in indices[:train_end]],
            [datalist[index] for index in indices[train_end:val_end]],
            [datalist[index] for index in indices[val_end:]],
        )

    if args.dataset.lower() == "rome" and len(datalist) >= 11000:
        return (
            get_split_data(datalist, "train"),
            get_split_data(datalist, "val"),
            get_split_data(datalist, "test"),
        )

    indices = np.arange(len(datalist))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    n = len(indices)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n)) if n >= 3 else 0
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if len(test_idx) == 0 and len(val_idx) > 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    return (
        [datalist[int(i)] for i in train_idx],
        [datalist[int(i)] for i in val_idx],
        [datalist[int(i)] for i in test_idx],
    )


def _stable_graph_seed(name, seed):
    payload = f"{name}|{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _partition_to_labels(partition, num_nodes):
    labels = np.zeros(int(num_nodes), dtype=np.int64)
    for community_id, nodes in enumerate(partition):
        for node in nodes:
            labels[int(node)] = int(community_id)
    return labels


def _community_labels_from_graph(data, source, seed):
    graph = nx.Graph(data.G)
    graph.add_nodes_from(range(int(data.num_nodes)))
    source = str(source).lower()

    if source == "ground_truth":
        labels = []
        for node in range(int(data.num_nodes)):
            attributes = graph.nodes[node]
            if "community" in attributes:
                labels.append(int(attributes["community"]))
            elif "block" in attributes:
                labels.append(int(attributes["block"]))
            else:
                raise ValueError(
                    "ground-truth community labels are unavailable for "
                    f"graph {getattr(data, 'name', 'unknown')}"
                )
        return np.asarray(labels, dtype=np.int64)

    if graph.number_of_edges() == 0 or int(data.num_nodes) <= 2:
        return np.zeros(int(data.num_nodes), dtype=np.int64)

    if source in {"greedy", "random"}:
        partition = nx.community.greedy_modularity_communities(graph)
    elif source == "louvain":
        partition = nx.community.louvain_communities(
            graph, weight=None, seed=int(seed)
        )
    elif source == "label_propagation":
        partition = nx.community.asyn_lpa_communities(graph, seed=int(seed))
    elif source == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as error:
            raise RuntimeError(
                "community-source=leiden requires python-igraph and leidenalg"
            ) from error
        edges = [(int(u), int(v)) for u, v in graph.edges()]
        ig_graph = ig.Graph(n=int(data.num_nodes), edges=edges, directed=False)
        partition_result = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            seed=int(seed),
        )
        partition = [set(members) for members in partition_result]
    else:
        raise ValueError(f"unknown community source: {source}")

    labels = _partition_to_labels(partition, int(data.num_nodes))
    if source == "random":
        rng = np.random.default_rng(int(seed))
        labels = labels[rng.permutation(len(labels))]
    return labels


def _perturb_community_labels(labels, fraction, seed):
    labels = np.asarray(labels, dtype=np.int64).copy()
    fraction = float(fraction)
    if fraction <= 0.0 or labels.size == 0:
        return labels
    unique = np.unique(labels)
    if unique.size <= 1:
        return labels
    rng = np.random.default_rng(int(seed))
    count = min(labels.size, max(1, int(round(fraction * labels.size))))
    selected = rng.choice(labels.size, size=count, replace=False)
    for node in selected:
        alternatives = unique[unique != labels[node]]
        labels[node] = int(rng.choice(alternatives))
    return labels


def augment_batch_community_labels(labels, batch_index, fraction, seed):
    """Deterministically perturb each graph partition for training robustness."""
    fraction = float(fraction)
    if fraction <= 0.0 or labels.numel() == 0:
        return labels
    result = labels.detach().cpu().numpy().copy()
    batch_cpu = batch_index.detach().cpu().numpy()
    for graph_id in np.unique(batch_cpu):
        nodes = np.flatnonzero(batch_cpu == graph_id)
        local = result[nodes]
        local_seed = int(seed) ^ ((int(graph_id) + 1) * 0x9E3779B1)
        result[nodes] = _perturb_community_labels(local, fraction, local_seed)
    return torch.as_tensor(result, dtype=torch.long, device=labels.device)


def attach_community_labels(datalist, args):
    source = getattr(args, "community_source", "greedy")
    noise = float(getattr(args, "community_noise", 0.0))
    base_seed = int(getattr(args, "community_seed", getattr(args, "seed", 42)))
    for graph_index, data in enumerate(datalist):
        name = getattr(data, "name", f"graph_{graph_index}")
        graph_seed = _stable_graph_seed(name, base_seed)
        labels = _community_labels_from_graph(data, source, graph_seed)
        labels = _perturb_community_labels(labels, noise, graph_seed ^ 0x9E3779B9)
        data.community_label = torch.as_tensor(labels, dtype=torch.long)


def load_graph_dataset(args):
    dataset_name = args.dataset.lower()
    print(f"[INFO] loading {dataset_name} dataset...")

    index = None
    if args.index_file and os.path.exists(args.index_file):
        index = pd.read_csv(args.index_file, header=None)[0].dropna().astype(str).tolist()

    dataset_cls = dataset_class_from_name(dataset_name)
    dataset = dataset_cls(index=index)

    datalist = list(dataset)
    if args.index_file and not os.path.exists(args.index_file):
        ensure_dir(os.path.dirname(args.index_file))
        names = [str(data.name) for data in datalist]
        pd.Series(names).to_csv(args.index_file, index=False, header=False)
        print(f"[INFO] saved dataset index to: {args.index_file}")

    layouts = load_or_generate_layouts(datalist, args.layouts_file)

    for i, data in enumerate(datalist):
        data.pos = torch.tensor(layouts[i]).float()

    attach_community_labels(datalist, args)

    attach_resistance_distance_targets(datalist, args)
    attach_cosine_stress_targets(datalist, args)

    train_data, val_data, test_data = split_graph_dataset(datalist, args)

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


def maybe_cleanup_memory(args, batch_id=None, force=False):
    interval = int(getattr(args, "memory_cleanup_every", 0))
    if not force and (interval <= 0 or batch_id is None or (batch_id + 1) % interval != 0):
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_layout_objective(
    pos,
    batch,
    community_labels,
    stress_metric,
    args,
):
    stress_loss = stress_metric(
        pos,
        batch.perm_index.long(),
        batch.apsp_attr,
        batch.batch.long(),
    )
    stress_optimization_loss = stress_loss
    balance_power = float(getattr(args, "stress_graph_balance_power", 0.0))
    if balance_power > 0.0:
        # Training-only reweighting: the reported Stress above remains the
        # original per-graph sum. A geometric-mean reference preserves its
        # scale when all graphs have the same number of APSP pairs.
        pair_index = batch.perm_index.long()
        graph_index = batch.batch.long()[pair_index[0]]
        pair_dist = (pos[pair_index[1]] - pos[pair_index[0]]).norm(dim=1)
        pair_error = ((pair_dist - batch.apsp_attr).abs() / batch.apsp_attr).square()
        num_graphs = int(batch.num_graphs)
        graph_sums = pyg.utils.scatter(
            pair_error, graph_index, dim_size=num_graphs, reduce="sum"
        )
        graph_counts = pyg.utils.scatter(
            torch.ones_like(pair_error), graph_index, dim_size=num_graphs,
            reduce="sum",
        )
        positive = graph_counts > 0
        if bool(positive.any()):
            reference_count = graph_counts[positive].log().mean().exp().detach()
            graph_weight = (reference_count / graph_counts.clamp_min(1)).pow(
                balance_power
            )
            stress_optimization_loss = (graph_sums * graph_weight).mean()

    if getattr(args, "use_rd_stress", False):
        rd_stress_loss = resistance_distance_stress_loss(pos, batch, args)
    else:
        rd_stress_loss = torch.tensor(0.0, device=pos.device)

    if getattr(args, "use_cosine_stress", False):
        cosine_stress_loss = cosine_stress_loss_fn(pos, batch, args)
    else:
        cosine_stress_loss = torch.tensor(0.0, device=pos.device)

    intra_loss, inter_loss = community_refine_loss(
        pos,
        community_labels,
        batch.batch.long(),
        margin=args.margin,
    )

    if args.lambda_pair > 0:
        pair_loss = community_pair_contrast_loss(
            pos,
            community_labels,
            batch.batch.long(),
            pair_margin=args.pair_margin,
            max_pairs=args.max_pairs,
        )
    else:
        pair_loss = torch.tensor(0.0, device=pos.device)

    objective = (
        args.lambda_stress * stress_optimization_loss
        + args.lambda_rd_stress * rd_stress_loss
        + args.lambda_cosine_stress * cosine_stress_loss
        + args.lambda_intra * intra_loss
        + args.lambda_inter * inter_loss
        + args.lambda_pair * pair_loss
    )

    return {
        "objective": objective,
        "stress": stress_loss,
        "stress_optimization": stress_optimization_loss,
        "rd_stress": rd_stress_loss,
        "cosine_stress": cosine_stress_loss,
        "intra": intra_loss,
        "inter": inter_loss,
        "pair": pair_loss,
    }


def hctf_graphwise_component_objectives(
    pos,
    batch,
    community_labels,
    crossing_count_per_graph,
    crossing_norm_per_graph,
):
    """Differentiable counterparts of the ten reported layout metrics.

    Crossing count and normalized crossing use separate reductions over the
    same fixed edge pairs. All ten outputs are lower-is-better.
    """
    batch_index = batch.batch.long()
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    stress = graphwise_stress(
        pos,
        batch.perm_index.long(),
        batch.apsp_attr,
        batch_index,
    )

    edge_index = batch.edge_index.long()
    source, target = edge_index
    unique_mask = (source != target) & (source < target)
    source = source[unique_mask]
    target = target[unique_mask]
    edge_lengths = torch.norm(pos[source] - pos[target], dim=1)
    edge_graph = batch_index[source]

    edge_variance = []
    edge_cv = []
    occlusion = []
    compactness = []
    separation = []
    sep_comp_ratio = []
    silhouette = []
    for graph_id in range(graph_count):
        graph_mask = batch_index == graph_id
        points = pos[graph_mask]
        labels = community_labels[graph_mask]
        selected_lengths = edge_lengths[edge_graph == graph_id]
        if selected_lengths.numel() == 0:
            zero = pos.sum() * 0.0
            edge_variance.append(zero)
            edge_cv.append(zero)
            length_scale = points.detach().std().clamp_min(1e-8)
        else:
            edge_variance.append(selected_lengths.var(unbiased=False))
            edge_cv.append(
                selected_lengths.std(unbiased=False)
                / selected_lengths.mean().clamp_min(1e-8)
            )
            length_scale = selected_lengths.detach().median().clamp_min(1e-8)

        pair_distances = torch.pdist(points)
        if pair_distances.numel() == 0:
            occlusion.append(pos.sum() * 0.0)
        else:
            threshold = 0.03 * length_scale
            softness = (0.20 * threshold).clamp_min(1e-6)
            occlusion.append(
                torch.sigmoid((threshold - pair_distances) / softness).mean()
            )

        unique_labels = labels.unique(sorted=True)
        if unique_labels.numel() <= 1:
            zero = pos.sum() * 0.0
            compactness.append(zero)
            separation.append(zero)
            sep_comp_ratio.append(zero)
            silhouette.append(zero)
            continue

        centers = []
        compact_values = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            center = cluster_points.mean(dim=0)
            centers.append(center)
            compact_values.append(torch.norm(cluster_points - center, dim=1).mean())
        centers = torch.stack(centers)
        compact = torch.stack(compact_values).mean()
        sep = torch.pdist(centers).mean()
        compactness.append(compact)
        separation.append(sep)
        sep_comp_ratio.append(sep / compact.clamp_min(1e-8))

        distances = torch.cdist(points, points)
        node_silhouette = []
        for node_id in range(points.size(0)):
            own = labels == labels[node_id]
            own[node_id] = False
            if not bool(own.any()):
                node_silhouette.append(pos.sum() * 0.0)
                continue
            intra = distances[node_id, own].mean()
            inter_values = []
            for other_label in unique_labels:
                if bool(other_label == labels[node_id]):
                    continue
                inter_values.append(
                    distances[node_id, labels == other_label].mean()
                )
            nearest_other = torch.stack(inter_values).min()
            node_silhouette.append(
                (nearest_other - intra)
                / torch.maximum(nearest_other, intra).clamp_min(1e-8)
            )
        silhouette.append(torch.stack(node_silhouette).mean())

    return torch.stack(
        [
            stress,
            crossing_count_per_graph,
            crossing_norm_per_graph,
            torch.stack(edge_variance),
            torch.stack(edge_cv),
            torch.stack(occlusion),
            torch.stack(compactness),
            -torch.stack(separation),
            -torch.stack(sep_comp_ratio),
            -torch.stack(silhouette),
        ],
        dim=1,
    )


def hctf_componentwise_dominance_loss(candidate, teacher, args):
    floors = candidate.new_tensor(
        [0.05, 1.0, 0.01, 0.01, 0.05, 0.001, 0.05, 0.05, 0.05, 0.05]
    )
    scale = teacher.detach().abs().clamp_min(floors)
    regret = (candidate - teacher.detach()) / scale
    temperature = max(float(args.hctf_dominance_temperature), 1e-4)
    target_spec = str(getattr(args, "hctf_component_targets", "")).strip()
    if target_spec:
        target_values = [float(value.strip()) for value in target_spec.split(",")]
        if len(target_values) == 9 and regret.size(1) == 10:
            target_values = target_values[:2] + [target_values[1]] + target_values[2:]
        if len(target_values) != regret.size(1):
            raise ValueError(
                "--hctf-component-targets must contain ten values (or nine legacy values)"
            )
        targets = regret.new_tensor(target_values).clamp_min(0.0)
    else:
        targets = regret.new_full(
            (regret.size(1),),
            max(float(args.hctf_target_improvement), 0.0),
        )
    weight_spec = str(getattr(args, "hctf_component_weights", "")).strip()
    if weight_spec:
        weight_values = [float(value.strip()) for value in weight_spec.split(",")]
        if len(weight_values) == 9 and regret.size(1) == 10:
            weight_values = weight_values[:2] + [weight_values[1]] + weight_values[2:]
        if len(weight_values) != regret.size(1):
            raise ValueError(
                "--hctf-component-weights must contain ten values (or nine legacy values)"
            )
        weights = regret.new_tensor(weight_values).clamp_min(0.0)
    else:
        weights = regret.new_ones(regret.size(1))
    weights = weights / weights.mean().clamp_min(1e-8)
    adjusted_regret = regret + targets.view(1, -1)
    smooth_max = temperature * (
        torch.logsumexp(adjusted_regret / temperature, dim=1)
        - math.log(regret.size(1))
    )
    component_penalty = (
        weights.view(1, -1) * torch.relu(adjusted_regret).square()
    ).mean(dim=1)
    excess = torch.relu(adjusted_regret).square()
    tail_fraction = min(
        max(float(getattr(args, "hctf_cvar_fraction", 0.20)), 0.0),
        1.0,
    )
    if tail_fraction > 0.0 and excess.size(0) > 0:
        tail_count = max(int(math.ceil(excess.size(0) * tail_fraction)), 1)
        tail_by_component = torch.topk(
            excess,
            k=tail_count,
            dim=0,
            largest=True,
        ).values.mean(dim=0)
        tail_penalty = (weights * tail_by_component).mean()
    else:
        tail_penalty = excess.sum() * 0.0
    loss = (smooth_max + component_penalty).mean()
    loss = loss + float(getattr(args, "hctf_cvar_weight", 0.0)) * tail_penalty
    return loss, regret.max(dim=1).values.mean(), regret, tail_penalty


HCTF_EXACT_METRICS = (
    "batch_stress",
    "edge_crossing",
    "edge_crossing_norm",
    "edge_length_variance",
    "edge_length_cv",
    "node_occlusion_rate",
    "community_compactness",
    "community_separation",
    "community_sep_comp_ratio",
    "community_silhouette",
)
HCTF_EXACT_HIGHER_IS_BETTER = {
    "community_separation",
    "community_sep_comp_ratio",
    "community_silhouette",
}
HCTF_EXACT_FLOORS = {
    "batch_stress": 0.05,
    "edge_crossing": 1.0,
    "edge_crossing_norm": 0.001,
    "edge_length_variance": 0.01,
    "edge_length_cv": 0.05,
    "node_occlusion_rate": 0.001,
    "community_compactness": 0.05,
    "community_separation": 0.05,
    "community_sep_comp_ratio": 0.05,
    "community_silhouette": 0.05,
}


@torch.no_grad()
def hctf_exact_batch_metric_sums(pos, batch):
    stress = graphwise_stress(
        pos,
        batch.perm_index.long(),
        batch.apsp_attr,
        batch.batch.long(),
    ).detach().cpu().tolist()
    rows = evaluate_pos_by_graph(
        pos=pos,
        batch=batch,
        method_name="hctf-validation",
        batch_id=0,
        batch_stress=0.0,
        infer_time=0.0,
    )
    sums = {metric: 0.0 for metric in HCTF_EXACT_METRICS}
    for graph_id, row in enumerate(rows):
        row["batch_stress"] = float(stress[graph_id])
        for metric in HCTF_EXACT_METRICS:
            sums[metric] += float(row[metric])
    return sums, len(rows)


def hctf_exact_validation_summary(candidate_sums, teacher_sums, graph_count):
    if graph_count <= 0:
        return {"val_hctf_exact_score": float("inf")}
    regrets = []
    summary = {}
    for metric in HCTF_EXACT_METRICS:
        candidate = float(candidate_sums[metric]) / graph_count
        teacher = float(teacher_sums[metric]) / graph_count
        scale = max(abs(teacher), HCTF_EXACT_FLOORS[metric])
        if metric in HCTF_EXACT_HIGHER_IS_BETTER:
            regret = (teacher - candidate) / scale
        else:
            regret = (candidate - teacher) / scale
        regrets.append(regret)
        summary[f"val_hctf_exact_{metric}"] = candidate
        summary[f"val_hctf_exact_{metric}_teacher"] = teacher
        summary[f"val_hctf_exact_{metric}_regret"] = regret
    max_regret = max(regrets)
    mean_regret = sum(regrets) / len(regrets)
    summary["val_hctf_exact_max_regret"] = max_regret
    summary["val_hctf_exact_mean_regret"] = mean_regret
    summary["val_hctf_exact_improved_metrics"] = sum(value < 0.0 for value in regrets)
    summary["val_hctf_exact_score"] = max_regret + 0.10 * mean_regret
    return summary


def parse_hctf_exact_gate_components(spec):
    values = tuple(int(value.strip()) for value in str(spec).split(",") if value.strip())
    if not values:
        raise ValueError("--hctf-exact-gate-components cannot be empty")
    if len(set(values)) != len(values) or any(
        value < 0 or value >= len(HCTF_EXACT_METRICS) for value in values
    ):
        raise ValueError(
            "--hctf-exact-gate-components must be unique metric indices in [0, 9]"
        )
    return values


@torch.no_grad()
def hctf_exact_selected_metric_tensor(pos, batch, component_indices):
    component_indices = tuple(component_indices)
    metric_names = [HCTF_EXACT_METRICS[index] for index in component_indices]
    requested = set(metric_names)
    graphs = split_batch_to_graphs(
        pos=pos,
        batch_edge_index=batch.edge_index.long(),
        batch_vector=batch.batch.long(),
    )
    stress_values = None
    if "batch_stress" in requested:
        stress_values = graphwise_stress(
            pos,
            batch.perm_index.long(),
            batch.apsp_attr,
            batch.batch.long(),
        ).detach().cpu().tolist()

    rows = []
    for graph_id, (local_pos, local_edge_index) in enumerate(graphs):
        values = {}
        if stress_values is not None:
            values["batch_stress"] = float(stress_values[graph_id])
        if {"edge_crossing", "edge_crossing_norm"} & requested:
            crossing, crossing_norm = edge_crossing_count(local_pos, local_edge_index)
            values["edge_crossing"] = float(crossing)
            values["edge_crossing_norm"] = float(crossing_norm)
        if {"edge_length_variance", "edge_length_cv"} & requested:
            variance, cv = edge_length_variance(local_pos, local_edge_index)
            values["edge_length_variance"] = float(variance)
            values["edge_length_cv"] = float(cv)
        if "node_occlusion_rate" in requested:
            occlusion, _ = node_occlusion_rate(
                local_pos,
                edge_index=local_edge_index,
            )
            values["node_occlusion_rate"] = float(occlusion)
        if {
            "community_compactness",
            "community_separation",
            "community_sep_comp_ratio",
            "community_silhouette",
        } & requested:
            labels = detect_communities(local_edge_index, int(local_pos.size(0)))
            values.update(community_metrics(local_pos, labels))
        rows.append([values[name] for name in metric_names])

    if not rows:
        return pos.new_zeros((0, len(component_indices)))
    return pos.new_tensor(rows)


def hctf_exact_risk_targets(
    current_risk,
    candidate_metrics,
    teacher_metrics,
    component_indices,
    tolerance,
    temperature,
):
    # Exact graph metrics are not defined for every graph.  In particular,
    # community silhouette is NaN when a partition has fewer than two usable
    # communities.  Treat a metric with a non-finite teacher value as
    # unavailable for that graph, while a non-finite candidate value against a
    # finite teacher is an unsafe regression.  This keeps the finite-metric
    # calculation unchanged and prevents an undefined auxiliary metric from
    # poisoning the BCE gate target.
    candidate_finite = torch.isfinite(candidate_metrics)
    teacher_finite = torch.isfinite(teacher_metrics)
    finite_pair = candidate_finite & teacher_finite
    candidate_metrics = torch.where(
        candidate_finite, candidate_metrics, torch.zeros_like(candidate_metrics)
    )
    teacher_metrics = torch.where(
        teacher_finite, teacher_metrics, torch.zeros_like(teacher_metrics)
    )
    floors = candidate_metrics.new_tensor(
        [HCTF_EXACT_FLOORS[HCTF_EXACT_METRICS[index]] for index in component_indices]
    )
    scale = teacher_metrics.abs().clamp_min(floors)
    regret = (candidate_metrics - teacher_metrics) / scale
    for local_index, component_index in enumerate(component_indices):
        if HCTF_EXACT_METRICS[component_index] in HCTF_EXACT_HIGHER_IS_BETTER:
            regret[:, local_index] = -regret[:, local_index]

    tolerance = max(float(tolerance), 0.0)
    temperature = max(float(temperature), 1e-6)
    finite_penalty = max(tolerance + 50.0 * temperature, 1.0)
    regret = torch.nan_to_num(
        regret,
        nan=finite_penalty,
        posinf=finite_penalty,
        neginf=-finite_penalty,
    )
    regret = torch.where(
        teacher_finite & ~candidate_finite,
        torch.full_like(regret, finite_penalty),
        regret,
    )
    regret = torch.where(
        finite_pair,
        regret,
        torch.where(
            teacher_finite,
            torch.full_like(regret, finite_penalty),
            torch.full_like(regret, -finite_penalty),
        ),
    )

    max_regret = regret.max(dim=1).values
    has_comparable_metric = teacher_finite.any(dim=1)
    max_regret = torch.where(
        has_comparable_metric, max_regret, torch.zeros_like(max_regret)
    )
    safe = max_regret <= tolerance
    excess = torch.relu(max_regret - tolerance)
    finite_risk = torch.nan_to_num(
        current_risk, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    reduced = finite_risk.detach() * torch.exp(-excess / temperature)
    target = torch.where(safe, torch.ones_like(reduced), reduced).clamp(0.0, 1.0)
    target = torch.where(has_comparable_metric, target, finite_risk.detach())
    return target, max_regret, safe


def hctf_exact_risk_gate_loss(
    current_risk,
    candidate_pos,
    teacher_pos,
    batch,
    args,
    component_spec=None,
    tolerance=None,
    temperature=None,
):
    component_indices = parse_hctf_exact_gate_components(
        args.hctf_exact_gate_components if component_spec is None else component_spec
    )
    candidate_metrics = hctf_exact_selected_metric_tensor(
        candidate_pos,
        batch,
        component_indices,
    )
    teacher_metrics = hctf_exact_selected_metric_tensor(
        teacher_pos,
        batch,
        component_indices,
    )
    target, max_regret, safe = hctf_exact_risk_targets(
        current_risk=current_risk,
        candidate_metrics=candidate_metrics,
        teacher_metrics=teacher_metrics,
        component_indices=component_indices,
        tolerance=(
            args.hctf_exact_gate_tolerance if tolerance is None else tolerance
        ),
        temperature=(
            args.hctf_exact_gate_temperature if temperature is None else temperature
        ),
    )
    finite_risk = torch.nan_to_num(
        current_risk, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(1e-6, 1.0 - 1e-6)
    per_graph = torch.nn.functional.binary_cross_entropy(
        finite_risk,
        target.detach(),
        reduction="none",
    )
    if bool(safe.any()) and bool((~safe).any()):
        loss = 0.5 * per_graph[safe].mean() + 0.5 * per_graph[~safe].mean()
    else:
        loss = per_graph.mean()
    return loss, target.mean(), (~safe).float().mean(), max_regret.mean()


def epoch_linear_schedule(args, start_epoch_attr, warmup_epoch_attr):
    start_epoch = int(getattr(args, start_epoch_attr, 1))
    warmup_epochs = max(int(getattr(args, warmup_epoch_attr, 0)), 0)
    epoch = int(getattr(args, "current_epoch", getattr(args, "epochs", start_epoch)))

    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch - start_epoch + 1) / float(warmup_epochs)))


def pmds_anchor_loss(
    pos, init_pos, huber_delta=0.0, node_mask=None,
    batch_index=None, center_per_graph=False,
):
    if init_pos is None or init_pos.numel() == 0:
        return pos.sum() * 0.0
    displacement = pos - init_pos.detach()
    if center_per_graph:
        if batch_index is None:
            raise ValueError("batch_index is required for graph-centered PMDS anchor")
        graph_mean = pyg.utils.scatter(
            displacement, batch_index.long(), dim=0, reduce="mean"
        )
        displacement = displacement - graph_mean[batch_index.long()]
    if huber_delta <= 0.0:
        per_node = displacement.pow(2).sum(dim=1)
        return (per_node * node_mask.to(per_node.dtype)).mean() if node_mask is not None else per_node.mean()
    # Match the original squared distance below delta, but bound the
    # coordinate gradient from rare, extremely distant PMDS initializations.
    radius = torch.linalg.vector_norm(displacement, dim=1)
    robust = torch.where(
        radius <= huber_delta,
        radius.square(),
        2.0 * huber_delta * radius - huber_delta ** 2,
    )
    return (robust * node_mask.to(robust.dtype)).mean() if node_mask is not None else robust.mean()


def pmds_anchor_training_options(args, batch):
    """Use the optimizer step, not model.training: frozen HCTF backbone is eval()."""
    if not bool(getattr(args, "_is_training_step", False)):
        return None, False
    min_nodes = int(getattr(args, "pmds_anchor_min_nodes", 0))
    node_mask = None
    if min_nodes > 0:
        graph_size = torch.bincount(
            batch.batch.long(), minlength=int(batch.num_graphs)
        )
        node_mask = graph_size[batch.batch.long()] >= min_nodes
    return node_mask, bool(getattr(args, "pmds_anchor_center_per_graph", False))


def centered_rescale_by_stress(pos, reference_pos, apsp, edge_index, batch_index):
    """Stress-rescale relative coordinates without amplifying graph translation.

    The pairwise geometry uses the existing rescale rule; only the arbitrary
    coordinate origin is fixed to the corresponding reference graph centroid.
    """
    batch_index = batch_index.long()
    pos_center = pyg.utils.scatter(pos, batch_index, dim=0, reduce="mean")
    reference_center = pyg.utils.scatter(
        reference_pos.detach(), batch_index, dim=0, reduce="mean"
    )
    centered = pos - pos_center[batch_index]
    scaled = rescale_by_stress(centered, apsp, edge_index, batch_index)
    return scaled + reference_center[batch_index]


def compute_resistance_distance_targets(data, normalize="mean", eps=1e-9):
    """
    Compute resistance-distance targets aligned with data.perm_index.

    The target is scaled per graph to the APSP target scale by default, so the
    resulting loss can be mixed with the existing DeepGD shortest-path stress.
    """
    n = int(data.num_nodes)
    if n <= 1 or not hasattr(data, "perm_index") or not hasattr(data, "apsp_attr"):
        return torch.empty(0, dtype=torch.float32)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return data.apsp_attr.detach().clone().float()

    edge_index_cpu = edge_index.detach().cpu().long()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edges = edge_index_cpu.t().tolist()
    graph.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))

    rd_matrix = np.full((n, n), np.nan, dtype=np.float64)
    for component in nx.connected_components(graph):
        nodes = sorted(int(v) for v in component)
        if len(nodes) == 1:
            rd_matrix[nodes[0], nodes[0]] = 0.0
            continue

        sub = graph.subgraph(nodes)
        lap = nx.laplacian_matrix(sub, nodelist=nodes).toarray().astype(np.float64)
        lap_pinv = np.linalg.pinv(lap, hermitian=True)
        diag = np.diag(lap_pinv)
        rd_sub = diag[:, None] + diag[None, :] - 2.0 * lap_pinv
        rd_sub = np.maximum(rd_sub, 0.0)

        idx = np.ix_(nodes, nodes)
        rd_matrix[idx] = rd_sub

    finite = np.isfinite(rd_matrix)
    finite_values = rd_matrix[finite]
    fallback = float(np.max(finite_values)) if finite_values.size > 0 else 1.0
    rd_matrix[~finite] = fallback

    perm_index = data.perm_index.detach().cpu().long()
    src = perm_index[0].numpy()
    dst = perm_index[1].numpy()
    rd_targets = rd_matrix[src, dst].astype(np.float64)
    rd_targets = np.maximum(rd_targets, eps)

    apsp = data.apsp_attr.detach().cpu().numpy().astype(np.float64)
    valid = np.isfinite(rd_targets) & (rd_targets > eps) & np.isfinite(apsp) & (apsp > eps)
    normalize = str(normalize).lower()
    if valid.any() and normalize != "none":
        if normalize == "median":
            rd_center = float(np.median(rd_targets[valid]))
            apsp_center = float(np.median(apsp[valid]))
        else:
            rd_center = float(np.mean(rd_targets[valid]))
            apsp_center = float(np.mean(apsp[valid]))
        if rd_center > eps:
            rd_targets = rd_targets * (apsp_center / rd_center)

    return torch.tensor(rd_targets, dtype=torch.float32)


def attach_resistance_distance_targets(data_list, args):
    if not getattr(args, "use_rd_stress", False):
        return data_list

    print(
        "[INFO] precomputing resistance-distance targets: "
        f"normalize={args.rd_normalize}"
    )
    for data in tqdm(data_list, desc="RD targets"):
        data.rd_attr = compute_resistance_distance_targets(
            data,
            normalize=args.rd_normalize,
        )
    return data_list


def resistance_distance_stress_loss(pos, batch, args, eps=1e-9):
    if not getattr(args, "use_rd_stress", False) or not hasattr(batch, "rd_attr"):
        return pos.sum() * 0.0

    rd_attr = batch.rd_attr.to(device=pos.device, dtype=pos.dtype).view(-1)
    if rd_attr.numel() == 0:
        return pos.sum() * 0.0

    perm_index = batch.perm_index.long().to(pos.device)
    src_all = perm_index[0]
    dst_all = perm_index[1]
    graph_all = batch.batch.long().to(pos.device)[src_all]
    total_counts = torch.bincount(
        graph_all,
        minlength=int(batch.batch.max().item()) + 1,
    ).to(dtype=pos.dtype)

    max_pairs = int(getattr(args, "rd_max_pairs", 0))
    if max_pairs > 0 and rd_attr.numel() > max_pairs:
        idx = torch.randperm(rd_attr.numel(), device=pos.device)[:max_pairs]
        src = src_all[idx]
        dst = dst_all[idx]
        targets = rd_attr[idx]
        graph_ids = graph_all[idx]
    else:
        src = src_all
        dst = dst_all
        targets = rd_attr
        graph_ids = graph_all

    dist = torch.norm(pos[dst] - pos[src], p=2, dim=1)
    edge_loss = dist.sub(targets).abs().div(targets.clamp_min(eps)).square()

    num_graphs = int(batch.batch.max().item()) + 1
    per_graph = torch.zeros(num_graphs, dtype=pos.dtype, device=pos.device)
    sample_counts = torch.zeros(num_graphs, dtype=pos.dtype, device=pos.device)
    per_graph.index_add_(0, graph_ids, edge_loss)
    sample_counts.index_add_(0, graph_ids, torch.ones_like(edge_loss))

    if max_pairs > 0 and rd_attr.numel() > max_pairs:
        scale = total_counts / sample_counts.clamp_min(1.0)
        per_graph = per_graph * scale

    valid_graphs = sample_counts > 0
    if not valid_graphs.any():
        return pos.sum() * 0.0
    return per_graph[valid_graphs].mean()


def compute_cosine_stress_targets(
    data,
    walk_steps=4,
    walk_decay=0.75,
    normalize="mean",
    eps=1e-9,
):
    """
    Compute Word2VecGD-inspired cosine-distance targets aligned with perm_index.

    Instead of training a separate skip-gram model, this lightweight variant
    builds a random-walk context matrix sum_k decay^(k-1) P^k and uses cosine
    dissimilarity between node context vectors as a semantic/topological target.
    """
    n = int(data.num_nodes)
    if n <= 1 or not hasattr(data, "perm_index") or not hasattr(data, "apsp_attr"):
        return torch.empty(0, dtype=torch.float32)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return data.apsp_attr.detach().clone().float()

    edge_index_cpu = edge_index.detach().cpu().long().numpy()
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edge_index_cpu.T:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        A[u, v] = 1.0
        A[v, u] = 1.0

    deg = A.sum(axis=1)
    P = np.zeros_like(A)
    nonzero = deg > eps
    P[nonzero] = A[nonzero] / deg[nonzero, None]

    steps = max(int(walk_steps), 1)
    decay = float(walk_decay)
    P_power = np.eye(n, dtype=np.float64)
    context = np.zeros((n, n), dtype=np.float64)
    for step in range(steps):
        P_power = P_power @ P
        context += (decay ** step) * P_power

    context = 0.5 * (context + context.T)
    context += 0.05 * np.eye(n, dtype=np.float64)
    row_norm = np.linalg.norm(context, axis=1, keepdims=True)
    context = context / np.maximum(row_norm, eps)

    sim = np.clip(context @ context.T, -1.0, 1.0)
    cos_dist = np.maximum(1.0 - sim, 0.0)

    perm_index = data.perm_index.detach().cpu().long()
    src = perm_index[0].numpy()
    dst = perm_index[1].numpy()
    targets = cos_dist[src, dst].astype(np.float64)

    apsp = data.apsp_attr.detach().cpu().numpy().astype(np.float64)
    valid = np.isfinite(targets) & (targets > eps) & np.isfinite(apsp) & (apsp > eps)
    normalize = str(normalize).lower()
    if valid.any() and normalize != "none":
        if normalize == "median":
            target_center = float(np.median(targets[valid]))
            apsp_center = float(np.median(apsp[valid]))
        else:
            target_center = float(np.mean(targets[valid]))
            apsp_center = float(np.mean(apsp[valid]))
        if target_center > eps:
            targets = targets * (apsp_center / target_center)

    targets = np.maximum(targets, eps)
    return torch.tensor(targets, dtype=torch.float32)


def attach_cosine_stress_targets(data_list, args):
    if not getattr(args, "use_cosine_stress", False):
        return data_list

    print(
        "[INFO] precomputing cosine-stress targets: "
        f"steps={args.cosine_walk_steps}, "
        f"decay={args.cosine_walk_decay}, "
        f"normalize={args.cosine_normalize}"
    )
    for data in tqdm(data_list, desc="Cosine targets"):
        data.cosine_attr = compute_cosine_stress_targets(
            data,
            walk_steps=args.cosine_walk_steps,
            walk_decay=args.cosine_walk_decay,
            normalize=args.cosine_normalize,
        )
    return data_list


def cosine_stress_loss_fn(pos, batch, args, eps=1e-9):
    if not getattr(args, "use_cosine_stress", False) or not hasattr(batch, "cosine_attr"):
        return pos.sum() * 0.0

    targets_all = batch.cosine_attr.to(device=pos.device, dtype=pos.dtype).view(-1)
    if targets_all.numel() == 0:
        return pos.sum() * 0.0

    perm_index = batch.perm_index.long().to(pos.device)
    src_all = perm_index[0]
    dst_all = perm_index[1]
    graph_all = batch.batch.long().to(pos.device)[src_all]
    total_counts = torch.bincount(
        graph_all,
        minlength=int(batch.batch.max().item()) + 1,
    ).to(dtype=pos.dtype)

    max_pairs = int(getattr(args, "cosine_max_pairs", 0))
    if max_pairs > 0 and targets_all.numel() > max_pairs:
        idx = torch.randperm(targets_all.numel(), device=pos.device)[:max_pairs]
        src = src_all[idx]
        dst = dst_all[idx]
        targets = targets_all[idx]
        graph_ids = graph_all[idx]
    else:
        src = src_all
        dst = dst_all
        targets = targets_all
        graph_ids = graph_all

    dist = torch.norm(pos[dst] - pos[src], p=2, dim=1)
    edge_loss = dist.sub(targets).abs().div(targets.clamp_min(eps)).square()

    num_graphs = int(batch.batch.max().item()) + 1
    per_graph = torch.zeros(num_graphs, dtype=pos.dtype, device=pos.device)
    sample_counts = torch.zeros(num_graphs, dtype=pos.dtype, device=pos.device)
    per_graph.index_add_(0, graph_ids, edge_loss)
    sample_counts.index_add_(0, graph_ids, torch.ones_like(edge_loss))

    if max_pairs > 0 and targets_all.numel() > max_pairs:
        scale = total_counts / sample_counts.clamp_min(1.0)
        per_graph = per_graph * scale

    valid_graphs = sample_counts > 0
    if not valid_graphs.any():
        return pos.sum() * 0.0
    return per_graph[valid_graphs].mean()


def unique_undirected_edge_tensor(edge_index, num_nodes):
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return edge_index.new_zeros((0, 2))

    src = edge_index[0]
    dst = edge_index[1]
    valid = src != dst
    if not valid.any():
        return edge_index.new_zeros((0, 2))

    src = src[valid]
    dst = dst[valid]
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    key = lo * int(num_nodes) + hi
    order = torch.argsort(key)
    key = key[order]
    edges = torch.stack([lo[order], hi[order]], dim=1)
    keep = torch.ones(key.numel(), dtype=torch.bool, device=edge_index.device)
    keep[1:] = key[1:] != key[:-1]
    return edges[keep]


def sampled_crossing_loss(
    pos,
    edge_index,
    batch_vector,
    max_pairs=2048,
    oversample_factor=8,
    temperature=8.0,
    margin=0.25,
    eps=1e-8,
):
    """
    Differentiable crossing surrogate over sampled non-adjacent edge pairs.

    It estimates whether two segments put each other's endpoints on opposite
    sides, using normalized signed areas. The loss is bounded and sampled, so
    it gives a crossing-aware signal without an O(E^2) training cost.
    """
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return pos.sum() * 0.0

    device = pos.device
    edge_index = edge_index.long().to(device)
    batch_vector = batch_vector.long().to(device)

    edges = unique_undirected_edge_tensor(edge_index, pos.size(0))
    num_edges = int(edges.size(0))
    if num_edges < 2:
        return pos.sum() * 0.0

    draw_count = max(max_pairs * max(int(oversample_factor), 1), max_pairs)
    first = torch.randint(num_edges, (draw_count,), device=device)
    second = torch.randint(num_edges, (draw_count,), device=device)

    e1 = edges[first]
    e2 = edges[second]

    same_graph = batch_vector[e1[:, 0]] == batch_vector[e2[:, 0]]
    disjoint = (
        (e1[:, 0] != e2[:, 0])
        & (e1[:, 0] != e2[:, 1])
        & (e1[:, 1] != e2[:, 0])
        & (e1[:, 1] != e2[:, 1])
    )
    ordered = first < second
    valid = same_graph & disjoint & ordered

    if not valid.any():
        return pos.sum() * 0.0

    valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if valid_idx.numel() > max_pairs:
        valid_idx = valid_idx[:max_pairs]

    e1 = e1[valid_idx]
    e2 = e2[valid_idx]

    a = pos[e1[:, 0]]
    b = pos[e1[:, 1]]
    c = pos[e2[:, 0]]
    d = pos[e2[:, 1]]

    ab = b - a
    cd = d - c

    def cross2(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    def normalized_cross(x, y):
        return cross2(x, y) / (x.norm(dim=1) * y.norm(dim=1) + eps)

    s1 = normalized_cross(ab, c - a)
    s2 = normalized_cross(ab, d - a)
    s3 = normalized_cross(cd, a - c)
    s4 = normalized_cross(cd, b - c)

    side_ab = torch.sigmoid(-float(temperature) * s1 * s2)
    side_cd = torch.sigmoid(-float(temperature) * s3 * s4)
    crossing_score = side_ab * side_cd

    return torch.relu(crossing_score - float(margin)).mean()


def need_spectral_pe(args):
    mode = effective_cser_mode(args)

    if mode in ["same_spectral", "full"] and not args.no_spectral_context:
        return True

    return False


def forward_model(batch, deepgd_model, refiner, stress_metric, args, gradient_probe=None):
    batch.perm_index = batch.perm_index.long()
    batch.batch = batch.batch.long()

    num_nodes = int(batch.batch.numel())

    init_pos = generate_init_pos(batch)

    if hasattr(batch, "community_label"):
        community_labels = batch.community_label.long().to(batch.batch.device)
    else:
        community_labels = build_batch_community_labels(
            edge_index=batch.edge_index.long(),
            batch_vector=batch.batch.long(),
            num_nodes=num_nodes,
        )
    augmentation = float(getattr(args, "hctf_community_augmentation", 0.0))
    if (
        augmentation > 0.0
        and getattr(args, "use_hctf", False)
        and getattr(args, "hctf_version", "v1") == "v2"
        and bool(deepgd_model.training)
    ):
        augmentation_seed = (
            int(getattr(args, "seed", 42))
            ^ (int(getattr(args, "current_epoch", 0)) * 0x85EBCA6B)
        )
        community_labels = augment_batch_community_labels(
            community_labels,
            batch.batch.long(),
            augmentation,
            augmentation_seed,
        )

    spectral_pe = None
    if need_spectral_pe(args):
        spectral_pe = build_batch_spectral_pe(
            edge_index=batch.edge_index.long(),
            batch_vector=batch.batch.long(),
            num_nodes=num_nodes,
            spectral_dim=args.spectral_dim,
            community_labels=community_labels,
            use_adaspec=args.use_adaspec_cser,
            adaspec_alpha_s=args.adaspec_alpha_s,
            adaspec_alpha_f=args.adaspec_alpha_f,
            adaspec_self_loop_scale=args.adaspec_self_loop_scale,
            adaspec_boundary_weight=args.adaspec_boundary_weight,
            adaspec_core_weight=args.adaspec_core_weight,
        )

    base_edge_attr = get_edge_features(batch.apsp_attr)

    cser_mode = effective_cser_mode(args)

    caid_context = None
    if getattr(args, "use_caid", False):
        caid_context = build_caid_context(
            apsp_attr=batch.apsp_attr,
            perm_index=batch.perm_index.long(),
            edge_index=batch.edge_index.long(),
            community_labels=community_labels,
            batch_vector=batch.batch.long(),
            intra_shrink=args.caid_intra_shrink,
            adj_expand=args.caid_adj_expand,
            far_expand=args.caid_far_expand,
            min_ratio=args.caid_min_ratio,
            max_ratio=args.caid_max_ratio,
            graph_local_communities=getattr(
                args,
                "graph_local_community_objectives",
                False,
            ),
        )

    model_edge_attr = build_cser_edge_attr(
        base_edge_attr=base_edge_attr,
        edge_index=batch.perm_index.long(),
        pos=init_pos,
        community_labels=community_labels,
        batch_vector=batch.batch.long(),
        spectral_pe=spectral_pe,
        mode=cser_mode,
        use_spectral=not args.no_spectral_context,
    )

    if caid_context is not None and not getattr(args, "no_caid_edge_features", False):
        caid_edge_context = torch.cat(
            [
                caid_context["ratio"].to(dtype=model_edge_attr.dtype),
                caid_context["boundary_pair"].to(dtype=model_edge_attr.dtype),
            ],
            dim=-1,
        )
        model_edge_attr = torch.cat([model_edge_attr, caid_edge_context], dim=-1)

    if getattr(args, "use_care", False):
        if caid_context is not None:
            care_edge_context = torch.cat(
                [
                    caid_context["adjacent_pair"].to(dtype=model_edge_attr.dtype),
                    caid_context["far_pair"].to(dtype=model_edge_attr.dtype),
                ],
                dim=-1,
            )
        else:
            care_edge_context = torch.zeros(
                model_edge_attr.size(0),
                2,
                dtype=model_edge_attr.dtype,
                device=model_edge_attr.device,
        )
        model_edge_attr = torch.cat([model_edge_attr, care_edge_context], dim=-1)

    if getattr(args, "use_topocser", False):
        topo_edge_context = build_topocser_edge_attr(
            pair_index=batch.perm_index.long(),
            graph_edge_index=batch.edge_index.long(),
            community_labels=community_labels,
            batch_vector=batch.batch.long(),
        )
        model_edge_attr = torch.cat(
            [
                model_edge_attr,
                topo_edge_context.to(dtype=model_edge_attr.dtype),
            ],
            dim=-1,
        )

    hctf_teacher_pos_raw = None
    hctf_structure_teacher_pos_raw = None
    hctf_structure_teacher_enabled = bool(
        getattr(args, "_hctf_teacher_enabled", False)
        and
        getattr(args, "use_hctf", False)
        and getattr(args, "hctf_adaptive_structure_gate", False)
        and float(getattr(args, "lambda_hctf_structure_gate_supervision", 0.0)) > 0.0
    )
    if hctf_structure_teacher_enabled:
        # Exact nested counterfactual: the same model and node residual are
        # evaluated with only the structural correction gate forced to zero.
        deepgd_model.set_hctf_structure_gate_override(0.0)
        try:
            with torch.no_grad():
                hctf_structure_teacher_pos_raw = deepgd_model(
                    init_pos=init_pos,
                    edge_index=batch.perm_index.long(),
                    edge_attr=model_edge_attr,
                    batch_index=batch.batch.long(),
                    graph_edge_index=batch.edge_index.long(),
                    community_labels=community_labels,
                )
        finally:
            deepgd_model.set_hctf_structure_gate_override(None)
    hctf_teacher_enabled = (
        getattr(args, "_hctf_teacher_enabled", False)
        and hctf_baseline_teacher_requested(args)
    )
    if hctf_teacher_enabled:
        previous_bypass = bool(getattr(deepgd_model, "hctf_bypass", False))
        previous_conflict_enabled = bool(getattr(
            deepgd_model, "hctf_conflict_head_enabled", True
        ))
        # The teacher must be deterministic even during joint v2 fine-tuning.
        # Preserve every submodule's mode because frozen-HCTF training keeps the
        # backbone in eval mode while the adapter itself remains in train mode.
        previous_training_modes = {
            module: bool(module.training) for module in deepgd_model.modules()
        }
        if getattr(args, "freeze_hctf_block_for_conflict_head", False):
            # Exact nested teacher for the single-factor conflict experiment:
            # keep the already-trained forest active and disable only the new head.
            deepgd_model.hctf_conflict_head_enabled = False
        else:
            deepgd_model.hctf_bypass = True
        try:
            deepgd_model.eval()
            with torch.no_grad():
                hctf_teacher_pos_raw = deepgd_model(
                    init_pos=init_pos,
                    edge_index=batch.perm_index.long(),
                    edge_attr=model_edge_attr,
                    batch_index=batch.batch.long(),
                    graph_edge_index=batch.edge_index.long(),
                    community_labels=community_labels,
                )
        finally:
            deepgd_model.hctf_bypass = previous_bypass
            deepgd_model.hctf_conflict_head_enabled = previous_conflict_enabled
            for module, training in previous_training_modes.items():
                module.training = training

    if getattr(args, "use_paretofield", False):
        base_pos_raw, deepgd_hidden = deepgd_model(
            init_pos=init_pos,
            edge_index=batch.perm_index.long(),
            edge_attr=model_edge_attr,
            batch_index=batch.batch.long(),
            graph_edge_index=batch.edge_index.long(),
            community_labels=community_labels,
            return_hidden=True,
        )
    else:
        base_pos_raw = deepgd_model(
            init_pos=init_pos,
            edge_index=batch.perm_index.long(),
            edge_attr=model_edge_attr,
            batch_index=batch.batch.long(),
            graph_edge_index=batch.edge_index.long(),
            community_labels=community_labels,
        )
        deepgd_hidden = None

    if args.rescale_before_refiner:
        if getattr(args, "hctf_centered_rescale", False):
            base_pos = centered_rescale_by_stress(
                base_pos_raw, init_pos, batch.apsp_attr,
                batch.perm_index.long(), batch.batch.long(),
            )
        else:
            base_pos = rescale_by_stress(
                base_pos_raw, batch.apsp_attr,
                batch.perm_index.long(), batch.batch.long(),
            )
    else:
        base_pos = base_pos_raw

    hctf_teacher_pos = None
    if hctf_teacher_pos_raw is not None:
        if args.rescale_before_refiner:
            if getattr(args, "hctf_centered_rescale", False):
                hctf_teacher_pos = centered_rescale_by_stress(
                    hctf_teacher_pos_raw, init_pos, batch.apsp_attr,
                    batch.perm_index.long(), batch.batch.long(),
                )
            else:
                hctf_teacher_pos = rescale_by_stress(
                    hctf_teacher_pos_raw, batch.apsp_attr,
                    batch.perm_index.long(), batch.batch.long(),
                )
        else:
            hctf_teacher_pos = hctf_teacher_pos_raw

    hctf_structure_teacher_pos = None
    if hctf_structure_teacher_pos_raw is not None:
        if args.rescale_before_refiner:
            if getattr(args, "hctf_centered_rescale", False):
                hctf_structure_teacher_pos = centered_rescale_by_stress(
                    hctf_structure_teacher_pos_raw, init_pos, batch.apsp_attr,
                    batch.perm_index.long(), batch.batch.long(),
                )
            else:
                hctf_structure_teacher_pos = rescale_by_stress(
                    hctf_structure_teacher_pos_raw, batch.apsp_attr,
                    batch.perm_index.long(), batch.batch.long(),
                )
        else:
            hctf_structure_teacher_pos = hctf_structure_teacher_pos_raw

    base_losses = compute_layout_objective(
        pos=base_pos,
        batch=batch,
        community_labels=community_labels,
        stress_metric=stress_metric,
        args=args,
    )

    if args.use_refiner and refiner is not None:
        spectral_edge_index = build_spectral_knn_edges(
            edge_index=batch.edge_index.long(),
            batch_vector=batch.batch.long(),
            num_nodes=num_nodes,
            spectral_dim=args.spectral_dim,
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

    pareto_seed_pos = refined_pos
    paretofield_preferences = None
    paretofield_effective_preferences = None
    paretofield_community_confidence = None
    paretofield_relation_uncertainty = None
    paretofield_edge_pair_count = 0
    paretofield_reference_positions = []
    paretofield_guard_stats = None
    paretofield_primary_steps = 0
    if getattr(args, "use_paretofield", False):
        topo_node_features = build_topocser_node_features(
            edge_index=batch.edge_index.long(),
            community_labels=community_labels,
            batch_vector=batch.batch.long(),
        ).to(dtype=pareto_seed_pos.dtype)
        field_node_features = torch.cat(
            [topo_node_features, deepgd_hidden.to(dtype=pareto_seed_pos.dtype)], dim=1
        )
        paretofield_preferences = build_paretofield_preferences(
            args=args,
            batch_index=batch.batch.long(),
            training=deepgd_model.paretofield.training,
            dtype=pareto_seed_pos.dtype,
        )
        pareto_candidate_pos = deepgd_model.paretofield(
            pos=pareto_seed_pos,
            node_features=field_node_features,
            graph_edges=batch.edge_index.long(),
            long_range_edges=batch.perm_index.long(),
            batch_index=batch.batch.long(),
            community_labels=community_labels,
            preference=paretofield_preferences,
        )
        paretofield_primary_steps = int(deepgd_model.paretofield.last_inference_steps)
        paretofield_effective_preferences = getattr(
            deepgd_model.paretofield, "last_effective_preference", None
        )
        if paretofield_effective_preferences is None:
            paretofield_effective_preferences = paretofield_preferences
        paretofield_community_confidence = getattr(
            deepgd_model.paretofield, "last_community_confidence", None
        )
        paretofield_relation_uncertainty = getattr(
            deepgd_model.paretofield, "last_relation_uncertainty", None
        )
        paretofield_edge_pair_count = int(
            getattr(deepgd_model.paretofield, "last_edge_pair_count", 0)
        )
        refined_pos = pareto_candidate_pos
        if (
            getattr(args, "paretofield_safe_guard", False)
            and not deepgd_model.paretofield.training
        ):
            seed_guard_crossing = None
            candidate_guard_crossing = None
            guard_crossing_evaluator = None
            if (
                args.paretofield_guard_mode == "pareto"
                and getattr(args, "paretofield_guard_crossing", False)
            ):
                guard_ratios = parse_float_list(
                    getattr(args, "hard_crossing_ratios", "0.40,0.30,0.20,0.10")
                )
                if len(guard_ratios) != 4 or sum(guard_ratios) <= 0:
                    raise ValueError(
                        "--hard-crossing-ratios must contain four positive-sum values"
                    )
                guard_ratios = tuple(
                    value / sum(guard_ratios) for value in guard_ratios
                )
                if getattr(args, "paretofield_exact_crossing_guard", False):
                    audit_budget = max(int(args.crossing_max_pairs) // 2, 1)
                    seed_first, seed_second = sample_hard_edge_pairs(
                        pos=pareto_seed_pos,
                        edge_index=batch.edge_index.long(),
                        batch_index=batch.batch.long(),
                        max_pairs=audit_budget,
                        oversample_factor=args.crossing_oversample_factor,
                        temperature=args.crossing_temperature,
                        ratios=guard_ratios,
                    )
                    candidate_first, candidate_second = sample_hard_edge_pairs(
                        pos=pareto_candidate_pos,
                        edge_index=batch.edge_index.long(),
                        batch_index=batch.batch.long(),
                        max_pairs=audit_budget,
                        oversample_factor=args.crossing_oversample_factor,
                        temperature=args.crossing_temperature,
                        ratios=guard_ratios,
                    )
                    audit_first = torch.cat(
                        [seed_first, candidate_first], dim=0
                    )
                    audit_second = torch.cat(
                        [seed_second, candidate_second], dim=0
                    )

                    def guard_crossing_evaluator(trial_pos):
                        return crossing_loss_from_edge_pairs_per_graph(
                            pos=trial_pos,
                            first_edges=audit_first,
                            second_edges=audit_second,
                            batch_index=batch.batch.long(),
                            temperature=args.crossing_temperature,
                            margin=args.crossing_margin,
                        )
                else:
                    seed_guard_crossing = hard_sampled_crossing_loss_per_graph(
                        pos=pareto_seed_pos,
                        edge_index=batch.edge_index.long(),
                        batch_index=batch.batch.long(),
                        max_pairs=args.crossing_max_pairs,
                        oversample_factor=args.crossing_oversample_factor,
                        temperature=args.crossing_temperature,
                        margin=args.crossing_margin,
                        ratios=guard_ratios,
                    )
                    candidate_guard_crossing = hard_sampled_crossing_loss_per_graph(
                        pos=pareto_candidate_pos,
                        edge_index=batch.edge_index.long(),
                        batch_index=batch.batch.long(),
                        max_pairs=args.crossing_max_pairs,
                        oversample_factor=args.crossing_oversample_factor,
                        temperature=args.crossing_temperature,
                        margin=args.crossing_margin,
                        ratios=guard_ratios,
                    )
            refined_pos, paretofield_guard_stats = graphwise_pareto_guard(
                seed_pos=pareto_seed_pos,
                candidate_pos=pareto_candidate_pos,
                pair_index=batch.perm_index.long(),
                target_distance=batch.apsp_attr,
                edge_index=batch.edge_index.long(),
                batch_index=batch.batch.long(),
                community_labels=community_labels,
                preference=paretofield_effective_preferences,
                seed_crossing=seed_guard_crossing,
                candidate_crossing=candidate_guard_crossing,
                max_geometry_regret=args.paretofield_guard_max_geometry_regret,
                max_visual_regret=args.paretofield_guard_max_visual_regret,
                max_community_regret=args.paretofield_guard_max_community_regret,
                max_weighted_regret=args.paretofield_guard_max_weighted_regret,
                mode=args.paretofield_guard_mode,
                step_fractions=(
                    parse_float_list(args.paretofield_guard_step_fractions)
                    if args.paretofield_continuous_guard
                    else None
                ),
                crossing_evaluator=guard_crossing_evaluator,
                componentwise=args.paretofield_componentwise_guard,
            )
        if (
            not deepgd_model.training
            and getattr(args, "mode", "train") == "eval"
            and getattr(args, "use_mtpf", False)
            and int(getattr(args, "paretofield_reference_divisions", 0)) > 0
        ):
            directions = reference_directions(
                3, divisions=int(args.paretofield_reference_divisions)
            )
            max_layouts = int(getattr(args, "paretofield_max_reference_layouts", 12))
            if max_layouts > 0 and directions.shape[0] > max_layouts:
                keep = np.linspace(0, directions.shape[0] - 1, max_layouts)
                directions = directions[np.unique(np.rint(keep).astype(np.int64))]
            for direction_id, direction in enumerate(directions):
                preference = torch.tensor(
                    direction,
                    dtype=pareto_seed_pos.dtype,
                    device=pareto_seed_pos.device,
                )[None, :].expand(paretofield_preferences.size(0), -1)
                reference_pos = deepgd_model.paretofield(
                    pos=pareto_seed_pos,
                    node_features=field_node_features,
                    graph_edges=batch.edge_index.long(),
                    long_range_edges=batch.perm_index.long(),
                    batch_index=batch.batch.long(),
                    community_labels=community_labels,
                    preference=preference,
                )
                paretofield_reference_positions.append(
                    (f"paretofield_ref_{direction_id:02d}", reference_pos.detach())
                )

    final_losses = compute_layout_objective(
        pos=refined_pos,
        batch=batch,
        community_labels=community_labels,
        stress_metric=stress_metric,
        args=args,
    )

    caid_weight = 0.0
    caid_loss = torch.tensor(0.0, device=refined_pos.device)
    if caid_context is not None and args.lambda_caid > 0:
        caid_weight = caid_schedule(args)
        caid_loss = stress_metric(
            refined_pos,
            batch.perm_index.long(),
            caid_context["adjusted_apsp"],
            batch.batch.long(),
        )

    cpsg_weight = 0.0
    cpsg_proto_loss = torch.tensor(0.0, device=refined_pos.device)
    cpsg_sil_loss = torch.tensor(0.0, device=refined_pos.device)
    cpsg_sep_loss = torch.tensor(0.0, device=refined_pos.device)
    cpsg_total_loss = torch.tensor(0.0, device=refined_pos.device)
    if getattr(args, "use_cpsg", False):
        cpsg_weight = cpsg_schedule(args)
        cpsg_proto_loss, cpsg_sil_loss, cpsg_sep_loss = cpsg_loss(
            pos=refined_pos,
            edge_index=batch.edge_index.long(),
            community_labels=community_labels,
            batch_vector=batch.batch.long(),
            args=args,
        )
        cpsg_total_loss = (
            args.lambda_cpsg_proto * cpsg_proto_loss
            + args.lambda_cpsg_sil * cpsg_sil_loss
            + args.lambda_cpsg_sep * cpsg_sep_loss
        )

    anchor_loss = ((refined_pos - base_pos.detach()) ** 2).sum(dim=1).mean()

    pmds_anchor_weight = 0.0
    pmds_anchor_loss_value = torch.tensor(0.0, device=refined_pos.device)
    if getattr(args, "lambda_pmds_anchor", 0.0) > 0:
        pmds_anchor_weight = epoch_linear_schedule(
            args,
            "pmds_anchor_start_epoch",
            "pmds_anchor_warmup_epochs",
        )
        pmds_node_mask, pmds_center_per_graph = pmds_anchor_training_options(
            args, batch
        )
        pmds_anchor_loss_value = pmds_anchor_loss(
            refined_pos,
            init_pos,
            huber_delta=float(getattr(args, "pmds_anchor_huber_delta", 0.0)),
            node_mask=pmds_node_mask,
            batch_index=batch.batch.long(),
            center_per_graph=pmds_center_per_graph,
        )

    crossing_ratios = parse_float_list(
        getattr(args, "hard_crossing_ratios", "0.40,0.30,0.20,0.10")
    )
    if len(crossing_ratios) != 4 or sum(crossing_ratios) <= 0:
        raise ValueError("--hard-crossing-ratios must contain four positive-sum values")
    crossing_ratios = tuple(value / sum(crossing_ratios) for value in crossing_ratios)

    crossing_per_graph = None
    paretofield_pure_eval = (
        getattr(args, "mode", "train") == "eval"
        and getattr(args, "use_paretofield", False)
        and not deepgd_model.paretofield.training
    )
    if (
        not paretofield_pure_eval
        and (
            getattr(args, "use_paretofield", False)
            or getattr(args, "hard_crossing_sampling", False)
        )
    ):
        crossing_per_graph = hard_sampled_crossing_loss_per_graph(
            pos=refined_pos,
            edge_index=batch.edge_index.long(),
            batch_index=batch.batch.long(),
            max_pairs=args.crossing_max_pairs,
            oversample_factor=args.crossing_oversample_factor,
            temperature=args.crossing_temperature,
            margin=args.crossing_margin,
            ratios=crossing_ratios,
        )

    crossing_weight = 0.0
    crossing_loss_value = torch.tensor(0.0, device=refined_pos.device)
    if getattr(args, "lambda_crossing", 0.0) > 0 and not paretofield_pure_eval:
        crossing_weight = epoch_linear_schedule(
            args,
            "crossing_start_epoch",
            "crossing_warmup_epochs",
        )
        if getattr(args, "hard_crossing_sampling", False):
            crossing_loss_value = crossing_per_graph.mean()
        else:
            crossing_loss_value = sampled_crossing_loss(
                pos=refined_pos,
                edge_index=batch.edge_index.long(),
                batch_vector=batch.batch.long(),
                max_pairs=args.crossing_max_pairs,
                oversample_factor=args.crossing_oversample_factor,
                temperature=args.crossing_temperature,
                margin=args.crossing_margin,
            )

    train_silhouette_weight = 0.0
    train_silhouette_loss_value = torch.tensor(0.0, device=refined_pos.device)
    if getattr(args, "lambda_train_silhouette", 0.0) > 0:
        train_silhouette_weight = epoch_linear_schedule(
            args,
            "train_silhouette_start_epoch",
            "train_silhouette_warmup_epochs",
        )
        train_silhouette_loss_value = mtpf_silhouette_surrogate_loss(
            refined_pos,
            batch.edge_index.long(),
            margin=float(args.train_silhouette_margin),
        )

    hctf_readability_weight = 0.0
    hctf_readability_loss_value = torch.tensor(0.0, device=base_pos.device)
    hctf_angular_loss_value = torch.tensor(0.0, device=base_pos.device)
    hctf_crossing_angle_loss_value = torch.tensor(0.0, device=base_pos.device)
    hctf_clearance_loss_value = torch.tensor(0.0, device=base_pos.device)
    hctf_neighborhood_loss_value = torch.tensor(0.0, device=base_pos.device)
    if float(getattr(args, "lambda_hctf_readability", 0.0)) > 0.0:
        hctf_readability_weight = epoch_linear_schedule(
            args,
            "hctf_readability_start_epoch",
            "hctf_readability_warmup_epochs",
        )
        readability_losses = readability_surrogate_losses(
            pos=base_pos,
            edge_index=batch.edge_index.long(),
            batch_index=batch.batch.long(),
            pair_index=batch.perm_index.long(),
            target_distance=batch.apsp_attr,
            max_pairs=int(getattr(args, "hctf_readability_max_pairs", 4096)),
            angular_target_ratio=float(
                getattr(args, "hctf_readability_angular_target_ratio", 0.5)
            ),
            angular_max_target_degrees=float(
                getattr(args, "hctf_readability_angular_max_degrees", 30.0)
            ),
            crossing_angle_tail_fraction=float(
                getattr(args, "hctf_readability_crossing_angle_tail_fraction", 0.20)
            ),
            clearance_margin=float(
                getattr(args, "hctf_readability_clearance_margin", 0.10)
            ),
            neighborhood_near_hops=float(
                getattr(args, "hctf_readability_near_hops", 2.0)
            ),
            neighborhood_margin=float(
                getattr(args, "hctf_readability_neighborhood_margin", 0.10)
            ),
        )
        hctf_angular_loss_value = readability_losses["angular"]
        hctf_crossing_angle_loss_value = readability_losses["crossing_angle"]
        hctf_clearance_loss_value = readability_losses["clearance"]
        hctf_neighborhood_loss_value = readability_losses["neighborhood"]
        component_weights = {
            "angular": float(getattr(args, "hctf_readability_w_angular", 1.0)),
            "crossing_angle": float(
                getattr(args, "hctf_readability_w_crossing_angle", 1.0)
            ),
            "clearance": float(getattr(args, "hctf_readability_w_clearance", 1.0)),
            "neighborhood": float(
                getattr(args, "hctf_readability_w_neighborhood", 1.0)
            ),
        }
        component_weight_sum = max(sum(component_weights.values()), 1e-12)
        readability_dominance_enabled = bool(
            getattr(args, "hctf_readability_dominance", False)
        )
        if readability_dominance_enabled and hctf_teacher_pos is not None:
            with torch.no_grad():
                teacher_readability_losses = readability_surrogate_losses(
                    pos=hctf_teacher_pos,
                    edge_index=batch.edge_index.long(),
                    batch_index=batch.batch.long(),
                    pair_index=batch.perm_index.long(),
                    target_distance=batch.apsp_attr,
                    max_pairs=int(getattr(args, "hctf_readability_max_pairs", 4096)),
                    angular_target_ratio=float(
                        getattr(args, "hctf_readability_angular_target_ratio", 0.5)
                    ),
                    angular_max_target_degrees=float(
                        getattr(args, "hctf_readability_angular_max_degrees", 30.0)
                    ),
                    crossing_angle_tail_fraction=float(
                        getattr(args, "hctf_readability_crossing_angle_tail_fraction", 0.20)
                    ),
                    clearance_margin=float(
                        getattr(args, "hctf_readability_clearance_margin", 0.10)
                    ),
                    neighborhood_near_hops=float(
                        getattr(args, "hctf_readability_near_hops", 2.0)
                    ),
                    neighborhood_margin=float(
                        getattr(args, "hctf_readability_neighborhood_margin", 0.10)
                    ),
                )
            hctf_readability_loss_value, _ = componentwise_readability_dominance_loss(
                readability_losses,
                teacher_readability_losses,
                component_weights,
                target_improvement=float(
                    getattr(args, "hctf_readability_target_improvement", 0.0)
                ),
            )
        else:
            # Pure checkpoint evaluation does not need a teacher because this
            # loss is telemetry only; the produced coordinates are unchanged.
            # Training/validation still enters the dominance branch above.
            hctf_readability_loss_value = sum(
                component_weights[name] * readability_losses[name]
                for name in component_weights
            ) / component_weight_sum

    hctf_dominance_loss_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_anchor_loss_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_max_regret_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_tail_risk_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_crossing_barrier_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_exact_gate_loss_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_exact_gate_target_value = torch.tensor(1.0, device=refined_pos.device)
    hctf_exact_gate_unsafe_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_exact_gate_regret_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_structure_gate_cost_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_structure_gate_loss_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_structure_gate_target_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_structure_gate_unsafe_value = torch.tensor(0.0, device=refined_pos.device)
    hctf_structure_gate_regret_value = torch.tensor(0.0, device=refined_pos.device)
    current_structure_gate = getattr(
        getattr(deepgd_model, "hctf_block", None),
        "last_structure_gate_tensor",
        None,
    )
    if current_structure_gate is not None:
        # An L0-style expected-compute prior: the structural correction must
        # earn its use through the task loss instead of staying open by default.
        hctf_structure_gate_cost_value = current_structure_gate.mean()
    if hctf_structure_teacher_pos is not None:
        if current_structure_gate is None:
            raise RuntimeError("adaptive HCTF structure gate did not produce graph weights")
        (
            hctf_structure_gate_loss_value,
            hctf_structure_gate_target_value,
            hctf_structure_gate_unsafe_value,
            hctf_structure_gate_regret_value,
        ) = hctf_exact_risk_gate_loss(
            current_risk=current_structure_gate,
            candidate_pos=base_pos,
            teacher_pos=hctf_structure_teacher_pos.detach(),
            batch=batch,
            args=args,
            component_spec=args.hctf_structure_gate_components,
            tolerance=args.hctf_structure_gate_tolerance,
            temperature=args.hctf_structure_gate_temperature,
        )
    if hctf_teacher_pos is not None:
        with torch.no_grad():
            if getattr(args, "hctf_dominance_deterministic_pairs", False):
                hctf_first_edges, hctf_second_edges = select_deterministic_conflict_pairs(
                    pos=hctf_teacher_pos,
                    edge_index=batch.edge_index.long(),
                    batch_index=batch.batch.long(),
                    max_pairs_per_graph=int(args.hctf_dominance_max_pairs),
                )
            else:
                hctf_first_edges, hctf_second_edges = select_metric_aligned_edge_pairs(
                    pos=hctf_teacher_pos,
                    edge_index=batch.edge_index.long(),
                    batch_index=batch.batch.long(),
                    max_pairs=int(args.hctf_dominance_max_pairs),
                    exact_max_edges=int(args.hctf_crossing_exact_max_edges),
                    oversample_factor=args.crossing_oversample_factor,
                    temperature=args.crossing_temperature,
                    ratios=crossing_ratios,
                )
            teacher_crossing_count, teacher_crossing_norm = (
                crossing_objectives_from_edge_pairs_per_graph(
                pos=hctf_teacher_pos,
                first_edges=hctf_first_edges,
                second_edges=hctf_second_edges,
                edge_index=batch.edge_index.long(),
                batch_index=batch.batch.long(),
                temperature=args.crossing_temperature,
                margin=args.crossing_margin,
                )
            )
            teacher_components = hctf_graphwise_component_objectives(
                pos=hctf_teacher_pos,
                batch=batch,
                community_labels=community_labels,
                crossing_count_per_graph=teacher_crossing_count,
                crossing_norm_per_graph=teacher_crossing_norm,
            )
        candidate_crossing_count, candidate_crossing_norm = (
            crossing_objectives_from_edge_pairs_per_graph(
            pos=base_pos,
            first_edges=hctf_first_edges,
            second_edges=hctf_second_edges,
            edge_index=batch.edge_index.long(),
            batch_index=batch.batch.long(),
            temperature=args.crossing_temperature,
            margin=args.crossing_margin,
            )
        )
        candidate_components = hctf_graphwise_component_objectives(
            pos=base_pos,
            batch=batch,
            community_labels=community_labels,
            crossing_count_per_graph=candidate_crossing_count,
            crossing_norm_per_graph=candidate_crossing_norm,
        )
        (
            hctf_dominance_loss_value,
            hctf_max_regret_value,
            _,
            hctf_tail_risk_value,
        ) = hctf_componentwise_dominance_loss(
            candidate_components,
            teacher_components,
            args,
        )
        hctf_crossing_barrier_value = crossing_topology_barrier_per_graph(
            candidate_pos=base_pos,
            teacher_pos=hctf_teacher_pos,
            first_edges=hctf_first_edges,
            second_edges=hctf_second_edges,
            batch_index=batch.batch.long(),
            margin=float(args.hctf_crossing_barrier_margin),
            preserve_only=bool(getattr(args, "hctf_crossing_preserve_only", False)),
            tail_fraction=float(getattr(args, "hctf_crossing_barrier_tail_fraction", 0.0)),
        ).mean()
        hctf_anchor_loss_value = (
            base_pos - hctf_teacher_pos.detach()
        ).square().sum(dim=1).mean()
        if (
            getattr(args, "hctf_exact_risk_gate", False)
            and float(getattr(args, "lambda_hctf_exact_gate", 0.0)) > 0.0
        ):
            current_risk = getattr(
                deepgd_model,
                "last_hctf_exact_risk_tensor",
                None,
            )
            if current_risk is None:
                raise RuntimeError("exact HCTF risk gate did not produce graph weights")
            (
                hctf_exact_gate_loss_value,
                hctf_exact_gate_target_value,
                hctf_exact_gate_unsafe_value,
                hctf_exact_gate_regret_value,
            ) = hctf_exact_risk_gate_loss(
                current_risk=current_risk,
                candidate_pos=base_pos,
                teacher_pos=hctf_teacher_pos.detach(),
                batch=batch,
                args=args,
            )

    paretofield_objective_loss = torch.tensor(0.0, device=refined_pos.device)
    paretofield_regret_loss = torch.tensor(0.0, device=refined_pos.device)
    paretofield_teacher_loss = torch.tensor(0.0, device=refined_pos.device)
    paretofield_anchor_loss = torch.tensor(0.0, device=refined_pos.device)
    paretofield_field_loss = torch.tensor(0.0, device=refined_pos.device)
    if getattr(args, "use_paretofield", False) and not paretofield_pure_eval:
        current_objectives = graphwise_pareto_objectives(
            pos=refined_pos,
            pair_index=batch.perm_index.long(),
            target_distance=batch.apsp_attr,
            edge_index=batch.edge_index.long(),
            batch_index=batch.batch.long(),
            community_labels=community_labels,
            crossing_per_graph=crossing_per_graph,
        )
        with torch.no_grad():
            seed_crossing = hard_sampled_crossing_loss_per_graph(
                pos=pareto_seed_pos.detach(),
                edge_index=batch.edge_index.long(),
                batch_index=batch.batch.long(),
                max_pairs=args.crossing_max_pairs,
                oversample_factor=args.crossing_oversample_factor,
                temperature=args.crossing_temperature,
                margin=args.crossing_margin,
                ratios=crossing_ratios,
            )
            seed_objectives = graphwise_pareto_objectives(
                pos=pareto_seed_pos.detach(),
                pair_index=batch.perm_index.long(),
                target_distance=batch.apsp_attr,
                edge_index=batch.edge_index.long(),
                batch_index=batch.batch.long(),
                community_labels=community_labels,
                crossing_per_graph=seed_crossing,
            )
            objective_scale = seed_objectives.abs().clamp_min(0.05)
        effective_preferences = (
            paretofield_effective_preferences
            if paretofield_effective_preferences is not None
            else paretofield_preferences
        )
        normalized_objectives = current_objectives / objective_scale
        paretofield_objective_loss = (
            effective_preferences * normalized_objectives
        ).sum(dim=1).mean()
        paretofield_regret_loss, _ = graphwise_regret_loss(
            seed_objectives=seed_objectives,
            candidate_objectives=current_objectives,
            preference=effective_preferences,
            limits=[
                args.paretofield_regret_max_geometry,
                args.paretofield_regret_max_visual,
                args.paretofield_regret_max_community,
            ],
            max_weighted_regret=args.paretofield_regret_max_weighted,
        )
        paretofield_anchor_loss = (
            refined_pos - pareto_seed_pos.detach()
        ).square().sum(dim=1).mean()

        teacher_archive = getattr(deepgd_model, "pareto_teacher_archive", None)
        if teacher_archive is not None:
            graph_count = int(paretofield_preferences.size(0))
            node_counts = torch.bincount(
                batch.batch.long(), minlength=graph_count
            ).detach().cpu().tolist()
            teacher_pos, teacher_mask, teacher_weights = teacher_archive.select_batch(
                names=get_batch_graph_names(batch, graph_count),
                preferences=effective_preferences,
                node_counts=node_counts,
                device=refined_pos.device,
                dtype=refined_pos.dtype,
            )
            paretofield_teacher_loss = similarity_aligned_mse(
                prediction=refined_pos,
                target=teacher_pos,
                batch_index=batch.batch.long(),
                graph_mask=teacher_mask,
                graph_weights=teacher_weights,
                allow_reflection=True,
            )
        paretofield_field_loss = (
            args.paretofield_anchor_weight * paretofield_anchor_loss
            + args.paretofield_teacher_weight * paretofield_teacher_loss
        )

    total_loss = (
        final_losses["objective"]
        + args.lambda_caid * caid_weight * caid_loss
        + cpsg_weight * cpsg_total_loss
        + args.lambda_pmds_anchor * pmds_anchor_weight * pmds_anchor_loss_value
        + args.lambda_crossing * crossing_weight * crossing_loss_value
        + args.lambda_train_silhouette * train_silhouette_weight * train_silhouette_loss_value
        + args.lambda_anchor * anchor_loss
        + args.lambda_delta_reg * delta_reg_loss
        + args.lambda_balance * balance_loss
        + args.lambda_paretofield * paretofield_objective_loss
        + args.paretofield_regret_weight * paretofield_regret_loss
        + paretofield_field_loss
    )

    hctf_training_loss = (
        args.lambda_hctf_dominance * hctf_dominance_loss_value
        + args.lambda_hctf_anchor * hctf_anchor_loss_value
        + args.lambda_hctf_crossing_barrier * hctf_crossing_barrier_value
        + args.lambda_hctf_exact_gate * hctf_exact_gate_loss_value
        + args.lambda_hctf_structure_gate_cost * hctf_structure_gate_cost_value
        + args.lambda_hctf_structure_gate_supervision * hctf_structure_gate_loss_value
        + float(getattr(args, "lambda_hctf_readability", 0.0))
        * hctf_readability_weight
        * hctf_readability_loss_value
    )
    if hctf_teacher_pos is not None and args.hctf_dominance_only:
        total_loss = hctf_training_loss
    else:
        total_loss = total_loss + hctf_training_loss

    if args.use_refiner and refiner is not None and args.lambda_base_aux > 0:
        total_loss = total_loss + args.lambda_base_aux * base_losses["objective"]

    # Diagnostic-only live tensors. The regular training path never passes a
    # probe, so this does not alter the objective or its backward pass.
    if gradient_probe is not None:
        gradient_probe.update({
            "_base_pos_raw": base_pos_raw,
            "_init_pos": init_pos.detach(),
            "stress": args.lambda_stress * final_losses["stress_optimization"],
            "layout_community": (
                args.lambda_intra * final_losses["intra"]
                + args.lambda_inter * final_losses["inter"]
                + args.lambda_pair * final_losses["pair"]
            ),
            "caid": args.lambda_caid * caid_weight * caid_loss,
            "cpsg": cpsg_weight * cpsg_total_loss,
            "crossing": args.lambda_crossing * crossing_weight * crossing_loss_value,
            "silhouette": (
                args.lambda_train_silhouette
                * train_silhouette_weight
                * train_silhouette_loss_value
            ),
            "pmds_anchor": (
                args.lambda_pmds_anchor
                * pmds_anchor_weight
                * pmds_anchor_loss_value
            ),
            "hctf_gate": (
                args.lambda_hctf_structure_gate_cost * hctf_structure_gate_cost_value
                + args.lambda_hctf_structure_gate_supervision * hctf_structure_gate_loss_value
            ),
        })

    loss_dict = {
        "total_loss": total_loss.detach(),
        "base_objective": base_losses["objective"].detach(),
        "final_objective": final_losses["objective"].detach(),

        "base_stress": base_losses["stress"].detach(),
        "base_rd_stress": base_losses["rd_stress"].detach(),
        "base_cosine_stress": base_losses["cosine_stress"].detach(),
        "base_intra": base_losses["intra"].detach(),
        "base_inter": base_losses["inter"].detach(),
        "base_pair": base_losses["pair"].detach(),

        "refined_stress": final_losses["stress"].detach(),
        "refined_rd_stress": final_losses["rd_stress"].detach(),
        "refined_cosine_stress": final_losses["cosine_stress"].detach(),
        "refined_intra": final_losses["intra"].detach(),
        "refined_inter": final_losses["inter"].detach(),
        "refined_pair": final_losses["pair"].detach(),

        "caid_loss": caid_loss.detach(),
        "caid_weight": torch.tensor(caid_weight, device=base_pos.device),
        "cpsg_total_loss": cpsg_total_loss.detach(),
        "cpsg_proto_loss": cpsg_proto_loss.detach(),
        "cpsg_sil_loss": cpsg_sil_loss.detach(),
        "cpsg_sep_loss": cpsg_sep_loss.detach(),
        "cpsg_weight": torch.tensor(cpsg_weight, device=base_pos.device),
        "pmds_anchor_loss": pmds_anchor_loss_value.detach(),
        "pmds_anchor_weight": torch.tensor(pmds_anchor_weight, device=base_pos.device),
        "crossing_loss": crossing_loss_value.detach(),
        "crossing_weight": torch.tensor(crossing_weight, device=base_pos.device),
        "train_silhouette_loss": train_silhouette_loss_value.detach(),
        "train_silhouette_weight": torch.tensor(train_silhouette_weight, device=base_pos.device),
        "hctf_readability_loss": hctf_readability_loss_value.detach(),
        "hctf_readability_weight": torch.tensor(
            hctf_readability_weight, device=base_pos.device
        ),
        "hctf_angular_loss": hctf_angular_loss_value.detach(),
        "hctf_crossing_angle_loss": hctf_crossing_angle_loss_value.detach(),
        "hctf_clearance_loss": hctf_clearance_loss_value.detach(),
        "hctf_neighborhood_loss": hctf_neighborhood_loss_value.detach(),
        "anchor_loss": anchor_loss.detach(),
        "delta_reg_loss": delta_reg_loss.detach(),
        "balance_loss": balance_loss.detach(),
        "paretofield_objective_loss": paretofield_objective_loss.detach(),
        "paretofield_regret_loss": paretofield_regret_loss.detach(),
        "paretofield_teacher_loss": paretofield_teacher_loss.detach(),
        "paretofield_anchor_loss": paretofield_anchor_loss.detach(),
        "paretofield_community_confidence": (
            paretofield_community_confidence.mean().detach()
            if paretofield_community_confidence is not None
            else torch.tensor(0.0, device=refined_pos.device)
        ),
        "paretofield_relation_uncertainty": (
            paretofield_relation_uncertainty.mean().detach()
            if paretofield_relation_uncertainty is not None
            else torch.tensor(0.0, device=refined_pos.device)
        ),
        "paretofield_edge_pair_count": torch.tensor(
            float(paretofield_edge_pair_count), device=refined_pos.device
        ),
        "hctf_dominance_loss": hctf_dominance_loss_value.detach(),
        "hctf_anchor_loss": hctf_anchor_loss_value.detach(),
        "hctf_max_regret": hctf_max_regret_value.detach(),
        "hctf_tail_risk": hctf_tail_risk_value.detach(),
        "hctf_crossing_barrier": hctf_crossing_barrier_value.detach(),
        "hctf_exact_gate_loss": hctf_exact_gate_loss_value.detach(),
        "hctf_exact_gate_target": hctf_exact_gate_target_value.detach(),
        "hctf_exact_gate_unsafe": hctf_exact_gate_unsafe_value.detach(),
        "hctf_exact_gate_regret": hctf_exact_gate_regret_value.detach(),
        "hctf_structure_gate_cost": hctf_structure_gate_cost_value.detach(),
        "hctf_structure_gate_loss": hctf_structure_gate_loss_value.detach(),
        "hctf_structure_gate_target": hctf_structure_gate_target_value.detach(),
        "hctf_structure_gate_unsafe": hctf_structure_gate_unsafe_value.detach(),
        "hctf_structure_gate_regret": hctf_structure_gate_regret_value.detach(),
    }

    if hctf_teacher_pos is not None:
        loss_dict["_hctf_teacher_pos"] = hctf_teacher_pos.detach()

    if getattr(args, "use_paretofield", False):
        loss_dict["_paretofield_conflict_losses"] = (
            paretofield_field_loss,
            (
                args.lambda_paretofield * paretofield_objective_loss
                + args.paretofield_regret_weight * paretofield_regret_loss
            ),
        )
        loss_dict["_paretofield_candidates"] = paretofield_reference_positions
        loss_dict["_paretofield_guard_stats"] = paretofield_guard_stats
        loss_dict["_paretofield_inference_steps"] = paretofield_primary_steps

    return total_loss, base_pos, refined_pos, loss_dict


def hctf_baseline_teacher_requested(args):
    """Whether an HCTF loss needs the bypassed backbone coordinates."""
    return bool(
        getattr(args, "use_hctf", False)
        and any(
            float(getattr(args, name, 0.0)) > 0.0
            for name in (
                "lambda_hctf_dominance",
                "lambda_hctf_anchor",
                "lambda_hctf_crossing_barrier",
                "lambda_hctf_exact_gate",
            )
        )
        and (
            getattr(args, "freeze_backbone_for_hctf", False)
            or getattr(args, "hctf_version", "v1") == "v2"
        )
    )


def hctf_teacher_requested(args):
    dominance_teacher = hctf_baseline_teacher_requested(args)
    structure_teacher = bool(
        getattr(args, "use_hctf", False)
        and getattr(args, "hctf_adaptive_structure_gate", False)
        and float(getattr(args, "lambda_hctf_structure_gate_supervision", 0.0)) > 0.0
        and getattr(args, "freeze_backbone_for_hctf", False)
    )
    return dominance_teacher or structure_teacher


def configure_hctf_training_modes(deepgd_model, args):
    """Freeze backbone state (including BatchNorm) for every frozen control."""
    if not (
        getattr(args, "use_hctf", False)
        and getattr(args, "freeze_backbone_for_hctf", False)
    ):
        return
    deepgd_model.eval()
    if getattr(args, "freeze_hctf_proposal_for_risk_gate", False):
        if deepgd_model.hctf_exact_risk_gate is not None:
            deepgd_model.hctf_exact_risk_gate.train()
    elif getattr(args, "freeze_hctf_block_for_conflict_head", False):
        if deepgd_model.hctf_conflict_head is not None:
            deepgd_model.hctf_conflict_head.train()
        if deepgd_model.hctf_graph_size_gate is not None:
            deepgd_model.hctf_graph_size_gate.train()
    else:
        deepgd_model.hctf_block.train()
        stage_adapters = getattr(deepgd_model, "hctf_stage_adapters", None)
        if stage_adapters is not None:
            stage_adapters.train()
        for module_name in (
            "hctf_coordinate_projection",
            "hctf_coordinate_trust",
            "hctf_coordinate_equivariant",
            "hctf_exact_risk_gate",
            "hctf_conflict_head",
        ):
            module = getattr(deepgd_model, module_name, None)
            if module is not None:
                module.train()


def stress_protected_crossing_gradients(stress_grads, crossing_grads, target_ratio):
    """Rescale a crossing gradient after removing its Stress-opposing part.

    The returned direction has nonnegative Euclidean dot product with the
    Stress gradient and norm at most ``target_ratio * ||stress_grad||``.
    This is a parameter-gradient rule, not a guarantee about Adam's step or
    exact validation metrics.
    """
    if target_ratio <= 0.0:
        raise ValueError("target_ratio must be positive")
    stress_sq = sum(
        (grad.detach().double().square().sum() for grad in stress_grads if grad is not None),
        start=0.0,
    )
    crossing_sq = sum(
        (grad.detach().double().square().sum() for grad in crossing_grads if grad is not None),
        start=0.0,
    )
    if not torch.isfinite(torch.as_tensor(stress_sq)) or not torch.isfinite(torch.as_tensor(crossing_sq)):
        raise FloatingPointError("non-finite Stress/crossing gradient norm")
    if stress_sq <= 0.0 or crossing_sq <= 0.0:
        return [None if grad is None else torch.zeros_like(grad) for grad in crossing_grads]
    dot = sum(
        (
            (stress.detach().double() * crossing.detach().double()).sum()
            for stress, crossing in zip(stress_grads, crossing_grads)
            if stress is not None and crossing is not None
        ),
        start=0.0,
    )
    if not torch.isfinite(torch.as_tensor(dot)):
        raise FloatingPointError("non-finite Stress/crossing gradient dot product")
    projection = min(float(dot / stress_sq), 0.0)
    projected = [
        None if crossing is None else (
            crossing.detach()
            - (projection * stress.detach() if stress is not None else 0.0)
        )
        for stress, crossing in zip(stress_grads, crossing_grads)
    ]
    projected_sq = sum(
        (grad.double().square().sum() for grad in projected if grad is not None),
        start=0.0,
    )
    if projected_sq <= 0.0:
        return [None if grad is None else torch.zeros_like(grad) for grad in crossing_grads]
    scale = float(target_ratio) * math.sqrt(float(stress_sq / projected_sq))
    return [None if grad is None else grad * scale for grad in projected]


def hctf_component_guard_gradients(pos, batch, args, parameters):
    """Lower-is-better surrogate gradients used by the optional step guard."""
    if hasattr(batch, "community_label"):
        labels = batch.community_label.long().to(pos.device)
    else:
        labels = build_batch_community_labels(
            edge_index=batch.edge_index.long(),
            batch_vector=batch.batch.long(),
            num_nodes=int(pos.size(0)),
        )
    ratios = parse_float_list(getattr(args, "hard_crossing_ratios", "0.40,0.30,0.20,0.10"))
    ratios = tuple(value / sum(ratios) for value in ratios)
    with torch.no_grad():
        first_edges, second_edges = select_metric_aligned_edge_pairs(
            pos=pos,
            edge_index=batch.edge_index.long(),
            batch_index=batch.batch.long(),
            max_pairs=int(args.hctf_dominance_max_pairs),
            exact_max_edges=int(args.hctf_crossing_exact_max_edges),
            oversample_factor=args.crossing_oversample_factor,
            temperature=args.crossing_temperature,
            ratios=ratios,
        )
    crossing_count, crossing_norm = crossing_objectives_from_edge_pairs_per_graph(
        pos=pos,
        first_edges=first_edges,
        second_edges=second_edges,
        edge_index=batch.edge_index.long(),
        batch_index=batch.batch.long(),
        temperature=args.crossing_temperature,
        margin=args.crossing_margin,
    )
    components = hctf_graphwise_component_objectives(
        pos=pos,
        batch=batch,
        community_labels=labels,
        crossing_count_per_graph=crossing_count,
        crossing_norm_per_graph=crossing_norm,
    ).mean(dim=0)
    gradients = []
    for component_index in (0, 1, 2, 4, 5, 6, 7, 8, 9):
        if not components[component_index].requires_grad:
            gradients.append(np.zeros(sum(parameter.numel() for parameter in parameters)))
            continue
        partial = torch.autograd.grad(
            components[component_index], parameters,
            retain_graph=True, allow_unused=True,
        )
        gradients.append(np.concatenate([
            (gradient if gradient is not None else torch.zeros_like(parameter))
            .detach().reshape(-1).float().cpu().numpy()
            for parameter, gradient in zip(parameters, partial)
        ]))
    return gradients


def run_train_epoch(epoch, loader, deepgd_model, refiner, optimizer, stress_metric, device, args):
    if os.environ.get("HCTF_AUTOGRAD_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    args.current_epoch = epoch
    args._is_training_step = True
    args._hctf_teacher_enabled = hctf_teacher_requested(args)

    deepgd_model.train()
    if (
        getattr(args, "use_paretofield", False)
        and getattr(args, "freeze_deepgd_for_paretofield", False)
    ):
        deepgd_model.eval()
        deepgd_model.paretofield.train()
    configure_hctf_training_modes(deepgd_model, args)
    if refiner is not None:
        if getattr(args, "freeze_deepgd_for_paretofield", False):
            refiner.eval()
        else:
            refiner.train()

    sums = {
        "total_loss": 0.0,
        "caid_loss": 0.0,
        "cpsg_total_loss": 0.0,
        "cpsg_proto_loss": 0.0,
        "cpsg_sil_loss": 0.0,
        "cpsg_sep_loss": 0.0,
        "pmds_anchor_loss": 0.0,
        "crossing_loss": 0.0,
        "train_silhouette_loss": 0.0,
        "base_stress": 0.0,
        "base_rd_stress": 0.0,
        "base_cosine_stress": 0.0,
        "refined_stress": 0.0,
        "refined_rd_stress": 0.0,
        "refined_cosine_stress": 0.0,
        "base_intra": 0.0,
        "refined_intra": 0.0,
        "base_inter": 0.0,
        "refined_inter": 0.0,
        "paretofield_objective_loss": 0.0,
        "paretofield_regret_loss": 0.0,
        "paretofield_teacher_loss": 0.0,
        "paretofield_anchor_loss": 0.0,
        "paretofield_community_confidence": 0.0,
        "paretofield_relation_uncertainty": 0.0,
        "paretofield_edge_pair_count": 0.0,
        "hctf_dominance_loss": 0.0,
        "hctf_anchor_loss": 0.0,
        "hctf_max_regret": 0.0,
        "hctf_tail_risk": 0.0,
        "hctf_crossing_barrier": 0.0,
        "hctf_exact_gate_loss": 0.0,
        "hctf_exact_gate_target": 0.0,
        "hctf_exact_gate_unsafe": 0.0,
        "hctf_exact_gate_regret": 0.0,
        "hctf_structure_gate_cost": 0.0,
        "hctf_structure_gate_loss": 0.0,
        "hctf_structure_gate_target": 0.0,
        "hctf_structure_gate_unsafe": 0.0,
        "hctf_structure_gate_regret": 0.0,
        "hctf_readability_loss": 0.0,
        "hctf_readability_weight": 0.0,
        "hctf_angular_loss": 0.0,
        "hctf_crossing_angle_loss": 0.0,
        "hctf_clearance_loss": 0.0,
        "hctf_neighborhood_loss": 0.0,
    }
    count = 0
    gradient_sums = {
        "gradient_cosine": 0.0,
        "gradient_conflict": 0.0,
        "field_gradient_norm": 0.0,
        "pareto_gradient_norm": 0.0,
    }
    gradient_count = 0
    hctf_gradient_sum = 0.0
    backbone_gradient_sum = 0.0
    guard_retained_sum = 0.0
    guard_projected_count = 0
    guard_step_count = 0
    guard_max_violation = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch_id, batch in enumerate(pbar):
        if args.max_train_batches > 0 and batch_id >= args.max_train_batches:
            break

        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        projected_crossing_ratio = float(
            getattr(args, "hctf_crossing_projected_ratio", 0.0)
        )
        anchor_trace_path = os.environ.get("HCTF_ANCHOR_TRACE_CSV")
        gradient_probe = {} if projected_crossing_ratio > 0.0 or anchor_trace_path else None
        total_loss, base_pos, refined_pos, loss_dict = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
            gradient_probe=gradient_probe,
        )
        guard_parameters = []
        guard_before = []
        guard_gradients = []
        if getattr(args, "hctf_component_gradient_guard", False):
            guard_parameters = [
                parameter for parameter in deepgd_model.hctf_block.parameters()
                if parameter.requires_grad
            ]
            guard_before = [parameter.detach().clone() for parameter in guard_parameters]
            guard_gradients = hctf_component_guard_gradients(
                base_pos, batch, args, guard_parameters
            )

        if anchor_trace_path:
            init_pos_trace = gradient_probe["_init_pos"]
            raw_pos_trace = gradient_probe["_base_pos_raw"].detach()
            graph_ids = batch.batch.long()
            row_fields = (
                "epoch", "batch_id", "graph_name", "num_nodes", "anchor_squared",
                "init_pos_rms", "refined_pos_rms", "raw_centroid_norm",
                "raw_centered_rms", "rescale_factor", "amplified_centroid_norm",
                "refined_centroid_norm", "centered_anchor_squared",
            )
            os.makedirs(os.path.dirname(os.path.abspath(anchor_trace_path)), exist_ok=True)
            write_header = not os.path.exists(anchor_trace_path)
            with open(anchor_trace_path, "a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=row_fields)
                if write_header:
                    writer.writeheader()
                for graph_id in range(int(batch.num_graphs)):
                    mask = graph_ids == graph_id
                    raw_graph_pos = raw_pos_trace[mask]
                    raw_centroid = raw_graph_pos.mean(dim=0)
                    edge_mask = graph_ids[batch.perm_index[0].long()] == graph_id
                    graph_pairs = batch.perm_index[:, edge_mask].long()
                    relative_distances = (
                        (raw_pos_trace[graph_pairs[0]] - raw_pos_trace[graph_pairs[1]])
                        .norm(dim=1) / (batch.apsp_attr[edge_mask] + 1e-9)
                    )
                    rescale_factor = relative_distances.square().sum() / (
                        relative_distances.sum() + 1e-9
                    )
                    refined_graph_pos = refined_pos[mask].detach()
                    init_graph_pos = init_pos_trace[mask]
                    writer.writerow({
                        "epoch": epoch,
                        "batch_id": batch_id,
                        "graph_name": str(batch.name[graph_id]),
                        "num_nodes": int(mask.sum()),
                        "anchor_squared": float(
                            (refined_pos[mask].detach() - init_pos_trace[mask])
                            .square().sum(dim=1).mean()
                        ),
                        "init_pos_rms": float(init_pos_trace[mask].square().sum(dim=1).mean().sqrt()),
                        "refined_pos_rms": float(refined_pos[mask].detach().square().sum(dim=1).mean().sqrt()),
                        "raw_centroid_norm": float(raw_centroid.norm()),
                        "raw_centered_rms": float(
                            (raw_graph_pos - raw_centroid).square().sum(dim=1).mean().sqrt()
                        ),
                        "rescale_factor": float(rescale_factor),
                        "amplified_centroid_norm": float(
                            raw_centroid.norm() / (rescale_factor + 1e-9)
                        ),
                        "refined_centroid_norm": float(
                            refined_graph_pos.mean(dim=0).norm()
                        ),
                        "centered_anchor_squared": float((
                            (refined_graph_pos - refined_graph_pos.mean(dim=0))
                            - (init_graph_pos - init_graph_pos.mean(dim=0))
                        ).square().sum(dim=1).mean()),
                    })

        if (
            getattr(args, "use_hctf", False)
            and getattr(args, "freeze_backbone_for_hctf", False)
            and not bool(torch.isfinite(total_loss).item())
        ):
            nonfinite_losses = [
                name for name, value in loss_dict.items()
                if isinstance(value, torch.Tensor)
                and value.numel() == 1
                and not bool(torch.isfinite(value).item())
            ]
            raise FloatingPointError(
                f"non-finite HCTF loss at epoch={epoch}, batch={batch_id}; "
                f"nonfinite_components={nonfinite_losses[:12]}"
            )

        hctf_crossing_parameters = []
        original_crossing_grads = []
        corrected_crossing_grads = []
        if projected_crossing_ratio > 0.0:
            hctf_crossing_parameters = [
                parameter for name, parameter in deepgd_model.named_parameters()
                if is_hctf_parameter(name) and parameter.requires_grad
            ]
            crossing_objective = gradient_probe["crossing"]
            stress_objective = gradient_probe["stress"]
            if crossing_objective.requires_grad and stress_objective.requires_grad:
                stress_grads = torch.autograd.grad(
                    stress_objective,
                    hctf_crossing_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                original_crossing_grads = torch.autograd.grad(
                    crossing_objective,
                    hctf_crossing_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                corrected_crossing_grads = stress_protected_crossing_gradients(
                    stress_grads, original_crossing_grads, projected_crossing_ratio
                )

        conflict_losses = loss_dict.get("_paretofield_conflict_losses")
        conflict_method = getattr(args, "paretofield_conflict_method", "weighted")
        if conflict_losses is not None and conflict_method in {"pcgrad", "config"}:
            total_loss.backward(retain_graph=True)
            conflict_stats = apply_conflict_free_gradients(
                losses=conflict_losses,
                parameters=list(deepgd_model.paretofield.parameters()),
                method=conflict_method,
            )
            for key in gradient_sums:
                gradient_sums[key] += float(conflict_stats[key])
            gradient_count += 1
        else:
            total_loss.backward()

        for parameter, original, corrected in zip(
            hctf_crossing_parameters, original_crossing_grads, corrected_crossing_grads
        ):
            if original is None and corrected is None:
                continue
            correction = (
                (corrected if corrected is not None else 0.0)
                - (original if original is not None else 0.0)
            )
            if parameter.grad is None:
                parameter.grad = correction.clone()
            else:
                parameter.grad.add_(correction)

        if getattr(args, "use_hctf", False):
            hctf_grad_sq = 0.0
            backbone_grad_sq = 0.0
            for name, parameter in deepgd_model.named_parameters():
                if parameter.grad is None:
                    continue
                value = float(parameter.grad.detach().square().sum().item())
                if is_hctf_parameter(name):
                    hctf_grad_sq += value
                elif not name.startswith("paretofield."):
                    backbone_grad_sq += value
            hctf_gradient_sum += math.sqrt(hctf_grad_sq)
            backbone_gradient_sum += math.sqrt(backbone_grad_sq)

        params = list(deepgd_model.parameters())
        if refiner is not None:
            params += list(refiner.parameters())

        if getattr(args, "use_hctf", False) and getattr(args, "freeze_backbone_for_hctf", False):
            nonfinite_gradients = [
                name for name, parameter in deepgd_model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().item())
            ]
            if nonfinite_gradients:
                raise FloatingPointError(
                    f"non-finite HCTF gradient at epoch={epoch}, batch={batch_id}; "
                    f"parameters={nonfinite_gradients[:12]}"
                )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

        optimizer.step()
        if guard_parameters:
            proposed_step = np.concatenate([
                (parameter.detach() - before).reshape(-1).float().cpu().numpy()
                for parameter, before in zip(guard_parameters, guard_before)
            ])
            corrected_step, guard_stats = project_step_to_guard_halfspaces(
                proposed_step, guard_gradients
            )
            offset = 0
            with torch.no_grad():
                for parameter, before in zip(guard_parameters, guard_before):
                    count_values = parameter.numel()
                    corrected_piece = torch.from_numpy(
                        corrected_step[offset:offset + count_values].copy()
                    ).to(device=parameter.device, dtype=parameter.dtype).view_as(parameter)
                    parameter.copy_(before + corrected_piece)
                    offset += count_values
            guard_retained_sum += guard_stats["retained_norm"]
            guard_projected_count += guard_stats["retained_norm"] < 0.999
            guard_step_count += 1
            guard_max_violation = max(
                guard_max_violation, guard_stats["max_guard_dot"]
            )

        for key in sums:
            sums[key] += float(loss_dict[key].item())

        count += 1

        pbar.set_postfix({
            "loss": sums["total_loss"] / max(count, 1),
            "caid": sums["caid_loss"] / max(count, 1),
            "cpsg": sums["cpsg_total_loss"] / max(count, 1),
            "anc": sums["pmds_anchor_loss"] / max(count, 1),
            "xcr": sums["crossing_loss"] / max(count, 1),
            "sil": sums["train_silhouette_loss"] / max(count, 1),
            "rd": sums["refined_rd_stress"] / max(count, 1),
            "cos": sums["refined_cosine_stress"] / max(count, 1),
            "base_s": sums["base_stress"] / max(count, 1),
            "ref_s": sums["refined_stress"] / max(count, 1),
            "pf": sums["paretofield_objective_loss"] / max(count, 1),
            "pfr": sums["paretofield_regret_loss"] / max(count, 1),
            "pfu": sums["paretofield_relation_uncertainty"] / max(count, 1),
            "ep": sums["paretofield_edge_pair_count"] / max(count, 1),
            "hdom": sums["hctf_dominance_loss"] / max(count, 1),
            "hreg": sums["hctf_max_regret"] / max(count, 1),
            "hgate": sums["hctf_exact_gate_loss"] / max(count, 1),
            "hread": sums["hctf_readability_loss"] / max(count, 1),
        })

        del total_loss, loss_dict, batch
        maybe_cleanup_memory(args, batch_id=batch_id)

    result = {
        "train_loss": sums["total_loss"] / max(count, 1),
        "train_caid_loss": sums["caid_loss"] / max(count, 1),
        "train_cpsg_loss": sums["cpsg_total_loss"] / max(count, 1),
        "train_cpsg_proto": sums["cpsg_proto_loss"] / max(count, 1),
        "train_cpsg_sil": sums["cpsg_sil_loss"] / max(count, 1),
        "train_cpsg_sep": sums["cpsg_sep_loss"] / max(count, 1),
        "train_pmds_anchor": sums["pmds_anchor_loss"] / max(count, 1),
        "train_crossing_loss": sums["crossing_loss"] / max(count, 1),
        "train_silhouette_loss": sums["train_silhouette_loss"] / max(count, 1),
        "train_base_stress": sums["base_stress"] / max(count, 1),
        "train_base_rd_stress": sums["base_rd_stress"] / max(count, 1),
        "train_base_cosine_stress": sums["base_cosine_stress"] / max(count, 1),
        "train_refined_stress": sums["refined_stress"] / max(count, 1),
        "train_refined_rd_stress": sums["refined_rd_stress"] / max(count, 1),
        "train_refined_cosine_stress": sums["refined_cosine_stress"] / max(count, 1),
        "train_base_intra": sums["base_intra"] / max(count, 1),
        "train_refined_intra": sums["refined_intra"] / max(count, 1),
        "train_base_inter": sums["base_inter"] / max(count, 1),
        "train_refined_inter": sums["refined_inter"] / max(count, 1),
        "train_paretofield_objective": sums["paretofield_objective_loss"] / max(count, 1),
        "train_paretofield_regret": sums["paretofield_regret_loss"] / max(count, 1),
        "train_paretofield_teacher": sums["paretofield_teacher_loss"] / max(count, 1),
        "train_paretofield_anchor": sums["paretofield_anchor_loss"] / max(count, 1),
        "train_paretofield_community_confidence": sums["paretofield_community_confidence"] / max(count, 1),
        "train_paretofield_relation_uncertainty": sums["paretofield_relation_uncertainty"] / max(count, 1),
        "train_paretofield_edge_pair_count": sums["paretofield_edge_pair_count"] / max(count, 1),
        "train_hctf_dominance": sums["hctf_dominance_loss"] / max(count, 1),
        "train_hctf_anchor": sums["hctf_anchor_loss"] / max(count, 1),
        "train_hctf_max_regret": sums["hctf_max_regret"] / max(count, 1),
        "train_hctf_tail_risk": sums["hctf_tail_risk"] / max(count, 1),
        "train_hctf_crossing_barrier": sums["hctf_crossing_barrier"] / max(count, 1),
        "train_hctf_exact_gate_loss": sums["hctf_exact_gate_loss"] / max(count, 1),
        "train_hctf_exact_gate_target": sums["hctf_exact_gate_target"] / max(count, 1),
        "train_hctf_exact_gate_unsafe": sums["hctf_exact_gate_unsafe"] / max(count, 1),
        "train_hctf_exact_gate_regret": sums["hctf_exact_gate_regret"] / max(count, 1),
        "train_hctf_structure_gate_cost": sums["hctf_structure_gate_cost"] / max(count, 1),
        "train_hctf_structure_gate_loss": sums["hctf_structure_gate_loss"] / max(count, 1),
        "train_hctf_structure_gate_target": sums["hctf_structure_gate_target"] / max(count, 1),
        "train_hctf_structure_gate_unsafe": sums["hctf_structure_gate_unsafe"] / max(count, 1),
        "train_hctf_structure_gate_regret": sums["hctf_structure_gate_regret"] / max(count, 1),
        "train_hctf_readability": sums["hctf_readability_loss"] / max(count, 1),
        "train_hctf_readability_weight": sums["hctf_readability_weight"] / max(count, 1),
        "train_hctf_angular": sums["hctf_angular_loss"] / max(count, 1),
        "train_hctf_crossing_angle": sums["hctf_crossing_angle_loss"] / max(count, 1),
        "train_hctf_clearance": sums["hctf_clearance_loss"] / max(count, 1),
        "train_hctf_neighborhood": sums["hctf_neighborhood_loss"] / max(count, 1),
    }
    for key, value in gradient_sums.items():
        result[f"train_{key}"] = value / max(gradient_count, 1)
    result["train_hctf_gradient_norm"] = hctf_gradient_sum / max(count, 1)
    result["train_backbone_gradient_norm"] = backbone_gradient_sum / max(count, 1)
    if guard_step_count:
        result["train_hctf_guard_retained"] = guard_retained_sum / guard_step_count
        result["train_hctf_guard_projected_batches"] = guard_projected_count
        result["train_hctf_guard_max_first_order_violation"] = guard_max_violation
    return result


@torch.no_grad()
def run_validation(epoch, loader, deepgd_model, refiner, stress_metric, device, args):
    args.current_epoch = epoch
    args._is_training_step = False
    args._hctf_teacher_enabled = hctf_teacher_requested(args)

    deepgd_model.eval()
    if refiner is not None:
        refiner.eval()

    sums = {
        "total_loss": 0.0,
        "caid_loss": 0.0,
        "cpsg_total_loss": 0.0,
        "cpsg_proto_loss": 0.0,
        "cpsg_sil_loss": 0.0,
        "cpsg_sep_loss": 0.0,
        "pmds_anchor_loss": 0.0,
        "crossing_loss": 0.0,
        "train_silhouette_loss": 0.0,
        "base_objective": 0.0,
        "final_objective": 0.0,
        "base_stress": 0.0,
        "base_rd_stress": 0.0,
        "base_cosine_stress": 0.0,
        "refined_stress": 0.0,
        "refined_rd_stress": 0.0,
        "refined_cosine_stress": 0.0,
        "base_intra": 0.0,
        "refined_intra": 0.0,
        "base_inter": 0.0,
        "refined_inter": 0.0,
        "base_pair": 0.0,
        "refined_pair": 0.0,
        "paretofield_objective_loss": 0.0,
        "paretofield_regret_loss": 0.0,
        "paretofield_teacher_loss": 0.0,
        "paretofield_anchor_loss": 0.0,
        "paretofield_community_confidence": 0.0,
        "paretofield_relation_uncertainty": 0.0,
        "paretofield_edge_pair_count": 0.0,
        "hctf_dominance_loss": 0.0,
        "hctf_anchor_loss": 0.0,
        "hctf_max_regret": 0.0,
        "hctf_tail_risk": 0.0,
        "hctf_crossing_barrier": 0.0,
        "hctf_exact_gate_loss": 0.0,
        "hctf_exact_gate_target": 0.0,
        "hctf_exact_gate_unsafe": 0.0,
        "hctf_exact_gate_regret": 0.0,
        "hctf_structure_gate_cost": 0.0,
        "hctf_structure_gate_loss": 0.0,
        "hctf_structure_gate_target": 0.0,
        "hctf_structure_gate_unsafe": 0.0,
        "hctf_structure_gate_regret": 0.0,
        "hctf_readability_loss": 0.0,
        "hctf_readability_weight": 0.0,
        "hctf_angular_loss": 0.0,
        "hctf_crossing_angle_loss": 0.0,
        "hctf_clearance_loss": 0.0,
        "hctf_neighborhood_loss": 0.0,
    }
    count = 0
    exact_validation = args.select_by == "val_hctf_exact"
    exact_candidate_sums = {metric: 0.0 for metric in HCTF_EXACT_METRICS}
    exact_teacher_sums = {metric: 0.0 for metric in HCTF_EXACT_METRICS}
    exact_graph_count = 0

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")

    for batch_id, batch in enumerate(pbar):
        if args.max_val_batches > 0 and batch_id >= args.max_val_batches:
            break

        batch = batch.to(device)

        _, candidate_pos, _, loss_dict = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        for key in sums:
            sums[key] += float(loss_dict[key].item())

        if exact_validation:
            teacher_pos = loss_dict.get("_hctf_teacher_pos")
            if teacher_pos is None:
                raise RuntimeError(
                    "val_hctf_exact requires the frozen HCTF teacher path"
                )
            candidate_metrics, candidate_count = hctf_exact_batch_metric_sums(
                candidate_pos,
                batch,
            )
            teacher_metrics, teacher_count = hctf_exact_batch_metric_sums(
                teacher_pos,
                batch,
            )
            if candidate_count != teacher_count:
                raise RuntimeError("candidate and teacher validation graph counts differ")
            for metric in HCTF_EXACT_METRICS:
                exact_candidate_sums[metric] += candidate_metrics[metric]
                exact_teacher_sums[metric] += teacher_metrics[metric]
            exact_graph_count += candidate_count

        count += 1

        pbar.set_postfix({
            "val_loss": sums["total_loss"] / max(count, 1),
            "caid": sums["caid_loss"] / max(count, 1),
            "cpsg": sums["cpsg_total_loss"] / max(count, 1),
            "anc": sums["pmds_anchor_loss"] / max(count, 1),
            "xcr": sums["crossing_loss"] / max(count, 1),
            "sil": sums["train_silhouette_loss"] / max(count, 1),
            "rd": sums["refined_rd_stress"] / max(count, 1),
            "cos": sums["refined_cosine_stress"] / max(count, 1),
            "base_s": sums["base_stress"] / max(count, 1),
            "ref_s": sums["refined_stress"] / max(count, 1),
            "pf": sums["paretofield_objective_loss"] / max(count, 1),
            "pfr": sums["paretofield_regret_loss"] / max(count, 1),
            "pfu": sums["paretofield_relation_uncertainty"] / max(count, 1),
            "ep": sums["paretofield_edge_pair_count"] / max(count, 1),
            "hdom": sums["hctf_dominance_loss"] / max(count, 1),
            "hreg": sums["hctf_max_regret"] / max(count, 1),
            "hgate": sums["hctf_exact_gate_loss"] / max(count, 1),
            "hread": sums["hctf_readability_loss"] / max(count, 1),
        })

        del loss_dict, batch
        maybe_cleanup_memory(args, batch_id=batch_id)

    result = {
        "val_loss": sums["total_loss"] / max(count, 1),
        "val_caid_loss": sums["caid_loss"] / max(count, 1),
        "val_cpsg_loss": sums["cpsg_total_loss"] / max(count, 1),
        "val_cpsg_proto": sums["cpsg_proto_loss"] / max(count, 1),
        "val_cpsg_sil": sums["cpsg_sil_loss"] / max(count, 1),
        "val_cpsg_sep": sums["cpsg_sep_loss"] / max(count, 1),
        "val_pmds_anchor": sums["pmds_anchor_loss"] / max(count, 1),
        "val_crossing_loss": sums["crossing_loss"] / max(count, 1),
        "val_train_silhouette_loss": sums["train_silhouette_loss"] / max(count, 1),
        "val_base_objective": sums["base_objective"] / max(count, 1),
        "val_final_objective": sums["final_objective"] / max(count, 1),

        "val_base_stress": sums["base_stress"] / max(count, 1),
        "val_base_rd_stress": sums["base_rd_stress"] / max(count, 1),
        "val_base_cosine_stress": sums["base_cosine_stress"] / max(count, 1),
        "val_refined_stress": sums["refined_stress"] / max(count, 1),
        "val_refined_rd_stress": sums["refined_rd_stress"] / max(count, 1),
        "val_refined_cosine_stress": sums["refined_cosine_stress"] / max(count, 1),

        "val_base_intra": sums["base_intra"] / max(count, 1),
        "val_refined_intra": sums["refined_intra"] / max(count, 1),

        "val_base_inter": sums["base_inter"] / max(count, 1),
        "val_refined_inter": sums["refined_inter"] / max(count, 1),

        "val_base_pair": sums["base_pair"] / max(count, 1),
        "val_refined_pair": sums["refined_pair"] / max(count, 1),
        "val_paretofield_objective": sums["paretofield_objective_loss"] / max(count, 1),
        "val_paretofield_regret": sums["paretofield_regret_loss"] / max(count, 1),
        "val_paretofield_teacher": sums["paretofield_teacher_loss"] / max(count, 1),
        "val_paretofield_anchor": sums["paretofield_anchor_loss"] / max(count, 1),
        "val_paretofield_community_confidence": sums["paretofield_community_confidence"] / max(count, 1),
        "val_paretofield_relation_uncertainty": sums["paretofield_relation_uncertainty"] / max(count, 1),
        "val_paretofield_edge_pair_count": sums["paretofield_edge_pair_count"] / max(count, 1),
        "val_hctf_dominance": sums["hctf_dominance_loss"] / max(count, 1),
        "val_hctf_anchor": sums["hctf_anchor_loss"] / max(count, 1),
        "val_hctf_max_regret": sums["hctf_max_regret"] / max(count, 1),
        "val_hctf_tail_risk": sums["hctf_tail_risk"] / max(count, 1),
        "val_hctf_crossing_barrier": sums["hctf_crossing_barrier"] / max(count, 1),
        "val_hctf_exact_gate_loss": sums["hctf_exact_gate_loss"] / max(count, 1),
        "val_hctf_exact_gate_target": sums["hctf_exact_gate_target"] / max(count, 1),
        "val_hctf_exact_gate_unsafe": sums["hctf_exact_gate_unsafe"] / max(count, 1),
        "val_hctf_exact_gate_regret": sums["hctf_exact_gate_regret"] / max(count, 1),
        "val_hctf_structure_gate_cost": sums["hctf_structure_gate_cost"] / max(count, 1),
        "val_hctf_structure_gate_loss": sums["hctf_structure_gate_loss"] / max(count, 1),
        "val_hctf_structure_gate_target": sums["hctf_structure_gate_target"] / max(count, 1),
        "val_hctf_structure_gate_unsafe": sums["hctf_structure_gate_unsafe"] / max(count, 1),
        "val_hctf_structure_gate_regret": sums["hctf_structure_gate_regret"] / max(count, 1),
        "val_hctf_readability": sums["hctf_readability_loss"] / max(count, 1),
        "val_hctf_readability_weight": sums["hctf_readability_weight"] / max(count, 1),
        "val_hctf_angular": sums["hctf_angular_loss"] / max(count, 1),
        "val_hctf_crossing_angle": sums["hctf_crossing_angle_loss"] / max(count, 1),
        "val_hctf_clearance": sums["hctf_clearance_loss"] / max(count, 1),
        "val_hctf_neighborhood": sums["hctf_neighborhood_loss"] / max(count, 1),
    }
    if exact_validation:
        result.update(
            hctf_exact_validation_summary(
                exact_candidate_sums,
                exact_teacher_sums,
                exact_graph_count,
            )
        )
    return result


def get_method_name(args, refined=False):
    mode = effective_cser_mode(args)

    if mode == "none":
        base = "DeepGD"
    elif mode == "same":
        base = "CAER-DeepGD"
    elif mode == "same_spectral":
        base = "CSER-Spectral-DeepGD"
    elif mode == "full":
        base = "CSER-Full-DeepGD"
    else:
        base = "CSER-DeepGD"

    if getattr(args, "use_hctf", False):
        base = "HCTF-" + base

    if getattr(args, "use_caid", False):
        base = base + "+CAID"

    if getattr(args, "use_care", False):
        base = base + "+CARE"

    if getattr(args, "use_topocser", False):
        base = base + "+Topo-CSER"

    if getattr(args, "use_rd_stress", False):
        base = base + "+RD-Stress"

    if getattr(args, "use_cosine_stress", False):
        base = base + "+Cosine-Stress"

    if getattr(args, "use_cpsg", False):
        if getattr(args, "use_macpsg", False):
            base = base + "+MA-CPSG"
        else:
            base = base + "+CPSG"

    if getattr(args, "lambda_pmds_anchor", 0.0) > 0:
        base = base + "+PMDSA"

    if getattr(args, "lambda_crossing", 0.0) > 0:
        base = base + "+SXL"

    if refined:
        if getattr(args, "use_refiner", False):
            base = base + "+Spectral-MoG-Refiner"
        if getattr(args, "use_paretofield", False):
            base = base + "+ParetoField"
            if (
                getattr(args, "paretofield_soft_community", False)
                or getattr(args, "paretofield_relation_consensus", False)
                or getattr(args, "paretofield_graph_preference_calibration", False)
                or getattr(args, "paretofield_regret_weight", 0.0) > 0
            ):
                base = base + "-Adaptive"
            if getattr(args, "paretofield_edge_pair_field", False):
                base = base + "-DualRelation"
            if getattr(args, "paretofield_safe_guard", False):
                base = base + "-Safe"
                if getattr(args, "paretofield_continuous_guard", False):
                    base = base + "Line"
                if getattr(args, "paretofield_exact_crossing_guard", False):
                    base = base + "Audit"
                if getattr(args, "paretofield_componentwise_guard", False):
                    base = base + "-AllMetric"
        return base

    return base


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]

    items = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            items.append(float(item))
    return items


def parse_string_list(value):
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value if str(v).strip()]

    items = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if item:
            items.append(item)
    return items


def local_deepgd_stress(pos, perm_index, apsp_attr, eps=1e-9):
    """
    DeepGD-style stress for one local graph.

    The official Stress metric sums relative squared shortest-path errors per
    graph, then averages across graphs. CG-RCC uses this local value only as a
    guard for accepting deterministic geometry calibration candidates.
    """
    if perm_index is None or apsp_attr is None or perm_index.numel() == 0:
        return 0.0

    perm_index = perm_index.long().to(pos.device)
    apsp_attr = apsp_attr.to(device=pos.device, dtype=pos.dtype).view(-1)

    src = perm_index[0]
    dst = perm_index[1]
    dist = torch.norm(pos[dst] - pos[src], p=2, dim=1)
    edge_stress = (dist.sub(apsp_attr).abs().div(apsp_attr.clamp_min(eps))).square()
    return float(edge_stress.sum().item())


def local_deepgd_stress_tensor(pos, perm_index, apsp_attr, eps=1e-9):
    if perm_index is None or apsp_attr is None or perm_index.numel() == 0:
        return pos.sum() * 0.0

    perm_index = perm_index.long().to(pos.device)
    apsp_attr = apsp_attr.to(device=pos.device, dtype=pos.dtype).view(-1)
    src = perm_index[0]
    dst = perm_index[1]
    dist = torch.norm(pos[dst] - pos[src], p=2, dim=1)
    edge_stress = (dist.sub(apsp_attr).abs().div(apsp_attr.clamp_min(eps))).square()
    return edge_stress.sum()


def local_rescale_by_stress(pos, perm_index, apsp_attr):
    if pos.size(0) <= 1 or perm_index is None or perm_index.numel() == 0:
        centered = pos - pos.mean(dim=0, keepdim=True)
        return centered / centered.std().clamp_min(1e-9)

    batch_index = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    return rescale_by_stress(
        pos=pos,
        apsp=apsp_attr.to(device=pos.device, dtype=pos.dtype).view(-1),
        edge_index=perm_index.long().to(pos.device),
        batch_index=batch_index,
    )


def mtpf_center_layout(pos, eps=1e-9):
    centered = pos - pos.mean(dim=0, keepdim=True)
    rms = torch.sqrt(centered.pow(2).sum(dim=1).mean()).clamp_min(eps)
    return centered / rms


def mtpf_procrustes_align(source, target, eps=1e-9):
    if source.size(0) <= 1:
        return target.clone()

    source_center = source.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    source_c = source - source_center
    target_c = target - target_center
    source_norm = torch.sqrt(source_c.pow(2).sum()).clamp_min(eps)
    target_norm = torch.sqrt(target_c.pow(2).sum()).clamp_min(eps)
    source_n = source_c / source_norm
    target_n = target_c / target_norm

    try:
        u, _, vh = torch.linalg.svd(source_n.t().matmul(target_n), full_matrices=False)
        rot = u.matmul(vh)
        if torch.det(rot) < 0:
            u = u.clone()
            u[:, -1] *= -1
            rot = u.matmul(vh)
        return source_n.matmul(rot) * target_norm + target_center
    except RuntimeError:
        return mtpf_center_layout(source).to(target.device, target.dtype) + target_center


def mtpf_build_local_graph_records(batch):
    device = batch.batch.device
    batch_vector_cpu = batch.batch.long().detach().cpu()
    edge_index_cpu = batch.edge_index.long().detach().cpu()
    perm_index_cpu = batch.perm_index.long().detach().cpu()
    apsp_attr_cpu = batch.apsp_attr.detach().cpu().view(-1)

    records = []
    for gid in batch_vector_cpu.unique(sorted=True).tolist():
        node_mask = batch_vector_cpu == gid
        global_nodes_cpu = torch.where(node_mask)[0]
        if int(global_nodes_cpu.numel()) == 0:
            continue

        node_set = set(global_nodes_cpu.tolist())
        mapping = {int(g): i for i, g in enumerate(global_nodes_cpu.tolist())}

        local_edges = []
        for u, v in edge_index_cpu.t().tolist():
            if u in node_set and v in node_set:
                local_edges.append([mapping[int(u)], mapping[int(v)]])
        if len(local_edges) == 0:
            local_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()

        local_perm = []
        local_apsp = []
        for edge_id, (u, v) in enumerate(perm_index_cpu.t().tolist()):
            if u in node_set and v in node_set:
                local_perm.append([mapping[int(u)], mapping[int(v)]])
                local_apsp.append(float(apsp_attr_cpu[edge_id].item()))
        if len(local_perm) == 0:
            local_perm_index = torch.empty((2, 0), dtype=torch.long)
            local_apsp_attr = torch.empty(0, dtype=torch.float32)
        else:
            local_perm_index = torch.tensor(local_perm, dtype=torch.long).t().contiguous()
            local_apsp_attr = torch.tensor(local_apsp, dtype=torch.float32)

        records.append(
            {
                "graph_id": int(gid),
                "global_nodes_cpu": global_nodes_cpu,
                "global_nodes_device": global_nodes_cpu.to(device),
                "edge_index": local_edge_index,
                "perm_index": local_perm_index,
                "apsp_attr": local_apsp_attr,
                "num_nodes": int(global_nodes_cpu.numel()),
            }
        )

    return records


def mtpf_networkx_layout(num_nodes, edge_index, method, seed=42, iterations=100):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    for u, v in edge_index.long().detach().cpu().t().tolist():
        u = int(u)
        v = int(v)
        if u != v:
            graph.add_edge(u, v)

    if method == "fr":
        pos_dict = nx.spring_layout(graph, seed=int(seed), iterations=int(iterations))
    elif method == "kk":
        pos_dict = nx.kamada_kawai_layout(graph)
    elif method == "spectral":
        pos_dict = nx.spectral_layout(graph)
    else:
        raise ValueError(f"Unknown MTPF teacher layout: {method}")

    pos_np = np.zeros((int(num_nodes), 2), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    for node_id in range(int(num_nodes)):
        if node_id in pos_dict:
            pos_np[node_id] = np.asarray(pos_dict[node_id], dtype=np.float32)
        else:
            pos_np[node_id] = rng.normal(size=2).astype(np.float32)
    return torch.tensor(pos_np, dtype=torch.float32)


def mtpf_mds_from_distance_matrix(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=np.float64)
    n = int(dist.shape[0])
    if n <= 1:
        return torch.zeros((n, 2), dtype=torch.float32)

    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.maximum(0.0, 0.5 * (dist + dist.T))
    np.fill_diagonal(dist, 0.0)

    dist2 = dist ** 2
    center = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * center @ dist2 @ center
    gram = np.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:2]
        eigvals = np.maximum(eigvals[order], 0.0)
        pos = eigvecs[:, order] * np.sqrt(eigvals)[None, :]
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        pos = rng.normal(size=(n, 2))

    if pos.shape[1] < 2:
        pos = np.pad(pos, ((0, 0), (0, 2 - pos.shape[1])))

    pos = pos[:, :2]
    pos = pos - pos.mean(axis=0, keepdims=True)
    scale = np.std(pos)
    if scale > 1e-9:
        pos = pos / scale
    return torch.tensor(pos.astype(np.float32), dtype=torch.float32)


def mtpf_cosine_context_layout(
    num_nodes,
    edge_index,
    perm_index,
    apsp_attr,
    walk_steps=4,
    walk_decay=0.75,
    normalize="mean",
    eps=1e-9,
):
    n = int(num_nodes)
    if n <= 1:
        return torch.zeros((n, 2), dtype=torch.float32)

    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edge_index.long().detach().cpu().t().tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        A[u, v] = 1.0
        A[v, u] = 1.0

    deg = A.sum(axis=1)
    P = np.zeros_like(A)
    nonzero = deg > eps
    P[nonzero] = A[nonzero] / deg[nonzero, None]

    steps = max(int(walk_steps), 1)
    decay = float(walk_decay)
    P_power = np.eye(n, dtype=np.float64)
    context = np.zeros((n, n), dtype=np.float64)
    for step in range(steps):
        P_power = P_power @ P
        context += (decay ** step) * P_power

    context = 0.5 * (context + context.T)
    context += 0.05 * np.eye(n, dtype=np.float64)
    context_norm = np.linalg.norm(context, axis=1, keepdims=True)
    context = context / np.maximum(context_norm, eps)

    sim = np.clip(context @ context.T, -1.0, 1.0)
    dist = np.maximum(1.0 - sim, 0.0)

    if perm_index is not None and perm_index.numel() > 0 and apsp_attr is not None:
        perm = perm_index.detach().cpu().long()
        src = perm[0].numpy()
        dst = perm[1].numpy()
        target_values = dist[src, dst]
        apsp_values = apsp_attr.detach().cpu().numpy().astype(np.float64).reshape(-1)
        valid = (
            np.isfinite(target_values)
            & (target_values > eps)
            & np.isfinite(apsp_values)
            & (apsp_values > eps)
        )
        normalize = str(normalize).lower()
        if valid.any() and normalize != "none":
            if normalize == "median":
                target_center = float(np.median(target_values[valid]))
                apsp_center = float(np.median(apsp_values[valid]))
            else:
                target_center = float(np.mean(target_values[valid]))
                apsp_center = float(np.mean(apsp_values[valid]))
            if target_center > eps:
                dist = dist * (apsp_center / target_center)

    return mtpf_mds_from_distance_matrix(dist)


def mtpf_community_shell_transform(pos, edge_index, scale, shrink, boundary_push):
    if pos.size(0) <= 2:
        return pos.clone()

    labels_np = detect_communities(edge_index, int(pos.size(0)))
    unique_labels = np.unique(labels_np)
    if len(unique_labels) <= 1:
        center = pos.mean(dim=0, keepdim=True)
        return center + float(shrink) * (pos - center)

    device = pos.device
    dtype = pos.dtype
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    label_to_local = {int(cid): i for i, cid in enumerate(unique_labels.tolist())}
    local_labels = torch.tensor(
        [label_to_local[int(label)] for label in labels_np],
        dtype=torch.long,
        device=device,
    )

    centers = []
    radii = []
    for cid in unique_labels.tolist():
        mask = labels == int(cid)
        points = pos[mask]
        center = points.mean(dim=0)
        centers.append(center)
        if points.size(0) > 1:
            radius = torch.norm(points - center, dim=1).median()
        else:
            radius = torch.tensor(0.0, device=device, dtype=dtype)
        radii.append(radius)

    centers = torch.stack(centers, dim=0)
    radii = torch.stack(radii, dim=0).to(dtype=dtype)
    graph_center = pos.mean(dim=0, keepdim=True)
    new_centers = graph_center + float(scale) * (centers - graph_center)
    own_centers = centers[local_labels]
    candidate = new_centers[local_labels] + float(shrink) * (pos - own_centers)

    if float(boundary_push) <= 0:
        return candidate

    directions = torch.zeros_like(pos)
    cross_degree = torch.zeros(pos.size(0), device=device, dtype=dtype)
    for u, v in edge_index.long().detach().cpu().t().tolist():
        u = int(u)
        v = int(v)
        lu = int(local_labels[u].item())
        lv = int(local_labels[v].item())
        if lu == lv:
            continue

        du = centers[lv] - centers[lu]
        dv = centers[lu] - centers[lv]
        du = du / du.norm().clamp_min(1e-9)
        dv = dv / dv.norm().clamp_min(1e-9)
        directions[u] += du
        directions[v] += dv
        cross_degree[u] += 1.0
        cross_degree[v] += 1.0

    active = cross_degree > 0
    if active.any():
        directions[active] = directions[active] / directions[active].norm(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-9)
        push = float(boundary_push) * radii[local_labels].unsqueeze(-1) * directions
        candidate = candidate + push

    return candidate


def mtpf_radial_community_layout(pos, edge_index, radius_scale=1.0, shrink=0.95):
    if pos.size(0) <= 2:
        return pos.clone()

    labels_np = detect_communities(edge_index, int(pos.size(0)))
    unique_labels = np.unique(labels_np)
    if len(unique_labels) <= 1:
        return mtpf_center_layout(pos)

    device = pos.device
    dtype = pos.dtype
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)

    centers = []
    radii = []
    sizes = []
    for cid in unique_labels.tolist():
        mask = labels == int(cid)
        points = pos[mask]
        center = points.mean(dim=0)
        centers.append(center)
        sizes.append(int(points.size(0)))
        if points.size(0) > 1:
            radii.append(torch.norm(points - center, dim=1).median())
        else:
            radii.append(torch.tensor(0.0, device=device, dtype=dtype))

    centers = torch.stack(centers, dim=0)
    radii = torch.stack(radii, dim=0).to(dtype=dtype)
    graph_center = pos.mean(dim=0, keepdim=True)

    center_offsets = centers - graph_center
    angles = torch.atan2(center_offsets[:, 1], center_offsets[:, 0]).detach().cpu().numpy()
    order = np.argsort(angles)

    base_radius = torch.norm(pos - graph_center, dim=1).median()
    if not torch.isfinite(base_radius) or float(base_radius.item()) <= 1e-9:
        base_radius = torch.tensor(1.0, device=device, dtype=dtype)
    max_comm_radius = radii.max().clamp_min(1e-6)
    ring_radius = base_radius + float(radius_scale) * (max_comm_radius + radii.mean())

    new_centers = torch.zeros_like(centers)
    total_size = max(sum(sizes), 1)
    angle_cursor = 0.0
    for rank, local_idx in enumerate(order.tolist()):
        fraction = max(sizes[local_idx] / total_size, 1.0 / (4.0 * len(order)))
        angle_cursor += np.pi * fraction
        angle = 2.0 * np.pi * (rank / max(len(order), 1)) + angle_cursor
        direction = torch.tensor(
            [np.cos(angle), np.sin(angle)],
            device=device,
            dtype=dtype,
        )
        new_centers[local_idx] = graph_center.squeeze(0) + ring_radius * direction

    label_to_local = {int(cid): i for i, cid in enumerate(unique_labels.tolist())}
    local_labels = torch.tensor(
        [label_to_local[int(label)] for label in labels_np],
        dtype=torch.long,
        device=device,
    )
    own_centers = centers[local_labels]
    own_new_centers = new_centers[local_labels]
    return own_new_centers + float(shrink) * (pos - own_centers)


def mtpf_edge_length_cv_tensor(pos, edge_index):
    edges = []
    seen = set()
    for u, v in edge_index.long().detach().cpu().t().tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)

    if len(edges) == 0:
        return pos.sum() * 0.0

    edge_tensor = torch.tensor(edges, dtype=torch.long, device=pos.device)
    lengths = torch.norm(pos[edge_tensor[:, 0]] - pos[edge_tensor[:, 1]], dim=1)
    return lengths.std(unbiased=False) / lengths.mean().clamp_min(1e-9)


def mtpf_community_surrogate_loss(pos, edge_index, sep_margin=1.25):
    labels_np = detect_communities(edge_index, int(pos.size(0)))
    unique_labels = np.unique(labels_np)
    if len(unique_labels) <= 1:
        return pos.sum() * 0.0

    device = pos.device
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    centers = []
    compact_losses = []
    for cid in unique_labels.tolist():
        mask = labels == int(cid)
        points = pos[mask]
        center = points.mean(dim=0)
        centers.append(center)
        compact_losses.append((points - center).pow(2).sum(dim=1).mean())

    compact = torch.stack(compact_losses).mean()
    centers = torch.stack(centers, dim=0)
    sep_losses = []
    for i in range(centers.size(0)):
        for j in range(i + 1, centers.size(0)):
            dist = torch.norm(centers[i] - centers[j], p=2)
            sep_losses.append(torch.relu(float(sep_margin) - dist).pow(2))
    sep = torch.stack(sep_losses).mean() if sep_losses else compact * 0.0
    return compact + sep


def mtpf_silhouette_surrogate_loss(pos, edge_index, margin=-1.0):
    labels_np = detect_communities(edge_index, int(pos.size(0)))
    unique_labels = np.unique(labels_np)
    if len(unique_labels) <= 1:
        return pos.sum() * 0.0

    device = pos.device
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    centers = []
    for cid in unique_labels.tolist():
        mask = labels == int(cid)
        centers.append(pos[mask].mean(dim=0))
    centers = torch.stack(centers, dim=0)

    label_to_local = {int(cid): i for i, cid in enumerate(unique_labels.tolist())}
    local_labels = torch.tensor(
        [label_to_local[int(label)] for label in labels_np],
        dtype=torch.long,
        device=device,
    )
    own_centers = centers[local_labels]
    intra_dist = torch.norm(pos - own_centers, dim=1)

    center_dists = torch.cdist(pos, centers)
    row_ids = torch.arange(pos.size(0), device=device)
    center_dists = center_dists.clone()
    center_dists[row_ids, local_labels] = torch.inf
    nearest_other = center_dists.min(dim=1).values

    denom = torch.maximum(intra_dist, nearest_other).clamp_min(1e-9)
    silhouette = (nearest_other - intra_dist) / denom
    if float(margin) > -1.0:
        return torch.relu(float(margin) - silhouette).mean()
    return -silhouette.mean()


def mtpf_node_occlusion_surrogate(pos, target=0.03):
    n = pos.size(0)
    if n <= 1:
        return pos.sum() * 0.0
    if n > 160:
        perm = torch.randperm(n, device=pos.device)[:160]
        pos_eval = pos[perm]
    else:
        pos_eval = pos
    dists = torch.pdist(pos_eval, p=2)
    if dists.numel() == 0:
        return pos.sum() * 0.0
    median_dist = dists.detach().median().clamp_min(1e-9)
    threshold = float(target) * median_dist
    return torch.relu(threshold - dists).pow(2).mean()


def mtpf_refine_profile_weights(args, refine_profile):
    refine_profile = str(refine_profile or "default").lower()
    base = {
        "stress": float(args.mtpf_refine_w_stress),
        "crossing": float(args.mtpf_refine_w_crossing),
        "edgecv": float(args.mtpf_refine_w_edgecv),
        "community": float(args.mtpf_refine_w_community),
        "silhouette": float(args.mtpf_refine_w_silhouette),
        "occlusion": float(args.mtpf_refine_w_occlusion),
        "anchor": float(args.mtpf_refine_w_anchor),
    }
    scales = {
        "default": {},
        "balanced": {},
        "stress": {
            "stress": 2.00,
            "crossing": 0.30,
            "edgecv": 0.60,
            "community": 0.35,
            "silhouette": 0.35,
            "occlusion": 0.50,
            "anchor": 0.85,
        },
        "crossing": {
            "stress": 0.80,
            "crossing": 2.25,
            "edgecv": 0.85,
            "community": 0.50,
            "silhouette": 0.50,
            "occlusion": 0.80,
            "anchor": 0.90,
        },
        "community": {
            "stress": 0.65,
            "crossing": 0.75,
            "edgecv": 0.65,
            "community": 2.50,
            "silhouette": 2.75,
            "occlusion": 0.80,
            "anchor": 1.00,
        },
        "smooth": {
            "stress": 1.20,
            "crossing": 0.40,
            "edgecv": 1.80,
            "community": 0.30,
            "silhouette": 0.55,
            "occlusion": 0.75,
            "anchor": 0.95,
        },
        "thesis": {
            "stress": 1.10,
            "crossing": 0.85,
            "edgecv": 0.95,
            "community": 1.45,
            "silhouette": 1.80,
            "occlusion": 0.75,
            "anchor": 0.95,
        },
    }
    for key, scale in scales.get(refine_profile, {}).items():
        base[key] *= float(scale)
    return base


def mtpf_local_refine(pos, edge_index, perm_index, apsp_attr, args, refine_profile="default"):
    steps = int(getattr(args, "mtpf_refine_steps", 0))
    if steps <= 0 or pos.size(0) <= 2:
        return pos

    device = pos.device
    edge_index = edge_index.long().to(device)
    perm_index = perm_index.long().to(device)
    apsp_attr = apsp_attr.to(device=device, dtype=pos.dtype).view(-1)
    batch_vector = torch.zeros(pos.size(0), dtype=torch.long, device=device)

    anchor = pos.detach()
    pos_var = pos.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos_var], lr=float(args.mtpf_refine_lr))
    pair_count = max(int(perm_index.size(1)), 1)
    weights = mtpf_refine_profile_weights(args, refine_profile)
    refine_mode = str(getattr(args, "mtpf_refine_mode", "adam")).lower()

    if refine_mode in {"sso", "stochastic", "stochastic_stress"}:
        max_pairs = int(getattr(args, "mtpf_sso_pairs", 2048))
        anchor_weight = float(getattr(args, "mtpf_sso_anchor", 0.10))
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)

                if max_pairs > 0 and perm_index.size(1) > max_pairs:
                    sample_ids = torch.randperm(perm_index.size(1), device=device)[:max_pairs]
                    sampled_perm = perm_index[:, sample_ids]
                    sampled_apsp = apsp_attr[sample_ids]
                else:
                    sampled_perm = perm_index
                    sampled_apsp = apsp_attr

                src = sampled_perm[0]
                dst = sampled_perm[1]
                dist = torch.norm(pos_var[dst] - pos_var[src], p=2, dim=1)
                stress = dist.sub(sampled_apsp).abs().div(sampled_apsp.clamp_min(1e-9)).square().mean()
                anchor_loss = (pos_var - anchor).pow(2).sum(dim=1).mean()
                loss = stress + anchor_weight * anchor_loss
                if not torch.isfinite(loss):
                    break

                loss.backward()
                grad_clip = float(getattr(args, "mtpf_refine_grad_clip", 1.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([pos_var], grad_clip)
                optimizer.step()

        return local_rescale_by_stress(pos_var.detach(), perm_index, apsp_attr)

    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            stress = local_deepgd_stress_tensor(pos_var, perm_index, apsp_attr) / pair_count
            crossing = sampled_crossing_loss(
                pos=pos_var,
                edge_index=edge_index,
                batch_vector=batch_vector,
                max_pairs=int(args.mtpf_refine_crossing_pairs),
                oversample_factor=int(args.crossing_oversample_factor),
                temperature=float(args.crossing_temperature),
                margin=float(args.crossing_margin),
            )
            edge_cv = mtpf_edge_length_cv_tensor(pos_var, edge_index)
            community = mtpf_community_surrogate_loss(pos_var, edge_index)
            silhouette = mtpf_silhouette_surrogate_loss(
                pos_var,
                edge_index,
                margin=float(getattr(args, "mtpf_refine_silhouette_margin", -1.0)),
            )
            occlusion = mtpf_node_occlusion_surrogate(pos_var)
            anchor_loss = (pos_var - anchor).pow(2).sum(dim=1).mean()

            loss = (
                float(weights["stress"]) * stress
                + float(weights["crossing"]) * crossing
                + float(weights["edgecv"]) * edge_cv
                + float(weights["community"]) * community
                + float(weights["silhouette"]) * silhouette
                + float(weights["occlusion"]) * occlusion
                + float(weights["anchor"]) * anchor_loss
            )

            if not torch.isfinite(loss):
                break

            loss.backward()
            grad_clip = float(getattr(args, "mtpf_refine_grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([pos_var], grad_clip)
            optimizer.step()

    return local_rescale_by_stress(pos_var.detach(), perm_index, apsp_attr)


def mtpf_refined_candidate_allowed(seed_row, refined_row, args):
    if getattr(args, "no_mtpf_refine_guard", False):
        return True

    def ratio_ok(new_value, old_value, max_ratio):
        if max_ratio <= 0:
            return True
        old_value = float(old_value)
        new_value = float(new_value)
        if old_value <= 1e-9:
            return new_value <= 1e-9
        return new_value <= old_value * float(max_ratio)

    if not ratio_ok(
        refined_row["local_stress"],
        seed_row["local_stress"],
        float(getattr(args, "mtpf_refine_max_stress_ratio", 1.03)),
    ):
        return False
    if not ratio_ok(
        refined_row["edge_crossing_norm"],
        seed_row["edge_crossing_norm"],
        float(getattr(args, "mtpf_refine_max_crossing_ratio", 1.03)),
    ):
        return False
    if not ratio_ok(
        refined_row["edge_length_cv"],
        seed_row["edge_length_cv"],
        float(getattr(args, "mtpf_refine_max_edgecv_ratio", 1.05)),
    ):
        return False

    max_occ_delta = float(getattr(args, "mtpf_refine_max_occlusion_delta", 0.005))
    if (
        max_occ_delta >= 0
        and float(refined_row["node_occlusion_rate"])
        > float(seed_row["node_occlusion_rate"]) + max_occ_delta
    ):
        return False

    min_sep_ratio = float(getattr(args, "mtpf_refine_min_sep_ratio", 0.95))
    if (
        min_sep_ratio > 0
        and float(seed_row["community_sep_comp_ratio"]) > 1e-9
        and float(refined_row["community_sep_comp_ratio"])
        < float(seed_row["community_sep_comp_ratio"]) * min_sep_ratio
    ):
        return False

    max_sil_drop = float(getattr(args, "mtpf_refine_max_sil_drop", 0.015))
    if (
        max_sil_drop >= 0
        and float(refined_row["community_silhouette"])
        < float(seed_row["community_silhouette"]) - max_sil_drop
    ):
        return False

    return True


def mtpf_effective_profile(args, rows=None):
    profile = str(getattr(args, "mtpf_profile", "balanced")).lower()
    if profile != "auto":
        return profile

    dataset = str(getattr(args, "dataset", "")).lower()
    if dataset in {"proteins", "enzymes", "north"}:
        return "thesis"
    if dataset in {"imdb_binary", "rome"}:
        return "allfirst"

    if rows:
        row = rows[0]
        num_nodes = float(row.get("num_nodes", 0.0))
        num_edges = float(row.get("num_edges", 0.0))
        num_comms = float(row.get("num_communities", 1.0))
        density = 0.0
        if num_nodes > 1:
            density = 2.0 * num_edges / max(num_nodes * (num_nodes - 1.0), 1.0)
        if num_comms >= 4 and density < 0.08:
            return "community"
        if num_nodes <= 120:
            return "stress"
    return "balanced"


def mtpf_profile_weight(args, name, rows=None):
    profiles = {
        "balanced": {
            "stress": 1.00,
            "crossing": 1.00,
            "edgecv": 0.70,
            "occlusion": 0.80,
            "compactness": 0.25,
            "separation": 0.35,
            "silhouette": 0.80,
        },
        "allfirst": {
            "stress": 1.15,
            "crossing": 1.15,
            "edgecv": 0.85,
            "occlusion": 1.00,
            "compactness": 0.35,
            "separation": 0.55,
            "silhouette": 0.95,
        },
        "stress": {
            "stress": 1.80,
            "crossing": 0.55,
            "edgecv": 0.75,
            "occlusion": 0.60,
            "compactness": 0.15,
            "separation": 0.20,
            "silhouette": 0.35,
        },
        "crossing": {
            "stress": 0.80,
            "crossing": 1.75,
            "edgecv": 0.70,
            "occlusion": 0.80,
            "compactness": 0.25,
            "separation": 0.35,
            "silhouette": 0.70,
        },
        "community": {
            "stress": 0.75,
            "crossing": 0.75,
            "edgecv": 0.55,
            "occlusion": 0.65,
            "compactness": 0.45,
            "separation": 0.90,
            "silhouette": 1.35,
        },
        "thesis": {
            "stress": 1.35,
            "crossing": 1.20,
            "edgecv": 1.05,
            "occlusion": 0.80,
            "compactness": 0.35,
            "separation": 0.95,
            "silhouette": 1.75,
        },
    }
    profile_name = mtpf_effective_profile(args, rows=rows)
    profile = profiles.get(profile_name, profiles["balanced"])
    override = getattr(args, f"mtpf_w_{name}", -1.0)
    if float(override) >= 0:
        return float(override)
    return float(profile[name])


def mtpf_metric_weight(args, name, rows=None):
    if name == "crossing_count":
        return mtpf_profile_weight(args, "crossing", rows=rows) * float(
            getattr(args, "mtpf_bc_crossing_count_weight", 0.35)
        )
    return mtpf_profile_weight(args, name, rows=rows)


def mtpf_candidate_family(name):
    base = str(name).split("+", 1)[0].lower()
    for prefix in (
        "neural", "pmds", "kk", "fr", "spectral", "cosine",
        "flingstress", "flingvis", "flingblend",
    ):
        if base == prefix or base.startswith(prefix + "_"):
            return prefix
    return base.split("_", 1)[0]


def mtpf_measure_candidate(pos, edge_index, perm_index, apsp_attr):
    pos = local_rescale_by_stress(pos, perm_index, apsp_attr)
    row = evaluate_single_graph(pos.detach().cpu(), edge_index.detach().cpu())
    row["local_stress"] = local_deepgd_stress(
        pos.detach().cpu(),
        perm_index.detach().cpu(),
        apsp_attr.detach().cpu(),
    )
    return pos, row


def mtpf_pareto_front_indices(transformed, directions, metric_names, eps=1e-12):
    if not metric_names:
        return set(range(len(next(iter(transformed.values()), []))))

    n = len(next(iter(transformed.values())))
    front = set()
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for metric_name in metric_names:
                vi = float(transformed[metric_name][i])
                vj = float(transformed[metric_name][j])
                if directions[metric_name] == "lower":
                    if vj > vi + eps:
                        better_or_equal = False
                        break
                    if vj < vi - eps:
                        strictly_better = True
                else:
                    if vj < vi - eps:
                        better_or_equal = False
                        break
                    if vj > vi + eps:
                        strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def mtpf_bc_metric_value(row, metric_name, sep_cap):
    if metric_name == "stress":
        return float(row["local_stress"])
    if metric_name == "crossing":
        return float(row["edge_crossing_norm"])
    if metric_name == "crossing_count":
        return float(row["edge_crossing"])
    if metric_name == "edgecv":
        return float(row["edge_length_cv"])
    if metric_name == "occlusion":
        return float(row["node_occlusion_rate"])
    if metric_name == "compactness":
        return float(row["community_compactness"])
    if metric_name == "separation":
        return min(float(row["community_sep_comp_ratio"]), sep_cap)
    if metric_name == "silhouette":
        return float(row["community_silhouette"])
    raise ValueError(f"Unknown BC-Pareto metric: {metric_name}")


def mtpf_bc_score_rows(rows, args):
    baseline_families = set(parse_string_list(getattr(args, "mtpf_bc_baselines", "kk,fr,pmds")))
    baseline_rows = [
        row
        for row in rows
        if mtpf_candidate_family(row.get("candidate_name", "")) in baseline_families
    ]
    if len(baseline_rows) == 0:
        return None

    sep_cap = float(getattr(args, "mtpf_sep_ratio_cap", 20.0))
    metric_names = [
        "stress",
        "crossing",
        "crossing_count",
        "edgecv",
        "occlusion",
        "compactness",
        "separation",
        "silhouette",
    ]
    directions = {
        "stress": "lower",
        "crossing": "lower",
        "crossing_count": "lower",
        "edgecv": "lower",
        "occlusion": "lower",
        "compactness": "lower",
        "separation": "higher",
        "silhouette": "higher",
    }
    active_metrics = [
        metric_name
        for metric_name in metric_names
        if mtpf_metric_weight(args, metric_name, rows=rows) > 0
    ]
    if len(active_metrics) == 0:
        return None

    values_by_metric = {}
    baseline_best = {}
    for metric_name in active_metrics:
        values = []
        for row in rows:
            value = mtpf_bc_metric_value(row, metric_name, sep_cap)
            if not np.isfinite(value):
                value = 1e9 if directions[metric_name] == "lower" else -1e9
            values.append(float(value))
        values_by_metric[metric_name] = values

        baseline_values = [
            mtpf_bc_metric_value(row, metric_name, sep_cap)
            for row in baseline_rows
        ]
        baseline_values = [
            float(value)
            for value in baseline_values
            if np.isfinite(float(value))
        ]
        if len(baseline_values) == 0:
            baseline_values = values
        if directions[metric_name] == "lower":
            baseline_best[metric_name] = min(baseline_values)
        else:
            baseline_best[metric_name] = max(baseline_values)

    pareto_front = mtpf_pareto_front_indices(
        values_by_metric,
        directions,
        active_metrics,
    )
    pareto_penalty = float(getattr(args, "mtpf_pareto_dominated_penalty", 1.0))
    max_regret_weight = float(getattr(args, "mtpf_bc_max_regret_weight", 1.0))
    improvement_bonus_weight = float(getattr(args, "mtpf_bc_improvement_bonus", 0.25))
    regret_cap = float(getattr(args, "mtpf_bc_regret_cap", 2.0))

    scored = []
    for row_id, _ in enumerate(rows):
        regrets = []
        improvement_bonus = 0.0
        score = 0.0

        for metric_name in active_metrics:
            value = float(values_by_metric[metric_name][row_id])
            best = float(baseline_best[metric_name])
            weight = mtpf_metric_weight(args, metric_name, rows=rows)
            if weight <= 0:
                continue

            if directions[metric_name] == "lower":
                denom = max(abs(best), 1e-9)
                regret = max(value / denom - 1.0, 0.0)
                improvement = max(best / max(abs(value), 1e-9) - 1.0, 0.0)
            else:
                # Difference-based regret is more stable for silhouette, which can be near zero.
                denom = max(abs(best), 0.05)
                regret = max((best - value) / denom, 0.0)
                improvement = max((value - best) / denom, 0.0)

            regret = float(np.clip(regret, 0.0, regret_cap))
            improvement = float(np.clip(improvement, 0.0, regret_cap))
            regrets.append(regret)
            score += weight * regret
            improvement_bonus += weight * improvement

        if regrets:
            score += max_regret_weight * max(regrets)

        if row_id not in pareto_front:
            score += pareto_penalty * (1.0 + len(active_metrics))

        if improvement_bonus_weight > 0:
            score -= improvement_bonus_weight * improvement_bonus

        scored.append((score, row_id))

    return scored


def mtpf_spread_refine_measured(measured, edge_index, perm_index, apsp_attr, args):
    if not getattr(args, "mtpf_spread_refine", False) or len(measured) <= 1:
        return []
    top_k = min(max(int(args.mtpf_spread_top_k), 2), len(measured))
    scored = mtpf_candidate_score_rows([item["row"] for item in measured], args)
    if not scored:
        return []
    seed_ids = [row_id for _, row_id in sorted(scored)[:top_k]]
    seed_items = [measured[row_id] for row_id in seed_ids]
    device = seed_items[0]["pos"].device
    dtype = seed_items[0]["pos"].dtype
    positions = torch.stack([item["pos"].to(device=device, dtype=dtype) for item in seed_items])
    directions = reference_directions(3, divisions=max(int(args.mtpf_spread_divisions), 1))
    if directions.shape[0] < top_k:
        repeats = int(np.ceil(top_k / max(directions.shape[0], 1)))
        directions = np.tile(directions, (repeats, 1))
    preferences = torch.tensor(directions[:top_k], dtype=dtype, device=device)
    edge_index_local = edge_index.long().to(device)
    perm_index_local = perm_index.long().to(device)
    apsp_attr_local = apsp_attr.to(device=device, dtype=dtype)
    batch_index = torch.zeros(positions.size(1), dtype=torch.long, device=device)
    labels = torch.as_tensor(
        detect_communities(edge_index.detach().cpu(), int(positions.size(1))),
        dtype=torch.long,
        device=device,
    )

    def raw_objectives(layout_set):
        outputs = []
        for layout in layout_set:
            crossing = hard_sampled_crossing_loss_per_graph(
                pos=layout,
                edge_index=edge_index_local,
                batch_index=batch_index,
                max_pairs=args.mtpf_refine_crossing_pairs,
                oversample_factor=args.crossing_oversample_factor,
                temperature=args.crossing_temperature,
                margin=args.crossing_margin,
            )
            outputs.append(
                graphwise_pareto_objectives(
                    pos=layout,
                    pair_index=perm_index_local,
                    target_distance=apsp_attr_local,
                    edge_index=edge_index_local,
                    batch_index=batch_index,
                    community_labels=labels,
                    crossing_per_graph=crossing,
                )[0]
            )
        return torch.stack(outputs)

    with torch.no_grad():
        objective_scale = raw_objectives(positions).abs().median(dim=0).values.clamp_min(0.05)

    refined_set = refine_pareto_set(
        positions=positions,
        preferences=preferences,
        objective_fn=lambda layouts: raw_objectives(layouts) / objective_scale,
        steps=args.mtpf_spread_steps,
        lr=args.mtpf_spread_lr,
        repulsion_weight=args.mtpf_spread_repulsion,
        sigma=args.mtpf_spread_sigma,
        anchor_weight=args.mtpf_spread_anchor,
        grad_clip=args.mtpf_refine_grad_clip,
    )
    accepted = []
    for local_id, (seed_item, refined_pos) in enumerate(zip(seed_items, refined_set)):
        try:
            refined_pos, refined_row = mtpf_measure_candidate(
                refined_pos.detach(), edge_index, perm_index, apsp_attr
            )
        except Exception:
            continue
        if not mtpf_refined_candidate_allowed(seed_item["row"], refined_row, args):
            continue
        name = f"spread_{seed_item['name']}_{local_id:02d}"
        refined_row["candidate_name"] = name
        accepted.append({"name": name, "pos": refined_pos, "row": refined_row})
    return accepted


def mtpf_gain_gate_score_rows(rows, args):
    neural_row = None
    for row in rows:
        if str(row.get("candidate_name", "")) == "neural":
            neural_row = row
            break
    if neural_row is None:
        neural_row = rows[0]

    sep_cap = float(getattr(args, "mtpf_sep_ratio_cap", 20.0))
    max_stress_ratio = float(getattr(args, "mtpf_max_neural_stress_ratio", 1.03))
    max_cross_norm_ratio = float(getattr(args, "mtpf_max_crossing_norm_ratio", 1.05))
    max_cross_count_ratio = float(getattr(args, "mtpf_max_crossing_count_ratio", 0.0))
    max_edgecv_ratio = float(getattr(args, "mtpf_max_edgecv_ratio", 1.08))
    max_occ_delta = float(getattr(args, "mtpf_max_occlusion_delta", 0.01))
    max_compactness_ratio = float(getattr(args, "mtpf_max_compactness_ratio", 0.0))
    min_sep_ratio = float(getattr(args, "mtpf_min_sep_ratio_ratio", 0.90))
    min_separation_ratio = float(getattr(args, "mtpf_min_separation_ratio", 0.0))
    max_sil_drop = float(getattr(args, "mtpf_max_sil_drop", 0.03))
    penalty_weight = float(getattr(args, "mtpf_gain_penalty_weight", 4.0))
    min_gain = float(getattr(args, "mtpf_gain_min_score", 0.0))

    specs = [
        ("stress", "local_stress", "lower", max_stress_ratio, 1e-9),
        ("crossing", "edge_crossing_norm", "lower", max_cross_norm_ratio, 1e-9),
        ("crossing_count", "edge_crossing", "lower", max_cross_count_ratio, 1.0),
        ("edgecv", "edge_length_cv", "lower", max_edgecv_ratio, 1e-9),
        ("occlusion", "node_occlusion_rate", "lower_delta", max_occ_delta, 1e-4),
        ("compactness", "community_compactness", "lower", max_compactness_ratio, 1e-9),
        ("separation_abs", "community_separation", "higher", min_separation_ratio, 0.05),
        ("separation", "community_sep_comp_ratio", "higher", min_sep_ratio, 0.05),
        ("silhouette", "community_silhouette", "higher_delta", max_sil_drop, 0.05),
    ]

    scored = []
    for row_id, row in enumerate(rows):
        gain = 0.0
        penalty = 0.0

        for weight_name, key, direction, budget, floor in specs:
            if weight_name == "crossing_count":
                weight = mtpf_metric_weight(args, "crossing_count", rows=rows)
            elif weight_name == "separation_abs":
                weight = mtpf_profile_weight(args, "separation", rows=rows) * 0.35
            else:
                weight = mtpf_profile_weight(args, weight_name, rows=rows)
            if weight <= 0:
                continue

            value = float(row[key])
            ref = float(neural_row[key])
            if key == "community_sep_comp_ratio":
                value = min(value, sep_cap)
                ref = min(ref, sep_cap)

            if not np.isfinite(value) or not np.isfinite(ref):
                penalty += weight
                continue

            denom = max(abs(ref), float(floor))
            if direction == "lower":
                rel_gain = (ref - value) / denom
                gain += weight * max(rel_gain, 0.0)
                if budget > 0 and ref > 1e-12:
                    penalty += weight * max(value / max(ref, 1e-12) - budget, 0.0)
            elif direction == "lower_delta":
                rel_gain = (ref - value) / denom
                gain += weight * max(rel_gain, 0.0)
                if budget >= 0:
                    penalty += weight * max(value - ref - budget, 0.0) / denom
            elif direction == "higher":
                rel_gain = (value - ref) / denom
                gain += weight * max(rel_gain, 0.0)
                if budget > 0:
                    penalty += weight * max(ref * budget - value, 0.0) / denom
            elif direction == "higher_delta":
                rel_gain = (value - ref) / denom
                gain += weight * max(rel_gain, 0.0)
                if budget >= 0:
                    penalty += weight * max(ref - value - budget, 0.0) / denom

        score = penalty_weight * penalty - gain
        if gain < min_gain:
            score += abs(min_gain - gain) + 1.0
        if str(row.get("candidate_name", "")) == "neural":
            score = min(score, 0.0)
        scored.append((float(score), row_id))

    return scored


def mtpf_candidate_score_rows(rows, args):
    if len(rows) == 0:
        return []

    sep_cap = float(getattr(args, "mtpf_sep_ratio_cap", 20.0))
    transforms = {
        "stress": lambda row: np.log1p(float(row["local_stress"])),
        "crossing": lambda row: float(row["edge_crossing_norm"]),
        "edgecv": lambda row: float(row["edge_length_cv"]),
        "occlusion": lambda row: float(row["node_occlusion_rate"]),
        "compactness": lambda row: float(row["community_compactness"]),
        "separation": lambda row: np.log1p(min(float(row["community_sep_comp_ratio"]), sep_cap)),
        "silhouette": lambda row: float(row["community_silhouette"]),
    }
    directions = {
        "stress": "lower",
        "crossing": "lower",
        "edgecv": "lower",
        "occlusion": "lower",
        "compactness": "lower",
        "separation": "higher",
        "silhouette": "higher",
    }

    transformed = {}
    for metric_name, fn in transforms.items():
        values = []
        for row in rows:
            value = fn(row)
            if not np.isfinite(value):
                value = 1e9
            values.append(float(value))
        transformed[metric_name] = values

    active_metrics = [
        metric_name
        for metric_name in transforms.keys()
        if mtpf_profile_weight(args, metric_name, rows=rows) > 0
    ]

    neural_row = None
    for row in rows:
        if str(row.get("candidate_name", "")) == "neural":
            neural_row = row
            break
    if neural_row is None:
        neural_row = rows[0]

    scored = []
    max_regret_weight = float(getattr(args, "mtpf_max_regret_weight", 0.75))
    improvement_bonus_weight = float(getattr(args, "mtpf_improvement_bonus", 1.25))
    selection_mode = str(getattr(args, "mtpf_selection_mode", "weighted")).lower()
    pareto_front = mtpf_pareto_front_indices(
        transformed,
        directions,
        active_metrics,
    )
    pareto_penalty = float(getattr(args, "mtpf_pareto_dominated_penalty", 1.0))

    if selection_mode == "thesis":
        selection_mode = "pareto"

    if selection_mode == "bc_pareto":
        bc_scored = mtpf_bc_score_rows(rows, args)
        if bc_scored is not None:
            return bc_scored
        selection_mode = "pareto"

    if selection_mode == "gain_gate":
        return mtpf_gain_gate_score_rows(rows, args)

    for row_id, _ in enumerate(rows):
        row = rows[row_id]
        regrets = []
        score = 0.0
        if selection_mode in {"rank", "pareto"}:
            for metric_name, values in transformed.items():
                weight = mtpf_profile_weight(args, metric_name, rows=rows)
                if weight <= 0:
                    continue

                reverse = directions[metric_name] == "higher"
                order = sorted(
                    range(len(values)),
                    key=lambda idx: values[idx],
                    reverse=reverse,
                )
                ranks = [0] * len(values)
                for rank_id, idx in enumerate(order):
                    ranks[idx] = rank_id
                rank_scale = max(len(values) - 1, 1)
                rank_regret = ranks[row_id] / rank_scale

                value = values[row_id]
                best = min(values) if directions[metric_name] == "lower" else max(values)
                worst = max(values) if directions[metric_name] == "lower" else min(values)
                span = max(abs(worst - best), 1e-9)
                if directions[metric_name] == "lower":
                    regret = (value - best) / span
                else:
                    regret = (best - value) / span
                regret = float(np.clip(regret, 0.0, 1.0))

                regrets.append(regret)
                score += weight * (0.65 * rank_regret + 0.35 * regret)

            if selection_mode == "pareto" and row_id not in pareto_front:
                score += pareto_penalty * (1.0 + len(active_metrics))
        else:
            for metric_name, values in transformed.items():
                weight = mtpf_profile_weight(args, metric_name, rows=rows)
                if weight <= 0:
                    continue

                value = values[row_id]
                best = min(values) if directions[metric_name] == "lower" else max(values)
                worst = max(values) if directions[metric_name] == "lower" else min(values)
                span = max(abs(worst - best), 1e-9)
                if directions[metric_name] == "lower":
                    regret = (value - best) / span
                else:
                    regret = (best - value) / span
                regret = float(np.clip(regret, 0.0, 1.0))
                regrets.append(regret)
                score += weight * regret

        if regrets:
            score += max_regret_weight * max(regrets)

        if improvement_bonus_weight > 0:
            bonus = 0.0
            neural_stress = max(float(neural_row["local_stress"]), 1e-9)
            neural_cross = max(float(neural_row["edge_crossing_norm"]), 1e-9)
            neural_edgecv = max(float(neural_row["edge_length_cv"]), 1e-9)
            neural_occ = max(float(neural_row["node_occlusion_rate"]), 1e-9)
            neural_sep = max(float(neural_row["community_sep_comp_ratio"]), 1e-9)

            bonus += mtpf_profile_weight(args, "stress", rows=rows) * max(
                (neural_stress - float(row["local_stress"])) / neural_stress,
                0.0,
            )
            bonus += mtpf_profile_weight(args, "crossing", rows=rows) * max(
                (neural_cross - float(row["edge_crossing_norm"])) / neural_cross,
                0.0,
            )
            bonus += mtpf_profile_weight(args, "edgecv", rows=rows) * max(
                (neural_edgecv - float(row["edge_length_cv"])) / neural_edgecv,
                0.0,
            )
            bonus += mtpf_profile_weight(args, "occlusion", rows=rows) * max(
                (neural_occ - float(row["node_occlusion_rate"])) / neural_occ,
                0.0,
            )
            bonus += mtpf_profile_weight(args, "separation", rows=rows) * max(
                (min(float(row["community_sep_comp_ratio"]), sep_cap) - min(neural_sep, sep_cap))
                / max(min(neural_sep, sep_cap), 1e-9),
                0.0,
            )
            bonus += mtpf_profile_weight(args, "silhouette", rows=rows) * max(
                float(row["community_silhouette"]) - float(neural_row["community_silhouette"]),
                0.0,
            )
            score -= improvement_bonus_weight * bonus

        scored.append((score, row_id))

    return scored


def mtpf_select_candidate(candidates, edge_index, perm_index, apsp_attr, args):
    measured = []
    for name, pos in candidates:
        try:
            measured_pos, row = mtpf_measure_candidate(pos, edge_index, perm_index, apsp_attr)
        except Exception:
            continue
        if not np.isfinite(float(row["local_stress"])):
            continue
        row["candidate_name"] = name
        measured.append({"name": name, "pos": measured_pos, "row": row})

    if not measured:
        return candidates[0][1], candidates[0][0], {}

    best_stress = min(float(item["row"]["local_stress"]) for item in measured)
    max_stress_ratio = float(getattr(args, "mtpf_max_stress_ratio", 0.0))
    if max_stress_ratio > 0 and best_stress > 1e-9:
        guarded = [
            item for item in measured
            if item["name"] == "neural"
            or float(item["row"]["local_stress"]) <= best_stress * max_stress_ratio
        ]
        if guarded:
            measured = guarded

    if not getattr(args, "no_mtpf_neural_guard", False):
        neural_item = None
        for item in measured:
            if item["name"] == "neural":
                neural_item = item
                break
        if neural_item is not None:
            neural_row = neural_item["row"]
            max_cross_ratio = float(getattr(args, "mtpf_max_crossing_norm_ratio", 1.15))
            max_neural_stress_ratio = float(getattr(args, "mtpf_max_neural_stress_ratio", 1.15))
            max_edgecv_ratio = float(getattr(args, "mtpf_max_edgecv_ratio", 1.25))
            max_crossing_count_ratio = float(getattr(args, "mtpf_max_crossing_count_ratio", 0.0))
            max_occ_delta = float(getattr(args, "mtpf_max_occlusion_delta", 0.02))
            max_compactness_ratio = float(getattr(args, "mtpf_max_compactness_ratio", 0.0))
            min_separation_ratio = float(getattr(args, "mtpf_min_separation_ratio", 0.0))
            min_sep_ratio = float(getattr(args, "mtpf_min_sep_ratio_ratio", 0.75))
            max_sil_drop = float(getattr(args, "mtpf_max_sil_drop", 0.04))
            guarded = []
            for item in measured:
                row = item["row"]
                if (
                    max_neural_stress_ratio > 0
                    and float(row["local_stress"])
                    > float(neural_row["local_stress"]) * max_neural_stress_ratio
                ):
                    continue
                if (
                    max_crossing_count_ratio > 0
                    and float(neural_row["edge_crossing"]) > 1e-9
                    and float(row["edge_crossing"])
                    > float(neural_row["edge_crossing"]) * max_crossing_count_ratio
                ):
                    continue
                if (
                    max_cross_ratio > 0
                    and float(neural_row["edge_crossing_norm"]) > 1e-9
                    and float(row["edge_crossing_norm"])
                    > float(neural_row["edge_crossing_norm"]) * max_cross_ratio
                ):
                    continue
                if (
                    max_edgecv_ratio > 0
                    and float(neural_row["edge_length_cv"]) > 1e-9
                    and float(row["edge_length_cv"])
                    > float(neural_row["edge_length_cv"]) * max_edgecv_ratio
                ):
                    continue
                if (
                    max_occ_delta >= 0
                    and float(row["node_occlusion_rate"])
                    > float(neural_row["node_occlusion_rate"]) + max_occ_delta
                ):
                    continue
                if (
                    max_compactness_ratio > 0
                    and float(neural_row["community_compactness"]) > 1e-9
                    and float(row["community_compactness"])
                    > float(neural_row["community_compactness"]) * max_compactness_ratio
                ):
                    continue
                if (
                    min_separation_ratio > 0
                    and float(row["community_separation"])
                    < float(neural_row["community_separation"]) * min_separation_ratio
                ):
                    continue
                if (
                    min_sep_ratio > 0
                    and float(row["community_sep_comp_ratio"])
                    < float(neural_row["community_sep_comp_ratio"]) * min_sep_ratio
                ):
                    continue
                if (
                    float(row["community_silhouette"])
                    < float(neural_row["community_silhouette"]) - max_sil_drop
                ):
                    continue
                guarded.append(item)
            if guarded:
                measured = guarded
            elif getattr(args, "mtpf_strict_neural_guard", False):
                return neural_item["pos"], neural_item["name"], neural_item["row"]

    if int(getattr(args, "mtpf_refine_steps", 0)) > 0:
        scored_before_refine = mtpf_candidate_score_rows([item["row"] for item in measured], args)
        if scored_before_refine:
            refine_top_k = max(int(getattr(args, "mtpf_refine_top_k", 1)), 1)
            refine_profiles = parse_string_list(
                getattr(args, "mtpf_refine_profiles", "default")
            )
            if len(refine_profiles) == 0:
                refine_profiles = ["default"]

            seed_ids = []
            seen_ids = set()
            for _, row_id in sorted(scored_before_refine, key=lambda item: item[0]):
                if row_id in seen_ids:
                    continue
                seed_ids.append(row_id)
                seen_ids.add(row_id)
                if len(seed_ids) >= refine_top_k:
                    break

            for row_id in seed_ids:
                seed_item = measured[row_id]
                for refine_profile in refine_profiles:
                    if refine_profile in {"none", "off", "false"}:
                        continue
                    refined_pos = mtpf_local_refine(
                        seed_item["pos"],
                        edge_index=edge_index,
                        perm_index=perm_index,
                        apsp_attr=apsp_attr,
                        args=args,
                        refine_profile=refine_profile,
                    )
                    refined_pos, refined_row = mtpf_measure_candidate(
                        refined_pos,
                        edge_index,
                        perm_index,
                        apsp_attr,
                    )
                    if not mtpf_refined_candidate_allowed(
                        seed_item["row"],
                        refined_row,
                        args,
                    ):
                        continue
                    suffix = "+TTR"
                    if refine_profile not in {"default", "balanced"}:
                        suffix += f"-{refine_profile}"
                    refined_name = seed_item["name"] + suffix
                    refined_row["candidate_name"] = refined_name
                    measured.append(
                        {
                            "name": refined_name,
                            "pos": refined_pos,
                            "row": refined_row,
                        }
                    )

    measured.extend(
        mtpf_spread_refine_measured(
            measured=measured,
            edge_index=edge_index,
            perm_index=perm_index,
            apsp_attr=apsp_attr,
            args=args,
        )
    )
    rows = [item["row"] for item in measured]
    scored = mtpf_candidate_score_rows(rows, args)
    if not scored:
        best_item = min(measured, key=lambda item: float(item["row"]["local_stress"]))
    else:
        _, best_row_id = min(
            scored,
            key=lambda item: (
                item[0],
                float(rows[item[1]]["local_stress"]),
                float(rows[item[1]]["edge_crossing_norm"]),
                -float(rows[item[1]]["community_silhouette"]),
            ),
        )
        best_item = measured[best_row_id]

    return best_item["pos"], best_item["name"], best_item["row"]


def mtpf_generate_candidates(local_record, neural_pos, pmds_pos, args):
    teachers = set(parse_string_list(args.mtpf_teachers))
    candidates = []

    edge_index = local_record["edge_index"]
    perm_index = local_record["perm_index"]
    apsp_attr = local_record["apsp_attr"]
    num_nodes = int(local_record["num_nodes"])
    seed = int(getattr(args, "seed", 42)) + int(local_record["graph_id"])

    neural_pos = local_rescale_by_stress(neural_pos.detach().cpu().float(), perm_index, apsp_attr)
    pmds_pos = local_rescale_by_stress(pmds_pos.detach().cpu().float(), perm_index, apsp_attr)

    if "neural" in teachers:
        candidates.append(("neural", neural_pos))

    if "pmds" in teachers:
        candidates.append(("pmds", pmds_pos))

    kk_pos = None
    fr_pos = None
    spectral_pos = None
    cosine_pos = None
    if "kk" in teachers:
        try:
            kk_pos = mtpf_networkx_layout(num_nodes, edge_index, "kk", seed=seed)
            kk_pos = local_rescale_by_stress(kk_pos, perm_index, apsp_attr)
            candidates.append(("kk", kk_pos))
        except Exception:
            kk_pos = None

    if "fr" in teachers:
        fr_seed_count = max(int(getattr(args, "mtpf_fr_seeds", 1)), 1)
        fr_iterations = max(int(getattr(args, "mtpf_fr_iterations", 100)), 1)
        for seed_offset in range(fr_seed_count):
            try:
                fr_seed = seed + seed_offset * 1009
                candidate_name = "fr" if seed_offset == 0 else f"fr_seed{seed_offset}"
                candidate_pos = mtpf_networkx_layout(
                    num_nodes,
                    edge_index,
                    "fr",
                    seed=fr_seed,
                    iterations=fr_iterations,
                )
                candidate_pos = local_rescale_by_stress(candidate_pos, perm_index, apsp_attr)
                candidates.append((candidate_name, candidate_pos))
                if fr_pos is None:
                    fr_pos = candidate_pos
            except Exception:
                continue

    if "spectral" in teachers:
        try:
            spectral_pos = mtpf_networkx_layout(num_nodes, edge_index, "spectral", seed=seed)
            spectral_pos = local_rescale_by_stress(spectral_pos, perm_index, apsp_attr)
            candidates.append(("spectral", spectral_pos))
        except Exception:
            spectral_pos = None

    fling_methods = [
        method for method in ("flingstress", "flingvis", "flingblend")
        if method in teachers
    ]
    if fling_methods:
        try:
            fling = FlingAdapter(root=args.fling_root, device=args.fling_device)
            graph = fling.graph_from_edge_index(local_record["num_nodes"], edge_index)
            for method in fling_methods:
                fling_pos = torch.as_tensor(
                    fling.layout(
                        method=method,
                        graph=graph,
                        seed=args.seed,
                        iterations=args.mtpf_fling_iterations,
                        blend_lambda=args.mtpf_fling_blend_lambda,
                    ),
                    dtype=neural_pos.dtype,
                )
                fling_pos = local_rescale_by_stress(fling_pos, perm_index, apsp_attr)
                candidates.append((method, fling_pos))
        except (FlingUnavailableError, RuntimeError, ValueError) as error:
            if not getattr(args, "_fling_warning_shown", False):
                print(f"[WARN] skipped Fling teachers: {error}")
                args._fling_warning_shown = True

    if "cosine" in teachers or "cosine_stress" in teachers or "word2vecgd" in teachers:
        try:
            cosine_pos = mtpf_cosine_context_layout(
                num_nodes=num_nodes,
                edge_index=edge_index,
                perm_index=perm_index,
                apsp_attr=apsp_attr,
                walk_steps=getattr(args, "cosine_walk_steps", 4),
                walk_decay=getattr(args, "cosine_walk_decay", 0.75),
                normalize=getattr(args, "cosine_normalize", "mean"),
            )
            cosine_pos = local_rescale_by_stress(cosine_pos, perm_index, apsp_attr)
            candidates.append(("cosine", cosine_pos))
        except Exception:
            cosine_pos = None

    if "blend" in teachers:
        blend_sources = []
        if kk_pos is not None:
            blend_sources.append(("kk", kk_pos))
        if fr_pos is not None:
            blend_sources.append(("fr", fr_pos))
        if spectral_pos is not None:
            blend_sources.append(("spectral", spectral_pos))
        if cosine_pos is not None:
            blend_sources.append(("cosine", cosine_pos))
        blend_sources.append(("pmds", pmds_pos))

        for source_name, source_pos in blend_sources:
            aligned = mtpf_procrustes_align(source_pos, neural_pos)
            for alpha in parse_float_list(args.mtpf_blend_alphas):
                alpha = float(alpha)
                if alpha <= 0 or alpha >= 1:
                    continue
                blended = (1.0 - alpha) * neural_pos + alpha * aligned
                candidates.append((f"blend_{source_name}_{alpha:.2f}", blended))

        if getattr(args, "use_mtpf_pair_blends", False):
            pair_sources = [("neural", neural_pos), ("pmds", pmds_pos)]
            if kk_pos is not None:
                pair_sources.append(("kk", kk_pos))
            if fr_pos is not None:
                pair_sources.append(("fr", fr_pos))
            if spectral_pos is not None:
                pair_sources.append(("spectral", spectral_pos))
            if cosine_pos is not None:
                pair_sources.append(("cosine", cosine_pos))

            for left_id in range(len(pair_sources)):
                for right_id in range(left_id + 1, len(pair_sources)):
                    left_name, left_pos = pair_sources[left_id]
                    right_name, right_pos = pair_sources[right_id]
                    aligned_right = mtpf_procrustes_align(right_pos, left_pos)
                    for alpha in parse_float_list(args.mtpf_blend_alphas):
                        alpha = float(alpha)
                        if alpha <= 0 or alpha >= 1:
                            continue
                        blended = (1.0 - alpha) * left_pos + alpha * aligned_right
                        name = f"pairblend_{left_name}_{right_name}_{alpha:.2f}"
                        candidates.append((name, blended))

    if "radial" in teachers or "radial_community" in teachers:
        radial_seed_items = [("neural", neural_pos), ("pmds", pmds_pos)]
        if kk_pos is not None:
            radial_seed_items.append(("kk", kk_pos))
        if fr_pos is not None:
            radial_seed_items.append(("fr", fr_pos))
        if spectral_pos is not None:
            radial_seed_items.append(("spectral", spectral_pos))
        if cosine_pos is not None:
            radial_seed_items.append(("cosine", cosine_pos))

        for source_name, source_pos in radial_seed_items:
            for radius_scale in parse_float_list(args.mtpf_radial_scales):
                for shrink in parse_float_list(args.mtpf_radial_shrinks):
                    radial_pos = mtpf_radial_community_layout(
                        source_pos,
                        edge_index=edge_index,
                        radius_scale=float(radius_scale),
                        shrink=float(shrink),
                    )
                    name = f"radial_{source_name}_{float(radius_scale):.2f}_{float(shrink):.3f}"
                    candidates.append((name, radial_pos))

    if "shell" in teachers:
        shell_seed_items = [("neural", neural_pos), ("pmds", pmds_pos)]
        if kk_pos is not None:
            shell_seed_items.append(("kk", kk_pos))
        if fr_pos is not None:
            shell_seed_items.append(("fr", fr_pos))
        if spectral_pos is not None:
            shell_seed_items.append(("spectral", spectral_pos))
        if cosine_pos is not None:
            shell_seed_items.append(("cosine", cosine_pos))
        for source_name, source_pos in shell_seed_items:
            for scale in parse_float_list(args.mtpf_shell_scales):
                for shrink in parse_float_list(args.mtpf_shell_shrinks):
                    for boundary_push in parse_float_list(args.mtpf_boundary_pushes):
                        if (
                            abs(float(scale) - 1.0) < 1e-9
                            and abs(float(shrink) - 1.0) < 1e-9
                            and abs(float(boundary_push)) < 1e-9
                        ):
                            continue
                        shell_pos = mtpf_community_shell_transform(
                            source_pos,
                            edge_index=edge_index,
                            scale=float(scale),
                            shrink=float(shrink),
                            boundary_push=float(boundary_push),
                        )
                        name = (
                            f"shell_{source_name}_"
                            f"{float(scale):.2f}_{float(shrink):.3f}_{float(boundary_push):.2f}"
                        )
                        candidates.append((name, shell_pos))

    if len(candidates) == 0:
        candidates.append(("neural", neural_pos))

    max_candidates = int(getattr(args, "mtpf_max_candidates", 0))
    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates


def mtpf_select_batch(neural_pos, pmds_pos, batch, args, additional_candidates=None):
    local_records = mtpf_build_local_graph_records(batch)
    selected = neural_pos.detach().clone()
    source_counts = {}
    candidate_count = 0

    for record in local_records:
        global_nodes = record["global_nodes_device"]
        local_neural_pos = neural_pos[global_nodes].detach().cpu().float()
        local_pmds_pos = pmds_pos[global_nodes].detach().cpu().float()

        candidates = mtpf_generate_candidates(
            local_record=record,
            neural_pos=local_neural_pos,
            pmds_pos=local_pmds_pos,
            args=args,
        )
        for candidate_name, candidate_batch_pos in additional_candidates or []:
            local_candidate = candidate_batch_pos[
                record["global_nodes_device"]
            ].detach().cpu().float()
            local_candidate = local_rescale_by_stress(
                local_candidate, record["perm_index"], record["apsp_attr"]
            )
            candidates.append((candidate_name, local_candidate))
        candidate_count += len(candidates)

        chosen_pos, chosen_name, _ = mtpf_select_candidate(
            candidates=candidates,
            edge_index=record["edge_index"],
            perm_index=record["perm_index"],
            apsp_attr=record["apsp_attr"],
            args=args,
        )

        source_key = chosen_name.split("_")[0].split("+")[0]
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected[global_nodes] = chosen_pos.to(device=selected.device, dtype=selected.dtype)

    selected = rescale_by_stress(
        selected,
        batch.apsp_attr,
        batch.perm_index.long(),
        batch.batch.long(),
    )
    return selected, source_counts, candidate_count


def cgrcc_calibrate_single_graph(
    pos,
    edge_index,
    perm_index,
    apsp_attr,
    args,
    eps=1e-9,
):
    """
    Stress-Guarded Crossing-Guided Readability Calibration.

    This is a deterministic evaluation-time calibration. It tries small
    community-center radial expansion and intra-community contraction, then
    keeps the candidate only if crossing, silhouette, and local stress guards
    are satisfied.
    """
    if pos.size(0) <= 2:
        return pos, False

    labels_np = detect_communities(edge_index, int(pos.size(0)))
    unique_labels = np.unique(labels_np)
    if len(unique_labels) <= 1:
        return pos, False

    labels = torch.tensor(labels_np, dtype=torch.long, device=pos.device)
    original_crossing, _ = edge_crossing_count(pos, edge_index)
    original_comm = community_metrics(pos, labels_np)
    original_ratio = float(original_comm["community_sep_comp_ratio"])
    original_sil = float(original_comm["community_silhouette"])
    original_stress = local_deepgd_stress(pos, perm_index, apsp_attr, eps=eps)

    centers = []
    for cid in unique_labels:
        mask = labels == int(cid)
        centers.append(pos[mask].mean(dim=0))
    centers = torch.stack(centers, dim=0)

    label_to_local = {int(cid): i for i, cid in enumerate(unique_labels.tolist())}
    local_inv = torch.tensor(
        [label_to_local[int(label)] for label in labels_np],
        dtype=torch.long,
        device=pos.device,
    )

    graph_center = pos.mean(dim=0, keepdim=True)
    own_centers = centers[local_inv]

    scale_candidates = parse_float_list(args.cgrcc_scale_candidates)
    shrink_candidates = parse_float_list(args.cgrcc_shrink_candidates)

    best_pos = pos
    best_ratio = original_ratio
    best_crossing = int(original_crossing)
    best_sil = original_sil
    best_stress = original_stress
    accepted = False
    max_stress_ratio = float(getattr(args, "cgrcc_max_stress_ratio", 1.03))
    stress_weight = float(getattr(args, "cgrcc_stress_weight", 0.10))
    best_score = 0.0

    for scale in scale_candidates:
        for shrink in shrink_candidates:
            if abs(scale - 1.0) < eps and abs(shrink - 1.0) < eps:
                continue

            new_centers = graph_center + float(scale) * (centers - graph_center)
            candidate = new_centers[local_inv] + float(shrink) * (pos - own_centers)

            crossing, _ = edge_crossing_count(candidate, edge_index)
            if crossing > original_crossing + int(args.cgrcc_crossing_allowance):
                continue

            candidate_stress = local_deepgd_stress(
                candidate,
                perm_index,
                apsp_attr,
                eps=eps,
            )
            if max_stress_ratio > 0 and original_stress > eps:
                if candidate_stress > original_stress * max_stress_ratio:
                    continue

            comm = community_metrics(candidate, labels_np)
            ratio = float(comm["community_sep_comp_ratio"])
            sil = float(comm["community_silhouette"])

            if sil < original_sil - float(args.cgrcc_max_sil_drop):
                continue

            ratio_gain = ratio - original_ratio
            if ratio_gain < float(args.cgrcc_min_ratio_gain):
                continue

            stress_increase = 0.0
            if original_stress > eps:
                stress_increase = max(candidate_stress / original_stress - 1.0, 0.0)
            score = ratio_gain - stress_weight * stress_increase

            better_score = score > best_score + eps
            same_score_better_crossing = (
                abs(score - best_score) <= eps and crossing < best_crossing
            )
            same_score_crossing_better_sil = (
                abs(score - best_score) <= eps
                and crossing == best_crossing
                and sil > best_sil
            )
            same_score_sil_better_stress = (
                abs(score - best_score) <= eps
                and crossing == best_crossing
                and abs(sil - best_sil) <= eps
                and candidate_stress < best_stress
            )
            if (
                better_score
                or same_score_better_crossing
                or same_score_crossing_better_sil
                or same_score_sil_better_stress
            ):
                best_pos = candidate
                best_ratio = ratio
                best_crossing = int(crossing)
                best_sil = sil
                best_stress = candidate_stress
                best_score = score
                accepted = True

    return best_pos, accepted


def cgrcc_calibrate_batch(pos, batch, args):
    device = pos.device
    batch_vector_cpu = batch.batch.long().detach().cpu()
    edge_index_cpu = batch.edge_index.long().detach().cpu()
    perm_index_cpu = batch.perm_index.long().detach().cpu()
    apsp_attr_cpu = batch.apsp_attr.detach().cpu().view(-1)
    calibrated = pos.detach().clone()

    accepted = 0
    total = 0

    for gid in batch_vector_cpu.unique(sorted=True).tolist():
        node_mask = batch_vector_cpu == gid
        global_nodes = torch.where(node_mask)[0]
        if int(global_nodes.numel()) == 0:
            continue

        node_set = set(global_nodes.tolist())
        mapping = {int(g): i for i, g in enumerate(global_nodes.tolist())}

        local_edges = []
        for u, v in edge_index_cpu.t().tolist():
            if u in node_set and v in node_set:
                local_edges.append([mapping[int(u)], mapping[int(v)]])

        if len(local_edges) == 0:
            local_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()

        local_perm = []
        local_apsp = []
        for edge_id, (u, v) in enumerate(perm_index_cpu.t().tolist()):
            if u in node_set and v in node_set:
                local_perm.append([mapping[int(u)], mapping[int(v)]])
                local_apsp.append(float(apsp_attr_cpu[edge_id].item()))

        if len(local_perm) == 0:
            local_perm_index = torch.empty((2, 0), dtype=torch.long)
            local_apsp_attr = torch.empty(0, dtype=torch.float32)
        else:
            local_perm_index = torch.tensor(local_perm, dtype=torch.long).t().contiguous()
            local_apsp_attr = torch.tensor(local_apsp, dtype=torch.float32)

        local_pos = pos[global_nodes.to(device)].detach().cpu()
        local_calibrated, did_accept = cgrcc_calibrate_single_graph(
            pos=local_pos,
            edge_index=local_edge_index,
            perm_index=local_perm_index,
            apsp_attr=local_apsp_attr,
            args=args,
        )

        calibrated[global_nodes.to(device)] = local_calibrated.to(device=device, dtype=pos.dtype)
        accepted += int(did_accept)
        total += 1

    return calibrated, accepted, total


@torch.no_grad()
def evaluate_split(loader, deepgd_model, refiner, stress_metric, device, args):
    deepgd_model.eval()
    # Counterfactual teacher passes are training/validation losses, not part of
    # deployable inference or the reported forward latency.
    args._hctf_teacher_enabled = False
    if refiner is not None:
        refiner.eval()

    all_rows = []
    pareto_front_rows = []
    base_stress_values = []
    refined_stress_values = []
    cgrcc_stress_values = []
    cgrcc_accept_count = 0
    cgrcc_total_count = 0
    mtpf_stress_values = []
    mtpf_source_counts = {}
    mtpf_candidate_total = 0
    mtpf_graph_total = 0
    paretofield_guard_accepted = 0
    paretofield_guard_total = 0
    paretofield_guard_step_sum = 0.0
    paretofield_guard_partial = 0
    paretofield_guard_full = 0
    paretofield_step_total = 0
    paretofield_step_batches = 0

    if getattr(args, "use_cgrcc", False):
        print(
            "[INFO] CG-RCC enabled: "
            f"scale={args.cgrcc_scale_candidates}, "
            f"shrink={args.cgrcc_shrink_candidates}, "
            f"crossing_allowance={args.cgrcc_crossing_allowance}, "
            f"max_sil_drop={args.cgrcc_max_sil_drop}, "
            f"min_ratio_gain={args.cgrcc_min_ratio_gain}, "
            f"max_stress_ratio={args.cgrcc_max_stress_ratio}, "
            f"stress_weight={args.cgrcc_stress_weight}"
        )

    if getattr(args, "use_mtpf", False):
        print(
            "[INFO] MTPF-GD enabled: "
            f"profile={args.mtpf_profile}, "
            f"selection={args.mtpf_selection_mode}, "
            f"teachers={args.mtpf_teachers}, "
            f"max_stress_ratio={args.mtpf_max_stress_ratio}, "
            f"refine_steps={args.mtpf_refine_steps}, "
            f"refine_top_k={args.mtpf_refine_top_k}, "
            f"refine_profiles={args.mtpf_refine_profiles}, "
            f"dcsg={args.mtpf_refine_w_silhouette}"
        )

    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(loader, desc="Evaluate test")

    for batch_id, batch in enumerate(pbar):
        if args.max_test_batches > 0 and batch_id >= args.max_test_batches:
            break

        batch = batch.to(device)

        if device_type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        _, base_pos, refined_pos, loss_dict = forward_model(
            batch=batch,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            args=args,
        )

        if device_type == "cuda":
            torch.cuda.synchronize(device)
        infer_time = time.perf_counter() - start_time

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

        base_name = get_method_name(args, refined=False)
        final_name = get_method_name(args, refined=True)

        if getattr(args, "paretofield_front_csv", ""):
            for candidate_name, candidate_pos in loss_dict.get("_paretofield_candidates", []):
                candidate_eval = rescale_by_stress(
                    candidate_pos,
                    batch.apsp_attr,
                    batch.perm_index.long(),
                    batch.batch.long(),
                )
                candidate_stress = stress_metric(
                    candidate_eval,
                    batch.perm_index.long(),
                    batch.apsp_attr,
                    batch.batch.long(),
                ).item()
                pareto_front_rows.extend(
                    evaluate_pos_by_graph(
                        pos=candidate_eval,
                        batch=batch,
                        method_name=candidate_name,
                        batch_id=batch_id,
                        batch_stress=candidate_stress,
                        infer_time=infer_time,
                    )
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

        guard_stats = loss_dict.get("_paretofield_guard_stats")
        if guard_stats is not None:
            accepted = guard_stats["accepted"]
            paretofield_guard_accepted += int(accepted.sum().item())
            paretofield_guard_total += int(accepted.numel())
            step_fraction = guard_stats.get("step_fraction")
            if step_fraction is not None:
                paretofield_guard_step_sum += float(step_fraction.sum().item())
                paretofield_guard_partial += int(
                    ((step_fraction > 0.0) & (step_fraction < 1.0)).sum().item()
                )
                paretofield_guard_full += int((step_fraction >= 1.0).sum().item())
        if getattr(args, "use_paretofield", False):
            paretofield_step_total += int(
                loss_dict.get("_paretofield_inference_steps", 0)
            )
            paretofield_step_batches += 1

        if getattr(args, "use_cgrcc", False):
            cgrcc_start_time = time.time()
            use_final_layout = args.use_refiner or getattr(args, "use_paretofield", False)
            cgrcc_input_pos = refined_pos_eval if use_final_layout else base_pos_eval
            cgrcc_pos_eval, accepted, total = cgrcc_calibrate_batch(
                pos=cgrcc_input_pos,
                batch=batch,
                args=args,
            )
            cgrcc_time = infer_time + (time.time() - cgrcc_start_time)
            cgrcc_stress = stress_metric(
                cgrcc_pos_eval,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long(),
            ).item()
            cgrcc_stress_values.append(cgrcc_stress)
            cgrcc_accept_count += accepted
            cgrcc_total_count += total

            cgrcc_suffix = "+SG-CG-RCC"
            if float(getattr(args, "cgrcc_max_stress_ratio", 1.03)) <= 0:
                cgrcc_suffix = "+CG-RCC"

            cgrcc_name = (
                final_name if use_final_layout else base_name
            ) + cgrcc_suffix
            all_rows.extend(
                evaluate_pos_by_graph(
                    pos=cgrcc_pos_eval,
                    batch=batch,
                    method_name=cgrcc_name,
                    batch_id=batch_id,
                    batch_stress=cgrcc_stress,
                    infer_time=cgrcc_time,
                )
            )

        if getattr(args, "use_mtpf", False):
            mtpf_start_time = time.time()
            use_final_layout = args.use_refiner or getattr(args, "use_paretofield", False)
            mtpf_input_pos = refined_pos_eval if use_final_layout else base_pos_eval
            pmds_pos_eval = rescale_by_stress(
                batch.pos,
                batch.apsp_attr,
                batch.perm_index.long(),
                batch.batch.long(),
            )
            mtpf_pos_eval, source_counts, candidate_count = mtpf_select_batch(
                neural_pos=mtpf_input_pos,
                pmds_pos=pmds_pos_eval,
                batch=batch,
                args=args,
                additional_candidates=loss_dict.get("_paretofield_candidates", []),
            )
            mtpf_time = infer_time + (time.time() - mtpf_start_time)
            mtpf_stress = stress_metric(
                mtpf_pos_eval,
                batch.perm_index.long(),
                batch.apsp_attr,
                batch.batch.long(),
            ).item()
            mtpf_stress_values.append(mtpf_stress)
            mtpf_candidate_total += int(candidate_count)
            mtpf_graph_total += int(sum(source_counts.values()))
            for key, value in source_counts.items():
                mtpf_source_counts[key] = mtpf_source_counts.get(key, 0) + int(value)

            mtpf_name = (final_name if use_final_layout else base_name) + "+MTPF-GD"
            all_rows.extend(
                evaluate_pos_by_graph(
                    pos=mtpf_pos_eval,
                    batch=batch,
                    method_name=mtpf_name,
                    batch_id=batch_id,
                    batch_stress=mtpf_stress,
                    infer_time=mtpf_time,
                )
            )

        if args.use_refiner or getattr(args, "use_paretofield", False):
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
    df["peak_cuda_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device_type == "cuda"
        else 0.0
    )

    print("\n========== Evaluation Stress Summary ==========")
    print(f"Base Mean Stress: {np.mean(base_stress_values):.6f}")

    if args.use_refiner or getattr(args, "use_paretofield", False):
        print(f"Final Mean Stress: {np.mean(refined_stress_values):.6f}")

    if getattr(args, "paretofield_safe_guard", False):
        print(
            "[INFO] ParetoField guard accepted "
            f"{paretofield_guard_accepted}/{max(paretofield_guard_total, 1)} graphs"
        )
        if getattr(args, "paretofield_continuous_guard", False):
            print(
                "[INFO] ParetoField safe-line average step: "
                f"{paretofield_guard_step_sum / max(paretofield_guard_total, 1):.4f}; "
                f"partial={paretofield_guard_partial}, full={paretofield_guard_full}"
            )
    if (
        getattr(args, "use_paretofield", False)
        and paretofield_step_batches > 0
    ):
        print(
            "[INFO] ParetoField average executed steps: "
            f"{paretofield_step_total / paretofield_step_batches:.2f}"
        )

    if getattr(args, "use_cgrcc", False) and len(cgrcc_stress_values) > 0:
        print(f"CG-RCC Mean Stress: {np.mean(cgrcc_stress_values):.6f}")
        print(
            "[INFO] CG-RCC accepted "
            f"{cgrcc_accept_count}/{max(cgrcc_total_count, 1)} graphs"
        )

    if getattr(args, "use_mtpf", False) and len(mtpf_stress_values) > 0:
        print(f"MTPF-GD Mean Stress: {np.mean(mtpf_stress_values):.6f}")
        print(
            "[INFO] MTPF-GD selected sources: "
            + ", ".join(f"{k}={v}" for k, v in sorted(mtpf_source_counts.items()))
        )
        print(
            "[INFO] MTPF-GD average candidates per graph: "
            f"{mtpf_candidate_total / max(mtpf_graph_total, 1):.2f}"
        )

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
        "angular_resolution_deg",
        "angular_resolution_ratio",
        "crossing_angle_deg",
        "crossing_angle_p05_deg",
        "node_edge_clearance",
        "node_edge_clearance_p05",
        "neighborhood_preservation_k5",
    ]

    print("\n========== Mean Metrics by Method ==========")
    print(df.groupby("method")[metric_cols].mean())

    ensure_dir(os.path.dirname(args.eval_csv))
    df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] saved evaluation csv to: {args.eval_csv}")

    if getattr(args, "paretofield_front_csv", ""):
        ensure_dir(os.path.dirname(args.paretofield_front_csv))
        pd.DataFrame(pareto_front_rows).to_csv(
            args.paretofield_front_csv, index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] saved Pareto front csv to: {args.paretofield_front_csv}")

    return df


def save_checkpoint(path, epoch, deepgd_model, refiner, optimizer, best_val, args):
    ensure_dir(os.path.dirname(path))
    save_optimizer = optimizer is not None and not getattr(args, "no_save_optimizer", False)

    ckpt = {
        "epoch": epoch,
        "deepgd_state_dict": deepgd_model.state_dict(),
        "refiner_state_dict": refiner.state_dict() if refiner is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if save_optimizer else None,
        "best_val": best_val,
        "args": {
            key: value for key, value in vars(args).items()
            if not str(key).startswith("_")
        },
    }

    checkpoint_dir = os.path.dirname(path) or "."
    base_name = os.path.basename(path)
    stem, ext = os.path.splitext(path)
    last_error = None

    for attempt in range(3):
        tmp_name = (
            f".{base_name}.tmp-"
            f"{os.getpid()}-{epoch}-{attempt}-{int(time.time() * 1000)}"
        )
        tmp_path = os.path.join(checkpoint_dir, tmp_name)

        try:
            torch.save(ckpt, tmp_path)
            try:
                os.replace(tmp_path, path)
                print(f"[INFO] saved checkpoint to: {path}")
                return path
            except OSError as replace_error:
                fallback_path = f"{stem}_epoch{epoch:03d}_{int(time.time() * 1000)}{ext}"
                os.replace(tmp_path, fallback_path)
                print(
                    f"[WARN] could not replace checkpoint {path}: {replace_error}. "
                    f"saved fallback checkpoint to: {fallback_path}"
                )
                return fallback_path
        except Exception as error:
            last_error = error
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            time.sleep(0.5 * (attempt + 1))

    fallback_path = f"{stem}_epoch{epoch:03d}_{int(time.time() * 1000)}{ext}"
    try:
        torch.save(ckpt, fallback_path)
        print(
            f"[WARN] saved checkpoint to fallback path after retries: {fallback_path}"
        )
        return fallback_path
    except Exception as error:
        raise RuntimeError(
            f"Failed to save checkpoint to {path} or fallback {fallback_path}"
        ) from (last_error or error)


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_checkpoint(path, deepgd_model, refiner, device, strict=True):
    ckpt = torch_load_trusted(path, map_location=device)
    state = ckpt["deepgd_state_dict"]
    model_has_paretofield = any(
        key.startswith("paretofield.") for key in deepgd_model.state_dict()
    )
    checkpoint_has_paretofield = any(key.startswith("paretofield.") for key in state)
    model_has_hctf = any(key.startswith("hctf_block.") for key in deepgd_model.state_dict())
    checkpoint_has_hctf = any(key.startswith("hctf_block.") for key in state)
    architecture_matches = (
        model_has_paretofield == checkpoint_has_paretofield
        and model_has_hctf == checkpoint_has_hctf
    )
    effective_strict = bool(strict) and architecture_matches
    incompatible = deepgd_model.load_state_dict(state, strict=effective_strict)
    if not effective_strict:
        print(
            "[INFO] checkpoint loaded with strict=False; "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    if refiner is not None and ckpt.get("refiner_state_dict") is not None:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=strict)

    print(f"[INFO] loaded checkpoint from: {path}")
    print(f"[INFO] checkpoint epoch = {ckpt.get('epoch', 'unknown')}")
    print(f"[INFO] checkpoint best_val = {ckpt.get('best_val', 'unknown')}")

    return ckpt


def select_validation_score(val_info, args):
    if args.select_by == "val_loss":
        return val_info["val_loss"]

    if args.select_by == "val_refined_stress":
        return val_info["val_refined_stress"]

    if args.select_by == "val_base_stress":
        return val_info["val_base_stress"]

    if args.select_by == "val_cpsg_loss":
        return val_info["val_cpsg_loss"]

    if args.select_by == "val_hctf_dominance":
        return val_info["val_hctf_dominance"]

    if args.select_by == "val_hctf_readability":
        return val_info["val_hctf_readability"]

    if args.select_by == "val_hctf_exact":
        return val_info["val_hctf_exact_score"]

    raise ValueError(f"Unknown select_by: {args.select_by}")


HCTF_PARAMETER_PREFIXES = (
    "hctf_block.",
    "hctf_conflict_head.",
    "hctf_graph_size_gate.",
    "hctf_stage_adapters.",
    "hctf_coordinate_",
    "hctf_exact_risk_gate.",
)


def is_hctf_parameter(name):
    return name.startswith(HCTF_PARAMETER_PREFIXES)


def configure_hctf_training_stage(optimizer, model, epoch, args):
    """Apply the v2 backbone freeze and residual ramp without rebuilding Adam."""
    warmup_epochs = max(int(getattr(args, "hctf_backbone_warmup_epochs", 0)), 0)
    for group in optimizer.param_groups:
        group_name = group.get("group_name", "")
        if group_name not in {"backbone", "refiner"}:
            continue
        target_lr = float(group.get("target_lr", group.get("lr", 0.0)))
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            group["lr"] = 0.0
            group["hctf_warmup_frozen"] = True
        elif group.pop("hctf_warmup_frozen", False) or float(group["lr"]) == 0.0:
            group["lr"] = target_lr

    residual_warmup = max(
        int(getattr(args, "hctf_residual_warmup_epochs", 0)), 0
    )
    progress = 1.0
    if residual_warmup > 0:
        progress = min(max(float(epoch) / float(residual_warmup), 0.0), 1.0)
    if hasattr(model, "set_hctf_residual_progress"):
        model.set_hctf_residual_progress(progress)
    return progress


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="rome",
        choices=[
            "rome",
            "north",
            "sbm",
            "fb15k237",
            "wn18rr",
            "imdb_binary",
            "proteins",
            "enzymes",
        ],
    )

    parser.add_argument("--index-file", type=str, default="assets/rome_index.txt")
    parser.add_argument("--layouts-file", type=str, default="assets/layouts/pmds.npy")

    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--history-dir", type=str, default="research_v2/graph_local/logs")
    parser.add_argument("--run-name", type=str, default="caer_deepgd")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="Warm-start training from model weights without resuming optimizer/epoch state.",
    )
    parser.add_argument("--inherit-args-from", type=str, default="",
                        help="Reuse saved checkpoint settings; explicit CLI flags override them.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-csv", type=str, default="results/caer_deepgd_test.csv")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=-1)
    parser.add_argument("--max-test-batches", type=int, default=-1)
    parser.add_argument("--train-only-pilot", action="store_true",
                        help="Train and save last checkpoint without reading validation or test metrics.")
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help=(
            "Stop after validation-selected checkpoint training. This keeps "
            "the test split untouched during candidate/policy development."
        ),
    )
    parser.add_argument("--memory-cleanup-every", type=int, default=0)
    parser.add_argument("--no-save-optimizer", action="store_true")
    parser.add_argument("--min-select-epoch", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; 0 disables it.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)

    parser.add_argument(
        "--community-source",
        choices=[
            "greedy",
            "louvain",
            "leiden",
            "label_propagation",
            "ground_truth",
            "random",
        ],
        default="greedy",
        help="Partition supplied to all community-aware model components.",
    )
    parser.add_argument(
        "--community-noise",
        type=float,
        default=0.0,
        help="Fraction of nodes reassigned to a different community after detection.",
    )
    parser.add_argument("--community-seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["random", "size_ood"],
        default="random",
        help="Use the standard random split or train on smaller graphs and test on larger graphs.",
    )

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
    parser.add_argument("--use-hctf", action="store_true")
    parser.add_argument(
        "--hctf-version",
        choices=["v1", "v2"],
        default="v1",
        help="Use the legacy synchronous HCTF block or the hierarchical v2 adapter.",
    )
    parser.add_argument("--hctf-num-trees", type=int, default=3)
    parser.add_argument("--hctf-hidden-dim", type=int, default=16)
    parser.add_argument("--hctf-dropout", type=float, default=0.05)
    parser.add_argument("--hctf-scale", type=float, default=0.20)
    parser.add_argument("--hctf-mix-init", type=float, default=0.05)
    parser.add_argument("--hctf-block-stride", type=int, default=1)
    parser.add_argument("--hctf-late-only", action="store_true")
    parser.add_argument(
        "--hctf-stochastic-depth-prob", type=float, default=0.0,
        help="Training-only per-graph drop probability for each HCTF insertion.",
    )
    parser.add_argument("--hctf-cache-size", type=int, default=4096)
    parser.add_argument("--hctf-graph-gate", action="store_true")
    parser.add_argument(
        "--hctf-adaptive-structure-gate",
        action="store_true",
        help=(
            "Use a nested node-only residual plus a graph-adaptive structural "
            "correction gate (HCTF-v1 only)."
        ),
    )
    parser.add_argument("--hctf-structure-gate-init", type=float, default=0.05)
    parser.add_argument(
        "--lambda-hctf-structure-gate-cost",
        type=float,
        default=0.0,
        help="Penalty on the mean adaptive structural gate (expected branch use).",
    )
    parser.add_argument(
        "--lambda-hctf-structure-gate-supervision", type=float, default=0.0
    )
    parser.add_argument("--hctf-structure-gate-components", default="0,4,9")
    parser.add_argument("--hctf-structure-gate-tolerance", type=float, default=0.0)
    parser.add_argument("--hctf-structure-gate-temperature", type=float, default=0.10)
    parser.add_argument("--hctf-zero-init", action="store_true")
    parser.add_argument("--hctf-coordinate-residual", action="store_true")
    parser.add_argument("--hctf-coordinate-max-step", type=float, default=0.08)
    parser.add_argument("--hctf-conflict-head", action="store_true",
                        help="Learn sparse nonincident-edge conflict messages inside HCTF forward.")
    parser.add_argument("--hctf-conflict-max-pairs", type=int, default=1024)
    parser.add_argument("--hctf-conflict-max-step", type=float, default=0.04)
    parser.add_argument("--hctf-conflict-deterministic-pairs", action="store_true",
                        help="Use reproducible per-graph conflict-pair selection.")
    parser.add_argument("--hctf-conflict-graph-context", action="store_true",
                        help="Condition conflict messages on graph size and conflict density.")
    parser.add_argument("--hctf-graph-size-gate", action="store_true",
                        help="Learn node-count-conditioned HCTF residual gain.")
    parser.add_argument("--hctf-graph-size-midpoint", type=float, default=20.0)
    parser.add_argument("--hctf-graph-size-width", type=float, default=5.0)
    parser.add_argument("--hctf-graph-size-gate-lr", type=float, default=0.0,
                        help="Optional separate size-gate learning rate; 0 inherits HCTF LR.")
    parser.add_argument("--freeze-hctf-block-for-conflict-head", action="store_true",
                        help="Single-factor pilot: train only the conflict head from a source checkpoint.")
    parser.add_argument("--hctf-exact-risk-gate", action="store_true")
    parser.add_argument("--hctf-disable-tree-messages", action="store_true")
    parser.add_argument("--hctf-disable-direction-conditioning", action="store_true")
    parser.add_argument("--hctf-disable-relation-conditioning", action="store_true")
    parser.add_argument("--hctf-disable-prototype-context", action="store_true")
    parser.add_argument(
        "--hctf-learn-direction-balance", action="store_true",
        help="Learn one shared top-down/bottom-up tree-flow preference (v1 only).",
    )
    parser.add_argument(
        "--hctf-unsaturated-degree-mean", action="store_true",
        help="Use log(1+degree)/3 instead of saturated tanh in the v1 graph gate.",
    )
    parser.add_argument(
        "--hctf-tree-identity-agnostic", action="store_true",
        help="Remove arbitrary tree ordinal from v1 relation conditioning.",
    )
    parser.add_argument(
        "--hctf-tree-coverage-relation", action="store_true",
        help="Use forest-edge multiplicity instead of tree ordinal in v1 relation conditioning.",
    )
    parser.add_argument(
        "--hctf-centered-prototype-context", action="store_true",
        help="Remove shared graph mean from v1 HCTF community prototype context.",
    )
    parser.add_argument(
        "--hctf-prototype-graph-mean-scale", type=float, default=1.0,
        help="Scale shared graph mean in v1 HCTF prototype context; 1 is legacy.",
    )
    parser.add_argument(
        "--hctf-independent-prototype-gate", action="store_true",
        help="Add a graphwise zero-logit gate for the v1 prototype input, initialized to one.",
    )
    parser.add_argument(
        "--hctf-route-graph-mean-to-node", action="store_true",
        help="Route v1 global graph context to node residual, keeping only community contrast in structural residual.",
    )
    parser.add_argument(
        "--hctf-relative-tree-messages", action="store_true",
        help="Use source-minus-target features for v1 forest message values; default keeps legacy source features.",
    )
    parser.add_argument(
        "--hctf-depth-scheduled-tree-flow", action="store_true",
        help="Use bounded leaf-to-root then root-to-leaf sweeps in each v1 forest; default keeps legacy one-hop aggregation.",
    )
    parser.add_argument(
        "--hctf-community-first-cover", action="store_true",
        help="Prefer intra-community edges when constructing each forest; default preserves the legacy cover.",
    )
    parser.add_argument(
        "--hctf-component-gradient-guard", action="store_true",
        help="Research-only: project the actual HCTF Adam step against nine training-batch surrogate gradients.",
    )
    parser.add_argument(
        "--hctf-node-relative-prototype", action="store_true",
        help="Use boundary-weighted own-community-minus-node features instead of absolute v1 prototype context.",
    )
    parser.add_argument(
        "--hctf-stage-residual-adapters", action="store_true",
        help="Zero-initialized feature-vector correction per hidden HCTF insertion stage.",
    )
    parser.add_argument(
        "--hctf-node-only-control",
        action="store_true",
        help=(
            "Ablation control: keep the residual MLP but remove tree messages, "
            "prototype context, and structural gate scalars."
        ),
    )
    parser.add_argument(
        "--hctf-preference",
        type=str,
        default="0.34,0.33,0.33",
        help="Geometry, visual, community preference for HCTF-v2 conditioning.",
    )
    parser.add_argument("--hctf-v2-disable-tree-attention", action="store_true")
    parser.add_argument(
        "--hctf-v2-disable-equivariant-coordinate", action="store_true"
    )
    parser.add_argument(
        "--hctf-lr",
        type=float,
        default=0.0,
        help="Explicit HCTF optimizer LR; zero uses deepgd_lr times the multiplier.",
    )
    parser.add_argument("--hctf-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--hctf-backbone-warmup-epochs", type=int, default=0)
    parser.add_argument("--hctf-residual-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--hctf-community-augmentation",
        type=float,
        default=0.0,
        help="Training-only per-graph partition-label perturbation for HCTF-v2.",
    )
    parser.add_argument("--lambda-hctf-exact-gate", type=float, default=0.0)
    parser.add_argument("--hctf-exact-gate-components", type=str, default="1,4,5")
    parser.add_argument("--hctf-exact-gate-tolerance", type=float, default=0.0)
    parser.add_argument("--hctf-exact-gate-temperature", type=float, default=0.10)
    parser.add_argument(
        "--freeze-hctf-proposal-for-risk-gate",
        action="store_true",
    )
    parser.add_argument("--freeze-backbone-for-hctf", action="store_true")
    parser.add_argument("--lambda-hctf-dominance", type=float, default=0.0)
    parser.add_argument("--lambda-hctf-anchor", type=float, default=0.0)
    parser.add_argument("--lambda-hctf-crossing-barrier", type=float, default=0.0)
    parser.add_argument("--hctf-crossing-barrier-margin", type=float, default=0.02)
    parser.add_argument("--hctf-crossing-preserve-only", action="store_true",
                        help="Penalize creation of new crossings without resolving every source crossing.")
    parser.add_argument("--hctf-crossing-barrier-tail-fraction", type=float, default=0.0,
                        help="If positive, average the worst fraction of pairwise barrier violations per graph.")
    parser.add_argument("--hctf-crossing-exact-max-edges", type=int, default=128)
    parser.add_argument("--hctf-cvar-fraction", type=float, default=0.20)
    parser.add_argument("--hctf-cvar-weight", type=float, default=0.50)
    parser.add_argument("--hctf-dominance-temperature", type=float, default=0.05)
    parser.add_argument("--hctf-target-improvement", type=float, default=0.002)
    parser.add_argument("--hctf-component-targets", type=str, default="")
    parser.add_argument("--hctf-component-weights", type=str, default="")
    parser.add_argument("--hctf-dominance-max-pairs", type=int, default=1024)
    parser.add_argument("--hctf-dominance-deterministic-pairs", action="store_true",
                        help="Use fixed per-graph edge-pair audit in dominance loss.")
    parser.add_argument("--hctf-dominance-only", action="store_true")
    parser.add_argument("--lambda-hctf-readability", type=float, default=0.0)
    parser.add_argument("--hctf-readability-dominance", action="store_true")
    parser.add_argument("--hctf-readability-target-improvement", type=float, default=0.0)
    parser.add_argument("--hctf-readability-start-epoch", type=int, default=1)
    parser.add_argument("--hctf-readability-warmup-epochs", type=int, default=5)
    parser.add_argument("--hctf-readability-max-pairs", type=int, default=4096)
    parser.add_argument("--hctf-readability-w-angular", type=float, default=1.0)
    parser.add_argument("--hctf-readability-w-crossing-angle", type=float, default=1.0)
    parser.add_argument("--hctf-readability-w-clearance", type=float, default=1.0)
    parser.add_argument("--hctf-readability-w-neighborhood", type=float, default=1.0)
    parser.add_argument("--hctf-readability-angular-target-ratio", type=float, default=0.5)
    parser.add_argument("--hctf-readability-angular-max-degrees", type=float, default=30.0)
    parser.add_argument(
        "--hctf-readability-crossing-angle-tail-fraction", type=float, default=0.20
    )
    parser.add_argument("--hctf-readability-clearance-margin", type=float, default=0.10)
    parser.add_argument("--hctf-readability-near-hops", type=float, default=2.0)
    parser.add_argument("--hctf-readability-neighborhood-margin", type=float, default=0.10)
    parser.add_argument("--use-care", action="store_true")
    parser.add_argument("--care-rank", type=int, default=2)
    parser.add_argument("--care-scale", type=float, default=0.10)
    parser.add_argument("--no-care-boundary-gate", action="store_true")
    parser.add_argument("--care-boundary-damp", type=float, default=0.50)
    parser.add_argument("--no-care-checkpoint", action="store_true")

    parser.add_argument("--use-cser", action="store_true")
    parser.add_argument(
        "--cser-mode",
        type=str,
        default="same",
        choices=["none", "same", "same_spectral", "full"],
    )
    parser.add_argument("--spectral-dim", type=int, default=4)
    parser.add_argument("--no-spectral-context", action="store_true")
    parser.add_argument("--use-adaspec-cser", action="store_true")
    parser.add_argument("--adaspec-alpha-s", type=float, default=0.03)
    parser.add_argument("--adaspec-alpha-f", type=float, default=0.08)
    parser.add_argument("--adaspec-self-loop-scale", type=float, default=1.0)
    parser.add_argument("--adaspec-boundary-weight", type=float, default=0.35)
    parser.add_argument("--adaspec-core-weight", type=float, default=0.10)
    parser.add_argument(
        "--use-topocser",
        action="store_true",
        help=(
            "Enable LEAP-inspired lightweight topology edge features "
            "(degree, clustering, boundary ratio, local density)."
        ),
    )

    parser.add_argument("--use-caid", action="store_true")
    parser.add_argument("--lambda-caid", type=float, default=0.15)
    parser.add_argument("--caid-intra-shrink", type=float, default=0.08)
    parser.add_argument("--caid-adj-expand", type=float, default=0.02)
    parser.add_argument("--caid-far-expand", type=float, default=0.10)
    parser.add_argument("--caid-min-ratio", type=float, default=0.90)
    parser.add_argument("--caid-max-ratio", type=float, default=1.12)
    parser.add_argument("--caid-warmup-epochs", type=int, default=20)
    parser.add_argument("--no-caid-edge-features", action="store_true")
    parser.add_argument(
        "--graph-local-community-objectives",
        action="store_true",
        help=(
            "Key CAID and CPSG communities by (graph id, local community id) "
            "to prevent cross-graph label collisions inside a mini-batch. "
            "Off by default to preserve legacy experiment reproduction."
        ),
    )

    parser.add_argument("--use-cpsg", action="store_true")
    parser.add_argument("--lambda-cpsg-proto", type=float, default=0.12)
    parser.add_argument("--lambda-cpsg-sil", type=float, default=0.45)
    parser.add_argument("--lambda-cpsg-sep", type=float, default=0.35)
    parser.add_argument("--cpsg-sil-margin", type=float, default=0.35)
    parser.add_argument("--cpsg-adj-margin", type=float, default=0.80)
    parser.add_argument("--cpsg-far-margin", type=float, default=1.60)
    parser.add_argument("--cpsg-core-gamma", type=float, default=1.50)
    parser.add_argument("--cpsg-boundary-floor", type=float, default=0.25)
    parser.add_argument("--cpsg-warmup-epochs", type=int, default=10)
    parser.add_argument("--use-macpsg", action="store_true")
    parser.add_argument("--macpsg-sep-scale-floor", type=float, default=1.0)

    parser.add_argument("--use-refiner", action="store_true")
    parser.add_argument("--refiner-hidden", type=int, default=64)
    parser.add_argument("--refiner-heads", type=int, default=4)

    parser.add_argument("--use-paretofield", action="store_true")
    parser.add_argument("--freeze-deepgd-for-paretofield", action="store_true")
    parser.add_argument("--paretofield-hidden", type=int, default=64)
    parser.add_argument("--paretofield-lr", type=float, default=0.0001)
    parser.add_argument("--paretofield-train-steps", type=int, default=1)
    parser.add_argument("--paretofield-inference-steps", type=int, default=4)
    parser.add_argument("--paretofield-step-size", type=float, default=0.10)
    parser.add_argument("--paretofield-max-pairs", type=int, default=4096)
    parser.add_argument("--paretofield-adaptive-tolerance", type=float, default=0.0)
    parser.add_argument("--paretofield-min-inference-steps", type=int, default=1)
    parser.add_argument(
        "--paretofield-soft-community",
        action="store_true",
        help="Use confidence- and boundary-aware soft community pair relations.",
    )
    parser.add_argument(
        "--paretofield-relation-consensus",
        action="store_true",
        help=(
            "Learn a bounded topology-evidence correction for uncertain "
            "community pair relations."
        ),
    )
    parser.add_argument(
        "--paretofield-relation-consensus-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--paretofield-graph-preference-calibration",
        action="store_true",
        help="Learn a bounded graph-conditioned correction to objective preferences.",
    )
    parser.add_argument(
        "--paretofield-preference-calibration-scale", type=float, default=0.75
    )
    parser.add_argument(
        "--paretofield-edge-pair-field",
        action="store_true",
        help="Add a sparse E(2)-equivariant edge-pair branch for crossings.",
    )
    parser.add_argument(
        "--paretofield-edge-pair-max-pairs", type=int, default=384
    )
    parser.add_argument(
        "--paretofield-edge-pair-oversample-factor", type=int, default=4
    )
    parser.add_argument("--paretofield-edge-pair-scale", type=float, default=0.15)
    parser.add_argument("--paretofield-preference", type=str, default="0.34,0.33,0.33")
    parser.add_argument("--paretofield-sample-preferences", action="store_true")
    parser.add_argument("--paretofield-dirichlet-alpha", type=float, default=0.70)
    parser.add_argument("--lambda-paretofield", type=float, default=0.10)
    parser.add_argument("--paretofield-regret-weight", type=float, default=0.0)
    parser.add_argument(
        "--paretofield-regret-max-geometry", type=float, default=0.0
    )
    parser.add_argument(
        "--paretofield-regret-max-visual", type=float, default=0.01
    )
    parser.add_argument(
        "--paretofield-regret-max-community", type=float, default=0.01
    )
    parser.add_argument(
        "--paretofield-regret-max-weighted", type=float, default=0.0
    )
    parser.add_argument("--paretofield-anchor-weight", type=float, default=0.05)
    parser.add_argument("--paretofield-teacher-weight", type=float, default=1.0)
    parser.add_argument("--paretofield-teacher-archive", type=str, default="")
    parser.add_argument("--paretofield-reference-divisions", type=int, default=0)
    parser.add_argument("--paretofield-max-reference-layouts", type=int, default=12)
    parser.add_argument("--paretofield-front-csv", type=str, default="")
    parser.add_argument("--paretofield-safe-guard", action="store_true")
    parser.add_argument(
        "--paretofield-continuous-guard",
        action="store_true",
        help="Search safe interpolation steps instead of binary accept/reject.",
    )
    parser.add_argument(
        "--paretofield-guard-step-fractions",
        type=str,
        default="0,0.0625,0.125,0.25,0.5,0.75,1",
    )
    parser.add_argument(
        "--paretofield-guard-mode",
        type=str,
        default="stress",
        choices=["stress", "pareto"],
    )
    parser.add_argument("--paretofield-guard-crossing", action="store_true")
    parser.add_argument(
        "--paretofield-exact-crossing-guard",
        action="store_true",
        help=(
            "Use one fixed edge-pair audit set and recompute crossing at every "
            "continuous guard step."
        ),
    )
    parser.add_argument(
        "--paretofield-componentwise-guard",
        action="store_true",
        help=(
            "Protect stress, crossing, edge CV, occlusion, compactness, "
            "separation, separation/compactness, and silhouette separately."
        ),
    )
    parser.add_argument(
        "--paretofield-guard-max-geometry-regret", type=float, default=0.0
    )
    parser.add_argument(
        "--paretofield-guard-max-visual-regret", type=float, default=0.01
    )
    parser.add_argument(
        "--paretofield-guard-max-community-regret", type=float, default=0.01
    )
    parser.add_argument(
        "--paretofield-guard-max-weighted-regret", type=float, default=0.0
    )
    parser.add_argument(
        "--paretofield-conflict-method",
        type=str,
        default="weighted",
        choices=["weighted", "pcgrad", "config"],
    )

    parser.add_argument("--gps-delta-scale", type=float, default=0.20)
    parser.add_argument("--ori-delta-scale", type=float, default=0.10)
    parser.add_argument("--intra-delta-scale", type=float, default=0.10)
    parser.add_argument("--inter-delta-scale", type=float, default=0.04)
    parser.add_argument("--spectral-delta-scale", type=float, default=0.04)

    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--intra-knn-k", type=int, default=3)
    parser.add_argument("--inter-comm-k", type=int, default=2)
    parser.add_argument("--spectral-knn-k", type=int, default=3)

    parser.add_argument("--lambda-stress", type=float, default=0.7)
    parser.add_argument(
        "--stress-graph-balance-power", type=float, default=0.0,
        help=(
            "Training-only per-graph APSP-pair-count balancing exponent in [0,1]; "
            "0 keeps legacy Stress. Reported/selection Stress is unchanged."
        ),
    )
    parser.add_argument("--use-rd-stress", action="store_true")
    parser.add_argument("--lambda-rd-stress", type=float, default=0.0)
    parser.add_argument("--rd-max-pairs", type=int, default=4096)
    parser.add_argument(
        "--rd-normalize",
        type=str,
        default="mean",
        choices=["mean", "median", "none"],
    )
    parser.add_argument("--use-cosine-stress", action="store_true")
    parser.add_argument("--lambda-cosine-stress", type=float, default=0.0)
    parser.add_argument("--cosine-max-pairs", type=int, default=4096)
    parser.add_argument("--cosine-walk-steps", type=int, default=4)
    parser.add_argument("--cosine-walk-decay", type=float, default=0.75)
    parser.add_argument(
        "--cosine-normalize",
        type=str,
        default="mean",
        choices=["mean", "median", "none"],
    )
    parser.add_argument("--lambda-intra", type=float, default=0.6)
    parser.add_argument("--lambda-inter", type=float, default=0.4)
    parser.add_argument("--lambda-pair", type=float, default=0.05)

    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--lambda-delta-reg", type=float, default=0.01)
    parser.add_argument("--lambda-balance", type=float, default=0.001)
    parser.add_argument("--lambda-base-aux", type=float, default=0.5)
    parser.add_argument("--lambda-pmds-anchor", type=float, default=0.0)
    parser.add_argument("--pmds-anchor-huber-delta", type=float, default=0.0)
    parser.add_argument(
        "--pmds-anchor-min-nodes", type=int, default=0,
        help="Training only: exclude anchor penalties on graphs with fewer nodes; 0 preserves legacy loss.",
    )
    parser.add_argument(
        "--pmds-anchor-center-per-graph", action="store_true",
        help="Training only: penalize within-graph PMDS displacement after removing arbitrary translation.",
    )
    parser.add_argument(
        "--hctf-centered-rescale", action="store_true",
        help=(
            "Opt-in HCTF rescaling about each graph centroid, returning to the "
            "input-layout centroid; avoids magnifying arbitrary translation."
        ),
    )
    parser.add_argument("--pmds-anchor-start-epoch", type=int, default=1)
    parser.add_argument("--pmds-anchor-warmup-epochs", type=int, default=1)
    parser.add_argument("--lambda-crossing", type=float, default=0.0)
    parser.add_argument(
        "--hctf-crossing-projected-ratio", type=float, default=0.0,
        help=(
            "Opt-in Stress-protected crossing gradient target norm as a fraction "
            "of the Stress gradient norm (frozen-backbone HCTF only)."
        ),
    )
    parser.add_argument("--crossing-max-pairs", type=int, default=2048)
    parser.add_argument("--crossing-oversample-factor", type=int, default=8)
    parser.add_argument("--crossing-temperature", type=float, default=8.0)
    parser.add_argument("--crossing-margin", type=float, default=0.25)
    parser.add_argument("--crossing-start-epoch", type=int, default=5)
    parser.add_argument("--crossing-warmup-epochs", type=int, default=10)
    parser.add_argument("--hard-crossing-sampling", action="store_true")
    parser.add_argument("--hard-crossing-ratios", type=str, default="0.40,0.30,0.20,0.10")
    parser.add_argument("--lambda-train-silhouette", type=float, default=0.0)
    parser.add_argument("--train-silhouette-margin", type=float, default=0.35)
    parser.add_argument("--train-silhouette-start-epoch", type=int, default=8)
    parser.add_argument("--train-silhouette-warmup-epochs", type=int, default=20)

    parser.add_argument("--margin", type=float, default=4.8)
    parser.add_argument("--pair-margin", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=4096)

    parser.add_argument(
        "--select-by",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "val_refined_stress",
            "val_base_stress",
            "val_cpsg_loss",
            "val_hctf_dominance",
            "val_hctf_readability",
            "val_hctf_exact",
        ],
    )
    parser.add_argument(
        "--save-every-epoch", action="store_true",
        help="Keep each epoch checkpoint for a separate preregistered Raw selection audit.",
    )

    parser.add_argument("--rescale-before-refiner", action="store_true")
    parser.add_argument("--use-cgrcc", action="store_true")
    parser.add_argument(
        "--cgrcc-scale-candidates",
        type=str,
        default="1.00,1.02,1.04,1.06",
    )
    parser.add_argument(
        "--cgrcc-shrink-candidates",
        type=str,
        default="1.00,0.99,0.98",
    )
    parser.add_argument("--cgrcc-crossing-allowance", type=int, default=0)
    parser.add_argument("--cgrcc-max-sil-drop", type=float, default=0.0025)
    parser.add_argument("--cgrcc-min-ratio-gain", type=float, default=0.0)
    parser.add_argument("--cgrcc-max-stress-ratio", type=float, default=1.03)
    parser.add_argument("--cgrcc-stress-weight", type=float, default=0.10)
    parser.add_argument("--use-mtpf", action="store_true")
    parser.add_argument(
        "--mtpf-profile",
        type=str,
        default="balanced",
        choices=["balanced", "allfirst", "stress", "crossing", "community", "thesis", "auto"],
    )
    parser.add_argument(
        "--mtpf-teachers",
        type=str,
        default="neural,pmds,kk,fr,blend,shell",
    )
    parser.add_argument(
        "--mtpf-selection-mode",
        type=str,
        default="weighted",
        choices=["weighted", "rank", "pareto", "thesis", "bc_pareto", "gain_gate"],
    )
    parser.add_argument("--mtpf-pareto-dominated-penalty", type=float, default=1.0)
    parser.add_argument("--mtpf-blend-alphas", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--use-mtpf-pair-blends", action="store_true")
    parser.add_argument("--mtpf-fr-seeds", type=int, default=1)
    parser.add_argument("--mtpf-fr-iterations", type=int, default=100)
    parser.add_argument("--fling-root", type=str, default="fling-layout-main")
    parser.add_argument("--fling-device", type=str, default="cpu")
    parser.add_argument("--mtpf-fling-iterations", type=int, default=200)
    parser.add_argument("--mtpf-fling-blend-lambda", type=float, default=0.50)
    parser.add_argument("--mtpf-shell-scales", type=str, default="1.00,1.02,1.04,1.06")
    parser.add_argument("--mtpf-shell-shrinks", type=str, default="1.00,0.99,0.98")
    parser.add_argument("--mtpf-boundary-pushes", type=str, default="0.00,0.04,0.08")
    parser.add_argument("--mtpf-radial-scales", type=str, default="0.80,1.00,1.20")
    parser.add_argument("--mtpf-radial-shrinks", type=str, default="0.94,0.98,1.00")
    parser.add_argument("--mtpf-max-candidates", type=int, default=0)
    parser.add_argument("--mtpf-max-stress-ratio", type=float, default=3.50)
    parser.add_argument("--no-mtpf-neural-guard", action="store_true")
    parser.add_argument("--mtpf-strict-neural-guard", action="store_true")
    parser.add_argument("--mtpf-max-neural-stress-ratio", type=float, default=1.15)
    parser.add_argument("--mtpf-max-crossing-count-ratio", type=float, default=0.0)
    parser.add_argument("--mtpf-max-crossing-norm-ratio", type=float, default=1.15)
    parser.add_argument("--mtpf-max-edgecv-ratio", type=float, default=1.25)
    parser.add_argument("--mtpf-max-occlusion-delta", type=float, default=0.02)
    parser.add_argument("--mtpf-max-compactness-ratio", type=float, default=0.0)
    parser.add_argument("--mtpf-min-separation-ratio", type=float, default=0.0)
    parser.add_argument("--mtpf-min-sep-ratio-ratio", type=float, default=0.75)
    parser.add_argument("--mtpf-max-sil-drop", type=float, default=0.04)
    parser.add_argument("--mtpf-sep-ratio-cap", type=float, default=20.0)
    parser.add_argument("--mtpf-max-regret-weight", type=float, default=0.35)
    parser.add_argument("--mtpf-improvement-bonus", type=float, default=1.25)
    parser.add_argument("--mtpf-bc-baselines", type=str, default="kk,fr,pmds")
    parser.add_argument("--mtpf-bc-crossing-count-weight", type=float, default=0.35)
    parser.add_argument("--mtpf-bc-max-regret-weight", type=float, default=1.0)
    parser.add_argument("--mtpf-bc-improvement-bonus", type=float, default=0.25)
    parser.add_argument("--mtpf-bc-regret-cap", type=float, default=2.0)
    parser.add_argument("--mtpf-gain-penalty-weight", type=float, default=4.0)
    parser.add_argument("--mtpf-gain-min-score", type=float, default=0.0)
    parser.add_argument("--mtpf-w-stress", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-crossing", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-edgecv", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-occlusion", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-compactness", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-separation", type=float, default=-1.0)
    parser.add_argument("--mtpf-w-silhouette", type=float, default=-1.0)
    parser.add_argument("--mtpf-refine-steps", type=int, default=0)
    parser.add_argument(
        "--mtpf-refine-mode",
        type=str,
        default="adam",
        choices=["adam", "sso", "stochastic", "stochastic_stress"],
    )
    parser.add_argument("--mtpf-refine-top-k", type=int, default=1)
    parser.add_argument("--mtpf-refine-profiles", type=str, default="default")
    parser.add_argument("--mtpf-refine-lr", type=float, default=0.02)
    parser.add_argument("--mtpf-refine-crossing-pairs", type=int, default=4096)
    parser.add_argument("--mtpf-refine-grad-clip", type=float, default=1.0)
    parser.add_argument("--mtpf-sso-pairs", type=int, default=2048)
    parser.add_argument("--mtpf-sso-anchor", type=float, default=0.10)
    parser.add_argument("--mtpf-refine-w-stress", type=float, default=1.0)
    parser.add_argument("--mtpf-refine-w-crossing", type=float, default=0.35)
    parser.add_argument("--mtpf-refine-w-edgecv", type=float, default=0.15)
    parser.add_argument("--mtpf-refine-w-community", type=float, default=0.08)
    parser.add_argument("--mtpf-refine-w-silhouette", type=float, default=0.0)
    parser.add_argument("--mtpf-refine-silhouette-margin", type=float, default=-1.0)
    parser.add_argument("--mtpf-refine-w-occlusion", type=float, default=0.10)
    parser.add_argument("--mtpf-refine-w-anchor", type=float, default=0.15)
    parser.add_argument("--no-mtpf-refine-guard", action="store_true")
    parser.add_argument("--mtpf-refine-max-stress-ratio", type=float, default=1.03)
    parser.add_argument("--mtpf-refine-max-crossing-ratio", type=float, default=1.03)
    parser.add_argument("--mtpf-refine-max-edgecv-ratio", type=float, default=1.05)
    parser.add_argument("--mtpf-refine-max-occlusion-delta", type=float, default=0.005)
    parser.add_argument("--mtpf-refine-min-sep-ratio", type=float, default=0.95)
    parser.add_argument("--mtpf-refine-max-sil-drop", type=float, default=0.015)
    parser.add_argument("--mtpf-spread-refine", action="store_true")
    parser.add_argument("--mtpf-spread-top-k", type=int, default=8)
    parser.add_argument("--mtpf-spread-divisions", type=int, default=4)
    parser.add_argument("--mtpf-spread-steps", type=int, default=8)
    parser.add_argument("--mtpf-spread-lr", type=float, default=0.006)
    parser.add_argument("--mtpf-spread-repulsion", type=float, default=0.05)
    parser.add_argument("--mtpf-spread-sigma", type=float, default=1.0)
    parser.add_argument("--mtpf-spread-anchor", type=float, default=0.10)

    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--inherit-args-from", type=str, default="")
    preliminary, _ = preparser.parse_known_args()
    if preliminary.inherit_args_from:
        source_path = Path(preliminary.inherit_args_from)
        if not source_path.is_file():
            parser.error(f"inheritance checkpoint not found: {source_path}")
        inherited = torch_load_trusted(str(source_path), map_location="cpu").get("args")
        if not isinstance(inherited, dict):
            parser.error("inheritance checkpoint has no saved argument dictionary")
        valid_names = {action.dest for action in parser._actions}
        parser.set_defaults(**{name: value for name, value in inherited.items()
                               if name in valid_names})
    args = parser.parse_args()

    if not args.graph_local_community_objectives:
        parser.error("HCTF-GD-GraphLocal-v1 requires --graph-local-community-objectives")

    if args.hctf_conflict_head and not args.use_hctf:
        parser.error("--hctf-conflict-head requires --use-hctf")
    if (args.hctf_conflict_deterministic_pairs or args.hctf_conflict_graph_context) and not args.hctf_conflict_head:
        parser.error("conflict pair/context flags require --hctf-conflict-head")
    if args.hctf_graph_size_midpoint <= 0 or args.hctf_graph_size_width <= 0:
        parser.error("graph-size gate midpoint and width must be positive")
    if args.hctf_graph_size_gate_lr < 0:
        parser.error("graph-size gate learning rate must be nonnegative")
    if not 0.0 <= args.hctf_crossing_barrier_tail_fraction <= 1.0:
        parser.error("crossing barrier tail fraction must lie in [0, 1]")
    if args.hctf_conflict_max_pairs <= 0 or args.hctf_conflict_max_step < 0:
        parser.error("conflict max pairs must be positive and max step nonnegative")
    if args.freeze_hctf_block_for_conflict_head and not (
        args.hctf_conflict_head and args.freeze_backbone_for_hctf
    ):
        parser.error("conflict-head-only training requires the head and frozen backbone")
    if args.train_only_pilot and not args.skip_final_test:
        parser.error("--train-only-pilot requires --skip-final-test")

    if not 0.0 <= args.stress_graph_balance_power <= 1.0:
        parser.error("--stress-graph-balance-power must be in [0,1]")
    if not math.isfinite(args.pmds_anchor_huber_delta) or args.pmds_anchor_huber_delta < 0.0:
        parser.error("--pmds-anchor-huber-delta must be finite and nonnegative")
    if args.pmds_anchor_min_nodes < 0:
        parser.error("--pmds-anchor-min-nodes must be nonnegative")

    if args.hctf_num_trees < 1:
        parser.error("--hctf-num-trees must be at least 1")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")
    if args.early_stopping_min_delta < 0.0:
        parser.error("--early-stopping-min-delta cannot be negative")
    if not 0.0 <= args.community_noise <= 1.0:
        parser.error("--community-noise must be in [0, 1]")
    if args.hctf_hidden_dim < 1:
        parser.error("--hctf-hidden-dim must be at least 1")
    if not 0.0 <= args.hctf_dropout < 1.0:
        parser.error("--hctf-dropout must be in [0, 1)")
    if args.hctf_scale < 0.0:
        parser.error("--hctf-scale cannot be negative")
    if args.hctf_mix_init < 0.0 or args.hctf_mix_init > args.hctf_scale:
        parser.error("--hctf-mix-init must be in [0, hctf-scale]")
    if args.hctf_block_stride < 1:
        parser.error("--hctf-block-stride must be at least 1")
    if args.hctf_tree_identity_agnostic and args.hctf_tree_coverage_relation:
        parser.error("choose identity-agnostic or coverage relation, not both")
    if not math.isfinite(args.hctf_stochastic_depth_prob) or not 0.0 <= args.hctf_stochastic_depth_prob < 1.0:
        parser.error("--hctf-stochastic-depth-prob must be finite and in [0, 1)")
    if args.hctf_cache_size < 0:
        parser.error("--hctf-cache-size cannot be negative")
    if args.hctf_coordinate_max_step < 0.0:
        parser.error("--hctf-coordinate-max-step cannot be negative")
    if not 0.0 < args.hctf_structure_gate_init < 1.0:
        parser.error("--hctf-structure-gate-init must lie strictly inside (0, 1)")
    if not math.isfinite(args.hctf_prototype_graph_mean_scale) or not 0.0 <= args.hctf_prototype_graph_mean_scale <= 1.0:
        parser.error("--hctf-prototype-graph-mean-scale must be in [0, 1]")
    if args.lambda_hctf_structure_gate_cost < 0.0:
        parser.error("--lambda-hctf-structure-gate-cost cannot be negative")
    if args.lambda_hctf_structure_gate_supervision < 0.0:
        parser.error("--lambda-hctf-structure-gate-supervision cannot be negative")
    if args.hctf_structure_gate_tolerance < 0.0:
        parser.error("--hctf-structure-gate-tolerance cannot be negative")
    if args.hctf_structure_gate_temperature <= 0.0:
        parser.error("--hctf-structure-gate-temperature must be positive")
    try:
        parse_hctf_exact_gate_components(args.hctf_structure_gate_components)
    except ValueError as error:
        parser.error(str(error).replace("exact-gate", "structure-gate"))
    if args.hctf_adaptive_structure_gate and args.hctf_version != "v1":
        parser.error("--hctf-adaptive-structure-gate currently requires HCTF-v1")
    if args.hctf_relative_tree_messages and not (
        args.use_hctf and args.hctf_version == "v1"
        and not args.hctf_disable_tree_messages and not args.hctf_node_only_control
    ):
        parser.error(
            "--hctf-relative-tree-messages requires HCTF-v1 tree messages "
            "and no node-only control"
        )
    if args.hctf_depth_scheduled_tree_flow and not (
        args.use_hctf and args.hctf_version == "v1"
        and not args.hctf_disable_tree_messages and not args.hctf_node_only_control
    ):
        parser.error(
            "--hctf-depth-scheduled-tree-flow requires HCTF-v1 tree messages "
            "and no node-only control"
        )
    if args.hctf_component_gradient_guard and not (
        args.mode == "train"
        and args.hctf_depth_scheduled_tree_flow
        and args.freeze_backbone_for_hctf
        and not args.use_refiner
        and not args.use_paretofield
        and args.hctf_crossing_projected_ratio == 0.0
    ):
        parser.error(
            "--hctf-component-gradient-guard requires frozen HCTF-v1 "
            "depth flow, no refiner/ParetoField, and no crossing projection"
        )
    if args.hctf_stage_residual_adapters and not (
        args.use_hctf and not args.hctf_coordinate_residual
    ):
        parser.error("--hctf-stage-residual-adapters requires hidden-block HCTF")
    if args.hctf_node_relative_prototype and not (
        args.use_hctf and args.hctf_version == "v1"
        and not args.hctf_disable_prototype_context
        and not args.hctf_node_only_control
        and not args.hctf_centered_prototype_context
        and args.hctf_prototype_graph_mean_scale == 1.0
        and not args.hctf_route_graph_mean_to_node
        and not args.hctf_independent_prototype_gate
        and not args.hctf_relative_tree_messages
    ):
        parser.error(
            "--hctf-node-relative-prototype requires baseline HCTF-v1 "
            "prototype/tree settings and no other prototype variant"
        )
    if args.hctf_independent_prototype_gate and not (
        args.use_hctf
        and args.hctf_version == "v1"
        and not args.hctf_disable_prototype_context
        and not args.hctf_node_only_control
    ):
        parser.error(
            "--hctf-independent-prototype-gate requires HCTF-v1 with "
            "prototype context enabled and without node-only control"
        )
    if args.hctf_route_graph_mean_to_node and not (
        args.use_hctf
        and args.hctf_version == "v1"
        and args.hctf_adaptive_structure_gate
        and not args.hctf_disable_prototype_context
        and not args.hctf_node_only_control
        and not args.hctf_centered_prototype_context
        and args.hctf_prototype_graph_mean_scale == 1.0
        and not args.hctf_independent_prototype_gate
    ):
        parser.error(
            "--hctf-route-graph-mean-to-node requires adaptive HCTF-v1 "
            "with unchanged prototype settings and no independent prototype gate"
        )
    if args.lambda_hctf_structure_gate_supervision > 0.0 and not (
        args.hctf_adaptive_structure_gate and args.freeze_backbone_for_hctf
    ):
        parser.error(
            "--lambda-hctf-structure-gate-supervision requires the adaptive gate "
            "and --freeze-backbone-for-hctf for an exact matched counterfactual"
        )
    if args.hctf_lr < 0.0:
        parser.error("--hctf-lr cannot be negative")
    if args.hctf_lr_multiplier <= 0.0:
        parser.error("--hctf-lr-multiplier must be positive")
    if args.hctf_backbone_warmup_epochs < 0:
        parser.error("--hctf-backbone-warmup-epochs cannot be negative")
    if args.hctf_residual_warmup_epochs < 0:
        parser.error("--hctf-residual-warmup-epochs cannot be negative")
    if not 0.0 <= args.hctf_community_augmentation <= 1.0:
        parser.error("--hctf-community-augmentation must be in [0, 1]")
    try:
        hctf_preference = parse_float_list(args.hctf_preference)
    except ValueError as error:
        parser.error(f"invalid --hctf-preference: {error}")
    if len(hctf_preference) != 3 or any(value < 0.0 for value in hctf_preference):
        parser.error("--hctf-preference must contain three non-negative values")
    if sum(hctf_preference) <= 0.0:
        parser.error("--hctf-preference must have a positive sum")
    if args.lambda_hctf_exact_gate < 0.0:
        parser.error("--lambda-hctf-exact-gate cannot be negative")
    if args.hctf_exact_gate_tolerance < 0.0:
        parser.error("--hctf-exact-gate-tolerance cannot be negative")
    if args.hctf_exact_gate_temperature <= 0.0:
        parser.error("--hctf-exact-gate-temperature must be positive")
    try:
        parse_hctf_exact_gate_components(args.hctf_exact_gate_components)
    except ValueError as error:
        parser.error(str(error))
    if args.lambda_hctf_dominance < 0.0:
        parser.error("--lambda-hctf-dominance cannot be negative")
    if args.lambda_hctf_anchor < 0.0:
        parser.error("--lambda-hctf-anchor cannot be negative")
    if args.lambda_hctf_crossing_barrier < 0.0:
        parser.error("--lambda-hctf-crossing-barrier cannot be negative")
    if args.hctf_crossing_barrier_margin < 0.0:
        parser.error("--hctf-crossing-barrier-margin cannot be negative")
    if args.hctf_crossing_exact_max_edges < 2:
        parser.error("--hctf-crossing-exact-max-edges must be at least 2")
    if not 0.0 <= args.hctf_cvar_fraction <= 1.0:
        parser.error("--hctf-cvar-fraction must be in [0, 1]")
    if args.hctf_cvar_weight < 0.0:
        parser.error("--hctf-cvar-weight cannot be negative")
    if args.hctf_dominance_temperature <= 0.0:
        parser.error("--hctf-dominance-temperature must be positive")
    if args.hctf_target_improvement < 0.0:
        parser.error("--hctf-target-improvement cannot be negative")
    if args.hctf_dominance_max_pairs < 1:
        parser.error("--hctf-dominance-max-pairs must be at least 1")
    if args.lambda_hctf_readability < 0.0:
        parser.error("--lambda-hctf-readability cannot be negative")
    readability_component_weights = [
        args.hctf_readability_w_angular,
        args.hctf_readability_w_crossing_angle,
        args.hctf_readability_w_clearance,
        args.hctf_readability_w_neighborhood,
    ]
    if any(weight < 0.0 for weight in readability_component_weights):
        parser.error("HCTF readability component weights cannot be negative")
    if args.lambda_hctf_readability > 0.0 and sum(readability_component_weights) <= 0.0:
        parser.error("active HCTF readability training requires a positive component weight")
    if args.lambda_hctf_readability > 0.0 and not args.use_hctf:
        parser.error("--lambda-hctf-readability requires --use-hctf")
    if args.hctf_readability_target_improvement < 0.0:
        parser.error("--hctf-readability-target-improvement cannot be negative")
    if args.hctf_readability_dominance and args.lambda_hctf_dominance <= 0.0:
        parser.error("--hctf-readability-dominance requires --lambda-hctf-dominance > 0")
    if args.hctf_readability_start_epoch < 1:
        parser.error("--hctf-readability-start-epoch must be at least 1")
    if args.hctf_readability_warmup_epochs < 0:
        parser.error("--hctf-readability-warmup-epochs cannot be negative")
    if args.hctf_readability_max_pairs < 1:
        parser.error("--hctf-readability-max-pairs must be at least 1")
    if not 0.0 < args.hctf_readability_angular_target_ratio <= 1.0:
        parser.error("--hctf-readability-angular-target-ratio must be in (0, 1]")
    if args.hctf_readability_angular_max_degrees <= 0.0:
        parser.error("--hctf-readability-angular-max-degrees must be positive")
    if not 0.0 < args.hctf_readability_crossing_angle_tail_fraction <= 1.0:
        parser.error(
            "--hctf-readability-crossing-angle-tail-fraction must be in (0, 1]"
        )
    if args.hctf_readability_clearance_margin < 0.0:
        parser.error("--hctf-readability-clearance-margin cannot be negative")
    if args.hctf_readability_near_hops <= 0.0:
        parser.error("--hctf-readability-near-hops must be positive")
    if args.hctf_readability_neighborhood_margin < 0.0:
        parser.error("--hctf-readability-neighborhood-margin cannot be negative")
    if args.freeze_backbone_for_hctf and not args.use_hctf:
        parser.error("--freeze-backbone-for-hctf requires --use-hctf")
    if args.hctf_learn_direction_balance and (
        not args.use_hctf or args.hctf_version != "v1"
    ):
        parser.error("--hctf-learn-direction-balance requires HCTF-v1")
    if args.hctf_centered_rescale and (
        not args.use_hctf
        or not args.freeze_backbone_for_hctf
        or not args.rescale_before_refiner
    ):
        parser.error(
            "--hctf-centered-rescale requires frozen-backbone HCTF "
            "and --rescale-before-refiner"
        )
    if not math.isfinite(args.hctf_crossing_projected_ratio) or args.hctf_crossing_projected_ratio < 0.0:
        parser.error("--hctf-crossing-projected-ratio must be finite and non-negative")
    if args.hctf_crossing_projected_ratio > 0.0 and (
        not args.use_hctf
        or not args.freeze_backbone_for_hctf
        or args.lambda_crossing <= 0.0
    ):
        parser.error(
            "--hctf-crossing-projected-ratio requires frozen-backbone HCTF "
            "and --lambda-crossing > 0"
        )
    if (
        args.lambda_hctf_dominance > 0.0
        and not args.freeze_backbone_for_hctf
        and args.hctf_version != "v2"
    ):
        parser.error(
            "joint dominance training is supported only by HCTF-v2; "
            "freeze the backbone for HCTF-v1"
        )
    if args.hctf_dominance_only and args.lambda_hctf_dominance <= 0.0:
        parser.error(
            "--hctf-dominance-only requires --lambda-hctf-dominance > 0"
        )
    if args.hctf_exact_risk_gate and not args.hctf_coordinate_residual:
        parser.error("--hctf-exact-risk-gate requires --hctf-coordinate-residual")
    if args.lambda_hctf_exact_gate > 0.0 and not (
        args.hctf_exact_risk_gate
        and (args.freeze_backbone_for_hctf or args.hctf_version == "v2")
        and args.lambda_hctf_dominance > 0.0
    ):
        parser.error(
            "--lambda-hctf-exact-gate requires the exact risk gate and frozen HCTF teacher"
        )
    if args.freeze_hctf_proposal_for_risk_gate and not (
        args.hctf_exact_risk_gate and args.lambda_hctf_exact_gate > 0.0
    ):
        parser.error(
            "--freeze-hctf-proposal-for-risk-gate requires an active exact risk gate"
        )
    if args.select_by == "val_hctf_exact" and not (
        args.use_hctf
        and (args.freeze_backbone_for_hctf or args.hctf_version == "v2")
        and args.lambda_hctf_dominance > 0.0
    ):
        parser.error(
            "--select-by val_hctf_exact requires frozen HCTF dominance training"
        )

    if args.dataset == "north":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/north_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/north_pmds.npy"
    elif args.dataset == "sbm":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/sbm_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/sbm_pmds.npy"
    elif args.dataset == "fb15k237":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/fb15k237_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/fb15k237_pmds.npy"
    elif args.dataset == "wn18rr":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/wn18rr_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/wn18rr_pmds.npy"
    elif args.dataset == "imdb_binary":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/imdb_binary_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/imdb_binary_pmds.npy"
    elif args.dataset == "proteins":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/proteins_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/proteins_pmds.npy"
    elif args.dataset == "enzymes":
        if args.index_file == "assets/rome_index.txt":
            args.index_file = "assets/enzymes_index.txt"
        if args.layouts_file == "assets/layouts/pmds.npy":
            args.layouts_file = "assets/layouts/enzymes_pmds.npy"

    set_random_seed(args.seed)

    device = get_device()
    print(f"[INFO] device = {device}")

    ensure_dir(args.save_dir)
    ensure_dir(args.history_dir)
    ensure_dir("results")
    ensure_dir("logs")

    train_data, val_data, test_data = load_graph_dataset(args)

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

    if args.init_ckpt:
        load_checkpoint(args.init_ckpt, deepgd_model, refiner, device, strict=False)
        print(f"[INFO] initialized training weights from: {args.init_ckpt}")

    if args.use_hctf and args.freeze_backbone_for_hctf:
        for name, parameter in deepgd_model.named_parameters():
            if not name.startswith(
                (
                    "hctf_block.",
                    "hctf_stage_adapters.",
                    "hctf_coordinate_projection.",
                    "hctf_coordinate_trust.",
                    "hctf_exact_risk_gate.",
                    "hctf_conflict_head.",
                )
            ):
                parameter.requires_grad_(False)
        if refiner is not None:
            for parameter in refiner.parameters():
                parameter.requires_grad_(False)
        print("[INFO] froze current-model backbone; training HCTF branch only")

    if args.freeze_hctf_block_for_conflict_head:
        for name, parameter in deepgd_model.named_parameters():
            parameter.requires_grad_(name.startswith(("hctf_conflict_head.", "hctf_graph_size_gate.")))
        print("[INFO] froze original HCTF; training conflict head only")

    if args.freeze_hctf_proposal_for_risk_gate:
        for name, parameter in deepgd_model.named_parameters():
            parameter.requires_grad_(name.startswith("hctf_exact_risk_gate."))
        print("[INFO] froze HCTF proposal; training exact risk gate only")

    if args.use_paretofield and args.freeze_deepgd_for_paretofield:
        for name, parameter in deepgd_model.named_parameters():
            if not name.startswith("paretofield."):
                parameter.requires_grad_(False)
        if refiner is not None:
            for parameter in refiner.parameters():
                parameter.requires_grad_(False)
        print("[INFO] froze DeepGD/refiner parameters; training ParetoField only")

    separate_hctf_optimizer = bool(
        args.use_hctf
        and (
            args.hctf_version == "v2"
            or args.hctf_lr > 0.0
            or args.hctf_lr_multiplier != 1.0
            or args.hctf_backbone_warmup_epochs > 0
        )
    )
    optim_groups = []
    if separate_hctf_optimizer:
        backbone_parameters = [
            parameter
            for name, parameter in deepgd_model.named_parameters()
            if parameter.requires_grad
            and not name.startswith("paretofield.")
            and not is_hctf_parameter(name)
        ]
        hctf_parameters = [
            parameter
            for name, parameter in deepgd_model.named_parameters()
            if parameter.requires_grad and is_hctf_parameter(name)
            and not (args.hctf_graph_size_gate_lr > 0
                     and name.startswith("hctf_graph_size_gate."))
        ]
        size_gate_parameters = [
            parameter for name, parameter in deepgd_model.named_parameters()
            if parameter.requires_grad and name.startswith("hctf_graph_size_gate.")
        ]
        hctf_lr = (
            float(args.hctf_lr)
            if args.hctf_lr > 0.0
            else float(args.deepgd_lr) * float(args.hctf_lr_multiplier)
        )
        if backbone_parameters:
            optim_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": 0.0 if args.hctf_backbone_warmup_epochs > 0 else args.deepgd_lr,
                    "target_lr": args.deepgd_lr,
                    "group_name": "backbone",
                    "hctf_warmup_frozen": args.hctf_backbone_warmup_epochs > 0,
                }
            )
        if hctf_parameters:
            optim_groups.append(
                {
                    "params": hctf_parameters,
                    "lr": hctf_lr,
                    "target_lr": hctf_lr,
                    "group_name": "hctf",
                }
            )
        if size_gate_parameters and args.hctf_graph_size_gate_lr > 0:
            optim_groups.append({
                "params": size_gate_parameters,
                "lr": float(args.hctf_graph_size_gate_lr),
                "target_lr": float(args.hctf_graph_size_gate_lr),
                "group_name": "hctf_size_gate",
            })
    else:
        base_parameters = [
            parameter
            for name, parameter in deepgd_model.named_parameters()
            if not name.startswith("paretofield.") and parameter.requires_grad
        ]
        if base_parameters:
            optim_groups.append({"params": base_parameters, "lr": args.deepgd_lr})

    if args.use_paretofield:
        pareto_parameters = [
            parameter
            for parameter in deepgd_model.paretofield.parameters()
            if parameter.requires_grad
        ]
        if pareto_parameters:
            optim_groups.append(
                {
                    "params": pareto_parameters,
                    "lr": args.paretofield_lr,
                    "target_lr": args.paretofield_lr,
                    "group_name": "paretofield",
                }
            )

    if refiner is not None:
        refiner_parameters = [
            parameter for parameter in refiner.parameters() if parameter.requires_grad
        ]
        if refiner_parameters:
            optim_groups.append(
                {
                    "params": refiner_parameters,
                    "lr": 0.0 if args.hctf_backbone_warmup_epochs > 0 else args.refiner_lr,
                    "target_lr": args.refiner_lr,
                    "group_name": "refiner",
                    "hctf_warmup_frozen": args.hctf_backbone_warmup_epochs > 0,
                }
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
    start_epoch = 1
    non_improving_epochs = 0

    if args.resume:
        resume_path = args.ckpt if args.ckpt != "" else last_ckpt_path
        ckpt = load_checkpoint(resume_path, deepgd_model, refiner, device)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"[INFO] loaded optimizer state from: {resume_path}")
        else:
            print(f"[INFO] no optimizer state in checkpoint; resume with fresh optimizer")
        loaded_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", -1))
        if best_epoch < 0 and os.path.exists(best_ckpt_path):
            try:
                best_ckpt = torch_load_trusted(best_ckpt_path, map_location="cpu")
                best_epoch = int(best_ckpt.get("epoch", -1))
                best_val = float(best_ckpt.get("best_val", best_val))
                print(
                    f"[INFO] restored best checkpoint metadata: "
                    f"epoch={best_epoch}, best_val={best_val:.6f}"
                )
            except Exception as error:
                print(f"[WARN] could not read best checkpoint metadata: {error}")
        if best_epoch < 0:
            best_epoch = loaded_epoch
        start_epoch = loaded_epoch + 1
        print(f"[INFO] resume training from epoch {start_epoch}")

    print("\n========== Start Training ==========")
    print(f"[INFO] run name = {args.run_name}")
    print(f"[INFO] dataset = {args.dataset}")
    print(f"[INFO] use_cser = {args.use_cser}")
    print(f"[INFO] cser_mode = {effective_cser_mode(args)}")
    print(
        "[INFO] community partition: "
        f"source={args.community_source}, noise={args.community_noise}, "
        f"seed={args.community_seed}"
    )
    print(
        "[INFO] adaspec_cser: "
        f"enabled={args.use_adaspec_cser}, "
        f"alpha_s={args.adaspec_alpha_s}, "
        f"alpha_f={args.adaspec_alpha_f}, "
        f"self_loop_scale={args.adaspec_self_loop_scale}, "
        f"boundary_weight={args.adaspec_boundary_weight}, "
        f"core_weight={args.adaspec_core_weight}"
    )
    if args.use_adaspec_cser and not need_spectral_pe(args):
        print(
            "[WARN] --use-adaspec-cser only affects cser_mode=same_spectral/full "
            "with spectral context enabled."
        )
    print(
        "[INFO] topocser: "
        f"enabled={args.use_topocser}, "
        "features=topo_dist,topo_cos,boundary_pair,core_pair"
    )
    print(f"[INFO] use_care = {args.use_care}")
    if args.use_care:
        care_indices = get_care_edge_indices(args)
        print(
            "[INFO] care: "
            f"rank={args.care_rank}, "
            f"scale={args.care_scale}, "
            f"boundary_gate={not args.no_care_boundary_gate}, "
            f"boundary_damp={args.care_boundary_damp}, "
            f"checkpoint={not args.no_care_checkpoint}, "
            f"indices={care_indices}"
        )
    print(
        "[INFO] hctf: "
        f"enabled={args.use_hctf}, "
        f"version={args.hctf_version}, "
        f"trees={args.hctf_num_trees}, "
        f"relative_tree_messages={args.hctf_relative_tree_messages}, "
        f"node_relative_prototype={args.hctf_node_relative_prototype}, "
        f"hidden={args.hctf_hidden_dim}, "
        f"dropout={args.hctf_dropout}, "
        f"scale={args.hctf_scale}, "
        f"mix_init={args.hctf_mix_init}, "
        f"block_stride={args.hctf_block_stride}, "
        f"late_only={args.hctf_late_only}, "
        f"cache_size={args.hctf_cache_size}, "
        f"graph_gate={args.hctf_graph_gate}, "
        f"adaptive_structure_gate={args.hctf_adaptive_structure_gate}, "
        f"structure_gate_init={args.hctf_structure_gate_init}, "
        f"unsaturated_degree_mean={args.hctf_unsaturated_degree_mean}, "
        f"centered_prototype_context={args.hctf_centered_prototype_context}, "
        f"prototype_graph_mean_scale={args.hctf_prototype_graph_mean_scale}, "
        f"independent_prototype_gate={args.hctf_independent_prototype_gate}, "
        f"route_graph_mean_to_node={args.hctf_route_graph_mean_to_node}, "
        f"structure_gate_cost={args.lambda_hctf_structure_gate_cost}, "
        f"structure_gate_supervision={args.lambda_hctf_structure_gate_supervision}, "
        f"structure_gate_components={args.hctf_structure_gate_components}, "
        f"zero_init={args.hctf_zero_init}, "
        f"coordinate_residual={args.hctf_coordinate_residual}, "
        f"coordinate_max_step={args.hctf_coordinate_max_step}, "
        f"exact_risk_gate={args.hctf_exact_risk_gate}, "
        f"exact_gate_lambda={args.lambda_hctf_exact_gate}, "
        f"exact_gate_components={args.hctf_exact_gate_components}, "
        f"exact_gate_tolerance={args.hctf_exact_gate_tolerance}, "
        f"freeze_proposal={args.freeze_hctf_proposal_for_risk_gate}, "
        f"freeze_backbone={args.freeze_backbone_for_hctf}, "
        f"lr={args.hctf_lr or args.deepgd_lr * args.hctf_lr_multiplier}, "
        f"backbone_warmup={args.hctf_backbone_warmup_epochs}, "
        f"residual_warmup={args.hctf_residual_warmup_epochs}, "
        f"community_augmentation={args.hctf_community_augmentation}, "
        f"preference={args.hctf_preference}, "
        f"dominance={args.lambda_hctf_dominance}, "
        f"anchor={args.lambda_hctf_anchor}, "
        f"crossing_barrier={args.lambda_hctf_crossing_barrier}, "
        f"cvar=({args.hctf_cvar_fraction}, {args.hctf_cvar_weight}), "
        f"dominance_only={args.hctf_dominance_only}"
    )
    print(f"[INFO] use_caid = {args.use_caid}")
    if args.use_caid:
        print(
            "[INFO] caid: "
            f"lambda={args.lambda_caid}, "
            f"intra_shrink={args.caid_intra_shrink}, "
            f"adj_expand={args.caid_adj_expand}, "
            f"far_expand={args.caid_far_expand}, "
            f"ratio=[{args.caid_min_ratio}, {args.caid_max_ratio}], "
            f"warmup={args.caid_warmup_epochs}, "
            f"edge_features={not args.no_caid_edge_features}, "
            f"graph_local_communities={args.graph_local_community_objectives}"
        )
    print(f"[INFO] use_cpsg = {args.use_cpsg}")
    if args.use_cpsg:
        print(
            "[INFO] cpsg: "
            f"lambda_proto={args.lambda_cpsg_proto}, "
            f"lambda_sil={args.lambda_cpsg_sil}, "
            f"lambda_sep={args.lambda_cpsg_sep}, "
            f"sil_margin={args.cpsg_sil_margin}, "
            f"adj_margin={args.cpsg_adj_margin}, "
            f"far_margin={args.cpsg_far_margin}, "
            f"core_gamma={args.cpsg_core_gamma}, "
            f"boundary_floor={args.cpsg_boundary_floor}, "
            f"warmup={args.cpsg_warmup_epochs}, "
            f"graph_local_communities={args.graph_local_community_objectives}"
        )
        print(
            "[INFO] metric-aligned cpsg: "
            f"enabled={args.use_macpsg}, "
            f"sep_scale_floor={args.macpsg_sep_scale_floor}"
        )
    print(
        "[INFO] pmds_anchor: "
        f"lambda={args.lambda_pmds_anchor}, "
        f"start={args.pmds_anchor_start_epoch}, "
        f"warmup={args.pmds_anchor_warmup_epochs}"
    )
    print(
        "[INFO] rd_stress: "
        f"enabled={args.use_rd_stress}, "
        f"lambda={args.lambda_rd_stress}, "
        f"max_pairs={args.rd_max_pairs}, "
        f"normalize={args.rd_normalize}"
    )
    print(
        "[INFO] cosine_stress: "
        f"enabled={args.use_cosine_stress}, "
        f"lambda={args.lambda_cosine_stress}, "
        f"max_pairs={args.cosine_max_pairs}, "
        f"walk_steps={args.cosine_walk_steps}, "
        f"walk_decay={args.cosine_walk_decay}, "
        f"normalize={args.cosine_normalize}"
    )
    print(
        "[INFO] sampled_crossing: "
        f"lambda={args.lambda_crossing}, "
        f"max_pairs={args.crossing_max_pairs}, "
        f"oversample={args.crossing_oversample_factor}, "
        f"temperature={args.crossing_temperature}, "
        f"margin={args.crossing_margin}, "
        f"start={args.crossing_start_epoch}, "
        f"warmup={args.crossing_warmup_epochs}"
    )
    print(
        "[INFO] hctf_readability: "
        f"lambda={args.lambda_hctf_readability}, "
        f"weights=({args.hctf_readability_w_angular},"
        f"{args.hctf_readability_w_crossing_angle},"
        f"{args.hctf_readability_w_clearance},"
        f"{args.hctf_readability_w_neighborhood}), "
        f"tail={args.hctf_readability_crossing_angle_tail_fraction}, "
        f"dominance={args.hctf_readability_dominance}, "
        f"target={args.hctf_readability_target_improvement}, "
        f"max_pairs={args.hctf_readability_max_pairs}, "
        f"start={args.hctf_readability_start_epoch}, "
        f"warmup={args.hctf_readability_warmup_epochs}"
    )
    print(
        "[INFO] train_dcsg: "
        f"lambda={args.lambda_train_silhouette}, "
        f"margin={args.train_silhouette_margin}, "
        f"start={args.train_silhouette_start_epoch}, "
        f"warmup={args.train_silhouette_warmup_epochs}"
    )
    print(
        "[INFO] mtpf_gd: "
        f"enabled={args.use_mtpf}, "
        f"profile={args.mtpf_profile}, "
        f"teachers={args.mtpf_teachers}, "
        f"refine_steps={args.mtpf_refine_steps}"
    )
    print(f"[INFO] use_refiner = {args.use_refiner}")
    print(
        "[INFO] paretofield: "
        f"enabled={args.use_paretofield}, "
        f"hidden={args.paretofield_hidden}, "
        f"steps={args.paretofield_train_steps}/{args.paretofield_inference_steps}, "
        f"preference={args.paretofield_preference}, "
        f"sample_preferences={args.paretofield_sample_preferences}, "
        f"soft_community={args.paretofield_soft_community}, "
        f"relation_consensus={args.paretofield_relation_consensus}, "
        f"graph_preference={args.paretofield_graph_preference_calibration}, "
        f"edge_pair_field={args.paretofield_edge_pair_field}, "
        f"edge_pair_pairs={args.paretofield_edge_pair_max_pairs}, "
        f"regret_weight={args.paretofield_regret_weight}, "
        f"conflict={args.paretofield_conflict_method}, "
        f"adaptive_tol={args.paretofield_adaptive_tolerance}, "
        f"safe_guard={args.paretofield_safe_guard}, "
        f"continuous_guard={args.paretofield_continuous_guard}, "
        f"guard_steps={args.paretofield_guard_step_fractions}, "
        f"guard_mode={args.paretofield_guard_mode}, "
        f"guard_crossing={args.paretofield_guard_crossing}, "
        f"exact_crossing_guard={args.paretofield_exact_crossing_guard}, "
        f"componentwise_guard={args.paretofield_componentwise_guard}, "
        f"freeze_deepgd={args.freeze_deepgd_for_paretofield}, "
        f"teacher_archive={args.paretofield_teacher_archive or 'none'}"
    )
    print(f"[INFO] no_spectral_context = {args.no_spectral_context}")
    print(f"[INFO] select_by = {args.select_by}")
    print(f"[INFO] min_select_epoch = {args.min_select_epoch}")
    print(f"[INFO] memory_cleanup_every = {args.memory_cleanup_every}")
    print(f"[INFO] save_optimizer = {not args.no_save_optimizer}")
    print(f"[INFO] best checkpoint = {best_ckpt_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        hctf_residual_progress = configure_hctf_training_stage(
            optimizer, deepgd_model, epoch, args
        )
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

        if args.train_only_pilot:
            history.append({
                "epoch": epoch,
                **train_info,
                "hctf_residual_progress": hctf_residual_progress,
                "split": "train_only_pilot",
            })
            save_checkpoint(
                path=last_ckpt_path, epoch=epoch, deepgd_model=deepgd_model,
                refiner=refiner, optimizer=optimizer,
                best_val="not_evaluated_train_only_pilot", args=args,
            )
            history_path = os.path.join(args.history_dir, f"{args.run_name}_history.csv")
            pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
            print(f"[TRAIN-ONLY] epoch={epoch} train_loss={train_info['train_loss']:.6f} "
                  f"last_checkpoint={last_ckpt_path}")
            maybe_cleanup_memory(args, force=True)
            continue

        val_info = run_validation(
            epoch=epoch,
            loader=val_loader,
            deepgd_model=deepgd_model,
            refiner=refiner,
            stress_metric=stress_metric,
            device=device,
            args=args,
        )

        val_score = select_validation_score(val_info, args)
        scheduler.step(val_score)

        row = {
            "epoch": epoch,
            **train_info,
            **val_info,
            "val_score": val_score,
            "select_by": args.select_by,
            "hctf_residual_progress": hctf_residual_progress,
        }
        for group in optimizer.param_groups:
            group_name = group.get("group_name")
            if group_name:
                row[f"lr_{group_name}"] = float(group["lr"])
        if getattr(deepgd_model, "hctf_block", None) is not None:
            row.update(
                {
                    "hctf_mean_gate": deepgd_model.hctf_block.last_mean_gate,
                    "hctf_mean_graph_gate": (
                        deepgd_model.hctf_block.last_mean_graph_gate
                    ),
                    "hctf_mix": deepgd_model.hctf_block.last_mix,
                    "hctf_structure_gate": getattr(
                        deepgd_model.hctf_block,
                        "last_mean_structure_gate",
                        0.0,
                    ),
                    "hctf_structure_gate_min": getattr(
                        deepgd_model.hctf_block,
                        "last_min_structure_gate",
                        0.0,
                    ),
                    "hctf_structure_gate_max": getattr(
                        deepgd_model.hctf_block,
                        "last_max_structure_gate",
                        0.0,
                    ),
                    "hctf_prototype_gate": getattr(
                        deepgd_model.hctf_block, "last_mean_prototype_gate", 1.0
                    ),
                    "hctf_prototype_gate_min": getattr(
                        deepgd_model.hctf_block, "last_min_prototype_gate", 1.0
                    ),
                    "hctf_prototype_gate_max": getattr(
                        deepgd_model.hctf_block, "last_max_prototype_gate", 1.0
                    ),
                    "hctf_coordinate_trust": deepgd_model.last_hctf_coordinate_trust,
                    "hctf_coordinate_limiter": deepgd_model.last_hctf_coordinate_limiter,
                    "hctf_exact_risk": deepgd_model.last_hctf_exact_risk,
                    "hctf_exact_risk_min": deepgd_model.last_hctf_exact_risk_min,
                    "hctf_exact_risk_max": deepgd_model.last_hctf_exact_risk_max,
                    "hctf_exact_risk_std": deepgd_model.last_hctf_exact_risk_std,
                    "hctf_cache_hits": deepgd_model.hctf_cover.cache_hits,
                    "hctf_cache_misses": deepgd_model.hctf_cover.cache_misses,
                    "hctf_tree_attention_entropy": getattr(
                        deepgd_model.hctf_block,
                        "last_tree_attention_entropy",
                        0.0,
                    ),
                    "hctf_partition_confidence": getattr(
                        deepgd_model.hctf_block,
                        "last_partition_confidence",
                        0.0,
                    ),
                }
            )
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_info['train_loss']:.6f} "
            f"train_caid={train_info['train_caid_loss']:.6f} "
            f"train_cpsg={train_info['train_cpsg_loss']:.6f} "
            f"train_pmds_anchor={train_info['train_pmds_anchor']:.6f} "
            f"train_crossing={train_info['train_crossing_loss']:.6f} "
            f"train_pf_regret={train_info['train_paretofield_regret']:.6f} "
            f"train_rd={train_info['train_refined_rd_stress']:.6f} "
            f"train_cos={train_info['train_refined_cosine_stress']:.6f} "
            f"train_base_stress={train_info['train_base_stress']:.6f} "
            f"train_refined_stress={train_info['train_refined_stress']:.6f} "
            f"val_loss={val_info['val_loss']:.6f} "
            f"val_caid={val_info['val_caid_loss']:.6f} "
            f"val_cpsg={val_info['val_cpsg_loss']:.6f} "
            f"val_pmds_anchor={val_info['val_pmds_anchor']:.6f} "
            f"val_crossing={val_info['val_crossing_loss']:.6f} "
            f"val_pf_regret={val_info['val_paretofield_regret']:.6f} "
            f"val_rd={val_info['val_refined_rd_stress']:.6f} "
            f"val_cos={val_info['val_refined_cosine_stress']:.6f} "
            f"val_base_stress={val_info['val_base_stress']:.6f} "
            f"val_refined_stress={val_info['val_refined_stress']:.6f} "
            f"val_base_intra={val_info['val_base_intra']:.6f} "
            f"val_refined_intra={val_info['val_refined_intra']:.6f} "
            f"best_val={best_val:.6f}"
        )
        if getattr(deepgd_model, "hctf_block", None) is not None:
            print(
                "[INFO] hctf_state: "
                f"mean_gate={deepgd_model.hctf_block.last_mean_gate:.6f}, "
                f"graph_gate={deepgd_model.hctf_block.last_mean_graph_gate:.6f}, "
                f"mix={deepgd_model.hctf_block.last_mix:.6f}, "
                f"structure_gate={getattr(deepgd_model.hctf_block, 'last_mean_structure_gate', 0.0):.6f}, "
                f"prototype_gate={getattr(deepgd_model.hctf_block, 'last_mean_prototype_gate', 1.0):.6f}, "
                f"coordinate_trust={deepgd_model.last_hctf_coordinate_trust:.6f}, "
                f"coordinate_limiter={deepgd_model.last_hctf_coordinate_limiter:.6f}, "
                f"exact_risk={deepgd_model.last_hctf_exact_risk:.6f}, "
                f"risk_range=[{deepgd_model.last_hctf_exact_risk_min:.6f}, "
                f"{deepgd_model.last_hctf_exact_risk_max:.6f}], "
                f"risk_std={deepgd_model.last_hctf_exact_risk_std:.6f}, "
                f"cache_hits={deepgd_model.hctf_cover.cache_hits}, "
                f"cache_misses={deepgd_model.hctf_cover.cache_misses}"
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
        if args.save_every_epoch:
            save_checkpoint(
                path=os.path.join(args.save_dir, f"{args.run_name}_epoch{epoch}.pt"),
                epoch=epoch,
                deepgd_model=deepgd_model,
                refiner=refiner,
                optimizer=optimizer,
                best_val=best_val,
                args=args,
            )
        maybe_cleanup_memory(args, force=True)

        if epoch < args.min_select_epoch:
            print(
                f"[INFO] skip best selection before epoch {args.min_select_epoch} "
                f"(current epoch = {epoch})"
            )
        elif val_score < best_val - float(args.early_stopping_min_delta):
            best_val = val_score
            best_epoch = epoch
            non_improving_epochs = 0

            save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                deepgd_model=deepgd_model,
                refiner=refiner,
                optimizer=optimizer,
                best_val=best_val,
                args=args,
            )
            maybe_cleanup_memory(args, force=True)

            print(f"[INFO] new best epoch = {best_epoch}, best val = {best_val:.6f}")
        elif epoch >= args.min_select_epoch:
            non_improving_epochs += 1

        history_path = os.path.join(args.history_dir, f"{args.run_name}_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        maybe_cleanup_memory(args, force=True)

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_select_epoch
            and non_improving_epochs >= args.early_stopping_patience
        ):
            print(
                "[INFO] early stopping: "
                f"no validation improvement for {non_improving_epochs} epochs"
            )
            break

    if args.train_only_pilot:
        print("[TRAIN-ONLY] finished without validation or test evaluation")
        print(f"[TRAIN-ONLY] last checkpoint = {last_ckpt_path}")
        return

    print("\n========== Training Finished ==========")
    print(f"[INFO] best epoch = {best_epoch}")
    print(f"[INFO] best val = {best_val:.6f}")
    print(f"[INFO] best checkpoint = {best_ckpt_path}")

    if args.skip_final_test:
        print("[INFO] final test evaluation skipped by --skip-final-test")
        return

    print("\n========== Evaluate Best on Test ==========")
    load_checkpoint(best_ckpt_path, deepgd_model, refiner, device)
    args._hctf_teacher_enabled = False

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
