from __future__ import annotations

import torch


def _graph_count(batch_index) -> int:
    return int(batch_index.max().item()) + 1 if batch_index.numel() else 0


def graphwise_stress(pos, pair_index, target_distance, batch_index, eps: float = 1e-8):
    src, dst = pair_index.long()
    values = (torch.norm(pos[src] - pos[dst], dim=1) - target_distance.view(-1)).square()
    values = values / target_distance.view(-1).square().clamp_min(eps)
    pair_graph = batch_index.long()[src]
    result = []
    for graph_id in range(_graph_count(batch_index)):
        selected = values[pair_graph == graph_id]
        result.append(selected.mean() if selected.numel() else pos.sum() * 0.0)
    return torch.stack(result)


def graphwise_edge_cv(pos, edge_index, batch_index):
    src, dst = edge_index.long()
    valid = (src != dst) & (src < dst)
    src, dst = src[valid], dst[valid]
    edge_graph = batch_index.long()[src]
    lengths = torch.norm(pos[src] - pos[dst], dim=1)
    result = []
    for graph_id in range(_graph_count(batch_index)):
        selected = lengths[edge_graph == graph_id]
        if selected.numel() <= 1:
            result.append(pos.sum() * 0.0)
        else:
            result.append(selected.std(unbiased=False) / selected.mean().clamp_min(1e-8))
    return torch.stack(result)


def graphwise_occlusion(pos, batch_index, target_ratio: float = 0.03, max_nodes: int = 160):
    result = []
    for graph_id in range(_graph_count(batch_index)):
        points = pos[batch_index == graph_id]
        if points.size(0) > int(max_nodes):
            sample = torch.randperm(points.size(0), device=pos.device)[: int(max_nodes)]
            points = points[sample]
        distances = torch.pdist(points)
        if distances.numel() == 0:
            result.append(pos.sum() * 0.0)
            continue
        threshold = float(target_ratio) * distances.detach().median().clamp_min(1e-8)
        result.append(torch.relu(threshold - distances).square().mean())
    return torch.stack(result)


def graphwise_community_loss(pos, community_labels, batch_index, sep_margin: float = 1.25):
    compact_result = []
    separation_result = []
    silhouette_result = []
    for graph_id in range(_graph_count(batch_index)):
        mask = batch_index == graph_id
        points = pos[mask]
        labels = community_labels[mask]
        unique = labels.unique(sorted=True)
        if unique.numel() <= 1:
            zero = pos.sum() * 0.0
            compact_result.append(zero)
            separation_result.append(zero)
            silhouette_result.append(zero)
            continue
        centers = torch.stack([points[labels == label].mean(dim=0) for label in unique])
        local_ids = torch.searchsorted(unique, labels)
        own_centers = centers[local_ids]
        intra = torch.norm(points - own_centers, dim=1)
        scale = torch.pdist(points).detach().median().clamp_min(1e-8)
        compact_result.append((intra / scale).mean())

        center_distances = torch.pdist(centers)
        separation_result.append(torch.relu(float(sep_margin) - center_distances / scale).square().mean())

        all_center_distances = torch.cdist(points, centers) / scale
        rows = torch.arange(points.size(0), device=pos.device)
        other = all_center_distances.clone()
        other[rows, local_ids] = torch.inf
        nearest_other = other.min(dim=1).values
        silhouette = (nearest_other - intra / scale) / torch.maximum(
            nearest_other, intra / scale
        ).clamp_min(1e-8)
        silhouette_result.append(-silhouette.mean())
    return (
        torch.stack(compact_result),
        torch.stack(separation_result),
        torch.stack(silhouette_result),
    )


def graphwise_community_quality(pos, community_labels, batch_index):
    """Return scale-free compactness, separation, ratio, and silhouette."""
    rows = []
    for graph_id in range(_graph_count(batch_index)):
        mask = batch_index == graph_id
        points = pos[mask]
        labels = community_labels[mask]
        unique = labels.unique(sorted=True)
        if points.size(0) <= 1 or unique.numel() <= 1:
            rows.append(pos.new_zeros(4) + pos.sum() * 0.0)
            continue

        centers = torch.stack([points[labels == label].mean(dim=0) for label in unique])
        local_ids = torch.searchsorted(unique, labels)
        intra = torch.norm(points - centers[local_ids], dim=1)
        scale = torch.pdist(points).detach().median().clamp_min(1e-8)
        compactness = (intra / scale).mean()
        separation = (torch.pdist(centers) / scale).mean()
        ratio = separation / compactness.clamp_min(1e-8)

        center_distances = torch.cdist(points, centers) / scale
        node_rows = torch.arange(points.size(0), device=pos.device)
        other = center_distances.clone()
        other[node_rows, local_ids] = torch.inf
        nearest_other = other.min(dim=1).values
        normalized_intra = intra / scale
        silhouette = (
            nearest_other - normalized_intra
        ) / torch.maximum(nearest_other, normalized_intra).clamp_min(1e-8)
        rows.append(torch.stack([compactness, separation, ratio, silhouette.mean()]))
    return torch.stack(rows)


def graphwise_pareto_objectives(
    pos,
    pair_index,
    target_distance,
    edge_index,
    batch_index,
    community_labels,
    crossing_per_graph,
):
    geometry = graphwise_stress(pos, pair_index, target_distance, batch_index)
    edge_cv = graphwise_edge_cv(pos, edge_index, batch_index)
    occlusion = graphwise_occlusion(pos, batch_index)
    compact, separation, silhouette = graphwise_community_loss(
        pos, community_labels, batch_index
    )
    visual = 0.50 * crossing_per_graph + 0.25 * edge_cv + 0.25 * occlusion
    community = 0.40 * compact + 0.30 * separation + 0.30 * silhouette
    return torch.stack([geometry, visual, community], dim=1)


def graphwise_regret_loss(
    seed_objectives,
    candidate_objectives,
    preference,
    limits=None,
    max_weighted_regret: float = 0.0,
    scale_floor: float = 0.05,
):
    """Penalize graphwise objective regressions beyond explicit regret limits.

    All objectives are lower-is-better. Improvements remain unpenalized, while
    per-objective and preference-weighted regressions receive squared hinge loss.
    """
    if seed_objectives.shape != candidate_objectives.shape:
        raise ValueError("seed and candidate objectives must have identical shapes")
    if seed_objectives.ndim != 2:
        raise ValueError("graphwise objectives must have shape [graphs, objectives]")
    if preference.shape != seed_objectives.shape:
        raise ValueError("preference must match graphwise objective shape")

    scale = seed_objectives.detach().abs().clamp_min(float(scale_floor))
    regret = (candidate_objectives - seed_objectives.detach()) / scale
    preference = preference.clamp_min(0.0)
    preference = preference / preference.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if limits is None:
        limits = regret.new_zeros(regret.size(1))
    else:
        limits = torch.as_tensor(limits, dtype=regret.dtype, device=regret.device)
    if limits.numel() != regret.size(1):
        raise ValueError("regret limits must contain one value per objective")

    objective_excess = torch.relu(regret - limits.view(1, -1))
    objective_penalty = (preference * objective_excess.square()).sum(dim=1)
    weighted_regret = (preference * regret).sum(dim=1)
    weighted_excess = torch.relu(weighted_regret - float(max_weighted_regret))
    loss = (objective_penalty + weighted_excess.square()).mean()
    return loss, {
        "regret": regret,
        "weighted_regret": weighted_regret,
        "objective_excess": objective_excess,
    }


def _stress_rescale(pos, pair_index, target_distance, batch_index):
    src, dst = pair_index.long()
    pair_graph = batch_index.long()[src]
    distance = torch.norm(pos[src] - pos[dst], dim=1)
    ratio = distance / target_distance.view(-1).clamp_min(1e-9)
    graph_count = _graph_count(batch_index)
    ratio_sum = torch.zeros(graph_count, dtype=pos.dtype, device=pos.device)
    ratio_sq_sum = torch.zeros_like(ratio_sum)
    ratio_sum.index_add_(0, pair_graph, ratio)
    ratio_sq_sum.index_add_(0, pair_graph, ratio.square())
    scale = ratio_sq_sum / ratio_sum.clamp_min(1e-9)
    return pos / scale[batch_index.long(), None].clamp_min(1e-9)


def _guard_objectives(
    pos,
    crossing,
    pair_index,
    target_distance,
    edge_index,
    batch_index,
    community_labels,
):
    scaled_pos = _stress_rescale(pos, pair_index, target_distance, batch_index)
    geometry = graphwise_stress(
        scaled_pos, pair_index, target_distance, batch_index
    )
    edge_cv = graphwise_edge_cv(scaled_pos, edge_index, batch_index)
    occlusion = graphwise_occlusion(scaled_pos, batch_index)
    compact, separation, silhouette = graphwise_community_loss(
        scaled_pos, community_labels, batch_index
    )
    if crossing is None:
        visual = 0.50 * edge_cv + 0.50 * occlusion
    else:
        visual = 0.50 * crossing + 0.25 * edge_cv + 0.25 * occlusion
    community = 0.40 * compact + 0.30 * separation + 0.30 * silhouette
    return torch.stack([geometry, visual, community], dim=1)


def _guard_components(
    pos,
    crossing,
    pair_index,
    target_distance,
    edge_index,
    batch_index,
    community_labels,
):
    """Return lower-is-better objectives for every protected layout metric."""
    scaled_pos = _stress_rescale(pos, pair_index, target_distance, batch_index)
    geometry = graphwise_stress(
        scaled_pos, pair_index, target_distance, batch_index
    )
    edge_cv = graphwise_edge_cv(scaled_pos, edge_index, batch_index)
    occlusion = graphwise_occlusion(scaled_pos, batch_index)
    quality = graphwise_community_quality(
        scaled_pos, community_labels, batch_index
    )
    if crossing is None:
        crossing = torch.zeros_like(geometry)
    return torch.stack(
        [
            geometry,
            crossing,
            edge_cv,
            occlusion,
            quality[:, 0],
            -quality[:, 1],
            -quality[:, 2],
            -quality[:, 3],
        ],
        dim=1,
    )


@torch.no_grad()
def graphwise_pareto_guard(
    seed_pos,
    candidate_pos,
    pair_index,
    target_distance,
    edge_index,
    batch_index,
    community_labels,
    preference,
    seed_crossing=None,
    candidate_crossing=None,
    max_geometry_regret: float = 0.0,
    max_visual_regret: float = 0.01,
    max_community_regret: float = 0.01,
    max_weighted_regret: float = 0.0,
    mode: str = "pareto",
    step_fractions=None,
    crossing_evaluator=None,
    componentwise: bool = False,
):
    """Accept candidate coordinates only for graphs satisfying Pareto regret limits.

    Regret is normalized by the magnitude of each seed objective. All three objectives
    are lower-is-better. When crossing estimates are omitted, the visual objective uses
    edge-length CV and occlusion only, which keeps the guard inexpensive. If a
    fixed-pair ``crossing_evaluator`` is supplied, crossing is recomputed for
    every line-search step instead of being linearly interpolated.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"stress", "pareto"}:
        raise ValueError("Pareto guard mode must be 'stress' or 'pareto'")

    def evaluate(pos, crossing):
        if normalized_mode == "stress":
            scaled = _stress_rescale(
                pos, pair_index, target_distance, batch_index
            )
            return graphwise_stress(
                scaled, pair_index, target_distance, batch_index
            )[:, None]
        if crossing_evaluator is not None:
            crossing = crossing_evaluator(pos)
        if componentwise:
            return _guard_components(
                pos=pos,
                crossing=crossing,
                pair_index=pair_index,
                target_distance=target_distance,
                edge_index=edge_index,
                batch_index=batch_index,
                community_labels=community_labels,
            )
        return _guard_objectives(
            pos=pos,
            crossing=crossing,
            pair_index=pair_index,
            target_distance=target_distance,
            edge_index=edge_index,
            batch_index=batch_index,
            community_labels=community_labels,
        )

    seed_objectives = evaluate(seed_pos, seed_crossing)
    candidate_objectives = evaluate(candidate_pos, candidate_crossing)
    scale = seed_objectives.abs().clamp_min(0.05)
    full_regret = (candidate_objectives - seed_objectives) / scale

    if normalized_mode == "stress":
        normalized_preference = torch.ones_like(seed_objectives)
        limits = full_regret.new_tensor([max_geometry_regret])
    elif componentwise:
        preference = preference.clamp_min(0.0)
        preference = preference / preference.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        normalized_preference = torch.cat(
            [
                preference[:, 0:1],
                preference[:, 1:2].expand(-1, 3) / 3.0,
                preference[:, 2:3].expand(-1, 4) / 4.0,
            ],
            dim=1,
        )
        limits = full_regret.new_tensor(
            [
                max_geometry_regret,
                max_visual_regret,
                max_visual_regret,
                max_visual_regret,
                max_community_regret,
                max_community_regret,
                max_community_regret,
                max_community_regret,
            ]
        )
    else:
        normalized_preference = preference.clamp_min(0.0)
        normalized_preference = normalized_preference / normalized_preference.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        limits = full_regret.new_tensor(
            [max_geometry_regret, max_visual_regret, max_community_regret]
        )

    if step_fractions is not None:
        fractions = sorted({float(value) for value in step_fractions} | {0.0})
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("guard step fractions must lie in [0, 1]")
        graph_count = int(seed_objectives.size(0))
        selected_fraction = seed_pos.new_zeros(graph_count)
        selected_objectives = seed_objectives.clone()
        selected_regret = torch.zeros_like(seed_objectives)
        selected_weighted_regret = seed_pos.new_zeros(graph_count)
        best_score = seed_pos.new_zeros(graph_count)
        direction = candidate_pos - seed_pos

        for fraction in fractions:
            if fraction == 0.0:
                continue
            trial_pos = seed_pos + fraction * direction
            if crossing_evaluator is not None:
                trial_crossing = None
            elif seed_crossing is not None and candidate_crossing is not None:
                trial_crossing = seed_crossing + fraction * (
                    candidate_crossing - seed_crossing
                )
            else:
                trial_crossing = None
            trial_objectives = evaluate(trial_pos, trial_crossing)
            trial_regret = (trial_objectives - seed_objectives) / scale
            trial_weighted = (normalized_preference * trial_regret).sum(dim=1)
            feasible = (
                torch.isfinite(trial_objectives).all(dim=1)
                & (trial_regret <= limits[None, :]).all(dim=1)
                & (trial_weighted <= float(max_weighted_regret))
            )
            score = trial_weighted
            better = feasible & (
                (score < best_score - 1e-12)
                | (
                    torch.isclose(score, best_score, atol=1e-12, rtol=0.0)
                    & (fraction > selected_fraction)
                )
            )
            selected_fraction = torch.where(
                better, selected_fraction.new_full((), fraction), selected_fraction
            )
            selected_objectives = torch.where(
                better[:, None], trial_objectives, selected_objectives
            )
            selected_regret = torch.where(
                better[:, None], trial_regret, selected_regret
            )
            selected_weighted_regret = torch.where(
                better, trial_weighted, selected_weighted_regret
            )
            best_score = torch.where(better, score, best_score)

        guarded_pos = seed_pos + selected_fraction[batch_index.long(), None] * direction
        accepted = selected_fraction > 0.0
        return guarded_pos, {
            "accepted": accepted,
            "step_fraction": selected_fraction,
            "regret": selected_regret,
            "weighted_regret": selected_weighted_regret,
            "seed_objectives": seed_objectives,
            "candidate_objectives": candidate_objectives,
            "selected_objectives": selected_objectives,
            "full_candidate_regret": full_regret,
            "componentwise": bool(componentwise),
        }

    if normalized_mode == "stress":
        regret = full_regret
        accepted = torch.isfinite(candidate_objectives[:, 0]) & (
            regret[:, 0] <= float(max_geometry_regret)
        )
        weighted_regret = regret[:, 0]
        guarded_pos = torch.where(
            accepted[batch_index.long(), None], candidate_pos, seed_pos
        )
        return guarded_pos, {
            "accepted": accepted,
            "regret": regret,
            "weighted_regret": weighted_regret,
            "seed_objectives": seed_objectives,
            "candidate_objectives": candidate_objectives,
        }

    regret = full_regret
    weighted_regret = (normalized_preference * regret).sum(dim=1)
    finite = torch.isfinite(candidate_objectives).all(dim=1)
    accepted = (
        finite
        & (regret <= limits[None, :]).all(dim=1)
        & (weighted_regret <= float(max_weighted_regret))
    )
    guarded_pos = torch.where(accepted[batch_index.long(), None], candidate_pos, seed_pos)
    return guarded_pos, {
        "accepted": accepted,
        "regret": regret,
        "weighted_regret": weighted_regret,
        "seed_objectives": seed_objectives,
        "candidate_objectives": candidate_objectives,
        "componentwise": bool(componentwise),
    }
