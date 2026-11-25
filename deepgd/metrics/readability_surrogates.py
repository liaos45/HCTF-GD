from __future__ import annotations

import math

import torch

from deepgd.paretofield.crossing import select_metric_aligned_edge_pairs


def _zero(pos: torch.Tensor) -> torch.Tensor:
    return pos.sum() * 0.0


def _acos_margin(value: torch.Tensor, eps: float) -> float:
    """Return a clamp margin that remains representable in the tensor dtype."""
    return max(float(eps), float(torch.finfo(value.dtype).eps) * 16.0)


def _unique_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.long()
    if edge_index.numel() == 0:
        return edge_index.new_zeros((0, 2))
    src, dst = edge_index
    valid = src != dst
    lo = torch.minimum(src[valid], dst[valid])
    hi = torch.maximum(src[valid], dst[valid])
    if lo.numel() == 0:
        return edge_index.new_zeros((0, 2))
    key = lo * int(num_nodes) + hi
    order = torch.argsort(key)
    key = key[order]
    edges = torch.stack([lo[order], hi[order]], dim=1)
    keep = torch.ones(key.numel(), dtype=torch.bool, device=edge_index.device)
    keep[1:] = key[1:] != key[:-1]
    return edges[keep]


def angular_resolution_surrogate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    target_ratio: float = 0.5,
    max_target_degrees: float = 30.0,
    max_pairs: int = 4096,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize incident edge pairs below a degree-adaptive angular target."""
    edges = _unique_undirected_edges(edge_index.to(pos.device), pos.size(0))
    if edges.numel() == 0:
        return _zero(pos)
    centers = torch.cat([edges[:, 0], edges[:, 1]])
    neighbours = torch.cat([edges[:, 1], edges[:, 0]])
    degree = torch.bincount(centers, minlength=pos.size(0))
    pair_centers = []
    first_neighbours = []
    second_neighbours = []
    for node in torch.nonzero(degree >= 2, as_tuple=False).flatten().tolist():
        local = neighbours[centers == node]
        pair_ids = torch.triu_indices(
            local.numel(), local.numel(), offset=1, device=pos.device
        )
        if pair_ids.numel() == 0:
            continue
        pair_centers.append(
            torch.full(
                (pair_ids.size(1),),
                int(node),
                dtype=torch.long,
                device=pos.device,
            )
        )
        first_neighbours.append(local[pair_ids[0]])
        second_neighbours.append(local[pair_ids[1]])
    if not pair_centers:
        return _zero(pos)
    center = torch.cat(pair_centers)
    first = torch.cat(first_neighbours)
    second = torch.cat(second_neighbours)
    first_vector = pos[first] - pos[center]
    second_vector = pos[second] - pos[center]
    cosine = torch.nn.functional.cosine_similarity(
        first_vector,
        second_vector,
        dim=1,
        eps=float(eps),
    )
    angle_margin = _acos_margin(cosine, eps)
    pair_angles = torch.acos(cosine.clamp(-1.0 + angle_margin, 1.0 - angle_margin))
    unique_centers = torch.unique(center, sorted=True)
    minimum_angles = torch.stack(
        [pair_angles[center == node].min() for node in unique_centers]
    )
    ideal_radians = (
        2.0
        * math.pi
        / degree[unique_centers].to(dtype=pos.dtype).clamp_min(2.0)
    )
    normalized = (minimum_angles / ideal_radians.clamp_min(float(eps))).clamp(max=1.0)
    target_radians = torch.minimum(
        ideal_radians * float(target_ratio),
        ideal_radians.new_full(ideal_radians.shape, float(max_target_degrees) * math.pi / 180.0),
    )
    # The first term is a direct differentiable counterpart of the reported
    # degree-normalized angular-resolution ratio.  The hinge keeps extra focus
    # on nodes below the requested target instead of becoming flat above it.
    losses = (1.0 - normalized) + torch.relu(
        (target_radians - minimum_angles) / target_radians.clamp_min(float(eps))
    ).square()
    if int(max_pairs) > 0 and losses.numel() > int(max_pairs):
        with torch.no_grad():
            selected = torch.topk(losses.detach(), int(max_pairs)).indices
        losses = losses[selected]
    return losses.mean() if losses.numel() else _zero(pos)


def crossing_angle_surrogate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
    max_pairs: int = 4096,
    exact_max_edges: int = 128,
    tail_fraction: float = 0.20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage currently crossing edge pairs to approach right angles."""
    with torch.no_grad():
        first, second = select_metric_aligned_edge_pairs(
            pos=pos.detach(),
            edge_index=edge_index.long(),
            batch_index=batch_index.long(),
            max_pairs=int(max_pairs),
            exact_max_edges=int(exact_max_edges),
        )
        if first.numel() == 0:
            return _zero(pos)
        detached = pos.detach()
        a, b = detached[first[:, 0]], detached[first[:, 1]]
        c, d = detached[second[:, 0]], detached[second[:, 1]]

        def cross2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]

        ab = b - a
        cd = d - c
        crossing = (
            cross2(ab, c - a) * cross2(ab, d - a) < 0.0
        ) & (
            cross2(cd, a - c) * cross2(cd, b - c) < 0.0
        )
        first = first[crossing]
        second = second[crossing]
    if first.numel() == 0:
        return _zero(pos)
    first_vector = pos[first[:, 1]] - pos[first[:, 0]]
    second_vector = pos[second[:, 1]] - pos[second[:, 0]]
    cosine = torch.nn.functional.cosine_similarity(
        first_vector,
        second_vector,
        dim=1,
        eps=float(eps),
    ).abs()
    angle_margin = _acos_margin(cosine, eps)
    acute_angle = torch.acos(cosine.clamp(0.0, 1.0 - angle_margin))
    # This is exactly monotone with the acute crossing angle reported by the
    # evaluator.  Tail-CVaR over the largest losses approximates its p05.
    losses = (1.0 - acute_angle / (0.5 * math.pi)).square()
    tail_count = max(int(math.ceil(losses.numel() * float(tail_fraction))), 1)
    return torch.topk(losses, min(tail_count, losses.numel())).values.mean()


def node_edge_clearance_surrogate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
    margin: float = 0.10,
    max_pairs: int = 4096,
    oversample_factor: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hinge loss on hard non-incident node--edge distances."""
    edges = _unique_undirected_edges(edge_index.to(pos.device), pos.size(0))
    if edges.numel() == 0 or int(max_pairs) <= 0:
        return _zero(pos)
    per_node_clearance = []
    for graph_id in torch.unique(batch_index, sorted=True).tolist():
        graph_nodes = torch.nonzero(batch_index == int(graph_id), as_tuple=False).flatten()
        graph_edges = edges[batch_index[edges[:, 0]] == int(graph_id)]
        if graph_nodes.numel() == 0 or graph_edges.numel() == 0:
            continue
        edge_lengths = (pos[graph_edges[:, 1]] - pos[graph_edges[:, 0]]).norm(dim=1)
        # Match the evaluator's graphwise median-edge normalization.  Detaching
        # the scale prevents the optimizer from gaming clearance by shrinking
        # or stretching all edges uniformly.
        scale = edge_lengths.detach().median().clamp_min(float(eps))
        possible = int(graph_nodes.numel()) * int(graph_edges.size(0))
        draw_count = min(
            possible,
            max(int(max_pairs) * max(int(oversample_factor), 1), int(max_pairs)),
        )
        if possible <= draw_count:
            nodes = graph_nodes.repeat_interleave(graph_edges.size(0))
            edge_ids = torch.arange(graph_edges.size(0), device=pos.device).repeat(
                graph_nodes.numel()
            )
        else:
            nodes = graph_nodes[
                torch.randint(graph_nodes.numel(), (draw_count,), device=pos.device)
            ]
            edge_ids = torch.randint(graph_edges.size(0), (draw_count,), device=pos.device)
        chosen_edges = graph_edges[edge_ids]
        valid = (nodes != chosen_edges[:, 0]) & (nodes != chosen_edges[:, 1])
        nodes = nodes[valid]
        chosen_edges = chosen_edges[valid]
        if nodes.numel() == 0:
            continue
        point = pos[nodes]
        start = pos[chosen_edges[:, 0]]
        end = pos[chosen_edges[:, 1]]
        segment = end - start
        denominator = segment.square().sum(dim=1).clamp_min(float(eps))
        fraction = ((point - start) * segment).sum(dim=1) / denominator
        projection = start + fraction.clamp(0.0, 1.0).unsqueeze(1) * segment
        distances = (point - projection).norm(dim=1) / scale
        for node in torch.unique(nodes, sorted=True):
            per_node_clearance.append(distances[nodes == node].min())
    if not per_node_clearance:
        return _zero(pos)
    normalized_distance = torch.stack(per_node_clearance)
    losses = torch.relu(float(margin) - normalized_distance).square()
    # The hinge naturally concentrates gradient on the low-clearance tail while
    # retaining the evaluator's per-node minimum reduction.
    return losses.mean()


def neighborhood_preservation_surrogate(
    pos: torch.Tensor,
    pair_index: torch.Tensor,
    target_distance: torch.Tensor,
    near_hops: float = 2.0,
    margin: float = 0.10,
    max_triplets: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hard triplet loss aligning Euclidean and shortest-path neighbourhoods."""
    pair_index = pair_index.long().to(pos.device)
    target_distance = target_distance.to(device=pos.device, dtype=pos.dtype).reshape(-1)
    if pair_index.numel() == 0 or target_distance.numel() == 0:
        return _zero(pos)
    source, target = pair_index
    valid = source != target
    positive_mask = valid & (target_distance > 0.0) & (target_distance <= float(near_hops))
    negative_mask = valid & (target_distance > float(near_hops))
    anchors = torch.unique(source[positive_mask], sorted=True)
    positive_ids = []
    negative_ids = []
    for anchor in anchors.tolist():
        positives = torch.nonzero(
            positive_mask & (source == int(anchor)), as_tuple=False
        ).flatten()
        negatives = torch.nonzero(
            negative_mask & (source == int(anchor)), as_tuple=False
        ).flatten()
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        with torch.no_grad():
            positive_distance = (pos[target[positives]] - pos[int(anchor)]).norm(dim=1)
            negative_distance = (pos[target[negatives]] - pos[int(anchor)]).norm(dim=1)
            positive_ids.append(positives[torch.argmax(positive_distance)])
            negative_ids.append(negatives[torch.argmin(negative_distance)])
    if not positive_ids:
        return _zero(pos)
    positive_ids = torch.stack(positive_ids)
    negative_ids = torch.stack(negative_ids)
    anchor_nodes = source[positive_ids]
    positive_distance = (pos[target[positive_ids]] - pos[anchor_nodes]).norm(dim=1)
    negative_distance = (pos[target[negative_ids]] - pos[anchor_nodes]).norm(dim=1)
    scale = (0.5 * (positive_distance + negative_distance)).detach().clamp_min(float(eps))
    losses = torch.relu(
        positive_distance / scale - negative_distance / scale + float(margin)
    ).square()
    if int(max_triplets) > 0 and losses.numel() > int(max_triplets):
        with torch.no_grad():
            selected = torch.topk(losses.detach(), int(max_triplets)).indices
        losses = losses[selected]
    return losses.mean()


def readability_surrogate_losses(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
    pair_index: torch.Tensor,
    target_distance: torch.Tensor,
    max_pairs: int = 4096,
    angular_target_ratio: float = 0.5,
    angular_max_target_degrees: float = 30.0,
    crossing_angle_tail_fraction: float = 0.20,
    clearance_margin: float = 0.10,
    neighborhood_near_hops: float = 2.0,
    neighborhood_margin: float = 0.10,
) -> dict[str, torch.Tensor]:
    return {
        "angular": angular_resolution_surrogate(
            pos,
            edge_index,
            target_ratio=angular_target_ratio,
            max_target_degrees=angular_max_target_degrees,
            max_pairs=max_pairs,
        ),
        "crossing_angle": crossing_angle_surrogate(
            pos,
            edge_index,
            batch_index,
            max_pairs=max_pairs,
            tail_fraction=crossing_angle_tail_fraction,
        ),
        "clearance": node_edge_clearance_surrogate(
            pos,
            edge_index,
            batch_index,
            margin=clearance_margin,
            max_pairs=max_pairs,
        ),
        "neighborhood": neighborhood_preservation_surrogate(
            pos,
            pair_index,
            target_distance,
            near_hops=neighborhood_near_hops,
            margin=neighborhood_margin,
            max_triplets=max_pairs,
        ),
    }


def componentwise_readability_dominance_loss(
    candidate_losses: dict[str, torch.Tensor],
    teacher_losses: dict[str, torch.Tensor],
    component_weights: dict[str, float],
    target_improvement: float = 0.0,
    scale_floor: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Minimize the worst relative readability regret against a teacher.

    Positive component weights activate a component; their magnitudes are not
    used so a large weight cannot authorize regression in another component.
    This Chebyshev form is deliberately different from a weighted sum: all
    active surrogate losses must improve before the maximum regret is reduced.
    """
    regrets: dict[str, torch.Tensor] = {}
    for name, weight in component_weights.items():
        if float(weight) <= 0.0:
            continue
        candidate = candidate_losses[name]
        teacher = teacher_losses[name].detach().to(candidate)
        scale = teacher.abs().clamp_min(float(scale_floor))
        regrets[name] = (
            (candidate - teacher) / scale + float(target_improvement)
        )
    if not regrets:
        reference = next(iter(candidate_losses.values()))
        return _zero(reference), regrets
    return torch.stack(list(regrets.values())).max(), regrets
