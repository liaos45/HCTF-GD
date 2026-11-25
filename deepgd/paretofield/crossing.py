from __future__ import annotations

import torch


def _unique_undirected_edges(edge_index, num_nodes):
    src, dst = edge_index.long()
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


def _cross2(first, second):
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _crossing_scores(pos, first_edges, second_edges, temperature, eps):
    a, b = pos[first_edges[:, 0]], pos[first_edges[:, 1]]
    c, d = pos[second_edges[:, 0]], pos[second_edges[:, 1]]
    ab, cd = b - a, d - c

    def normalized_cross(first, second):
        return _cross2(first, second) / (first.norm(dim=1) * second.norm(dim=1) + eps)

    s1 = normalized_cross(ab, c - a)
    s2 = normalized_cross(ab, d - a)
    s3 = normalized_cross(cd, a - c)
    s4 = normalized_cross(cd, b - c)
    return torch.sigmoid(-float(temperature) * s1 * s2) * torch.sigmoid(
        -float(temperature) * s3 * s4
    )


def edge_pair_crossing_scores(
    pos,
    first_edges,
    second_edges,
    temperature: float = 8.0,
    eps: float = 1e-8,
):
    """Return differentiable crossing probabilities for fixed edge pairs."""
    if first_edges.numel() == 0:
        return pos.new_zeros(0)
    return _crossing_scores(
        pos,
        first_edges.long(),
        second_edges.long(),
        temperature,
        eps,
    )


def _non_crossing_clearance(pos, first_edges, second_edges, eps=1e-8):
    """Return a signed segment-separation witness for fixed edge pairs.

    Ignoring degenerate collinear cases, two segments cross only when both
    orientation products are negative.  The maximum product is therefore
    non-negative whenever at least one segment separates the other pair.
    """
    if first_edges.numel() == 0:
        return pos.new_zeros(0)
    first_edges = first_edges.long().to(pos.device)
    second_edges = second_edges.long().to(pos.device)
    a, b = pos[first_edges[:, 0]], pos[first_edges[:, 1]]
    c, d = pos[second_edges[:, 0]], pos[second_edges[:, 1]]
    ab, cd = b - a, d - c

    def normalized_cross(first, second):
        return _cross2(first, second) / (
            first.norm(dim=1) * second.norm(dim=1) + float(eps)
        )

    side_ab = normalized_cross(ab, c - a) * normalized_cross(ab, d - a)
    side_cd = normalized_cross(cd, a - c) * normalized_cross(cd, b - c)
    return torch.maximum(side_ab, side_cd)


def _select_hard_pairs(pos, edges, batch_index, max_pairs, oversample_factor, ratios, temperature):
    num_edges = int(edges.size(0))
    draw_count = max(int(max_pairs) * max(int(oversample_factor), 1), int(max_pairs))
    first = torch.randint(num_edges, (draw_count,), device=pos.device)
    second = torch.randint(num_edges, (draw_count,), device=pos.device)
    lo, hi = torch.minimum(first, second), torch.maximum(first, second)
    e1, e2 = edges[lo], edges[hi]
    valid = (
        (lo < hi)
        & (batch_index[e1[:, 0]] == batch_index[e2[:, 0]])
        & (e1[:, 0] != e2[:, 0])
        & (e1[:, 0] != e2[:, 1])
        & (e1[:, 1] != e2[:, 0])
        & (e1[:, 1] != e2[:, 1])
    )
    if not valid.any():
        return e1[:0], e2[:0]
    lo, hi, e1, e2 = lo[valid], hi[valid], e1[valid], e2[valid]
    key = lo * num_edges + hi
    order = torch.argsort(key)
    key, e1, e2 = key[order], e1[order], e2[order]
    keep = torch.ones(key.numel(), dtype=torch.bool, device=pos.device)
    keep[1:] = key[1:] != key[:-1]
    e1, e2 = e1[keep], e2[keep]

    with torch.no_grad():
        scores = _crossing_scores(pos.detach(), e1, e2, temperature, 1e-8)
        a, b = pos.detach()[e1[:, 0]], pos.detach()[e1[:, 1]]
        c, d = pos.detach()[e2[:, 0]], pos.detach()[e2[:, 1]]
        bbox_overlap = (
            (torch.maximum(torch.minimum(a[:, 0], b[:, 0]), torch.minimum(c[:, 0], d[:, 0]))
             <= torch.minimum(torch.maximum(a[:, 0], b[:, 0]), torch.maximum(c[:, 0], d[:, 0])))
            &
            (torch.maximum(torch.minimum(a[:, 1], b[:, 1]), torch.minimum(c[:, 1], d[:, 1]))
             <= torch.minimum(torch.maximum(a[:, 1], b[:, 1]), torch.maximum(c[:, 1], d[:, 1])))
        )
        midpoint_distance = torch.norm(0.5 * (a + b) - 0.5 * (c + d), dim=1)
        edge_scale = 0.5 * (torch.norm(a - b, dim=1) + torch.norm(c - d, dim=1)).clamp_min(1e-8)
        near_score = -(midpoint_distance / edge_scale)

        selected = []
        used = torch.zeros(e1.size(0), dtype=torch.bool, device=pos.device)

        def take(values, count, mask=None):
            if count <= 0:
                return
            available = ~used if mask is None else (~used & mask)
            ids = torch.nonzero(available, as_tuple=False).view(-1)
            if ids.numel() == 0:
                return
            top = ids[torch.argsort(values[ids], descending=True)[:count]]
            used[top] = True
            selected.append(top)

        hard_count = int(max_pairs * ratios[0])
        bbox_count = int(max_pairs * ratios[1])
        near_count = int(max_pairs * ratios[2])
        take(scores, hard_count)
        take(scores, bbox_count, bbox_overlap)
        take(near_score, near_count)
        random_count = max(int(max_pairs) - sum(index.numel() for index in selected), 0)
        random_ids = torch.nonzero(~used, as_tuple=False).view(-1)
        if random_ids.numel() > 0 and random_count > 0:
            random_ids = random_ids[torch.randperm(random_ids.numel(), device=pos.device)[:random_count]]
            selected.append(random_ids)
        if not selected:
            return e1[:0], e2[:0]
        selected = torch.cat(selected)[: int(max_pairs)]
    return e1[selected], e2[selected]


@torch.no_grad()
def sample_hard_edge_pairs(
    pos,
    edge_index,
    batch_index,
    max_pairs=2048,
    oversample_factor=8,
    temperature=8.0,
    ratios=(0.40, 0.30, 0.20, 0.10),
):
    """Sample one fixed set of difficult, non-adjacent edge pairs per batch.

    Keeping the sampled pairs separate from score evaluation lets the layout
    field and line-search guard compare multiple coordinate candidates against
    exactly the same edge-pair audit set.
    """
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    edges = _unique_undirected_edges(edge_index.to(pos.device), pos.size(0))
    empty = edge_index.new_zeros((0, 2), device=pos.device)
    if graph_count == 0 or edges.size(0) < 2 or int(max_pairs) <= 0:
        return empty, empty

    per_graph_budget = max(int(max_pairs) // max(graph_count, 1), 1)
    first_parts = []
    second_parts = []
    for graph_id in range(graph_count):
        graph_edges = edges[batch_index[edges[:, 0]] == graph_id]
        if graph_edges.size(0) < 2:
            continue
        first, second = _select_hard_pairs(
            pos,
            graph_edges,
            batch_index,
            per_graph_budget,
            oversample_factor,
            ratios,
            temperature,
        )
        if first.numel():
            first_parts.append(first)
            second_parts.append(second)

    if not first_parts:
        return empty, empty
    return torch.cat(first_parts, dim=0), torch.cat(second_parts, dim=0)


@torch.no_grad()
def select_metric_aligned_edge_pairs(
    pos,
    edge_index,
    batch_index,
    max_pairs=4096,
    exact_max_edges=128,
    oversample_factor=8,
    temperature=8.0,
    ratios=(0.40, 0.30, 0.20, 0.10),
):
    """Select fixed pairs, exhaustively for small graphs and sampled otherwise."""
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    edges = _unique_undirected_edges(edge_index.to(pos.device), pos.size(0))
    empty = edge_index.new_zeros((0, 2), device=pos.device)
    if graph_count == 0 or edges.size(0) < 2:
        return empty, empty

    per_graph_budget = max(int(max_pairs) // max(graph_count, 1), 1)
    first_parts = []
    second_parts = []
    for graph_id in range(graph_count):
        graph_edges = edges[batch_index[edges[:, 0]] == graph_id]
        edge_count = int(graph_edges.size(0))
        if edge_count < 2:
            continue
        if edge_count <= int(exact_max_edges):
            pair_ids = torch.triu_indices(
                edge_count,
                edge_count,
                offset=1,
                device=pos.device,
            )
            first = graph_edges[pair_ids[0]]
            second = graph_edges[pair_ids[1]]
            valid = (
                (first[:, 0] != second[:, 0])
                & (first[:, 0] != second[:, 1])
                & (first[:, 1] != second[:, 0])
                & (first[:, 1] != second[:, 1])
            )
            first, second = first[valid], second[valid]
        else:
            first, second = _select_hard_pairs(
                pos,
                graph_edges,
                batch_index,
                per_graph_budget,
                oversample_factor,
                ratios,
                temperature,
            )
        if first.numel():
            first_parts.append(first)
            second_parts.append(second)

    if not first_parts:
        return empty, empty
    return torch.cat(first_parts, dim=0), torch.cat(second_parts, dim=0)


def eligible_edge_pair_counts(edge_index, batch_index, num_nodes):
    """Count non-adjacent undirected edge pairs for every graph."""
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    result = torch.zeros(graph_count, dtype=torch.float32, device=batch_index.device)
    if graph_count == 0:
        return result
    edges = _unique_undirected_edges(edge_index.to(batch_index.device), int(num_nodes))
    for graph_id in range(graph_count):
        graph_edges = edges[batch_index[edges[:, 0]] == graph_id]
        edge_count = int(graph_edges.size(0))
        if edge_count < 2:
            continue
        nodes = graph_edges.reshape(-1)
        degree = torch.bincount(nodes, minlength=int(num_nodes)).to(torch.float32)
        all_pairs = edge_count * (edge_count - 1) / 2.0
        adjacent_pairs = (degree * (degree - 1.0) / 2.0).sum()
        result[graph_id] = max(all_pairs - float(adjacent_pairs.item()), 0.0)
    return result


def crossing_objectives_from_edge_pairs_per_graph(
    pos,
    first_edges,
    second_edges,
    edge_index,
    batch_index,
    temperature=8.0,
    margin=0.25,
):
    """Return count-like and normalized crossing objectives per graph."""
    normalized = crossing_loss_from_edge_pairs_per_graph(
        pos=pos,
        first_edges=first_edges,
        second_edges=second_edges,
        batch_index=batch_index,
        temperature=temperature,
        margin=margin,
    )
    eligible = eligible_edge_pair_counts(
        edge_index=edge_index,
        batch_index=batch_index,
        num_nodes=pos.size(0),
    ).to(dtype=pos.dtype)
    return normalized * eligible, normalized


def crossing_topology_barrier_per_graph(
    candidate_pos,
    teacher_pos,
    first_edges,
    second_edges,
    batch_index,
    margin=0.02,
    preserve_only=False,
    tail_fraction=0.0,
):
    """Protect noncrossings, optionally without pushing old crossings apart."""
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    result = candidate_pos.new_zeros(graph_count)
    if graph_count == 0 or first_edges.numel() == 0:
        return result + candidate_pos.sum() * 0.0

    with torch.no_grad():
        teacher_clearance = _non_crossing_clearance(
            teacher_pos.detach(), first_edges, second_edges
        )
        preserved_margin = teacher_clearance.clamp(min=0.0, max=float(margin))
        target = torch.where(
            teacher_clearance >= 0.0,
            preserved_margin,
            preserved_margin.new_zeros(()),
        )
    candidate_clearance = _non_crossing_clearance(
        candidate_pos, first_edges, second_edges
    )
    losses = torch.relu(target - candidate_clearance).square()
    if preserve_only:
        losses = losses * (teacher_clearance >= 0.0).to(losses.dtype)
    pair_graph = batch_index.long()[first_edges.long().to(batch_index.device)[:, 0]]
    if tail_fraction > 0.0:
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must lie in (0, 1]")
        for graph_id in range(graph_count):
            mask = pair_graph == graph_id
            if preserve_only:
                mask = mask & (teacher_clearance >= 0.0)
            graph_losses = losses[mask]
            if graph_losses.numel() > 0:
                keep = max(1, int(torch.ceil(
                    graph_losses.new_tensor(graph_losses.numel() * tail_fraction)
                ).item()))
                result[graph_id] = torch.topk(graph_losses, k=keep).values.mean()
        return result
    counts = candidate_pos.new_zeros(graph_count)
    result.index_add_(0, pair_graph, losses)
    counts.index_add_(0, pair_graph, torch.ones_like(losses))
    return result / counts.clamp_min(1.0)


def crossing_loss_from_edge_pairs_per_graph(
    pos,
    first_edges,
    second_edges,
    batch_index,
    temperature=8.0,
    margin=0.25,
):
    """Evaluate a fixed edge-pair audit set and reduce scores per graph."""
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    result = pos.new_zeros(graph_count)
    if graph_count == 0 or first_edges.numel() == 0:
        return result + pos.sum() * 0.0

    first_edges = first_edges.long().to(pos.device)
    second_edges = second_edges.long().to(pos.device)
    scores = edge_pair_crossing_scores(
        pos,
        first_edges,
        second_edges,
        temperature=temperature,
    )
    losses = torch.relu(scores - float(margin))
    pair_graph = batch_index.long()[first_edges[:, 0]]
    counts = pos.new_zeros(graph_count)
    result.index_add_(0, pair_graph, losses)
    counts.index_add_(0, pair_graph, torch.ones_like(losses))
    return result / counts.clamp_min(1.0)


def hard_sampled_crossing_loss_per_graph(
    pos,
    edge_index,
    batch_index,
    max_pairs=2048,
    oversample_factor=8,
    temperature=8.0,
    margin=0.25,
    ratios=(0.40, 0.30, 0.20, 0.10),
):
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    if graph_count == 0:
        return pos.new_zeros(0)
    first_edges, second_edges = sample_hard_edge_pairs(
        pos=pos,
        edge_index=edge_index,
        batch_index=batch_index,
        max_pairs=max_pairs,
        oversample_factor=oversample_factor,
        temperature=temperature,
        ratios=ratios,
    )
    return crossing_loss_from_edge_pairs_per_graph(
        pos=pos,
        first_edges=first_edges,
        second_edges=second_edges,
        batch_index=batch_index,
        temperature=temperature,
        margin=margin,
    )


def hard_sampled_crossing_loss(
    pos,
    edge_index,
    batch_index,
    max_pairs=2048,
    oversample_factor=8,
    temperature=8.0,
    margin=0.25,
    ratios=(0.40, 0.30, 0.20, 0.10),
):
    values = hard_sampled_crossing_loss_per_graph(
        pos=pos,
        edge_index=edge_index,
        batch_index=batch_index,
        max_pairs=max_pairs,
        oversample_factor=oversample_factor,
        temperature=temperature,
        margin=margin,
        ratios=ratios,
    )
    return values.mean() if values.numel() else pos.sum() * 0.0
