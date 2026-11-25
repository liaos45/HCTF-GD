"""Optional in-model sparse geometric-conflict messages for HCTF layouts.

The module consumes only the graph, its current neural coordinates, and hidden
features. It never reads exact evaluation metrics or chooses among layouts.
"""

from __future__ import annotations

import torch
from torch import nn

from deepgd.paretofield.crossing import (
    edge_pair_crossing_scores, select_metric_aligned_edge_pairs,
)


@torch.no_grad()
def select_deterministic_conflict_pairs(
    pos: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor,
    max_pairs_per_graph: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select geometry-hard nonincident pairs with no RNG or batch-size budget.

    Small graphs enumerate every pair. Large graphs use a stateless integer
    candidate sequence, then rank those candidates by the same soft crossing
    score on the current layout. The per-graph budget is independent of how
    many other graphs happen to share a batch.
    """
    if max_pairs_per_graph <= 0:
        raise ValueError("max_pairs_per_graph must be positive")
    empty = edge_index.new_empty((0, 2))
    if edge_index.numel() == 0 or batch_index.numel() == 0:
        return empty, empty
    source, target = edge_index.long()
    valid = source != target
    lo = torch.minimum(source[valid], target[valid])
    hi = torch.maximum(source[valid], target[valid])
    if lo.numel() == 0:
        return empty, empty
    edges = torch.unique(torch.stack((lo, hi), dim=1), dim=0, sorted=True)
    first_parts, second_parts = [], []
    graph_count = int(batch_index.max().item()) + 1
    for graph_id in range(graph_count):
        graph_edges = edges[batch_index[edges[:, 0]] == graph_id]
        count = int(graph_edges.size(0))
        if count < 2:
            continue
        if count <= 128:
            pair_ids = torch.triu_indices(count, count, offset=1, device=pos.device)
            left, right = pair_ids[0], pair_ids[1]
        else:
            draw_count = max(int(max_pairs_per_graph) * 32, 2048)
            indices = torch.arange(draw_count, dtype=torch.long, device=pos.device)
            prime = 4294967291
            left_raw = ((indices * 1664525 + 1013904223) % prime) % count
            right_raw = ((indices * 22695477 + 2531011) % prime) % count
            left = torch.minimum(left_raw, right_raw)
            right = torch.maximum(left_raw, right_raw)
            pair_keys = torch.unique(left * count + right, sorted=True)
            left, right = pair_keys // count, pair_keys % count
        first = graph_edges[left]
        second = graph_edges[right]
        nonincident = (
            (left < right)
            & (first[:, 0] != second[:, 0])
            & (first[:, 0] != second[:, 1])
            & (first[:, 1] != second[:, 0])
            & (first[:, 1] != second[:, 1])
        )
        first, second = first[nonincident], second[nonincident]
        if first.numel() == 0:
            continue
        if first.size(0) > max_pairs_per_graph:
            scores = edge_pair_crossing_scores(pos.detach(), first, second)
            chosen = torch.topk(scores, k=max_pairs_per_graph, sorted=False).indices
            first, second = first[chosen], second[chosen]
        first_parts.append(first)
        second_parts.append(second)
    if not first_parts:
        return empty, empty
    return torch.cat(first_parts, dim=0), torch.cat(second_parts, dim=0)


class HCTFConflictHead(nn.Module):
    """Predict bounded SE(2)-equivariant coordinate messages from edge pairs."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32,
                 max_pairs: int = 1024, max_step: float = 0.04,
                 deterministic_pairs: bool = False,
                 graph_context: bool = False):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or max_pairs <= 0:
            raise ValueError("feature_dim, hidden_dim, max_pairs must be positive")
        if max_step < 0:
            raise ValueError("max_step must be nonnegative")
        self.max_pairs = int(max_pairs)
        self.max_step = float(max_step)
        self.deterministic_pairs = bool(deterministic_pairs)
        self.graph_context = bool(graph_context)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.message_net = nn.Sequential(
            nn.Linear(2 * feature_dim + 6 + (4 if self.graph_context else 0), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # The enabled architecture exactly nests the source checkpoint.
        nn.init.zeros_(self.message_net[-1].weight)
        nn.init.zeros_(self.message_net[-1].bias)
        self.last_candidate_pairs = 0
        self.last_active_pairs = 0
        self.last_mean_displacement_fraction = 0.0

    @staticmethod
    def _graph_edge_scales(pos: torch.Tensor, edge_index: torch.Tensor,
                           batch_index: torch.Tensor) -> torch.Tensor:
        graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        scales = pos.new_ones(graph_count)
        if graph_count == 0 or edge_index.numel() == 0:
            return scales
        source, target = edge_index.long()
        valid = source != target
        source, target = source[valid], target[valid]
        if source.numel() == 0:
            return scales
        lo, hi = torch.minimum(source, target), torch.maximum(source, target)
        undirected = torch.unique(torch.stack([lo, hi], dim=1), dim=0)
        lengths = (pos[undirected[:, 0]] - pos[undirected[:, 1]]).norm(dim=1)
        for graph_id in range(graph_count):
            mask = batch_index[undirected[:, 0]] == graph_id
            if mask.any():
                scales[graph_id] = lengths[mask].detach().median().clamp_min(1e-6)
            else:
                node_mask = batch_index == graph_id
                if node_mask.any():
                    scales[graph_id] = pos[node_mask].detach().std().clamp_min(1e-6)
        return scales

    def forward(self, pos: torch.Tensor, hidden: torch.Tensor,
                edge_index: torch.Tensor, community_labels: torch.Tensor | None,
                batch_index: torch.Tensor) -> torch.Tensor:
        if pos.numel() == 0 or edge_index is None or edge_index.numel() == 0:
            return pos
        if hidden is None or hidden.size(0) != pos.size(0):
            raise ValueError("conflict head requires aligned per-node hidden features")
        if self.deterministic_pairs:
            first, second = select_deterministic_conflict_pairs(
                pos.detach(), edge_index, batch_index, self.max_pairs,
            )
        else:
            first, second = select_metric_aligned_edge_pairs(
                pos.detach(), edge_index, batch_index,
                max_pairs=self.max_pairs, exact_max_edges=32,
            )
        self.last_candidate_pairs = int(first.size(0))
        if first.numel() == 0:
            self.last_active_pairs = 0
            self.last_mean_displacement_fraction = 0.0
            return pos
        scores = edge_pair_crossing_scores(pos, first, second)
        weights = torch.relu(scores - 0.25)
        self.last_active_pairs = int((weights.detach() > 0.25).sum().item())
        if float(weights.detach().max()) <= 0.0:
            self.last_mean_displacement_fraction = 0.0
            return pos
        graph_scales = self._graph_edge_scales(pos, edge_index, batch_index)
        pair_graph = batch_index[first[:, 0]].long()
        scale = graph_scales[pair_graph].clamp_min(1e-6)
        a, b = first[:, 0], first[:, 1]
        c, d = second[:, 0], second[:, 1]
        node = torch.cat((a, b, c, d))
        own_start = torch.cat((a, a, c, c))
        own_end = torch.cat((b, b, d, d))
        other_start = torch.cat((c, c, a, a))
        other_end = torch.cat((d, d, b, b))
        frame = pos[own_end] - pos[own_start]
        frame_length = frame.norm(dim=1, keepdim=True).clamp_min(1e-8)
        tangent = frame / frame_length
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
        other_midpoint = 0.5 * (pos[other_start] + pos[other_end])
        own_midpoint = 0.5 * (pos[own_start] + pos[own_end])
        relative = other_midpoint - pos[node]
        local_scale = scale.repeat(4)
        other_length = (pos[other_end] - pos[other_start]).norm(dim=1).clamp_min(1e-8)
        pair_score = scores.repeat(4)
        if community_labels is None:
            same_community = pos.new_zeros(node.numel())
        else:
            labels = community_labels.long().to(pos.device)
            same_community = 0.5 * (
                (labels[node] == labels[other_start]).to(pos.dtype)
                + (labels[node] == labels[other_end]).to(pos.dtype)
            )
        geometry = torch.stack((
            pair_score,
            (frame_length.squeeze(1) / other_length).log().clamp(-5, 5),
            (relative * tangent).sum(dim=1) / local_scale,
            (relative * normal).sum(dim=1) / local_scale,
            ((pos[node] - own_midpoint) * tangent).sum(dim=1) / local_scale,
            same_community,
        ), dim=1)
        normalized_hidden = self.feature_norm(hidden)
        other_hidden = 0.5 * (
            normalized_hidden[other_start] + normalized_hidden[other_end]
        )
        message_features = [normalized_hidden[node], other_hidden, geometry]
        if self.graph_context:
            graph_count = graph_scales.numel()
            graph_size = pos.new_zeros(graph_count)
            graph_size.index_add_(0, batch_index.long(), pos.new_ones(pos.size(0)))
            source, target = edge_index.long()
            valid_edges = source != target
            unique_edges = torch.unique(torch.stack((
                torch.minimum(source[valid_edges], target[valid_edges]),
                torch.maximum(source[valid_edges], target[valid_edges]),
            ), dim=1), dim=0)
            source, target = unique_edges[:, 0], unique_edges[:, 1]
            edge_graph = batch_index[source].long()
            edge_count = pos.new_zeros(graph_count)
            edge_count.index_add_(0, edge_graph, pos.new_ones(edge_graph.numel()))
            density = edge_count / (
                graph_size * (graph_size - 1.0) / 2.0
            ).clamp_min(1.0)
            conflict_sum = pos.new_zeros(graph_count)
            conflict_count = pos.new_zeros(graph_count)
            conflict_sum.index_add_(0, pair_graph, (scores.detach() > 0.5).to(pos.dtype))
            conflict_count.index_add_(0, pair_graph, pos.new_ones(scores.numel()))
            if community_labels is None:
                boundary_ratio = pos.new_zeros(graph_count)
            else:
                labels = community_labels.long().to(pos.device)
                boundary = (
                    labels[source] != labels[target]
                ).to(pos.dtype)
                boundary_sum = pos.new_zeros(graph_count)
                boundary_sum.index_add_(0, edge_graph, boundary)
                boundary_ratio = boundary_sum / edge_count.clamp_min(1.0)
            context = torch.stack((
                torch.log1p(graph_size) / 6.0,
                density,
                conflict_sum / conflict_count.clamp_min(1.0),
                boundary_ratio,
            ), dim=1)
            message_features.append(context[pair_graph].repeat(4, 1))
        coefficients = self.message_net(torch.cat(message_features, dim=1))
        message = (coefficients[:, :1] * tangent
                   + coefficients[:, 1:] * normal)
        repeated_weights = weights.repeat(4).unsqueeze(1)
        message = message * repeated_weights
        delta = pos.new_zeros(pos.shape)
        normalizer = pos.new_zeros((pos.size(0), 1))
        delta.index_add_(0, node, message)
        normalizer.index_add_(0, node, repeated_weights)
        delta = delta / normalizer.clamp_min(1e-8)
        graph_count = graph_scales.numel()
        means = pos.new_zeros((graph_count, 2))
        counts = pos.new_zeros((graph_count, 1))
        means.index_add_(0, batch_index.long(), delta)
        counts.index_add_(0, batch_index.long(), pos.new_ones((pos.size(0), 1)))
        delta = delta - (means / counts.clamp_min(1.0))[batch_index.long()]
        cap = (self.max_step * graph_scales[batch_index.long()]).unsqueeze(1)
        magnitude = delta.norm(dim=1, keepdim=True)
        # A hard norm cap or tanh(norm)/norm formulation has zero/undefined
        # derivative at the deliberately zero-initialized output. This smooth
        # rational cap is identity to first order at zero and remains bounded.
        delta = delta * (cap / (cap + magnitude).clamp_min(1e-8))
        self.last_mean_displacement_fraction = float(
            (delta.detach().norm(dim=1) / graph_scales[batch_index.long()]).mean()
        )
        return pos + delta


class HCTFGraphSizeGate(nn.Module):
    """Learn a graph-local HCTF residual gain while exactly nesting the source.

    The gate uses node count only, never quality metrics or a reference layout.
    A zero raw strength gives gain one with a nonzero training gradient.
    """

    def __init__(self, midpoint: float = 20.0, width: float = 5.0):
        super().__init__()
        if midpoint <= 0 or width <= 0:
            raise ValueError("midpoint and width must be positive")
        self.midpoint = float(midpoint)
        self.width = float(width)
        self.raw_strength = nn.Parameter(torch.zeros(()))

    def forward(self, before: torch.Tensor, after: torch.Tensor,
                batch_index: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape or before.size(0) != batch_index.numel():
            raise ValueError("graph-size gate requires aligned node tensors")
        if batch_index.numel() == 0:
            return after
        graph_count = int(batch_index.max().item()) + 1
        sizes = before.new_zeros(graph_count)
        sizes.index_add_(0, batch_index.long(), before.new_ones(batch_index.numel()))
        smallness = torch.sigmoid((self.midpoint - sizes) / self.width)
        gain = 1.0 + torch.tanh(self.raw_strength) * smallness[batch_index.long()]
        return after + (gain - 1.0).unsqueeze(1) * (after - before)
