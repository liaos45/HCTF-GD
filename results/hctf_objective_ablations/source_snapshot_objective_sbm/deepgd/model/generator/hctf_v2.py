"""Second-generation hierarchical community tree field for DeepGD.

The v2 path is intentionally separate from :mod:`hctf` so that legacy HCTF
experiments remain bit-for-bit reproducible.  It adds four mechanisms that the
original synchronous block did not provide: depth-ordered bottom-up/top-down
tree sweeps, per-tree attention, a core/bridge prototype controller, and an
E(2)-equivariant coordinate residual with graph-level confidence gating.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.nn import functional as F

from .hctf import TreeCoverBatch


def _normalized_preference(values: Sequence[float]) -> Tensor:
    preference = torch.as_tensor(tuple(values), dtype=torch.float32)
    if preference.numel() != 3:
        raise ValueError("HCTF preference must contain geometry, visual, community")
    preference = preference.clamp_min(0.0)
    return preference / preference.sum().clamp_min(1e-8)


class HCTFv2Block(nn.Module):
    """Hierarchical multi-tree feature adapter with confidence-aware fusion."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_trees: int,
        dropout: float = 0.0,
        scale: float = 0.20,
        mix_init: float = 0.05,
        preference: Sequence[float] = (0.34, 0.33, 0.33),
        graph_gate_enabled: bool = True,
        tree_attention_enabled: bool = True,
        zero_init: bool = False,
        tree_messages_enabled: bool = True,
        direction_conditioning_enabled: bool = True,
        relation_conditioning_enabled: bool = True,
        prototype_context_enabled: bool = True,
        node_only_control: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or num_trees < 1:
            raise ValueError("feature_dim, hidden_dim, and num_trees must be positive")
        if scale < 0.0 or mix_init < 0.0:
            raise ValueError("scale and mix_init cannot be negative")

        self.feature_dim = int(feature_dim)
        self.num_trees = int(num_trees)
        self.scale = float(scale)
        self.graph_gate_enabled = bool(graph_gate_enabled)
        self.tree_attention_enabled = bool(tree_attention_enabled)
        self.tree_messages_enabled = bool(tree_messages_enabled)
        self.direction_conditioning_enabled = bool(direction_conditioning_enabled)
        self.relation_conditioning_enabled = bool(relation_conditioning_enabled)
        self.prototype_context_enabled = bool(prototype_context_enabled)
        self.node_only_control = bool(node_only_control)
        self.register_buffer("preference", _normalized_preference(preference))
        self.register_buffer("residual_progress", torch.tensor(1.0))

        self.input_norm = nn.LayerNorm(feature_dim)
        self.up_message = nn.Linear(feature_dim, feature_dim, bias=False)
        self.down_message = nn.Linear(feature_dim, feature_dim, bias=False)
        self.up_update = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.down_update = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.up_direction = nn.Linear(feature_dim, 1, bias=False)
        self.down_direction = nn.Linear(feature_dim, 1, bias=False)
        self.relation_gate = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.tree_attention = nn.Sequential(
            nn.Linear(2 * feature_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.prototype_controller = nn.Sequential(
            nn.Linear(feature_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.update_net = nn.Sequential(
            nn.Linear(3 * feature_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(3 * feature_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.graph_gate = None
        if self.graph_gate_enabled:
            self.graph_gate = nn.Sequential(
                nn.Linear(12, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.dropout = nn.Dropout(dropout)

        if scale <= 0.0 or mix_init <= 0.0:
            initial_logit = -20.0
        else:
            ratio = min(max(float(mix_init) / float(scale), 1e-5), 1.0 - 1e-5)
            initial_logit = math.log(ratio / (1.0 - ratio))
        self.mix_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

        # Neutral relation weights and useful, non-saturated residual gates.
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)
        nn.init.zeros_(self.prototype_controller[-1].weight)
        nn.init.zeros_(self.prototype_controller[-1].bias)
        nn.init.zeros_(self.node_gate[-2].weight)
        nn.init.constant_(self.node_gate[-2].bias, 1.0)
        if self.graph_gate is not None:
            nn.init.zeros_(self.graph_gate[-1].weight)
            nn.init.constant_(self.graph_gate[-1].bias, 1.0)
        if zero_init:
            nn.init.zeros_(self.update_net[-1].weight)
            nn.init.zeros_(self.update_net[-1].bias)

        self.last_mean_gate = 0.0
        self.last_mean_graph_gate = 1.0
        self.last_mix = 0.0
        self.last_tree_attention_entropy = 0.0
        self.last_partition_confidence = 0.0

    def set_residual_progress(self, progress: float) -> None:
        self.residual_progress.fill_(min(max(float(progress), 0.0), 1.0))

    def set_preference(self, preference: Sequence[float]) -> None:
        self.preference.copy_(_normalized_preference(preference).to(self.preference))

    @staticmethod
    def _segment_sum(values: Tensor, index: LongTensor, count: int) -> Tensor:
        output = values.new_zeros((count, values.size(-1)))
        if values.numel():
            output.index_add_(0, index, values)
        return output

    @classmethod
    def _segment_mean(cls, values: Tensor, index: LongTensor, count: int) -> Tensor:
        output = cls._segment_sum(values, index, count)
        normalizer = values.new_zeros((count, 1))
        if values.numel():
            normalizer.index_add_(0, index, values.new_ones((values.size(0), 1)))
        return output / normalizer.clamp_min(1.0)

    @staticmethod
    def _labels(labels: Optional[LongTensor], node_count: int, device) -> LongTensor:
        if labels is None:
            return torch.zeros(node_count, dtype=torch.long, device=device)
        return labels.to(device=device, dtype=torch.long)

    @staticmethod
    def _boundary_statistics(
        graph_edge_index: LongTensor,
        labels: LongTensor,
        node_count: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        degree = torch.zeros(node_count, dtype=dtype, device=labels.device)
        boundary = torch.zeros_like(degree)
        if graph_edge_index.numel() == 0:
            return degree, boundary
        source, target = graph_edge_index.long()
        valid = (
            (source >= 0)
            & (target >= 0)
            & (source < node_count)
            & (target < node_count)
            & (source != target)
        )
        source, target = source[valid], target[valid]
        if source.numel() == 0:
            return degree, boundary
        ones = degree.new_ones(source.numel())
        cross = (labels[source] != labels[target]).to(dtype)
        degree.index_add_(0, source, ones)
        degree.index_add_(0, target, ones)
        boundary.index_add_(0, source, cross)
        boundary.index_add_(0, target, cross)
        return degree, boundary / degree.clamp_min(1.0)

    @classmethod
    def graph_context(
        cls,
        graph_edge_index: LongTensor,
        labels: LongTensor,
        batch_index: LongTensor,
        dtype: torch.dtype,
    ) -> Tuple[LongTensor, Tensor, Tensor, Tensor, Tensor]:
        """Return inverse graph ids, descriptors, confidence, degree, boundary."""
        _, graph_inverse = torch.unique(
            batch_index.long(), sorted=True, return_inverse=True
        )
        graph_count = int(graph_inverse.max().item()) + 1 if graph_inverse.numel() else 0
        node_count = int(batch_index.numel())
        degree, boundary = cls._boundary_statistics(
            graph_edge_index, labels, node_count, dtype
        )
        if graph_count == 0:
            return graph_inverse, torch.empty((0, 9), device=batch_index.device), (
                torch.empty((0, 1), device=batch_index.device)
            ), degree, boundary

        boundary_col = boundary[:, None]
        degree_col = torch.log1p(degree)[:, None]
        boundary_mean = cls._segment_mean(boundary_col, graph_inverse, graph_count)
        degree_mean = cls._segment_mean(degree_col, graph_inverse, graph_count)
        boundary_var = cls._segment_mean(
            (boundary_col - boundary_mean[graph_inverse]).square(),
            graph_inverse,
            graph_count,
        )
        degree_var = cls._segment_mean(
            (degree_col - degree_mean[graph_inverse]).square(),
            graph_inverse,
            graph_count,
        )
        node_sizes = degree.new_zeros((graph_count, 1))
        node_sizes.index_add_(0, graph_inverse, degree.new_ones((node_count, 1)))
        edge_counts = cls._segment_sum(degree[:, None], graph_inverse, graph_count) / 2.0
        density = edge_counts / (node_sizes * (node_sizes - 1.0)).clamp_min(1.0)
        density = density.clamp(0.0, 1.0)

        community_counts = degree.new_ones((graph_count, 1))
        entropy = degree.new_zeros((graph_count, 1))
        modularity = degree.new_zeros((graph_count, 1))
        edge_source = edge_target = None
        if graph_edge_index.numel():
            edge_source, edge_target = graph_edge_index.long()
            edge_valid = (
                (edge_source >= 0)
                & (edge_target >= 0)
                & (edge_source < node_count)
                & (edge_target < node_count)
                & (edge_source != edge_target)
                & (graph_inverse[edge_source] == graph_inverse[edge_target])
            )
            edge_source, edge_target = edge_source[edge_valid], edge_target[edge_valid]
        for graph_id in range(graph_count):
            local_labels = labels[graph_inverse == graph_id]
            if local_labels.numel() == 0:
                continue
            _, counts = torch.unique(local_labels, return_counts=True)
            community_counts[graph_id] = float(counts.numel())
            probabilities = counts.to(dtype=dtype) / counts.sum().clamp_min(1)
            local_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
            if counts.numel() > 1:
                local_entropy = local_entropy / math.log(float(counts.numel()))
            entropy[graph_id] = local_entropy
            if edge_source is not None and edge_source.numel():
                local_edge_mask = graph_inverse[edge_source] == graph_id
                local_source = edge_source[local_edge_mask]
                local_target = edge_target[local_edge_mask]
                if local_source.numel():
                    within = (
                        labels[local_source] == labels[local_target]
                    ).to(dtype).mean()
                    local_nodes = torch.nonzero(
                        graph_inverse == graph_id, as_tuple=False
                    ).flatten()
                    local_degree = degree[local_nodes]
                    total_volume = local_degree.sum().clamp_min(1.0)
                    volume_square = degree.new_tensor(0.0)
                    for community in torch.unique(labels[local_nodes]):
                        volume = local_degree[
                            labels[local_nodes] == community
                        ].sum() / total_volume
                        volume_square = volume_square + volume.square()
                    modularity[graph_id] = within - volume_square

        within_fraction = 1.0 - boundary_mean
        confidence = torch.sigmoid(
            2.0 * (within_fraction - 0.5) + 4.0 * modularity
        )
        confidence = confidence * (0.5 + 0.5 * entropy.clamp(0.0, 1.0))
        descriptor = torch.cat(
            [
                boundary_mean,
                boundary_var.sqrt(),
                torch.tanh(degree_mean),
                torch.tanh(degree_var.sqrt()),
                torch.log1p(node_sizes) / 6.0,
                density,
                torch.log1p(community_counts) / 3.0,
                modularity,
                confidence,
            ],
            dim=1,
        )
        return graph_inverse, descriptor, confidence, degree, boundary

    def _edge_weights(
        self,
        normalized: Tensor,
        source: LongTensor,
        target: LongTensor,
        relation: Tensor,
        upward: bool,
    ) -> Tensor:
        if self.relation_conditioning_enabled:
            relation_weight = 2.0 * torch.sigmoid(
                self.relation_gate(relation).squeeze(-1)
            )
        else:
            relation_weight = normalized.new_ones(source.numel())
        if not self.direction_conditioning_enabled:
            return relation_weight
        difference = normalized[source] - normalized[target]
        scorer = self.up_direction if upward else self.down_direction
        return relation_weight * (2.0 * torch.sigmoid(scorer(difference).squeeze(-1)))

    def _sweep_tree(
        self,
        normalized: Tensor,
        cover: TreeCoverBatch,
        tree_id: int,
        relation: Tensor,
    ) -> Tensor:
        source_all, target_all = cover.edge_index.long()
        orientation = cover.orientation
        depth = cover.depth
        if depth is None or depth.numel() != orientation.numel():
            depth = torch.ones_like(cover.tree_id)
        depth = depth.long()
        tree_mask = cover.tree_id.long() == int(tree_id)
        max_depth = int(depth[tree_mask].max().item()) if bool(tree_mask.any()) else 0

        state = normalized
        for level in range(max_depth, 0, -1):
            mask = tree_mask & (orientation < 0) & (depth == level)
            if not bool(mask.any()):
                continue
            source, target = source_all[mask], target_all[mask]
            weight = self._edge_weights(
                state, source, target, relation[mask], upward=True
            )
            messages = self.up_message(state[source]) * weight[:, None]
            aggregate = state.new_zeros(state.shape)
            normalizer = state.new_zeros((state.size(0), 1))
            aggregate.index_add_(0, target, messages)
            normalizer.index_add_(0, target, weight[:, None])
            aggregate = aggregate / normalizer.clamp_min(1.0)
            candidate = state + self.up_update(torch.cat([state, aggregate], dim=-1))
            state = torch.where(normalizer > 0, candidate, state)

        for level in range(1, max_depth + 1):
            mask = tree_mask & (orientation > 0) & (depth == level)
            if not bool(mask.any()):
                continue
            source, target = source_all[mask], target_all[mask]
            weight = self._edge_weights(
                state, source, target, relation[mask], upward=False
            )
            messages = self.down_message(state[source]) * weight[:, None]
            aggregate = state.new_zeros(state.shape)
            normalizer = state.new_zeros((state.size(0), 1))
            aggregate.index_add_(0, target, messages)
            normalizer.index_add_(0, target, weight[:, None])
            aggregate = aggregate / normalizer.clamp_min(1.0)
            candidate = state + self.down_update(torch.cat([state, aggregate], dim=-1))
            state = torch.where(normalizer > 0, candidate, state)
        return state

    def _prototype_context(
        self,
        normalized: Tensor,
        graph_edge_index: LongTensor,
        labels: LongTensor,
        graph_inverse: LongTensor,
        graph_confidence: Tensor,
        boundary: Tensor,
    ) -> Tensor:
        graph_count = int(graph_confidence.size(0))
        graph_mean = self._segment_mean(normalized, graph_inverse, graph_count)
        graph_pairs = torch.stack([graph_inverse, labels], dim=1)
        _, community_inverse = torch.unique(graph_pairs, dim=0, return_inverse=True)
        community_mean = self._segment_mean(
            normalized,
            community_inverse,
            int(community_inverse.max().item()) + 1,
        )
        own = community_mean[community_inverse]

        bridge_sum = normalized.new_zeros(normalized.shape)
        bridge_count = normalized.new_zeros((normalized.size(0), 1))
        if graph_edge_index.numel():
            source, target = graph_edge_index.long()
            valid = (
                (source >= 0)
                & (target >= 0)
                & (source < normalized.size(0))
                & (target < normalized.size(0))
                & (source != target)
                & (labels[source] != labels[target])
            )
            source, target = source[valid], target[valid]
            if source.numel():
                bridge_sum.index_add_(0, source, normalized[target])
                bridge_count.index_add_(0, source, normalized.new_ones((source.numel(), 1)))
                bridge_sum.index_add_(0, target, normalized[source])
                bridge_count.index_add_(0, target, normalized.new_ones((target.numel(), 1)))
        bridge = bridge_sum / bridge_count.clamp_min(1.0)
        bridge = torch.where(
            bridge_count > 0,
            bridge,
            graph_mean[graph_inverse],
        )

        preference = self.preference.to(normalized).expand(normalized.size(0), -1)
        confidence = graph_confidence[graph_inverse]
        controller_input = torch.cat(
            [normalized, boundary[:, None], confidence, preference], dim=1
        )
        logits = self.prototype_controller(controller_input)
        logits = logits.clone()
        logits[:, 1] = logits[:, 1] + 2.0 * (1.0 - boundary)
        logits[:, 2] = logits[:, 2] + 2.0 * boundary
        weights = torch.softmax(logits, dim=1)
        return (
            weights[:, 0:1] * normalized
            + weights[:, 1:2] * own
            + weights[:, 2:3] * bridge
        )

    def forward(
        self,
        node_feat: FloatTensor,
        cover: TreeCoverBatch,
        graph_edge_index: LongTensor,
        community_labels: Optional[LongTensor],
        batch_index: LongTensor,
    ) -> FloatTensor:
        node_count = int(node_feat.size(0))
        if node_count == 0:
            return node_feat
        labels = self._labels(community_labels, node_count, node_feat.device)
        normalized = self.input_norm(node_feat)
        (
            graph_inverse,
            graph_descriptor,
            graph_confidence,
            degree,
            boundary,
        ) = self.graph_context(
            graph_edge_index, labels, batch_index, node_feat.dtype
        )
        preference = self.preference.to(node_feat)

        tree_fused = torch.zeros_like(normalized)
        attention = normalized.new_full((node_count, self.num_trees), 1.0 / self.num_trees)
        if (
            self.tree_messages_enabled
            and not self.node_only_control
            and cover.edge_index.numel() > 0
        ):
            source, target = cover.edge_index.long()
            tree_fraction = cover.tree_id.to(dtype=node_feat.dtype)
            tree_fraction = tree_fraction / max(self.num_trees - 1, 1)
            same = (labels[source] == labels[target]).to(node_feat.dtype)
            boundary_mean = 0.5 * (boundary[source] + boundary[target])
            boundary_gap = (boundary[source] - boundary[target]).abs()
            relation = torch.cat(
                [
                    torch.stack([same, boundary_mean, boundary_gap, tree_fraction], dim=1),
                    preference.expand(source.numel(), -1),
                ],
                dim=1,
            )
            tree_states = torch.stack(
                [
                    self._sweep_tree(normalized, cover, tree_id, relation)
                    for tree_id in range(self.num_trees)
                ],
                dim=1,
            )
            if self.tree_attention_enabled:
                confidence = graph_confidence[graph_inverse]
                common = torch.cat(
                    [boundary[:, None], confidence, preference.expand(node_count, -1)],
                    dim=1,
                )
                attention_input = torch.cat(
                    [
                        normalized[:, None, :].expand(-1, self.num_trees, -1),
                        tree_states,
                        common[:, None, :].expand(-1, self.num_trees, -1),
                    ],
                    dim=2,
                )
                attention = torch.softmax(
                    self.tree_attention(attention_input).squeeze(-1), dim=1
                )
            tree_fused = (attention[:, :, None] * tree_states).sum(dim=1)

        if self.prototype_context_enabled and not self.node_only_control:
            prototype = self._prototype_context(
                normalized,
                graph_edge_index,
                labels,
                graph_inverse,
                graph_confidence,
                boundary,
            )
            gate_boundary = boundary[:, None]
            gate_confidence = graph_confidence[graph_inverse]
        else:
            prototype = torch.zeros_like(normalized)
            gate_boundary = torch.zeros((node_count, 1), device=node_feat.device, dtype=node_feat.dtype)
            gate_confidence = torch.zeros_like(gate_boundary)

        preference_nodes = preference.expand(node_count, -1)
        update_input = torch.cat(
            [normalized, tree_fused, prototype, preference_nodes], dim=1
        )
        delta = self.update_net(update_input)
        node_gate = self.node_gate(
            torch.cat(
                [
                    normalized,
                    tree_fused,
                    prototype,
                    gate_boundary,
                    gate_confidence,
                    preference_nodes,
                ],
                dim=1,
            )
        )
        if self.graph_gate is not None and not self.node_only_control:
            graph_preference = preference.expand(graph_descriptor.size(0), -1)
            graph_logits = self.graph_gate(
                torch.cat([graph_descriptor, graph_preference], dim=1)
            )
            graph_logits = graph_logits + 2.0 * (graph_confidence - 0.5)
            graph_gate = torch.sigmoid(graph_logits)[graph_inverse]
        else:
            graph_gate = torch.ones_like(node_gate)

        mix = torch.sigmoid(self.mix_logit) * self.scale * self.residual_progress
        combined_gate = node_gate * graph_gate
        output = node_feat + mix * combined_gate * self.dropout(delta)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / (
            math.log(float(self.num_trees)) if self.num_trees > 1 else 1.0
        )
        self.last_mean_gate = float(combined_gate.detach().mean().cpu())
        self.last_mean_graph_gate = float(graph_gate.detach().mean().cpu())
        self.last_mix = float(mix.detach().cpu())
        self.last_tree_attention_entropy = float(entropy.detach().mean().cpu())
        self.last_partition_confidence = float(graph_confidence.detach().mean().cpu())
        return output


class HCTFv2EquivariantCoordinateHead(nn.Module):
    """Produce translation-conserving coordinate forces from invariant scalars."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_trees: int,
        preference: Sequence[float] = (0.34, 0.33, 0.33),
        zero_init: bool = False,
        node_only_control: bool = False,
        tree_messages_enabled: bool = True,
        prototype_context_enabled: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.num_trees = int(num_trees)
        self.eps = float(eps)
        self.node_only_control = bool(node_only_control)
        self.tree_messages_enabled = bool(tree_messages_enabled)
        self.prototype_context_enabled = bool(prototype_context_enabled)
        self.register_buffer("preference", _normalized_preference(preference))
        self.pair_force = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.prototype_force = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.graph_trust = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.graph_trust[-1].weight)
        nn.init.constant_(self.graph_trust[-1].bias, 1.0)
        if zero_init:
            nn.init.zeros_(self.pair_force[-2].weight)
            nn.init.zeros_(self.pair_force[-2].bias)
            nn.init.zeros_(self.prototype_force[-2].weight)
            nn.init.zeros_(self.prototype_force[-2].bias)

    def set_preference(self, preference: Sequence[float]) -> None:
        self.preference.copy_(_normalized_preference(preference).to(self.preference))

    @staticmethod
    def _unit(vector: Tensor, eps: float) -> Tensor:
        return vector / vector.norm(dim=1, keepdim=True).clamp_min(eps)

    def forward(
        self,
        base_pos: FloatTensor,
        hidden: FloatTensor,
        enhanced: FloatTensor,
        cover: TreeCoverBatch,
        graph_edge_index: LongTensor,
        community_labels: Optional[LongTensor],
        batch_index: LongTensor,
    ) -> Tuple[Tensor, Tensor]:
        node_count = int(base_pos.size(0))
        labels = HCTFv2Block._labels(community_labels, node_count, base_pos.device)
        context_edge_index = graph_edge_index
        if self.node_only_control:
            labels = torch.zeros_like(labels)
            context_edge_index = graph_edge_index.new_empty((2, 0))
        (
            graph_inverse,
            graph_descriptor,
            confidence,
            _,
            boundary,
        ) = HCTFv2Block.graph_context(
            context_edge_index, labels, batch_index, base_pos.dtype
        )
        graph_count = int(graph_descriptor.size(0))
        if graph_count == 0:
            return torch.zeros_like(base_pos), base_pos.new_zeros(0)
        preference = self.preference.to(base_pos)
        delta = torch.zeros_like(base_pos)
        normalizer = base_pos.new_zeros((node_count, 1))

        if (
            cover.edge_index.numel()
            and not self.node_only_control
            and self.tree_messages_enabled
        ):
            mask = cover.orientation > 0
            source, target = cover.edge_index[:, mask].long()
            if source.numel():
                left, right = hidden[source], hidden[target]
                left_norm = left.norm(dim=1)
                right_norm = right.norm(dim=1)
                difference_norm = (left - right).norm(dim=1)
                cosine = F.cosine_similarity(left, right, dim=1, eps=self.eps)
                same = (labels[source] == labels[target]).to(base_pos.dtype)
                boundary_mean = 0.5 * (boundary[source] + boundary[target])
                boundary_gap = (boundary[source] - boundary[target]).abs()
                tree_fraction = cover.tree_id[mask].to(base_pos.dtype)
                tree_fraction = tree_fraction / max(self.num_trees - 1, 1)
                pair_confidence = confidence[graph_inverse[source], 0]
                pair_features = torch.cat(
                    [
                        torch.stack(
                            [
                                0.5 * (left_norm + right_norm),
                                (left_norm - right_norm).abs(),
                                difference_norm,
                                cosine,
                                same,
                                boundary_mean,
                                boundary_gap,
                                tree_fraction,
                                pair_confidence,
                            ],
                            dim=1,
                        ),
                        preference.expand(source.numel(), -1),
                    ],
                    dim=1,
                )
                strength = self.pair_force(pair_features)
                direction = self._unit(base_pos[target] - base_pos[source], self.eps)
                message = strength * direction
                delta.index_add_(0, source, message)
                delta.index_add_(0, target, -message)
                normalizer.index_add_(0, source, torch.ones_like(strength))
                normalizer.index_add_(0, target, torch.ones_like(strength))
        pair_delta = delta / normalizer.clamp_min(1.0)

        graph_mean = HCTFv2Block._segment_mean(base_pos, graph_inverse, graph_count)
        graph_pairs = torch.stack([graph_inverse, labels], dim=1)
        _, community_inverse = torch.unique(graph_pairs, dim=0, return_inverse=True)
        community_mean = HCTFv2Block._segment_mean(
            base_pos,
            community_inverse,
            int(community_inverse.max().item()) + 1,
        )
        own_vector = community_mean[community_inverse] - base_pos

        bridge_sum = torch.zeros_like(base_pos)
        bridge_count = base_pos.new_zeros((node_count, 1))
        if context_edge_index.numel():
            source, target = context_edge_index.long()
            cross = (
                (source >= 0)
                & (target >= 0)
                & (source < node_count)
                & (target < node_count)
                & (source != target)
                & (labels[source] != labels[target])
            )
            source, target = source[cross], target[cross]
            if source.numel():
                bridge_sum.index_add_(0, source, base_pos[target])
                bridge_count.index_add_(0, source, base_pos.new_ones((source.numel(), 1)))
                bridge_sum.index_add_(0, target, base_pos[source])
                bridge_count.index_add_(0, target, base_pos.new_ones((target.numel(), 1)))
        bridge_center = bridge_sum / bridge_count.clamp_min(1.0)
        bridge_center = torch.where(
            bridge_count > 0,
            bridge_center,
            graph_mean[graph_inverse],
        )
        bridge_vector = bridge_center - base_pos

        invariant_node_features = torch.cat(
            [
                hidden.norm(dim=1, keepdim=True),
                enhanced.norm(dim=1, keepdim=True),
                (enhanced - hidden).norm(dim=1, keepdim=True),
                boundary[:, None],
                confidence[graph_inverse],
                preference.expand(node_count, -1),
            ],
            dim=1,
        )
        if self.prototype_context_enabled:
            prototype_strength = self.prototype_force(invariant_node_features)
            own_force = (
                prototype_strength[:, 0:1]
                * (1.0 - boundary[:, None])
                * self._unit(own_vector, self.eps)
            )
            bridge_force = (
                prototype_strength[:, 1:2]
                * boundary[:, None]
                * self._unit(bridge_vector, self.eps)
            )
        else:
            own_force = torch.zeros_like(base_pos)
            bridge_force = torch.zeros_like(base_pos)
        raw_delta = pair_delta + own_force + bridge_force

        # Remove graph-wise translation exactly.  Rotation/reflection equivariance
        # follows because only invariant scalars multiply relative directions.
        raw_delta = raw_delta - HCTFv2Block._segment_mean(
            raw_delta, graph_inverse, graph_count
        )[graph_inverse]
        rms = HCTFv2Block._segment_mean(
            raw_delta.square().sum(dim=1, keepdim=True), graph_inverse, graph_count
        ).clamp_min(self.eps).sqrt()
        trust_input = torch.cat(
            [
                confidence,
                graph_descriptor[:, 0:1],
                rms,
                graph_descriptor[:, 4:5],
                graph_descriptor[:, 5:6],
                graph_descriptor[:, 7:8],
                preference.expand(graph_count, -1),
            ],
            dim=1,
        )
        trust = torch.sigmoid(
            self.graph_trust(trust_input) + 2.0 * (confidence - 0.5)
        ).squeeze(-1)
        return raw_delta, trust
