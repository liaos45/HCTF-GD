from __future__ import annotations

import torch
from torch import nn

from .crossing import edge_pair_crossing_scores, sample_hard_edge_pairs


def _graph_count(batch_index, fallback: int = 0) -> int:
    if batch_index.numel():
        return int(batch_index.max().item()) + 1
    return int(fallback)


def graph_community_confidence(
    edge_index,
    batch_index,
    community_labels,
    num_graphs: int | None = None,
):
    """Estimate how informative hard community labels are for each graph.

    The score is the excess fraction of intra-community edges above the random
    label-frequency baseline. It is bounded to [0, 1] and requires no spectral
    decomposition or additional learned parameters.
    """
    batch_index = batch_index.long()
    community_labels = community_labels.long()
    graph_count = _graph_count(batch_index, fallback=num_graphs or 0)
    dtype = torch.get_default_dtype()
    if edge_index.is_floating_point():
        dtype = edge_index.dtype
    confidence = torch.zeros(graph_count, dtype=dtype, device=batch_index.device)
    if graph_count == 0:
        return confidence

    src, dst = edge_index.long()
    valid = (src != dst) & (batch_index[src] == batch_index[dst])
    src, dst = src[valid], dst[valid]
    observed = confidence.clone()
    edge_count = confidence.clone()
    if src.numel():
        edge_graph = batch_index[src]
        edge_count.index_add_(
            0, edge_graph, torch.ones(src.numel(), dtype=dtype, device=src.device)
        )
        observed.index_add_(
            0,
            edge_graph,
            (community_labels[src] == community_labels[dst]).to(dtype),
        )
        observed = observed / edge_count.clamp_min(1.0)

    expected = confidence.clone()
    for graph_id in range(graph_count):
        labels = community_labels[batch_index == graph_id]
        if labels.numel() == 0:
            continue
        _, counts = labels.unique(return_counts=True)
        probabilities = counts.to(dtype) / float(labels.numel())
        expected[graph_id] = probabilities.square().sum()

    informative = (edge_count > 0) & (expected < 1.0 - 1e-6)
    excess = (observed - expected) / (1.0 - expected).clamp_min(1e-6)
    return torch.where(informative, excess.clamp(0.0, 1.0), confidence)


def soft_community_relation(
    src,
    dst,
    batch_index,
    community_labels,
    graph_confidence,
    boundary_ratio=None,
    dtype=None,
):
    """Turn hard same-community labels into confidence-aware pair relations."""
    if dtype is None:
        dtype = graph_confidence.dtype
    hard_relation = (community_labels[src] == community_labels[dst]).to(dtype)
    reliability = graph_confidence[batch_index[src].long()].to(dtype)
    if boundary_ratio is not None:
        boundary_ratio = boundary_ratio.clamp(0.0, 1.0)
        boundary_reliability = 1.0 - 0.5 * (
            boundary_ratio[src] + boundary_ratio[dst]
        )
        reliability = reliability * boundary_reliability.clamp(0.0, 1.0)
    return 0.5 + (hard_relation - 0.5) * reliability


def topology_consensus_relation(
    src,
    dst,
    batch_index,
    community_labels,
    graph_confidence,
    node_features,
    relation_net,
    consensus_scale: float = 1.0,
):
    """Fuse a hard partition with learned local-topology evidence.

    The hard partition remains dominant when its graph-level confidence is
    high. On boundary nodes or weak partitions, a zero-initialized relation
    network can learn a bounded correction from topology descriptors. This
    keeps the initial behavior identical to the soft hard-label relation.
    """
    boundary_ratio = node_features[:, 2] if node_features.size(1) > 2 else None
    base = soft_community_relation(
        src=src,
        dst=dst,
        batch_index=batch_index,
        community_labels=community_labels,
        graph_confidence=graph_confidence,
        boundary_ratio=boundary_ratio,
        dtype=node_features.dtype,
    )
    topology = node_features[:, : min(node_features.size(1), 4)]
    if topology.size(1) < 4:
        topology = torch.nn.functional.pad(topology, (0, 4 - topology.size(1)))
    pair_graph = batch_index[src].long()
    if boundary_ratio is None:
        pair_boundary = base.new_zeros(base.size(0), 1)
    else:
        pair_boundary = 0.5 * (
            boundary_ratio[src, None] + boundary_ratio[dst, None]
        )
    relation_features = torch.cat(
        [
            (topology[src] - topology[dst]).abs(),
            0.5 * (topology[src] + topology[dst]),
            graph_confidence[pair_graph, None].to(topology.dtype),
            pair_boundary.to(topology.dtype),
        ],
        dim=1,
    )
    evidence = torch.tanh(relation_net(relation_features)).squeeze(1)
    reliability = graph_confidence[pair_graph].to(base.dtype)
    reliability = reliability * (1.0 - pair_boundary.squeeze(1)).clamp(0.0, 1.0)
    correction = float(consensus_scale) * (1.0 - reliability) * evidence
    base_logit = torch.logit(base.clamp(0.025, 0.975))
    return torch.sigmoid(base_logit + correction)


def sample_preferences(
    num_graphs: int,
    num_objectives: int,
    alpha: float,
    device,
    dtype,
):
    concentration = torch.full(
        (int(num_objectives),),
        max(float(alpha), 1e-3),
        device=device,
        dtype=dtype,
    )
    return torch.distributions.Dirichlet(concentration).sample((int(num_graphs),))


class PreferenceEncoder(nn.Module):
    def __init__(self, num_objectives: int, hidden_dim: int):
        super().__init__()
        self.num_objectives = int(num_objectives)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_objectives, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 3),
        )

    def forward(self, preference):
        preference = preference.clamp_min(0.0)
        preference = preference / preference.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        embedding, gamma, beta = self.net(preference).chunk(3, dim=-1)
        return embedding, 0.5 * torch.tanh(gamma), beta


class EquivariantLayoutField(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 4,
        hidden_dim: int = 64,
        num_objectives: int = 3,
        use_soft_community: bool = False,
        use_relation_consensus: bool = False,
        relation_consensus_scale: float = 1.0,
        use_graph_preference_calibration: bool = False,
        preference_calibration_scale: float = 0.75,
        use_edge_pair_field: bool = False,
        edge_pair_scale: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_objectives = int(num_objectives)
        self.use_soft_community = bool(use_soft_community)
        self.use_relation_consensus = bool(use_relation_consensus)
        self.relation_consensus_scale = float(relation_consensus_scale)
        self.use_edge_pair_field = bool(use_edge_pair_field)
        self.edge_pair_scale = float(edge_pair_scale)
        self.use_graph_preference_calibration = bool(
            use_graph_preference_calibration
        )
        self.preference_calibration_scale = float(preference_calibration_scale)
        self.node_encoder = nn.Sequential(
            nn.Linear(int(node_feature_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.preference_encoder = PreferenceEncoder(num_objectives, self.hidden_dim)
        if self.use_graph_preference_calibration:
            calibration_hidden = max(self.hidden_dim // 2, 8)
            self.preference_calibrator = nn.Sequential(
                nn.Linear(6, calibration_hidden),
                nn.SiLU(),
                nn.Linear(calibration_hidden, self.num_objectives),
            )
            nn.init.zeros_(self.preference_calibrator[-1].weight)
            nn.init.zeros_(self.preference_calibrator[-1].bias)
        else:
            self.preference_calibrator = None
        if self.use_relation_consensus:
            relation_hidden = max(self.hidden_dim // 2, 8)
            self.relation_consensus_net = nn.Sequential(
                nn.Linear(10, relation_hidden),
                nn.SiLU(),
                nn.Linear(relation_hidden, 1),
            )
            nn.init.zeros_(self.relation_consensus_net[-1].weight)
            nn.init.zeros_(self.relation_consensus_net[-1].bias)
        else:
            self.relation_consensus_net = None
        self.coefficient_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh(),
        )
        if self.use_edge_pair_field:
            edge_pair_input_dim = self.hidden_dim * 2 + self.num_objectives + 5
            self.edge_pair_coefficient_net = nn.Sequential(
                nn.Linear(edge_pair_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.edge_pair_coefficient_net[-2].weight)
            nn.init.zeros_(self.edge_pair_coefficient_net[-2].bias)
        else:
            self.edge_pair_coefficient_net = None
        self.last_effective_preference = None
        self.last_community_confidence = None
        self.last_relation_uncertainty = None
        self.last_edge_pair_count = 0

    def _community_relation(
        self,
        src,
        dst,
        batch_index,
        community_labels,
        graph_confidence,
        node_features,
        dtype,
    ):
        if self.relation_consensus_net is not None:
            return topology_consensus_relation(
                src=src,
                dst=dst,
                batch_index=batch_index,
                community_labels=community_labels,
                graph_confidence=graph_confidence,
                node_features=node_features,
                relation_net=self.relation_consensus_net,
                consensus_scale=self.relation_consensus_scale,
            ).to(dtype)
        if self.use_soft_community:
            boundary_ratio = node_features[:, 2] if node_features.size(1) > 2 else None
            return soft_community_relation(
                src=src,
                dst=dst,
                batch_index=batch_index,
                community_labels=community_labels,
                graph_confidence=graph_confidence,
                boundary_ratio=boundary_ratio,
                dtype=dtype,
            )
        return (community_labels[src] == community_labels[dst]).to(dtype)

    def _edge_pair_delta(
        self,
        pos,
        node_embedding,
        node_features,
        batch_index,
        community_labels,
        graph_confidence,
        effective_preference,
        edge_pair_index,
    ):
        if self.edge_pair_coefficient_net is None or edge_pair_index is None:
            self.last_edge_pair_count = 0
            return torch.zeros_like(pos)
        first_edges, second_edges = edge_pair_index
        if first_edges.numel() == 0:
            self.last_edge_pair_count = 0
            return torch.zeros_like(pos)

        self.last_edge_pair_count = int(first_edges.size(0))

        first_edges = first_edges.long()
        second_edges = second_edges.long()
        a, b = first_edges[:, 0], first_edges[:, 1]
        c, d = second_edges[:, 0], second_edges[:, 1]
        first_embedding = 0.5 * (node_embedding[a] + node_embedding[b])
        second_embedding = 0.5 * (node_embedding[c] + node_embedding[d])

        first_vector = pos[b] - pos[a]
        second_vector = pos[d] - pos[c]
        first_length = first_vector.norm(dim=1, keepdim=True).clamp_min(1e-8)
        second_length = second_vector.norm(dim=1, keepdim=True).clamp_min(1e-8)
        pair_scale = 0.5 * (first_length + second_length)
        midpoint_distance = (
            0.5 * (pos[a] + pos[b] - pos[c] - pos[d])
        ).norm(dim=1, keepdim=True) / pair_scale
        cosine = (
            first_vector * second_vector
        ).sum(dim=1, keepdim=True) / (first_length * second_length)
        length_imbalance = (first_length - second_length).abs() / (
            first_length + second_length
        )
        crossing_score = edge_pair_crossing_scores(
            pos,
            first_edges,
            second_edges,
        )[:, None]
        cross_relations = torch.stack(
            [
                self._community_relation(
                    left,
                    right,
                    batch_index,
                    community_labels,
                    graph_confidence,
                    node_features,
                    pos.dtype,
                )
                for left, right in ((a, c), (a, d), (b, c), (b, d))
            ],
            dim=1,
        ).mean(dim=1, keepdim=True)
        pair_graph = batch_index[a].long()
        pair_features = torch.cat(
            [
                0.5 * (first_embedding + second_embedding),
                (first_embedding - second_embedding).abs(),
                midpoint_distance,
                cosine.abs(),
                length_imbalance,
                crossing_score,
                cross_relations,
                effective_preference[pair_graph],
            ],
            dim=1,
        )
        coefficient = self.edge_pair_coefficient_net(pair_features)
        strength = coefficient * crossing_score

        edge_delta = torch.zeros_like(pos)
        edge_count = pos.new_zeros(pos.size(0))
        for left, right in ((a, c), (a, d), (b, c), (b, d)):
            relative = pos[left] - pos[right]
            unit = relative / relative.norm(dim=1, keepdim=True).clamp_min(1e-8)
            message = -strength * unit
            edge_delta.index_add_(0, left, message)
            edge_delta.index_add_(0, right, -message)
            edge_count.index_add_(0, left, torch.ones_like(left, dtype=pos.dtype))
            edge_count.index_add_(0, right, torch.ones_like(right, dtype=pos.dtype))
        return edge_delta / edge_count.clamp_min(1.0)[:, None]

    def _effective_preference(
        self,
        preference,
        node_features,
        batch_index,
        graph_confidence,
    ):
        preference = preference.clamp_min(0.0)
        preference = preference / preference.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.preference_calibrator is None:
            return preference

        graph_count = int(preference.size(0))
        topology = node_features[:, : min(node_features.size(1), 4)]
        if topology.size(1) < 4:
            topology = torch.nn.functional.pad(topology, (0, 4 - topology.size(1)))
        graph_sum = torch.zeros(
            (graph_count, 4), dtype=node_features.dtype, device=node_features.device
        )
        graph_sum.index_add_(0, batch_index, topology)
        graph_size = torch.bincount(batch_index, minlength=graph_count).to(
            node_features.dtype
        )
        graph_mean = graph_sum / graph_size.clamp_min(1.0)[:, None]
        size_feature = torch.log1p(graph_size).div(5.0).clamp(max=2.0)[:, None]
        context = torch.cat(
            [
                graph_mean,
                graph_confidence.to(node_features.dtype)[:, None],
                size_feature,
            ],
            dim=1,
        )
        adjustment = self.preference_calibration_scale * torch.tanh(
            self.preference_calibrator(context)
        )
        effective = preference * torch.exp(adjustment)
        return effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        pos,
        node_features,
        edge_index,
        batch_index,
        community_labels,
        preference,
        step_size: float,
        edge_pair_index=None,
    ):
        edge_index = edge_index.long()
        batch_index = batch_index.long()
        src, dst = edge_index
        valid = (src != dst) & (batch_index[src] == batch_index[dst])
        src, dst = src[valid], dst[valid]
        if src.numel() == 0:
            return pos

        graph_confidence = graph_community_confidence(
            edge_index=edge_index,
            batch_index=batch_index,
            community_labels=community_labels,
            num_graphs=int(preference.size(0)),
        ).to(dtype=pos.dtype)
        effective_preference = self._effective_preference(
            preference=preference,
            node_features=node_features,
            batch_index=batch_index,
            graph_confidence=graph_confidence,
        )
        self.last_effective_preference = effective_preference.detach()
        self.last_community_confidence = graph_confidence.detach()

        node_embedding = self.node_encoder(node_features)
        pref_embedding, gamma, beta = self.preference_encoder(effective_preference)
        node_pref = pref_embedding[batch_index]
        node_embedding = (1.0 + gamma[batch_index]) * node_embedding + beta[batch_index]

        relative = pos[src] - pos[dst]
        distance_sq = relative.square().sum(dim=1, keepdim=True)
        same_community = self._community_relation(
            src=src,
            dst=dst,
            batch_index=batch_index,
            community_labels=community_labels,
            graph_confidence=graph_confidence,
            node_features=node_features,
            dtype=pos.dtype,
        )[:, None]
        relation_uncertainty = 4.0 * same_community.squeeze(1) * (
            1.0 - same_community.squeeze(1)
        )
        uncertainty_sum = pos.new_zeros(int(effective_preference.size(0)))
        uncertainty_count = pos.new_zeros(int(effective_preference.size(0)))
        pair_graph = batch_index[src]
        uncertainty_sum.index_add_(0, pair_graph, relation_uncertainty)
        uncertainty_count.index_add_(
            0, pair_graph, torch.ones_like(relation_uncertainty)
        )
        self.last_relation_uncertainty = (
            uncertainty_sum / uncertainty_count.clamp_min(1.0)
        ).detach()
        pair_features = torch.cat(
            [
                0.5 * (node_embedding[src] + node_embedding[dst]),
                (node_embedding[src] - node_embedding[dst]).abs(),
                distance_sq,
                same_community,
                (node_pref[src] * node_pref[dst]).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        coefficient = self.coefficient_net(pair_features)
        messages = coefficient * relative
        delta = torch.zeros_like(pos)
        delta.index_add_(0, src, messages)
        degree = torch.zeros(pos.size(0), dtype=pos.dtype, device=pos.device)
        degree.index_add_(0, src, torch.ones_like(src, dtype=pos.dtype))
        delta = delta / degree.clamp_min(1.0)[:, None]
        edge_pair_delta = self._edge_pair_delta(
            pos=pos,
            node_embedding=node_embedding,
            node_features=node_features,
            batch_index=batch_index,
            community_labels=community_labels,
            graph_confidence=graph_confidence,
            effective_preference=effective_preference,
            edge_pair_index=edge_pair_index,
        )
        delta = delta + self.edge_pair_scale * edge_pair_delta

        graph_count = int(effective_preference.size(0))
        graph_sum = torch.zeros(
            (graph_count, delta.size(1)), dtype=delta.dtype, device=delta.device
        )
        graph_sum.index_add_(0, batch_index, delta)
        graph_size = torch.bincount(batch_index, minlength=graph_count).to(delta.dtype)
        graph_mean = graph_sum / graph_size.clamp_min(1.0)[:, None]
        delta = delta - graph_mean[batch_index]
        return pos - float(step_size) * delta


class ParetoFieldRefiner(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 4,
        hidden_dim: int = 64,
        num_objectives: int = 3,
        train_steps: int = 1,
        inference_steps: int = 4,
        step_size: float = 0.10,
        max_sampled_pairs: int = 4096,
        adaptive_tolerance: float = 0.0,
        min_inference_steps: int = 1,
        use_soft_community: bool = False,
        use_relation_consensus: bool = False,
        relation_consensus_scale: float = 1.0,
        use_graph_preference_calibration: bool = False,
        preference_calibration_scale: float = 0.75,
        use_edge_pair_field: bool = False,
        edge_pair_max_pairs: int = 384,
        edge_pair_oversample_factor: int = 4,
        edge_pair_temperature: float = 8.0,
        edge_pair_scale: float = 0.15,
    ):
        super().__init__()
        self.field = EquivariantLayoutField(
            node_feature_dim,
            hidden_dim,
            num_objectives,
            use_soft_community=use_soft_community,
            use_relation_consensus=use_relation_consensus,
            relation_consensus_scale=relation_consensus_scale,
            use_graph_preference_calibration=use_graph_preference_calibration,
            preference_calibration_scale=preference_calibration_scale,
            use_edge_pair_field=use_edge_pair_field,
            edge_pair_scale=edge_pair_scale,
        )
        self.train_steps = int(train_steps)
        self.inference_steps = int(inference_steps)
        self.step_size = float(step_size)
        self.max_sampled_pairs = int(max_sampled_pairs)
        self.adaptive_tolerance = float(adaptive_tolerance)
        self.min_inference_steps = max(int(min_inference_steps), 1)
        self.use_edge_pair_field = bool(use_edge_pair_field)
        self.edge_pair_max_pairs = int(edge_pair_max_pairs)
        self.edge_pair_oversample_factor = int(edge_pair_oversample_factor)
        self.edge_pair_temperature = float(edge_pair_temperature)
        self.last_inference_steps = 0
        self.last_effective_preference = None
        self.last_community_confidence = None
        self.last_relation_uncertainty = None
        self.last_edge_pair_count = 0

    def _sparse_pairs(self, graph_edges, long_range_edges):
        pairs = [graph_edges.long()]
        if long_range_edges is not None and long_range_edges.numel() > 0:
            long_range_edges = long_range_edges.long()
            if self.max_sampled_pairs > 0 and long_range_edges.size(1) > self.max_sampled_pairs:
                sample = torch.randperm(long_range_edges.size(1), device=long_range_edges.device)
                long_range_edges = long_range_edges[:, sample[: self.max_sampled_pairs]]
            pairs.append(long_range_edges)
        directed = torch.cat(pairs, dim=1)
        return torch.cat([directed, directed.flip(0)], dim=1)

    def forward(
        self,
        pos,
        node_features,
        graph_edges,
        long_range_edges,
        batch_index,
        community_labels,
        preference,
    ):
        pair_index = self._sparse_pairs(graph_edges, long_range_edges)
        steps = self.train_steps if self.training else self.inference_steps
        output = pos
        graph_count = int(preference.size(0))
        active_graphs = torch.ones(graph_count, dtype=torch.bool, device=pos.device)
        self.last_inference_steps = 0
        for step_id in range(max(int(steps), 0)):
            edge_pair_index = None
            if self.use_edge_pair_field:
                edge_pair_index = sample_hard_edge_pairs(
                    pos=output.detach(),
                    edge_index=graph_edges.long(),
                    batch_index=batch_index.long(),
                    max_pairs=self.edge_pair_max_pairs,
                    oversample_factor=self.edge_pair_oversample_factor,
                    temperature=self.edge_pair_temperature,
                    ratios=(0.70, 0.0, 0.20, 0.10),
                )
                self.last_edge_pair_count = int(edge_pair_index[0].size(0))
            candidate = self.field(
                output,
                node_features,
                pair_index,
                batch_index,
                community_labels,
                preference,
                self.step_size,
                edge_pair_index=edge_pair_index,
            )
            self.last_effective_preference = self.field.last_effective_preference
            self.last_community_confidence = self.field.last_community_confidence
            self.last_relation_uncertainty = self.field.last_relation_uncertainty
            self.last_edge_pair_count = self.field.last_edge_pair_count or self.last_edge_pair_count
            if not self.training and self.adaptive_tolerance > 0:
                candidate = torch.where(
                    active_graphs[batch_index.long(), None], candidate, output
                )
                change = (candidate - output).square().sum(dim=1)
                graph_sum = torch.zeros(
                    (graph_count, pos.size(1)), dtype=pos.dtype, device=pos.device
                )
                graph_sum.index_add_(0, batch_index.long(), output)
                graph_size = torch.bincount(
                    batch_index.long(), minlength=graph_count
                ).to(pos.dtype)
                center = graph_sum / graph_size.clamp_min(1.0)[:, None]
                scale = (output - center[batch_index.long()]).square().sum(dim=1)
                change_sum = torch.zeros(graph_count, dtype=pos.dtype, device=pos.device)
                scale_sum = torch.zeros_like(change_sum)
                change_sum.index_add_(0, batch_index.long(), change)
                scale_sum.index_add_(0, batch_index.long(), scale)
                relative_update = torch.sqrt(
                    change_sum / graph_size.clamp_min(1.0)
                ) / torch.sqrt(scale_sum / graph_size.clamp_min(1.0)).clamp_min(1e-8)
                if step_id + 1 >= self.min_inference_steps:
                    active_graphs = active_graphs & (
                        relative_update > self.adaptive_tolerance
                    )
            output = candidate
            self.last_inference_steps = step_id + 1
            if (
                not self.training
                and self.adaptive_tolerance > 0
                and step_id + 1 >= self.min_inference_steps
                and not bool(active_graphs.any())
            ):
                break
        return output
