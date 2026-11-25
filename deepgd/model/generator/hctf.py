from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor, nn


@dataclass(frozen=True)
class TreeCoverBatch:
    edge_index: LongTensor
    orientation: FloatTensor
    tree_id: LongTensor
    # Depth of the child endpoint for each directed copy of a tree edge.  Both
    # parent->child and child->parent copies carry the same positive depth.
    # Optional keeps checkpoints/tests that construct the legacy three-field
    # batch source-compatible.
    depth: Optional[LongTensor] = None


class CoverageAwareTreeCover:
    """Build a small deterministic spanning-forest cover for each graph in a batch.

    The cover is topology-only and cached on CPU. Each selected undirected tree edge is
    emitted in both directions. ``orientation`` is +1 for parent-to-child flow and -1
    for child-to-parent flow.
    """

    def __init__(
        self,
        num_trees: int = 3,
        max_cache_size: int = 4096,
        community_first: bool = False,
    ):
        if num_trees < 1:
            raise ValueError("num_trees must be at least 1")
        if max_cache_size < 0:
            raise ValueError("max_cache_size cannot be negative")
        self.num_trees = int(num_trees)
        self.max_cache_size = int(max_cache_size)
        self.community_first = bool(community_first)
        self._cache: OrderedDict[Tuple, Tuple[Tuple[int, ...], ...]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _wl_colors(num_nodes: int, adjacency: Sequence[Sequence[int]]) -> List[int]:
        colors = [len(adjacency[node]) for node in range(num_nodes)]
        for _ in range(2):
            signatures = [
                (colors[node], tuple(sorted(colors[neighbor] for neighbor in adjacency[node])))
                for node in range(num_nodes)
            ]
            palette = {signature: rank for rank, signature in enumerate(sorted(set(signatures)))}
            colors = [palette[signature] for signature in signatures]
        return colors

    @staticmethod
    def _find(parent: List[int], node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    @classmethod
    def _union(cls, parent: List[int], rank: List[int], left: int, right: int) -> bool:
        root_left = cls._find(parent, left)
        root_right = cls._find(parent, right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    def _build_local_cover(
        self,
        num_nodes: int,
        undirected_edges: Tuple[Tuple[int, int], ...],
        community_labels: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Tuple[int, ...], ...]:
        if self.community_first and (
            community_labels is None or len(community_labels) != num_nodes
        ):
            raise ValueError("community-first cover requires one label per node")
        key = (
            num_nodes, self.num_trees, undirected_edges,
            community_labels if self.community_first else None,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return cached

        self.cache_misses += 1
        adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
        for left, right in undirected_edges:
            adjacency[left].append(right)
            adjacency[right].append(left)
        for neighbors in adjacency:
            neighbors.sort()

        degree = [len(neighbors) for neighbors in adjacency]
        colors = self._wl_colors(num_nodes, adjacency)
        coverage: Dict[Tuple[int, int], int] = defaultdict(int)
        all_src: List[int] = []
        all_dst: List[int] = []
        all_orientation: List[int] = []
        all_tree_id: List[int] = []
        all_depth: List[int] = []

        for tree_id in range(self.num_trees):
            def edge_priority(edge: Tuple[int, int]) -> Tuple[int, ...]:
                left, right = edge
                degree_pair = sorted((degree[left], degree[right]), reverse=True)
                color_pair = sorted((colors[left], colors[right]), reverse=True)
                # The endpoint term only resolves exact structural symmetries.
                tie_break = ((left + 1) * 2654435761 ^ (right + 1) * 2246822519 ^ tree_id) & 0xFFFFFFFF
                return (
                    int(community_labels[left] != community_labels[right])
                    if self.community_first else 0,
                    coverage[edge],
                    -degree_pair[0],
                    -degree_pair[1],
                    -color_pair[0],
                    -color_pair[1],
                    tie_break,
                )

            parent = list(range(num_nodes))
            rank = [0] * num_nodes
            selected: List[Tuple[int, int]] = []
            for left, right in sorted(undirected_edges, key=edge_priority):
                if self._union(parent, rank, left, right):
                    selected.append((left, right))
                    coverage[(left, right)] += 1

            tree_adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
            for left, right in selected:
                tree_adjacency[left].append(right)
                tree_adjacency[right].append(left)
            for neighbors in tree_adjacency:
                neighbors.sort(key=lambda node: (-degree[node], -colors[node], node))

            visited = [False] * num_nodes
            root_order = sorted(
                range(num_nodes),
                key=lambda node: (-degree[node], -colors[node], node),
            )
            for root in root_order:
                if visited[root]:
                    continue
                visited[root] = True
                node_depth = {root: 0}
                queue = deque([root])
                while queue:
                    current = queue.popleft()
                    for child in tree_adjacency[current]:
                        if visited[child]:
                            continue
                        visited[child] = True
                        child_depth = node_depth[current] + 1
                        node_depth[child] = child_depth
                        queue.append(child)
                        all_src.extend((current, child))
                        all_dst.extend((child, current))
                        all_orientation.extend((1, -1))
                        all_tree_id.extend((tree_id, tree_id))
                        all_depth.extend((child_depth, child_depth))

        result = (
            tuple(all_src),
            tuple(all_dst),
            tuple(all_orientation),
            tuple(all_tree_id),
            tuple(all_depth),
        )
        if self.max_cache_size > 0:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
        return result

    def build(
        self,
        graph_edge_index: LongTensor,
        batch_index: LongTensor,
        device: Optional[torch.device] = None,
        community_labels: Optional[LongTensor] = None,
    ) -> TreeCoverBatch:
        device = device or batch_index.device
        num_nodes = int(batch_index.numel())
        if num_nodes == 0:
            empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            return TreeCoverBatch(
                edge_index=empty_edges,
                orientation=torch.empty(0, dtype=torch.float32, device=device),
                tree_id=torch.empty(0, dtype=torch.long, device=device),
                depth=torch.empty(0, dtype=torch.long, device=device),
            )

        batch_cpu = batch_index.detach().to(device="cpu", dtype=torch.long)
        edge_cpu = graph_edge_index.detach().to(device="cpu", dtype=torch.long)
        if self.community_first:
            if community_labels is None or int(community_labels.numel()) != num_nodes:
                raise ValueError("community-first cover requires one label per node")
            community_cpu = community_labels.detach().to(device="cpu", dtype=torch.long)
        else:
            community_cpu = None
        graph_nodes: Dict[int, List[int]] = defaultdict(list)
        for node, graph_id in enumerate(batch_cpu.tolist()):
            graph_nodes[int(graph_id)].append(node)

        graph_edges: Dict[int, set[Tuple[int, int]]] = defaultdict(set)
        if edge_cpu.numel() > 0:
            for source, target in edge_cpu.t().tolist():
                if source == target or source < 0 or target < 0:
                    continue
                if source >= num_nodes or target >= num_nodes:
                    continue
                graph_id = int(batch_cpu[source])
                if graph_id != int(batch_cpu[target]):
                    continue
                left, right = sorted((int(source), int(target)))
                graph_edges[graph_id].add((left, right))

        all_src: List[int] = []
        all_dst: List[int] = []
        all_orientation: List[int] = []
        all_tree_id: List[int] = []
        all_depth: List[int] = []
        for graph_id in sorted(graph_nodes):
            global_nodes = graph_nodes[graph_id]
            global_to_local = {node: local for local, node in enumerate(global_nodes)}
            local_edges = tuple(
                sorted(
                    (global_to_local[left], global_to_local[right])
                    for left, right in graph_edges.get(graph_id, set())
                )
            )
            src, dst, orientation, tree_id, depth = self._build_local_cover(
                len(global_nodes),
                local_edges,
                (tuple(int(community_cpu[node]) for node in global_nodes)
                 if community_cpu is not None else None),
            )
            all_src.extend(global_nodes[index] for index in src)
            all_dst.extend(global_nodes[index] for index in dst)
            all_orientation.extend(orientation)
            all_tree_id.extend(tree_id)
            all_depth.extend(depth)

        if all_src:
            cover_edge_index = torch.tensor(
                [all_src, all_dst],
                dtype=torch.long,
                device=device,
            )
        else:
            cover_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return TreeCoverBatch(
            edge_index=cover_edge_index,
            orientation=torch.tensor(all_orientation, dtype=torch.float32, device=device),
            tree_id=torch.tensor(all_tree_id, dtype=torch.long, device=device),
            depth=torch.tensor(all_depth, dtype=torch.long, device=device),
        )


class HCTFBlock(nn.Module):
    """Shared hierarchical community-conditioned tree-flow update."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_trees: int,
        dropout: float = 0.05,
        scale: float = 0.20,
        graph_gate_enabled: bool = False,
        zero_init: bool = False,
        tree_messages_enabled: bool = True,
        direction_conditioning_enabled: bool = True,
        relation_conditioning_enabled: bool = True,
        prototype_context_enabled: bool = True,
        node_only_control: bool = False,
        adaptive_structure_gate: bool = False,
        structure_gate_init: float = 0.05,
        mix_init: Optional[float] = None,
        learn_direction_balance: bool = False,
        unsaturated_degree_mean: bool = False,
        centered_prototype_context: bool = False,
        prototype_graph_mean_scale: float = 1.0,
        independent_prototype_gate: bool = False,
        route_graph_mean_to_node: bool = False,
        relative_tree_messages: bool = False,
        node_relative_prototype: bool = False,
        depth_scheduled_tree_flow: bool = False,
        tree_identity_agnostic: bool = False,
        tree_coverage_relation: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_trees = int(num_trees)
        self.scale = float(scale)
        self.graph_gate_enabled = bool(graph_gate_enabled)
        self.tree_messages_enabled = bool(tree_messages_enabled)
        self.direction_conditioning_enabled = bool(direction_conditioning_enabled)
        self.relation_conditioning_enabled = bool(relation_conditioning_enabled)
        self.prototype_context_enabled = bool(prototype_context_enabled)
        self.node_only_control = bool(node_only_control)
        self.adaptive_structure_gate = bool(adaptive_structure_gate)
        if not 0.0 < float(structure_gate_init) < 1.0:
            raise ValueError("structure_gate_init must be strictly between zero and one")
        self.structure_gate_init = float(structure_gate_init)
        self.learn_direction_balance = bool(learn_direction_balance)
        self.unsaturated_degree_mean = bool(unsaturated_degree_mean)
        self.centered_prototype_context = bool(centered_prototype_context)
        if not 0.0 <= float(prototype_graph_mean_scale) <= 1.0:
            raise ValueError("prototype_graph_mean_scale must be in [0, 1]")
        self.prototype_graph_mean_scale = (
            0.0 if self.centered_prototype_context
            else float(prototype_graph_mean_scale)
        )
        self.independent_prototype_gate = bool(independent_prototype_gate)
        self.route_graph_mean_to_node = bool(route_graph_mean_to_node)
        self.relative_tree_messages = bool(relative_tree_messages)
        self.node_relative_prototype = bool(node_relative_prototype)
        self.depth_scheduled_tree_flow = bool(depth_scheduled_tree_flow)
        self.tree_identity_agnostic = bool(tree_identity_agnostic)
        self.tree_coverage_relation = bool(tree_coverage_relation)
        if self.tree_identity_agnostic and self.tree_coverage_relation:
            raise ValueError("choose identity-agnostic or coverage relation, not both")
        if self.node_relative_prototype and (
            self.centered_prototype_context
            or self.route_graph_mean_to_node
            or self.prototype_graph_mean_scale != 1.0
        ):
            raise ValueError("node_relative_prototype requires unchanged prototype routing")
        if self.route_graph_mean_to_node and not self.adaptive_structure_gate:
            raise ValueError("route_graph_mean_to_node requires adaptive_structure_gate")
        # A single shared preference for the two forest-flow orientations.
        # Zero gives unit weight to both directions, exactly the legacy path.
        self.direction_balance_logit = (
            nn.Parameter(torch.zeros(())) if self.learn_direction_balance else None
        )
        self.structure_gate_override: Optional[float] = None
        # Evaluation-only same-weight counterfactual.  ``None`` retains the
        # implemented bidirectional forest aggregation; the two named modes
        # keep only one orientation without changing learned parameters.
        self.tree_flow_override: Optional[str] = None
        # Same-checkpoint inference audit only; absent means the trained path.
        self.tree_message_scale_override: Optional[float] = None
        self.tree_community_coherence_override: Optional[float] = None

        self.input_norm = nn.LayerNorm(feature_dim)
        self.top_down_value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.bottom_up_value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.direction_score = nn.Linear(feature_dim, 1, bias=False)
        self.direction_relation = nn.Linear(4, 1, bias=False)
        self.relation_gate = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update_net = nn.Sequential(
            nn.Linear(3 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.node_update_net = None
        self.structure_gate = None
        if self.adaptive_structure_gate:
            # The node path is a genuine nested baseline.  The structural path
            # is additive and can be closed without changing parameter count.
            self.node_update_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
            )
            self.structure_gate = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        self.adaptive_gate = nn.Sequential(
            nn.Linear(3 * feature_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.graph_gate = None
        if self.graph_gate_enabled:
            self.graph_gate = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        self.prototype_gate = None
        if self.independent_prototype_gate:
            # Add this after all legacy modules so a matched random seed still
            # initializes every existing weight identically. Restore the CPU
            # RNG as well: this zero-initialized layer must not change later
            # data-loader shuffles in a paired experiment.
            rng_state = torch.random.get_rng_state()
            self.prototype_gate = nn.Linear(feature_dim + 2, 1)
            torch.random.set_rng_state(rng_state)
        self.dropout = nn.Dropout(dropout)
        if mix_init is None:
            mix_logit = -3.0
        else:
            if self.scale <= 0.0 or not 0.0 < float(mix_init) < self.scale:
                raise ValueError("mix_init must lie strictly between zero and scale")
            mix_fraction = float(mix_init) / self.scale
            mix_logit = math.log(mix_fraction / (1.0 - mix_fraction))
        self.mix_logit = nn.Parameter(torch.tensor(mix_logit))
        self.last_mean_gate = 0.0
        self.last_mean_graph_gate = 1.0
        self.last_mix = 0.0
        self.last_mean_structure_gate = 0.0
        self.last_min_structure_gate = 0.0
        self.last_max_structure_gate = 0.0
        self.last_structure_gate_tensor = None
        self.last_mean_prototype_gate = 1.0
        self.last_min_prototype_gate = 1.0
        self.last_max_prototype_gate = 1.0
        self.last_prototype_gate_tensor = None

        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)
        if self.graph_gate is not None:
            nn.init.zeros_(self.graph_gate[-2].weight)
            nn.init.constant_(self.graph_gate[-2].bias, -1.0)
        if self.structure_gate is not None:
            nn.init.zeros_(self.structure_gate[-2].weight)
            initial_logit = math.log(self.structure_gate_init / (1.0 - self.structure_gate_init))
            nn.init.constant_(self.structure_gate[-2].bias, initial_logit)
        if self.prototype_gate is not None:
            nn.init.zeros_(self.prototype_gate.weight)
            nn.init.zeros_(self.prototype_gate.bias)
        if zero_init:
            nn.init.zeros_(self.update_net[-1].weight)
            nn.init.zeros_(self.update_net[-1].bias)
            if self.node_update_net is not None:
                nn.init.zeros_(self.node_update_net[-1].weight)
                nn.init.zeros_(self.node_update_net[-1].bias)

    def set_structure_gate_override(self, value: Optional[float]) -> None:
        """Force the structured branch gate for matched counterfactual passes."""
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError("structure gate override must lie in [0, 1]")
        self.structure_gate_override = None if value is None else float(value)

    def set_tree_flow_override(self, value: Optional[str]) -> None:
        """Restrict tree messages to one orientation for a matched audit pass.

        This switch is deliberately not a trainable model option.  It is used
        to ask whether a trained bidirectional block still benefits from
        having both parent-to-child (``top_down``) and child-to-parent
        (``bottom_up``) messages available at inference time.
        """
        if value not in {None, "top_down", "bottom_up"}:
            raise ValueError(
                "tree flow override must be None, 'top_down', or 'bottom_up'"
            )
        self.tree_flow_override = value

    def set_tree_message_scale_override(self, value: Optional[float]) -> None:
        """Scale forest features for an inference-only dose-response audit."""
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError("tree message scale override must lie in [0, 1]")
        self.tree_message_scale_override = None if value is None else float(value)

    def set_tree_community_coherence_override(self, value: Optional[float]) -> None:
        """Blend forest features toward each community's mean for an audit."""
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError("tree community coherence override must lie in [0, 1]")
        self.tree_community_coherence_override = None if value is None else float(value)

    @staticmethod
    def _segment_mean(values: Tensor, index: LongTensor, num_segments: int) -> Tensor:
        output = values.new_zeros((num_segments, values.size(-1)))
        counts = values.new_zeros((num_segments, 1))
        output.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones((values.size(0), 1)))
        return output / counts.clamp_min(1.0)

    @staticmethod
    def _boundary_statistics(
        graph_edge_index: LongTensor,
        community_labels: LongTensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        degree = torch.zeros(num_nodes, dtype=dtype, device=community_labels.device)
        boundary = torch.zeros_like(degree)
        if graph_edge_index.numel() == 0:
            return degree, boundary
        source, target = graph_edge_index
        valid = (source != target) & (source >= 0) & (target >= 0)
        valid &= (source < num_nodes) & (target < num_nodes)
        source = source[valid]
        target = target[valid]
        if source.numel() == 0:
            return degree, boundary
        ones = degree.new_ones(source.numel())
        is_boundary = (community_labels[source] != community_labels[target]).to(dtype=dtype)
        degree.index_add_(0, source, ones)
        degree.index_add_(0, target, ones)
        boundary.index_add_(0, source, is_boundary)
        boundary.index_add_(0, target, is_boundary)
        return degree, boundary / degree.clamp_min(1.0)

    def _depth_scheduled_tree_aggregate(
        self,
        normalized: Tensor,
        source: LongTensor,
        target: LongTensor,
        orientation: Tensor,
        tree_id: LongTensor,
        depth: LongTensor,
        message_weight: Tensor,
    ) -> Tensor:
        """Run a bounded leaf-to-root sweep followed by a root-to-leaf sweep.

        Each tree has its own state, so information cannot leak between trees
        before their final mean. The half-step and tanh bound every node's
        per-direction update while retaining gradients through the full path.
        """
        tree_sum = torch.zeros_like(normalized)
        num_nodes = normalized.size(0)
        for tree in range(self.num_trees):
            tree_mask = tree_id == tree
            if not bool(tree_mask.any()):
                continue
            state = normalized
            for sign, reverse in ((-1, True), (1, False)):
                direction_mask = tree_mask & (orientation * sign > 0)
                levels = sorted(depth[direction_mask].unique().tolist(), reverse=reverse)
                for level in levels:
                    selected = direction_mask & (depth == level)
                    local_source = source[selected]
                    local_target = target[selected]
                    local_weight = message_weight[selected]
                    value_input = (
                        state[local_source] - state[local_target]
                        if self.relative_tree_messages else state[local_source]
                    )
                    if self.direction_conditioning_enabled:
                        value_net = (
                            self.bottom_up_value if sign < 0
                            else self.top_down_value
                        )
                        value = value_net(value_input)
                    else:
                        value = 0.5 * (
                            self.bottom_up_value(value_input)
                            + self.top_down_value(value_input)
                        )
                    aggregate = torch.zeros_like(normalized)
                    denominator = normalized.new_zeros((num_nodes, 1))
                    aggregate.index_add_(
                        0, local_target, value * local_weight.unsqueeze(-1)
                    )
                    denominator.index_add_(
                        0, local_target, local_weight.unsqueeze(-1)
                    )
                    state = state + 0.5 * torch.tanh(
                        aggregate / denominator.clamp_min(1.0)
                    )
            tree_sum = tree_sum + (state - normalized)
        return tree_sum / self.num_trees

    def forward(
        self,
        node_feat: FloatTensor,
        cover: TreeCoverBatch,
        graph_edge_index: LongTensor,
        community_labels: Optional[LongTensor],
        batch_index: LongTensor,
    ) -> FloatTensor:
        num_nodes = int(node_feat.size(0))
        if num_nodes == 0:
            return node_feat
        if community_labels is None:
            community_labels = torch.zeros(
                num_nodes,
                dtype=torch.long,
                device=node_feat.device,
            )
        else:
            community_labels = community_labels.to(device=node_feat.device, dtype=torch.long)

        normalized = self.input_norm(node_feat)
        degree, boundary_ratio = self._boundary_statistics(
            graph_edge_index=graph_edge_index,
            community_labels=community_labels,
            num_nodes=num_nodes,
            dtype=node_feat.dtype,
        )

        tree_aggregate = node_feat.new_zeros(node_feat.shape)
        if (
            self.tree_messages_enabled
            and not self.node_only_control
            and cover.edge_index.numel() > 0
        ):
            source, target = cover.edge_index
            orientation = cover.orientation.to(dtype=node_feat.dtype)
            tree_id = cover.tree_id
            depth = cover.depth
            if self.depth_scheduled_tree_flow and depth is None:
                raise ValueError("depth-scheduled tree flow requires cover.depth")
            tree_fraction = tree_id.to(dtype=node_feat.dtype)
            if self.tree_coverage_relation:
                # The number of trees reusing a directed graph edge is
                # structural evidence: bridges recur; alternate cycle edges
                # need not. Unlike tree_id, it has no arbitrary tree order.
                _, inverse, counts = torch.unique(
                    cover.edge_index.t(), dim=0,
                    return_inverse=True, return_counts=True,
                )
                tree_fraction = (
                    counts[inverse].to(dtype=node_feat.dtype) - 1.0
                ) / max(self.num_trees - 1, 1)
            if self.tree_flow_override is not None:
                keep = orientation > 0 if self.tree_flow_override == "top_down" else orientation < 0
                source = source[keep]
                target = target[keep]
                orientation = orientation[keep]
                tree_id = tree_id[keep]
                if depth is not None:
                    depth = depth[keep]
                tree_fraction = tree_fraction[keep]
            if not self.tree_coverage_relation:
                tree_fraction = tree_fraction / max(self.num_trees - 1, 1)
            if self.tree_identity_agnostic:
                # A cover is an unordered set of trees. Keep tree_id for
                # depth scheduling, but remove its arbitrary ordinal value
                # from the relation descriptor.
                tree_fraction = torch.full_like(tree_fraction, 0.5)
            same_community = (
                community_labels[source] == community_labels[target]
            ).to(dtype=node_feat.dtype)
            boundary_mean = 0.5 * (boundary_ratio[source] + boundary_ratio[target])
            boundary_gap = (boundary_ratio[source] - boundary_ratio[target]).abs()
            relation = torch.stack(
                [same_community, boundary_mean, boundary_gap, tree_fraction],
                dim=-1,
            )

            difference = normalized[source] - normalized[target]
            if self.direction_conditioning_enabled:
                directional_logit = self.direction_score(difference).squeeze(-1)
                if self.relation_conditioning_enabled:
                    directional_logit = directional_logit + orientation * self.direction_relation(
                        relation
                    ).squeeze(-1)
                directional_weight = torch.sigmoid(directional_logit)
            else:
                directional_weight = node_feat.new_ones(source.numel())
            if self.relation_conditioning_enabled:
                relation_weight = 2.0 * torch.sigmoid(
                    self.relation_gate(relation).squeeze(-1)
                )
            else:
                relation_weight = node_feat.new_ones(source.numel())
            message_weight = directional_weight * relation_weight
            if self.direction_balance_logit is not None:
                top_down_scale = 2.0 * torch.sigmoid(self.direction_balance_logit)
                orientation_scale = torch.where(
                    orientation > 0,
                    top_down_scale,
                    2.0 - top_down_scale,
                )
                message_weight = message_weight * orientation_scale

            if self.depth_scheduled_tree_flow:
                tree_aggregate = self._depth_scheduled_tree_aggregate(
                    normalized, source, target, orientation, tree_id,
                    depth, message_weight,
                )
            else:
                tree_value_input = difference if self.relative_tree_messages else normalized[source]
                top_down = self.top_down_value(tree_value_input)
                bottom_up = self.bottom_up_value(tree_value_input)
                if self.direction_conditioning_enabled:
                    message = torch.where(
                        (orientation > 0).unsqueeze(-1),
                        top_down,
                        bottom_up,
                    )
                else:
                    message = 0.5 * (top_down + bottom_up)
                message = message * message_weight.unsqueeze(-1)
                normalizer = node_feat.new_zeros((num_nodes, 1))
                tree_aggregate.index_add_(0, target, message)
                normalizer.index_add_(0, target, message_weight.unsqueeze(-1))
                tree_aggregate = tree_aggregate / normalizer.clamp_min(1.0)

        _, graph_inverse = torch.unique(batch_index.long(), sorted=True, return_inverse=True)
        graph_mean = self._segment_mean(
            normalized,
            graph_inverse,
            int(graph_inverse.max().item()) + 1,
        )
        if self.prototype_gate is not None:
            graph_count = int(graph_mean.size(0))
            boundary_mean_for_prototype = self._segment_mean(
                boundary_ratio.unsqueeze(-1), graph_inverse, graph_count
            )
            graph_size_for_prototype = node_feat.new_zeros((graph_count, 1))
            graph_size_for_prototype.index_add_(
                0, graph_inverse, node_feat.new_ones((num_nodes, 1))
            )
            prototype_descriptor = torch.cat(
                [
                    graph_mean,
                    boundary_mean_for_prototype,
                    torch.log1p(graph_size_for_prototype) / 6.0,
                ],
                dim=-1,
            )
            graph_prototype_gate = 2.0 * torch.sigmoid(
                self.prototype_gate(prototype_descriptor).squeeze(-1)
            )
            detached_prototype_gate = graph_prototype_gate.detach()
            self.last_mean_prototype_gate = float(detached_prototype_gate.mean().cpu())
            self.last_min_prototype_gate = float(detached_prototype_gate.min().cpu())
            self.last_max_prototype_gate = float(detached_prototype_gate.max().cpu())
            self.last_prototype_gate_tensor = graph_prototype_gate
        else:
            graph_prototype_gate = None
            self.last_mean_prototype_gate = 1.0
            self.last_min_prototype_gate = 1.0
            self.last_max_prototype_gate = 1.0
            self.last_prototype_gate_tensor = None
        if self.prototype_context_enabled and not self.node_only_control:
            graph_pairs = torch.stack([batch_index.long(), community_labels], dim=-1)
            _, community_inverse = torch.unique(graph_pairs, dim=0, return_inverse=True)
            community_mean = self._segment_mean(
                normalized,
                community_inverse,
                int(community_inverse.max().item()) + 1,
            )
            community_contrast = community_mean[community_inverse] - graph_mean[graph_inverse]
            # Boundary nodes receive stronger community-vs-graph contrast in the
            # representation branch; MA-CPSG remains responsible for weak geometric pull.
            if self.node_relative_prototype:
                # Relative node-to-own-community evidence replaces an
                # absolute prototype vector; no coordinates are postprocessed.
                community_context = (
                    community_mean[community_inverse] - normalized
                ) * (1.0 + boundary_ratio.unsqueeze(-1))
            elif self.route_graph_mean_to_node:
                # The global part is handled by the node path below; only
                # community-relative evidence enters the structural path.
                community_context = community_contrast + (
                    boundary_ratio.unsqueeze(-1) * community_contrast
                )
            elif self.prototype_graph_mean_scale == 1.0:
                # Preserve the legacy arithmetic exactly for old checkpoints.
                community_context = community_mean[community_inverse] + (
                    boundary_ratio.unsqueeze(-1) * community_contrast
                )
            else:
                community_context = (
                    self.prototype_graph_mean_scale * graph_mean[graph_inverse]
                    + community_contrast
                    + boundary_ratio.unsqueeze(-1) * community_contrast
                )
            if graph_prototype_gate is not None:
                community_context = (
                    community_context * graph_prototype_gate[graph_inverse, None]
                )
        else:
            community_context = torch.zeros_like(normalized)

        if self.tree_community_coherence_override is not None:
            graph_community = torch.stack(
                [batch_index.long(), community_labels], dim=-1
            )
            _, community_inverse = torch.unique(
                graph_community, dim=0, return_inverse=True
            )
            community_tree_mean = self._segment_mean(
                tree_aggregate,
                community_inverse,
                int(community_inverse.max().item()) + 1,
            )
            coherence = self.tree_community_coherence_override
            tree_aggregate = (
                (1.0 - coherence) * tree_aggregate
                + coherence * community_tree_mean[community_inverse]
            )
        if self.tree_message_scale_override is not None:
            tree_aggregate = tree_aggregate * self.tree_message_scale_override
        update_input = torch.cat([normalized, tree_aggregate, community_context], dim=-1)
        structural_delta = self.update_net(update_input)
        gate_boundary = boundary_ratio.unsqueeze(-1)
        gate_degree = torch.log1p(degree).unsqueeze(-1)
        if self.node_only_control:
            gate_boundary = torch.zeros_like(gate_boundary)
            gate_degree = torch.zeros_like(gate_degree)
        if self.adaptive_structure_gate:
            # Keep the outer node gate identical for the full and nested
            # node-only paths; all structural evidence acts through the
            # explicit graph-level structure gate below.
            gate_tree = torch.zeros_like(tree_aggregate)
            gate_community = torch.zeros_like(community_context)
            gate_boundary = torch.zeros_like(gate_boundary)
            gate_degree = torch.zeros_like(gate_degree)
        else:
            gate_tree = tree_aggregate
            gate_community = community_context
        gate_input = torch.cat(
            [
                normalized,
                gate_tree,
                gate_community,
                gate_boundary,
                gate_degree,
            ],
            dim=-1,
        )
        adaptive_gate = self.adaptive_gate(gate_input)
        if self.graph_gate is not None or self.structure_gate is not None:
            graph_count = int(graph_mean.size(0))
            boundary_feature = boundary_ratio.unsqueeze(-1)
            degree_feature = torch.log1p(degree).unsqueeze(-1)
            boundary_mean = self._segment_mean(
                boundary_feature, graph_inverse, graph_count
            )
            degree_mean = self._segment_mean(
                degree_feature, graph_inverse, graph_count
            )
            boundary_std = self._segment_mean(
                (boundary_feature - boundary_mean[graph_inverse]).square(),
                graph_inverse,
                graph_count,
            ).sqrt()
            degree_std = self._segment_mean(
                (degree_feature - degree_mean[graph_inverse]).square(),
                graph_inverse,
                graph_count,
            ).sqrt()
            feature_dispersion = self._segment_mean(
                (normalized - graph_mean[graph_inverse]).square().mean(
                    dim=-1, keepdim=True
                ),
                graph_inverse,
                graph_count,
            )
            graph_size = node_feat.new_zeros((graph_count, 1))
            graph_size.index_add_(
                0,
                graph_inverse,
                node_feat.new_ones((num_nodes, 1)),
            )
            graph_descriptor = torch.cat(
                [
                    boundary_mean,
                    boundary_std,
                    (
                        degree_mean / 3.0
                        if self.unsaturated_degree_mean
                        else torch.tanh(degree_mean)
                    ),
                    torch.tanh(degree_std),
                    torch.log1p(graph_size) / 6.0,
                    torch.tanh(feature_dispersion),
                ],
                dim=-1,
            )
        else:
            graph_descriptor = None
        if self.graph_gate is not None:
            graph_gate = self.graph_gate(graph_descriptor)[graph_inverse]
        else:
            graph_gate = adaptive_gate.new_ones(adaptive_gate.shape)
        if self.structure_gate is not None:
            graph_structure_gate = self.structure_gate(graph_descriptor).squeeze(-1)
            if self.node_only_control:
                graph_structure_gate = torch.zeros_like(graph_structure_gate)
            if self.structure_gate_override is not None:
                graph_structure_gate = torch.full_like(
                    graph_structure_gate, self.structure_gate_override
                )
            structure_gate = graph_structure_gate[graph_inverse, None]
            node_input = (
                normalized + graph_mean[graph_inverse]
                if self.route_graph_mean_to_node else normalized
            )
            node_delta = self.node_update_net(node_input)
            delta = node_delta + structure_gate * structural_delta
            detached_structure_gate = graph_structure_gate.detach()
            self.last_mean_structure_gate = float(detached_structure_gate.mean().cpu())
            self.last_min_structure_gate = float(detached_structure_gate.min().cpu())
            self.last_max_structure_gate = float(detached_structure_gate.max().cpu())
            self.last_structure_gate_tensor = graph_structure_gate
        else:
            delta = structural_delta
            self.last_mean_structure_gate = 0.0
            self.last_min_structure_gate = 0.0
            self.last_max_structure_gate = 0.0
            self.last_structure_gate_tensor = None
        mix = torch.sigmoid(self.mix_logit) * self.scale
        combined_gate = adaptive_gate * graph_gate
        output = node_feat + mix * combined_gate * self.dropout(delta)
        self.last_mean_gate = float(combined_gate.detach().mean().cpu())
        self.last_mean_graph_gate = float(graph_gate.detach().mean().cpu())
        self.last_mix = float(mix.detach().cpu())
        return output
