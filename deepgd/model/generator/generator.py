from deepgd.constants import EPS
from ..common import EdgeFeatureExpansion
from .generator_block import GeneratorBlock
from .generator_layer import GeneratorLayer
from .hctf import CoverageAwareTreeCover, HCTFBlock
from .hctf_v2 import HCTFv2Block, HCTFv2EquivariantCoordinateHead
from .hctf_conflict import HCTFConflictHead, HCTFGraphSizeGate

from attrs import define
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, FloatTensor, LongTensor


@define(kw_only=True, eq=False, repr=False, slots=False)
class Generator(nn.Module):
    @dataclass(kw_only=True)
    class Params:
        num_blocks: int = 10
        block_depth: int = 2
        block_width: int = 8
        block_output_dim: int = 8
        edge_net_depth: int = 1
        edge_net_width: int = 16
        edge_attr_dim: int = 2
        node_attr_dim: int = 2
        care_enabled: bool = False
        care_rank: int = 2
        care_scale: float = 0.10
        care_same_index: int = -1
        care_adj_index: int = -1
        care_far_index: int = -1
        care_boundary_index: int = -1
        care_use_boundary_gate: bool = True
        care_boundary_damp: float = 0.50
        care_checkpoint: bool = True
        hctf_enabled: bool = False
        hctf_version: str = "v1"
        hctf_num_trees: int = 3
        hctf_hidden_dim: int = 16
        hctf_dropout: float = 0.05
        hctf_scale: float = 0.20
        hctf_mix_init: float = 0.05
        hctf_block_stride: int = 1
        hctf_late_only: bool = False
        hctf_stochastic_depth_prob: float = 0.0
        hctf_stochastic_depth_seed: int = 42
        hctf_cache_size: int = 4096
        hctf_graph_gate: bool = False
        hctf_adaptive_structure_gate: bool = False
        hctf_structure_gate_init: float = 0.05
        hctf_zero_init: bool = False
        hctf_coordinate_residual: bool = False
        hctf_coordinate_max_step: float = 0.08
        hctf_conflict_head: bool = False
        hctf_conflict_max_pairs: int = 1024
        hctf_conflict_max_step: float = 0.04
        hctf_conflict_deterministic_pairs: bool = False
        hctf_conflict_graph_context: bool = False
        hctf_graph_size_gate: bool = False
        hctf_graph_size_midpoint: float = 20.0
        hctf_graph_size_width: float = 5.0
        hctf_exact_risk_gate: bool = False
        hctf_tree_messages: bool = True
        hctf_direction_conditioning: bool = True
        hctf_relation_conditioning: bool = True
        hctf_prototype_context: bool = True
        hctf_node_only_control: bool = False
        hctf_learn_direction_balance: bool = False
        hctf_unsaturated_degree_mean: bool = False
        hctf_centered_prototype_context: bool = False
        hctf_prototype_graph_mean_scale: float = 1.0
        hctf_independent_prototype_gate: bool = False
        hctf_route_graph_mean_to_node: bool = False
        hctf_relative_tree_messages: bool = False
        hctf_depth_scheduled_tree_flow: bool = False
        hctf_tree_identity_agnostic: bool = False
        hctf_tree_coverage_relation: bool = False
        hctf_community_first_cover: bool = False
        hctf_node_relative_prototype: bool = False
        hctf_stage_residual_adapters: bool = False
        hctf_preference: tuple[float, float, float] = (0.34, 0.33, 0.33)
        hctf_v2_tree_attention: bool = True
        hctf_v2_equivariant_coordinate: bool = True

    @dataclass(kw_only=True)
    class BlockConfig:
        relative_edge_feat_mode: Optional[str] = "jump"
        residual: bool = True

    @dataclass(kw_only=True)
    class EdgeNetConfig:
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = "tanh"
        bn: Optional[str] = "batch_norm"
        dp: float = 0.2
        residual: bool = False

    @dataclass(kw_only=True)
    class GNNConfig:
        root_weight: bool = True
        bn: Optional[str] = "pyg_batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.1

    params: Params = Params()
    block_config: BlockConfig = BlockConfig()
    edge_net_config: EdgeNetConfig = EdgeNetConfig()
    gnn_config: GNNConfig = GNNConfig()
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(
        src_feat=False,
        dst_feat=False,
        diff_vec=False,
        unit_vec=True,
        vec_norm=True,
        vec_norm_inv=False,
        vec_norm_square=False,
        vec_norm_inv_square=False,
        edge_attr_inv=False,
        edge_attr_square=False,
        edge_attr_inv_square=False
    )
    eps: float = EPS

    def __attrs_post_init__(self):
        super().__init__()

        main_block_config = GeneratorBlock.Config(
            in_dim=self.params.block_output_dim,
            hidden_dims=[self.params.block_width] * self.params.block_depth,
            out_dim=self.params.block_output_dim,
            edge_attr_dim=self.params.edge_attr_dim,
            node_attr_dim=self.params.node_attr_dim,
            dynamic_edge_feat_mode=self.block_config.relative_edge_feat_mode,
            residual=self.block_config.residual
        )
        main_edge_net_config = GeneratorLayer.EdgeNetConfig(
            width=self.params.edge_net_width,
            depth=self.params.edge_net_depth,
            hidden_act=self.edge_net_config.hidden_act,
            out_act=self.edge_net_config.out_act,
            bn=self.edge_net_config.bn,
            dp=self.edge_net_config.dp,
            residual=self.edge_net_config.residual,
            care_enabled=self.params.care_enabled,
            care_rank=self.params.care_rank,
            care_scale=self.params.care_scale,
            care_same_index=self.params.care_same_index,
            care_adj_index=self.params.care_adj_index,
            care_far_index=self.params.care_far_index,
            care_boundary_index=self.params.care_boundary_index,
            care_use_boundary_gate=self.params.care_use_boundary_gate,
            care_boundary_damp=self.params.care_boundary_damp,
            care_checkpoint=self.params.care_checkpoint,
        )
        main_gnn_config = GeneratorLayer.GNNConfig(
            aggr="mean",
            root_weight=self.gnn_config.root_weight,
            dense=False,
            bn=self.gnn_config.bn,
            act=self.gnn_config.act,
            dp=self.gnn_config.dp
        )

        # TODO: Normalization
        #   * add [Center, Standardization] before input
        #   * add [Center, ParametricScaling] after output

        self.block_list: nn.ModuleList = nn.ModuleList()
        self.block_list.append(
            GeneratorBlock(
                start_layer_index=0,
                config=GeneratorBlock.Config(
                    in_dim=self.params.node_attr_dim,
                    hidden_dims=[],
                    out_dim=self.params.block_output_dim,
                    edge_attr_dim=self.params.edge_attr_dim,
                    node_attr_dim=self.params.node_attr_dim,
                    dynamic_edge_feat_mode=None,
                    residual=False
                ),
                edge_net_config=main_edge_net_config,
                gnn_config=main_gnn_config,
                eps=self.eps
            )
        )

        for _ in range(self.params.num_blocks):
            self.block_list.append(
                GeneratorBlock(
                    start_layer_index=self.block_list[-1].next_layer_index,
                    config=main_block_config,
                    edge_net_config=main_edge_net_config,
                    gnn_config=main_gnn_config,
                    edge_feat_expansion=self.edge_feat_expansion,
                    eps=self.eps
                )
            )
        self.block_list.append(
            GeneratorBlock(
                start_layer_index=self.block_list[-1].next_layer_index,
                config=GeneratorBlock.Config(
                    in_dim=self.params.block_output_dim,
                    hidden_dims=[],
                    out_dim=2,
                    edge_attr_dim=self.params.edge_attr_dim,
                    node_attr_dim=self.params.node_attr_dim,
                    dynamic_edge_feat_mode=None,
                    residual=False,
                ),
                edge_net_config=main_edge_net_config,
                gnn_config=GeneratorLayer.GNNConfig(
                    aggr="mean",
                    root_weight=True,
                    dense=False,
                    bn=None,
                    act=None,
                    dp=0.0,
                ),
                eps=self.eps
            )
        )

        # Build optional modules after the original DeepGD blocks so a fixed seed
        # initializes the unchanged backbone identically in ablation runs.
        self.hctf_cover = None
        self.hctf_block = None
        self.hctf_stage_adapters = None
        self.hctf_coordinate_projection = None
        self.hctf_coordinate_equivariant = None
        self.hctf_coordinate_trust = None
        self.hctf_conflict_head = None
        self.hctf_graph_size_gate = None
        self.hctf_conflict_head_enabled = True
        self.hctf_exact_risk_gate = None
        self.hctf_bypass = False
        # Read-only same-checkpoint ablation; None preserves every configured
        # injection point. This is not serialized or used for formal training.
        self.hctf_active_block_ids_override = None
        if self.params.hctf_stage_residual_adapters and (
            not self.params.hctf_enabled or self.params.hctf_coordinate_residual
        ):
            raise ValueError("stage adapters require hidden-block HCTF injection")
        if not 0.0 <= self.params.hctf_stochastic_depth_prob < 1.0:
            raise ValueError("hctf_stochastic_depth_prob must be in [0, 1)")
        self._hctf_stochastic_forward_counter = 0
        self.last_hctf_coordinate_trust = 1.0
        self.last_hctf_coordinate_limiter = 1.0
        self.last_hctf_exact_risk = 1.0
        self.last_hctf_exact_risk_min = 1.0
        self.last_hctf_exact_risk_max = 1.0
        self.last_hctf_exact_risk_std = 0.0
        self.last_hctf_coordinate_trust_tensor = None
        self.last_hctf_exact_risk_tensor = None
        self.hctf_residual_progress = 1.0
        if self.params.hctf_enabled:
            if self.params.hctf_version not in {"v1", "v2"}:
                raise ValueError("hctf_version must be 'v1' or 'v2'")
            self.hctf_cover = CoverageAwareTreeCover(
                num_trees=self.params.hctf_num_trees,
                max_cache_size=self.params.hctf_cache_size,
                community_first=self.params.hctf_community_first_cover,
            )
            block_type = HCTFv2Block if self.params.hctf_version == "v2" else HCTFBlock
            block_kwargs = dict(
                feature_dim=self.params.block_output_dim,
                hidden_dim=self.params.hctf_hidden_dim,
                num_trees=self.params.hctf_num_trees,
                dropout=self.params.hctf_dropout,
                scale=self.params.hctf_scale,
                graph_gate_enabled=self.params.hctf_graph_gate,
                zero_init=self.params.hctf_zero_init,
                tree_messages_enabled=self.params.hctf_tree_messages,
                direction_conditioning_enabled=self.params.hctf_direction_conditioning,
                relation_conditioning_enabled=self.params.hctf_relation_conditioning,
                prototype_context_enabled=self.params.hctf_prototype_context,
                node_only_control=self.params.hctf_node_only_control,
            )
            if self.params.hctf_version == "v1":
                block_kwargs.update(
                    adaptive_structure_gate=self.params.hctf_adaptive_structure_gate,
                    structure_gate_init=self.params.hctf_structure_gate_init,
                    learn_direction_balance=self.params.hctf_learn_direction_balance,
                    unsaturated_degree_mean=self.params.hctf_unsaturated_degree_mean,
                    centered_prototype_context=self.params.hctf_centered_prototype_context,
                    prototype_graph_mean_scale=self.params.hctf_prototype_graph_mean_scale,
                    independent_prototype_gate=self.params.hctf_independent_prototype_gate,
                    route_graph_mean_to_node=self.params.hctf_route_graph_mean_to_node,
                    relative_tree_messages=self.params.hctf_relative_tree_messages,
                    depth_scheduled_tree_flow=self.params.hctf_depth_scheduled_tree_flow,
                    tree_identity_agnostic=self.params.hctf_tree_identity_agnostic,
                    tree_coverage_relation=self.params.hctf_tree_coverage_relation,
                    node_relative_prototype=self.params.hctf_node_relative_prototype,
                    mix_init=(
                        self.params.hctf_mix_init
                        if self.params.hctf_adaptive_structure_gate
                        else None
                    ),
                )
            if self.params.hctf_version == "v2":
                block_kwargs.update(
                    mix_init=self.params.hctf_mix_init,
                    preference=self.params.hctf_preference,
                    tree_attention_enabled=self.params.hctf_v2_tree_attention,
                )
            self.hctf_block = block_type(**block_kwargs)
            if self.params.hctf_graph_size_gate:
                self.hctf_graph_size_gate = HCTFGraphSizeGate(
                    midpoint=self.params.hctf_graph_size_midpoint,
                    width=self.params.hctf_graph_size_width,
                )
            if self.params.hctf_conflict_head:
                self.hctf_conflict_head = HCTFConflictHead(
                    feature_dim=self.params.block_output_dim,
                    hidden_dim=self.params.hctf_hidden_dim,
                    max_pairs=self.params.hctf_conflict_max_pairs,
                    max_step=self.params.hctf_conflict_max_step,
                    deterministic_pairs=self.params.hctf_conflict_deterministic_pairs,
                    graph_context=self.params.hctf_conflict_graph_context,
                )
            if self.params.hctf_coordinate_residual:
                if (
                    self.params.hctf_version == "v2"
                    and self.params.hctf_v2_equivariant_coordinate
                ):
                    self.hctf_coordinate_equivariant = HCTFv2EquivariantCoordinateHead(
                        feature_dim=self.params.block_output_dim,
                        hidden_dim=self.params.hctf_hidden_dim,
                        num_trees=self.params.hctf_num_trees,
                        preference=self.params.hctf_preference,
                        zero_init=self.params.hctf_zero_init,
                        node_only_control=self.params.hctf_node_only_control,
                        tree_messages_enabled=self.params.hctf_tree_messages,
                        prototype_context_enabled=self.params.hctf_prototype_context,
                        eps=self.eps,
                    )
                else:
                    self.hctf_coordinate_projection = nn.Linear(
                        self.params.block_output_dim,
                        2,
                        bias=False,
                    )
                    nn.init.xavier_uniform_(self.hctf_coordinate_projection.weight)
                    self.hctf_coordinate_trust = nn.Sequential(
                        nn.Linear(5, self.params.hctf_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(self.params.hctf_hidden_dim, 1),
                        nn.Sigmoid(),
                    )
                    nn.init.zeros_(self.hctf_coordinate_trust[-2].weight)
                    nn.init.constant_(self.hctf_coordinate_trust[-2].bias, -2.0)
                if self.params.hctf_exact_risk_gate:
                    self.hctf_exact_risk_gate = nn.Sequential(
                        nn.Linear(11, self.params.hctf_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(self.params.hctf_hidden_dim, 1),
                        nn.Sigmoid(),
                    )
                    nn.init.zeros_(self.hctf_exact_risk_gate[-2].weight)
                    nn.init.constant_(self.hctf_exact_risk_gate[-2].bias, 4.0)
            if self.params.hctf_stage_residual_adapters:
                # A separate feature-space correction for each hidden injection
                # point. Zero initialization exactly preserves the shared-HCTF
                # forward before training. Preserve the legacy RNG stream too:
                # nn.Linear initializes randomly before we zero its weight,
                # and that otherwise changes matched data-loader shuffling.
                rng_state = torch.random.get_rng_state()
                try:
                    self.hctf_stage_adapters = nn.ModuleList(
                        nn.Linear(self.params.block_output_dim,
                                  self.params.block_output_dim, bias=False)
                        for _ in range(len(self.block_list) - 1)
                    )
                    for adapter in self.hctf_stage_adapters:
                        nn.init.zeros_(adapter.weight)
                finally:
                    torch.random.set_rng_state(rng_state)

    def _apply_hctf_coordinate_residual(
        self,
        base_pos,
        hidden,
        tree_cover,
        graph_edge_index,
        community_labels,
        batch_index,
    ):
        enhanced = self.hctf_block(
            node_feat=hidden,
            cover=tree_cover,
            graph_edge_index=graph_edge_index,
            community_labels=community_labels,
            batch_index=batch_index,
        )
        v2_trust = None
        if self.hctf_coordinate_equivariant is not None:
            raw_delta, v2_trust = self.hctf_coordinate_equivariant(
                base_pos=base_pos,
                hidden=hidden,
                enhanced=enhanced,
                cover=tree_cover,
                graph_edge_index=graph_edge_index,
                community_labels=community_labels,
                batch_index=batch_index,
            )
            learned_mix = (
                torch.sigmoid(self.hctf_block.mix_logit) * self.hctf_block.scale
            )
            raw_delta = (
                float(self.hctf_residual_progress) * learned_mix * raw_delta
            )
        else:
            raw_delta = self.hctf_coordinate_projection(enhanced - hidden)
        proposal = base_pos + raw_delta
        graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if graph_count == 0:
            return base_pos

        source, target = graph_edge_index.long()
        valid = (source != target) & (source < target)
        source, target = source[valid], target[valid]
        descriptors = []
        risk_descriptors = []
        limiters = []
        for graph_id in range(graph_count):
            node_mask = batch_index == graph_id
            graph_delta = raw_delta[node_mask]
            edge_mask = batch_index[source] == graph_id
            graph_source, graph_target = source[edge_mask], target[edge_mask]
            node_count = int(node_mask.sum().item())
            edge_count = int(graph_source.numel())
            if graph_source.numel():
                base_vec = base_pos[graph_target] - base_pos[graph_source]
                proposal_vec = proposal[graph_target] - proposal[graph_source]
                base_length = base_vec.norm(dim=1).clamp_min(self.eps)
                proposal_length = proposal_vec.norm(dim=1)
                edge_scale = base_length.detach().median().clamp_min(self.eps)
                stretch = ((proposal_length - base_length).abs() / base_length).mean()
                direction_change = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        base_vec,
                        proposal_vec,
                        dim=1,
                        eps=self.eps,
                    )
                ).mean()
                base_cv = base_length.std(unbiased=False) / base_length.mean().clamp_min(
                    self.eps
                )
                proposal_cv = proposal_length.std(
                    unbiased=False
                ) / proposal_length.mean().clamp_min(self.eps)
            else:
                edge_scale = base_pos[node_mask].detach().std().clamp_min(self.eps)
                stretch = raw_delta.sum() * 0.0
                direction_change = raw_delta.sum() * 0.0
                base_cv = raw_delta.sum() * 0.0
                proposal_cv = raw_delta.sum() * 0.0

            displacement = graph_delta.norm(dim=1)
            rms = (
                displacement.square().mean() + float(self.eps) ** 2
            ).sqrt()
            maximum = displacement.max() if displacement.numel() else rms
            displacement_cv = displacement.std(
                unbiased=False
            ) / displacement.mean().clamp_min(self.eps)
            graph_size = base_pos.new_tensor(float(node_count))
            graph_edges = base_pos.new_tensor(float(edge_count))
            density = graph_edges / max(float(node_count * (node_count - 1) / 2), 1.0)
            if community_labels is not None and node_count:
                community_count = community_labels[node_mask].unique().numel()
            else:
                community_count = 1
            graph_communities = base_pos.new_tensor(float(community_count))
            descriptors.append(
                torch.stack(
                    [
                        torch.log1p(graph_size) / 6.0,
                        rms / edge_scale,
                        maximum / edge_scale,
                        stretch,
                        direction_change,
                    ]
                )
            )
            risk_descriptors.append(
                torch.stack(
                    [
                        torch.log1p(graph_size) / 6.0,
                        torch.log1p(graph_edges) / 7.0,
                        density,
                        torch.log1p(graph_communities) / 3.0,
                        rms / edge_scale,
                        maximum / edge_scale,
                        stretch,
                        direction_change,
                        base_cv,
                        proposal_cv,
                        displacement_cv,
                    ]
                )
            )
            max_rms = float(self.params.hctf_coordinate_max_step) * edge_scale
            limiters.append((max_rms / rms.clamp_min(self.eps)).clamp(max=1.0))

        descriptor = torch.stack(descriptors)
        if v2_trust is None:
            trust = self.hctf_coordinate_trust(descriptor).squeeze(-1)
        else:
            trust = v2_trust
        limiter = torch.stack(limiters)
        if self.hctf_exact_risk_gate is not None:
            risk_descriptor = torch.stack(risk_descriptors).detach()
            exact_risk = self.hctf_exact_risk_gate(risk_descriptor).squeeze(-1)
        else:
            exact_risk = torch.ones_like(trust)
        graph_weight = trust * limiter * exact_risk
        self.last_hctf_coordinate_trust = float(trust.detach().mean().cpu())
        self.last_hctf_coordinate_limiter = float(limiter.detach().mean().cpu())
        self.last_hctf_exact_risk = float(exact_risk.detach().mean().cpu())
        self.last_hctf_exact_risk_min = float(exact_risk.detach().min().cpu())
        self.last_hctf_exact_risk_max = float(exact_risk.detach().max().cpu())
        self.last_hctf_exact_risk_std = float(
            exact_risk.detach().std(unbiased=False).cpu()
        )
        self.last_hctf_coordinate_trust_tensor = trust
        self.last_hctf_exact_risk_tensor = exact_risk
        return base_pos + graph_weight[batch_index.long(), None] * raw_delta

    def set_hctf_residual_progress(self, progress: float) -> None:
        self.hctf_residual_progress = min(max(float(progress), 0.0), 1.0)
        if self.hctf_block is not None and hasattr(
            self.hctf_block, "set_residual_progress"
        ):
            self.hctf_block.set_residual_progress(self.hctf_residual_progress)

    def set_hctf_active_block_ids_override(self, block_ids) -> None:
        """Keep only selected hidden-block HCTF insertions for an audit pass."""
        if block_ids is None:
            self.hctf_active_block_ids_override = None
            return
        valid = set(range(len(self.block_list) - 1))
        selected = frozenset(int(block_id) for block_id in block_ids)
        if not selected.issubset(valid):
            raise ValueError("HCTF block IDs must name non-coordinate blocks")
        self.hctf_active_block_ids_override = selected

    def _apply_hctf_stochastic_depth(
        self, before: FloatTensor, enhanced: FloatTensor,
        batch_index: LongTensor, block_id: int, forward_counter: int,
    ) -> FloatTensor:
        """Drop a shared HCTF residual independently per graph during training.

        A private CPU generator keeps this trial from shifting the global RNG
        stream that controls the matched training data and existing dropout.
        """
        probability = self.params.hctf_stochastic_depth_prob
        if probability == 0.0 or not self.hctf_block.training:
            return enhanced
        graph_count = int(batch_index.max().item()) + 1
        random_state = torch.Generator(device="cpu")
        random_state.manual_seed(
            int(self.params.hctf_stochastic_depth_seed)
            + 1009 * forward_counter + block_id
        )
        keep = (
            torch.rand(graph_count, generator=random_state) >= probability
        ).to(device=before.device, dtype=before.dtype)
        return before + (enhanced - before) * keep[batch_index, None] / (1.0 - probability)

    def set_hctf_preference(self, preference) -> None:
        for module in (self.hctf_block, self.hctf_coordinate_equivariant):
            if module is not None and hasattr(module, "set_preference"):
                module.set_preference(preference)

    def set_hctf_structure_gate_override(self, value: Optional[float]) -> None:
        if self.hctf_block is not None and hasattr(
            self.hctf_block, "set_structure_gate_override"
        ):
            self.hctf_block.set_structure_gate_override(value)

    def set_hctf_tree_flow_override(self, value: Optional[str]) -> None:
        if self.hctf_block is not None and hasattr(
            self.hctf_block, "set_tree_flow_override"
        ):
            self.hctf_block.set_tree_flow_override(value)

    def forward(self,
                init_pos: FloatTensor,
                edge_index: LongTensor,
                edge_attr: FloatTensor,
                batch_index: LongTensor,
                graph_edge_index: Optional[LongTensor] = None,
                community_labels: Optional[LongTensor] = None,
                num_sampled_nodes_per_hop: Optional[list[int]] = None,
                num_sampled_edges_per_hop: Optional[list[int]] = None,
                return_hidden: bool = False):
        # num_sampled_nodes_per_hop = num_sampled_nodes_per_hop or [len(init_pos)] * self.total_layers
        # num_sampled_edges_per_hop = num_sampled_edges_per_hop or [len(edge_attr)] * self.total_layers
        inputs = outputs = init_pos
        hidden = None
        tree_cover = None
        stochastic_forward_counter = self._hctf_stochastic_forward_counter
        if (
            self.hctf_block is not None
            and self.hctf_block.training
            and self.params.hctf_stochastic_depth_prob > 0.0
        ):
            self._hctf_stochastic_forward_counter += 1
        if self.hctf_block is not None and graph_edge_index is not None:
            tree_cover = self.hctf_cover.build(
                graph_edge_index=graph_edge_index,
                batch_index=batch_index,
                device=init_pos.device,
                community_labels=community_labels,
            )
        for block_id, block in enumerate(self.block_list):
            if block_id == len(self.block_list) - 1:
                hidden = outputs
            outputs, inputs, edge_index, edge_attr = block(
                node_feat=outputs,
                node_attr=inputs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_index=batch_index,
                num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop=num_sampled_edges_per_hop
            )
            is_coordinate_head = block_id == len(self.block_list) - 1
            use_hctf_here = (
                self.hctf_block is not None
                and not self.hctf_bypass
                and tree_cover is not None
                and not self.params.hctf_coordinate_residual
                and not is_coordinate_head
                and block_id % max(int(self.params.hctf_block_stride), 1) == 0
                and (
                    self.hctf_active_block_ids_override is None
                    or block_id in self.hctf_active_block_ids_override
                )
                and (
                    not self.params.hctf_late_only
                    or block_id == len(self.block_list) - 2
                )
            )
            if use_hctf_here:
                before_hctf = outputs
                enhanced = self.hctf_block(
                    node_feat=outputs,
                    cover=tree_cover,
                    graph_edge_index=graph_edge_index,
                    community_labels=community_labels,
                    batch_index=batch_index,
                )
                if self.hctf_stage_adapters is not None:
                    enhanced = enhanced + self.hctf_stage_adapters[block_id](
                        enhanced - before_hctf
                    )
                if self.hctf_graph_size_gate is not None:
                    enhanced = self.hctf_graph_size_gate(
                        before_hctf, enhanced, batch_index,
                    )
                outputs = self._apply_hctf_stochastic_depth(
                    outputs, enhanced, batch_index, block_id,
                    stochastic_forward_counter,
                )

        if (
            self.params.hctf_coordinate_residual
            and self.hctf_block is not None
            and not self.hctf_bypass
            and tree_cover is not None
            and hidden is not None
        ):
            outputs = self._apply_hctf_coordinate_residual(
                base_pos=outputs,
                hidden=hidden,
                tree_cover=tree_cover,
                graph_edge_index=graph_edge_index,
                community_labels=community_labels,
                batch_index=batch_index,
            )

        if (
            self.hctf_conflict_head is not None
            and self.hctf_conflict_head_enabled
            and not self.hctf_bypass
            and graph_edge_index is not None
            and hidden is not None
        ):
            outputs = self.hctf_conflict_head(
                pos=outputs,
                hidden=hidden,
                edge_index=graph_edge_index,
                community_labels=community_labels,
                batch_index=batch_index,
            )

        # TODO: Use input as initial layout
        #   outputs += normalized_inputs
        #   outputs = parametric_scaling(outputs)

        if return_hidden:
            return outputs, hidden
        return outputs  # , edge_index, edge_attr

    @property
    def total_layers(self):
        return self.block_list[-1].next_layer_index


Generator.__annotations__.clear()
