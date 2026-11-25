from .mlp import MLP
from .nnconv_basic_layer import NNConvBasicLayer
from .care_edge_net import CommunityAdaptiveResidualEdgeNet

from dataclasses import dataclass, field
from typing import Optional
from attrs import define, frozen

import torch
from torch import nn, jit, FloatTensor, LongTensor


@define(kw_only=True, eq=False, repr=False, slots=False)
class NNConvLayer(nn.Module):

    @dataclass(kw_only=True)
    class NNConvConfig:
        dense: bool = False
        bn: Optional[str] = "pyg_batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        residual: bool = False
        aggr: str = "mean"
        root_weight: bool = True

    @dataclass(kw_only=True)
    class EdgeNetConfig:
        hidden_dims: list[int] = field(default_factory=list)
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = None
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = True
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

    layer_index: int
    params: NNConvBasicLayer.Params
    nnconv_config: NNConvConfig = NNConvConfig()
    edge_net_config: EdgeNetConfig = EdgeNetConfig()

    def __attrs_post_init__(self):
        super().__init__()

        base_edge_net = MLP(
            params=MLP.Params(
                in_dim=self.params.edge_feat_dim,
                out_dim=self.params.in_dim * self.params.out_dim,
                hidden_dims=self.edge_net_config.hidden_dims,
            ),
            config=MLP.Config(
                hidden_act=self.edge_net_config.hidden_act,
                out_act=self.edge_net_config.out_act,
                bn=self.edge_net_config.bn,
                dp=self.edge_net_config.dp,
                residual=self.edge_net_config.residual
            )
        )

        edge_net = base_edge_net
        if self.edge_net_config.care_enabled:
            edge_net = CommunityAdaptiveResidualEdgeNet(
                base_edge_net=base_edge_net,
                in_dim=self.params.in_dim,
                out_dim=self.params.out_dim,
                rank=self.edge_net_config.care_rank,
                scale=self.edge_net_config.care_scale,
                same_index=self.edge_net_config.care_same_index,
                adj_index=self.edge_net_config.care_adj_index,
                far_index=self.edge_net_config.care_far_index,
                boundary_index=self.edge_net_config.care_boundary_index,
                use_boundary_gate=self.edge_net_config.care_use_boundary_gate,
                boundary_damp=self.edge_net_config.care_boundary_damp,
                checkpoint_base=self.edge_net_config.care_checkpoint,
            )

        self.nnconv_layer: NNConvBasicLayer = NNConvBasicLayer(
            layer_index=self.layer_index,
            params=self.params,
            config=NNConvBasicLayer.Config(
                edge_net=edge_net,
                dense=self.nnconv_config.dense,
                bn=self.nnconv_config.bn,
                act=self.nnconv_config.act,
                dp=self.nnconv_config.dp,
                residual=self.nnconv_config.residual,
                aggr=self.nnconv_config.aggr,
                root_weight=self.nnconv_config.root_weight
            )
        )

    def forward(self, *,
                node_feat: FloatTensor,
                edge_feat: FloatTensor,
                edge_index: LongTensor,
                batch_index: LongTensor,
                num_sampled_nodes_per_hop: list[int],
                num_sampled_edges_per_hop: list[int]) -> tuple[FloatTensor, LongTensor, FloatTensor]:
        return self.nnconv_layer(
            node_feat=node_feat,
            edge_feat=edge_feat,
            edge_index=edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop
        )


NNConvLayer.__annotations__.clear()
