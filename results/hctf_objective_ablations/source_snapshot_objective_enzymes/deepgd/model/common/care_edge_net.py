from torch import nn
import torch
from torch.utils.checkpoint import checkpoint


class CommunityAdaptiveResidualEdgeNet(nn.Module):
    """
    Wrap an edge network with lightweight community-relation residual experts.

    The wrapped base edge network still predicts the standard NNConv matrix.
    CARE adds a low-rank residual matrix selected by same/adjacent/far community
    indicators that are stored in the leading raw edge-attribute channels.
    """

    def __init__(
        self,
        base_edge_net: nn.Module,
        in_dim: int,
        out_dim: int,
        rank: int = 2,
        scale: float = 0.10,
        same_index: int = -1,
        adj_index: int = -1,
        far_index: int = -1,
        boundary_index: int = -1,
        use_boundary_gate: bool = True,
        boundary_damp: float = 0.50,
        checkpoint_base: bool = True,
    ):
        super().__init__()

        self.base_edge_net = base_edge_net
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = max(1, int(rank))
        self.scale = float(scale)

        self.same_index = int(same_index)
        self.adj_index = int(adj_index)
        self.far_index = int(far_index)
        self.boundary_index = int(boundary_index)
        self.use_boundary_gate = bool(use_boundary_gate)
        self.boundary_damp = float(boundary_damp)
        self.checkpoint_base = bool(checkpoint_base)

        self.left = nn.Parameter(torch.empty(3, self.in_dim, self.rank))
        self.right = nn.Parameter(torch.zeros(3, self.rank, self.out_dim))
        nn.init.xavier_uniform_(self.left)

    @staticmethod
    def _column(edge_feat: torch.Tensor, index: int) -> torch.Tensor:
        if index < 0 or index >= edge_feat.size(-1):
            return edge_feat.new_zeros(edge_feat.size(0))
        return edge_feat[:, index].clamp(0.0, 1.0)

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_base and self.training:
            base = checkpoint(
                self.base_edge_net,
                edge_feat,
                use_reentrant=False,
            )
        else:
            base = self.base_edge_net(edge_feat)

        same = self._column(edge_feat, self.same_index)
        adj = self._column(edge_feat, self.adj_index)
        far = self._column(edge_feat, self.far_index)

        if self.use_boundary_gate:
            boundary = self._column(edge_feat, self.boundary_index)
            same = same * (1.0 - boundary)
            inter_gate = (1.0 - self.boundary_damp * boundary).clamp(0.0, 1.0)
            adj = adj * inter_gate
            far = far * inter_gate

        weights = torch.stack([same, adj, far], dim=-1)
        adapters = torch.einsum("kir,kro->kio", self.left, self.right)
        adapters = adapters.reshape(3, self.in_dim * self.out_dim)

        return torch.addmm(base, weights, adapters, beta=1.0, alpha=self.scale)
