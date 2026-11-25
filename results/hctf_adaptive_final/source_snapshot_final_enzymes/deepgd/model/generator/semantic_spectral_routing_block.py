# deepgd/model/generator/semantic_spectral_routing_block.py
# -*- coding: utf-8 -*-

"""
SSRB: Semantic-Spectral Routing Block

作用：
在 DeepGD Generator 的中间 block 之间注入社区上下文和谱结构信息。

输入：
    node_feat: 当前 DeepGD 隐藏状态 [N, hidden_dim]
    pos: 初始布局坐标或当前坐标 [N, 2]
    batch_index: batch 图编号 [N]
    community_labels: 社区标签 [N]
    spectral_pe: Laplacian 谱位置编码 [N, spectral_dim]

输出：
    enhanced node_feat [N, hidden_dim]
"""

from typing import Optional

import torch
from torch import nn, FloatTensor, LongTensor


class SemanticSpectralRoutingBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        spectral_dim: int = 4,
        pos_dim: int = 2,
        mlp_hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.spectral_dim = spectral_dim
        self.pos_dim = pos_dim

        self.community_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        fusion_dim = hidden_dim * 4

        self.update_mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def _safe_spectral_pe(
        self,
        node_feat: FloatTensor,
        spectral_pe: Optional[FloatTensor],
    ) -> FloatTensor:
        if spectral_pe is None:
            return torch.zeros(
                node_feat.shape[0],
                self.spectral_dim,
                dtype=node_feat.dtype,
                device=node_feat.device,
            )

        spectral_pe = spectral_pe.to(device=node_feat.device, dtype=node_feat.dtype)

        if spectral_pe.shape[1] == self.spectral_dim:
            return spectral_pe

        if spectral_pe.shape[1] > self.spectral_dim:
            return spectral_pe[:, : self.spectral_dim]

        pad_dim = self.spectral_dim - spectral_pe.shape[1]
        pad = torch.zeros(
            spectral_pe.shape[0],
            pad_dim,
            dtype=spectral_pe.dtype,
            device=spectral_pe.device,
        )
        return torch.cat([spectral_pe, pad], dim=-1)

    def _safe_pos(
        self,
        node_feat: FloatTensor,
        pos: Optional[FloatTensor],
    ) -> FloatTensor:
        if pos is None:
            return torch.zeros(
                node_feat.shape[0],
                self.pos_dim,
                dtype=node_feat.dtype,
                device=node_feat.device,
            )

        pos = pos.to(device=node_feat.device, dtype=node_feat.dtype)

        if pos.shape[1] == self.pos_dim:
            return pos

        if pos.shape[1] > self.pos_dim:
            return pos[:, : self.pos_dim]

        pad_dim = self.pos_dim - pos.shape[1]
        pad = torch.zeros(
            pos.shape[0],
            pad_dim,
            dtype=pos.dtype,
            device=pos.device,
        )
        return torch.cat([pos, pad], dim=-1)

    def _community_pool(
        self,
        node_feat: FloatTensor,
        batch_index: LongTensor,
        community_labels: Optional[LongTensor],
    ) -> FloatTensor:
        """
        对每张图内部的每个社区做 mean pooling，然后广播回节点。
        如果没有 community_labels，则退化为 graph-level pooling。
        """
        device = node_feat.device
        batch_index = batch_index.long().to(device)

        comm_context = torch.zeros_like(node_feat)

        if community_labels is None:
            for gid in torch.unique(batch_index):
                mask = batch_index == gid
                if mask.sum() == 0:
                    continue
                mean_feat = node_feat[mask].mean(dim=0, keepdim=True)
                comm_context[mask] = mean_feat
            return comm_context

        community_labels = community_labels.long().to(device)

        for gid in torch.unique(batch_index):
            graph_mask = batch_index == gid
            if graph_mask.sum() == 0:
                continue

            node_indices_g = torch.where(graph_mask)[0]
            labels_g = community_labels[graph_mask]

            for cid in torch.unique(labels_g):
                local_mask = labels_g == cid
                if local_mask.sum() == 0:
                    continue

                global_indices = node_indices_g[local_mask]
                mean_feat = node_feat[global_indices].mean(dim=0, keepdim=True)
                comm_context[global_indices] = mean_feat

        return comm_context

    def forward(
        self,
        node_feat: FloatTensor,
        pos: Optional[FloatTensor],
        batch_index: LongTensor,
        community_labels: Optional[LongTensor] = None,
        spectral_pe: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        spectral_pe = self._safe_spectral_pe(node_feat, spectral_pe)
        pos = self._safe_pos(node_feat, pos)

        community_context = self._community_pool(
            node_feat=node_feat,
            batch_index=batch_index,
            community_labels=community_labels,
        )

        community_context = self.community_encoder(community_context)
        spectral_context = self.spectral_encoder(spectral_pe)
        pos_context = self.pos_encoder(pos)

        fusion = torch.cat(
            [
                node_feat,
                community_context,
                spectral_context,
                pos_context,
            ],
            dim=-1,
        )

        gate = self.gate_mlp(fusion)
        delta_h = self.update_mlp(fusion)

        out = node_feat + gate * delta_h
        out = self.out_norm(out)

        return out