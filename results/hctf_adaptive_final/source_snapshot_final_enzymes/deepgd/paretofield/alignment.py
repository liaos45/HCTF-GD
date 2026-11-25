from __future__ import annotations

import torch


def similarity_align(target, reference, allow_reflection: bool = True, eps: float = 1e-8):
    """Align target to reference under translation, scale, and O(2)/SO(2)."""
    if target.shape != reference.shape or target.ndim != 2 or target.size(-1) != 2:
        raise ValueError("target and reference must both have shape [N, 2]")
    target_centered = target - target.mean(dim=0, keepdim=True)
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    covariance = target_centered.transpose(0, 1) @ reference_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if not allow_reflection and torch.det(rotation) < 0:
        correction = torch.eye(2, dtype=target.dtype, device=target.device)
        correction[-1, -1] = -1.0
        rotation = u @ correction @ vh
    rotated = target_centered @ rotation
    scale = (rotated * reference_centered).sum() / rotated.square().sum().clamp_min(eps)
    return scale * rotated + reference.mean(dim=0, keepdim=True)


def similarity_aligned_mse(
    prediction,
    target,
    batch_index,
    graph_mask=None,
    graph_weights=None,
    allow_reflection: bool = True,
):
    batch_index = batch_index.long()
    graph_ids = batch_index.unique(sorted=True)
    if graph_mask is None:
        graph_mask = torch.ones(graph_ids.numel(), dtype=torch.bool, device=prediction.device)
    if graph_weights is None:
        graph_weights = torch.ones(graph_ids.numel(), dtype=prediction.dtype, device=prediction.device)
    losses = []
    weights = []
    for local_id, graph_id in enumerate(graph_ids):
        if local_id >= graph_mask.numel() or not bool(graph_mask[local_id]):
            continue
        node_mask = batch_index == graph_id
        aligned = similarity_align(
            target[node_mask],
            prediction[node_mask].detach(),
            allow_reflection=allow_reflection,
        ).detach()
        losses.append((prediction[node_mask] - aligned).square().sum(dim=1).mean())
        weights.append(graph_weights[local_id].clamp_min(0.0))
    if not losses:
        return prediction.sum() * 0.0
    losses = torch.stack(losses)
    weights = torch.stack(weights).to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)
