from __future__ import annotations

import torch


def objective_space_repulsion(objectives, sigma: float = 1.0):
    if objectives.size(0) <= 1:
        return objectives.sum() * 0.0
    distances_sq = torch.pdist(objectives, p=2).square()
    return torch.exp(-distances_sq / (2.0 * max(float(sigma), 1e-8) ** 2)).mean()


def refine_pareto_set(
    positions,
    preferences,
    objective_fn,
    steps: int = 5,
    lr: float = 0.01,
    repulsion_weight: float = 0.05,
    sigma: float = 1.0,
    anchor_weight: float = 0.10,
    grad_clip: float = 1.0,
):
    if int(steps) <= 0:
        return positions.detach()
    with torch.enable_grad():
        anchor = positions.detach()
        values = positions.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([values], lr=float(lr))
        preferences = preferences.to(device=values.device, dtype=values.dtype)
        preferences = preferences / preferences.sum(dim=1, keepdim=True).clamp_min(1e-8)
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            objectives = objective_fn(values)
            scalarized = (preferences * objectives).sum(dim=1).mean()
            repulsion = objective_space_repulsion(objectives, sigma=sigma)
            anchor_loss = (values - anchor).square().sum(dim=2).mean()
            loss = scalarized + float(repulsion_weight) * repulsion + float(anchor_weight) * anchor_loss
            if not torch.isfinite(loss):
                break
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([values], float(grad_clip))
            optimizer.step()
    return values.detach()
