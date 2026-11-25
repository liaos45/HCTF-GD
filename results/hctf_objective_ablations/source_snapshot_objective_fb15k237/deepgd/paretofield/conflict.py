from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _flatten_gradients(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = []
    for parameter, gradient in zip(parameters, gradients):
        flat.append(torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1))
    return torch.cat(flat) if flat else torch.empty(0, device=loss.device)


def _assign_flat_gradient(parameters, flat_gradient):
    cursor = 0
    for parameter in parameters:
        count = parameter.numel()
        gradient = flat_gradient[cursor : cursor + count].view_as(parameter)
        parameter.grad = gradient.detach().clone()
        cursor += count


def _pcgrad(first, second, eps=1e-12):
    dot = torch.dot(first, second)
    first_projected = first
    second_projected = second
    if dot < 0:
        first_projected = first - dot * second / second.square().sum().clamp_min(eps)
        second_projected = second - dot * first / first.square().sum().clamp_min(eps)
    return 0.5 * (first_projected + second_projected)


def _config(first, second, eps=1e-12):
    first_norm = first.norm().clamp_min(eps)
    second_norm = second.norm().clamp_min(eps)
    first_unit = first / first_norm
    second_unit = second / second_norm
    cosine = torch.dot(first_unit, second_unit)
    if cosine > 1.0 - 1e-6:
        return 0.5 * (first + second)
    first_orthogonal = first_unit - cosine * second_unit
    second_orthogonal = second_unit - cosine * first_unit
    direction = first_orthogonal / first_orthogonal.norm().clamp_min(eps)
    direction = direction + second_orthogonal / second_orthogonal.norm().clamp_min(eps)
    direction = direction / direction.norm().clamp_min(eps)
    scale = torch.dot(first, direction) + torch.dot(second, direction)
    if scale <= eps:
        return _pcgrad(first, second, eps=eps)
    return scale * direction


def apply_conflict_free_gradients(losses, parameters, method="config"):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if len(losses) != 2:
        raise ValueError("lightweight conflict-free update expects exactly two losses")
    first = _flatten_gradients(losses[0], parameters, retain_graph=True)
    second = _flatten_gradients(losses[1], parameters, retain_graph=True)
    denominator = first.norm() * second.norm()
    cosine = torch.dot(first, second) / denominator.clamp_min(1e-12)
    if method == "pcgrad":
        combined = _pcgrad(first, second)
    elif method == "config":
        combined = _config(first, second)
    else:
        raise ValueError(f"unknown conflict-free method: {method}")
    _assign_flat_gradient(parameters, combined)
    return {
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": float(cosine.detach().cpu() < 0),
        "field_gradient_norm": float(first.norm().detach().cpu()),
        "pareto_gradient_norm": float(second.norm().detach().cpu()),
    }


@dataclass
class GradientMonitor:
    totals: dict = field(default_factory=lambda: {
        "gradient_cosine": 0.0,
        "gradient_conflict": 0.0,
        "field_gradient_norm": 0.0,
        "pareto_gradient_norm": 0.0,
    })
    count: int = 0

    def update(self, values):
        for key in self.totals:
            self.totals[key] += float(values.get(key, 0.0))
        self.count += 1

    def mean(self):
        return {key: value / max(self.count, 1) for key, value in self.totals.items()}
