"""Exact componentwise safe selection across trained HCTF candidates.

For every test graph, candidate layouts are Procrustes-aligned to the matched
baseline and interpolated along a fixed line-search grid.  A trial is accepted
only when none of the selected audit metrics deteriorates (up to numerical
tolerance) and at least one metric improves.  The baseline is therefore an
explicit per-graph fallback rather than an assumed average-case safeguard.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_cser_spectral_mog_end2end as train  # noqa: E402
from deepgd.metrics import Stress  # noqa: E402
from evaluate_extra_metrics import (  # noqa: E402
    community_metrics,
    detect_communities,
    edge_crossing_count,
    edge_length_variance,
    independent_visual_metrics,
    node_occlusion_rate,
    unique_undirected_edges,
)


METRICS = [
    "batch_stress",
    "edge_crossing",
    "edge_crossing_norm",
    "edge_length_variance",
    "edge_length_cv",
    "node_occlusion_rate",
    "community_compactness",
    "community_separation",
    "community_sep_comp_ratio",
    "community_silhouette",
]
HOLDOUT_METRICS = [
    "angular_resolution_deg",
    "angular_resolution_ratio",
    "crossing_angle_deg",
    "crossing_angle_p05_deg",
    "node_edge_clearance",
    "node_edge_clearance_p05",
    "neighborhood_preservation_k5",
]
ALL_METRICS = METRICS + HOLDOUT_METRICS
LOWER_IS_BETTER = {
    "batch_stress",
    "edge_crossing",
    "edge_crossing_norm",
    "edge_length_variance",
    "edge_length_cv",
    "node_occlusion_rate",
    "community_compactness",
}
SCORE_FLOORS = {
    "batch_stress": 1.0,
    "edge_crossing": 1.0,
    "edge_crossing_norm": 0.01,
    "edge_length_variance": 0.01,
    "edge_length_cv": 0.05,
    "node_occlusion_rate": 0.001,
    "community_compactness": 0.05,
    "community_separation": 0.05,
    "community_sep_comp_ratio": 0.05,
    "community_silhouette": 0.05,
    "angular_resolution_deg": 1.0,
    "angular_resolution_ratio": 0.01,
    "crossing_angle_deg": 1.0,
    "crossing_angle_p05_deg": 1.0,
    "node_edge_clearance": 0.01,
    "node_edge_clearance_p05": 0.001,
    "neighborhood_preservation_k5": 0.01,
}
ABS_TOLERANCE = {
    "batch_stress": 1e-6,
    "edge_crossing": 0.0,
    "edge_crossing_norm": 1e-12,
    "edge_length_variance": 1e-9,
    "edge_length_cv": 1e-9,
    "node_occlusion_rate": 0.0,
    "community_compactness": 1e-8,
    "community_separation": 1e-8,
    "community_sep_comp_ratio": 1e-8,
    "community_silhouette": 1e-8,
    "angular_resolution_deg": 1e-8,
    "angular_resolution_ratio": 1e-10,
    "crossing_angle_deg": 1e-8,
    "crossing_angle_p05_deg": 1e-8,
    "node_edge_clearance": 1e-10,
    "node_edge_clearance_p05": 1e-10,
    "neighborhood_preservation_k5": 1e-10,
}


def objective_metric_weights(spec: str) -> dict[str, float]:
    if not str(spec).strip() or str(spec).strip().lower() == "legacy":
        return {metric: 1.0 for metric in METRICS}
    values = [float(value.strip()) for value in str(spec).split(",") if value.strip()]
    if len(values) != 3 or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError("objective preference must be three non-negative positive-sum values")
    geometry, visual, community = (value / sum(values) for value in values)
    groups = {
        "batch_stress": (geometry, 3),
        "edge_length_variance": (geometry, 3),
        "edge_length_cv": (geometry, 3),
        "edge_crossing": (visual, 3),
        "edge_crossing_norm": (visual, 3),
        "node_occlusion_rate": (visual, 3),
        "community_compactness": (community, 4),
        "community_separation": (community, 4),
        "community_sep_comp_ratio": (community, 4),
        "community_silhouette": (community, 4),
    }
    return {metric: weight / count for metric, (weight, count) in groups.items()}


def monotone_refinement_alphas(alphas: list[float], levels: int) -> list[float]:
    """Return midpoint refinements without removing any original alpha.

    The caller first runs the original grid unchanged, then considers only the
    returned points under an incumbent-dominance rule.  Consequently refinement
    cannot alter the legacy result unless every audited metric is preserved and
    at least one improves.
    """
    if levels < 0:
        raise ValueError("refinement levels cannot be negative")
    original = {float(value) for value in alphas}
    grid = sorted(original | {0.0, 1.0})
    for _ in range(levels):
        midpoints = {
            0.5 * (left + right)
            for left, right in zip(grid[:-1], grid[1:])
            if right > left
        }
        grid = sorted(set(grid) | midpoints)
    return [value for value in grid if value not in original and 0.0 < value <= 1.0]


def monotone_extrapolation_alphas(step: float, steps: int) -> list[float]:
    """Return proposal-line points beyond alpha=1 for audited line search."""
    if steps < 0:
        raise ValueError("extrapolation steps cannot be negative")
    if step <= 0.0:
        raise ValueError("extrapolation step must be positive")
    return [1.0 + step * index for index in range(1, steps + 1)]


def portfolio_alpha_plan(
    candidate_count: int,
    core_candidate_count: int,
    base_alphas: list[float],
    refinement_alphas: list[float],
) -> tuple[int, list[list[float]]]:
    """Plan monotone search without letting extra candidates replace the core unfairly.

    Core candidates participate in the unchanged first-stage grid and receive
    only new refinement points later. Extra candidates are introduced only in
    the incumbent-dominance stage, where their original and new alpha points
    must dominate the already selected core result.
    """
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if core_candidate_count < 0 or core_candidate_count > candidate_count:
        raise ValueError("core_candidate_count must be in [0, candidate_count]")
    effective_core = candidate_count if core_candidate_count == 0 else core_candidate_count
    base_nonzero = [float(alpha) for alpha in base_alphas if float(alpha) > 0.0]
    plans = []
    for index in range(candidate_count):
        values = refinement_alphas if index < effective_core else base_nonzero + refinement_alphas
        plans.append(sorted(set(float(value) for value in values)))
    return effective_core, plans


def baseline_chain_steps_were_searched(
    core_candidate_count: int,
    base_alphas: list[float],
    refinement_alpha_plan: list[list[float]],
    chain_steps: list[float],
) -> bool:
    """Whether a chain beginning at baseline repeats earlier line-search trials."""
    core_alphas = {float(alpha) for alpha in base_alphas if alpha > 0.0}
    for candidate_index, refinement_alphas in enumerate(refinement_alpha_plan):
        searched = {float(alpha) for alpha in refinement_alphas}
        if candidate_index < core_candidate_count:
            searched.update(core_alphas)
        if any(float(step) not in searched for step in chain_steps):
            return False
    return True


def refinement_trial_schedule(
    alpha_plan: list[list[float]],
    prefix_alpha: float = 0.0,
) -> list[tuple[int, float]]:
    """Keep an existing search as a prefix when extending its alpha range.

    Historical runs visit all alpha values for one candidate before moving to
    the next.  With that order, adding a larger alpha to an early candidate can
    make a later candidate's old result inaccessible: the two safe layouts may
    be Pareto-incomparable.  When a prefix cutoff is supplied, preserve the
    historical candidate-major order through that cutoff.  Visit new alpha
    values across candidates in ascending order, so a larger grid only appends
    trials and can never undo the locked search result.
    """
    if not np.isfinite(prefix_alpha) or prefix_alpha < 0.0:
        raise ValueError("prefix_alpha must be finite and non-negative")
    if prefix_alpha == 0.0:
        return [
            (candidate_index, float(alpha))
            for candidate_index, alphas in enumerate(alpha_plan)
            for alpha in alphas
        ]
    prefix = [
        (candidate_index, float(alpha))
        for candidate_index, alphas in enumerate(alpha_plan)
        for alpha in alphas
        if float(alpha) <= prefix_alpha
    ]
    extension_values = sorted({
        float(alpha)
        for alphas in alpha_plan
        for alpha in alphas
        if float(alpha) > prefix_alpha
    })
    extension = [
        (candidate_index, alpha)
        for alpha in extension_values
        for candidate_index, alphas in enumerate(alpha_plan)
        if alpha in alphas
    ]
    return prefix + extension


def parse_pairwise_mixture_weights(spec: str) -> list[float]:
    """Parse deterministic convex weights for refinement-only proposal mixtures."""
    if not str(spec).strip():
        return []
    weights = sorted({float(value.strip()) for value in str(spec).split(",") if value.strip()})
    if any(not 0.0 < weight < 1.0 for weight in weights):
        raise ValueError("pairwise mixture weights must lie strictly between 0 and 1")
    return weights


def parse_pareto_chain_steps(spec: str) -> list[float]:
    """Parse interpolation fractions used by monotone incumbent chaining."""
    steps = sorted({float(value.strip()) for value in str(spec).split(",") if value.strip()})
    if not steps or any(not 0.0 < step <= 1.0 for step in steps):
        raise ValueError("Pareto-chain steps must be a non-empty grid in (0,1]")
    return steps


def pairwise_aligned_proposals(
    aligned_candidates: list[tuple[str, torch.Tensor]],
    seed_pos: torch.Tensor,
    weights: list[float],
) -> list[tuple[str, torch.Tensor]]:
    """Create convex combinations of already Procrustes-aligned directions.

    These virtual proposals are intended only for the incumbent-dominance
    refinement stage.  Combining displacement directions can cancel a metric
    conflict between two trained proposals while the downstream componentwise
    guard still supplies the exact non-regression guarantee.
    """
    mixtures: list[tuple[str, torch.Tensor]] = []
    for left_index, (left_name, left_pos) in enumerate(aligned_candidates):
        left_delta = left_pos - seed_pos
        for right_name, right_pos in aligned_candidates[left_index + 1 :]:
            right_delta = right_pos - seed_pos
            for weight in weights:
                mixed_pos = seed_pos + (1.0 - weight) * left_delta + weight * right_delta
                mixtures.append(
                    (
                        f"mix[{left_name},{right_name};w={weight:g}]",
                        mixed_pos,
                    )
                )
    return mixtures


def local_stress_repair_candidate(
    seed_pos: torch.Tensor,
    record: dict,
    steps: int,
    lr: float,
) -> torch.Tensor:
    """Make a graph-local, stress-only direction without changing model weights.

    This optional direction uses no held-out readability endpoint.  It is
    subsequently subject to the same exact componentwise audit as all other
    extra candidates; default ``steps=0`` leaves historical runs unchanged.
    """
    if steps <= 0:
        raise ValueError("local stress repair requires positive steps")
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("local stress repair learning rate must be positive")
    anchor = seed_pos.detach()
    position = anchor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([position], lr=float(lr))
    pair_count = max(int(record["perm_index"].size(1)), 1)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            stress = train.local_deepgd_stress_tensor(
                position, record["perm_index"], record["apsp_attr"]
            ) / pair_count
            anchor_loss = (position - anchor).square().sum(dim=1).mean()
            loss = stress + 0.10 * anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite local stress repair loss")
            loss.backward()
            if position.grad is None or not torch.isfinite(position.grad).all():
                raise FloatingPointError("non-finite local stress repair gradient")
            torch.nn.utils.clip_grad_norm_([position], 1.0)
            optimizer.step()
    return train.local_rescale_by_stress(
        position.detach(), record["perm_index"], record["apsp_attr"]
    )


def local_stress_cv_candidate(
    seed_pos: torch.Tensor,
    record: dict,
    step_scale: float,
) -> torch.Tensor:
    """One shared descent direction for rescaled stress and edge-length CV.

    Equal-length normalized gradients have a negative-average direction that
    locally decreases both objectives unless they are exactly opposed.  The
    exact SafeBlend gate still decides whether *all* audited metrics are safe.
    No held-out visual endpoint participates in generating this direction.
    """
    if not np.isfinite(step_scale) or step_scale <= 0.0:
        raise ValueError("local stress/CV step scale must be finite and positive")
    if seed_pos.size(0) <= 2:
        return seed_pos.detach().clone()
    with torch.enable_grad():
        position = seed_pos.detach().clone().requires_grad_(True)
        rescaled = train.local_rescale_by_stress(
            position, record["perm_index"], record["apsp_attr"]
        )
        stress = train.local_deepgd_stress_tensor(
            rescaled, record["perm_index"], record["apsp_attr"]
        )
        cv = train.mtpf_edge_length_cv_tensor(rescaled, record["edge_index"])
        stress_grad = torch.autograd.grad(
            stress, position, retain_graph=True
        )[0]
        cv_grad = torch.autograd.grad(cv, position)[0]
    if not torch.isfinite(stress_grad).all() or not torch.isfinite(cv_grad).all():
        return seed_pos.detach().clone()
    stress_norm = torch.linalg.vector_norm(stress_grad)
    cv_norm = torch.linalg.vector_norm(cv_grad)
    if stress_norm <= 1e-12 or cv_norm <= 1e-12:
        return seed_pos.detach().clone()
    direction = -0.5 * (
        stress_grad / stress_norm + cv_grad / cv_norm
    )
    max_node_norm = torch.linalg.vector_norm(direction, dim=1).max()
    if max_node_norm <= 1e-8:
        return seed_pos.detach().clone()
    edges = unique_undirected_edges(record["edge_index"])
    if not edges:
        return seed_pos.detach().clone()
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=seed_pos.device)
    lengths = torch.linalg.vector_norm(
        seed_pos[edge_tensor[:, 0]] - seed_pos[edge_tensor[:, 1]], dim=1
    )
    typical_length = lengths.median().clamp_min(1e-9)
    displacement = float(step_scale) * typical_length * direction / max_node_norm
    return (seed_pos.detach() + displacement.detach()).detach()


def checkpoint_paths(dataset: str) -> tuple[Path, list[tuple[str, Path]]]:
    baseline = ROOT / "checkpoints" / f"{dataset}_hctf_all8_baseline_pilot_best.pt"
    candidates = [
        (
            "original",
            ROOT / "checkpoints" / f"{dataset}_hctf_all8_hctf_pilot_best.pt",
        ),
        (
            "conservative",
            ROOT
            / "checkpoints"
            / f"{dataset}_hctf_conservative_t2_h16_d10_s008_b2_best.pt",
        ),
        (
            "balanced",
            ROOT
            / "checkpoints"
            / f"{dataset}_hctf_balanced_t2_h16_d05_s010_b2_best.pt",
        ),
    ]
    return baseline, [(name, path) for name, path in candidates if path.exists()]


def resolve_checkpoint_paths(
    dataset: str,
    baseline_override: str = "",
    candidate_specs: str = "",
) -> tuple[Path, list[tuple[str, Path]]]:
    baseline, candidates = checkpoint_paths(dataset)
    if baseline_override:
        baseline = Path(baseline_override.format(dataset=dataset))
        if not baseline.is_absolute():
            baseline = ROOT / baseline
    if candidate_specs:
        candidates = []
        for index, item in enumerate(candidate_specs.split(";"), start=1):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, raw_path = item.split("=", 1)
            else:
                name, raw_path = f"candidate_{index}", item
            path = Path(raw_path.strip().format(dataset=dataset))
            if not path.is_absolute():
                path = ROOT / path
            candidates.append((name.strip(), path))
    return baseline, candidates


def namespace_from_checkpoint(path: Path) -> argparse.Namespace:
    checkpoint = train.torch_load_trusted(path, map_location="cpu")
    args = argparse.Namespace(**copy.deepcopy(checkpoint["args"]))
    compatibility_defaults = {
        "freeze_backbone_for_hctf": False,
        "freeze_hctf_proposal_for_risk_gate": False,
        "hctf_dominance_only": False,
        "hctf_exact_risk_gate": False,
        "hctf_exact_gate_components": "1,4,5",
        "hctf_exact_gate_tolerance": 0.0,
        "hctf_exact_gate_temperature": 0.10,
        "lambda_hctf_dominance": 0.0,
        "lambda_hctf_anchor": 0.0,
        "lambda_hctf_crossing_barrier": 0.0,
        "lambda_hctf_exact_gate": 0.0,
        "hctf_adaptive_structure_gate": False,
        "hctf_structure_gate_init": 0.05,
        "lambda_hctf_structure_gate_cost": 0.0,
        "lambda_hctf_structure_gate_supervision": 0.0,
        "hctf_structure_gate_components": "0,4,9",
        "hctf_structure_gate_tolerance": 0.0,
        "hctf_structure_gate_temperature": 0.10,
    }
    for name, value in compatibility_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    args.mode = "eval"
    args.ckpt = str(path)
    args.current_epoch = int(checkpoint.get("epoch", 0))
    return args


def load_model(path: Path, device: str):
    args = namespace_from_checkpoint(path)
    model = train.build_deepgd_model(args, device)
    train.load_checkpoint(str(path), model, None, device)
    model.eval()
    return args, model


def configure_evaluation_device(device_override: str, deterministic_eval: bool) -> str:
    """Resolve the device, using single-threaded CPU for reproducible auditing."""
    if deterministic_eval:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        return "cpu"
    return train.get_device() if device_override == "auto" else device_override


def as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def measure_layout(pos, record, labels, include_holdout: bool = True) -> tuple[torch.Tensor, dict]:
    pos = train.local_rescale_by_stress(
        pos,
        record["perm_index"],
        record["apsp_attr"],
    )
    pos_cpu = pos.detach().cpu()
    edge_index = record["edge_index"]
    crossing, crossing_norm = edge_crossing_count(pos_cpu, edge_index)
    edge_variance, edge_cv = edge_length_variance(pos_cpu, edge_index)
    occlusion, occlusion_pairs = node_occlusion_rate(pos_cpu, edge_index=edge_index)
    metrics = {
        "batch_stress": as_float(
            train.local_deepgd_stress(
                pos_cpu,
                record["perm_index"],
                record["apsp_attr"],
            )
        ),
        "edge_crossing": float(crossing),
        "edge_crossing_norm": float(crossing_norm),
        "edge_length_variance": float(edge_variance),
        "edge_length_cv": float(edge_cv),
        "node_occlusion_rate": float(occlusion),
        "node_occlusion_pairs": float(occlusion_pairs),
        "num_nodes": int(record["num_nodes"]),
        "num_edges": int(len(unique_undirected_edges(edge_index))),
    }
    metrics.update(community_metrics(pos_cpu, labels))
    # Hold-out visual metrics are intentionally absent from METRICS and hence
    # cannot influence interpolation, acceptance, or fallback decisions.
    if include_holdout:
        metrics.update(independent_visual_metrics(pos_cpu, edge_index))
    return pos, metrics


def attach_postselection_holdout(metrics: dict, pos: torch.Tensor, record: dict) -> dict:
    """Evaluate the seven independent endpoints only for a selected layout."""
    if all(metric in metrics for metric in HOLDOUT_METRICS):
        return metrics
    return {
        **metrics,
        **independent_visual_metrics(pos.detach().cpu(), record["edge_index"]),
    }


def improvement(seed: dict, trial: dict, metric: str) -> float:
    scale = max(abs(float(seed[metric])), SCORE_FLOORS[metric])
    if metric in LOWER_IS_BETTER:
        return (float(seed[metric]) - float(trial[metric])) / scale
    return (float(trial[metric]) - float(seed[metric])) / scale


def meets_minimum_total_gain(score: float, minimum_total_gain: float) -> bool:
    """Exclude finite-precision-only changes without altering the audit gate."""
    return bool(np.isfinite(score) and score >= minimum_total_gain)


def is_componentwise_safe(
    seed: dict,
    trial: dict,
    relative_tolerance: float = 0.0,
    metrics: list[str] | None = None,
) -> bool:
    return not componentwise_violations(
        seed,
        trial,
        relative_tolerance,
        metrics=metrics,
    )


def componentwise_violations(
    seed: dict,
    trial: dict,
    relative_tolerance: float = 0.0,
    metrics: list[str] | None = None,
) -> list[str]:
    """Return every audit metric that rejects a trial.

    This is the diagnostic form of :func:`is_componentwise_safe`; keeping the
    tolerance calculation in one place guarantees that blocker reporting does
    not alter the SafeBlend feasibility rule.
    """
    violations: list[str] = []
    checked_metrics = METRICS if metrics is None else metrics
    for metric in checked_metrics:
        seed_value = float(seed[metric])
        trial_value = float(trial[metric])
        tolerance = ABS_TOLERANCE[metric] + max(float(relative_tolerance), 0.0) * max(
            abs(seed_value), SCORE_FLOORS[metric]
        )
        if not np.isfinite(trial_value):
            violations.append(metric)
            continue
        if metric in LOWER_IS_BETTER:
            if trial_value > seed_value + tolerance:
                violations.append(metric)
        elif trial_value < seed_value - tolerance:
            violations.append(metric)
    return violations


def row_with_metadata(metrics: dict, method: str, graph_id: int, batch_id: int, **extra):
    return {
        **metrics,
        "method": method,
        "graph_id": graph_id,
        "batch_id": batch_id,
        **extra,
    }


@torch.no_grad()
def evaluate_dataset(
    dataset: str,
    alphas: list[float],
    max_test_batches: int,
    output_dir: Path,
    baseline_checkpoint: str = "",
    candidate_checkpoints: str = "",
    device_override: str = "auto",
    objective_preference: str = "",
    relative_tolerance: float = 0.0,
    monotone_refinement_levels: int = 0,
    monotone_extrapolation_step: float = 0.25,
    monotone_extrapolation_steps: int = 0,
    data_split: str = "test",
    selection_metric_set: str = "audited",
    core_candidate_count: int = 0,
    deterministic_eval: bool = False,
    pairwise_mixture_weights: list[float] | None = None,
    pareto_chain_sweeps: int = 0,
    pareto_chain_steps: list[float] | None = None,
    monotone_extension_prefix_alpha: float = 0.0,
    minimum_total_gain: float = 0.0,
    local_stress_repair_steps: int = 0,
    local_stress_repair_lr: float = 0.01,
    local_stress_cv_step_scale: float = 0.0,
) -> dict:
    if not np.isfinite(minimum_total_gain) or minimum_total_gain < 0.0:
        raise ValueError("minimum_total_gain must be finite and non-negative")
    if local_stress_repair_steps < 0:
        raise ValueError("local_stress_repair_steps cannot be negative")
    if not np.isfinite(local_stress_repair_lr) or local_stress_repair_lr <= 0.0:
        raise ValueError("local_stress_repair_lr must be finite and positive")
    if not np.isfinite(local_stress_cv_step_scale) or local_stress_cv_step_scale < 0.0:
        raise ValueError("local_stress_cv_step_scale must be finite and non-negative")
    baseline_path, candidate_paths = resolve_checkpoint_paths(
        dataset,
        baseline_override=baseline_checkpoint,
        candidate_specs=candidate_checkpoints,
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline checkpoint: {baseline_path}")
    if not candidate_paths:
        raise FileNotFoundError(f"no HCTF candidate checkpoint found for {dataset}")

    device = configure_evaluation_device(device_override, deterministic_eval)
    if deterministic_eval:
        print(
            "[INFO] deterministic evaluation enabled: single-threaded CPU inference",
            flush=True,
        )
    baseline_args, baseline_model = load_model(baseline_path, device)
    candidate_models = []
    for name, path in candidate_paths:
        candidate_args, candidate_model = load_model(path, device)
        candidate_models.append((name, candidate_args, candidate_model))

    train.set_random_seed(int(getattr(baseline_args, "seed", 42)))
    _, validation_data, test_data = train.load_graph_dataset(baseline_args)
    if data_split not in {"validation", "test"}:
        raise ValueError("data_split must be 'validation' or 'test'")
    evaluation_data = validation_data if data_split == "validation" else test_data
    loader = train.make_loader(
        evaluation_data,
        int(baseline_args.batch_size),
        shuffle=False,
    )
    stress_metric = Stress()
    if selection_metric_set not in {"audited", "all"}:
        raise ValueError("selection_metric_set must be 'audited' or 'all'")
    selection_metrics = METRICS if selection_metric_set == "audited" else ALL_METRICS
    include_trial_holdout = selection_metric_set == "all"
    if selection_metric_set == "all" and str(objective_preference).strip():
        raise ValueError(
            "--objective-preference is only defined for audited selection; "
            "leave it empty in all-metric mode"
        )
    metric_weights = (
        objective_metric_weights(objective_preference)
        if selection_metric_set == "audited"
        else {metric: 1.0 for metric in selection_metrics}
    )
    midpoint_alphas = monotone_refinement_alphas(alphas, monotone_refinement_levels)
    extrapolation_alphas = monotone_extrapolation_alphas(
        monotone_extrapolation_step,
        monotone_extrapolation_steps,
    )
    refinement_alphas = midpoint_alphas + extrapolation_alphas
    mixture_weights = list(pairwise_mixture_weights or [])
    if any(not 0.0 < weight < 1.0 for weight in mixture_weights):
        raise ValueError("pairwise mixture weights must lie strictly between 0 and 1")
    mixture_weights = sorted(set(float(weight) for weight in mixture_weights))
    if pareto_chain_sweeps < 0:
        raise ValueError("pareto_chain_sweeps cannot be negative")
    if monotone_extension_prefix_alpha > 0.0 and pareto_chain_sweeps > 0:
        raise ValueError(
            "prefix-locked extension does not support Pareto chaining: "
            "the chain would not be part of the locked prefix"
        )
    chain_steps = sorted(set(float(step) for step in (pareto_chain_steps or [0.0625])))
    if any(not 0.0 < step <= 1.0 for step in chain_steps):
        raise ValueError("Pareto-chain steps must lie in (0,1]")
    mixture_candidate_count = (
        len(candidate_models) * (len(candidate_models) - 1) // 2 * len(mixture_weights)
    )
    proposal_direction_count = (
        len(candidate_models)
        + mixture_candidate_count
        + int(local_stress_repair_steps > 0)
        + int(local_stress_cv_step_scale > 0.0)
    )
    requested_core_count = (
        len(candidate_models) if core_candidate_count == 0 else core_candidate_count
    )
    if requested_core_count > len(candidate_models):
        raise ValueError("core_candidate_count cannot exceed the trained candidate count")
    effective_core_candidate_count, refinement_alpha_plan = portfolio_alpha_plan(
        proposal_direction_count,
        requested_core_count,
        alphas,
        refinement_alphas,
    )
    refinement_schedule = refinement_trial_schedule(
        refinement_alpha_plan,
        prefix_alpha=monotone_extension_prefix_alpha,
    )
    rows: list[dict] = []
    alpha_rows: list[dict] = []
    blocker_rows: list[dict] = []
    accepted_graphs = 0
    graph_counter = 0
    started = time.perf_counter()

    for batch_id, batch in enumerate(loader):
        if max_test_batches > 0 and batch_id >= max_test_batches:
            break
        batch = batch.to(device)
        _, baseline_pos, _, _ = train.forward_model(
            batch=batch,
            deepgd_model=baseline_model,
            refiner=None,
            stress_metric=stress_metric,
            args=baseline_args,
        )
        candidate_positions = []
        for name, candidate_args, candidate_model in candidate_models:
            _, candidate_pos, _, _ = train.forward_model(
                batch=batch,
                deepgd_model=candidate_model,
                refiner=None,
                stress_metric=stress_metric,
                args=candidate_args,
            )
            candidate_positions.append((name, candidate_pos))

        records = train.mtpf_build_local_graph_records(batch)
        for record in records:
            audit_started = time.perf_counter()
            nodes = record["global_nodes_device"]
            labels = detect_communities(record["edge_index"], record["num_nodes"])
            seed_pos, seed_metrics = measure_layout(
                baseline_pos[nodes], record, labels,
                include_holdout=include_trial_holdout,
            )
            best_pos = seed_pos
            best_metrics = seed_metrics
            best_score = 0.0
            best_alpha = 0.0
            best_source = "baseline"
            graph_trial_count = 0
            graph_safe_trial_count = 0
            graph_improving_safe_trial_count = 0
            graph_block_counts = {metric: 0 for metric in selection_metrics}
            refinement_trial_count = 0
            refinement_incumbent_rejections = 0
            refinement_improvements = 0
            pareto_chain_trials = 0
            pareto_chain_improvements = 0
            aligned_candidates = []
            raw_candidate_rows = []

            for candidate_index, (source_name, candidate_batch_pos) in enumerate(
                candidate_positions
            ):
                candidate_pos, candidate_metrics = measure_layout(
                    candidate_batch_pos[nodes], record, labels,
                    include_holdout=include_trial_holdout,
                )
                raw_candidate_rows.append(
                    (candidate_pos, candidate_metrics, f"HCTF-{source_name}", source_name)
                )
                aligned = train.mtpf_procrustes_align(candidate_pos, seed_pos)
                aligned_candidates.append((source_name, aligned))
                if candidate_index >= effective_core_candidate_count:
                    continue
                for alpha in alphas:
                    if alpha <= 0.0:
                        continue
                    graph_trial_count += 1
                    trial_pos, trial_metrics = measure_layout(
                        seed_pos + float(alpha) * (aligned - seed_pos),
                        record,
                        labels,
                        include_holdout=include_trial_holdout,
                    )
                    violations = componentwise_violations(
                        seed_metrics,
                        trial_metrics,
                        relative_tolerance=relative_tolerance,
                        metrics=selection_metrics,
                    )
                    if violations:
                        for metric in violations:
                            graph_block_counts[metric] += 1
                        continue
                    graph_safe_trial_count += 1
                    gains = [
                        improvement(seed_metrics, trial_metrics, metric)
                        for metric in selection_metrics
                    ]
                    score = float(
                        sum(
                            metric_weights[metric] * gain
                            for metric, gain in zip(selection_metrics, gains)
                        )
                    )
                    strictly_better = any(gain > 1e-10 for gain in gains)
                    graph_improving_safe_trial_count += int(strictly_better)
                    if strictly_better and (
                        score > best_score + 1e-12
                        or (
                            abs(score - best_score) <= 1e-12
                            and float(alpha) > best_alpha
                        )
                    ):
                        best_pos = trial_pos
                        best_metrics = trial_metrics
                        best_score = score
                        best_alpha = float(alpha)
                        best_source = source_name

            aligned_candidates.extend(
                pairwise_aligned_proposals(
                    aligned_candidates,
                    seed_pos,
                    mixture_weights,
                )
            )
            if local_stress_repair_steps > 0:
                repaired_pos = local_stress_repair_candidate(
                    seed_pos,
                    record,
                    local_stress_repair_steps,
                    local_stress_repair_lr,
                )
                repaired_pos, repaired_metrics = measure_layout(
                    repaired_pos, record, labels,
                    include_holdout=include_trial_holdout,
                )
                raw_candidate_rows.append(
                    (repaired_pos, repaired_metrics, "Local-Stress-Repair", "local_stress_repair")
                )
                aligned_candidates.append(("local_stress_repair", repaired_pos))
            if local_stress_cv_step_scale > 0.0:
                paired_pos = local_stress_cv_candidate(
                    seed_pos, record, local_stress_cv_step_scale
                )
                paired_pos, paired_metrics = measure_layout(
                    paired_pos, record, labels,
                    include_holdout=include_trial_holdout,
                )
                raw_candidate_rows.append(
                    (paired_pos, paired_metrics, "Local-Stress-CV", "local_stress_cv")
                )
                aligned_candidates.append(("local_stress_cv", paired_pos))
            if len(aligned_candidates) != proposal_direction_count:
                raise AssertionError("proposal portfolio size changed unexpectedly")

            legacy_best_metrics = dict(best_metrics)
            legacy_best_pos = best_pos
            legacy_best_alpha = best_alpha
            legacy_best_source = best_source
            legacy_best_score = best_score

            # Every accepted refinement dominates the current incumbent.
            # The optional prefix schedule also preserves the complete result
            # of a shorter alpha search before examining larger alpha values.
            for candidate_index, alpha in refinement_schedule:
                source_name, aligned = aligned_candidates[candidate_index]
                refinement_trial_count += 1
                graph_trial_count += 1
                trial_pos, trial_metrics = measure_layout(
                    seed_pos + float(alpha) * (aligned - seed_pos),
                    record,
                    labels,
                    include_holdout=include_trial_holdout,
                )
                violations = componentwise_violations(
                    seed_metrics,
                    trial_metrics,
                    relative_tolerance=relative_tolerance,
                    metrics=selection_metrics,
                )
                if violations:
                    for metric in violations:
                        graph_block_counts[metric] += 1
                    continue
                graph_safe_trial_count += 1
                seed_gains = [
                    improvement(seed_metrics, trial_metrics, metric)
                    for metric in selection_metrics
                ]
                seed_strictly_better = any(gain > 1e-10 for gain in seed_gains)
                graph_improving_safe_trial_count += int(seed_strictly_better)

                incumbent_violations = componentwise_violations(
                    best_metrics,
                    trial_metrics,
                    # Do not compound the user-configurable baseline
                    # tolerance along the refinement chain.
                    relative_tolerance=0.0,
                    metrics=selection_metrics,
                )
                if incumbent_violations:
                    refinement_incumbent_rejections += 1
                    continue
                incumbent_gains = [
                    improvement(best_metrics, trial_metrics, metric)
                    for metric in selection_metrics
                ]
                strictly_better = any(gain > 1e-10 for gain in incumbent_gains)
                if not strictly_better:
                    continue

                best_score = float(
                    sum(
                        metric_weights[metric] * gain
                        for metric, gain in zip(selection_metrics, seed_gains)
                    )
                )
                best_pos = trial_pos
                best_metrics = trial_metrics
                best_alpha = float(alpha)
                best_source = source_name
                refinement_improvements += 1

            # Compose individually trained directions only through a monotone
            # Pareto chain.  Each accepted link must dominate the current
            # incumbent on every selected metric, so later links cannot undo
            # any gain already secured by the core/refinement search.
            redundant_baseline_chain = (
                pareto_chain_sweeps > 0
                and best_alpha <= 0.0
                and baseline_chain_steps_were_searched(
                    effective_core_candidate_count,
                    alphas,
                    refinement_alpha_plan,
                    chain_steps,
                )
            )
            for _ in range(0 if redundant_baseline_chain else pareto_chain_sweeps):
                sweep_improved = False
                for source_name, aligned in aligned_candidates:
                    chain_origin = best_pos
                    source_best = None
                    source_best_score = best_score
                    for step in chain_steps:
                        pareto_chain_trials += 1
                        graph_trial_count += 1
                        trial_pos, trial_metrics = measure_layout(
                            chain_origin + float(step) * (aligned - chain_origin),
                            record,
                            labels,
                            include_holdout=include_trial_holdout,
                        )
                        violations = componentwise_violations(
                            seed_metrics,
                            trial_metrics,
                            relative_tolerance=relative_tolerance,
                            metrics=selection_metrics,
                        )
                        if violations:
                            for metric in violations:
                                graph_block_counts[metric] += 1
                            continue
                        graph_safe_trial_count += 1
                        seed_gains = [
                            improvement(seed_metrics, trial_metrics, metric)
                            for metric in selection_metrics
                        ]
                        seed_strictly_better = any(gain > 1e-10 for gain in seed_gains)
                        graph_improving_safe_trial_count += int(seed_strictly_better)

                        incumbent_violations = componentwise_violations(
                            best_metrics,
                            trial_metrics,
                            relative_tolerance=0.0,
                            metrics=selection_metrics,
                        )
                        if incumbent_violations:
                            refinement_incumbent_rejections += 1
                            continue
                        incumbent_gains = [
                            improvement(best_metrics, trial_metrics, metric)
                            for metric in selection_metrics
                        ]
                        if not any(gain > 1e-10 for gain in incumbent_gains):
                            continue
                        score = float(
                            sum(
                                metric_weights[metric] * gain
                                for metric, gain in zip(selection_metrics, seed_gains)
                            )
                        )
                        if source_best is None or score > source_best_score + 1e-12:
                            source_best = (trial_pos, trial_metrics, float(step), score)
                            source_best_score = score

                    if source_best is None:
                        continue
                    best_pos, best_metrics, chain_step, best_score = source_best
                    if best_alpha <= 0.0:
                        best_alpha = chain_step
                    best_source = f"{best_source}+chain[{source_name}]"
                    pareto_chain_improvements += 1
                    sweep_improved = True
                if not sweep_improved:
                    break

            legacy_regressions = componentwise_violations(
                legacy_best_metrics,
                best_metrics,
                relative_tolerance=0.0,
                metrics=selection_metrics,
            )
            if legacy_regressions:
                raise AssertionError(
                    "monotone refinement regressed the legacy selection on: "
                    + ", ".join(legacy_regressions)
                )

            # A nearly identical proposal can satisfy strict Pareto checks
            # through floating-point-scale changes.  The optional aggregate
            # gain floor removes such selections only at the final boundary,
            # so small intermediate chain steps may still lead to a material
            # final result.  Zero preserves the historical protocol.
            if legacy_best_alpha > 0.0 and not meets_minimum_total_gain(
                legacy_best_score, minimum_total_gain
            ):
                legacy_best_metrics = dict(seed_metrics)
                legacy_best_pos = seed_pos
                legacy_best_alpha = 0.0
                legacy_best_source = "baseline"
            if best_alpha > 0.0 and not meets_minimum_total_gain(
                best_score, minimum_total_gain
            ):
                best_pos = seed_pos
                best_metrics = seed_metrics
                best_score = 0.0
                best_alpha = 0.0
                best_source = "baseline"

            seed_metrics = attach_postselection_holdout(seed_metrics, seed_pos, record)
            legacy_best_metrics = attach_postselection_holdout(
                legacy_best_metrics, legacy_best_pos, record
            )
            best_metrics = attach_postselection_holdout(best_metrics, best_pos, record)

            for raw_pos, raw_metrics, raw_method, raw_source in raw_candidate_rows:
                rows.append(
                    row_with_metadata(
                        attach_postselection_holdout(raw_metrics, raw_pos, record),
                        raw_method,
                        graph_counter,
                        batch_id,
                        selected_alpha=1.0,
                        selected_source=raw_source,
                        accepted=False,
                    )
                )

            accepted = best_alpha > 0.0
            accepted_graphs += int(accepted)
            audit_seconds = time.perf_counter() - audit_started
            rows.append(
                row_with_metadata(
                    seed_metrics,
                    "Current-Model-Baseline",
                    graph_counter,
                    batch_id,
                    selected_alpha=0.0,
                    selected_source="baseline",
                    accepted=False,
                    audit_seconds=audit_seconds,
                )
            )
            rows.append(
                row_with_metadata(
                    legacy_best_metrics,
                    "HCTF-SafeBlend-LegacyGrid",
                    graph_counter,
                    batch_id,
                    selected_alpha=legacy_best_alpha,
                    selected_source=legacy_best_source,
                    accepted=legacy_best_alpha > 0.0,
                    audit_seconds=audit_seconds,
                )
            )
            rows.append(
                row_with_metadata(
                    best_metrics,
                    "HCTF-SafeBlend",
                    graph_counter,
                    batch_id,
                    selected_alpha=best_alpha,
                    selected_source=best_source,
                    accepted=accepted,
                    audit_seconds=audit_seconds,
                )
            )
            alpha_rows.append(
                {
                    "dataset": dataset,
                    "graph_id": graph_counter,
                    "batch_id": batch_id,
                    "accepted": accepted,
                    "selected_source": best_source,
                    "selected_alpha": best_alpha,
                    "legacy_selected_source": legacy_best_source,
                    "legacy_selected_alpha": legacy_best_alpha,
                    "refinement_changed_selection": (
                        best_source != legacy_best_source
                        or best_alpha != legacy_best_alpha
                    ),
                    "score": best_score,
                    "num_nodes": int(record["num_nodes"]),
                    "num_edges": int(len(unique_undirected_edges(record["edge_index"]))),
                    "audit_seconds": audit_seconds,
                }
            )
            blocker_rows.append(
                {
                    "dataset": dataset,
                    "graph_id": graph_counter,
                    "batch_id": batch_id,
                    "accepted": accepted,
                    "selected_source": best_source,
                    "selected_alpha": best_alpha,
                    "legacy_selected_source": legacy_best_source,
                    "legacy_selected_alpha": legacy_best_alpha,
                    "refinement_changed_selection": (
                        best_source != legacy_best_source
                        or best_alpha != legacy_best_alpha
                    ),
                    "num_nodes": int(record["num_nodes"]),
                    "num_edges": int(len(unique_undirected_edges(record["edge_index"]))),
                    "trial_count": graph_trial_count,
                    "componentwise_safe_trials": graph_safe_trial_count,
                    "improving_safe_trials": graph_improving_safe_trial_count,
                    "safe_but_nonimproving_trials": (
                        graph_safe_trial_count - graph_improving_safe_trial_count
                    ),
                    "monotone_refinement_levels": monotone_refinement_levels,
                    "selection_metric_set": selection_metric_set,
                    "core_candidate_count": effective_core_candidate_count,
                    "monotone_extrapolation_step": monotone_extrapolation_step,
                    "monotone_extrapolation_steps": monotone_extrapolation_steps,
                    "monotone_extension_prefix_alpha": monotone_extension_prefix_alpha,
                    "monotone_refinement_trials": refinement_trial_count,
                    "refinement_incumbent_rejections": refinement_incumbent_rejections,
                    "refinement_improvements": refinement_improvements,
                    "pareto_chain_trials": pareto_chain_trials,
                    "pareto_chain_improvements": pareto_chain_improvements,
                    "pareto_chain_skipped_redundant": redundant_baseline_chain,
                    **{
                        f"blocked_{metric}_trials": graph_block_counts[metric]
                        for metric in selection_metrics
                    },
                }
            )
            graph_counter += 1

    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    alpha_frame = pd.DataFrame(alpha_rows)
    blocker_frame = pd.DataFrame(blocker_rows)
    file_stem = dataset if data_split == "test" else f"{dataset}_{data_split}"
    frame.to_csv(output_dir / f"{file_stem}_eval.csv", index=False, encoding="utf-8-sig")
    alpha_frame.to_csv(
        output_dir / f"{file_stem}_selection.csv", index=False, encoding="utf-8-sig"
    )
    blocker_frame.to_csv(
        output_dir / f"{file_stem}_blocker_by_graph.csv",
        index=False,
        encoding="utf-8-sig",
    )
    blocker_summary_rows = []
    total_trials = int(blocker_frame["trial_count"].sum())
    rejected = blocker_frame[~blocker_frame["accepted"]]
    rejected_trials = int(rejected["trial_count"].sum())
    for metric in selection_metrics:
        column = f"blocked_{metric}_trials"
        blocker_summary_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "blocked_trials": int(blocker_frame[column].sum()),
                "blocked_trial_fraction": float(
                    blocker_frame[column].sum() / max(total_trials, 1)
                ),
                "graphs_blocked_at_least_once": int((blocker_frame[column] > 0).sum()),
                "graph_fraction": float((blocker_frame[column] > 0).mean()),
                "rejected_graph_blocked_trials": int(rejected[column].sum()),
                "rejected_graph_blocked_trial_fraction": float(
                    rejected[column].sum() / max(rejected_trials, 1)
                ),
            }
        )
    pd.DataFrame(blocker_summary_rows).sort_values(
        "blocked_trials",
        ascending=False,
    ).to_csv(
        output_dir / f"{file_stem}_blocker_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped = frame[frame["method"].isin(["Current-Model-Baseline", "HCTF-SafeBlend"])].groupby(
        "method"
    )[METRICS].mean()
    baseline = grouped.loc["Current-Model-Baseline"]
    safe = grouped.loc["HCTF-SafeBlend"]
    legacy_grouped = frame[frame["method"] == "HCTF-SafeBlend-LegacyGrid"].groupby(
        "method"
    )[METRICS].mean()
    legacy_safe = legacy_grouped.loc["HCTF-SafeBlend-LegacyGrid"]
    summary = {
        "dataset": dataset,
        "split": data_split,
        "selection_metric_set": selection_metric_set,
        "deterministic_eval": deterministic_eval,
        "graphs": graph_counter,
        "candidates": len(candidate_models),
        "pairwise_mixture_candidates": mixture_candidate_count,
        "proposal_directions": proposal_direction_count,
        "core_candidates": effective_core_candidate_count,
        "minimum_total_gain": minimum_total_gain,
        "local_stress_repair_steps": local_stress_repair_steps,
        "local_stress_repair_lr": local_stress_repair_lr,
        "local_stress_cv_step_scale": local_stress_cv_step_scale,
        "accepted_graphs": accepted_graphs,
        "acceptance_rate": accepted_graphs / max(graph_counter, 1),
        "elapsed_seconds": elapsed,
        "monotone_refinement_levels": monotone_refinement_levels,
        "midpoint_refinement_alpha_count": len(midpoint_alphas),
        "monotone_extrapolation_step": monotone_extrapolation_step,
        "monotone_extrapolation_steps": monotone_extrapolation_steps,
        "monotone_extension_prefix_alpha": monotone_extension_prefix_alpha,
        "refinement_alpha_count": len(refinement_alphas),
        "refinement_improvements": int(blocker_frame["refinement_improvements"].sum()),
        "pareto_chain_sweeps": pareto_chain_sweeps,
        "pareto_chain_improvements": int(
            blocker_frame["pareto_chain_improvements"].sum()
        ),
    }
    improved = 0
    for metric in METRICS:
        base_value = float(baseline[metric])
        safe_value = float(safe[metric])
        legacy_value = float(legacy_safe[metric])
        gain = improvement({metric: base_value}, {metric: safe_value}, metric) * 100.0
        refinement_gain = (
            improvement({metric: legacy_value}, {metric: safe_value}, metric) * 100.0
        )
        summary[f"{metric}_baseline"] = base_value
        summary[f"{metric}_legacy_grid"] = legacy_value
        summary[f"{metric}_safe"] = safe_value
        summary[f"{metric}_improvement_pct"] = gain
        summary[f"{metric}_refinement_improvement_pct"] = refinement_gain
        improved += int(gain > 1e-10)
    summary["improved_metrics"] = improved
    holdout_grouped = frame[
        frame["method"].isin(
            [
                "Current-Model-Baseline",
                "HCTF-SafeBlend-LegacyGrid",
                "HCTF-SafeBlend",
            ]
        )
    ].groupby("method")[HOLDOUT_METRICS].mean()
    holdout_baseline = holdout_grouped.loc["Current-Model-Baseline"]
    holdout_legacy = holdout_grouped.loc["HCTF-SafeBlend-LegacyGrid"]
    holdout_safe = holdout_grouped.loc["HCTF-SafeBlend"]
    holdout_improved = 0
    for metric in HOLDOUT_METRICS:
        base_value = float(holdout_baseline[metric])
        legacy_value = float(holdout_legacy[metric])
        safe_value = float(holdout_safe[metric])
        scale = max(abs(base_value), 1e-9)
        gain = 100.0 * (safe_value - base_value) / scale
        refinement_scale = max(abs(legacy_value), 1e-9)
        refinement_gain = 100.0 * (safe_value - legacy_value) / refinement_scale
        summary[f"holdout_{metric}_baseline"] = base_value
        summary[f"holdout_{metric}_legacy_grid"] = legacy_value
        summary[f"holdout_{metric}_safe"] = safe_value
        summary[f"holdout_{metric}_improvement_pct"] = gain
        summary[f"holdout_{metric}_refinement_change_pct"] = refinement_gain
        holdout_improved += int(gain > 1e-10)
    summary["holdout_improved_metrics"] = holdout_improved
    summary["all_metric_improved_metrics"] = improved + holdout_improved
    summary["mean_audit_ms_per_graph"] = 1000.0 * float(
        alpha_frame["audit_seconds"].mean()
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sbm", "enzymes", "proteins", "wn18rr", "fb15k237", "rome", "north", "imdb_binary"],
    )
    parser.add_argument("--alphas", default="0,0.0625,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--max-test-batches", type=int, default=-1)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Validation is for policy design; test must remain a one-shot final evaluation.",
    )
    parser.add_argument(
        "--selection-metric-set",
        choices=["audited", "all"],
        default="audited",
        help=(
            "audited preserves the seven independent endpoints; all constrains "
            "all 17 metrics but relinquishes their held-out status."
        ),
    )
    parser.add_argument(
        "--core-candidate-count",
        type=int,
        default=0,
        help=(
            "Number of leading candidates in the incumbent pool. Zero keeps "
            "historical behavior (all candidates are core); later candidates "
            "may only enter through incumbent-dominating portfolio refinement."
        ),
    )
    parser.add_argument(
        "--pairwise-mixture-weights",
        default="",
        help=(
            "Optional comma-separated convex weights in (0,1). Pairwise mixtures "
            "of aligned trained proposals enter only through the componentwise-"
            "dominating refinement stage; empty preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--pareto-chain-sweeps",
        type=int,
        default=0,
        help=(
            "Number of deterministic sweeps that compose proposal directions. "
            "Every link must componentwise-dominate the current incumbent; zero "
            "preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--pareto-chain-steps",
        default="0.03125,0.0625,0.125,0.25",
        help="Comma-separated interpolation fractions in (0,1] for Pareto chaining.",
    )
    parser.add_argument("--output-subdir", default="hctf_safe_blend")
    parser.add_argument(
        "--objective-preference",
        default="",
        help="Optional geometry, visual, community ranking weights; empty preserves legacy equal-metric scoring.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="Optional per-metric relative epsilon; zero gives strict non-regression.",
    )
    parser.add_argument(
        "--minimum-total-gain",
        type=float,
        default=0.0,
        help=(
            "Minimum sum of direction-correct normalized metric gains for a "
            "selected layout. Zero preserves historical selection."
        ),
    )
    parser.add_argument(
        "--local-stress-repair-steps",
        type=int,
        default=0,
        help=(
            "Optional deterministic graph-local stress refinement as an extra "
            "candidate; zero preserves historical evaluation."
        ),
    )
    parser.add_argument("--local-stress-repair-lr", type=float, default=0.01)
    parser.add_argument(
        "--local-stress-cv-step-scale",
        type=float,
        default=0.0,
        help=(
            "Optional graph-local joint stress/CV descent direction. "
            "Zero preserves historical evaluation."
        ),
    )
    parser.add_argument(
        "--monotone-refinement-levels",
        type=int,
        default=0,
        help=(
            "Midpoint-refine the alpha grid after legacy selection. New points "
            "may replace the incumbent only by componentwise Pareto dominance."
        ),
    )
    parser.add_argument(
        "--monotone-extrapolation-step",
        type=float,
        default=0.25,
        help="Alpha increment beyond 1.0 for audited proposal-line extrapolation.",
    )
    parser.add_argument(
        "--monotone-extrapolation-steps",
        type=int,
        default=0,
        help=(
            "Number of alpha points beyond 1.0. Each point must Pareto-dominate "
            "the current incumbent on all audited metrics before it is accepted."
        ),
    )
    parser.add_argument(
        "--monotone-extension-prefix-alpha",
        type=float,
        default=0.0,
        help=(
            "Lock candidate-major refinement through this alpha before visiting "
            "larger alpha values in ascending order across candidates. Use the "
            "old search maximum (for example 2.0) when extending a prior policy."
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="",
        help="Optional path template; {dataset} is expanded.",
    )
    parser.add_argument(
        "--candidate-checkpoints",
        default="",
        help="Semicolon-separated name=path templates; {dataset} is expanded.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help=(
            "Force single-threaded CPU inference and deterministic PyTorch "
            "algorithms for reproducible boundary decisions."
        ),
    )
    args = parser.parse_args()

    alphas = sorted({float(value.strip()) for value in args.alphas.split(",") if value.strip()})
    if not alphas or min(alphas) < 0.0 or max(alphas) > 1.0:
        raise ValueError("--alphas must contain values in [0, 1]")
    objective_metric_weights(args.objective_preference)
    pairwise_mixture_weights = parse_pairwise_mixture_weights(
        args.pairwise_mixture_weights
    )
    pareto_chain_steps = parse_pareto_chain_steps(args.pareto_chain_steps)
    if args.pareto_chain_sweeps < 0:
        raise ValueError("--pareto-chain-sweeps cannot be negative")
    if args.relative_tolerance < 0.0:
        raise ValueError("--relative-tolerance cannot be negative")
    if not np.isfinite(args.minimum_total_gain) or args.minimum_total_gain < 0.0:
        raise ValueError("--minimum-total-gain must be finite and non-negative")
    if args.monotone_refinement_levels < 0:
        raise ValueError("--monotone-refinement-levels cannot be negative")
    monotone_extrapolation_alphas(
        args.monotone_extrapolation_step,
        args.monotone_extrapolation_steps,
    )
    if (
        not np.isfinite(args.monotone_extension_prefix_alpha)
        or args.monotone_extension_prefix_alpha < 0.0
    ):
        raise ValueError(
            "--monotone-extension-prefix-alpha must be finite and non-negative"
        )
    output_dir = ROOT / "results" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summaries: list[dict] = []
    failures: list[dict] = []
    for index, dataset in enumerate(args.datasets, start=1):
        print(f"[{index}/{len(args.datasets)}] safe HCTF evaluation: {dataset}", flush=True)
        try:
            summary = evaluate_dataset(
                dataset=dataset,
                alphas=alphas,
                max_test_batches=args.max_test_batches,
                output_dir=output_dir,
                baseline_checkpoint=args.baseline_checkpoint,
                candidate_checkpoints=args.candidate_checkpoints,
                device_override=args.device,
                objective_preference=args.objective_preference,
                relative_tolerance=args.relative_tolerance,
                monotone_refinement_levels=args.monotone_refinement_levels,
                monotone_extrapolation_step=args.monotone_extrapolation_step,
                monotone_extrapolation_steps=args.monotone_extrapolation_steps,
                data_split=args.split,
                selection_metric_set=args.selection_metric_set,
                core_candidate_count=args.core_candidate_count,
                deterministic_eval=args.deterministic_eval,
                pairwise_mixture_weights=pairwise_mixture_weights,
                pareto_chain_sweeps=args.pareto_chain_sweeps,
                pareto_chain_steps=pareto_chain_steps,
                monotone_extension_prefix_alpha=args.monotone_extension_prefix_alpha,
                minimum_total_gain=args.minimum_total_gain,
                local_stress_repair_steps=args.local_stress_repair_steps,
                local_stress_repair_lr=args.local_stress_repair_lr,
                local_stress_cv_step_scale=args.local_stress_cv_step_scale,
            )
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(
                output_dir / "summary.csv", index=False, encoding="utf-8-sig"
            )
            print(
                f"[SUCCESS] {dataset}: accepted={summary['accepted_graphs']}/"
                f"{summary['graphs']}, improved="
                f"{summary['all_metric_improved_metrics'] if args.selection_metric_set == 'all' else summary['improved_metrics']}/"
                f"{17 if args.selection_metric_set == 'all' else 10}",
                flush=True,
            )
        except Exception as error:
            failures.append({"dataset": dataset, "error": repr(error)})
            pd.DataFrame(failures).to_csv(
                output_dir / "failures.csv", index=False, encoding="utf-8-sig"
            )
            print(f"[FAILED] {dataset}: {error!r}", flush=True)

    improved_key = (
        "all_metric_improved_metrics"
        if args.selection_metric_set == "all"
        else "improved_metrics"
    )
    metric_count = 17 if args.selection_metric_set == "all" else 10
    total_improved = sum(int(row[improved_key]) for row in summaries)
    print(
        f"[DONE] success={len(summaries)}, failures={len(failures)}, "
        f"improved_metrics={total_improved}/{metric_count * len(summaries)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
