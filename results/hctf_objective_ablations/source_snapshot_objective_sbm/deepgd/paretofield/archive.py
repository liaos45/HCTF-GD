from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


DEFAULT_METRICS = (
    ("geometry", "lower"),
    ("visual", "lower"),
    ("community", "lower"),
)


def _as_2d(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("objective values must be a 2D array")
    return array


def orient_to_minimize(values, directions: Sequence[str]) -> np.ndarray:
    array = _as_2d(values).copy()
    if array.shape[1] != len(directions):
        raise ValueError("directions must match objective columns")
    for column, direction in enumerate(directions):
        if direction not in {"lower", "higher"}:
            raise ValueError(f"unknown objective direction: {direction}")
        if direction == "higher":
            array[:, column] *= -1.0
    return array


def dominance_counts(values, eps: float = 1e-12) -> np.ndarray:
    values = _as_2d(values)
    counts = np.zeros(values.shape[0], dtype=np.int64)
    for i in range(values.shape[0]):
        no_worse = values <= values[i] + eps
        strictly_better = values < values[i] - eps
        dominates_i = np.all(no_worse, axis=1) & np.any(strictly_better, axis=1)
        counts[i] = int(dominates_i.sum())
    return counts


def non_dominated_sort(values, eps: float = 1e-12) -> np.ndarray:
    values = _as_2d(values)
    remaining = list(range(values.shape[0]))
    ranks = np.full(values.shape[0], -1, dtype=np.int64)
    rank = 0
    while remaining:
        current = values[remaining]
        counts = dominance_counts(current, eps=eps)
        front_local = np.flatnonzero(counts == 0)
        if front_local.size == 0:
            front_local = np.array([0], dtype=np.int64)
        front_global = [remaining[int(index)] for index in front_local]
        ranks[front_global] = rank
        front_set = set(front_global)
        remaining = [index for index in remaining if index not in front_set]
        rank += 1
    return ranks


def crowding_distance(values) -> np.ndarray:
    values = _as_2d(values)
    count, objectives = values.shape
    if count <= 2:
        return np.full(count, np.inf, dtype=np.float64)
    distance = np.zeros(count, dtype=np.float64)
    for column in range(objectives):
        order = np.argsort(values[:, column], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[order[-1], column] - values[order[0], column]
        if abs(span) <= 1e-12:
            continue
        interior = order[1:-1]
        distance[interior] += (
            values[order[2:], column] - values[order[:-2], column]
        ) / span
    return distance


def reference_directions(num_objectives: int = 3, divisions: int = 4) -> np.ndarray:
    if num_objectives < 2:
        return np.ones((1, 1), dtype=np.float64)
    if divisions <= 0:
        raise ValueError("divisions must be positive")

    compositions = []

    def visit(prefix, remaining, dimensions):
        if dimensions == 1:
            compositions.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, dimensions - 1)

    visit([], divisions, num_objectives)
    return np.asarray(compositions, dtype=np.float64) / float(divisions)


@dataclass
class RobustObjectiveNormalizer:
    directions: Sequence[str]
    clip: float = 5.0
    eps: float = 1e-8
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values, baseline=None):
        oriented = orient_to_minimize(values, self.directions)
        if baseline is None:
            center = np.median(oriented, axis=0)
        else:
            center = orient_to_minimize(baseline, self.directions)[0]
        q25, q75 = np.percentile(oriented, [25.0, 75.0], axis=0)
        scale = np.maximum(q75 - q25, self.eps)
        fallback = np.maximum(np.std(oriented, axis=0), self.eps)
        self.center_ = center
        self.scale_ = np.where(scale > self.eps, scale, fallback)
        return self

    def transform(self, values) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("normalizer must be fitted before transform")
        oriented = orient_to_minimize(values, self.directions)
        normalized = (oriented - self.center_) / self.scale_
        return np.clip(normalized, -float(self.clip), float(self.clip))

    def fit_transform(self, values, baseline=None) -> np.ndarray:
        return self.fit(values, baseline=baseline).transform(values)


def pareto_sample_weights(
    ranks,
    crowding,
    temperature: float = 1.0,
    density_floor: float = 1e-3,
) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.float64)
    crowding = np.asarray(crowding, dtype=np.float64)
    finite = crowding[np.isfinite(crowding)]
    cap = float(finite.max()) if finite.size else 1.0
    diversity = np.where(np.isfinite(crowding), crowding, cap + 1.0)
    diversity = diversity - diversity.min() + float(density_floor)
    quality = np.exp(-ranks / max(float(temperature), 1e-8))
    weights = quality * diversity
    return weights / max(weights.sum(), 1e-12)


def achievement_scalarization(values, preference, rho: float = 0.05) -> np.ndarray:
    values = _as_2d(values)
    preference = np.asarray(preference, dtype=np.float64).reshape(-1)
    if values.shape[1] != preference.size:
        raise ValueError("preference must match objective columns")
    preference = np.maximum(preference, 1e-6)
    preference /= preference.sum()
    shifted = values - values.min(axis=0, keepdims=True)
    weighted = shifted * preference[None, :]
    return weighted.max(axis=1) + float(rho) * weighted.sum(axis=1)


def build_pareto_archive(
    candidates: Sequence[Mapping],
    objective_keys=("geometry", "visual", "community"),
    directions=("lower", "lower", "lower"),
    max_front: int = 6,
    max_sparse: int = 6,
) -> list[dict]:
    if not candidates:
        return []
    values = np.asarray(
        [[float(candidate["objectives"][key]) for key in objective_keys] for candidate in candidates],
        dtype=np.float64,
    )
    baseline = values[0]
    normalized = RobustObjectiveNormalizer(directions).fit_transform(values, baseline=baseline)
    ranks = non_dominated_sort(normalized)
    crowding = crowding_distance(normalized)
    weights = pareto_sample_weights(ranks, crowding)

    front = np.flatnonzero(ranks == 0)
    front_order = sorted(front.tolist(), key=lambda index: (-crowding[index], index))
    selected = front_order[: max(int(max_front), 0)]

    remaining = [index for index in range(len(candidates)) if index not in set(selected)]
    sparse_order = sorted(remaining, key=lambda index: (-crowding[index], ranks[index], index))
    selected.extend(sparse_order[: max(int(max_sparse), 0)])

    for column in range(values.shape[1]):
        selected.append(int(np.argmin(normalized[:, column])))
    selected.append(0)
    selected = list(dict.fromkeys(selected))

    archive = []
    for index in selected:
        item = dict(candidates[index])
        item["objective_vector"] = normalized[index].astype(np.float32)
        item["pareto_rank"] = int(ranks[index])
        item["crowding_distance"] = float(crowding[index])
        item["sample_weight"] = float(weights[index])
        archive.append(item)
    return archive


class ParetoTeacherArchive:
    def __init__(self, graphs: Mapping[str, Sequence[Mapping]], rho: float = 0.05):
        self.graphs = {str(name): list(items) for name, items in graphs.items()}
        self.rho = float(rho)

    @classmethod
    def load(cls, path):
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
        graphs = payload.get("graphs", payload)
        return cls(graphs=graphs, rho=float(payload.get("rho", 0.05)))

    def select(self, graph_name: str, preference) -> Mapping | None:
        candidates = self.graphs.get(str(graph_name), [])
        if not candidates:
            return None
        vectors = np.asarray([candidate["objective_vector"] for candidate in candidates])
        scores = achievement_scalarization(vectors, preference, rho=self.rho)
        weighted_scores = []
        for index, score in enumerate(scores):
            weight = max(float(candidates[index].get("sample_weight", 1.0)), 1e-8)
            weighted_scores.append(float(score) - 0.01 * np.log(weight))
        return candidates[int(np.argmin(weighted_scores))]

    def select_batch(self, names: Iterable[str], preferences, node_counts, device, dtype):
        preferences = torch.as_tensor(preferences).detach().cpu().numpy()
        targets = []
        masks = []
        weights = []
        for name, preference, count in zip(names, preferences, node_counts):
            candidate = self.select(str(name), preference)
            if candidate is None:
                targets.append(torch.zeros((int(count), 2), dtype=dtype, device=device))
                masks.append(False)
                weights.append(0.0)
                continue
            pos = torch.as_tensor(candidate["pos"], dtype=dtype, device=device)
            if pos.shape != (int(count), 2):
                targets.append(torch.zeros((int(count), 2), dtype=dtype, device=device))
                masks.append(False)
                weights.append(0.0)
                continue
            targets.append(pos)
            masks.append(True)
            weights.append(float(candidate.get("sample_weight", 1.0)))
        return (
            torch.cat(targets, dim=0),
            torch.tensor(masks, dtype=torch.bool, device=device),
            torch.tensor(weights, dtype=dtype, device=device),
        )
