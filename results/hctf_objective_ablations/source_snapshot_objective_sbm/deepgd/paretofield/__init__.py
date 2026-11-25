"""Preference-conditioned Pareto graph drawing components."""

from .alignment import similarity_aligned_mse, similarity_align
from .archive import (
    DEFAULT_METRICS,
    ParetoTeacherArchive,
    RobustObjectiveNormalizer,
    build_pareto_archive,
    non_dominated_sort,
    reference_directions,
)
from .conflict import GradientMonitor, apply_conflict_free_gradients
from .crossing import (
    crossing_objectives_from_edge_pairs_per_graph,
    crossing_topology_barrier_per_graph,
    crossing_loss_from_edge_pairs_per_graph,
    edge_pair_crossing_scores,
    eligible_edge_pair_counts,
    hard_sampled_crossing_loss,
    hard_sampled_crossing_loss_per_graph,
    select_metric_aligned_edge_pairs,
    sample_hard_edge_pairs,
)
from .fling_adapter import FlingAdapter, FlingUnavailableError
from .losses import (
    graphwise_pareto_guard,
    graphwise_pareto_objectives,
    graphwise_regret_loss,
)
from .model import (
    ParetoFieldRefiner,
    graph_community_confidence,
    sample_preferences,
    soft_community_relation,
    topology_consensus_relation,
)
from .refinement import objective_space_repulsion, refine_pareto_set

__all__ = [
    "DEFAULT_METRICS",
    "FlingAdapter",
    "FlingUnavailableError",
    "GradientMonitor",
    "ParetoFieldRefiner",
    "ParetoTeacherArchive",
    "RobustObjectiveNormalizer",
    "apply_conflict_free_gradients",
    "build_pareto_archive",
    "crossing_objectives_from_edge_pairs_per_graph",
    "crossing_topology_barrier_per_graph",
    "crossing_loss_from_edge_pairs_per_graph",
    "edge_pair_crossing_scores",
    "eligible_edge_pair_counts",
    "graphwise_pareto_objectives",
    "graphwise_pareto_guard",
    "graphwise_regret_loss",
    "graph_community_confidence",
    "hard_sampled_crossing_loss",
    "hard_sampled_crossing_loss_per_graph",
    "non_dominated_sort",
    "objective_space_repulsion",
    "reference_directions",
    "refine_pareto_set",
    "sample_preferences",
    "sample_hard_edge_pairs",
    "select_metric_aligned_edge_pairs",
    "similarity_align",
    "similarity_aligned_mse",
    "soft_community_relation",
    "topology_consensus_relation",
]
