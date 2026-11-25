"""Optional research-only projection of HCTF parameter steps.

The halfspaces use gradients of lower-is-better surrogate objectives. This
guarantees only their local first-order condition, not exact metric safety.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def project_step_to_guard_halfspaces(step, guard_gradients):
    """Return the nearest Euclidean step with no positive guard dot product.

    The implementation uses Moreau decomposition: subtract the NNLS
    projection of the proposed step onto the cone of normalized guard normals.
    Normalization is only for numerical conditioning and leaves each
    halfspace unchanged.
    """
    proposed = np.asarray(step, dtype=np.float64).reshape(-1)
    if not np.isfinite(proposed).all():
        raise ValueError("proposed step contains nonfinite values")
    length = float(np.linalg.norm(proposed))
    if length == 0.0:
        return proposed.copy(), {
            "retained_norm": 1.0, "task_alignment": 1.0,
            "max_guard_dot": 0.0, "active_guards": 0,
        }
    normals = []
    for gradient in guard_gradients:
        normal = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if normal.shape != proposed.shape or not np.isfinite(normal).all():
            raise ValueError("guard gradient shape or finiteness mismatch")
        normal_length = float(np.linalg.norm(normal))
        if normal_length > 0.0:
            normals.append(normal / normal_length)
    if not normals:
        return proposed.copy(), {
            "retained_norm": 1.0, "task_alignment": 1.0,
            "max_guard_dot": 0.0, "active_guards": 0,
        }
    matrix = np.stack(normals)
    unit_step = proposed / length
    multipliers, _ = nnls(matrix.T, unit_step, maxiter=500)
    projected_unit = unit_step - matrix.T @ multipliers
    guard_dots = matrix @ projected_unit
    max_guard_dot = float(guard_dots.max())
    if max_guard_dot > 1e-7:
        raise FloatingPointError(
            f"projected HCTF step violates a guard: {max_guard_dot}"
        )
    return projected_unit * length, {
        "retained_norm": float(np.linalg.norm(projected_unit)),
        "task_alignment": float(unit_step @ projected_unit),
        "max_guard_dot": max_guard_dot,
        "active_guards": len(normals),
    }
