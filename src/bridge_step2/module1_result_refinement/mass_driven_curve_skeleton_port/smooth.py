from __future__ import annotations

import numpy as np

from .interpolate import interpolate_skeleton
from .optimize_skeleton_with_line import optimize_skeleton_with_line


def smooth_skeleton(
    points: np.ndarray,
    adjacency: np.ndarray,
    skeleton_points: np.ndarray,
    skeleton_mass: np.ndarray,
    lambda_value: float,
    weight_value: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del skeleton_mass
    point_mass = np.ones((len(points), 1), dtype=float) / max(len(points), 1)
    if n == 0:
        result_skeleton = skeleton_points.copy()
        new_adjacency = adjacency.copy()
    else:
        result_skeleton, new_adjacency = interpolate_skeleton(adjacency, skeleton_points, n)
    new_skeleton, new_mass = optimize_skeleton_with_line(
        points,
        result_skeleton,
        new_adjacency,
        point_mass,
        lambda_value,
        weight_value,
    )
    return new_skeleton, new_adjacency, new_mass
