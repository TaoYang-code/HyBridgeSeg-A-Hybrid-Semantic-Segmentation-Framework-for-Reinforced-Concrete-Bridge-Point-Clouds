from __future__ import annotations

import numpy as np

from .center_with_line import center_points_with_line
from .find_nei import find_nearest_indices
from .ot import compute_optimal_transport


def optimize_skeleton_with_line(
    points: np.ndarray,
    skeleton_points: np.ndarray,
    neighbor_matrix: np.ndarray,
    point_mass: np.ndarray,
    lambda_value: float,
    weight_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    transport_previous = 9_999_999.0
    threshold = 1e-4
    skeleton = skeleton_points.copy()
    nearest = find_nearest_indices(points, skeleton)
    skeleton_mass = np.zeros(len(skeleton), dtype=float)
    for index in range(len(skeleton)):
        skeleton_mass[index] = np.sum(nearest == index)
    skeleton_mass = skeleton_mass / max(len(nearest), 1)

    while True:
        transport_plan, transport_value = compute_optimal_transport(
            points,
            skeleton,
            point_mass,
            skeleton_mass[:, None],
            lambda_value,
        )
        skeleton = center_points_with_line(points, skeleton, transport_plan, neighbor_matrix, weight_value)
        if abs(transport_previous - transport_value) < threshold:
            break
        transport_previous = transport_value

    return skeleton, skeleton_mass
