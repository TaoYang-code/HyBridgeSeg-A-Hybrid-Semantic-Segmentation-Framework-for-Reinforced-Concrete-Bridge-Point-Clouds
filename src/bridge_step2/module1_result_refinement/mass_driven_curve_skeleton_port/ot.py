from __future__ import annotations

import numpy as np

from .common import pairwise_distances
from .sinkhorn_transport import sinkhorn_transport


def compute_optimal_transport(
    points: np.ndarray,
    skeleton_points: np.ndarray,
    point_mass: np.ndarray,
    skeleton_mass: np.ndarray,
    lambda_value: float,
) -> tuple[np.ndarray, float]:
    distance_matrix = pairwise_distances(skeleton_points, points)
    k_matrix = np.exp(-lambda_value * distance_matrix)
    u_matrix = k_matrix * distance_matrix
    _, _, left_scalings, right_scalings = sinkhorn_transport(
        skeleton_mass,
        point_mass,
        k_matrix,
        u_matrix,
        lambda_value,
        verbose=0,
    )
    transport_plan = left_scalings * (k_matrix @ np.diagflat(right_scalings[:, 0]) if right_scalings.shape[1] == 1 else 0)
    if right_scalings.shape[1] != 1:
        transport_plan = np.multiply(right_scalings.T, (left_scalings * k_matrix))
    transport_plan = right_scalings[:, 0][None, :] * (left_scalings[:, 0][:, None] * k_matrix) if right_scalings.shape[1] == 1 else transport_plan
    transport_value = float(np.sum(transport_plan * distance_matrix))
    return transport_plan, transport_value
