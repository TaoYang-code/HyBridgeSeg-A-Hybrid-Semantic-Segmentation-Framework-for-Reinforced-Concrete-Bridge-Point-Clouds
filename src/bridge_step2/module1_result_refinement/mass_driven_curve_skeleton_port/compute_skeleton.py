from __future__ import annotations

import numpy as np

from .center import center_points
from .im_merge import im_merge
from .ot import compute_optimal_transport


def compute_skeleton(
    points: np.ndarray,
    skeleton_points: np.ndarray,
    point_mass: np.ndarray,
    skeleton_mass: np.ndarray,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_transport_value = 0.0
    threshold_transport_value = 1e-4
    merge_points: list[np.ndarray] = []
    condition = True

    while condition:
        initial_skeleton_count = skeleton_points.shape[0]
        while True:
            transport_plan, transport_value = compute_optimal_transport(
                points,
                skeleton_points,
                point_mass,
                skeleton_mass,
                lambda_value,
            )
            skeleton_points = center_points(points, skeleton_points, transport_plan)
            if abs(previous_transport_value - transport_value) < threshold_transport_value:
                break
            previous_transport_value = transport_value

        skeleton_points, skeleton_mass, transport_plan, merge_pair, non_skeleton_points = im_merge(
            points,
            transport_plan,
            skeleton_points,
            point_mass,
        )
        if len(merge_pair) > 0:
            merge_points.append(merge_pair)

        keep_mask = skeleton_mass >= 1e-5
        skeleton_points = skeleton_points[keep_mask]
        skeleton_mass = skeleton_mass[keep_mask]
        transport_plan = transport_plan[keep_mask]
        if initial_skeleton_count == skeleton_points.shape[0]:
            condition = False

    merged = np.vstack(merge_points) if merge_points else np.empty((0, 2), dtype=int)
    return skeleton_mass, skeleton_points, transport_plan, merged, non_skeleton_points
