from __future__ import annotations

import numpy as np

from .common import normalize_mass


def merge_circle(
    skeleton_points: np.ndarray,
    skeleton_mass: np.ndarray,
    non_circle: list[list[int]],
    circle: list[list[int]],
    bool_circle: bool,
    bool_non_circle: bool,
    transport_plan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del non_circle, bool_non_circle
    skeleton_points = skeleton_points.copy()
    skeleton_mass = skeleton_mass.copy()

    if bool_circle and circle:
        delete_point = circle[0][0]
        min_mass = skeleton_mass[delete_point]
        for loop_nodes in circle:
            for node in loop_nodes:
                if min_mass > skeleton_mass[node]:
                    min_mass = skeleton_mass[node]
                    delete_point = node

        assigned_points = np.where(transport_plan[delete_point] > 0)[0]
        for point_idx in assigned_points:
            max_indices = np.argsort(transport_plan[:, point_idx])[::-1]
            if max_indices[0] == delete_point and len(max_indices) > 1:
                target = max_indices[1]
            else:
                target = max_indices[0]
            transport_plan[target, point_idx] += transport_plan[delete_point, point_idx]

        keep_mask = np.ones(len(skeleton_points), dtype=bool)
        keep_mask[delete_point] = False
        skeleton_points = skeleton_points[keep_mask]
        transport_plan = transport_plan[keep_mask]
        skeleton_mass = normalize_mass(np.sum(transport_plan, axis=1))

    return skeleton_points, skeleton_mass
