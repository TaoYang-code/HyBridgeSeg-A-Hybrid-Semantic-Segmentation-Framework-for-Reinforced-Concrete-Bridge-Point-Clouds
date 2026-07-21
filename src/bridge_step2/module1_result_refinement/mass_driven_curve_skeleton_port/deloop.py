from __future__ import annotations

import numpy as np

from .center import center_points
from .merging_circle import merge_circle
from .ot import compute_optimal_transport
from .show_skel import show_skeleton
from .subgraphs import sub_graphs
from .symmetry import test_symmetry
from .loops import find_circles


def deloop(
    seed_indices: np.ndarray,
    transport_plan: np.ndarray,
    skeleton_points: np.ndarray,
    adjacency: np.ndarray,
    skeleton_mass: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del seed_indices
    class_threshold = 0.15
    condition_refine = True

    while condition_refine:
        result = test_symmetry(points, skeleton_points, transport_plan)
        tmp_adjacency, class_a, class_groups, _ = sub_graphs(adjacency, result, class_threshold)
        del tmp_adjacency, class_a
        bool_circle, circle_index, circle, bool_nocircle, nocircle_index, nocircle = find_circles(
            class_groups,
            skeleton_points,
            adjacency,
        )
        del circle_index, nocircle_index
        if not bool_circle:
            condition_refine = False

        if bool_circle or bool_nocircle:
            skeleton_points, skeleton_mass = merge_circle(
                skeleton_points,
                skeleton_mass,
                nocircle,
                circle,
                bool_circle,
                bool_nocircle,
                transport_plan,
            )
            transport_previous = 9_999_999.0
            threshold = 1e-4
            lambda_value = 500.0
            point_mass = np.ones((len(points), 1), dtype=float) / max(len(points), 1)
            while True:
                transport_plan, transport_value = compute_optimal_transport(
                    points,
                    skeleton_points,
                    point_mass,
                    skeleton_mass[:, None],
                    lambda_value,
                )
                skeleton_points = center_points(points, skeleton_points, transport_plan)
                if abs(transport_previous - transport_value) < threshold:
                    break
                transport_previous = transport_value
            adjacency = show_skeleton(points, skeleton_points, transport_plan)

    return transport_plan, skeleton_points, adjacency, skeleton_mass
