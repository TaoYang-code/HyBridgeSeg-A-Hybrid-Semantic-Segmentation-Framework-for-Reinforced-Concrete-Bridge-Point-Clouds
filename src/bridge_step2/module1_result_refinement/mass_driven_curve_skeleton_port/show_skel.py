from __future__ import annotations

import numpy as np


def show_skeleton(points: np.ndarray, skeleton_points: np.ndarray, transport_plan: np.ndarray) -> np.ndarray:
    del points, skeleton_points
    connect_relation_value = 0.12
    adjacency = np.zeros((transport_plan.shape[0], transport_plan.shape[0]), dtype=int)
    shared_mass = np.zeros_like(adjacency, dtype=float)

    for point_index in range(transport_plan.shape[1]):
        order = np.argsort(transport_plan[:, point_index])[::-1]
        if len(order) < 2:
            continue
        i, j = sorted((int(order[0]), int(order[1])))
        shared_mass[i, j] += transport_plan[order[0], point_index] + transport_plan[order[1], point_index]

    skeleton_mass = np.sum(transport_plan, axis=1)
    for i in range(len(skeleton_mass) - 1):
        for j in range(i + 1, len(skeleton_mass)):
            if shared_mass[i, j] > connect_relation_value * (skeleton_mass[i] + skeleton_mass[j]):
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return adjacency
