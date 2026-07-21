from __future__ import annotations

import numpy as np

from .graph_utils import delete_degree_one_edges, delete_degree_two_loop_nodes, find_loop, find_nonzero_degree_nodes


def find_circles(
    class_groups: list[list[int]],
    skeleton_points: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[bool, list[int], list[list[int]], bool, list[int], list[list[int]]]:
    del skeleton_points
    local_adjacency = adjacency.copy()
    nocircle_index = []
    circle: list[list[int]] = []
    circle_index: list[int] = []

    for group_index, group in enumerate(class_groups):
        nocircle_index.append(group_index)
        group_adjacency = np.zeros_like(local_adjacency)
        for node_i in group:
            for node_j in group:
                if local_adjacency[node_i, node_j] == 1:
                    group_adjacency[node_i, node_j] = 1

        loop: list[int] = []
        for _ in range(adjacency.shape[0]):
            group_adjacency = delete_degree_two_loop_nodes(loop, group_adjacency)
            group_adjacency = delete_degree_one_edges(group_adjacency)
            if not np.any(group_adjacency):
                break
            nonzero_nodes = find_nonzero_degree_nodes(group_adjacency)
            loop = find_loop(nonzero_nodes, group_adjacency)
            if loop:
                circle.append(loop)
                circle_index.append(group_index)

    bool_circle = len(circle) > 0
    remaining_indices = [index for index in nocircle_index if index not in circle_index]
    nocircle = [class_groups[index] for index in remaining_indices]
    bool_nocircle = len(nocircle) > 0
    return bool_circle, circle_index, circle, bool_nocircle, remaining_indices, nocircle
