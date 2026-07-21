from __future__ import annotations

import numpy as np


def delete_degree_one_edges(adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.copy()
    for _ in range(adjacency.shape[0]):
        for node in range(adjacency.shape[0]):
            neighbors = np.where(adjacency[node] > 0)[0]
            if len(neighbors) == 1:
                adjacency[node, neighbors[0]] = 0
                adjacency[neighbors[0], node] = 0
    return adjacency


def delete_degree_two_loop_nodes(loop_nodes: list[int] | np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.copy()
    for node in loop_nodes:
        neighbors = np.where(adjacency[node] > 0)[0]
        if len(neighbors) == 2:
            adjacency[node, :] = 0
            adjacency[:, node] = 0
    return adjacency


def find_nonzero_degree_nodes(adjacency: np.ndarray) -> np.ndarray:
    _, cols = np.nonzero(adjacency)
    unique_cols = np.unique(cols)
    if len(unique_cols) == 0:
        return unique_cols
    return np.random.permutation(unique_cols)


def find_loop(start_nodes: np.ndarray, adjacency: np.ndarray) -> list[int]:
    if len(start_nodes) == 0:
        return []

    stack = [int(start_nodes[0])]
    visited_order = [0]
    loop: list[int] = []

    while stack:
        previous_len = len(stack)
        current = stack[-1]
        for index, neighbor in enumerate(start_nodes):
            if adjacency[current, neighbor] == 1 and index in visited_order:
                loop_start = next((i for i, value in enumerate(stack) if value == neighbor), None)
                if loop_start is not None and loop_start != len(stack) - 2:
                    loop = stack[loop_start:]
                continue
            if adjacency[current, neighbor] == 1 and index not in visited_order:
                stack.append(int(neighbor))
                visited_order.append(index)
                break

        if previous_len == len(stack):
            stack.pop()

    return loop


def find_branch_path(adjacency: np.ndarray, leaf: int, remaining_indices: np.ndarray) -> np.ndarray:
    path = [int(leaf)]
    remaining = [idx for idx in remaining_indices if idx != leaf]
    step_count = 1
    while remaining:
        step_count += 1
        if step_count > 20:
            break
        current = path[-1]
        neighbors = np.where(adjacency[current] == 1)[0]
        if len(neighbors) > 2:
            break
        for neighbor in neighbors:
            if neighbor in remaining:
                path.append(int(neighbor))
                remaining.remove(int(neighbor))
                break
        else:
            break
    return np.asarray(path, dtype=int)
