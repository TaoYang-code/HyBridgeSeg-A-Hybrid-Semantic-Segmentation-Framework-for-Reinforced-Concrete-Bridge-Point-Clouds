from __future__ import annotations

import numpy as np


def dfs_subgraph(adjacency: np.ndarray, result: np.ndarray, class_threshold: float) -> tuple[np.ndarray, list[int]]:
    solution: list[int] = []
    if not np.any(result >= class_threshold):
        return np.empty((0, 2), dtype=int), solution

    start_index = int(np.where(result >= class_threshold)[0][0])
    stack = [start_index]
    visited = {start_index}
    edges: list[tuple[int, int]] = []

    while stack:
        previous_len = len(stack)
        current = stack[-1]
        for neighbor in range(len(result)):
            if adjacency[current, neighbor] == 1 and neighbor not in visited and result[neighbor] >= class_threshold:
                stack.append(neighbor)
                visited.add(neighbor)
                edges.append((current, neighbor))
                break
        if len(edges) == 0:
            result[current] = 0
            solution.append(current)
        if len(stack) == previous_len:
            stack.pop()

    return np.asarray(edges, dtype=int), solution


def sub_graphs(
    adjacency: np.ndarray,
    result: np.ndarray,
    class_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], int]:
    current_result = result.copy()
    tmp_adjacency = np.zeros_like(adjacency, dtype=int)
    class_a: list[int] = []
    class_groups: list[list[int]] = []

    while True:
        edges, isolated = dfs_subgraph(adjacency, current_result, class_threshold)
        if len(edges) == 0 and not isolated:
            break
        if len(edges) > 0:
            nodes = [int(edges[0, 0]), int(edges[0, 1])]
            tmp_adjacency[edges[0, 0], edges[0, 1]] = 1
            tmp_adjacency[edges[0, 1], edges[0, 0]] = 1
            for edge in edges[1:]:
                if int(edge[1]) not in nodes:
                    tmp_adjacency[edge[0], edge[1]] = 1
                    tmp_adjacency[edge[1], edge[0]] = 1
                    nodes.append(int(edge[1]))
            for node_i in nodes:
                for node_j in nodes:
                    if adjacency[node_i, node_j] == 1:
                        tmp_adjacency[node_i, node_j] = 1
                        tmp_adjacency[node_j, node_i] = 1
            for node in nodes:
                current_result[node] = 0
            class_a.extend(nodes)
            class_groups.append(nodes)
        else:
            node = int(isolated[0])
            current_result[node] = 0
            class_a.append(node)
            class_groups.append([node])

    class_a_unique = np.unique(class_a)
    bool_class_num = int(len(class_a_unique) == len(result) or len(class_a_unique) == 0)

    if len(class_a_unique) > 0 and not bool_class_num:
        class_b: list[int] = []
        for node in class_a_unique:
            neighbors = np.where(adjacency[node] != 0)[0]
            for neighbor in neighbors:
                if current_result[neighbor] < class_threshold and current_result[neighbor] > 0 and neighbor not in class_b:
                    class_b.append(int(neighbor))

        for node in class_b:
            neighbors = np.where(adjacency[node] != 0)[0]
            for neighbor in neighbors:
                if current_result[neighbor] == 0:
                    tmp_adjacency[node, neighbor] = 1
                    tmp_adjacency[neighbor, node] = 1

        enriched_groups: list[list[int]] = []
        for group in class_groups:
            extra_nodes: list[int] = []
            for node in group:
                neighbors = np.where(tmp_adjacency[node] != 0)[0]
                for neighbor in neighbors:
                    if 0 < current_result[neighbor] < class_threshold:
                        extra_nodes.append(int(neighbor))
            enriched_groups.append(group + extra_nodes)
        class_groups = enriched_groups

    return tmp_adjacency, class_a_unique, class_groups, bool_class_num
