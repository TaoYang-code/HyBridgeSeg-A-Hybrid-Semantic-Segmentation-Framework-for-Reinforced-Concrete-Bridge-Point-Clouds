from __future__ import annotations

import numpy as np

from .common import pairwise_distances
from .graph_utils import find_branch_path


def _interpolate_segment(segment_points: np.ndarray, n: int) -> list[np.ndarray]:
    if len(segment_points) <= 1:
        return [segment_points]

    distance_matrix = pairwise_distances(segment_points, segment_points)
    total_length = float(np.sum(np.diag(distance_matrix, k=1)))
    nonzero = distance_matrix[distance_matrix > 0]
    if len(nonzero) == 0:
        return [segment_points]
    min_distance = float(np.min(nonzero))
    spacing = total_length / max(int(np.floor(total_length / (min_distance * (1 / n)))), 1)
    result_parts: list[np.ndarray] = []
    for idx in range(len(segment_points) - 1):
        p1 = segment_points[idx] if idx == 0 else result_parts[-1][-1]
        p2 = segment_points[idx + 1]
        segment_length = float(np.linalg.norm(p2 - p1))
        if segment_length < 1e-12:
            interpolated = p2[None, :]
        else:
            step = spacing / segment_length
            alpha = np.arange(0, 1 + step, step)
            interpolated = p1[None, :] + alpha[:, None] * (p2 - p1)[None, :]
        if idx > 0 and len(interpolated) > 0:
            interpolated = interpolated[1:]
        if idx == len(segment_points) - 2:
            interpolated = np.vstack((interpolated, p2))
        if len(interpolated) > 1 and np.allclose(interpolated[-1], interpolated[-2]):
            interpolated = interpolated[:-1]
        result_parts.append(interpolated)
    return result_parts


def interpolate_skeleton(adjacency: np.ndarray, skeleton_points: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    result_skeleton = []
    result_adjacency = np.zeros((0, 0), dtype=int)
    leaves = [node for node in range(adjacency.shape[0]) if np.sum(adjacency[node] > 0) == 1]
    index = np.arange(adjacency.shape[0])

    cskel = skeleton_points.copy()
    cadj = adjacency.copy()
    trunk_size = 0
    for leaf in leaves:
        branch = find_branch_path(adjacency, leaf, index)
        for node in branch[:-1]:
            cskel[node] = 0
            cadj[node, :] = 0
            cadj[:, node] = 0

    trunk_nodes = np.where(cskel[:, 0] != 0)[0]
    trunk_path = []
    if len(trunk_nodes) > 1:
        start = next((node for node in trunk_nodes if np.sum(cadj[node] == 1) == 1), int(trunk_nodes[0]))
        trunk_path = [start]
        while len(trunk_path) < len(trunk_nodes):
            neighbors = list(np.where(cadj[trunk_path[-1]] == 1)[0])
            for neighbor in neighbors:
                if neighbor not in trunk_path:
                    trunk_path.append(int(neighbor))
                    break
            else:
                break
        trunk_parts = _interpolate_segment(skeleton_points[np.asarray(trunk_path)], n)
        for part in trunk_parts:
            result_skeleton.append(part)
        trunk_size = len(np.vstack(result_skeleton)) if result_skeleton else 0

    snode: list[int] = []
    lnode: list[int] = []
    for leaf in leaves:
        branch = find_branch_path(adjacency, leaf, index)
        branch_parts = _interpolate_segment(skeleton_points[branch], n)
        for part in branch_parts:
            result_skeleton.append(part)
        merged = np.vstack(result_skeleton)
        duplicate = np.where(np.all(merged == merged[-1], axis=1))[0]
        duplicate = duplicate[:-1]
        if len(duplicate) > 0:
            snode.extend(duplicate.tolist())
            merged = merged[:-1]
            lnode.append(len(merged) - 1)
        result_skeleton = [merged]

    merged_skeleton = np.vstack(result_skeleton) if result_skeleton else skeleton_points.copy()
    result_adjacency = np.zeros((len(merged_skeleton), len(merged_skeleton)), dtype=int)
    unique_snode = set(snode)
    lnode_set = set(lnode)
    for idx in range(len(merged_skeleton) - 1):
        if idx in lnode_set:
            continue
        if idx >= trunk_size and idx in unique_snode:
            continue
        result_adjacency[idx, idx + 1] = 1
        result_adjacency[idx + 1, idx] = 1
    for source, target in zip(snode, lnode):
        if source < len(merged_skeleton) and target < len(merged_skeleton):
            result_adjacency[source, target] = 1
            result_adjacency[target, source] = 1
    return merged_skeleton, result_adjacency
