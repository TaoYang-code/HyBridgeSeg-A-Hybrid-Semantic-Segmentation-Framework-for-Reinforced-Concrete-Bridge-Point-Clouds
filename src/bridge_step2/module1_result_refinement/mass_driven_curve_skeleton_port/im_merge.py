from __future__ import annotations

import numpy as np

from .common import normalize_mass, pairwise_distances
from .find_nei import find_nearest_indices


def im_merge(
    points: np.ndarray,
    transport_plan: np.ndarray,
    skeleton_points: np.ndarray,
    point_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    skeleton_count = skeleton_points.shape[0]
    mass_skeleton = np.zeros(skeleton_count, dtype=float)
    max_number_skeleton = np.zeros(skeleton_count, dtype=float)
    line_number_skeleton = np.zeros(skeleton_count, dtype=float)
    skeleton_index = np.zeros(skeleton_count, dtype=int)
    threshold_skeleton_point = 0.92
    relation_skeleton = np.zeros((skeleton_count, skeleton_count), dtype=int)
    need_delete = False

    for point_index in range(transport_plan.shape[1]):
        threshold_transport = point_mass[point_index] * 0.001
        current_transport = transport_plan[:, point_index]
        active = np.where(current_transport > threshold_transport)[0]
        if len(active) == 0:
            continue
        masses = np.column_stack((active, current_transport[active]))
        if 1 < len(active) < 3:
            max_idx = int(masses[np.argmax(masses[:, 1]), 0])
            max_number_skeleton[max_idx] += 1
            line_number_skeleton[max_idx] += 1
            other = [idx for idx in active if idx != max_idx]
            if other:
                relation_skeleton[max_idx, other[0]] = 1
                relation_skeleton[other[0], max_idx] = 1
        elif len(active) > 2:
            order = active[np.argsort(current_transport[active])[::-1]]
            max_number_skeleton[order[0]] += 1
            side_c = np.sum((skeleton_points[order[1]] - skeleton_points[order[2]]) ** 2)
            side_a = np.sum((skeleton_points[order[1]] - skeleton_points[order[0]]) ** 2)
            side_b = np.sum((skeleton_points[order[2]] - skeleton_points[order[0]]) ** 2)
            cos_angle = (side_a + side_b - side_c) / (2 * np.sqrt(side_a) * np.sqrt(side_b))
            if cos_angle < -0.9:
                line_number_skeleton[order[0]] += 1
            relation_skeleton[order[0], order[1]] = 1
            relation_skeleton[order[0], order[2]] = 1

    distances = pairwise_distances(skeleton_points, points)
    transport_cost = transport_plan * distances
    cost = np.sum(transport_cost, axis=1)
    for index in range(skeleton_count):
        mass_skeleton[index] = np.sum(transport_plan[index])
        if max_number_skeleton[index] > 0 and line_number_skeleton[index] / max_number_skeleton[index] > threshold_skeleton_point:
            skeleton_index[index] = 1

    non_skeleton_points = np.where(skeleton_index == 0)[0]
    delete_points: list[int] = []
    merge_pairs: list[tuple[int, int]] = []
    sorted_non_skeleton = non_skeleton_points[np.argsort(mass_skeleton[non_skeleton_points])]

    for candidate in sorted_non_skeleton:
        candidate_neighbors = np.where(relation_skeleton[candidate] == 1)[0]
        candidate_neighbors = np.array([idx for idx in candidate_neighbors if idx in non_skeleton_points], dtype=int)
        if len(candidate_neighbors) == 0:
            continue
        nearest_idx = find_nearest_indices(skeleton_points[candidate][None, :], skeleton_points[candidate_neighbors])[0]
        pair_index = int(candidate_neighbors[nearest_idx])
        if pair_index in delete_points:
            continue
        p1 = skeleton_points[candidate]
        p2 = skeleton_points[pair_index]
        m1 = mass_skeleton[candidate]
        m2 = mass_skeleton[pair_index]
        skeleton_points[pair_index] = (p1 * m1 + p2 * m2) / (m1 + m2)
        mass_skeleton[pair_index] = m1 + m2
        delete_points.append(int(candidate))
        merge_pairs.append((int(candidate), int(pair_index)))
        need_delete = True

    if need_delete:
        for delete_index in delete_points:
            assigned_points = np.where(transport_plan[delete_index] > 0)[0]
            for point_idx in assigned_points:
                max_indices = np.argsort(transport_plan[:, point_idx])[::-1]
                if max_indices[0] == delete_index and len(max_indices) > 1:
                    target = max_indices[1]
                else:
                    target = max_indices[0]
                transport_plan[target, point_idx] += transport_plan[delete_index, point_idx]
        keep_mask = np.ones(skeleton_count, dtype=bool)
        keep_mask[delete_points] = False
        transport_plan = transport_plan[keep_mask]
        skeleton_points = skeleton_points[keep_mask]
        mass_skeleton = normalize_mass(np.sum(transport_plan, axis=1))
    else:
        mass_skeleton = normalize_mass(np.sum(transport_plan, axis=1))

    return skeleton_points, mass_skeleton, transport_plan, np.asarray(merge_pairs, dtype=int), non_skeleton_points
