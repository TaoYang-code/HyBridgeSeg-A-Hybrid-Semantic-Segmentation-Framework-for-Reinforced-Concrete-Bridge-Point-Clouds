from __future__ import annotations

import numpy as np

from .common import pairwise_distances


def center_points_with_line(
    points: np.ndarray,
    skeleton_points: np.ndarray,
    transport_plan: np.ndarray,
    neighbor_matrix: np.ndarray,
    weight_value: float,
) -> np.ndarray:
    lambda_value = weight_value
    distance = 9_999_999.0
    threshold_distance = 1e-4
    theta = 0.2
    skeleton = skeleton_points.copy()
    current = skeleton_points.copy()

    while distance > threshold_distance:
        distances = np.sqrt(np.maximum(pairwise_distances(current, points), 1e-6))
        distance = 0.0
        for i in range(transport_plan.shape[0]):
            weights = transport_plan[i] / distances[i]
            normalizer = np.sum(weights)
            xyz = np.sum(weights[:, None] * points, axis=0)

            neighbors_one = np.where(neighbor_matrix[i] > 0.5)[0]
            num_neighbors_one = len(neighbors_one)
            if num_neighbors_one > 1:
                normalizer += 2 * lambda_value
            for neighbor in neighbors_one:
                neighbors_two = np.where(neighbor_matrix[neighbor] > 0.5)[0]
                num_neighbors_two = len(neighbors_two)
                if num_neighbors_two > 1:
                    normalizer += 2 * lambda_value / num_neighbors_two / num_neighbors_two

            if num_neighbors_one > 1:
                neighbor_sum = np.sum(current[neighbors_one], axis=0)
                xyz += neighbor_sum * 2 * lambda_value / num_neighbors_one

            for neighbor in neighbors_one:
                neighbors_two = np.where(neighbor_matrix[neighbor] > 0.5)[0]
                num_neighbors_two = len(neighbors_two)
                if num_neighbors_two > 1:
                    second_sum = np.sum([current[idx] for idx in neighbors_two if idx != i], axis=0)
                    second_avg = second_sum / num_neighbors_two
                    offset = (current[neighbor] - second_avg) / num_neighbors_two
                    xyz += 2 * lambda_value * offset

            updated_point = (1 - theta) * current[i] + theta * xyz / normalizer
            distance += float(np.linalg.norm(updated_point - current[i]))
            skeleton[i] = updated_point

        current = skeleton.copy()

    return skeleton
