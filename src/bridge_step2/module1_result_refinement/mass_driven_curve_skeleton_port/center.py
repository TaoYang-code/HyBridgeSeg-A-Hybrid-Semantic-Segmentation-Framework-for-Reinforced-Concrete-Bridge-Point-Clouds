from __future__ import annotations

import numpy as np

from .common import pairwise_distances


def center_points(points: np.ndarray, skeleton_points: np.ndarray, transport_plan: np.ndarray) -> np.ndarray:
    distance = 9_999_999.0
    threshold_distance = 0.1
    theta = 0.2
    skeleton = skeleton_points.copy()
    current = skeleton_points.copy()

    while distance > threshold_distance:
        distances = np.sqrt(np.maximum(pairwise_distances(current, points), 1e-6))
        distance = 0.0
        for index in range(transport_plan.shape[0]):
            weights = transport_plan[index] / distances[index]
            normalizer = np.sum(weights)
            updated_point = (1 - theta) * current[index] + theta * np.sum(weights[:, None] * points, axis=0) / normalizer
            distance += float(np.linalg.norm(updated_point - current[index]))
            skeleton[index] = updated_point
        current = skeleton.copy()

    return skeleton
