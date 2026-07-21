from __future__ import annotations

import numpy as np


def test_value(
    points: np.ndarray,
    skeleton_point: np.ndarray,
    eigenvectors: np.ndarray,
    indices: np.ndarray,
    transport_plan: np.ndarray,
    skeleton_index: int,
) -> tuple[float, float, float]:
    transport_row = transport_plan[skeleton_index]
    values = []
    axis_pairs = ((0, 1), (0, 2), (1, 2))
    for first, second in axis_pairs:
        left = 0.0
        right = 0.0
        a = eigenvectors[:, first]
        b = eigenvectors[:, second]
        cross_ab = np.cross(a, b)
        for idx in indices:
            vector = skeleton_point - points[idx]
            flag = float(np.dot(vector, cross_ab))
            if flag < 0:
                left += transport_row[idx]
            else:
                right += transport_row[idx]
        values.append(abs(right - left) / max(np.sum(transport_row), 1e-12))
    return float(values[0]), float(values[1]), float(values[2])


def test_value_new(
    points: np.ndarray,
    skeleton_point: np.ndarray,
    eigenvectors: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    local_points = points[indices]
    normalized_vectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    centered_points = local_points - skeleton_point[None, :]
    mass = np.full(len(local_points), 1.0 / max(len(local_points), 1), dtype=float)

    results = []
    for axis in range(3):
        positive = 0.0
        negative = 0.0
        direction = normalized_vectors[:, axis]
        projections = centered_points @ direction
        for index, projection in enumerate(projections):
            if projection > 0:
                positive += projection * mass[index]
            else:
                negative -= projection * mass[index]
        denominator = abs(positive + negative)
        results.append(0.0 if denominator < 1e-12 else abs(positive - negative) / denominator)
    return float(results[0]), float(results[1]), float(results[2])


def test_symmetry(points: np.ndarray, skeleton_points: np.ndarray, transport_plan: np.ndarray) -> np.ndarray:
    result = []
    threshold = (1.0 / max(len(points), 1)) * 0.01
    for skeleton_index in range(len(skeleton_points)):
        support_indices = np.where(transport_plan[skeleton_index] > threshold)[0]
        support_points = points[support_indices]
        if len(support_points) == 0:
            result.append(0.0)
            continue

        centered = support_points - skeleton_points[skeleton_index]
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        first = sorted(test_value(points, skeleton_points[skeleton_index], eigenvectors, support_indices, transport_plan, skeleton_index), reverse=True)
        second = sorted(test_value_new(points, skeleton_points[skeleton_index], eigenvectors, support_indices), reverse=True)
        result.append(max(first[1], second[1]) if len(first) > 1 and len(second) > 1 else 0.0)
    return np.asarray(result, dtype=float)
