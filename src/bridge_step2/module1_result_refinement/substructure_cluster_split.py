from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(points: np.ndarray, eps: float = 0.1, min_samples: int = 1) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)


def split_by_cluster_size(
    points: np.ndarray,
    labels: np.ndarray,
    size_threshold: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    large_clusters: list[np.ndarray] = []
    small_clusters: list[np.ndarray] = []

    for label in unique_labels:
        cluster_points = points[labels == label]
        if len(cluster_points) >= size_threshold:
            large_clusters.append(cluster_points)
        else:
            small_clusters.append(cluster_points)

    large_points = np.vstack(large_clusters) if large_clusters else np.empty((0, points.shape[1]))
    small_points = np.vstack(small_clusters) if small_clusters else np.empty((0, points.shape[1]))
    return large_points, small_points
