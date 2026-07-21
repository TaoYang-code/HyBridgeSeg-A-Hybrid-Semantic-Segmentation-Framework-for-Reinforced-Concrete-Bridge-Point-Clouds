from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(points: np.ndarray, eps: float = 0.1, min_samples: int = 1) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)


def extract_largest_non_noise_cluster(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = unique_labels >= 0
    if not np.any(valid_mask):
        return np.empty((0, points.shape[1])), -1

    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    largest_cluster_label = int(valid_labels[np.argmax(valid_counts)])
    return points[labels == largest_cluster_label], largest_cluster_label
