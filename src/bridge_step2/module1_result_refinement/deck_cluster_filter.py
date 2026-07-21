from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(points: np.ndarray, eps: float = 0.1, min_samples: int = 1) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)


def extract_largest_cluster(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0:
        return np.empty((0, points.shape[1])), -1

    largest_cluster_label = int(unique_labels[np.argmax(counts)])
    return points[labels == largest_cluster_label], largest_cluster_label
