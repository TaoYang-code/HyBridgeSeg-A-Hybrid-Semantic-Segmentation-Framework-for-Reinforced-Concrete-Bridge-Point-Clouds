from __future__ import annotations

import numpy as np

from .common import pairwise_distances


def find_nearest_indices(points: np.ndarray, skeleton_points: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(points, skeleton_points)
    return np.argmin(distances, axis=1)
