from __future__ import annotations

import numpy as np


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def normalize_mass(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return values / total
