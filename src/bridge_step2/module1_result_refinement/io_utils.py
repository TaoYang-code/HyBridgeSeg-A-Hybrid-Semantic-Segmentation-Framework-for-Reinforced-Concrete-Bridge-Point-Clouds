from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_xyz_points(file_path: str | Path) -> np.ndarray:
    return np.loadtxt(file_path, delimiter=" ")[:, :3]


def save_xyz_points(points: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, points, delimiter=" ")


def split_module1_labels(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deck_points = data[data[:, 3] == 1][:, :3]
    substructure_points = data[data[:, 3] == 2][:, :3]
    superstructure_points = data[data[:, 3] == 3][:, :3]
    return deck_points, substructure_points, superstructure_points


def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        feature_dim = points.shape[1] if points.ndim == 2 else 3
        return np.empty((0, feature_dim))

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    downsampled_points = np.zeros((len(unique_voxels), points.shape[1]), dtype=float)
    counts = np.bincount(inverse)

    for axis in range(points.shape[1]):
        downsampled_points[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts

    return downsampled_points


def map_labels_from_downsampled_points(
    original_points: np.ndarray,
    downsampled_points: np.ndarray,
    downsampled_labels: np.ndarray,
) -> np.ndarray:
    if len(original_points) == 0:
        return np.empty((0,), dtype=downsampled_labels.dtype)
    if len(downsampled_points) == 0:
        return np.empty((0,), dtype=downsampled_labels.dtype)

    nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(downsampled_points)
    _, indices = nearest_neighbors.kneighbors(original_points)
    return downsampled_labels[indices.flatten()]
