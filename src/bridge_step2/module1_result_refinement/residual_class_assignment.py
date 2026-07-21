from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .io_utils import save_xyz_points, voxel_downsample_numpy


RESIDUAL_ASSIGNMENT_VOXEL_SIZE = 0.07


def assign_to_nearest_class(
    unclassified_points: np.ndarray,
    classified_points: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if len(unclassified_points) == 0:
        return np.empty((0,), dtype=int)

    kd_tree = cKDTree(classified_points)
    _, indices = kd_tree.query(unclassified_points, k=1)
    return labels[np.asarray(indices).flatten()]


def assign_residual_groups(
    deck_points: np.ndarray,
    superstructure_points: np.ndarray,
    substructure_points: np.ndarray,
    residual_substructure_points: np.ndarray,
    residual_superstructure_points: np.ndarray,
    output_folder: str | Path | None = None,
    save_outputs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    downsampled_deck_points = voxel_downsample_numpy(deck_points, voxel_size=RESIDUAL_ASSIGNMENT_VOXEL_SIZE)
    downsampled_superstructure_points = voxel_downsample_numpy(
        superstructure_points,
        voxel_size=RESIDUAL_ASSIGNMENT_VOXEL_SIZE,
    )
    downsampled_substructure_points = voxel_downsample_numpy(
        substructure_points,
        voxel_size=RESIDUAL_ASSIGNMENT_VOXEL_SIZE,
    )

    classified_points = np.vstack(
        (
            downsampled_deck_points,
            downsampled_superstructure_points,
            downsampled_substructure_points,
        )
    )
    labels = np.concatenate(
        (
            np.full(len(downsampled_deck_points), 1, dtype=int),
            np.full(len(downsampled_superstructure_points), 2, dtype=int),
            np.full(len(downsampled_substructure_points), 3, dtype=int),
        )
    )

    residual_groups = [group for group in (residual_substructure_points, residual_superstructure_points) if len(group) > 0]
    residual_points = np.vstack(residual_groups) if residual_groups else np.empty((0, deck_points.shape[1]))

    downsampled_residual_points = voxel_downsample_numpy(
        residual_points,
        voxel_size=RESIDUAL_ASSIGNMENT_VOXEL_SIZE,
    )
    downsampled_assigned_labels = assign_to_nearest_class(
        downsampled_residual_points,
        classified_points,
        labels,
    )

    if len(residual_points) > 0 and len(downsampled_residual_points) > 0:
        residual_kd_tree = cKDTree(downsampled_residual_points)
        _, residual_to_downsampled_indices = residual_kd_tree.query(residual_points, k=1)
        assigned_labels = downsampled_assigned_labels[np.asarray(residual_to_downsampled_indices).flatten()]
    else:
        assigned_labels = np.empty((0,), dtype=int)

    updated_deck_points = np.vstack((deck_points, residual_points[assigned_labels == 1]))
    updated_superstructure_points = np.vstack((superstructure_points, residual_points[assigned_labels == 2]))
    updated_substructure_points = np.vstack((substructure_points, residual_points[assigned_labels == 3]))

    if output_folder is not None and save_outputs:
        output_folder = Path(output_folder)
        save_xyz_points(updated_deck_points, output_folder / "deck.txt")
        save_xyz_points(updated_superstructure_points, output_folder / "sup.txt")
        save_xyz_points(updated_substructure_points, output_folder / "sub.txt")

    return updated_deck_points, updated_superstructure_points, updated_substructure_points
