from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_point_cloud_txt(file_path: Path) -> np.ndarray:
    """Load an XYZ point cloud text file as an ``(N, 3)`` float array."""
    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    try:
        points = np.loadtxt(file_path, dtype=float)
    except ValueError:
        return np.empty((0, 3), dtype=float)

    if points.size == 0:
        return np.empty((0, 3), dtype=float)

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {file_path}, got {points.shape[1]}")

    return points[:, :3]


def save_point_cloud_txt(file_path: Path, points: np.ndarray) -> None:
    """Save an XYZ point cloud using a consistent text format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        file_path.write_text("")
        return
    np.savetxt(file_path, points[:, :3], fmt="%.6f", delimiter=" ")


def scene_directories(input_root: Path) -> List[Path]:
    """Return sorted scene directories under the input root."""
    return sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda path: path.name)
