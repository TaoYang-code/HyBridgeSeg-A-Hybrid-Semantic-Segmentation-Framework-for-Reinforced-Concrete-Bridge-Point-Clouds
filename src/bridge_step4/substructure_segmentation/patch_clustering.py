from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .io_utils import load_point_cloud_txt, save_point_cloud_txt, scene_directories


@dataclass(frozen=True)
class PatchClusteringConfig:
    input_file_name: str = "sub.txt"
    output_dir_name: str = "sub_patch"
    voxel_size: float = 0.4
    dbscan_eps: float = 3
    dbscan_min_samples: int = 1
    create_manual_review_dirs: bool = True
    manual_review_dir_names: Tuple[str, ...] = ("abutment", "only_pier", "pier_piercap")


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points with Open3D voxel sampling."""
    if points.size == 0:
        return np.empty((0, 3), dtype=float)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled.points, dtype=float)


def cluster_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster points with DBSCAN."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[:, :3])


def project_labels_to_original_points(
    original_points: np.ndarray,
    downsampled_points: np.ndarray,
    downsampled_labels: np.ndarray,
) -> np.ndarray:
    """Transfer cluster labels from downsampled points back to original points."""
    if original_points.size == 0:
        return np.empty((0,), dtype=int)
    if downsampled_points.size == 0:
        return np.full(len(original_points), -1, dtype=int)

    tree = cKDTree(downsampled_points[:, :3])
    _, indices = tree.query(original_points[:, :3], k=1)
    return downsampled_labels[indices]


def ensure_manual_review_dirs(output_dir: Path, config: PatchClusteringConfig) -> None:
    """Create directories used for manual downstream patch classification."""
    if not config.create_manual_review_dirs:
        return
    for dir_name in config.manual_review_dir_names:
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)


def save_cluster_files(points: np.ndarray, labels: np.ndarray, output_dir: Path) -> int:
    """Save each non-noise cluster to ``sub_<label>.txt``."""
    cluster_count = 0
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        save_point_cloud_txt(output_dir / f"sub_{label}.txt", cluster_points)
        cluster_count += 1
    return cluster_count


def process_scene(scene_dir: Path, config: PatchClusteringConfig) -> bool:
    """Cluster one scene's substructure points into coarse sub-patches."""
    load_time = 0.0
    input_file = scene_dir / config.input_file_name
    if not input_file.exists():
        print(f"Skipping {scene_dir.name}: missing {config.input_file_name}.")
        return False

    scene_start_time = perf_counter()
    load_start_time = perf_counter()
    points = load_point_cloud_txt(input_file)
    load_time += perf_counter() - load_start_time
    if points.size == 0:
        print(f"Skipping {scene_dir.name}: empty {config.input_file_name}.")
        return False

    output_dir = scene_dir / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manual_review_dirs(output_dir, config)

    voxel_start_time = perf_counter()
    downsampled_points = voxel_downsample(points, voxel_size=config.voxel_size)
    voxel_time = perf_counter() - voxel_start_time

    clustering_start_time = perf_counter()
    downsampled_labels = cluster_points(
        downsampled_points,
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
    )
    labels = project_labels_to_original_points(points, downsampled_points, downsampled_labels)
    clustering_time = perf_counter() - clustering_start_time

    save_start_time = perf_counter()
    cluster_count = save_cluster_files(points, labels, output_dir)
    save_time = perf_counter() - save_start_time

    total_time = perf_counter() - scene_start_time
    print(
        f"Clustered scene: {scene_dir.name} "
        f"(points={len(points)}, downsampled={len(downsampled_points)}, clusters={cluster_count}, "
        f"load={load_time:.2f}s, voxel={voxel_time:.2f}s, clustering={clustering_time:.2f}s, "
        f"save={save_time:.2f}s, total={total_time:.2f}s)"
    )
    return True


def run_substructure_patch_clustering(input_root: Path, config: Optional[PatchClusteringConfig] = None) -> None:
    """Run substructure patch clustering for every scene under the input root."""
    clustering_config = config or PatchClusteringConfig()
    input_root = Path(input_root)

    print(f"Running substructure patch clustering on: {input_root}")
    for scene_dir in scene_directories(input_root):
        process_scene(scene_dir, clustering_config)
