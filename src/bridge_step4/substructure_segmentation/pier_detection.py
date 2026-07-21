from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .io_utils import save_point_cloud_txt, scene_directories


@dataclass(frozen=True)
class PierDetectionConfig:
    sub_patch_dir_name: str = "sub_patch"
    input_group_dir_name: str = "pier_piercap"
    output_dir_name: str = "pier_detection_result"
    max_workers: int = 5
    normal_max_nn: int = 32
    orient_normals_consistently: bool = False
    bottom_grid_size: float = 0.3
    bottom_dbscan_voxel_size: float = 0.1
    bottom_cluster_eps: float = 1#1.6
    bottom_cluster_min_samples: int = 1
    bottom_seed_height: float = 0.5
    region_angle_threshold_rad: float = np.pi / 18
    region_search_knn: int = 32
    axis_radius_percentile: float = 95
    axis_radius_scale: float = 1.05
    noise_dbscan_eps: float = 0.1
    noise_dbscan_min_samples: int = 20
    noise_point_threshold: int = 500


def load_point_cloud(file_path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from a text file."""
    points = np.loadtxt(file_path, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    return point_cloud


def compute_normals(
    point_cloud: o3d.geometry.PointCloud,
    max_nn: int,
    orient_normals_consistently: bool,
) -> o3d.geometry.PointCloud:
    """Estimate and orient point normals."""
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
    )
    if orient_normals_consistently:
        point_cloud.orient_normals_consistent_tangent_plane(k=max_nn)
    return point_cloud


def filter_bottom_grid_points(point_cloud: o3d.geometry.PointCloud, grid_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Keep XY grid cells whose minimum Z suggests contact with the lower part of the scene."""
    points = np.asarray(point_cloud.points)
    point_indices = np.arange(len(points))
    grid_indices = np.floor(points[:, :2] / grid_size).astype(int)
    unique_grids = np.unique(grid_indices, axis=0)

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_threshold = (z_max + z_min) / 2.0

    valid_grids = []
    for grid_index in unique_grids:
        mask = np.all(grid_indices == grid_index, axis=1)
        grid_points = points[mask]
        if grid_points.size == 0:
            continue
        if grid_points[:, 2].min() <= z_threshold - 1.0:
            valid_grids.append(tuple(grid_index.tolist()))

    valid_mask = np.array([tuple(row.tolist()) in valid_grids for row in grid_indices])
    return points[valid_mask], point_indices[valid_mask]


def cluster_bottom_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster filtered bottom points to separate different pier bases."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3]).labels_


def voxelize_points_with_inverse(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize points and return voxel centroids plus inverse indices to original points."""
    if points.size == 0 or voxel_size <= 0:
        return points[:, :3].copy(), np.arange(len(points), dtype=int)

    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    voxel_centroids = np.zeros((len(unique_voxels), 3), dtype=float)
    for voxel_id in range(len(unique_voxels)):
        voxel_centroids[voxel_id] = points[inverse_indices == voxel_id, :3].mean(axis=0)

    return voxel_centroids, inverse_indices


def cluster_bottom_points_with_voxelization(
    points: np.ndarray,
    voxel_size: float,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Cluster filtered bottom points after voxelization, then map labels back."""
    voxel_points, inverse_indices = voxelize_points_with_inverse(points, voxel_size=voxel_size)
    voxel_labels = cluster_bottom_points(voxel_points, eps=eps, min_samples=min_samples)
    return voxel_labels[inverse_indices]


def extract_seed_indices(
    filtered_points: np.ndarray,
    filtered_indices: np.ndarray,
    labels: np.ndarray,
    z_threshold: float,
) -> np.ndarray:
    """Extract low-elevation seed points from each bottom cluster."""
    seed_index_groups: List[np.ndarray] = []
    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster_points = filtered_points[cluster_mask]
        cluster_indices = filtered_indices[cluster_mask]
        if cluster_points.size == 0:
            continue
        z_min = cluster_points[:, 2].min()
        seed_mask = cluster_points[:, 2] < z_min + z_threshold
        seed_index_groups.append(cluster_indices[seed_mask])

    if not seed_index_groups:
        return np.empty((0,), dtype=int)
    return np.hstack(seed_index_groups)


def extract_seed_index_groups(
    filtered_points: np.ndarray,
    filtered_indices: np.ndarray,
    labels: np.ndarray,
    z_threshold: float,
) -> List[np.ndarray]:
    """Extract low-elevation seed points for each bottom cluster separately."""
    seed_index_groups: List[np.ndarray] = []
    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster_points = filtered_points[cluster_mask]
        cluster_indices = filtered_indices[cluster_mask]
        if cluster_points.size == 0:
            continue
        z_min = cluster_points[:, 2].min()
        seed_mask = cluster_points[:, 2] < z_min + z_threshold
        seed_indices = cluster_indices[seed_mask]
        if seed_indices.size > 0:
            seed_index_groups.append(seed_indices)
    return seed_index_groups


def dist_to_vertical_axis(points: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Compute XY distance from points to a vertical axis passing through center_xy."""
    dx = points[:, 0] - center_xy[0]
    dy = points[:, 1] - center_xy[1]
    return np.sqrt(dx * dx + dy * dy)


def estimate_pier_center_and_rmax(
    point_cloud: o3d.geometry.PointCloud,
    seed_indices: np.ndarray,
    percentile: float,
    scale: float,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Estimate one pier's XY center and a conservative radial growth limit from its bottom seeds."""
    points = np.asarray(point_cloud.points)
    seed_indices = np.asarray(seed_indices, dtype=int).ravel()
    if seed_indices.size == 0:
        return None, None

    seed_points = points[seed_indices]
    if seed_points.size == 0:
        return None, None

    center_xy = seed_points[:, :2].mean(axis=0)
    distances = dist_to_vertical_axis(seed_points, center_xy)
    radius = float(np.percentile(distances, percentile)) * scale
    return center_xy, radius


def region_grow_pier(
    point_cloud: o3d.geometry.PointCloud,
    seed_indices: np.ndarray,
    angle_threshold_rad: float,
    search_knn: int,
) -> np.ndarray:
    """Region-grow pier-like points from bottom seeds using normal similarity."""
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    labels = np.full(len(points), -1, dtype=int)
    queue: deque[int] = deque(seed_indices.tolist())

    for seed_index in seed_indices:
        labels[seed_index] = 0

    while queue:
        current_index = queue.popleft()
        current_normal = normals[current_index]
        _, neighbor_indices, _ = tree.search_knn_vector_3d(points[current_index], search_knn)

        for neighbor_index in neighbor_indices:
            if labels[neighbor_index] != -1:
                continue
            neighbor_normal = normals[neighbor_index]
            angle = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1.0, 1.0))
            if angle < angle_threshold_rad:
                labels[neighbor_index] = 0
                queue.append(neighbor_index)

    return labels


def region_grow_pier_with_vertical_axis(
    point_cloud: o3d.geometry.PointCloud,
    seed_indices: np.ndarray,
    angle_threshold_rad: float,
    search_knn: int,
    center_xy: np.ndarray,
    max_radius: float,
) -> np.ndarray:
    """Region-grow one pier using normal similarity plus XY distance to its vertical axis."""
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    labels = np.full(len(points), -1, dtype=int)

    seed_indices = np.asarray(seed_indices, dtype=int).ravel()
    if seed_indices.size == 0:
        return labels

    seed_points = points[seed_indices]
    seed_distances = dist_to_vertical_axis(seed_points, center_xy)
    valid_seed_indices = seed_indices[seed_distances <= max_radius]
    if valid_seed_indices.size == 0:
        return labels

    queue: deque[int] = deque(valid_seed_indices.tolist())
    for seed_index in valid_seed_indices:
        labels[seed_index] = 0

    while queue:
        current_index = queue.popleft()
        current_normal = normals[current_index]
        _, neighbor_indices, _ = tree.search_knn_vector_3d(points[current_index], search_knn)

        for neighbor_index in neighbor_indices:
            if labels[neighbor_index] != -1:
                continue

            neighbor_distance = dist_to_vertical_axis(points[neighbor_index : neighbor_index + 1], center_xy)[0]
            if neighbor_distance > max_radius:
                continue

            neighbor_normal = normals[neighbor_index]
            angle = np.arccos(np.clip(np.dot(current_normal, neighbor_normal), -1.0, 1.0))
            if angle < angle_threshold_rad:
                labels[neighbor_index] = 0
                queue.append(neighbor_index)

    return labels


def assign_noise_points(
    point_cloud: o3d.geometry.PointCloud,
    labels: np.ndarray,
    output_dir: Path,
    file_stem: str,
    config: PierDetectionConfig,
) -> None:
    """Split grown points into pier and other parts, then reassign small noisy groups by nearest neighbor."""
    points = np.asarray(point_cloud.points)
    pier_points = points[labels == 0]
    other_points = points[labels != 0]

    if other_points.size == 0:
        save_point_cloud_txt(output_dir / f"{file_stem}_pier.txt", pier_points)
        save_point_cloud_txt(output_dir / f"{file_stem}_other.txt", np.empty((0, 3), dtype=float))
        return

    dbscan_labels = DBSCAN(
        eps=config.noise_dbscan_eps,
        min_samples=config.noise_dbscan_min_samples,
    ).fit(other_points[:, :3]).labels_

    unique_labels = np.unique(dbscan_labels[dbscan_labels != -1])
    counts = np.array([np.sum(dbscan_labels == label) for label in unique_labels]) if len(unique_labels) else np.array([])
    noise_labels = unique_labels[counts < config.noise_point_threshold] if len(unique_labels) else np.array([])
    noise_mask = np.isin(dbscan_labels, noise_labels)

    noise_points = other_points[noise_mask | (dbscan_labels == -1)]
    core_other_points = other_points[~noise_mask & (dbscan_labels != -1)]

    combined_points = np.vstack((pier_points, core_other_points)) if len(core_other_points) else pier_points.copy()
    combined_labels = np.hstack(
        (
            np.full(len(pier_points), "pier", dtype=object),
            np.full(len(core_other_points), "other", dtype=object),
        )
    ) if len(core_other_points) else np.full(len(pier_points), "pier", dtype=object)

    if len(combined_points) == 0:
        final_pier_points = np.empty((0, 3), dtype=float)
        final_other_points = np.empty((0, 3), dtype=float)
    else:
        tree = cKDTree(combined_points[:, :3])
        assigned_to_pier: List[np.ndarray] = []
        assigned_to_other: List[np.ndarray] = []

        for point in noise_points:
            _, nearest_index = tree.query(point[:3], k=1)
            if combined_labels[nearest_index] == "pier":
                assigned_to_pier.append(point)
            else:
                assigned_to_other.append(point)

        final_pier_points = np.vstack([pier_points] + assigned_to_pier) if assigned_to_pier else pier_points
        final_other_points = (
            np.vstack([core_other_points] + assigned_to_other)
            if assigned_to_other
            else core_other_points
        )

    save_point_cloud_txt(output_dir / f"{file_stem}_pier.txt", final_pier_points)
    save_point_cloud_txt(output_dir / f"{file_stem}_other.txt", final_other_points)


def process_pier_candidate(file_path: Path, output_dir: Path, config: PierDetectionConfig) -> bool:
    """Detect pier points inside one manually grouped sub-patch file."""
    scene_start_time = perf_counter()

    load_time = 0.0
    normal_time = 0.0
    bottom_filter_time = 0.0
    bottom_dbscan_time = 0.0
    seed_extraction_time = 0.0
    region_growing_time = 0.0
    noise_reassignment_time = 0.0

    load_start_time = perf_counter()
    point_cloud = load_point_cloud(file_path)
    load_time += perf_counter() - load_start_time

    normal_start_time = perf_counter()
    point_cloud = compute_normals(
        point_cloud,
        max_nn=config.normal_max_nn,
        orient_normals_consistently=config.orient_normals_consistently,
    )
    normal_time += perf_counter() - normal_start_time

    bottom_filter_start_time = perf_counter()
    filtered_points, filtered_indices = filter_bottom_grid_points(point_cloud, grid_size=config.bottom_grid_size)
    bottom_filter_time += perf_counter() - bottom_filter_start_time
    if filtered_points.size == 0:
        print(f"Skipping {file_path.name}: no valid bottom grid points were found.")
        return False

    save_point_cloud_txt(output_dir / f"{file_path.stem}_filtered_bottom_points.txt", filtered_points)

    bottom_dbscan_start_time = perf_counter()
    bottom_labels = cluster_bottom_points(
        filtered_points,
        eps=config.bottom_cluster_eps,
        min_samples=config.bottom_cluster_min_samples,
    ) if config.bottom_dbscan_voxel_size <= 0 else cluster_bottom_points_with_voxelization(
        filtered_points,
        voxel_size=config.bottom_dbscan_voxel_size,
        eps=config.bottom_cluster_eps,
        min_samples=config.bottom_cluster_min_samples,
    )
    bottom_dbscan_time += perf_counter() - bottom_dbscan_start_time

    seed_extraction_start_time = perf_counter()
    seed_index_groups = extract_seed_index_groups(
        filtered_points,
        filtered_indices,
        bottom_labels,
        z_threshold=config.bottom_seed_height,
    )
    seed_indices = np.hstack(seed_index_groups) if seed_index_groups else np.empty((0,), dtype=int)
    seed_extraction_time += perf_counter() - seed_extraction_start_time
    if seed_indices.size == 0:
        print(f"Skipping {file_path.name}: no region-growing seeds were extracted.")
        return False

    save_point_cloud_txt(output_dir / f"{file_path.stem}_merged_patch.txt", np.asarray(point_cloud.points)[seed_indices])

    region_growing_start_time = perf_counter()
    grown_labels = np.full(len(np.asarray(point_cloud.points)), -1, dtype=int)
    for seed_group in seed_index_groups:
        center_xy, max_radius = estimate_pier_center_and_rmax(
            point_cloud,
            seed_group,
            percentile=config.axis_radius_percentile,
            scale=config.axis_radius_scale,
        )
        if center_xy is None or max_radius is None:
            continue
        grown_one = region_grow_pier_with_vertical_axis(
            point_cloud,
            seed_group,
            angle_threshold_rad=config.region_angle_threshold_rad,
            search_knn=config.region_search_knn,
            center_xy=center_xy,
            max_radius=max_radius,
        )
        grown_labels[grown_one == 0] = 0
    region_growing_time += perf_counter() - region_growing_start_time

    noise_reassignment_start_time = perf_counter()
    assign_noise_points(point_cloud, grown_labels, output_dir, file_path.stem, config)
    noise_reassignment_time += perf_counter() - noise_reassignment_start_time

    total_time = perf_counter() - scene_start_time
    print(
        f"Detected pier candidate: {file_path.name} "
        f"(seeds={len(seed_indices)}, load={load_time:.2f}s, normals={normal_time:.2f}s, "
        f"bottom_filter={bottom_filter_time:.2f}s, bottom_dbscan={bottom_dbscan_time:.2f}s, "
        f"seed_extraction={seed_extraction_time:.2f}s, region_growing={region_growing_time:.2f}s, "
        f"noise_reassignment={noise_reassignment_time:.2f}s, total={total_time:.2f}s)"
    )
    return True


def process_scene(scene_dir: Path, config: PierDetectionConfig) -> bool:
    """Run pier detection on all files under ``sub_patch/pier_piercap`` for one scene."""
    input_dir = scene_dir / config.sub_patch_dir_name / config.input_group_dir_name
    if not input_dir.exists():
        print(f"Skipping {scene_dir.name}: missing {input_dir}.")
        return False

    output_dir = input_dir / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted([path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"])
    if not input_files:
        print(f"Skipping {scene_dir.name}: no pier candidate files found in {input_dir}.")
        return False

    scene_start_time = perf_counter()
    processed_count = 0
    if config.max_workers > 1 and len(input_files) > 1:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_pier_candidate, file_path, output_dir, config): file_path
                for file_path in input_files
            }
            for future in as_completed(futures):
                if future.result():
                    processed_count += 1
    else:
        for file_path in input_files:
            if process_pier_candidate(file_path, output_dir, config):
                processed_count += 1

    total_time = perf_counter() - scene_start_time
    print(
        f"Pier detection scene complete: {scene_dir.name} "
        f"(files={processed_count}, workers={config.max_workers}, total={total_time:.2f}s)"
    )
    return True


def run_pier_detection(input_root: Path, config: Optional[PierDetectionConfig] = None) -> None:
    """Run pier detection for every scene under the input root."""
    detection_config = config or PierDetectionConfig()
    input_root = Path(input_root)

    print(f"Running pier detection on: {input_root}")
    for scene_dir in scene_directories(input_root):
        process_scene(scene_dir, detection_config)
