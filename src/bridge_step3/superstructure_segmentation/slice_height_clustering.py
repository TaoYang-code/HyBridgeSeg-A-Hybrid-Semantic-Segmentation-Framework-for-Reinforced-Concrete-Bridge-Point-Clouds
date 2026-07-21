from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def fit_plane_ransac(
    cluster_points: np.ndarray,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Optional[np.ndarray]:
    """Fit a plane with RANSAC and return ``(a, b, c, d)`` plane parameters."""
    if len(cluster_points) < 3:
        return None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cluster_points[:, :3])
    plane_model, _ = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return np.asarray(plane_model, dtype=float)


def point_to_plane_distance(point: np.ndarray, plane_model: Optional[np.ndarray]) -> float:
    """Return the orthogonal distance from a point to a plane."""
    if plane_model is None:
        return 0.0

    a, b, c, d = plane_model
    denominator = np.sqrt(a**2 + b**2 + c**2)
    if denominator < 1e-12:
        return 0.0
    x, y, z = point[:3]
    return abs(a * x + b * y + c * z + d) / denominator


def height_region_grow(
    point_cloud: np.ndarray,
    z_threshold: float = 0.01,
    neighbor_distance: float = 0.5,#0.2
    max_plane_distance: float = 0.05,
    fit_interval: int = 50,
    min_cluster_size: int = 1,
) -> List[np.ndarray]:
    """Cluster deck points using Z consistency with periodic local plane fitting."""
    if point_cloud.size == 0:
        return []

    num_points = point_cloud.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    tree = cKDTree(point_cloud[:, :3])
    clusters: List[np.ndarray] = []
    fallback_clusters: List[np.ndarray] = []

    def region_grow(seed_index: int) -> Optional[np.ndarray]:
        cluster_indices: List[int] = []
        queue: deque[int] = deque([seed_index])
        plane_model: Optional[np.ndarray] = None

        while queue:
            index = queue.popleft()
            if visited[index]:
                continue

            visited[index] = True
            cluster_indices.append(index)

            if len(cluster_indices) % fit_interval == 0:
                plane_model = fit_plane_ransac(point_cloud[cluster_indices])

            neighbors = tree.query_ball_point(point_cloud[index, :3], neighbor_distance)
            for neighbor_index in neighbors:
                if visited[neighbor_index]:
                    continue

                z_difference = abs(point_cloud[neighbor_index, 2] - point_cloud[index, 2])
                plane_distance = point_to_plane_distance(point_cloud[neighbor_index], plane_model)
                if z_difference <= z_threshold and plane_distance <= max_plane_distance:
                    queue.append(neighbor_index)

        if len(cluster_indices) < min_cluster_size:
            return None
        return point_cloud[cluster_indices]

    for index in range(num_points):
        if visited[index]:
            continue
        cluster = region_grow(index)
        if cluster is not None:
            clusters.append(cluster)
        else:
            # Keep a raw fallback cluster so completely small-but-valid slices do not become empty.
            fallback_indices: List[int] = []
            queue: deque[int] = deque([index])
            local_visited: set[int] = set()
            plane_model: Optional[np.ndarray] = None

            while queue:
                current_index = queue.popleft()
                if current_index in local_visited:
                    continue
                local_visited.add(current_index)
                fallback_indices.append(current_index)

                if len(fallback_indices) % fit_interval == 0:
                    plane_model = fit_plane_ransac(point_cloud[fallback_indices])

                neighbors = tree.query_ball_point(point_cloud[current_index, :3], neighbor_distance)
                for neighbor_index in neighbors:
                    if neighbor_index in local_visited or visited[neighbor_index]:
                        continue
                    z_difference = abs(point_cloud[neighbor_index, 2] - point_cloud[current_index, 2])
                    plane_distance = point_to_plane_distance(point_cloud[neighbor_index], plane_model)
                    if z_difference <= z_threshold and plane_distance <= max_plane_distance:
                        queue.append(neighbor_index)

            if fallback_indices:
                fallback_clusters.append(point_cloud[fallback_indices])

    if clusters:
        return clusters

    if fallback_clusters:
        fallback_clusters.sort(key=len, reverse=True)
        return fallback_clusters[:2]

    return []


def coarse_height_region_grow(
    point_cloud: np.ndarray,
    z_threshold: float = 0.01,
    neighbor_distance: float = 0.2,
) -> List[np.ndarray]:
    """Cluster points using only local Z consistency and neighborhood distance."""
    if point_cloud.size == 0:
        return []

    num_points = point_cloud.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    tree = cKDTree(point_cloud[:, :3])
    clusters: List[np.ndarray] = []

    def region_grow(seed_index: int) -> np.ndarray:
        cluster_indices: List[int] = []
        queue: deque[int] = deque([seed_index])

        while queue:
            index = queue.popleft()
            if visited[index]:
                continue

            visited[index] = True
            cluster_indices.append(index)

            neighbors = tree.query_ball_point(point_cloud[index, :3], neighbor_distance)
            for neighbor_index in neighbors:
                if visited[neighbor_index]:
                    continue
                if abs(point_cloud[neighbor_index, 2] - point_cloud[index, 2]) <= z_threshold:
                    queue.append(neighbor_index)

        return point_cloud[cluster_indices]

    for index in range(num_points):
        if visited[index]:
            continue
        clusters.append(region_grow(index))

    return clusters
