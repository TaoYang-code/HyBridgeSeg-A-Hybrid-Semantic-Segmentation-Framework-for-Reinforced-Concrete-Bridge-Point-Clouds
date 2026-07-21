from __future__ import annotations

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def extract_planar_regions(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    normal_threshold: float,
    distance_threshold: float,
    min_cluster_size: int,
) -> list[list[int]]:
    neighbors = NearestNeighbors(n_neighbors=16).fit(point_cloud)
    visited = np.zeros(len(point_cloud), dtype=bool)
    clusters: list[list[int]] = []

    for start_index in range(len(point_cloud)):
        if visited[start_index]:
            continue

        cluster: list[int] = []
        seeds = [start_index]

        while seeds:
            seed_index = seeds.pop()
            if visited[seed_index]:
                continue

            visited[seed_index] = True
            cluster.append(seed_index)
            distances, neighbor_indices = neighbors.kneighbors([point_cloud[seed_index]], n_neighbors=16)

            for neighbor_distance, neighbor_index in zip(distances[0], neighbor_indices[0]):
                if visited[neighbor_index]:
                    continue

                angle = np.arccos(np.clip(np.dot(normals[seed_index], normals[neighbor_index]), -1.0, 1.0))
                if angle < normal_threshold and neighbor_distance < distance_threshold:
                    seeds.append(neighbor_index)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def estimate_normals(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    o3d_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    o3d_point_cloud.orient_normals_consistent_tangent_plane(k=16)
    return np.asarray(o3d_point_cloud.points), np.asarray(o3d_point_cloud.normals)
