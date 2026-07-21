from __future__ import annotations

import time

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    return vh[2, :], centroid


def point_to_plane_distance(point: np.ndarray, normal: np.ndarray, point_on_plane: np.ndarray) -> float:
    return float(abs(np.dot(normal, point - point_on_plane)))


def compute_candidate_normals(
    candidate_points: np.ndarray,
    reference_points: np.ndarray | None = None,
) -> np.ndarray:
    if len(candidate_points) == 0:
        return np.empty((0, 3))

    if reference_points is not None and len(reference_points) > 0:
        combined_points = np.vstack((reference_points[:, :3], candidate_points[:, :3]))
        candidate_start_index = len(reference_points)
    else:
        combined_points = candidate_points[:, :3]
        candidate_start_index = 0

    combined_cloud = o3d.geometry.PointCloud()
    combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
    combined_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=32))
    combined_cloud.orient_normals_consistent_tangent_plane(k=32)
    combined_normals = np.asarray(combined_cloud.normals)
    return combined_normals[candidate_start_index:]


def grow_deck_region(
    deck_points: np.ndarray,
    candidate_points_with_labels: np.ndarray,
    candidate_normals: np.ndarray,
    normal_threshold: float,
    plane_distance_threshold: float = 0.05,
    seed_plane_radius: float = 0.5,
    voxel_size: float = 0.2,
    report_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deck_cloud = o3d.geometry.PointCloud()
    deck_cloud.points = o3d.utility.Vector3dVector(deck_points)
    seed_points = np.asarray(deck_cloud.voxel_down_sample(voxel_size=voxel_size).points)

    candidate_search = NearestNeighbors(n_neighbors=32).fit(candidate_points_with_labels[:, :3])
    deck_search = NearestNeighbors(radius=seed_plane_radius).fit(deck_points)
    visited = np.zeros(len(candidate_points_with_labels), dtype=bool)
    accepted_indices: list[int] = []
    total_seeds = len(seed_points)
    loop_start_time = time.perf_counter()
    block_start_time = loop_start_time
    candidate_neighbor_distances, candidate_neighbor_indices = candidate_search.kneighbors(
        candidate_points_with_labels[:, :3],
        n_neighbors=32,
    )

    for seed_index, seed_point in enumerate(tqdm(seed_points, desc="Region growing"), start=1):
        _, candidate_indices = candidate_search.kneighbors([seed_point], n_neighbors=32)
        available_candidate_indices = [
            int(candidate_index)
            for candidate_index in candidate_indices[0]
            if not visited[candidate_index]
        ]
        if len(available_candidate_indices) == 0:
            continue

        _, neighborhood_indices = deck_search.radius_neighbors([seed_point])
        local_deck_region = deck_points[neighborhood_indices[0], :]
        if len(local_deck_region) < 3:
            pass
        else:
            fitted_normal, fitted_centroid = fit_plane_pca(local_deck_region)

            for candidate_index in available_candidate_indices:
                if visited[candidate_index]:
                    continue

                angle = np.arccos(
                    np.clip(
                        np.abs(np.dot(fitted_normal, candidate_normals[candidate_index])),
                        -1.0,
                        1.0,
                    )
                )
                if angle >= normal_threshold:
                    continue

                plane_distance = point_to_plane_distance(
                    candidate_points_with_labels[candidate_index, :3],
                    fitted_normal,
                    fitted_centroid,
                )
                if plane_distance > plane_distance_threshold:
                    continue

                visited[candidate_index] = True
                seeds = [candidate_index]
                region_indices = [candidate_index]

                while seeds:
                    current_seed = seeds.pop()
                    for neighbor_distance, neighbor_index in zip(
                        candidate_neighbor_distances[current_seed],
                        candidate_neighbor_indices[current_seed],
                    ):
                        if visited[neighbor_index]:
                            continue

                        neighbor_angle = np.arccos(
                            np.clip(
                                np.abs(
                                    np.dot(
                                        candidate_normals[current_seed],
                                        candidate_normals[neighbor_index],
                                    )
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                        if neighbor_angle >= normal_threshold:
                            continue

                        plane_distance = point_to_plane_distance(
                            candidate_points_with_labels[neighbor_index, :3],
                            fitted_normal,
                            fitted_centroid,
                        )
                        if plane_distance > plane_distance_threshold:
                            continue

                        visited[neighbor_index] = True
                        seeds.append(neighbor_index)
                        region_indices.append(neighbor_index)

                accepted_indices.extend(region_indices)

        if report_every > 0 and (seed_index % report_every == 0 or seed_index == total_seeds):
            now = time.perf_counter()
            block_elapsed = now - block_start_time
            total_elapsed = now - loop_start_time
            print(
                f"[RegionGrowing] processed {seed_index}/{total_seeds} seed points "
                f"(last {min(report_every, seed_index)} seeds: {block_elapsed:.2f}s, total: {total_elapsed:.2f}s, "
                f"accepted candidates: {len(accepted_indices)})"
            )
            block_start_time = now

    unique_indices = np.unique(accepted_indices)
    grown_deck_points = np.vstack((deck_points, candidate_points_with_labels[unique_indices, :3]))
    remaining_points = np.delete(candidate_points_with_labels, unique_indices, axis=0)
    grown_candidate_points_with_labels = candidate_points_with_labels[unique_indices]
    return grown_deck_points, remaining_points, grown_candidate_points_with_labels
