from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    sampled_points = [points[np.random.randint(len(points))]]
    distances = np.full(len(points), np.inf)

    for _ in range(1, num_samples):
        dist_to_sampled = np.linalg.norm(points - sampled_points[-1], axis=1)
        distances = np.minimum(distances, dist_to_sampled)
        sampled_points.append(points[np.argmax(distances)])

    return np.array(sampled_points)


def voxel_sampling(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.voxel_down_sample(voxel_size=voxel_size).points)


def remove_non_planar_points(
    points: np.ndarray,
    sampled_points: np.ndarray,
    radius: float = 0.1,
    distance_threshold: float = 0.01,
    visualize: bool = False,
    report_every: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    planar_mask = np.ones(len(points), dtype=bool)
    neighbors_search = NearestNeighbors(radius=radius).fit(points)
    total_samples = len(sampled_points)
    loop_start_time = time.perf_counter()
    block_start_time = loop_start_time

    for sample_index, sample_point in enumerate(sampled_points, start=1):
        indices = neighbors_search.radius_neighbors([sample_point], return_distance=False)[0]
        if len(indices) < 3:
            planar_mask[indices] = False
        else:
            local_points = points[indices]
            pca = PCA(n_components=3)
            pca.fit(local_points)
            normal_vector = pca.components_[-1]
            point_on_plane = local_points.mean(axis=0)
            plane_bias = -point_on_plane.dot(normal_vector)
            distances = np.abs((local_points @ normal_vector + plane_bias) / np.linalg.norm(normal_vector))
            inlier_mask = distances < distance_threshold

            if np.sum(inlier_mask) < len(local_points):
                planar_mask[indices[~inlier_mask]] = False

            if visualize:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(*(local_points[inlier_mask].T), color="blue", label="Inliers")
                ax.scatter(*(local_points[~inlier_mask].T), color="red", label="Outliers")
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
                zz = (-normal_vector[0] * xx - normal_vector[1] * yy - plane_bias) / normal_vector[2]
                ax.plot_surface(xx, yy, zz, color="cyan", alpha=0.5)
                ax.scatter(sample_point[0], sample_point[1], sample_point[2], color="black", s=100, label="Seed")
                ax.legend()
                plt.show()

        if report_every > 0 and (sample_index % report_every == 0 or sample_index == total_samples):
            now = time.perf_counter()
            block_elapsed = now - block_start_time
            total_elapsed = now - loop_start_time
            print(
                f"[Planarity] processed {sample_index}/{total_samples} sampled points "
                f"(last {min(report_every, sample_index)} seeds: {block_elapsed:.2f}s, total: {total_elapsed:.2f}s)"
            )
            block_start_time = now

    planar_points = points[planar_mask]
    residual_points = points[~planar_mask]
    return planar_points, residual_points
