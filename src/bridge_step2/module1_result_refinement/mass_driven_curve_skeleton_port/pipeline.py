from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from ..io_utils import save_xyz_points
from .compute_skeleton import compute_skeleton
from .deloop import deloop
from .show_skel import show_skeleton
from .smooth import smooth_skeleton


@dataclass
class MassDrivenSkeletonConfig:
    voxel_size: float = 0.5
    sample_ratio: float = 0.05
    lambda_value: float = 200.0
    smooth_lambda: float = 200.0
    smooth_weight: float = 0.3
    interpolation_count: int = 0
    random_seed: int = 42


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    return np.asarray(cloud.voxel_down_sample(voxel_size=voxel_size).points)


def prepare_sampled_deck_points(points: np.ndarray, voxel_size: float = 0.5) -> np.ndarray:
    sampled_points = voxel_downsample(points[:, :3], voxel_size=voxel_size)
    if len(sampled_points) == 0:
        sampled_points = points[:, :3]
    return sampled_points


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    min_xyz = np.min(points[:, :3], axis=0)
    shifted_points = points[:, :3] - min_xyz
    min_coord_all = float(np.min(shifted_points))
    max_coord_all = float(np.max(shifted_points))
    scale = max(max_coord_all - min_coord_all, 1e-12)
    normalized_points = (shifted_points - min_coord_all) / scale
    return normalized_points, min_xyz, min_coord_all, max_coord_all


def restore_points(points: np.ndarray, min_xyz: np.ndarray, min_coord_all: float, max_coord_all: float) -> np.ndarray:
    scale = max(max_coord_all - min_coord_all, 1e-12)
    return points * scale + min_coord_all + min_xyz


def estimate_deck_principal_frame(deck_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(deck_points) <= 1:
        return np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float), 1.0

    xy = deck_points[:, :2].astype(float)
    centered_xy = xy - np.mean(xy, axis=0, keepdims=True)
    covariance = np.cov(centered_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    principal_axis = eigenvectors[:, order[0]]
    principal_axis = principal_axis / max(np.linalg.norm(principal_axis), 1e-12)
    secondary_axis = eigenvectors[:, order[1]] if eigenvectors.shape[1] > 1 else np.array([-principal_axis[1], principal_axis[0]])
    secondary_axis = secondary_axis / max(np.linalg.norm(secondary_axis), 1e-12)

    long_span = float(np.max(xy @ principal_axis) - np.min(xy @ principal_axis))
    short_span = float(np.max(xy @ secondary_axis) - np.min(xy @ secondary_axis))
    short_to_long_ratio = short_span / max(long_span, 1e-12)
    return principal_axis, secondary_axis, short_to_long_ratio


def sort_points_by_axis(points: np.ndarray, axis_xy: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    projections = points[:, :2] @ axis_xy
    order = np.argsort(projections)
    return points[order]


def generate_pca_centerline_skeleton(
    deck_points: np.ndarray,
    principal_axis: np.ndarray,
) -> np.ndarray:
    if len(deck_points) == 0:
        return np.empty((0, 3))
    if len(deck_points) == 1:
        return deck_points[:, :3].copy()

    xyz = deck_points[:, :3].astype(float)
    line_origin_xy = np.mean(xyz[:, :2], axis=0)
    centered_xy = xyz[:, :2] - line_origin_xy
    projections = centered_xy @ principal_axis
    proj_min = float(np.min(projections))
    proj_max = float(np.max(projections))
    long_span = proj_max - proj_min

    if long_span <= 1e-8:
        return np.mean(xyz, axis=0, keepdims=True)

    # For short, near-rectangular bridges we assume the deck centerline is
    # straight. Build 5 equally spaced samples along the PCA bridge axis and
    # keep the middle 3 so downstream deck cutting is less sensitive to noisy
    # bridge-head / bridge-tail geometry.
    sample_positions = np.linspace(proj_min, proj_max, num=5)
    keep_positions = sample_positions[1:4]
    skeleton_points = []

    for position in keep_positions:
        point_xy = line_origin_xy + position * principal_axis
        local_mask = np.abs(projections - position) <= max(long_span * 0.05, 0.5)
        if np.any(local_mask):
            point_z = float(np.median(xyz[local_mask, 2]))
        else:
            point_z = float(np.median(xyz[:, 2]))
        skeleton_points.append([point_xy[0], point_xy[1], point_z])

    return np.asarray(skeleton_points, dtype=float)


def order_skeleton_points_original(skeleton_points: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    if len(skeleton_points) == 0:
        return skeleton_points
    degree = np.sum(adjacency, axis=1)
    endpoints = np.where(degree == 1)[0]
    start_node = int(endpoints[0]) if len(endpoints) > 0 else 0
    ordered = [skeleton_points[start_node]]
    visited = np.zeros(len(skeleton_points), dtype=bool)
    visited[start_node] = True
    stack = [start_node]

    while stack:
        current = stack[-1]
        next_nodes = np.where((adjacency[current] > 0) & (~visited))[0]
        if len(next_nodes) == 0:
            stack.pop()
        else:
            next_node = int(next_nodes[0])
            ordered.append(skeleton_points[next_node])
            visited[next_node] = True
            stack.append(next_node)

    return np.vstack(ordered)


def order_skeleton_points(
    skeleton_points: np.ndarray,
    adjacency: np.ndarray,
    deck_points: np.ndarray,
    short_to_long_ratio_threshold: float = 0.4,
) -> np.ndarray:
    if len(skeleton_points) <= 1:
        return skeleton_points.copy()

    principal_axis, _, short_to_long_ratio = estimate_deck_principal_frame(deck_points)
    print(short_to_long_ratio)
    if short_to_long_ratio >= short_to_long_ratio_threshold:
        return generate_pca_centerline_skeleton(deck_points, principal_axis)

    return order_skeleton_points_original(skeleton_points, adjacency)


def generate_ordered_skeleton_from_deck_points(
    deck_points: np.ndarray,
    config: MassDrivenSkeletonConfig | None = None,
) -> np.ndarray:
    config = config or MassDrivenSkeletonConfig()
    xyz_points = deck_points[:, :3]
    if len(xyz_points) == 0:
        return np.empty((0, 3))

    sampled_points = prepare_sampled_deck_points(xyz_points, voxel_size=config.voxel_size)
    principal_axis, _, short_to_long_ratio = estimate_deck_principal_frame(sampled_points)
    if short_to_long_ratio >= 0.3:
        return generate_pca_centerline_skeleton(sampled_points, principal_axis)

    downsampled_points = sampled_points.copy()

    normalized_points, min_xyz, min_coord_all, max_coord_all = normalize_points(downsampled_points)
    rng = np.random.default_rng(config.random_seed)
    sample_count = max(1, int(np.floor(len(normalized_points) * config.sample_ratio)))
    sample_indices = rng.integers(0, len(normalized_points), size=sample_count)
    skeleton_points = normalized_points[sample_indices]

    point_mass = np.ones((len(normalized_points), 1), dtype=float) / max(len(normalized_points), 1)
    skeleton_mass = np.ones((len(skeleton_points), 1), dtype=float) / max(len(skeleton_points), 1)

    skeleton_mass, skeleton_points, transport_plan, _, _ = compute_skeleton(
        normalized_points,
        skeleton_points,
        point_mass,
        skeleton_mass,
        config.lambda_value,
    )

    adjacency = show_skeleton(normalized_points, skeleton_points, transport_plan)
    transport_plan, skeleton_points, adjacency, skeleton_mass = deloop(
        sample_indices,
        transport_plan,
        skeleton_points,
        adjacency,
        skeleton_mass,
        normalized_points,
    )

    smoothed_skeleton, smoothed_adjacency, _ = smooth_skeleton(
        normalized_points,
        adjacency,
        skeleton_points,
        skeleton_mass,
        config.smooth_lambda,
        config.smooth_weight,
        config.interpolation_count,
    )

    restored_skeleton = restore_points(smoothed_skeleton, min_xyz, min_coord_all, max_coord_all)
    return order_skeleton_points_original(restored_skeleton, smoothed_adjacency)


def generate_ordered_skeleton_file(
    deck_file: str | Path,
    output_file: str | Path,
    config: MassDrivenSkeletonConfig | None = None,
) -> np.ndarray:
    deck_points = np.loadtxt(deck_file, delimiter=" ")
    ordered_skeleton = generate_ordered_skeleton_from_deck_points(deck_points, config=config)
    save_xyz_points(ordered_skeleton, output_file)
    return ordered_skeleton


def generate_and_save_ordered_skeleton(
    deck_points: np.ndarray,
    output_file: str | Path,
    config: MassDrivenSkeletonConfig | None = None,
) -> np.ndarray:
    ordered_skeleton = generate_ordered_skeleton_from_deck_points(deck_points, config=config)
    save_xyz_points(ordered_skeleton, output_file)
    return ordered_skeleton
