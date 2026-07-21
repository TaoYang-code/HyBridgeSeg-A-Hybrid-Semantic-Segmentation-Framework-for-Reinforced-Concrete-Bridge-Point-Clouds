from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .io_utils import load_point_cloud_txt, save_point_cloud_txt, scene_directories


@dataclass(frozen=True)
class SliceSegmentationConfig:
    deck_file_name: str = "deck_for_cutting.txt"
    skeleton_file_name: str = "ordered_skeleton.txt"
    superstructure_file_name: str = "sup.txt"
    deck_output_dir_name: str = "deck"
    superstructure_output_dir_name: str = "sup"
    deck_slice_voxel_size: float = 0.1
    merge_ratio_threshold: float = 0.5
    save_intermediate_slices: bool = True
    num_workers: int = 1
    log_progress: bool = True


@dataclass(frozen=True)
class SceneSliceSegment:
    deck_points: np.ndarray
    deck_half_1: np.ndarray
    deck_half_2: np.ndarray
    superstructure_points: np.ndarray
    superstructure_half_1: np.ndarray
    superstructure_half_2: np.ndarray


@dataclass(frozen=True)
class SceneSliceData:
    scene_dir: Path
    skeleton_points: np.ndarray
    segments: List[SceneSliceSegment]


def _build_scene_slice_data(
    scene_dir: Path,
    deck_points: np.ndarray,
    skeleton_points: np.ndarray,
    superstructure_points: np.ndarray,
    config: SliceSegmentationConfig,
) -> Optional[SceneSliceData]:
    if len(skeleton_points) < 1:
        if config.log_progress:
            print(f"Skipping {scene_dir.name}: ordered skeleton has fewer than 1 points.")
        return None

    deck_output_dir = scene_dir / config.deck_output_dir_name
    sup_output_dir = scene_dir / config.superstructure_output_dir_name
    if config.save_intermediate_slices:
        deck_output_dir.mkdir(parents=True, exist_ok=True)
        sup_output_dir.mkdir(parents=True, exist_ok=True)

    remaining_deck_points = deck_points
    remaining_superstructure_points = superstructure_points
    deck_segments: List[np.ndarray] = []
    sup_segments: List[np.ndarray] = []
    skeleton_segments: List[np.ndarray] = []

    for segment_index in range(len(skeleton_points) - 1):
        point_a = skeleton_points[segment_index]
        point_b = skeleton_points[segment_index + 1]
        remaining_deck_points, deck_back = split_points_by_skeleton_plane(remaining_deck_points, point_a, point_b)
        remaining_superstructure_points, sup_back = split_points_by_skeleton_plane(
            remaining_superstructure_points,
            point_a,
            point_b,
        )
        deck_segments.append(deck_back)
        sup_segments.append(sup_back)
        skeleton_segments.append(np.array([point_a, point_b], dtype=float))

    if len(remaining_deck_points) > 0:
        deck_segments.append(remaining_deck_points)
        sup_segments.append(remaining_superstructure_points)
        skeleton_segments.append(np.array([skeleton_points[-2], skeleton_points[-1]], dtype=float))

    deck_segments, sup_segments, skeleton_segments = merge_small_adjacent_segments(
        deck_segments,
        sup_segments,
        skeleton_segments,
        merge_ratio_threshold=config.merge_ratio_threshold,
        merge_voxel_size=config.deck_slice_voxel_size,
    )

    scene_segments: List[SceneSliceSegment] = []
    for segment_index, (deck_segment, sup_segment, skeleton_segment) in enumerate(
        zip(deck_segments, sup_segments, skeleton_segments),
        start=1,
    ):
        downsampled_deck = voxel_downsample(deck_segment, voxel_size=config.deck_slice_voxel_size)

        reference_segment = skeleton_segment
        if segment_index == len(deck_segments) and len(skeleton_segments) > 1:
            reference_segment = skeleton_segments[-2]

        deck_half_1, deck_half_2 = split_deck_cross_section(
            downsampled_deck,
            reference_segment[0],
            reference_segment[1],
        )
        superstructure_half_1, superstructure_half_2 = split_deck_cross_section(
            sup_segment,
            reference_segment[0],
            reference_segment[1],
        )

        scene_segments.append(
            SceneSliceSegment(
                deck_points=downsampled_deck,
                deck_half_1=deck_half_1,
                deck_half_2=deck_half_2,
                superstructure_points=sup_segment,
                superstructure_half_1=superstructure_half_1,
                superstructure_half_2=superstructure_half_2,
            )
        )

        if config.save_intermediate_slices:
            save_point_cloud_txt(deck_output_dir / f"deck_{segment_index}.txt", downsampled_deck)
            save_point_cloud_txt(deck_output_dir / f"deck_{segment_index}_1.txt", deck_half_1)
            save_point_cloud_txt(deck_output_dir / f"deck_{segment_index}_2.txt", deck_half_2)
            save_point_cloud_txt(sup_output_dir / f"sup_{segment_index}.txt", sup_segment)
            save_point_cloud_txt(sup_output_dir / f"sup_{segment_index}_1.txt", superstructure_half_1)
            save_point_cloud_txt(sup_output_dir / f"sup_{segment_index}_2.txt", superstructure_half_2)

    return SceneSliceData(
        scene_dir=scene_dir,
        skeleton_points=skeleton_points,
        segments=scene_segments,
    )


def build_scene_slice_data_from_points(
    scene_name: str,
    deck_points: np.ndarray,
    skeleton_points: np.ndarray,
    superstructure_points: np.ndarray,
    config: SliceSegmentationConfig,
) -> Optional[SceneSliceData]:
    return _build_scene_slice_data(
        scene_dir=Path(scene_name),
        deck_points=deck_points,
        skeleton_points=skeleton_points,
        superstructure_points=superstructure_points,
        config=config,
    )


def split_points_by_skeleton_plane(points: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split points in XY using the perpendicular plane through the skeleton segment midpoint."""
    point_a_xy = point_a[:2]
    point_b_xy = point_b[:2]
    direction = point_b_xy - point_a_xy
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-12:
        return points.copy(), np.empty((0, 3), dtype=float)

    direction_unit = direction / direction_norm
    midpoint = (point_a_xy + point_b_xy) / 2.0
    projected_distances = (points[:, :2] - midpoint) @ direction_unit
    return points[projected_distances >= 0], points[projected_distances <= 0]


def split_deck_cross_section(deck_segment: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a deck slice into two halves using the normal of its skeleton direction."""
    if deck_segment.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    direction = point_b[:2] - point_a[:2]
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-12:
        return deck_segment.copy(), np.empty((0, 3), dtype=float)

    direction_unit = direction / direction_norm
    normal_unit = np.array([-direction_unit[1], direction_unit[0]], dtype=float)
    midpoint = (point_a[:2] + point_b[:2]) / 2.0
    signed_distances = (deck_segment[:, :2] - midpoint) @ normal_unit
    signed_distances[np.abs(signed_distances) < 1e-6] = 0.0

    left_half = deck_segment[signed_distances > 0]
    right_half = deck_segment[signed_distances <= 0]
    return left_half, right_half


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points with Open3D voxel sampling."""
    if points.size == 0:
        return np.empty((0, 3), dtype=float)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    downsampled = cloud.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points, dtype=float)


def merge_small_adjacent_segments(
    deck_segments: List[np.ndarray],
    sup_segments: List[np.ndarray],
    skeleton_segments: List[np.ndarray],
    merge_ratio_threshold: float,
    merge_voxel_size: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Merge neighboring slices when one of them is much smaller than the average voxelized slice size."""
    if not deck_segments:
        return deck_segments, sup_segments, skeleton_segments

    def representative_size(points: np.ndarray) -> int:
        return len(voxel_downsample(points, voxel_size=merge_voxel_size))

    mean_point_count = float(np.mean([representative_size(segment) for segment in deck_segments]))
    merged_once = True

    while merged_once:
        merged_once = False
        merged_deck: List[np.ndarray] = []
        merged_sup: List[np.ndarray] = []
        merged_skeleton: List[np.ndarray] = []
        index = 0

        while index < len(deck_segments) - 1:
            current_deck = deck_segments[index]
            next_deck = deck_segments[index + 1]
            current_size = representative_size(current_deck)
            next_size = representative_size(next_deck)

            should_merge = (
                current_size <= merge_ratio_threshold * mean_point_count
                or next_size <= merge_ratio_threshold * mean_point_count
            )

            if should_merge:
                merged_deck.append(np.vstack((current_deck, next_deck)))
                merged_sup.append(np.vstack((sup_segments[index], sup_segments[index + 1])))
                merged_skeleton.append(np.vstack((skeleton_segments[index], skeleton_segments[index + 1])))
                merged_once = True
                index += 2
                continue

            merged_deck.append(current_deck)
            merged_sup.append(sup_segments[index])
            merged_skeleton.append(skeleton_segments[index])
            index += 1

        if index == len(deck_segments) - 1:
            merged_deck.append(deck_segments[-1])
            merged_sup.append(sup_segments[-1])
            merged_skeleton.append(skeleton_segments[-1])

        deck_segments, sup_segments, skeleton_segments = merged_deck, merged_sup, merged_skeleton

    return deck_segments, sup_segments, skeleton_segments


def slice_scene(scene_dir: Path, config: SliceSegmentationConfig) -> Optional[SceneSliceData]:
    """Slice one scene into deck-aligned segments for later superstructure segmentation."""
    scene_start_time = perf_counter()
    load_time = 0.0
    slicing_time = 0.0
    merge_time = 0.0
    voxel_time = 0.0
    save_time = 0.0
    deck_file = scene_dir / config.deck_file_name
    skeleton_file = scene_dir / config.skeleton_file_name
    sup_file = scene_dir / config.superstructure_file_name

    if not all(path.exists() for path in (deck_file, skeleton_file, sup_file)):
        if config.log_progress:
            print(f"Skipping {scene_dir.name}: missing required inputs for slicing.")
        return None

    load_start_time = perf_counter()
    deck_points = load_point_cloud_txt(deck_file)
    skeleton_points = load_point_cloud_txt(skeleton_file)
    superstructure_points = load_point_cloud_txt(sup_file)
    load_time += perf_counter() - load_start_time

    slicing_start_time = perf_counter()
    scene_slice_data = _build_scene_slice_data(
        scene_dir=scene_dir,
        deck_points=deck_points,
        skeleton_points=skeleton_points,
        superstructure_points=superstructure_points,
        config=config,
    )
    slicing_time += perf_counter() - slicing_start_time
    merge_time = 0.0
    voxel_time = 0.0
    if config.save_intermediate_slices and scene_slice_data is not None:
        save_time = 0.0

    elapsed_time = perf_counter() - scene_start_time
    if scene_slice_data is None:
        return None
    scene_segments = scene_slice_data.segments
    if config.log_progress:
        print(
            f"Sliced scene: {scene_dir.name} "
            f"(segments={len(scene_segments)}, deck_points={len(deck_points)}, sup_points={len(superstructure_points)}, "
            f"load={load_time:.2f}s, split={slicing_time:.2f}s, merge={merge_time:.2f}s, "
            f"voxel={voxel_time:.2f}s, save={save_time:.2f}s, total={elapsed_time:.2f}s)"
        )
    return scene_slice_data


def _slice_scene_worker(scene_dir: Path, config: SliceSegmentationConfig) -> tuple[str, Optional[SceneSliceData]]:
    """Picklable worker for scene-level parallel slicing."""
    scene_slice_data = slice_scene(scene_dir, config)
    return scene_dir.name, scene_slice_data


def slice_superstructure_inputs(input_root: Path, config: SliceSegmentationConfig) -> Dict[str, SceneSliceData]:
    """Slice every scene under the input root and return in-memory slice data."""
    scene_slice_map: Dict[str, SceneSliceData] = OrderedDict()
    scene_dirs = scene_directories(input_root)
    worker_count = max(1, int(config.num_workers))

    if worker_count == 1 or len(scene_dirs) <= 1:
        for scene_dir in scene_dirs:
            scene_slice_data = slice_scene(scene_dir, config)
            if scene_slice_data is not None:
                scene_slice_map[scene_dir.name] = scene_slice_data
        return scene_slice_map

    if config.log_progress:
        print(f"Slicing scenes in parallel with {worker_count} workers")
    completed_scene_data: Dict[str, SceneSliceData] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_scene = {
            executor.submit(_slice_scene_worker, scene_dir, config): scene_dir.name
            for scene_dir in scene_dirs
        }
        for future in as_completed(future_to_scene):
            scene_name, scene_slice_data = future.result()
            if scene_slice_data is not None:
                completed_scene_data[scene_name] = scene_slice_data

    for scene_dir in scene_dirs:
        scene_slice_data = completed_scene_data.get(scene_dir.name)
        if scene_slice_data is not None:
            scene_slice_map[scene_dir.name] = scene_slice_data
    return scene_slice_map
