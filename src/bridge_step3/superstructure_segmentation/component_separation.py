from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .deck_sup_slicing import SceneSliceData, SceneSliceSegment
from .io_utils import load_point_cloud_txt, save_point_cloud_txt, scene_directories
from .slice_height_clustering import coarse_height_region_grow, height_region_grow


@dataclass(frozen=True)
class SuperstructureSegmentationConfig:
    deck_dir_name: str = "deck"
    superstructure_dir_name: str = "sup"
    sidewalk_cluster_point_threshold: int = 200
    no_sidewalk_plane_offset: float = 0
    sidewalk_plane_offset: float = 0
    plane_distance_threshold: float = 0.01
    plane_ransac_n: int = 3
    plane_num_iterations: int = 1000
    above_output_name: str = "sup_above.txt"
    below_output_name: str = "sup_below.txt"
    cutting_reference_dir_name: str = "cutting_reference_deck"
    segment_workers: int = 5
    log_progress: bool = True


@dataclass(frozen=True)
class SliceSegmentationResult:
    slice_index: int
    above_points: np.ndarray
    below_points: np.ndarray
    reference_deck_1: np.ndarray
    reference_deck_2: np.ndarray
    debug_side_1: Optional[Dict[str, List[np.ndarray]]]
    debug_side_2: Optional[Dict[str, List[np.ndarray]]]
    step_timing: Dict[str, float]
    elapsed_time: float
    superstructure_point_count: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True)
class SceneSuperstructureResult:
    above_points: np.ndarray
    below_points: np.ndarray
    has_sidewalk: bool
    processed_slice_count: int


def sort_segment_files(files: List[Path], pattern: str) -> List[Path]:
    """Sort files by the integer embedded in their file name."""
    regex = re.compile(pattern)

    def sort_key(file_path: Path) -> int:
        match = regex.fullmatch(file_path.name)
        if match is None:
            return 10**9
        return int(match.group(1))

    return sorted(files, key=sort_key)


def fit_deck_plane(
    deck_points: np.ndarray,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> np.ndarray:
    """Fit a plane to a deck slice using Open3D RANSAC."""
    if len(deck_points) < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(deck_points[:, :3])
    plane_model, _ = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return np.asarray(plane_model, dtype=float)


def split_points_by_plane(points: np.ndarray, plane_model: np.ndarray, plane_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split points into the plane's positive and negative sides."""
    a, b, c, d = plane_model
    signed_distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    above_mask = signed_distances > plane_offset
    below_mask = ~above_mask
    return points[above_mask], points[below_mask]


def detect_sidewalk_presence(deck_reference_points: np.ndarray, config: SuperstructureSegmentationConfig) -> bool:
    """Detect whether the reference deck slice contains two sidewalk bands."""
    clusters = coarse_height_region_grow(deck_reference_points)
    # clusters = height_region_grow(deck_reference_points)
    clusters.sort(key=len, reverse=True)
    large_cluster_count = sum(len(cluster) > config.sidewalk_cluster_point_threshold for cluster in clusters)
    return large_cluster_count >= 2


def get_middle_slice_zero_based(slice_count: int) -> int:
    """Return the zero-based index of the middle slice, biased slightly forward for even counts."""
    if slice_count <= 0:
        raise ValueError("slice_count must be positive")
    return slice_count // 2


def select_sidewalk_reference_cluster(
    deck_side_points: np.ndarray,
    sup_side_points: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
    """Pick the sidewalk-supporting deck cluster from the three largest deck clusters on one side."""
    all_deck_clusters = height_region_grow(deck_side_points, min_cluster_size=1)
    all_deck_clusters.sort(key=len, reverse=True)
    if not all_deck_clusters:
        raise ValueError("No deck clusters were found on the current side.")

    filtered_clusters = [cluster for cluster in all_deck_clusters if len(cluster) >= 300]
    candidate_clusters = filtered_clusters if filtered_clusters else all_deck_clusters[:1]
    top_clusters = candidate_clusters[:3]
    comparison_clusters = top_clusters

    if sup_side_points.size == 0:
        selected_cluster = comparison_clusters[0]
        debug_info = {
            "top_clusters": [cluster[:, :3].copy() for cluster in top_clusters],
            "comparison_clusters": [cluster[:, :3].copy() for cluster in comparison_clusters],
            "selected_cluster": [selected_cluster[:, :3].copy()],
        }
        return selected_cluster, debug_info

    best_cluster = comparison_clusters[0]
    best_distance = float("inf")
    sup_tree = cKDTree(sup_side_points[:, :3])
    for cluster in comparison_clusters:
        distances, _ = sup_tree.query(cluster[:, :3], k=1)
        percentile_distance = float(np.percentile(distances, 20))
        if percentile_distance < best_distance:
            best_distance = percentile_distance
            best_cluster = cluster
    debug_info = {
        "top_clusters": [cluster[:, :3].copy() for cluster in top_clusters],
        "comparison_clusters": [cluster[:, :3].copy() for cluster in comparison_clusters],
        "selected_cluster": [best_cluster[:, :3].copy()],
    }
    return best_cluster, debug_info


def segment_superstructure_slice_without_sidewalk(
    deck_half_1: np.ndarray,
    deck_half_2: np.ndarray,
    superstructure_half_1: np.ndarray,
    superstructure_half_2: np.ndarray,
    config: SuperstructureSegmentationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, List[np.ndarray]]], Optional[Dict[str, List[np.ndarray]]], Dict[str, float]]:
    """Split one slice into superstructure-above and remaining points without sidewalk bands."""
    timing: Dict[str, float] = {
        "plane_fit_1": 0.0,
        "plane_fit_2": 0.0,
    }
    debug_side_1 = None
    debug_side_2 = None

    above_parts: List[np.ndarray] = []
    below_parts: List[np.ndarray] = []

    if len(deck_half_1) >= 3 and len(superstructure_half_1) > 0:
        plane_fit_start_time = perf_counter()
        plane_model_1 = fit_deck_plane(
            deck_half_1,
            distance_threshold=config.plane_distance_threshold,
            ransac_n=config.plane_ransac_n,
            num_iterations=config.plane_num_iterations,
        )
        timing["plane_fit_1"] = perf_counter() - plane_fit_start_time
        above_part_1, below_part_1 = split_points_by_plane(
            superstructure_half_1,
            plane_model_1,
            plane_offset=config.no_sidewalk_plane_offset,
        )
        if above_part_1.size > 0:
            above_parts.append(above_part_1)
        if below_part_1.size > 0:
            below_parts.append(below_part_1)
    elif len(superstructure_half_1) > 0:
        below_parts.append(superstructure_half_1)

    if len(deck_half_2) >= 3 and len(superstructure_half_2) > 0:
        plane_fit_start_time = perf_counter()
        plane_model_2 = fit_deck_plane(
            deck_half_2,
            distance_threshold=config.plane_distance_threshold,
            ransac_n=config.plane_ransac_n,
            num_iterations=config.plane_num_iterations,
        )
        timing["plane_fit_2"] = perf_counter() - plane_fit_start_time
        above_part_2, below_part_2 = split_points_by_plane(
            superstructure_half_2,
            plane_model_2,
            plane_offset=config.no_sidewalk_plane_offset,
        )
        if above_part_2.size > 0:
            above_parts.append(above_part_2)
        if below_part_2.size > 0:
            below_parts.append(below_part_2)
    elif len(superstructure_half_2) > 0:
        below_parts.append(superstructure_half_2)

    if above_parts:
        return (
            np.vstack(above_parts),
            np.vstack(below_parts) if below_parts else np.empty((0, 3), dtype=float),
            (deck_half_1[:, :3].copy() if deck_half_1.size > 0 else np.empty((0, 3), dtype=float)),
            (deck_half_2[:, :3].copy() if deck_half_2.size > 0 else np.empty((0, 3), dtype=float)),
            debug_side_1,
            debug_side_2,
            timing,
        )
    return (
        np.empty((0, 3), dtype=float),
        np.vstack(below_parts) if below_parts else np.empty((0, 3), dtype=float),
        (deck_half_1[:, :3].copy() if deck_half_1.size > 0 else np.empty((0, 3), dtype=float)),
        (deck_half_2[:, :3].copy() if deck_half_2.size > 0 else np.empty((0, 3), dtype=float)),
        debug_side_1,
        debug_side_2,
        timing,
    )


def segment_superstructure_slice_with_sidewalk(
    deck_half_1: np.ndarray,
    deck_half_2: np.ndarray,
    superstructure_half_1: np.ndarray,
    superstructure_half_2: np.ndarray,
    config: SuperstructureSegmentationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, List[np.ndarray]]], Optional[Dict[str, List[np.ndarray]]], Dict[str, float]]:
    """Split one slice into superstructure-above and remaining points using per-side sidewalk deck clusters."""
    timing: Dict[str, float] = {
        "height_region_grow_1": 0.0,
        "plane_fit_1": 0.0,
        "height_region_grow_2": 0.0,
        "plane_fit_2": 0.0,
    }

    above_parts: List[np.ndarray] = []
    below_parts: List[np.ndarray] = []
    deck_part_1 = np.empty((0, 3), dtype=float)
    deck_part_2 = np.empty((0, 3), dtype=float)
    debug_side_1 = None
    debug_side_2 = None

    if len(deck_half_1) >= 3 and len(superstructure_half_1) > 0:
        region_grow_start_time = perf_counter()
        deck_part_1, debug_side_1 = select_sidewalk_reference_cluster(deck_half_1, superstructure_half_1)
        timing["height_region_grow_1"] = perf_counter() - region_grow_start_time
        plane_fit_start_time = perf_counter()
        plane_model_1 = fit_deck_plane(
            deck_part_1,
            distance_threshold=config.plane_distance_threshold,
            ransac_n=config.plane_ransac_n,
            num_iterations=config.plane_num_iterations,
        )
        timing["plane_fit_1"] = perf_counter() - plane_fit_start_time
        above_part_1, below_part_1 = split_points_by_plane(
            superstructure_half_1,
            plane_model_1,
            plane_offset=config.sidewalk_plane_offset,
        )
        if above_part_1.size > 0:
            above_parts.append(above_part_1)
        if below_part_1.size > 0:
            below_parts.append(below_part_1)
    elif len(superstructure_half_1) > 0:
        below_parts.append(superstructure_half_1)

    if len(deck_half_2) >= 3 and len(superstructure_half_2) > 0:
        region_grow_start_time = perf_counter()
        deck_part_2, debug_side_2 = select_sidewalk_reference_cluster(deck_half_2, superstructure_half_2)
        timing["height_region_grow_2"] = perf_counter() - region_grow_start_time
        plane_fit_start_time = perf_counter()
        plane_model_2 = fit_deck_plane(
            deck_part_2,
            distance_threshold=config.plane_distance_threshold,
            ransac_n=config.plane_ransac_n,
            num_iterations=config.plane_num_iterations,
        )
        timing["plane_fit_2"] = perf_counter() - plane_fit_start_time
        above_part_2, below_part_2 = split_points_by_plane(
            superstructure_half_2,
            plane_model_2,
            plane_offset=config.sidewalk_plane_offset,
        )
        if above_part_2.size > 0:
            above_parts.append(above_part_2)
        if below_part_2.size > 0:
            below_parts.append(below_part_2)
    elif len(superstructure_half_2) > 0:
        below_parts.append(superstructure_half_2)

    if above_parts:
        return (
            np.vstack(above_parts),
            np.vstack(below_parts) if below_parts else np.empty((0, 3), dtype=float),
            deck_part_1[:, :3].copy() if deck_part_1.size > 0 else np.empty((0, 3), dtype=float),
            deck_part_2[:, :3].copy() if deck_part_2.size > 0 else np.empty((0, 3), dtype=float),
            debug_side_1,
            debug_side_2,
            timing,
        )
    return (
        np.empty((0, 3), dtype=float),
        np.vstack(below_parts) if below_parts else np.empty((0, 3), dtype=float),
        deck_part_1[:, :3].copy() if deck_part_1.size > 0 else np.empty((0, 3), dtype=float),
        deck_part_2[:, :3].copy() if deck_part_2.size > 0 else np.empty((0, 3), dtype=float),
        debug_side_1,
        debug_side_2,
        timing,
    )


def _segment_scene_slice_worker(
    slice_index: int,
    scene_segment: SceneSliceSegment,
    has_sidewalk: bool,
    config: SuperstructureSegmentationConfig,
) -> SliceSegmentationResult:
    """Process one sliced deck/sup block independently."""
    slice_start_time = perf_counter()
    superstructure_points = scene_segment.superstructure_points
    if superstructure_points.size == 0:
        return SliceSegmentationResult(
            slice_index=slice_index,
            above_points=np.empty((0, 3), dtype=float),
            below_points=np.empty((0, 3), dtype=float),
            reference_deck_1=np.empty((0, 3), dtype=float),
            reference_deck_2=np.empty((0, 3), dtype=float),
            debug_side_1=None,
            debug_side_2=None,
            step_timing={},
            elapsed_time=perf_counter() - slice_start_time,
            superstructure_point_count=0,
            skipped=True,
            skip_reason="empty superstructure slice",
        )

    if has_sidewalk:
        try:
            above_points, below_points, reference_deck_1, reference_deck_2, debug_side_1, debug_side_2, step_timing = segment_superstructure_slice_with_sidewalk(
                scene_segment.deck_half_1,
                scene_segment.deck_half_2,
                scene_segment.superstructure_half_1,
                scene_segment.superstructure_half_2,
                config,
            )
        except ValueError:
            return SliceSegmentationResult(
                slice_index=slice_index,
                above_points=np.empty((0, 3), dtype=float),
                below_points=np.empty((0, 3), dtype=float),
                reference_deck_1=np.empty((0, 3), dtype=float),
                reference_deck_2=np.empty((0, 3), dtype=float),
                debug_side_1=None,
                debug_side_2=None,
                step_timing={},
                elapsed_time=perf_counter() - slice_start_time,
                superstructure_point_count=len(superstructure_points),
                skipped=True,
                skip_reason="insufficient sidewalk deck clusters for superstructure separation",
            )
    else:
        above_points, below_points, reference_deck_1, reference_deck_2, debug_side_1, debug_side_2, step_timing = segment_superstructure_slice_without_sidewalk(
            scene_segment.deck_half_1,
            scene_segment.deck_half_2,
            scene_segment.superstructure_half_1,
            scene_segment.superstructure_half_2,
            config,
        )

    return SliceSegmentationResult(
        slice_index=slice_index,
        above_points=above_points,
        below_points=below_points,
        reference_deck_1=reference_deck_1,
        reference_deck_2=reference_deck_2,
        debug_side_1=debug_side_1,
        debug_side_2=debug_side_2,
        step_timing=step_timing,
        elapsed_time=perf_counter() - slice_start_time,
        superstructure_point_count=len(superstructure_points),
    )


def segment_scene(scene_dir: Path, config: SuperstructureSegmentationConfig) -> bool:
    """Run superstructure-above / superstructure-below segmentation for one scene."""
    return segment_scene_from_slice_data(scene_dir, None, config) is not None


def segment_scene_from_slice_data(
    scene_dir: Path,
    scene_slice_data: Optional[SceneSliceData],
    config: SuperstructureSegmentationConfig,
    save_outputs: bool = True,
) -> Optional[SceneSuperstructureResult]:
    """Run superstructure-above / superstructure-below segmentation for one scene."""
    scene_start_time = perf_counter()
    load_time = 0.0
    sidewalk_detection_time = 0.0
    slice_processing_time = 0.0
    save_time = 0.0
    processed_slice_count = 0
    cutting_reference_dir = scene_dir / config.cutting_reference_dir_name
    if scene_slice_data is None:
        deck_dir = scene_dir / config.deck_dir_name
        sup_dir = scene_dir / config.superstructure_dir_name
        deck_files = sort_segment_files(
            [path for path in deck_dir.iterdir() if re.fullmatch(r"deck_(\d+)\.txt", path.name)],
            r"deck_(\d+)\.txt",
        )
        sup_files = sort_segment_files(
            [path for path in sup_dir.iterdir() if re.fullmatch(r"sup_(\d+)\.txt", path.name)],
            r"sup_(\d+)\.txt",
        )
        if not deck_files:
            if config.log_progress:
                print(f"Skipping {scene_dir.name}: no deck slices were generated.")
            return None

        reference_slice_zero_based = get_middle_slice_zero_based(len(deck_files))
        reference_deck_file = deck_files[reference_slice_zero_based]
        sidewalk_start_time = perf_counter()
        has_sidewalk = detect_sidewalk_presence(load_point_cloud_txt(reference_deck_file), config)
        sidewalk_detection_time += perf_counter() - sidewalk_start_time

        scene_segments: List[SceneSliceSegment] = []
        for deck_file, sup_file in zip(deck_files, sup_files):
            load_start_time = perf_counter()
            scene_segments.append(
                SceneSliceSegment(
                    deck_points=load_point_cloud_txt(deck_file),
                    deck_half_1=load_point_cloud_txt(deck_file.with_name(deck_file.stem + "_1.txt")),
                    deck_half_2=load_point_cloud_txt(deck_file.with_name(deck_file.stem + "_2.txt")),
                    superstructure_points=load_point_cloud_txt(sup_file),
                    superstructure_half_1=load_point_cloud_txt(sup_file.with_name(sup_file.stem + "_1.txt")),
                    superstructure_half_2=load_point_cloud_txt(sup_file.with_name(sup_file.stem + "_2.txt")),
                )
            )
            load_time += perf_counter() - load_start_time
    else:
        if len(scene_slice_data.segments) == 0:
            if config.log_progress:
                print(f"Skipping {scene_dir.name}: no slices were generated for sidewalk detection.")
            return None
        reference_slice_zero_based = get_middle_slice_zero_based(len(scene_slice_data.segments))
        sidewalk_start_time = perf_counter()
        has_sidewalk = detect_sidewalk_presence(
            scene_slice_data.segments[reference_slice_zero_based].deck_points,
            config,
        )
        sidewalk_detection_time += perf_counter() - sidewalk_start_time
        scene_segments = scene_slice_data.segments

    above_segments: List[np.ndarray] = []
    below_segments: List[np.ndarray] = []
    segment_worker_count = max(1, int(config.segment_workers))
    slice_results: List[SliceSegmentationResult] = []
    indexed_scene_segments = list(enumerate(scene_segments, start=1))

    if segment_worker_count == 1 or len(indexed_scene_segments) <= 1:
        for slice_index, scene_segment in indexed_scene_segments:
            slice_results.append(
                _segment_scene_slice_worker(
                    slice_index,
                    scene_segment,
                    has_sidewalk,
                    config,
                )
            )
    else:
        if config.log_progress:
            print(f"  Processing scene slices in parallel with {segment_worker_count} workers")
        with ProcessPoolExecutor(max_workers=segment_worker_count) as executor:
            future_map = {
                executor.submit(
                    _segment_scene_slice_worker,
                    slice_index,
                    scene_segment,
                    has_sidewalk,
                    config,
                ): slice_index
                for slice_index, scene_segment in indexed_scene_segments
            }
            for future in as_completed(future_map):
                slice_results.append(future.result())

    slice_results.sort(key=lambda result: result.slice_index)

    for slice_result in slice_results:
        slice_processing_time += slice_result.elapsed_time
        if slice_result.skipped:
            if slice_result.skip_reason == "empty superstructure slice":
                if config.log_progress:
                    print(
                        f"  Slice {slice_result.slice_index}: empty superstructure slice, "
                        f"time={slice_result.elapsed_time:.2f}s"
                    )
            else:
                if config.log_progress:
                    print(
                        f"  Slice {slice_result.slice_index}: skipped ({slice_result.skip_reason}), "
                        f"time={slice_result.elapsed_time:.2f}s"
                    )
            continue

        if save_outputs:
            save_reference_start_time = perf_counter()
            save_point_cloud_txt(
                cutting_reference_dir / f"slice_{slice_result.slice_index}_deck_ref_1.txt",
                slice_result.reference_deck_1,
            )
            save_point_cloud_txt(
                cutting_reference_dir / f"slice_{slice_result.slice_index}_deck_ref_2.txt",
                slice_result.reference_deck_2,
            )

            for side_index, debug_info in ((1, slice_result.debug_side_1), (2, slice_result.debug_side_2)):
                if not debug_info:
                    continue
                for cluster_index, cluster_points in enumerate(debug_info.get("top_clusters", []), start=1):
                    save_point_cloud_txt(
                        cutting_reference_dir / f"slice_{slice_result.slice_index}_side_{side_index}_top3_cluster_{cluster_index}.txt",
                        cluster_points,
                    )
                for cluster_index, cluster_points in enumerate(debug_info.get("comparison_clusters", []), start=1):
                    save_point_cloud_txt(
                        cutting_reference_dir / f"slice_{slice_result.slice_index}_side_{side_index}_comparison_cluster_{cluster_index}.txt",
                        cluster_points,
                    )
                for cluster_index, cluster_points in enumerate(debug_info.get("selected_cluster", []), start=1):
                    save_point_cloud_txt(
                        cutting_reference_dir / f"slice_{slice_result.slice_index}_side_{side_index}_selected_cluster_{cluster_index}.txt",
                        cluster_points,
                    )
            save_time += perf_counter() - save_reference_start_time

        if slice_result.above_points.size > 0:
            above_segments.append(slice_result.above_points)
        if slice_result.below_points.size > 0:
            below_segments.append(slice_result.below_points)
        processed_slice_count += 1
        if config.log_progress:
            print(
                f"  Slice {slice_result.slice_index}: sup_points={slice_result.superstructure_point_count}, "
                f"above_points={len(slice_result.above_points)}, below_points={len(slice_result.below_points)}, "
                f"height_rg_1={slice_result.step_timing.get('height_region_grow_1', 0.0):.2f}s, "
                f"plane1={slice_result.step_timing.get('plane_fit_1', 0.0):.2f}s, "
                f"height_rg_2={slice_result.step_timing.get('height_region_grow_2', 0.0):.2f}s, "
                f"plane2={slice_result.step_timing.get('plane_fit_2', 0.0):.2f}s, "
                f"time={slice_result.elapsed_time:.2f}s"
            )

    combined_above_points = np.vstack(above_segments) if above_segments else np.empty((0, 3), dtype=float)
    combined_below_points = np.vstack(below_segments) if below_segments else np.empty((0, 3), dtype=float)
    if save_outputs:
        save_start_time = perf_counter()
        save_point_cloud_txt(
            scene_dir / config.above_output_name,
            combined_above_points,
        )
        save_point_cloud_txt(
            scene_dir / config.below_output_name,
            combined_below_points,
        )
        save_time += perf_counter() - save_start_time
    elapsed_time = perf_counter() - scene_start_time
    total_above_points = sum(len(points) for points in above_segments)
    total_below_points = sum(len(points) for points in below_segments)
    if config.log_progress:
        print(
            f"Segmented scene: {scene_dir.name} "
            f"({'with sidewalk' if has_sidewalk else 'without sidewalk'}, "
            f"above_points={total_above_points}, below_points={total_below_points}, "
            f"sidewalk_detection={sidewalk_detection_time:.2f}s, load={load_time:.2f}s, "
            f"slice_processing={slice_processing_time:.2f}s, save={save_time:.2f}s, "
            f"slices={processed_slice_count}, total={elapsed_time:.2f}s)"
        )
    return SceneSuperstructureResult(
        above_points=combined_above_points,
        below_points=combined_below_points,
        has_sidewalk=has_sidewalk,
        processed_slice_count=processed_slice_count,
    )


def segment_superstructure_components(
    input_root: Path,
    config: SuperstructureSegmentationConfig,
    scene_slice_map: Optional[Dict[str, SceneSliceData]] = None,
) -> None:
    """Run superstructure segmentation for every scene under the input root."""
    if scene_slice_map is not None:
        for scene_name, scene_slice_data in scene_slice_map.items():
            segment_scene_from_slice_data(scene_slice_data.scene_dir, scene_slice_data, config)
        return

    for scene_dir in scene_directories(input_root):
        segment_scene_from_slice_data(scene_dir, None, config)
