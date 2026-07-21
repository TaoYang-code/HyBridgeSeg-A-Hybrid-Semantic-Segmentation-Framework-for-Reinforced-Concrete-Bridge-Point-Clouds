from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import deck_candidate_region_growing
from . import deck_cluster_filter
from . import deck_planarity_filter
from .mass_driven_curve_skeleton_port.pipeline import (
    generate_and_save_ordered_skeleton,
    generate_ordered_skeleton_from_deck_points,
)
from . import residual_class_assignment
from . import substructure_cluster_split
from . import superstructure_cluster_filter
from .io_utils import (
    map_labels_from_downsampled_points,
    save_xyz_points,
    split_module1_labels,
    voxel_downsample_numpy,
)


DEFAULT_INPUT_DIR = Path(
    r"C:\1 new data\PTv3\real_to_real\bridge\with_background\combined_method\2 three_class_segmentation_result"
)
DEFAULT_OUTPUT_DIR = Path("predicted_vis_result")
DECK_CLUSTER_DBSCAN_VOXEL_SIZE = 0.07
DECK_GROWTH_Z_MARGIN = 0.2
DECK_REGION_GROWING_SEED_VOXEL_SIZE = 0.4
SUP_SUB_DBSCAN_VOXEL_SIZE = 0.07


@dataclass(frozen=True)
class RefinedModule1Result:
    deck_for_cutting_points: np.ndarray
    deck_points: np.ndarray
    superstructure_points: np.ndarray
    substructure_points: np.ndarray
    ordered_skeleton_points: np.ndarray


def refine_module1_prediction(
    data: np.ndarray,
    output_folder: Path,
    save_intermediate: bool = True,
    persist_outputs: bool = True,
) -> RefinedModule1Result:
    pipeline_start_time = time.perf_counter()
    deck_points, substructure_points, superstructure_points = split_module1_labels(data)
    print("Loaded Module 1 pre-segmentation result.")

    stage_start_time = time.perf_counter()
    downsampled_deck_points = voxel_downsample_numpy(
        deck_points,
        voxel_size=DECK_CLUSTER_DBSCAN_VOXEL_SIZE,
    )
    deck_labels_downsampled = deck_cluster_filter.cluster_points(
        downsampled_deck_points,
        eps=0.2,
        min_samples=1,
    )
    _, main_deck_label = deck_cluster_filter.extract_largest_cluster(
        downsampled_deck_points,
        deck_labels_downsampled,
    )
    deck_labels = map_labels_from_downsampled_points(
        deck_points,
        downsampled_deck_points,
        deck_labels_downsampled,
    )
    main_deck_patch = deck_points[deck_labels == main_deck_label]
    residual_deck_points = (
        deck_points[deck_labels != main_deck_label]
        if main_deck_label >= 0
        else np.empty((0, 3))
    )
    print(
        f"[Timing] deck cluster filtering: {time.perf_counter() - stage_start_time:.2f}s "
        f"(voxelized {len(deck_points)} -> {len(downsampled_deck_points)})"
    )

    if save_intermediate:
        save_xyz_points(main_deck_patch, output_folder / "largest_patch.txt")
        save_xyz_points(residual_deck_points, output_folder / "non_largest_patch.txt")

    stage_start_time = time.perf_counter()
    sampled_deck_points = deck_planarity_filter.voxel_sampling(main_deck_patch, voxel_size=0.2)
    planar_deck_points, non_planar_deck_points = deck_planarity_filter.remove_non_planar_points(
        main_deck_patch,
        sampled_deck_points,
        radius=0.5,
        distance_threshold=0.05,
        visualize=False,
    )
    print(
        f"[Timing] deck planarity filtering: {time.perf_counter() - stage_start_time:.2f}s "
        f"(sampled seeds: {len(sampled_deck_points)})"
    )

    if save_intermediate:
        save_xyz_points(planar_deck_points, output_folder / "plane_patch.txt")
        save_xyz_points(non_planar_deck_points, output_folder / "Non_plane_patch.txt")

    candidate_points_with_labels = np.vstack(
        (
            np.hstack((non_planar_deck_points, np.ones((len(non_planar_deck_points), 1)))),
            np.hstack((residual_deck_points, np.ones((len(residual_deck_points), 1)))),
            np.hstack((superstructure_points, np.ones((len(superstructure_points), 1)))),
            np.hstack((substructure_points, np.zeros((len(substructure_points), 1)))),
        )
    )

    deck_z_min = float(np.min(planar_deck_points[:, 2]))
    deck_z_max = float(np.max(planar_deck_points[:, 2]))
    nearby_candidate_mask = (
        (candidate_points_with_labels[:, 2] >= deck_z_min - DECK_GROWTH_Z_MARGIN)
        & (candidate_points_with_labels[:, 2] <= deck_z_max + DECK_GROWTH_Z_MARGIN)
    )

    nearby_candidate_points_with_labels = candidate_points_with_labels[nearby_candidate_mask]

    far_candidate_points_with_labels = candidate_points_with_labels[~nearby_candidate_mask]
    print(
        f"[Timing] candidate Z filtering: kept {len(nearby_candidate_points_with_labels)}/{len(candidate_points_with_labels)} "
        f"points within deck Z range [{deck_z_min - DECK_GROWTH_Z_MARGIN:.2f}, {deck_z_max + DECK_GROWTH_Z_MARGIN:.2f}]"
    )
    if save_intermediate:
        np.savetxt(
            output_folder / "nearby_candidate_points_with_labels.txt",
            nearby_candidate_points_with_labels,
            fmt="%.6f %.6f %.6f %.0f",
        )

    stage_start_time = time.perf_counter()
    candidate_normals = deck_candidate_region_growing.compute_candidate_normals(
        nearby_candidate_points_with_labels[:, :3],
        reference_points=planar_deck_points,
    )
    print(
        f"[Timing] candidate normal estimation: {time.perf_counter() - stage_start_time:.2f}s "
        f"(nearby candidates: {len(nearby_candidate_points_with_labels)})"
    )

    stage_start_time = time.perf_counter()
    refined_deck_points, nearby_remaining_points_with_labels, grown_candidate_points_with_labels = deck_candidate_region_growing.grow_deck_region(
        deck_points=planar_deck_points,
        candidate_points_with_labels=nearby_candidate_points_with_labels,
        candidate_normals=candidate_normals,
        normal_threshold=np.deg2rad(10),
        voxel_size=DECK_REGION_GROWING_SEED_VOXEL_SIZE,
    )
    remaining_points_with_labels = np.vstack(
        (nearby_remaining_points_with_labels, far_candidate_points_with_labels)
    )
    print(
        f"[Timing] deck region growing: {time.perf_counter() - stage_start_time:.2f}s "
        f"(nearby candidate points: {len(nearby_candidate_points_with_labels)})"
    )
    if save_intermediate:
        np.savetxt(
            output_folder / "grown_candidate_points_with_labels.txt",
            grown_candidate_points_with_labels,
            fmt="%.6f %.6f %.6f %.0f",
        )
    if persist_outputs:
        save_xyz_points(refined_deck_points, output_folder / "deck_for_cutting.txt")

    updated_substructure_points = remaining_points_with_labels[remaining_points_with_labels[:, -1] == 0][:, :3]
    updated_superstructure_points = remaining_points_with_labels[remaining_points_with_labels[:, -1] == 1][:, :3]
    if save_intermediate:
        save_xyz_points(updated_substructure_points, output_folder / "updated_sub.txt")
        save_xyz_points(updated_superstructure_points, output_folder / "updated_super.txt")

    stage_start_time = time.perf_counter()
    downsampled_superstructure_points = voxel_downsample_numpy(
        updated_superstructure_points,
        voxel_size=SUP_SUB_DBSCAN_VOXEL_SIZE,
    )
    superstructure_labels_downsampled = superstructure_cluster_filter.cluster_points(
        downsampled_superstructure_points,
        eps=0.2,
        min_samples=1,
    )
    _, main_superstructure_label = (
        superstructure_cluster_filter.extract_largest_non_noise_cluster(
            downsampled_superstructure_points,
            superstructure_labels_downsampled,
        )
    )
    superstructure_labels = map_labels_from_downsampled_points(
        updated_superstructure_points,
        downsampled_superstructure_points,
        superstructure_labels_downsampled,
    )
    main_superstructure_points = (
        updated_superstructure_points[superstructure_labels == main_superstructure_label]
        if main_superstructure_label >= 0
        else np.empty((0, 3))
    )
    residual_superstructure_points = (
        updated_superstructure_points[superstructure_labels != main_superstructure_label]
        if main_superstructure_label >= 0
        else updated_superstructure_points
    )
    print(
        f"[Timing] superstructure cluster filtering: {time.perf_counter() - stage_start_time:.2f}s "
        f"(voxelized {len(updated_superstructure_points)} -> {len(downsampled_superstructure_points)})"
    )

    if save_intermediate:
        save_xyz_points(main_superstructure_points, output_folder / "sup_largest_patch.txt")
        save_xyz_points(residual_superstructure_points, output_folder / "sup_non_largest_patch.txt")

    stage_start_time = time.perf_counter()
    downsampled_substructure_points = voxel_downsample_numpy(
        updated_substructure_points,
        voxel_size=SUP_SUB_DBSCAN_VOXEL_SIZE,
    )
    substructure_labels_downsampled = substructure_cluster_split.cluster_points(
        downsampled_substructure_points,
        eps=0.2,
        min_samples=1,
    )
    unique_labels_downsampled = np.unique(substructure_labels_downsampled)
    original_size_threshold = 5000
    downsampled_size_threshold = max(
        1,
        int(round(original_size_threshold * len(downsampled_substructure_points) / max(len(updated_substructure_points), 1))),
    )
    main_label_mask = np.zeros(len(downsampled_substructure_points), dtype=bool)
    for label in unique_labels_downsampled:
        label_mask = substructure_labels_downsampled == label
        if np.sum(label_mask) >= downsampled_size_threshold:
            main_label_mask |= label_mask

    downsampled_main_substructure_labels = main_label_mask.astype(int)
    substructure_main_mask = map_labels_from_downsampled_points(
        updated_substructure_points,
        downsampled_substructure_points,
        downsampled_main_substructure_labels,
    ).astype(bool)
    main_substructure_points = updated_substructure_points[substructure_main_mask]
    residual_substructure_points = updated_substructure_points[~substructure_main_mask]
    print(
        f"[Timing] substructure cluster split: {time.perf_counter() - stage_start_time:.2f}s "
        f"(voxelized {len(updated_substructure_points)} -> {len(downsampled_substructure_points)}, "
        f"threshold {original_size_threshold} -> {downsampled_size_threshold})"
    )

    stage_start_time = time.perf_counter()
    final_deck_points, final_superstructure_points, final_substructure_points = residual_class_assignment.assign_residual_groups(
        deck_points=refined_deck_points,
        superstructure_points=main_superstructure_points,
        substructure_points=main_substructure_points,
        residual_substructure_points=residual_substructure_points,
        residual_superstructure_points=residual_superstructure_points,
        output_folder=output_folder,
        save_outputs=False,
    )
    if persist_outputs:
        save_xyz_points(final_deck_points, output_folder / "deck.txt")
        save_xyz_points(final_superstructure_points, output_folder / "sup.txt")
        save_xyz_points(final_substructure_points, output_folder / "sub.txt")
    print(f"[Timing] residual class assignment: {time.perf_counter() - stage_start_time:.2f}s")

    stage_start_time = time.perf_counter()
    ordered_skeleton_points = (
        generate_and_save_ordered_skeleton(
            final_deck_points,
            output_folder / "ordered_skeleton.txt",
        )
        if persist_outputs
        else generate_ordered_skeleton_from_deck_points(final_deck_points)
    )
    print(
        f"[Timing] mass-driven skeleton generation: {time.perf_counter() - stage_start_time:.2f}s "
        f"(skeleton points: {len(ordered_skeleton_points)})"
    )

    print(f"[Timing] total scene refinement: {time.perf_counter() - pipeline_start_time:.2f}s")
    return RefinedModule1Result(
        deck_for_cutting_points=refined_deck_points,
        deck_points=final_deck_points,
        superstructure_points=final_superstructure_points,
        substructure_points=final_substructure_points,
        ordered_skeleton_points=ordered_skeleton_points,
    )


def process_input_folder(
    input_folder: Path,
    output_folder: Path,
    save_intermediate: bool = True,
    persist_outputs: bool = True,
) -> dict[str, RefinedModule1Result]:
    input_files = sorted(input_folder.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No .txt files found in {input_folder}")

    results: dict[str, RefinedModule1Result] = {}
    for input_file in input_files:
        print(f"\nRefining {input_file.name}...")
        scene_output_folder = output_folder / input_file.stem
        scene_output_folder.mkdir(parents=True, exist_ok=True)
        data = np.loadtxt(input_file, delimiter=" ")
        results[input_file.stem] = refine_module1_prediction(
            data,
            scene_output_folder,
            save_intermediate=save_intermediate,
            persist_outputs=persist_outputs,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine Module 1 three-class bridge pre-segmentation results."
    )
    parser.add_argument("--input-folder", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-intermediate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_input_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        save_intermediate=not args.no_intermediate,
        persist_outputs=True,
    )
