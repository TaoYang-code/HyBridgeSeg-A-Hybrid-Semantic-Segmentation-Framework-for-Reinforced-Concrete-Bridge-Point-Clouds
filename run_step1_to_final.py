from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from src.bridge_step1.module1_pipeline import run_step1_pipeline
from src.bridge_step2.refine_pipeline import run_step2_refinement
from src.bridge_step3.superstructure_pipeline import (
    run_step3_superstructure,
    run_step3_superstructure_from_arrays,
)
from src.bridge_step4.substructure_pipeline import run_step4_substructure
from src.bridge_step5.final_result_pipeline import run_step5_final_output


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _find_input_txt_files(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.glob("*.txt") if path.is_file())


def _save_xyz_points(target_file: Path, points: np.ndarray) -> None:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        target_file.write_text("", encoding="utf-8")
        return
    np.savetxt(target_file, points[:, :3], fmt="%.6f %.6f %.6f")


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _copy_numbered_files(source_dir: Path, prefix: str, target_dir: Path) -> list[str]:
    copied = []
    if not source_dir.exists():
        return copied
    for index, source_file in enumerate(sorted(source_dir.glob("*.txt")), start=1):
        target_name = f"{prefix}_{index}.txt"
        shutil.copy2(source_file, target_dir / target_name)
        copied.append(target_name)
    return copied


def _copy_pier_detection_outputs(
    source_dir: Path,
    target_dir: Path,
    pier_start_index: int = 1,
    piercap_start_index: int = 1,
) -> dict[str, list[str]]:
    result = {"pier": [], "piercap": []}
    if not source_dir.exists():
        return result

    pier_files = sorted(path for path in source_dir.glob("*.txt") if "_pier" in path.stem)
    other_files = sorted(path for path in source_dir.glob("*.txt") if "_other" in path.stem)
    for index, source_file in enumerate(pier_files, start=pier_start_index):
        target_name = f"pier_{index}.txt"
        shutil.copy2(source_file, target_dir / target_name)
        result["pier"].append(target_name)
    for index, source_file in enumerate(other_files, start=piercap_start_index):
        target_name = f"piercap_{index}.txt"
        shutil.copy2(source_file, target_dir / target_name)
        result["piercap"].append(target_name)
    return result


def _final_scene_dir_name(scene_name: str) -> str:
    return f"val-{scene_name}_merged_xyz_pred_gt"


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the full bridge segmentation pipeline from raw point clouds to final integrated outputs."
    )
    parser.add_argument("--input-root", type=Path, default=project_root / "data_input")
    parser.add_argument(
        "--module1-checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "Pre-segmentation_in_Module-1.pth",
    )
    parser.add_argument(
        "--module2-checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "classification_in_Module-2.pth",
    )
    parser.add_argument(
        "--step1-output-root",
        type=Path,
        default=project_root / "runs" / "step1_module1",
    )
    parser.add_argument(
        "--step2-output-folder",
        type=Path,
        default=project_root / "runs" / "step2_module1_refine" / "predicted_vis_result",
    )
    parser.add_argument(
        "--step5-output-root",
        type=Path,
        default=project_root / "runs" / "step5_final_output" / "whole_result",
    )
    parser.add_argument("--python-exe", type=str, default="python")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Persist step1/step2/step4 intermediate files under the configured runs directory.",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--segment-workers", type=int, default=5)
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--dbscan-eps", type=float, default=3.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=1)
    parser.add_argument("--orient-normals-consistently", action="store_true")
    parser.add_argument("--pier-workers", type=int, default=5)
    parser.add_argument(
        "--sequential-branches",
        action="store_true",
        help="Run superstructure and substructure stages sequentially instead of in parallel.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    save_intermediate = args.save_intermediate and not args.no_intermediate
    input_root = args.input_root.resolve()
    input_files = _find_input_txt_files(input_root)
    if not input_files:
        raise FileNotFoundError(f"No txt files found in {input_root}")

    total_start_time = perf_counter()
    timing_summary: dict[str, object] = {
        "save_intermediate": save_intermediate,
        "sequential_branches": args.sequential_branches,
        "input_root": str(input_root),
        "bridges": [],
    }
    if not save_intermediate and args.step5_output_root.exists():
        shutil.rmtree(args.step5_output_root)
    args.step5_output_root.mkdir(parents=True, exist_ok=True)

    for bridge_index, input_file in enumerate(input_files, start=1):
        scene_name = input_file.stem
        bridge_start_time = perf_counter()
        bridge_timing: dict[str, object] = {
            "scene_name": scene_name,
            "input_file": str(input_file.resolve()),
            "stages": {},
        }
        print(f"\n[Pipeline] Processing bridge {bridge_index}/{len(input_files)}: {scene_name}")

        with ExitStack() as stack:
            if save_intermediate:
                bridge_root = project_root / "runs" / "per_bridge" / scene_name
                bridge_input_root = bridge_root / "input"
                bridge_input_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_file, bridge_input_root / input_file.name)
                step1_output_root = args.step1_output_root / scene_name
                step2_output_folder = args.step2_output_folder / scene_name / "predicted_vis_result"
                step4_analysis_root = project_root / "runs" / "step4_substructure" / "classification_runtime" / scene_name
            else:
                temp_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=f"bridge_segmentation_{scene_name}_")))
                bridge_input_root = temp_root / "input"
                bridge_input_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_file, bridge_input_root / input_file.name)
                step1_output_root = temp_root / "step1_module1"
                step2_output_folder = temp_root / "step2_module1_refine" / "predicted_vis_result"
                step4_analysis_root = temp_root / "step4_substructure" / "classification_runtime"

            stage_start_time = perf_counter()
            print(f"[Pipeline][{scene_name}] Step 1/5: pre-segmentation")
            run_step1_pipeline(
                project_root=project_root,
                input_root=bridge_input_root,
                output_root=step1_output_root,
                checkpoint_path=args.module1_checkpoint,
                python_exe=args.python_exe,
                prepare_only=False,
                verbose=False,
            )
            step1_elapsed = perf_counter() - stage_start_time
            bridge_timing["stages"]["pre_segmentation"] = step1_elapsed
            print(f"[Pipeline Timing][{scene_name}] pre-segmentation: {_format_seconds(step1_elapsed)}")

            stage_start_time = perf_counter()
            print(f"[Pipeline][{scene_name}] Step 2/5: refine")
            refinement_results = run_step2_refinement(
                input_folder=step1_output_root / "merged_predictions",
                output_folder=step2_output_folder,
                save_intermediate=save_intermediate,
                persist_outputs=save_intermediate,
                verbose=False,
            )
            refined_scene_result = refinement_results.get(scene_name)
            if refined_scene_result is None:
                raise RuntimeError(f"Missing refinement result for scene {scene_name}.")
            step2_elapsed = perf_counter() - stage_start_time
            bridge_timing["stages"]["refine"] = step2_elapsed
            print(f"[Pipeline Timing][{scene_name}] refine: {_format_seconds(step2_elapsed)}")

            final_scene_dir = args.step5_output_root / _final_scene_dir_name(scene_name)
            final_scene_dir.mkdir(parents=True, exist_ok=True)
            background_file = step1_output_root / "four_part_result" / scene_name / "background.txt"
            _copy_if_exists(background_file, final_scene_dir / "background.txt")
            _save_xyz_points(final_scene_dir / "deck.txt", refined_scene_result.deck_points)

            branch_metrics: dict[str, float] = {}

            def run_superstructure_branch() -> None:
                branch_start_time = perf_counter()
                print(f"[Pipeline][{scene_name}] Step 3/5: superstructure")
                if save_intermediate:
                    run_step3_superstructure(
                        input_root=step2_output_folder,
                        workers=args.workers,
                        segment_workers=args.segment_workers,
                        save_intermediate_slices=save_intermediate,
                        verbose=False,
                    )
                    _copy_if_exists(step2_output_folder / scene_name / "sup_above.txt", final_scene_dir / "parapet.txt")
                    _copy_if_exists(step2_output_folder / scene_name / "sup_below.txt", final_scene_dir / "girder.txt")
                else:
                    superstructure_result = run_step3_superstructure_from_arrays(
                        scene_name=scene_name,
                        deck_points=refined_scene_result.deck_for_cutting_points,
                        skeleton_points=refined_scene_result.ordered_skeleton_points,
                        superstructure_points=refined_scene_result.superstructure_points,
                        output_dir=final_scene_dir,
                        segment_workers=args.segment_workers,
                        save_intermediate_slices=False,
                    )
                    _save_xyz_points(final_scene_dir / "parapet.txt", superstructure_result.above_points)
                    _save_xyz_points(final_scene_dir / "girder.txt", superstructure_result.below_points)
                branch_metrics["superstructure"] = perf_counter() - branch_start_time

            def run_substructure_branch() -> None:
                branch_start_time = perf_counter()
                print(f"[Pipeline][{scene_name}] Step 4/5: substructure")
                if save_intermediate:
                    substructure_input_root = step2_output_folder
                else:
                    substructure_input_root = temp_root / "step4_substructure_input"
                    substructure_scene_dir = substructure_input_root / scene_name
                    substructure_scene_dir.mkdir(parents=True, exist_ok=True)
                    _save_xyz_points(substructure_scene_dir / "sub.txt", refined_scene_result.substructure_points)
                run_step4_substructure(
                    input_root=substructure_input_root,
                    project_root=project_root,
                    checkpoint_path=args.module2_checkpoint,
                    python_exe=args.python_exe,
                    voxel_size=args.voxel_size,
                    dbscan_eps=args.dbscan_eps,
                    dbscan_min_samples=args.dbscan_min_samples,
                    orient_normals_consistently=args.orient_normals_consistently,
                    pier_workers=args.pier_workers,
                    analysis_root=step4_analysis_root,
                    verbose=False,
                )
                substructure_scene_root = substructure_input_root / scene_name / "sub_patch"
                _copy_numbered_files(
                    substructure_scene_root / "abutment",
                    "abutment",
                    final_scene_dir,
                )
                copied_only_pier = _copy_numbered_files(
                    substructure_scene_root / "only_pier",
                    "pier",
                    final_scene_dir,
                )
                _copy_pier_detection_outputs(
                    substructure_scene_root / "pier_piercap" / "pier_detection_result",
                    final_scene_dir,
                    pier_start_index=len(copied_only_pier) + 1,
                )
                branch_metrics["substructure"] = perf_counter() - branch_start_time

            branch_wall_start_time = perf_counter()
            if args.sequential_branches:
                run_superstructure_branch()
                run_substructure_branch()
            else:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(run_superstructure_branch),
                        executor.submit(run_substructure_branch),
                    ]
                    for future in futures:
                        future.result()
            branch_wall_elapsed = perf_counter() - branch_wall_start_time
            bridge_timing["stages"]["superstructure"] = branch_metrics["superstructure"]
            bridge_timing["stages"]["substructure"] = branch_metrics["substructure"]
            bridge_timing["stages"]["branch_wall_time"] = branch_wall_elapsed
            print(
                f"[Pipeline Timing][{scene_name}] superstructure: {_format_seconds(branch_metrics['superstructure'])}"
            )
            print(
                f"[Pipeline Timing][{scene_name}] substructure: {_format_seconds(branch_metrics['substructure'])}"
            )
            print(
                f"[Pipeline Timing][{scene_name}] parallel branch wall time: {_format_seconds(branch_wall_elapsed)}"
            )

            stage_start_time = perf_counter()
            print(f"[Pipeline][{scene_name}] Step 5/5: final integration")
            if save_intermediate:
                run_step5_final_output(
                    step1_output_root=step1_output_root,
                    refined_scene_root=step2_output_folder,
                    output_root=args.step5_output_root,
                    clean_output_root=bridge_index == 1,
                    write_summary=False,
                )
            step5_elapsed = perf_counter() - stage_start_time
            bridge_timing["stages"]["final_integration"] = step5_elapsed
            print(f"[Pipeline Timing][{scene_name}] final integration: {_format_seconds(step5_elapsed)}")

        bridge_total_elapsed = perf_counter() - bridge_start_time
        bridge_timing["stages"]["total"] = bridge_total_elapsed
        timing_summary["bridges"].append(bridge_timing)
        print(f"[Pipeline Timing][{scene_name}] total: {_format_seconds(bridge_total_elapsed)}")

    total_elapsed = perf_counter() - total_start_time
    timing_summary["total"] = total_elapsed
    timing_summary["output_root"] = str(args.step5_output_root.resolve())
    timing_summary_path = args.step5_output_root.parent / "pipeline_timing_summary.json"
    timing_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_summary_path, "w", encoding="utf-8") as file:
        json.dump(timing_summary, file, indent=2, ensure_ascii=False)
    print(f"[Pipeline Timing] total: {_format_seconds(total_elapsed)}")
    print(f"[Pipeline Timing] summary saved to: {timing_summary_path}")


if __name__ == "__main__":
    main()
