from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np

from .superstructure_segmentation.component_separation import (
    SceneSuperstructureResult,
    SuperstructureSegmentationConfig,
    segment_scene_from_slice_data,
)
from .superstructure_segmentation.deck_sup_slicing import (
    SliceSegmentationConfig,
    build_scene_slice_data_from_points,
)
from .superstructure_segmentation.pipeline import (
    SuperstructureSegmentationPipelineConfig,
    run_superstructure_segmentation,
)


def run_step3_superstructure(
    input_root: Path,
    workers: int = 1,
    segment_workers: int = 5,
    save_intermediate_slices: bool = True,
    verbose: bool = True,
) -> None:
    input_root = Path(input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Superstructure input root does not exist: {input_root}")

    pipeline_config = SuperstructureSegmentationPipelineConfig(
        slicing=replace(
            SliceSegmentationConfig(),
            save_intermediate_slices=save_intermediate_slices,
            num_workers=max(1, workers),
            log_progress=verbose,
        ),
        segmentation=replace(
            SuperstructureSegmentationConfig(),
            segment_workers=max(1, segment_workers),
            log_progress=verbose,
        ),
    )
    if verbose:
        run_superstructure_segmentation(input_root, config=pipeline_config)
        return

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            run_superstructure_segmentation(input_root, config=pipeline_config)
    except Exception as exc:
        tail_text = "\n".join((captured_stderr.getvalue() or captured_stdout.getvalue()).splitlines()[-40:])
        raise RuntimeError(f"Superstructure processing failed for input root {input_root}.\nLast logs:\n{tail_text}") from exc


def run_step3_superstructure_from_arrays(
    scene_name: str,
    deck_points: np.ndarray,
    skeleton_points: np.ndarray,
    superstructure_points: np.ndarray,
    output_dir: Path,
    segment_workers: int = 5,
    save_intermediate_slices: bool = False,
) -> SceneSuperstructureResult:
    slicing_config = replace(
        SliceSegmentationConfig(),
        save_intermediate_slices=save_intermediate_slices,
        log_progress=False,
    )
    segmentation_config = replace(
        SuperstructureSegmentationConfig(),
        segment_workers=max(1, segment_workers),
        log_progress=False,
    )
    scene_slice_data = build_scene_slice_data_from_points(
        scene_name=scene_name,
        deck_points=deck_points,
        skeleton_points=skeleton_points,
        superstructure_points=superstructure_points,
        config=slicing_config,
    )
    if scene_slice_data is None:
        raise RuntimeError(f"Superstructure slicing produced no usable slices for scene {scene_name}.")
    result = segment_scene_from_slice_data(
        Path(output_dir),
        scene_slice_data,
        segmentation_config,
        save_outputs=save_intermediate_slices,
    )
    if result is None:
        raise RuntimeError(f"Superstructure segmentation produced no result for scene {scene_name}.")
    return result
