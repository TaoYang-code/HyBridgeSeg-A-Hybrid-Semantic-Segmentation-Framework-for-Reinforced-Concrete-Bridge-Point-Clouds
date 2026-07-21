from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Optional

from .component_separation import (
    SuperstructureSegmentationConfig,
    segment_superstructure_components,
)
from .deck_sup_slicing import SliceSegmentationConfig, slice_superstructure_inputs


@dataclass(frozen=True)
class SuperstructureSegmentationPipelineConfig:
    slicing: SliceSegmentationConfig = field(default_factory=SliceSegmentationConfig)
    segmentation: SuperstructureSegmentationConfig = field(default_factory=SuperstructureSegmentationConfig)


def run_superstructure_segmentation(
    input_root: Path,
    config: Optional[SuperstructureSegmentationPipelineConfig] = None,
) -> None:
    """Run the complete superstructure segmentation pipeline on every scene folder."""
    pipeline_config = config or SuperstructureSegmentationPipelineConfig()
    input_root = Path(input_root)

    print(f"Running superstructure segmentation on: {input_root}")
    total_start_time = perf_counter()

    slicing_start_time = perf_counter()
    scene_slice_map = slice_superstructure_inputs(input_root, pipeline_config.slicing)
    slicing_elapsed_time = perf_counter() - slicing_start_time
    print(f"Stage timing - deck/superstructure slicing: {slicing_elapsed_time:.2f}s")

    segmentation_start_time = perf_counter()
    segment_superstructure_components(
        input_root,
        pipeline_config.segmentation,
        scene_slice_map=scene_slice_map,
    )
    segmentation_elapsed_time = perf_counter() - segmentation_start_time
    print(f"Stage timing - above/below component separation: {segmentation_elapsed_time:.2f}s")

    total_elapsed_time = perf_counter() - total_start_time
    print(f"Stage timing - total superstructure segmentation: {total_elapsed_time:.2f}s")
