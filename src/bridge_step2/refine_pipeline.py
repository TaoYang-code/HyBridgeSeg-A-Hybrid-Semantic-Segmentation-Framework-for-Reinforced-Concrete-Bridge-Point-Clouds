from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from .module1_result_refinement.pipeline import RefinedModule1Result, refine_module1_prediction


def _load_prediction_rows(input_file: Path) -> np.ndarray:
    rows = np.loadtxt(input_file, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] < 4:
        raise ValueError(f"{input_file} must contain at least 4 columns: xyz + label.")
    return rows


def _scene_name_from_prediction_file(input_file: Path) -> str:
    stem = input_file.stem
    suffix = "_merged_xyz_pred_gt"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def run_step2_refinement(
    input_folder: Path,
    output_folder: Path,
    save_intermediate: bool = True,
    persist_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, RefinedModule1Result]:
    input_folder = Path(input_folder).resolve()
    output_folder = Path(output_folder).resolve()

    if not input_folder.exists():
        raise FileNotFoundError(f"Refinement input folder does not exist: {input_folder}")

    input_files = sorted(input_folder.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No .txt prediction files found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    scene_results: dict[str, RefinedModule1Result] = {}
    for input_file in input_files:
        data = _load_prediction_rows(input_file)
        scene_name = _scene_name_from_prediction_file(input_file)
        scene_output_folder = output_folder / scene_name
        scene_output_folder.mkdir(parents=True, exist_ok=True)
        if verbose:
            scene_results[scene_name] = refine_module1_prediction(
                data=data,
                output_folder=scene_output_folder,
                save_intermediate=save_intermediate,
                persist_outputs=persist_outputs,
            )
        else:
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            try:
                with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                    scene_results[scene_name] = refine_module1_prediction(
                        data=data,
                        output_folder=scene_output_folder,
                        save_intermediate=save_intermediate,
                        persist_outputs=persist_outputs,
                    )
            except Exception as exc:
                tail_text = "\n".join(
                    (captured_stderr.getvalue() or captured_stdout.getvalue()).splitlines()[-40:]
                )
                raise RuntimeError(f"Refinement failed for scene {scene_name}.\nLast logs:\n{tail_text}") from exc
    return scene_results
