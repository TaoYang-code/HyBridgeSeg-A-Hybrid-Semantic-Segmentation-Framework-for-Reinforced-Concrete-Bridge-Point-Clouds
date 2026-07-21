from __future__ import annotations

import json
import os
import re
import stat
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABEL_TO_NAME = {
    0: "background",
    1: "deck",
    2: "sub",
    3: "sup_withoutdeck",
}

CHUNK_PATTERN = re.compile(r"^(?:val-)?(?P<scene>.+)__part(?P<part>\d+)$")


@dataclass(frozen=True)
class SceneChunk:
    chunk_name: str
    scene_name: str
    part_index: int
    points: np.ndarray


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    path.mkdir(parents=True, exist_ok=True)


def _handle_remove_readonly(func, target_path, exc_info) -> None:
    os.chmod(target_path, stat.S_IWRITE)
    func(target_path)


def load_raw_point_cloud(file_path: Path) -> np.ndarray:
    points = np.loadtxt(file_path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{file_path} must contain at least xyz columns.")
    return points


def find_input_txt_files(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.glob("*.txt") if path.is_file())


def split_points_into_five_chunks(points: np.ndarray, scene_name: str) -> list[SceneChunk]:
    xyz = points[:, :3]
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    span_x = float(max_xyz[0] - min_xyz[0])
    span_y = float(max_xyz[1] - min_xyz[1])
    split_axis = 0 if span_x >= span_y else 1
    edges = np.linspace(min_xyz[split_axis], max_xyz[split_axis], 6, dtype=np.float32)

    chunks: list[SceneChunk] = []
    for idx in range(5):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == 4:
            mask = (xyz[:, split_axis] >= left) & (xyz[:, split_axis] <= right)
        else:
            mask = (xyz[:, split_axis] >= left) & (xyz[:, split_axis] < right)
        chunk_points = points[mask]
        if chunk_points.size == 0:
            continue
        chunks.append(
            SceneChunk(
                chunk_name=f"{scene_name}__part{idx + 1}",
                scene_name=scene_name,
                part_index=idx + 1,
                points=chunk_points,
            )
        )
    return chunks


def save_preseg_dataset(chunks: list[SceneChunk], dataset_root: Path) -> list[dict]:
    val_root = dataset_root / "val"
    val_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for chunk in chunks:
        chunk_dir = val_root / chunk.chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(chunk_dir / "background.txt", chunk.points, fmt="%.6f")
        summaries.append(
            {
                "scene_name": chunk.scene_name,
                "chunk_name": chunk.chunk_name,
                "part_index": chunk.part_index,
                "num_points": int(chunk.points.shape[0]),
            }
        )
    return summaries


def build_inference_command(
    python_exe: str,
    vendor_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    save_path: Path,
) -> list[str]:
    config_file = vendor_root / "configs" / "bridge" / "semseg-pt-v3m1-0-4class-raw-long5.py"
    return [
        python_exe,
        "tools/test.py",
        "--config-file",
        str(config_file),
        "--num-gpus",
        "1",
        "--options",
        f"weight={checkpoint_path}",
        f"save_path={save_path}",
        f"data.train.data_root={dataset_root}",
        f"data.val.data_root={dataset_root}",
        f"data.test.data_root={dataset_root}",
    ]


def run_module1_inference(
    python_exe: str,
    vendor_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    save_path: Path,
    verbose: bool = True,
) -> None:
    command = build_inference_command(
        python_exe=python_exe,
        vendor_root=vendor_root,
        checkpoint_path=checkpoint_path.resolve(),
        dataset_root=dataset_root.resolve(),
        save_path=save_path.resolve(),
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(vendor_root.resolve())
        if not existing_pythonpath
        else str(vendor_root.resolve()) + os.pathsep + existing_pythonpath
    )
    if verbose:
        subprocess.run(command, cwd=vendor_root, check=True, env=env)
        return

    completed = subprocess.run(
        command,
        cwd=vendor_root,
        check=False,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if completed.returncode != 0:
        tail_lines = (completed.stderr or completed.stdout or "").splitlines()[-40:]
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            "Module-1 inference failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Last logs:\n{tail_text}"
        )


def parse_chunk_result_name(stem: str) -> tuple[str, int]:
    normalized = stem
    if normalized.endswith("_xyz_pred_gt"):
        normalized = normalized[: -len("_xyz_pred_gt")]
    match = CHUNK_PATTERN.match(normalized)
    if not match:
        raise ValueError(f"Unexpected chunk result name: {stem}")
    return match.group("scene"), int(match.group("part"))


def merge_chunk_predictions(result_root: Path, merged_root: Path, four_part_root: Path) -> dict:
    merged_root.mkdir(parents=True, exist_ok=True)
    four_part_root.mkdir(parents=True, exist_ok=True)

    scene_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for txt_path in sorted(result_root.glob("*_xyz_pred_gt.txt")):
        rows = np.loadtxt(txt_path, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        scene_name, _ = parse_chunk_result_name(txt_path.stem)
        scene_rows[scene_name].append(rows)

    summary: dict[str, dict] = {}
    for scene_name, rows_list in scene_rows.items():
        merged_rows = np.vstack(rows_list)
        np.savetxt(
            merged_root / f"{scene_name}_merged_xyz_pred_gt.txt",
            merged_rows,
            fmt="%.6f %.6f %.6f %d %d",
        )

        scene_output_dir = four_part_root / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        scene_summary = {"total_points": int(merged_rows.shape[0]), "classes": {}}

        for label, class_name in LABEL_TO_NAME.items():
            class_points = merged_rows[merged_rows[:, 3].astype(np.int32) == label][:, :3]
            output_path = scene_output_dir / f"{class_name}.txt"
            if class_points.size == 0:
                output_path.write_text("")
                count = 0
            else:
                np.savetxt(output_path, class_points, fmt="%.6f %.6f %.6f")
                count = int(class_points.shape[0])
            scene_summary["classes"][class_name] = count

        summary[scene_name] = scene_summary
    return summary


def run_step1_pipeline(
    project_root: Path,
    input_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    python_exe: str,
    prepare_only: bool,
    verbose: bool = True,
) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    vendor_root = (project_root / "vendor" / "module1_model").resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    dataset_root = output_root / "prepared_dataset"
    raw_chunk_root = output_root / "raw_chunks"
    inference_save_root = output_root / "module1_inference"
    merged_root = output_root / "merged_predictions"
    four_part_root = output_root / "four_part_result"

    ensure_clean_dir(output_root)
    raw_chunk_root.mkdir(parents=True, exist_ok=True)

    all_chunks: list[SceneChunk] = []
    input_files = find_input_txt_files(input_root)
    if not input_files:
        raise FileNotFoundError(f"No txt files found in {input_root}")

    input_summary = []
    for txt_path in input_files:
        scene_name = txt_path.stem
        points = load_raw_point_cloud(txt_path)
        chunks = split_points_into_five_chunks(points, scene_name=scene_name)
        if not chunks:
            raise ValueError(f"No valid chunks generated for {txt_path}")
        all_chunks.extend(chunks)
        scene_dir = raw_chunk_root / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            np.savetxt(scene_dir / f"{chunk.chunk_name}.txt", chunk.points, fmt="%.6f")
        input_summary.append(
            {
                "scene_name": scene_name,
                "input_file": str(txt_path),
                "num_points": int(points.shape[0]),
                "num_chunks": len(chunks),
            }
        )

    ensure_clean_dir(dataset_root)
    chunk_summary = save_preseg_dataset(all_chunks, dataset_root=dataset_root)

    summary = {
        "input_root": str(input_root),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "prepare_only": prepare_only,
        "scenes": input_summary,
        "chunks": chunk_summary,
    }

    if not prepare_only:
        inference_save_root.mkdir(parents=True, exist_ok=True)
        run_module1_inference(
            python_exe=python_exe,
            vendor_root=vendor_root,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            save_path=inference_save_root,
            verbose=verbose,
        )
        result_root = inference_save_root / "result"
        if not result_root.exists():
            raise FileNotFoundError(f"Expected inference result directory was not created: {result_root}")
        summary["merged_results"] = merge_chunk_predictions(
            result_root=result_root,
            merged_root=merged_root,
            four_part_root=four_part_root,
        )

    with open(output_root / "step1_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if verbose:
        print(f"Prepared {len(input_summary)} raw bridge files into {len(all_chunks)} long-axis chunks.")
        print("Step-1 outputs were written to the configured output directory.")
        if prepare_only:
            print("Model inference was skipped because --prepare-only was used.")
