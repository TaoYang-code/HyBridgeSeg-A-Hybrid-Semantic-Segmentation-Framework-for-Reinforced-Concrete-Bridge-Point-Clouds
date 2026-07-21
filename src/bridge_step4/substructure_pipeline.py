from __future__ import annotations

import csv
import io
import json
import os
import shutil
import stat
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from .substructure_segmentation.patch_clustering import (
    PatchClusteringConfig,
    run_substructure_patch_clustering,
)
from .substructure_segmentation.pier_detection import (
    PierDetectionConfig,
    run_pier_detection,
)


CLASS_NAME_TO_TARGET_DIR = {
    "onlyabutment": "abutment",
    "onlypier": "only_pier",
    "pierwithpiercap": "pier_piercap",
}

PLACEHOLDER_CLASS_NAME = "onlypier"


def _handle_remove_readonly(func, target_path, exc_info) -> None:
    os.chmod(target_path, stat.S_IWRITE)
    func(target_path)


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    path.mkdir(parents=True, exist_ok=True)


def _prepare_classifier_dataset(
    input_root: Path,
    dataset_root: Path,
) -> tuple[Path, dict[str, dict[str, str]]]:
    val_dir = dataset_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    sample_lookup: dict[str, dict[str, str]] = {}

    for scene_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        sub_patch_dir = scene_dir / "sub_patch"
        if not sub_patch_dir.exists():
            continue
        for cluster_file in sorted(sub_patch_dir.glob("sub_*.txt")):
            sample_name = f"{scene_dir.name}__{cluster_file.stem}.txt"
            target_file = val_dir / sample_name
            shutil.copy2(cluster_file, target_file)
            sample_lookup[sample_name] = {
                "scene_name": scene_dir.name,
                "cluster_file": str(cluster_file.resolve()),
            }
            manifest_rows.append(
                {
                    "split": "val",
                    "sample_name": sample_name,
                    "class_name": PLACEHOLDER_CLASS_NAME,
                    "source_bridge": scene_dir.name,
                    "source_files": cluster_file.name,
                    "cluster_index": cluster_file.stem,
                    "num_points": sum(1 for _ in open(cluster_file, "r", encoding="utf-8", errors="ignore")),
                }
            )

    if not manifest_rows:
        raise FileNotFoundError(f"No clustered substructure files found under: {input_root}")

    manifest_path = dataset_root / "val_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "sample_name",
                "class_name",
                "source_bridge",
                "source_files",
                "cluster_index",
                "num_points",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path, sample_lookup


def _build_classifier_command(
    python_exe: str,
    vendor_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    save_path: Path,
    score_dump_path: Path,
) -> list[str]:
    config_file = vendor_root / "configs" / "bridge" / "cls-ptv3-module2-base.py"
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
        f"data_root={dataset_root}",
        f"data.train.data_root={dataset_root}",
        f"data.val.data_root={dataset_root}",
        f"data.test.data_root={dataset_root}",
        "dump_prior_scores=True",
        f"prior_score_dump_path={score_dump_path}",
    ]


def _run_classifier_inference(
    python_exe: str,
    vendor_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    save_path: Path,
    score_dump_path: Path,
    verbose: bool = True,
) -> None:
    command = _build_classifier_command(
        python_exe=python_exe,
        vendor_root=vendor_root,
        checkpoint_path=checkpoint_path.resolve(),
        dataset_root=dataset_root.resolve(),
        save_path=save_path.resolve(),
        score_dump_path=score_dump_path.resolve(),
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(vendor_root.resolve())
        if not existing_pythonpath
        else str(vendor_root.resolve()) + os.pathsep + existing_pythonpath
    )
    if verbose:
        try:
            subprocess.run(command, cwd=vendor_root, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            if score_dump_path.exists() and score_dump_path.stat().st_size > 0:
                print(
                    "Classifier test script exited non-zero after producing the score dump. "
                    "Continuing with the saved predictions."
                )
                return
            raise exc
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
    if completed.returncode == 0:
        return
    try:
        # The vendored classifier test script can crash after evaluation while
        # finalizing metrics, even though the score dump we need was already written.
        if score_dump_path.exists() and score_dump_path.stat().st_size > 0:
            return
    except OSError:
        pass
    tail_lines = (completed.stderr or completed.stdout or "").splitlines()[-40:]
    tail_text = "\n".join(tail_lines)
    raise RuntimeError(
        "Module-2 classifier inference failed.\n"
        f"Command: {' '.join(command)}\n"
        f"Last logs:\n{tail_text}"
    )


def _read_classifier_predictions(score_dump_path: Path) -> dict[str, str]:
    if not score_dump_path.exists():
        raise FileNotFoundError(f"Classifier score dump was not created: {score_dump_path}")

    predictions: dict[str, str] = {}
    with open(score_dump_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            predictions[row["sample_name"]] = row["raw_pred"].strip().lower()
    if not predictions:
        raise ValueError(f"No predictions found in classifier score dump: {score_dump_path}")
    return predictions


def _reset_classification_dirs(input_root: Path) -> None:
    for scene_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        sub_patch_dir = scene_dir / "sub_patch"
        if not sub_patch_dir.exists():
            continue
        for folder_name in ("abutment", "only_pier", "pier_piercap"):
            target_dir = sub_patch_dir / folder_name
            _ensure_clean_dir(target_dir)


def _dispatch_clusters_by_prediction(
    input_root: Path,
    sample_lookup: dict[str, dict[str, str]],
    predictions: dict[str, str],
) -> dict[str, dict[str, int]]:
    _reset_classification_dirs(input_root)
    summary: dict[str, dict[str, int]] = {}

    for sample_name, meta in sample_lookup.items():
        pred_class = predictions.get(sample_name)
        if pred_class is None:
            raise KeyError(f"Missing classifier prediction for sample: {sample_name}")
        if pred_class not in CLASS_NAME_TO_TARGET_DIR:
            raise KeyError(f"Unexpected classifier label '{pred_class}' for sample: {sample_name}")

        scene_name = meta["scene_name"]
        source_file = Path(meta["cluster_file"])
        target_dir = input_root / scene_name / "sub_patch" / CLASS_NAME_TO_TARGET_DIR[pred_class]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_dir / source_file.name)

        scene_summary = summary.setdefault(
            scene_name,
            {"onlyabutment": 0, "onlypier": 0, "pierwithpiercap": 0},
        )
        scene_summary[pred_class] += 1

    return summary


def run_step4_substructure(
    input_root: Path,
    project_root: Path,
    checkpoint_path: Path,
    python_exe: str = "python",
    voxel_size: float = 0.4,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 1,
    orient_normals_consistently: bool = False,
    pier_workers: int = 5,
    analysis_root: Path | None = None,
    verbose: bool = True,
) -> dict:
    input_root = Path(input_root).resolve()
    project_root = Path(project_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Substructure input root does not exist: {input_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint_path}")

    vendor_root = project_root / "vendor" / "module2_classifier"
    if analysis_root is None:
        analysis_root = project_root / "runs" / "step4_substructure" / "classification_runtime"
    analysis_root = Path(analysis_root).resolve()
    dataset_root = analysis_root / "classifier_dataset"
    save_path = analysis_root / "classifier_inference"
    score_dump_path = analysis_root / "classifier_score_dump.csv"

    _ensure_clean_dir(analysis_root)
    _ensure_clean_dir(dataset_root)
    save_path.mkdir(parents=True, exist_ok=True)

    clustering_config = replace(
        PatchClusteringConfig(),
        voxel_size=voxel_size,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    if verbose:
        run_substructure_patch_clustering(input_root, config=clustering_config)
    else:
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                run_substructure_patch_clustering(input_root, config=clustering_config)
        except Exception as exc:
            tail_text = "\n".join((captured_stderr.getvalue() or captured_stdout.getvalue()).splitlines()[-40:])
            raise RuntimeError(f"Substructure clustering failed for input root {input_root}.\nLast logs:\n{tail_text}") from exc

    _, sample_lookup = _prepare_classifier_dataset(input_root, dataset_root)
    _run_classifier_inference(
        python_exe=python_exe,
        vendor_root=vendor_root,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        save_path=save_path,
        score_dump_path=score_dump_path,
        verbose=verbose,
    )
    predictions = _read_classifier_predictions(score_dump_path)
    dispatch_summary = _dispatch_clusters_by_prediction(input_root, sample_lookup, predictions)

    pier_config = replace(
        PierDetectionConfig(),
        orient_normals_consistently=orient_normals_consistently,
        max_workers=max(1, pier_workers),
    )
    if verbose:
        run_pier_detection(input_root, config=pier_config)
    else:
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                run_pier_detection(input_root, config=pier_config)
        except Exception as exc:
            tail_text = "\n".join((captured_stderr.getvalue() or captured_stdout.getvalue()).splitlines()[-40:])
            raise RuntimeError(f"Pier detection failed for input root {input_root}.\nLast logs:\n{tail_text}") from exc

    summary = {
        "input_root": str(input_root),
        "checkpoint_path": str(checkpoint_path),
        "num_cluster_samples": len(sample_lookup),
        "scene_cluster_predictions": dispatch_summary,
        "classifier_score_dump": str(score_dump_path),
    }
    summary_path = analysis_root / "step4_substructure_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary
