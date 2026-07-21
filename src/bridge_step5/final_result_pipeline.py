from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path


def _handle_remove_readonly(func, target_path, exc_info) -> None:
    os.chmod(target_path, stat.S_IWRITE)
    func(target_path)


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _write_merged_or_copy(deck_file: Path, deck_side_file: Path, target_file: Path) -> dict:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if not deck_file.exists():
        return {"written": False, "used_side_file": False}

    merged_lines = deck_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    used_side = False
    if deck_side_file.exists():
        merged_lines += deck_side_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        used_side = True
    target_file.write_text("\n".join(line for line in merged_lines if line.strip()) + ("\n" if merged_lines else ""), encoding="utf-8")
    return {"written": True, "used_side_file": used_side}


def _copy_numbered_files(source_dir: Path, prefix: str, target_dir: Path) -> list[str]:
    copied = []
    if not source_dir.exists():
        return copied
    source_files = sorted(path for path in source_dir.glob("*.txt") if path.is_file())
    for index, source_file in enumerate(source_files, start=1):
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

    pier_files = sorted(
        path for path in source_dir.glob("*.txt") if path.is_file() and "_pier" in path.stem
    )
    other_files = sorted(
        path for path in source_dir.glob("*.txt") if path.is_file() and "_other" in path.stem
    )

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


def run_step5_final_output(
    step1_output_root: Path,
    refined_scene_root: Path,
    output_root: Path,
    clean_output_root: bool = True,
    write_summary: bool = True,
) -> dict:
    step1_output_root = Path(step1_output_root).resolve()
    refined_scene_root = Path(refined_scene_root).resolve()
    output_root = Path(output_root).resolve()

    if not step1_output_root.exists():
        raise FileNotFoundError(f"Step-1 output root does not exist: {step1_output_root}")
    if not refined_scene_root.exists():
        raise FileNotFoundError(f"Refined scene root does not exist: {refined_scene_root}")

    four_part_root = step1_output_root / "four_part_result"
    if not four_part_root.exists():
        raise FileNotFoundError(f"Step-1 four-part result root does not exist: {four_part_root}")

    if clean_output_root:
        _ensure_clean_dir(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    scene_names = sorted(path.name for path in refined_scene_root.iterdir() if path.is_dir())
    for scene_name in scene_names:
        refined_scene_dir = refined_scene_root / scene_name
        step1_scene_dir = four_part_root / scene_name
        final_scene_dir = output_root / _final_scene_dir_name(scene_name)
        final_scene_dir.mkdir(parents=True, exist_ok=True)

        scene_summary: dict[str, object] = {}

        background_written = _copy_if_exists(
            step1_scene_dir / "background.txt",
            final_scene_dir / "background.txt",
        )
        scene_summary["background"] = background_written

        deck_result = _write_merged_or_copy(
            refined_scene_dir / "deck.txt",
            refined_scene_dir / "deck_with_side" / "deck_side.txt",
            final_scene_dir / "deck.txt",
        )
        scene_summary["deck"] = deck_result

        parapet_written = _copy_if_exists(
            refined_scene_dir / "sup_above.txt",
            final_scene_dir / "parapet.txt",
        )
        girder_written = _copy_if_exists(
            refined_scene_dir / "sup_below.txt",
            final_scene_dir / "girder.txt",
        )
        scene_summary["parapet"] = parapet_written
        scene_summary["girder"] = girder_written

        abutment_files = _copy_numbered_files(
            refined_scene_dir / "sub_patch" / "abutment",
            "abutment",
            final_scene_dir,
        )
        pier_files = _copy_numbered_files(
            refined_scene_dir / "sub_patch" / "only_pier",
            "pier",
            final_scene_dir,
        )
        piercap_result = _copy_pier_detection_outputs(
            refined_scene_dir / "sub_patch" / "pier_piercap" / "pier_detection_result",
            final_scene_dir,
            pier_start_index=len(pier_files) + 1,
        )

        scene_summary["abutment_files"] = abutment_files
        scene_summary["pier_files"] = pier_files
        scene_summary["pier_detection_files"] = piercap_result
        summary[scene_name] = scene_summary

    if write_summary:
        summary_path = output_root.parent / "step5_final_output_summary.json"
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "step1_output_root": str(step1_output_root),
                    "refined_scene_root": str(refined_scene_root),
                    "output_root": str(output_root),
                    "scenes": summary,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
    return summary
