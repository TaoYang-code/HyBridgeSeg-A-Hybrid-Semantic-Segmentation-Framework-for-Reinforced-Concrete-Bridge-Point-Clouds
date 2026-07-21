from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect_substructure_outputs(source_dir: str | Path, target_dir: str | Path, keyword: str = "sub") -> None:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for scene_dir in sorted(source_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        for txt_file in scene_dir.glob("*.txt"):
            if keyword not in txt_file.name:
                continue

            shutil.copy(txt_file, target_dir / f"{scene_dir.name}.txt")
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect substructure outputs from refinement scene folders.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--keyword", default="sub")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_substructure_outputs(args.source_dir, args.target_dir, keyword=args.keyword)
