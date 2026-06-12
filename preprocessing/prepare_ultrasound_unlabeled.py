"""Preprocess unlabeled ultrasound images for reconstruction pretraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.ultrasound_preprocessing import (  # noqa: E402
    list_image_files,
    preprocess_ultrasound_rgb,
    read_image_rgb,
    relpath,
    resolve_project_path,
    save_gray_png,
)


def unique_name(source_path: Path, input_root: Path, used_names: set[str]) -> str:
    try:
        relative = source_path.relative_to(input_root)
        stem = "_".join(relative.with_suffix("").parts)
    except ValueError:
        stem = source_path.stem

    name = f"{stem}.png"
    if name not in used_names:
        used_names.add(name)
        return name

    index = 1
    while True:
        candidate = f"{stem}_{index:03d}.png"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def preprocess_sources(args: argparse.Namespace) -> None:
    output_dir = resolve_project_path(args.output_dir)
    image_out = output_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    preprocessing_config = {
        "target_size": args.target_size,
        "resize_mode": args.resize_mode,
        "median_ksize": args.median_ksize,
        "apply_clahe": args.apply_clahe,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
    }
    with (output_dir / "preprocessing_config.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessing_config, handle, indent=2)
    print(
        "Preprocessing: "
        f"resize_mode={args.resize_mode}, "
        f"target_size={args.target_size}, "
        f"median_ksize={args.median_ksize}, "
        f"apply_clahe={args.apply_clahe}, "
        f"clahe_clip_limit={args.clahe_clip_limit}, "
        f"clahe_tile_grid_size={args.clahe_tile_grid_size}"
    )

    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    scanned = 0
    written = 0

    input_dirs = [resolve_project_path(path_value) for path_value in args.input_dirs]
    for input_dir in input_dirs:
        image_paths = list_image_files(input_dir)
        if args.limit:
            remaining = max(args.limit - scanned, 0)
            image_paths = image_paths[:remaining]

        for source_path in image_paths:
            scanned += 1
            image_rgb = read_image_rgb(source_path)
            if image_rgb is None:
                rows.append({"filename": "", "source": relpath(source_path), "status": "unreadable"})
                continue

            prepared = preprocess_ultrasound_rgb(
                image_rgb,
                target_size=args.target_size,
                resize_mode=args.resize_mode,
                median_ksize=args.median_ksize,
                apply_clahe=args.apply_clahe,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid_size=args.clahe_tile_grid_size,
            )
            output_name = unique_name(source_path, input_dir, used_names)
            output_path = image_out / output_name
            save_gray_png(output_path, prepared)
            rows.append(
                {
                    "filename": relpath(output_path),
                    "source": relpath(source_path),
                    "status": "ok",
                    "mean_intensity": float(np.mean(prepared)),
                }
            )
            written += 1

            if written % args.report_every == 0:
                print(f"Prepared {written} images, scanned {scanned}")

        if args.limit and scanned >= args.limit:
            break

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "source", "status", "mean_intensity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Output directory: {output_dir}")
    print(f"Readable images: {written}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess unlabeled ultrasound images.")
    parser.add_argument("--input-dirs", nargs="+", default=["data/kaggle_dataset", "data/unlabel"])
    parser.add_argument("--output-dir", default="data/ultrasound_unlabeled_processed")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--median-ksize", type=int, default=3)
    parser.add_argument("--apply-clahe", action="store_true", help="Apply CLAHE contrast enhancement before resizing.")
    parser.add_argument("--clahe-clip-limit", type=float, default=3.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_sources(parse_args())
