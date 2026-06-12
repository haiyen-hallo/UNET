"""Prepare LabelMe bongcan annotations for segmentation fine-tuning.

This script keeps the raw LabelMe polygon semantics intact: it first creates a
mask at the original image size, then resizes image and mask with the shared
ultrasound preprocessing pipeline. It writes train/val/test manifests so the
test set is never used by training callbacks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.ultrasound_preprocessing import (  # noqa: E402
    IMAGE_EXTENSIONS,
    make_overlay,
    preprocess_ultrasound_rgb,
    relpath,
    resize_mask,
    save_gray_png,
    valid_region_mask,
)


@dataclass
class PreparedSample:
    stem: str
    source_json: Path
    source_image: Path | None
    image: np.ndarray
    mask: np.ndarray
    valid_mask: np.ndarray


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def find_json_files(data_root: Path, exclude_root: Path | None = None) -> list[Path]:
    files: list[Path] = []
    for path in data_root.rglob("*.json"):
        path_text = path.as_posix().lower()
        if ".backup" in path_text:
            continue
        if exclude_root is not None:
            try:
                path.resolve().relative_to(exclude_root.resolve())
                continue
            except ValueError:
                pass
        files.append(path)
    return sorted(set(files))


def read_labelme_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_image_from_json(data: dict[str, Any]) -> np.ndarray | None:
    image_data = data.get("imageData")
    if not image_data:
        return None
    raw = base64.b64decode(image_data)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def find_image_path(json_path: Path, data: dict[str, Any]) -> Path | None:
    json_dir = json_path.parent
    candidates: list[Path] = []

    image_path = data.get("imagePath")
    if image_path:
        image_path_obj = Path(image_path)
        candidates.append(image_path_obj if image_path_obj.is_absolute() else json_dir / image_path_obj)

    for extension in sorted(IMAGE_EXTENSIONS):
        candidates.append(json_dir / f"{json_path.stem}{extension}")
        candidates.append(json_dir / f"{json_path.stem}{extension.upper()}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_image_rgb(json_path: Path, data: dict[str, Any]) -> tuple[np.ndarray | None, Path | None]:
    image = decode_image_from_json(data)
    if image is not None:
        return image, None

    image_path = find_image_path(json_path, data)
    if image_path is None:
        return None, None

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None, image_path
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), image_path


def shape_to_points(shape: dict[str, Any]) -> np.ndarray | None:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.size == 0:
        return None

    if shape.get("shape_type", "polygon") == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    if len(points) < 3:
        return None
    return points


def build_original_mask(data: dict[str, Any], target_label: str, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") != target_label:
            continue
        points = shape_to_points(shape)
        if points is None:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_sample(
    json_path: Path,
    target_label: str,
    target_size: int,
    resize_mode: str,
    median_ksize: int,
    apply_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid_size: int,
) -> tuple[PreparedSample | None, str]:
    data = read_labelme_json(json_path)
    if not any(shape.get("label") == target_label for shape in data.get("shapes", [])):
        return None, "missing_target_label"

    image_rgb, image_path = read_image_rgb(json_path, data)
    if image_rgb is None:
        return None, "missing_or_unreadable_image"

    old_height, old_width = image_rgb.shape[:2]
    original_mask = build_original_mask(data, target_label, old_width, old_height)
    if not np.any(original_mask):
        return None, "empty_target_mask"

    image = preprocess_ultrasound_rgb(
        image_rgb,
        target_size=target_size,
        resize_mode=resize_mode,
        median_ksize=median_ksize,
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )
    mask = resize_mask(original_mask, target_size, resize_mode)
    valid_mask = valid_region_mask(old_width, old_height, target_size, resize_mode)
    if not np.any(mask):
        return None, "empty_resized_mask"

    return (
        PreparedSample(
            stem=json_path.stem,
            source_json=json_path,
            source_image=image_path,
            image=image,
            mask=mask,
            valid_mask=valid_mask,
        ),
        "",
    )


def save_sample(sample: PreparedSample, split: str, output_dir: Path, preview_count: int, preview_index: int) -> dict[str, Any]:
    image_path = output_dir / split / "images" / f"{sample.stem}.png"
    mask_path = output_dir / split / "masks" / f"{sample.stem}.png"
    valid_mask_path = output_dir / split / "valid_masks" / f"{sample.stem}.png"
    save_gray_png(image_path, sample.image)
    save_gray_png(mask_path, sample.mask)
    save_gray_png(valid_mask_path, sample.valid_mask)

    if preview_count < 0 or preview_index < preview_count:
        preview_path = output_dir / "previews" / split / f"{sample.stem}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(make_overlay(sample.image, sample.mask), mode="RGB").save(preview_path)

    return {
        "filename": relpath(image_path),
        "mask": relpath(mask_path),
        "valid_mask": relpath(valid_mask_path),
        "source_json": relpath(sample.source_json),
        "source_image": relpath(sample.source_image) if sample.source_image is not None else "",
        "split": split,
        "augmented": "none",
    }


def split_samples(
    samples: list[PreparedSample],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[PreparedSample], list[PreparedSample], list[PreparedSample]]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("--val-fraction and --test-fraction must be > 0 and sum to < 1.")

    samples = samples.copy()
    random.Random(seed).shuffle(samples)
    total = len(samples)
    test_count = max(1, int(round(total * test_fraction)))
    val_count = max(1, int(round(total * val_fraction)))
    test_samples = samples[:test_count]
    val_samples = samples[test_count : test_count + val_count]
    train_samples = samples[test_count + val_count :]
    if not train_samples:
        raise ValueError("Training split is empty. Reduce val/test fractions or add data.")
    return train_samples, val_samples, test_samples


def prepare_dataset(args: argparse.Namespace) -> None:
    data_root = resolve_project_path(args.data_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_config = {
        "target_size": args.target_size,
        "resize_mode": args.resize_mode,
        "median_ksize": args.median_ksize,
        "apply_clahe": args.apply_clahe,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
        "offline_augmentation": "none",
    }
    with (output_dir / "preprocessing_config.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessing_config, handle, indent=2)

    json_files = find_json_files(data_root, exclude_root=output_dir)
    print(f"Found {len(json_files)} JSON files under {data_root}")
    print(
        "Preprocessing: "
        f"resize_mode={args.resize_mode}, "
        f"target_size={args.target_size}, "
        f"median_ksize={args.median_ksize}, "
        f"apply_clahe={args.apply_clahe}, "
        f"clahe_clip_limit={args.clahe_clip_limit}, "
        f"clahe_tile_grid_size={args.clahe_tile_grid_size}"
    )

    samples: list[PreparedSample] = []
    skipped_rows: list[dict[str, Any]] = []
    for index, json_path in enumerate(json_files, start=1):
        sample, skip_reason = prepare_sample(
            json_path,
            args.label,
            args.target_size,
            args.resize_mode,
            args.median_ksize,
            args.apply_clahe,
            args.clahe_clip_limit,
            args.clahe_tile_grid_size,
        )
        if sample is None:
            skipped_rows.append({"source_json": relpath(json_path), "reason": skip_reason})
        else:
            samples.append(sample)

        if index % args.report_every == 0 or index == len(json_files):
            print(f"Prepared {len(samples)} usable samples, skipped {len(skipped_rows)}, scanned {index}/{len(json_files)}")

    if not samples:
        raise ValueError(f"No usable samples found for label '{args.label}'.")

    train_samples, val_samples, test_samples = split_samples(samples, args.val_fraction, args.test_fraction, args.seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    preview_index = 0
    for sample in train_samples:
        row = save_sample(sample, "train", output_dir, args.preview_count, preview_index)
        train_rows.append(row)
        manifest_rows.append(row)
        preview_index += 1

    for sample in val_samples:
        row = save_sample(sample, "val", output_dir, args.preview_count, preview_index)
        val_rows.append(row)
        manifest_rows.append(row)
        preview_index += 1

    for sample in test_samples:
        row = save_sample(sample, "test", output_dir, args.preview_count, preview_index)
        test_rows.append(row)
        manifest_rows.append(row)
        preview_index += 1

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "val.csv", val_rows)
    write_csv(output_dir / "test.csv", test_rows)
    write_csv(output_dir / "manifest.csv", manifest_rows)
    if skipped_rows:
        write_csv(output_dir / "skipped.csv", skipped_rows)

    print(f"Done. Output directory: {output_dir}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare /data/bongcan LabelMe annotations.")
    parser.add_argument("--data-root", default="data/bongcan")
    parser.add_argument("--output-dir", default="data/bongcan_processed")
    parser.add_argument("--label", default="bc")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--median-ksize", type=int, default=3)
    parser.add_argument("--apply-clahe", action="store_true", help="Apply CLAHE contrast enhancement before resizing.")
    parser.add_argument("--clahe-clip-limit", type=float, default=3.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preview-count", type=int, default=40, help="Use -1 to save previews for all samples.")
    parser.add_argument("--report-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_dataset(parse_args())
