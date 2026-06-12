"""Visualize bongcan test predictions against doctor masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.finetune_bongcan_xunet import (  # noqa: E402
    bce_dice_loss,
    combo_loss,
    dice_coef,
    dice_coef_loss,
    focal_tversky_loss,
    hard_dice_coef,
    hard_iou,
    iou,
    masked_binary_accuracy,
)
from src.sector_geometry import sector_context_features_np  # noqa: E402
from src.training.pretrain_ultrasound_autoencoder import reconstruction_loss  # noqa: E402


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def relpath(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def infer_input_mode(model: tf.keras.Model, requested_mode: str) -> str:
    if requested_mode != "gray_sector":
        raise ValueError(f"Only the B baseline gray_sector input mode is supported, got: {requested_mode}")
    channels = int(model.input_shape[-1])
    if channels != 4:
        raise ValueError(f"B baseline visualization expects a 4-channel gray_sector model, got {channels} channel(s).")
    return requested_mode


def sector_context_features(height: int, width: int) -> list[np.ndarray]:
    return sector_context_features_np(height, width)


def make_input_features(gray: np.ndarray, input_mode: str) -> np.ndarray:
    image = gray.astype(np.float32) / 255.0
    image = image[..., None]
    if input_mode == "gray_sector":
        height, width = gray.shape
        return np.concatenate([image] + sector_context_features(height, width), axis=-1)

    raise ValueError(f"Unsupported input mode: {input_mode}")


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "dice_coef_loss": dice_coef_loss,
            "iou": iou,
            "bce_dice_loss": bce_dice_loss,
            "combo_loss": combo_loss,
            "focal_tversky_loss": focal_tversky_loss,
            "hard_dice_coef": hard_dice_coef,
            "hard_iou": hard_iou,
            "masked_binary_accuracy": masked_binary_accuracy,
            "reconstruction_loss": reconstruction_loss,
        },
        compile=False,
    )


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, panel.shape[1], 18), fill=(0, 0, 0))
    draw.text((5, 3), title, fill=(255, 255, 255))
    return np.asarray(image)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_gray(gray: np.ndarray, display_size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(gray).resize((display_size, display_size), Image.Resampling.BILINEAR))


def resize_bool(mask: np.ndarray, display_size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((display_size, display_size), Image.Resampling.NEAREST)
    ) > 127


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    mask_alpha = mask.astype(np.float32)[..., None] * alpha
    return np.clip(rgb * (1.0 - mask_alpha) + color_arr * mask_alpha, 0, 255).astype(np.uint8)


def apply_color(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    color_arr = np.zeros_like(rgb, dtype=np.float32)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    mask_alpha = mask.astype(np.float32)[..., None] * alpha
    return rgb * (1.0 - mask_alpha) + color_arr * mask_alpha


def draw_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(rgb, contours, -1, color, thickness=thickness)
    return rgb


def segmentation_stats(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    gt = gt & valid
    pred = pred & valid
    inter = gt & pred
    union = gt | pred

    doctor_pixels = int(np.sum(gt))
    model_pixels = int(np.sum(pred))
    intersection_pixels = int(np.sum(inter))
    union_pixels = int(np.sum(union))

    dice = (2.0 * intersection_pixels + 1.0) / (doctor_pixels + model_pixels + 1.0)
    iou_value = (intersection_pixels + 1.0) / (union_pixels + 1.0)
    sensitivity = (intersection_pixels + 1.0) / (doctor_pixels + 1.0)
    precision = (intersection_pixels + 1.0) / (model_pixels + 1.0)

    return {
        "dice": float(dice),
        "iou": float(iou_value),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "doctor_pixels": doctor_pixels,
        "model_pixels": model_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
    }


def postprocess_prediction(
    pred: np.ndarray,
    valid: np.ndarray,
    min_area: int = 0,
    keep_largest: bool = False,
    close_size: int = 0,
) -> np.ndarray:
    pred_uint8 = (pred & valid).astype(np.uint8)

    if min_area > 0 or keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_uint8, connectivity=8)
        kept = np.zeros_like(pred_uint8)
        if num_labels > 1:
            component_ids = list(range(1, num_labels))
            if keep_largest:
                largest = max(component_ids, key=lambda label: stats[label, cv2.CC_STAT_AREA])
                component_ids = [largest]
            for label in component_ids:
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    kept[labels == label] = 1
        pred_uint8 = kept

    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        pred_uint8 = cv2.morphologyEx(pred_uint8, cv2.MORPH_CLOSE, kernel)

    return (pred_uint8 > 0) & valid


def make_overlay_panel(
    gray: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    display_size: int,
) -> np.ndarray:
    gray_big = resize_gray(gray, display_size)
    gt_big = resize_bool(gt & valid, display_size)
    pred_big = resize_bool(pred & valid, display_size)

    overlap = gt_big & pred_big
    gt_only = gt_big & ~pred_big
    pred_only = pred_big & ~gt_big

    rgb = np.stack([gray_big, gray_big, gray_big], axis=-1).astype(np.float32)
    rgb = apply_color(rgb, gt_only, (255, 55, 48), 0.35)
    rgb = apply_color(rgb, pred_only, (65, 105, 255), 0.35)
    rgb = apply_color(rgb, overlap, (190, 70, 255), 0.58)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    thickness = max(2, display_size // 128)
    rgb = draw_contours(rgb, gt_big, (255, 55, 48), thickness)
    rgb = draw_contours(rgb, pred_big, (65, 105, 255), thickness)
    rgb = draw_contours(rgb, overlap, (190, 70, 255), thickness)
    return rgb


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=fill, font=font)


def draw_swatch(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], size: int = 14) -> None:
    draw.rectangle((x, y, x + size, y + size), fill=color)


def make_info_panel(
    row: dict,
    stats: dict[str, float],
    threshold: float,
    panel_height: int,
    panel_width: int,
) -> np.ndarray:
    panel = Image.new("RGB", (panel_width, panel_height), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    title_font = load_font(20, bold=True)
    metric_font = load_font(18, bold=True)
    body_font = load_font(16)
    small_font = load_font(14)

    filename = Path(row["filename"]).stem
    y = 20
    draw_text(draw, (18, y), filename[:32], (220, 220, 220), title_font)
    y += 34
    draw_text(draw, (18, y), f"Threshold: {threshold:.2f}", (180, 180, 180), small_font)
    y += 36

    green = (160, 230, 140)
    draw_text(draw, (18, y), f"IoU  : {stats['iou'] * 100:5.1f}%", green, metric_font)
    y += 26
    draw_text(draw, (18, y), f"Dice : {stats['dice'] * 100:5.1f}%", green, metric_font)
    y += 26
    draw_text(draw, (18, y), f"Sens : {stats['sensitivity'] * 100:5.1f}%", green, metric_font)
    y += 26
    draw_text(draw, (18, y), f"Prec : {stats['precision'] * 100:5.1f}%", green, metric_font)
    y += 48

    rows = [
        ("Bac si", stats["doctor_pixels"], (255, 55, 48)),
        ("Model", stats["model_pixels"], (65, 105, 255)),
        ("Trung", stats["intersection_pixels"], (190, 70, 255)),
        ("Union", stats["union_pixels"], (210, 210, 210)),
    ]
    for label, value, color in rows:
        draw_swatch(draw, 20, y + 4, color)
        draw_text(draw, (44, y), f"{label:<6}: {int(value):,} px", color, body_font)
        y += 28

    legend_y = panel_height - 112
    draw.line((18, legend_y - 18, panel_width - 18, legend_y - 18), fill=(60, 60, 60), width=1)
    legend_items = [
        ("Do", "Bac si", (255, 55, 48)),
        ("Xanh", "Model", (65, 105, 255)),
        ("Tim", "Trung nhau", (190, 70, 255)),
    ]
    for key, label, color in legend_items:
        draw_swatch(draw, 20, legend_y + 4, color)
        draw_text(draw, (44, legend_y), f"{key:<5}= {label}", color, body_font)
        legend_y += 28

    return np.asarray(panel)


def make_comparison_panel(
    row: dict,
    gray: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    stats: dict[str, float],
    threshold: float,
    display_size: int,
    info_width: int,
) -> np.ndarray:
    gray_big = resize_gray(gray, display_size)
    original = np.stack([gray_big, gray_big, gray_big], axis=-1)
    overlay = make_overlay_panel(gray, gt, pred, valid, display_size)
    info = make_info_panel(row, stats, threshold, display_size, info_width)
    divider = np.full((display_size, 4, 3), 35, dtype=np.uint8)
    return np.concatenate([original, divider, overlay, divider, info], axis=1)


def make_error_panel(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) * 0.55
    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)
    rgb[tp] = (0, 220, 80)
    rgb[fp] = (255, 40, 40)
    rgb[fn] = (40, 120, 255)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def hard_metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float]:
    if valid is None:
        valid = np.ones_like(gt, dtype=bool)
    gt_f = (gt & valid).astype(np.float32)
    pred_f = (pred & valid).astype(np.float32)
    inter = float(np.sum(gt_f * pred_f))
    dice = (2.0 * inter + 1.0) / (float(np.sum(gt_f) + np.sum(pred_f)) + 1.0)
    union = float(np.sum(gt_f) + np.sum(pred_f) - inter)
    iou_value = (inter + 1.0) / (union + 1.0)
    return dice, iou_value


def make_grid(panel_paths: list[Path], output_path: Path, columns: int = 2, max_items: int = 12) -> None:
    selected = panel_paths[:max_items]
    if not selected:
        return
    thumbs = []
    for path in selected:
        image = Image.open(path).convert("RGB")
        image.thumbnail((960, 420), Image.Resampling.BILINEAR)
        thumbs.append(image.copy())

    width = max(image.width for image in thumbs)
    height = max(image.height for image in thumbs)
    rows = int(np.ceil(len(thumbs) / columns))
    grid = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    for index, image in enumerate(thumbs):
        x = (index % columns) * width
        y = (index // columns) * height
        grid.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def visualize(args: argparse.Namespace) -> None:
    model_path = resolve_project_path(args.model)
    manifest_path = resolve_project_path(args.manifest)
    output_dir = resolve_project_path(args.output_dir)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    input_mode = infer_input_mode(model, args.input_mode)
    expected_channels = int(model.input_shape[-1])
    actual_channels = make_input_features(np.zeros((args.display_size, args.display_size), dtype=np.uint8), input_mode).shape[-1]
    if actual_channels != expected_channels:
        raise ValueError(
            f"Input mode {input_mode} creates {actual_channels} channel(s), but model expects {expected_channels}."
        )
    print(f"Input mode: {input_mode} ({expected_channels} channel(s))")
    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    images = []
    gt_masks = []
    valid_masks = []
    rows = []
    for row in df.to_dict("records"):
        image_path = resolve_project_path(row["filename"])
        mask_path = resolve_project_path(row["mask"])
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        gt = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
        valid_path_value = row.get("valid_mask", "")
        if valid_path_value and not pd.isna(valid_path_value):
            valid = np.asarray(Image.open(resolve_project_path(valid_path_value)).convert("L"), dtype=np.uint8) > 127
        else:
            valid = np.ones_like(gt, dtype=bool)
        images.append(image)
        gt_masks.append(gt)
        valid_masks.append(valid)
        rows.append(row)

    x = np.stack([make_input_features(image, input_mode) for image in images])
    probabilities = np.squeeze(model.predict(x, batch_size=args.batch_size, verbose=1), axis=-1)

    metric_rows = []
    panel_paths = []
    for row, gray, gt, valid, prob in zip(rows, images, gt_masks, valid_masks, probabilities):
        pred = (prob >= args.threshold) & valid
        pred = postprocess_prediction(
            pred,
            valid,
            min_area=args.min_area,
            keep_largest=args.keep_largest,
            close_size=args.close_size,
        )
        stats = segmentation_stats(gt, pred, valid)

        stem = Path(row["filename"]).stem
        panel_path = panel_dir / f"{stem}.png"

        panel = make_comparison_panel(
            row=row,
            gray=gray,
            gt=gt,
            pred=pred,
            valid=valid,
            stats=stats,
            threshold=args.threshold,
            display_size=args.display_size,
            info_width=args.info_width,
        )
        Image.fromarray(panel, mode="RGB").save(panel_path)
        panel_paths.append(panel_path)

        metric_rows.append(
            {
                "filename": row["filename"],
                "mask": row["mask"],
                "valid_mask": row.get("valid_mask", ""),
                "panel": relpath(panel_path),
                "threshold": args.threshold,
                "input_mode": input_mode,
                "min_area": args.min_area,
                "keep_largest": args.keep_largest,
                "close_size": args.close_size,
                "dice": stats["dice"],
                "iou": stats["iou"],
                "sensitivity": stats["sensitivity"],
                "precision": stats["precision"],
                "doctor_pixels": stats["doctor_pixels"],
                "model_pixels": stats["model_pixels"],
                "intersection_pixels": stats["intersection_pixels"],
                "union_pixels": stats["union_pixels"],
                "gt_positive_ratio": float(np.sum(gt & valid) / (np.sum(valid) + 1e-6)),
                "pred_positive_ratio": float(np.sum(pred & valid) / (np.sum(valid) + 1e-6)),
                "valid_region_ratio": float(np.mean(valid)),
            }
        )

    metrics_path = output_dir / "per_image_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = pd.DataFrame(metric_rows)
    summary[
        [
            "dice",
            "iou",
            "sensitivity",
            "precision",
            "gt_positive_ratio",
            "pred_positive_ratio",
            "valid_region_ratio",
        ]
    ].describe().to_csv(output_dir / "metrics_summary.csv")
    make_grid(panel_paths, output_dir / "test_visualizations_grid.png", columns=args.grid_columns, max_items=args.grid_items)

    print(f"Saved panels: {panel_dir}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Mean hard Dice: {summary['dice'].mean():.4f}")
    print(f"Mean hard IoU: {summary['iou'].mean():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize test predictions against doctor masks.")
    parser.add_argument("--model", default="outputs/bongcan_xunet_v4_attention_sector_autoenc/checkpoints/xunet_bongcan_best.keras")
    parser.add_argument("--manifest", default="data/bongcan_processed_clean/test.csv")
    parser.add_argument("--output-dir", default="outputs/bongcan_xunet_v4_attention_sector_autoenc/test_visualizations")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--input-mode",
        choices=["gray_sector"],
        default="gray_sector",
    )
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--close-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--display-size", type=int, default=768)
    parser.add_argument("--info-width", type=int, default=320)
    parser.add_argument("--grid-items", type=int, default=12)
    parser.add_argument("--grid-columns", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
