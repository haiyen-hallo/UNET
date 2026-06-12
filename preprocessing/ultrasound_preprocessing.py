"""Shared ultrasound preprocessing utilities.

All current bongcan pipelines should use this module so pretrain, finetune,
and unlabeled prediction see the same image distribution.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_gray(gray: np.ndarray, target_size: int, resize_mode: str, interpolation: int) -> np.ndarray:
    old_height, old_width = gray.shape[:2]

    if resize_mode == "stretch":
        return cv2.resize(gray, (target_size, target_size), interpolation=interpolation)

    if resize_mode == "letterbox":
        scale = min(target_size / old_width, target_size / old_height)
        new_width = int(round(old_width * scale))
        new_height = int(round(old_height * scale))
        resized = cv2.resize(gray, (new_width, new_height), interpolation=interpolation)
        canvas = np.zeros((target_size, target_size), dtype=gray.dtype)
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
        return canvas

    raise ValueError(f"Unsupported resize mode: {resize_mode}")


def valid_region_mask(old_width: int, old_height: int, target_size: int, resize_mode: str) -> np.ndarray:
    """Return 255 inside real resized image content and 0 inside letterbox padding."""
    if resize_mode == "stretch":
        return np.full((target_size, target_size), 255, dtype=np.uint8)

    if resize_mode == "letterbox":
        scale = min(target_size / old_width, target_size / old_height)
        new_width = int(round(old_width * scale))
        new_height = int(round(old_height * scale))
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        mask = np.zeros((target_size, target_size), dtype=np.uint8)
        mask[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = 255
        return mask

    raise ValueError(f"Unsupported resize mode: {resize_mode}")


def transform_points(
    points: np.ndarray,
    old_width: int,
    old_height: int,
    target_size: int,
    resize_mode: str,
) -> np.ndarray:
    points = points.astype(np.float32).copy()

    if resize_mode == "stretch":
        points[:, 0] *= target_size / old_width
        points[:, 1] *= target_size / old_height
    elif resize_mode == "letterbox":
        scale = min(target_size / old_width, target_size / old_height)
        new_width = int(round(old_width * scale))
        new_height = int(round(old_height * scale))
        pad_x = (target_size - new_width) / 2.0
        pad_y = (target_size - new_height) / 2.0
        points[:, 0] = points[:, 0] * scale + pad_x
        points[:, 1] = points[:, 1] * scale + pad_y
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    points[:, 0] = np.clip(points[:, 0], 0, target_size - 1)
    points[:, 1] = np.clip(points[:, 1], 0, target_size - 1)
    return points


def preprocess_ultrasound_rgb(
    image_rgb: np.ndarray,
    target_size: int = 256,
    resize_mode: str = "letterbox",
    median_ksize: int = 3,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if median_ksize > 0:
        if median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd.")
        gray = cv2.medianBlur(gray, ksize=median_ksize)

    if apply_clahe:
        if clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive.")
        if clahe_tile_grid_size <= 0:
            raise ValueError("clahe_tile_grid_size must be positive.")
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        gray = clahe.apply(gray)

    return resize_gray(gray, target_size, resize_mode, interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, target_size: int = 256, resize_mode: str = "letterbox") -> np.ndarray:
    resized = resize_gray(mask, target_size, resize_mode, interpolation=cv2.INTER_NEAREST)
    return ((resized > 127).astype(np.uint8) * 255)


def save_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def make_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
