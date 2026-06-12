"""Domain-matched X-Unet pretraining with masked reconstruction.

Input: preprocessed ultrasound image with random patch masking.
Target: original preprocessed ultrasound image.

Use the same --input-mode here and in bongcan segmentation fine-tuning, so the
pretrained checkpoint has matching input-channel shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model import unet_model_v4_attention  # noqa: E402
from src.sector_geometry import INPUT_MODES, input_channels_for_mode, sector_context_channels_tf  # noqa: E402


AUTOTUNE = tf.data.AUTOTUNE


def append_input_context_channels(image: tf.Tensor, target_size: int, input_mode: str) -> tf.Tensor:
    if input_mode == "gray_sector":
        return tf.concat([image] + sector_context_channels_tf(target_size, image.dtype), axis=-1)

    raise ValueError(f"Unsupported input mode: {input_mode}")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_manifest_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["status"] == "ok"].reset_index(drop=True) if "status" in df.columns else df
    df["filename"] = df["filename"].map(
        lambda value: str((ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    )
    return df


def print_preprocessing_config(manifest_path: Path) -> None:
    config_path = manifest_path.parent / "preprocessing_config.json"
    if not config_path.exists():
        print(f"Preprocessing config: not found beside {manifest_path.name}")
        return

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    print(
        "Preprocessing config: "
        f"apply_clahe={config.get('apply_clahe')}, "
        f"clahe_clip_limit={config.get('clahe_clip_limit')}, "
        f"clahe_tile_grid_size={config.get('clahe_tile_grid_size')}, "
        f"median_ksize={config.get('median_ksize')}, "
        f"resize_mode={config.get('resize_mode')}"
    )


def split_dataframe(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_count = max(1, int(round(len(df) * val_fraction)))
    val_df = df.iloc[:val_count].reset_index(drop=True)
    train_df = df.iloc[val_count:].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training split is empty.")
    return train_df, val_df


def load_image(image_path: tf.Tensor, target_size: int) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [target_size, target_size], method="bilinear")
    return tf.cast(image, tf.float32) / 255.0


def apply_patch_mask(
    image: tf.Tensor,
    mask_fraction: float,
    patch_size: int,
    target_size: int,
    input_mode: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    coarse_size = max(target_size // patch_size, 1)
    mask_small = tf.cast(tf.random.uniform([coarse_size, coarse_size, 1]) < mask_fraction, tf.float32)
    mask = tf.image.resize(mask_small, [target_size, target_size], method="nearest")
    masked_image = image * (1.0 - mask)
    masked_features = append_input_context_channels(masked_image, target_size, input_mode)
    return masked_features, image


def build_dataset(
    df: pd.DataFrame,
    target_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    mask_fraction: float,
    patch_size: int,
    input_mode: str,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(df["filename"].tolist())
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(df), 8192), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda path: load_image(path, target_size), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(
        lambda image: apply_patch_mask(image, mask_fraction, patch_size, target_size, input_mode),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def reconstruction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mae + 0.5 * (1.0 - ssim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain X-Unet with masked ultrasound reconstruction.")
    parser.add_argument("--manifest", default="data/ultrasound_unlabeled_processed/manifest.csv")
    parser.add_argument("--output-dir", default="outputs/pretrain_ultrasound_autoencoder_v4_attention_sector_geom45")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--mask-fraction", type=float, default=0.75)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-version", choices=["v4_attention"], default="v4_attention")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="gray_sector",
        help="B baseline input: grayscale ultrasound + sector ray coordinate channels.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    manifest_path = resolve_project_path(args.manifest)
    output_dir = resolve_project_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run src/preprocessing/prepare_ultrasound_unlabeled.py first."
        )

    df = resolve_manifest_paths(pd.read_csv(manifest_path))
    missing = [path for path in df["filename"].tolist() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} preprocessed images. First: {missing[:5]}")
    if args.limit:
        df = df.sample(frac=1.0, random_state=args.seed).head(args.limit).reset_index(drop=True)

    train_df, val_df = split_dataframe(df, args.val_fraction, args.seed)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)

    train_ds = build_dataset(
        train_df,
        args.target_size,
        args.batch_size,
        True,
        args.seed,
        args.mask_fraction,
        args.patch_size,
        args.input_mode,
    )
    val_ds = build_dataset(
        val_df,
        args.target_size,
        args.batch_size,
        False,
        args.seed,
        args.mask_fraction,
        args.patch_size,
        args.input_mode,
    )

    input_channels = input_channels_for_mode(args.input_mode)
    if args.model_version == "v4_attention":
        model = unet_model_v4_attention((args.target_size, args.target_size, input_channels))
    else:
        raise ValueError(f"Unsupported model version: {args.model_version}")
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=reconstruction_loss, metrics=["mae", "mse"])

    first_x, first_y = next(iter(train_ds))
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input mode: {args.input_mode} ({input_channels} channel(s))")
    print_preprocessing_config(manifest_path)
    print(f"Input batch shape: {first_x.shape}")
    print(f"Target batch shape: {first_y.shape}")
    if args.dry_run:
        print("Dry run finished. No training was started.")
        return

    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "xunet_ultrasound_autoencoder_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        ModelCheckpoint(filepath=str(checkpoint_dir / "xunet_ultrasound_autoencoder_latest.keras"), save_best_only=False),
        CSVLogger(str(log_dir / "pretrain_history.csv")),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(output_dir / "xunet_ultrasound_autoencoder_final.keras")
    model.save_weights(output_dir / "xunet_ultrasound_autoencoder_final.weights.h5")


if __name__ == "__main__":
    main()
