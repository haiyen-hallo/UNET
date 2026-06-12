
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model import (  
    bce_dice_loss as model_bce_dice_loss,
    dice_coef as model_dice_coef,
    dice_coef_loss as model_dice_coef_loss,
    iou as model_iou,
    unet_model_v4_attention,
)
from src.preprocessing.ultrasound_preprocessing import resolve_project_path
from src.sector_geometry import INPUT_MODES, input_channels_for_mode, sector_context_channels_tf
from src.training.pretrain_ultrasound_autoencoder import reconstruction_loss  # noqa: E402


AUTOTUNE = tf.data.AUTOTUNE
TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5
TVERSKY_GAMMA = 0.75
COMBO_BCE_WEIGHT = 0.5


def append_input_context_channels(image: tf.Tensor, target_size: int, input_mode: str) -> tf.Tensor:
    if input_mode == "gray_sector":
        return tf.concat([image] + sector_context_channels_tf(target_size, image.dtype), axis=-1)

    raise ValueError(f"Unsupported input mode: {input_mode}")


def resolve_manifest_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "valid_mask" not in df.columns:
        df["valid_mask"] = ""
    for column in ["filename", "mask", "valid_mask"]:
        df[column] = df[column].map(
            lambda value: ""
            if pd.isna(value) or str(value) == ""
            else str((ROOT / str(value)).resolve())
            if not Path(str(value)).is_absolute()
            else str(Path(str(value)))
        )
    return df


def validate_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    required_columns = {"filename", "mask"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing_columns)}")

    missing_files = []
    for column in ["filename", "mask", "valid_mask"]:
        if column not in df.columns:
            continue
        missing_files.extend(path for path in df[column].tolist() if str(path) and not Path(path).exists())
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(f"Missing {len(missing_files)} files from {manifest_path}. First entries:\n{preview}")


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


def load_pair(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    valid_mask_path: tf.Tensor,
    target_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [target_size, target_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, [target_size, target_size], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)

    def read_valid_mask() -> tf.Tensor:
        valid = tf.io.read_file(valid_mask_path)
        valid = tf.image.decode_image(valid, channels=1, expand_animations=False)
        valid.set_shape([None, None, 1])
        valid = tf.image.resize(valid, [target_size, target_size], method="nearest")
        valid = tf.cast(valid, tf.float32) / 255.0
        return tf.cast(valid > 0.5, tf.float32)

    valid_mask = tf.cond(
        tf.strings.length(valid_mask_path) > 0,
        read_valid_mask,
        lambda: tf.ones_like(mask),
    )

    y_true = tf.concat([mask, valid_mask], axis=-1)
    return image, y_true


def random_horizontal_flip_pair(
    image: tf.Tensor,
    mask: tf.Tensor,
    probability: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    should_flip = tf.random.uniform(()) < probability
    return tf.cond(
        should_flip,
        lambda: (tf.image.flip_left_right(image), tf.image.flip_left_right(mask)),
        lambda: (image, mask),
    )


def augment_pair(
    image: tf.Tensor,
    mask: tf.Tensor,
    hflip_prob: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    image, mask = random_horizontal_flip_pair(image, mask, probability=hflip_prob)

    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)

    gamma = tf.random.uniform((), minval=0.70, maxval=1.30)
    image = tf.pow(tf.clip_by_value(image, 1e-6, 1.0), gamma)

    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.03)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    mask = tf.cast(mask > 0.5, tf.float32)
    return image, mask


def build_labeled_dataset(
    df: pd.DataFrame,
    target_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augment: bool,
    hflip_prob: float,
    input_mode: str,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["filename"].tolist(), df["mask"].tolist(), df["valid_mask"].tolist())
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(df), 4096), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, mask, valid_mask: load_pair(image, mask, valid_mask, target_size),
        num_parallel_calls=AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(
            lambda image, mask: augment_pair(image, mask, hflip_prob),
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.map(
        lambda image, mask: (append_input_context_channels(image, target_size, input_mode), mask),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def split_mask_and_valid(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    if y_true.shape[-1] is not None and y_true.shape[-1] >= 2:
        return y_true[..., :1], y_true[..., 1:2]
    return y_true, tf.ones_like(y_true)


def valid_weighted_mean(values: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
    valid_mask = tf.cast(valid_mask, values.dtype)
    if values.shape.rank is not None and valid_mask.shape.rank is not None and values.shape.rank == valid_mask.shape.rank - 1:
        valid_mask = tf.squeeze(valid_mask, axis=-1)
    return tf.reduce_sum(values * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(mask * y_pred * valid_mask)
    mask_sum = tf.reduce_sum(mask * valid_mask)
    pred_sum = tf.reduce_sum(y_pred * valid_mask)
    return (2.0 * intersection + smooth) / (mask_sum + pred_sum + smooth)


def dice_coef_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


def iou(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(mask * y_pred * valid_mask)
    union = tf.reduce_sum((mask + y_pred - mask * y_pred) * valid_mask)
    return (intersection + smooth) / (union + smooth)


def focal_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    tp = tf.reduce_sum(mask * y_pred * valid_mask)
    fn = tf.reduce_sum(mask * (1.0 - y_pred) * valid_mask)
    fp = tf.reduce_sum((1.0 - mask) * y_pred * valid_mask)
    tversky = (tp + 1e-6) / (tp + TVERSKY_ALPHA * fn + TVERSKY_BETA * fp + 1e-6)
    return tf.pow(1.0 - tversky, TVERSKY_GAMMA)


def combo_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    bce = valid_weighted_mean(tf.keras.backend.binary_crossentropy(mask, y_pred), valid_mask)
    return COMBO_BCE_WEIGHT * bce + (1.0 - COMBO_BCE_WEIGHT) * focal_tversky_loss(y_true, y_pred)


def hard_dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    return dice_coef(y_true, y_pred)


def hard_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    return iou(y_true, y_pred)


def masked_binary_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    correct = tf.cast(tf.equal(tf.cast(y_pred >= 0.5, tf.float32), mask), tf.float32)
    return tf.reduce_sum(correct * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mask, valid_mask = split_mask_and_valid(y_true)
    bce = valid_weighted_mean(tf.keras.backend.binary_crossentropy(mask, y_pred), valid_mask)
    return bce + (1.0 - dice_coef(y_true, y_pred))


def build_segmentation_model(model_version: str, target_size: int, channels: int = 1) -> tf.keras.Model:
    input_shape = (target_size, target_size, channels)
    if model_version == "v4_attention":
        return unet_model_v4_attention(input_shape)
    raise ValueError(f"Unsupported model version: {model_version}")


def model_custom_objects() -> dict[str, Any]:
    return {
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "iou": iou,
        "bce_dice_loss": bce_dice_loss,
        "focal_tversky_loss": focal_tversky_loss,
        "combo_loss": combo_loss,
        "hard_dice_coef": hard_dice_coef,
        "hard_iou": hard_iou,
        "masked_binary_accuracy": masked_binary_accuracy,
        "reconstruction_loss": reconstruction_loss,
        "model_dice_coef": model_dice_coef,
        "model_dice_coef_loss": model_dice_coef_loss,
        "model_bce_dice_loss": model_bce_dice_loss,
        "model_iou": model_iou,
    }


def weighted_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    return [layer for layer in model.layers if layer.get_weights()]


def weight_shapes(layer: tf.keras.layers.Layer) -> list[tuple[int, ...]]:
    return [tuple(weight.shape) for weight in layer.get_weights()]


def checkpoint_has_attention(model: tf.keras.Model) -> bool:
    return any(layer.name == "bottleneck_attention_mha" for layer in model.layers)


def validate_checkpoint_for_finetune(
    checkpoint_model: tf.keras.Model,
    expected_model: tf.keras.Model,
    checkpoint_path: Path,
    model_version: str,
    target_size: int,
) -> None:
    expected_channels = int(expected_model.input_shape[-1])
    checkpoint_channels = int(checkpoint_model.input_shape[-1])
    if checkpoint_channels != expected_channels:
        raise ValueError(
            f"Pretrained checkpoint input channels = {checkpoint_channels}, but fine-tune uses "
            f"{expected_channels} channel(s): {checkpoint_path}\n"
            "Use the B baseline autoencoder checkpoint trained with gray_sector input."
        )

    checkpoint_hw = tuple(checkpoint_model.input_shape[1:3])
    expected_hw = tuple(expected_model.input_shape[1:3])
    if checkpoint_hw != expected_hw or checkpoint_hw != (target_size, target_size):
        raise ValueError(
            f"Pretrained checkpoint input size = {checkpoint_hw}, but fine-tune expects {expected_hw}: {checkpoint_path}"
        )

    if int(checkpoint_model.output_shape[-1]) != int(expected_model.output_shape[-1]):
        raise ValueError(
            f"Pretrained checkpoint output channels = {checkpoint_model.output_shape[-1]}, but fine-tune expects "
            f"{expected_model.output_shape[-1]}: {checkpoint_path}"
        )

    expected_attention = model_version == "v4_attention"
    checkpoint_attention = checkpoint_has_attention(checkpoint_model)
    if checkpoint_attention != expected_attention:
        expected_name = "v4_attention" if expected_attention else model_version
        checkpoint_name = "v4_attention" if checkpoint_attention else "non-attention"
        raise ValueError(
            f"Pretrained checkpoint architecture looks like {checkpoint_name}, but --model-version is {expected_name}: "
            f"{checkpoint_path}"
        )


def transfer_and_verify_pretrained_weights(
    checkpoint_model: tf.keras.Model,
    finetune_model: tf.keras.Model,
    print_layer_report: bool,
) -> None:
    checkpoint_weighted_layers = weighted_layers(checkpoint_model)
    finetune_weighted_layers = weighted_layers(finetune_model)
    if len(checkpoint_weighted_layers) != len(finetune_weighted_layers):
        raise ValueError(
            "Pretrained checkpoint does not have the same number of weighted layers as the fine-tune model: "
            f"{len(checkpoint_weighted_layers)} vs {len(finetune_weighted_layers)}"
        )

    loaded_count = 0
    mismatches: list[str] = []
    report_rows: list[tuple[str, str, bool]] = []
    for checkpoint_layer, finetune_layer in zip(checkpoint_weighted_layers, finetune_weighted_layers):
        checkpoint_shapes = weight_shapes(checkpoint_layer)
        finetune_shapes = weight_shapes(finetune_layer)
        if checkpoint_shapes != finetune_shapes:
            mismatches.append(
                f"{checkpoint_layer.name} -> {finetune_layer.name}: {checkpoint_shapes} != {finetune_shapes}"
            )
            report_rows.append((checkpoint_layer.name, finetune_layer.name, False))
            continue

        checkpoint_weights = checkpoint_layer.get_weights()
        finetune_layer.set_weights(checkpoint_weights)
        loaded = all(
            np.allclose(checkpoint_weight, finetune_weight)
            for checkpoint_weight, finetune_weight in zip(checkpoint_weights, finetune_layer.get_weights())
        )
        loaded_count += int(loaded)
        report_rows.append((checkpoint_layer.name, finetune_layer.name, loaded))

    if mismatches:
        preview = "\n".join(mismatches[:10])
        raise ValueError(f"Pretrained weight shape mismatch. First mismatches:\n{preview}")

    print(
        "Pretrained weight transfer verified: "
        f"{loaded_count}/{len(finetune_weighted_layers)} weighted layers loaded into the fine-tune model."
    )
    if print_layer_report:
        for checkpoint_name, finetune_name, loaded in report_rows:
            print(f"  {checkpoint_name} -> {finetune_name}: loaded={loaded}")


def load_saved_model_checkpoint(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, custom_objects=model_custom_objects(), compile=False)


def load_or_create_model(
    pretrained_model: Path | None,
    target_size: int,
    model_version: str,
    input_mode: str,
    print_pretrained_layer_report: bool,
) -> tf.keras.Model:
    input_channels = input_channels_for_mode(input_mode)
    if pretrained_model is None or not pretrained_model.exists():
        raise FileNotFoundError(
            "B baseline fine-tuning requires the autoencoder pretrained checkpoint. "
            f"Missing checkpoint: {pretrained_model}"
        )

    print(f"Loading and verifying pretrained checkpoint: {pretrained_model}")
    finetune_model = build_segmentation_model(model_version, target_size, channels=input_channels)
    checkpoint_model = load_saved_model_checkpoint(pretrained_model)
    validate_checkpoint_for_finetune(checkpoint_model, finetune_model, pretrained_model, model_version, target_size)
    transfer_and_verify_pretrained_weights(checkpoint_model, finetune_model, print_pretrained_layer_report)
    return finetune_model


def set_encoder_trainable(model: tf.keras.Model, trainable: bool) -> None:
    for layer in model.layers:
        if isinstance(layer, Conv2DTranspose):
            break
        layer.trainable = trainable


def set_all_trainable(model: tf.keras.Model, trainable: bool) -> None:
    for layer in model.layers:
        layer.trainable = trainable


def count_trainable_params(model: tf.keras.Model) -> int:
    return int(np.sum([tf.keras.backend.count_params(weight) for weight in model.trainable_weights]))


def select_loss(name: str):
    if name == "dice":
        return dice_coef_loss
    if name == "bce_dice":
        return bce_dice_loss
    if name == "focal_tversky":
        return focal_tversky_loss
    if name == "combo":
        return combo_loss
    raise ValueError(f"Unsupported loss: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune X-Unet on bongcan segmentation.")
    parser.add_argument("--train-manifest", default="data/bongcan_processed_clean/train.csv")
    parser.add_argument("--val-manifest", default="data/bongcan_processed_clean/val.csv")
    parser.add_argument("--test-manifest", default="data/bongcan_processed_clean/test.csv")
    parser.add_argument("--output-dir", default="outputs/bongcan_xunet_v4_attention_sector_autoenc")
    parser.add_argument(
        "--pretrained-model",
        default="outputs/pretrain_ultrasound_autoencoder_v4_attention_sector_geom45/checkpoints/xunet_ultrasound_autoencoder_best.keras",
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40, help="Total supervised epochs, including frozen warmup epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate after the encoder is unfrozen.")
    parser.add_argument("--warmup-learning-rate", type=float, default=2e-4, help="Learning rate while encoder+bridge are frozen.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--loss", choices=["dice", "bce_dice", "focal_tversky", "combo"], default="combo")
    parser.add_argument("--model-version", choices=["v4_attention"], default="v4_attention")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="gray_sector",
        help="B baseline input: grayscale ultrasound + sector ray coordinate channels.",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=10,
        help="Initial epochs that train only decoder/head when a pretrained checkpoint is loaded.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--monitor-metric", choices=["val_dice_coef", "val_hard_dice_coef", "val_iou", "val_hard_iou"], default="val_hard_iou")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Probability of horizontal flip augmentation on train image+mask pairs.")
    parser.add_argument("--tversky-alpha", type=float, default=0.5, help="False-negative weight in focal Tversky loss.")
    parser.add_argument("--tversky-beta", type=float, default=0.5, help="False-positive weight in focal Tversky loss.")
    parser.add_argument("--tversky-gamma", type=float, default=0.75, help="Focal exponent in focal Tversky loss.")
    parser.add_argument("--combo-bce-weight", type=float, default=0.5, help="BCE share in combo loss; the remainder is focal Tversky.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--no-pretrained-layer-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    global TVERSKY_ALPHA, TVERSKY_BETA, TVERSKY_GAMMA, COMBO_BCE_WEIGHT

    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    if args.tversky_alpha < 0.0 or args.tversky_beta < 0.0 or args.tversky_gamma <= 0.0:
        raise ValueError("Tversky alpha/beta must be non-negative and gamma must be positive.")
    if not 0.0 <= args.combo_bce_weight <= 1.0:
        raise ValueError("combo-bce-weight must be in [0, 1].")
    if not 0.0 <= args.hflip_prob <= 1.0:
        raise ValueError("hflip-prob must be in [0, 1].")

    TVERSKY_ALPHA = args.tversky_alpha
    TVERSKY_BETA = args.tversky_beta
    TVERSKY_GAMMA = args.tversky_gamma
    COMBO_BCE_WEIGHT = args.combo_bce_weight

    train_manifest = resolve_project_path(args.train_manifest)
    val_manifest = resolve_project_path(args.val_manifest)
    test_manifest = resolve_project_path(args.test_manifest)
    output_dir = resolve_project_path(args.output_dir)
    pretrained_model = resolve_project_path(args.pretrained_model) if args.pretrained_model else None
    pretrained_available = pretrained_model is not None and pretrained_model.exists()

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_df = resolve_manifest_paths(pd.read_csv(train_manifest))
    val_df = resolve_manifest_paths(pd.read_csv(val_manifest))
    test_df = resolve_manifest_paths(pd.read_csv(test_manifest))
    validate_manifest(train_df, train_manifest)
    validate_manifest(val_df, val_manifest)
    validate_manifest(test_df, test_manifest)

    train_ds = build_labeled_dataset(
        train_df,
        args.target_size,
        args.batch_size,
        True,
        args.seed,
        augment=not args.no_augment and not args.predict_only,
        hflip_prob=args.hflip_prob,
        input_mode=args.input_mode,
    )
    val_ds = build_labeled_dataset(
        val_df,
        args.target_size,
        args.batch_size,
        False,
        args.seed,
        augment=False,
        hflip_prob=0.0,
        input_mode=args.input_mode,
    )
    test_ds = build_labeled_dataset(
        test_df,
        args.target_size,
        args.batch_size,
        False,
        args.seed,
        augment=False,
        hflip_prob=0.0,
        input_mode=args.input_mode,
    )

    model = load_or_create_model(
        pretrained_model,
        args.target_size,
        args.model_version,
        args.input_mode,
        print_pretrained_layer_report=not args.no_pretrained_layer_report,
    )
    loss = select_loss(args.loss)
    use_frozen_warmup = args.freeze_encoder_epochs > 0 and pretrained_available and not args.predict_only
    if use_frozen_warmup:
        set_encoder_trainable(model, False)
        initial_learning_rate = args.warmup_learning_rate
    else:
        initial_learning_rate = args.learning_rate
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss=loss,
        metrics=[iou, dice_coef, hard_dice_coef, hard_iou, masked_binary_accuracy],
    )

    first_images, first_masks = next(iter(train_ds))
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Input mode: {args.input_mode} ({input_channels_for_mode(args.input_mode)} channel(s))")
    print_preprocessing_config(train_manifest)
    print(f"Freeze encoder warmup epochs: {args.freeze_encoder_epochs if use_frozen_warmup else 0}")
    print(f"Warmup learning rate: {args.warmup_learning_rate if use_frozen_warmup else 'unused'}")
    print(f"Fine-tune learning rate: {args.learning_rate}")
    augment_enabled = not args.no_augment and not args.predict_only
    print(f"Online augmentation: {augment_enabled} (tf.data train-time only)")
    print(f"Horizontal flip probability: {args.hflip_prob if augment_enabled else 0.0}")
    print(f"Intensity augmentation: {'brightness/contrast/gamma/noise' if augment_enabled else 'disabled'}")
    print(
        "Loss settings: "
        f"name={args.loss}, "
        f"tversky_alpha={TVERSKY_ALPHA}, "
        f"tversky_beta={TVERSKY_BETA}, "
        f"tversky_gamma={TVERSKY_GAMMA}, "
        f"combo_bce_weight={COMBO_BCE_WEIGHT}"
    )
    print(f"First batch image shape: {first_images.shape}")
    print(f"First batch y_true shape: {first_masks.shape} (channel 0 = doctor mask, channel 1 = valid letterbox region)")
    print(f"First batch valid-region ratio: {float(tf.reduce_mean(first_masks[..., 1:2])):.4f}")
    if args.dry_run:
        print("Dry run finished. No training or prediction was started.")
        return

    if args.predict_only:
        scores = model.evaluate(test_ds, verbose=1, return_dict=True)
        pd.DataFrame([scores]).to_csv(output_dir / "test_metrics.csv", index=False)
        print(f"Predict-only run finished. Output directory: {output_dir}")
        return

    def make_callbacks(best_name: str, latest_name: str, history_name: str) -> list[tf.keras.callbacks.Callback]:
        return [
            ModelCheckpoint(
                filepath=str(checkpoint_dir / best_name),
                monitor=args.monitor_metric,
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            ModelCheckpoint(filepath=str(checkpoint_dir / latest_name), save_best_only=False, verbose=0),
            CSVLogger(str(log_dir / history_name)),
            ReduceLROnPlateau(monitor=args.monitor_metric, mode="max", factor=0.5, patience=6, min_lr=1e-7, verbose=1),
            EarlyStopping(
                monitor=args.monitor_metric,
                mode="max",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

    initial_epoch = 0
    if use_frozen_warmup:
        warmup_epochs = min(args.freeze_encoder_epochs, args.epochs)
        print(f"Warmup: training decoder with frozen encoder for {warmup_epochs} epochs.")
        print(f"Trainable parameters during warmup: {count_trainable_params(model):,}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=make_callbacks(
                best_name="xunet_bongcan_warmup_best.keras",
                latest_name="xunet_bongcan_warmup_latest.keras",
                history_name="finetune_history_warmup.csv",
            ),
        )
        initial_epoch = warmup_epochs
        set_all_trainable(model, True)
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss=loss,
            metrics=[iou, dice_coef, hard_dice_coef, hard_iou, masked_binary_accuracy],
        )

    if initial_epoch < args.epochs:
        print(f"Fine-tune: training all layers from epoch {initial_epoch + 1} to {args.epochs}.")
        print(f"Trainable parameters after unfreeze: {count_trainable_params(model):,}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            callbacks=make_callbacks(
                best_name="xunet_bongcan_best.keras",
                latest_name="xunet_bongcan_latest.keras",
                history_name="finetune_history.csv",
            ),
        )

    final_model_path = output_dir / "xunet_bongcan_final.keras"
    model.save(final_model_path)
    model.save_weights(output_dir / "xunet_bongcan_final.weights.h5")

    best_model_path = checkpoint_dir / "xunet_bongcan_best.keras"
    warmup_best_model_path = checkpoint_dir / "xunet_bongcan_warmup_best.keras"
    evaluation_model_path = best_model_path if best_model_path.exists() else warmup_best_model_path
    if evaluation_model_path.exists():
        print(f"Loading best validation checkpoint for final test evaluation: {evaluation_model_path}")
        model = load_saved_model_checkpoint(evaluation_model_path)
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss=loss,
            metrics=[iou, dice_coef, hard_dice_coef, hard_iou, masked_binary_accuracy],
        )

    scores = model.evaluate(test_ds, verbose=1, return_dict=True)
    pd.DataFrame([scores]).to_csv(output_dir / "test_metrics.csv", index=False)

    print(f"Done. Output directory: {output_dir}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
