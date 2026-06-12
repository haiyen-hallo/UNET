"""Shared sector geometry for gray_sector model inputs.

These channels are a geometric prior, not a detected ultrasound fan. Keep every
training and evaluation script on the same constants so checkpoints are used
with the same input distribution they were trained with.
"""

from __future__ import annotations

import numpy as np


INPUT_MODES = ("gray_sector",)
DEFAULT_HALF_ANGLE_DEG = 45.0
DEFAULT_APEX_X_SCALE = 0.5
DEFAULT_APEX_Y_SCALE = 0.0


def input_channels_for_mode(input_mode: str) -> int:
    if input_mode == "gray_sector":
        return 4
    raise ValueError(f"Unsupported input mode: {input_mode}")


def sector_maps_np(
    height: int,
    width: int,
    half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG,
    apex_x_scale: float = DEFAULT_APEX_X_SCALE,
    apex_y_scale: float = DEFAULT_APEX_Y_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    apex_x = (width - 1) * apex_x_scale
    apex_y = height * apex_y_scale
    dx = xx - apex_x
    dy = yy - apex_y

    radius = np.sqrt(dx * dx + dy * dy)
    radius_norm = radius / (np.max(radius) + 1e-6)

    half_angle = np.float32(np.deg2rad(half_angle_deg))
    theta = np.arctan2(dx, dy)
    theta_norm = np.clip(theta / half_angle, -1.0, 1.0)
    fan_mask = ((np.abs(theta) <= half_angle) & (dy >= 0.0)).astype(np.float32)
    return radius_norm, theta_norm, fan_mask, theta, (float(apex_x), float(apex_y))


def sector_context_features_np(height: int, width: int) -> list[np.ndarray]:
    radius, theta_norm, fan_mask, _theta, _apex = sector_maps_np(height, width)
    return [radius[..., None], theta_norm[..., None], fan_mask[..., None]]


def sector_context_channels_tf(target_size: int, dtype):
    import tensorflow as tf

    coords = tf.linspace(0.0, tf.cast(target_size - 1, tf.float32), target_size)
    yy = tf.tile(coords[:, tf.newaxis], [1, target_size])
    xx = tf.tile(coords[tf.newaxis, :], [target_size, 1])

    apex_x = tf.cast(target_size - 1, tf.float32) * DEFAULT_APEX_X_SCALE
    apex_y = tf.cast(target_size, tf.float32) * DEFAULT_APEX_Y_SCALE
    dx = xx - apex_x
    dy = yy - apex_y
    radius = tf.sqrt(dx * dx + dy * dy)
    radius = radius / (tf.reduce_max(radius) + 1e-6)

    half_angle = tf.constant(DEFAULT_HALF_ANGLE_DEG * np.pi / 180.0, dtype=tf.float32)
    theta = tf.atan2(dx, dy)
    theta_norm = tf.clip_by_value(theta / half_angle, -1.0, 1.0)
    fan_mask = tf.cast((tf.abs(theta) <= half_angle) & (dy >= 0.0), tf.float32)

    return [
        tf.cast(radius[..., tf.newaxis], dtype),
        tf.cast(theta_norm[..., tf.newaxis], dtype),
        tf.cast(fan_mask[..., tf.newaxis], dtype),
    ]
