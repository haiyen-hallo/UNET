import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, SpatialDropout2D, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU, Add, LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def dice_coef(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred, smooth=1.):
    bce = K.mean(tf.keras.backend.binary_crossentropy(y_true, y_pred))
    dice = (2. * K.sum(y_true * y_pred) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return bce + (1.0 - dice)


# Evaluation metrics: iou
def iou(y_true, y_pred, smooth = 1.):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true) + K.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def conv_block_v4(x, filters, l2_reg=1e-5, dropout=0.0):
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout > 0:
        x = SpatialDropout2D(dropout)(x)
    return x


def spatial_self_attention_block_v4(x, num_heads=4, mlp_ratio=2, dropout=0.1, name="bottleneck_attention"):
    """Apply self-attention across bottleneck spatial patches."""
    height = x.shape[1]
    width = x.shape[2]
    channels = x.shape[3]
    if height is None or width is None or channels is None:
        raise ValueError("spatial_self_attention_block_v4 requires static H, W, and C dimensions.")

    token_count = int(height) * int(width)
    channels = int(channels)
    key_dim = max(channels // num_heads, 1)

    tokens = Reshape((token_count, channels), name=f"{name}_flatten")(x)
    norm_tokens = LayerNormalization(epsilon=1e-6, name=f"{name}_ln_1")(tokens)
    attended = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f"{name}_mha",
    )(norm_tokens, norm_tokens)
    tokens = Add(name=f"{name}_attn_residual")([tokens, attended])

    mlp = LayerNormalization(epsilon=1e-6, name=f"{name}_ln_2")(tokens)
    mlp = Dense(channels * mlp_ratio, activation="relu", name=f"{name}_mlp_dense_1")(mlp)
    mlp = Dropout(dropout, name=f"{name}_mlp_dropout")(mlp)
    mlp = Dense(channels, name=f"{name}_mlp_dense_2")(mlp)
    tokens = Add(name=f"{name}_mlp_residual")([tokens, mlp])

    return Reshape((int(height), int(width), channels), name=f"{name}_unflatten")(tokens)


def unet_model_v4(
    input_shape,
    base_filters=32,
    l2_reg=1e-5,
    skip_dropout_rates=(0.20, 0.15, 0.10, 0.05),
    use_attention=False,
):
    """Smaller regularized U-Net for limited labeled ultrasound data."""
    inp = Input(input_shape)
    skip4_rate, skip3_rate, skip2_rate, skip1_rate = skip_dropout_rates

    conv1 = conv_block_v4(inp, base_filters, l2_reg, dropout=0.0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block_v4(pool1, base_filters * 2, l2_reg, dropout=0.0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block_v4(pool2, base_filters * 4, l2_reg, dropout=0.1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block_v4(pool3, base_filters * 8, l2_reg, dropout=0.2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block_v4(pool4, base_filters * 16, l2_reg, dropout=0.3)
    if use_attention:
        conv5 = spatial_self_attention_block_v4(conv5, name="bottleneck_attention")

    up1 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat_1 = concatenate([up1, Dropout(skip4_rate, name="skip_dropout_4")(conv4)], axis=3)
    conv6 = conv_block_v4(concat_1, base_filters * 8, l2_reg, dropout=0.2)

    up2 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat_2 = concatenate([up2, Dropout(skip3_rate, name="skip_dropout_3")(conv3)], axis=3)
    conv7 = conv_block_v4(concat_2, base_filters * 4, l2_reg, dropout=0.15)

    up3 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat_3 = concatenate([up3, Dropout(skip2_rate, name="skip_dropout_2")(conv2)], axis=3)
    conv8 = conv_block_v4(concat_3, base_filters * 2, l2_reg, dropout=0.1)

    up4 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat_4 = concatenate([up4, Dropout(skip1_rate, name="skip_dropout_1")(conv1)], axis=3)
    conv9 = conv_block_v4(concat_4, base_filters, l2_reg, dropout=0.0)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return Model(inputs=[inp], outputs=[conv10])


def unet_model_v4_attention(input_shape, base_filters=32, l2_reg=1e-5):
    return unet_model_v4(
        input_shape,
        base_filters=base_filters,
        l2_reg=l2_reg,
        use_attention=True,
    )
    