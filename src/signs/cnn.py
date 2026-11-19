"""Small Keras CNN with in-model preprocessing and augmentation."""

from __future__ import annotations

import keras
from keras import layers

from signs import NUM_CLASSES
from signs.config import CnnConfig


def augmentation() -> keras.Sequential:
    """Camera-like variation only. No horizontal flip: it turns 'Keep left' into 'Keep right'
    and mirrors the digits on speed-limit signs, i.e. it would create wrongly labelled images."""
    return keras.Sequential(
        [
            layers.RandomRotation(0.04),  # about ±15 degrees
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.3, value_range=(0, 1)),
        ],
        name="augment",
    )


def conv_block(x, filters: int):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D()(x)


def build_cnn(cfg: CnnConfig, size: int = 32) -> keras.Model:
    inputs = keras.Input((size, size, 3), dtype="uint8")
    x = layers.Rescaling(1 / 255.0)(inputs)  # raw uint8 images in, so serving needs no preprocessing code
    x = augmentation()(x)
    for filters in (32, 64, 128):  # 32 → 16 → 8 → 4 spatial
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(cfg.dropout / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="sign_cnn")
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
