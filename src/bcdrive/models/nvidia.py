"""NVIDIA "PilotNet"-inspired CNN (Bojarski et al., 2016, arXiv:1604.07316, Fig. 4).

Layer table from the paper (input 66x200x3 YUV):
    conv 5x5 stride 2 -> 24 maps   (31x98)
    conv 5x5 stride 2 -> 36 maps   (14x47)
    conv 5x5 stride 2 -> 48 maps   (5x22)
    conv 3x3 stride 1 -> 64 maps   (3x20)
    conv 3x3 stride 1 -> 64 maps   (1x18)
    flatten (1152) -> FC 100 -> FC 50 -> FC 10 -> 1 output
That is ~252k trainable weights; the paper reports "250 thousand parameters".

Differences from the paper, deliberately:
    * The paper normalises inside the network with a fixed layer; here normalisation is
      done once in ``preprocess`` (shared with inference), so the network has none.
    * The paper does not state its activation; ELU is used (ReLU is configurable).
    * Dropout before the dense layers, and He initialisation, are additions.
    * The output predicts the simulator's normalised steering in [-1, 1], not 1/r.
So this is "inspired by", not an exact reproduction of, NVIDIA's implementation.
"""

from __future__ import annotations

import keras
from keras import layers

from bcdrive.config import ModelConfig


def _reg(cfg: ModelConfig):
    return keras.regularizers.L2(cfg.l2) if cfg.l2 > 0 else None


def build_nvidia_model(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    act, init = cfg.activation, "he_normal"
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for i, (filters, kernel, stride) in enumerate(
        [(24, 5, 2), (36, 5, 2), (48, 5, 2), (64, 3, 1), (64, 3, 1)], start=1
    ):
        x = layers.Conv2D(filters, kernel, strides=stride, activation=act, kernel_initializer=init,
                          kernel_regularizer=_reg(cfg), name=f"conv{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    for units in (100, 50, 10):
        if cfg.dropout > 0:
            x = layers.Dropout(cfg.dropout, name=f"dropout_fc{units}")(x)
        x = layers.Dense(units, activation=act, kernel_initializer=init,
                         kernel_regularizer=_reg(cfg), name=f"fc{units}")(x)
    outputs = layers.Dense(1, name="steering")(x)
    return keras.Model(inputs, outputs, name="pilotnet_inspired")


def build_tiny_model(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    """SMOKE-TEST FALLBACK ONLY: a ~10k-parameter CNN for fast CPU pipeline tests."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Conv2D(8, 5, strides=3, activation=cfg.activation, name="conv1")(inputs)
    x = layers.Conv2D(16, 5, strides=3, activation=cfg.activation, name="conv2")(x)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dense(16, activation=cfg.activation, name="fc16")(x)
    outputs = layers.Dense(1, name="steering")(x)
    return keras.Model(inputs, outputs, name="tiny_smoke_test_model")


def build_model(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    if cfg.architecture == "nvidia":
        return build_nvidia_model(input_shape, cfg)
    if cfg.architecture == "tiny":
        return build_tiny_model(input_shape, cfg)
    raise ValueError(f"unknown architecture {cfg.architecture!r}")


def model_summary_text(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lambda s, **_: lines.append(s))
    return "\n".join(lines)


def trainable_params(model: keras.Model) -> int:
    return int(sum(keras.ops.size(w) for w in model.trainable_weights))
