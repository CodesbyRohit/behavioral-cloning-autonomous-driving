"""Training-only augmentations, applied to the raw RGB frame *before* ``preprocess``.

Steering convention (Udacity simulator): steering is normalised to [-1, 1] and
POSITIVE MEANS TURN RIGHT. Every function returns a new (image, steering) pair so the
effect on the label is explicit and testable. Validation/test data is never augmented.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from bcdrive.config import AugmentConfig


def clip_steering(steering: float) -> float:
    return float(np.clip(steering, -1.0, 1.0))


def flip_horizontal(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    """Mirror left/right. A right turn becomes an equally sharp left turn."""
    return np.ascontiguousarray(image[:, ::-1]), -steering


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale the V channel in HSV space (exposure change; hue is preserved)."""
    hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.uint8)


def shift_horizontal(image: np.ndarray, steering: float, shift_px: int,
                     steering_per_px: float) -> tuple[np.ndarray, float]:
    """Translate the frame sideways by ``shift_px`` (positive = content moves right).

    If the road appears further right than usual, the car is left of the lane centre
    and should steer right, hence ``steering + shift_px * steering_per_px``. This is a
    small-shift approximation of a lateral offset; keep max_shift_px small. Borders
    are filled by replicating the edge column.
    """
    if shift_px == 0:
        return image.copy(), steering
    w = image.shape[1]
    if abs(shift_px) >= w:
        raise ValueError(f"shift of {shift_px}px is not smaller than the frame width {w}")
    out = np.empty_like(image)
    if shift_px > 0:
        out[:, shift_px:] = image[:, :-shift_px]
        out[:, :shift_px] = image[:, :1]
    else:
        s = -shift_px
        out[:, :-s] = image[:, s:]
        out[:, -s:] = image[:, -1:]
    return out, steering + shift_px * steering_per_px


class Augmenter:
    """Applies the configured random augmentations with an explicit RNG."""

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, image: np.ndarray, steering: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
        c = self.cfg
        if c.shift_probability > 0 and c.max_shift_px > 0 and rng.random() < c.shift_probability:
            shift = int(rng.integers(-c.max_shift_px, c.max_shift_px + 1))
            image, steering = shift_horizontal(image, steering, shift, c.steering_per_shift_px)
        if c.brightness_probability > 0 and rng.random() < c.brightness_probability:
            image = adjust_brightness(image, float(rng.uniform(*c.brightness_range)))
        if c.flip_probability > 0 and rng.random() < c.flip_probability:
            image, steering = flip_horizontal(image, steering)
        return image, clip_steering(steering)
