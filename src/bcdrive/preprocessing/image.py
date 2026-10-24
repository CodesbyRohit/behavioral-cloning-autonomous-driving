"""The one canonical image pipeline shared by training, evaluation and live driving.

Decoding: every image, whether it is a JPEG file on disk (training) or a base64 JPEG
string from the simulator (inference), is decoded by Pillow and converted to RGB uint8.
Using one decoder for both paths is what guarantees identical tensors.

``preprocess`` then applies, in order:
    1. validation (dtype uint8, 3 channels, optional expected size)
    2. crop (sky / bonnet removal, configurable)
    3. resize to out_height x out_width (Pillow bilinear)
    4. colour conversion (YUV as in the NVIDIA paper, or keep RGB)
    5. normalisation to float32 in [-1, 1]  (x / 127.5 - 1)
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from bcdrive.config import PreprocessConfig


class ImageError(ValueError):
    """An image could not be read, or is not a valid RGB uint8 camera frame."""


def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(arr)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (H, W, 3)."""
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"image file not found: {path}")
    try:
        with Image.open(path) as img:
            img.load()
            return _pil_to_rgb_array(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"corrupt or unreadable image {path}: {exc}") from exc


def decode_image_bytes(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return _pil_to_rgb_array(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"corrupt or undecodable image bytes: {exc}") from exc


def decode_image_base64(data: str) -> np.ndarray:
    """Decode the simulator's base64-encoded JPEG frame to RGB uint8."""
    if not isinstance(data, str) or not data:
        raise ImageError("image payload is empty or not a string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"image payload is not valid base64: {exc}") from exc
    return decode_image_bytes(raw)


def validate_frame(image: np.ndarray, cfg: PreprocessConfig) -> None:
    if not isinstance(image, np.ndarray):
        raise ImageError(f"expected a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ImageError(f"expected dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"expected shape (H, W, 3), got {image.shape}")
    h, w = image.shape[:2]
    if cfg.expected_input_hw is not None and [h, w] != list(cfg.expected_input_hw):
        raise ImageError(
            f"frame is {h}x{w} (HxW) but preprocess.expected_input_hw is {cfg.expected_input_hw}; "
            "the crop was tuned for a different camera layout"
        )
    if cfg.crop_top + cfg.crop_bottom >= h or cfg.crop_left + cfg.crop_right >= w:
        raise ImageError(f"crop {crop_box(cfg, h, w)} leaves no pixels in a {h}x{w} frame")


def crop_box(cfg: PreprocessConfig, height: int, width: int) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) row/column bounds kept by the crop (end exclusive)."""
    return cfg.crop_top, height - cfg.crop_bottom, cfg.crop_left, width - cfg.crop_right


def crop_and_resize(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Steps 1-3: validated, cropped, resized RGB uint8 (out_height, out_width, 3)."""
    validate_frame(image, cfg)
    top, bottom, left, right = crop_box(cfg, *image.shape[:2])
    cropped = image[top:bottom, left:right]
    resized = Image.fromarray(cropped).resize((cfg.out_width, cfg.out_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def to_color_space(rgb: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return rgb
    if color_space == "yuv":
        # Pillow's YCbCr is full-range BT.601, i.e. the YUV variant used for JPEG.
        return np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    raise ImageError(f"unknown color_space {color_space!r}")


def normalize(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 127.5 - 1.0


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """RGB uint8 camera frame -> float32 network input of shape (out_h, out_w, 3)."""
    return normalize(to_color_space(crop_and_resize(image, cfg), cfg.color_space))


def preprocess_batch(images: list[np.ndarray], cfg: PreprocessConfig) -> np.ndarray:
    if not images:
        return np.zeros((0, cfg.out_height, cfg.out_width, 3), dtype=np.float32)
    return np.stack([preprocess(img, cfg) for img in images]).astype(np.float32, copy=False)


def input_shape(cfg: PreprocessConfig) -> tuple[int, int, int]:
    return cfg.out_height, cfg.out_width, 3
