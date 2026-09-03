"""The ``npr_genimage_v1`` preprocessing profile (Pillow + numpy, no Torch).

Follows the official NPR README GenImage test guidance: translate-and-duplicate
undersized images instead of resizing, deterministic center crop to 224, no
test-time flip, ImageNet normalization.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

__all__ = [
    "CROP_SIZE",
    "PROFILE_NAME",
    "center_crop",
    "preprocess_npr_genimage_v1",
    "translate_duplicate",
]

PROFILE_NAME = "npr_genimage_v1"
CROP_SIZE = 224

# ImageNet normalization values used by the official NPR data pipeline.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def translate_duplicate(img: Image.Image, crop_size: int = CROP_SIZE) -> Image.Image:
    """Tile the image so both sides are at least ``crop_size``.

    Mirrors the official README's ``translate_duplicate``: the original image is
    pasted at integer offsets; it is never resized.
    """
    width, height = img.size
    if min(width, height) >= crop_size:
        return img
    new_width = width * math.ceil(crop_size / width)
    new_height = height * math.ceil(crop_size / height)
    canvas = Image.new("RGB", (new_width, new_height))
    for i in range(0, new_width, width):
        for j in range(0, new_height, height):
            canvas.paste(img, (i, j))
    return canvas


def center_crop(img: Image.Image, crop_size: int) -> Image.Image:
    """Deterministic center crop, matching torchvision's rounding behavior."""
    width, height = img.size
    left = int(round((width - crop_size) / 2.0))
    top = int(round((height - crop_size) / 2.0))
    return img.crop((left, top, left + crop_size, top + crop_size))


def preprocess_npr_genimage_v1(img: Image.Image) -> np.ndarray:
    """Convert a PIL image into a normalized CHW float32 array of shape (3, 224, 224)."""
    rgb = img.convert("RGB")
    tiled = translate_duplicate(rgb)
    cropped = center_crop(tiled, CROP_SIZE)

    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
