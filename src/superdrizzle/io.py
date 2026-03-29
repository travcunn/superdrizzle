"""Image I/O: read/write images, convert between formats."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from superdrizzle.load import load


def read_images(paths: list[str]) -> list[np.ndarray]:
    """Read images from paths, return as list of float32 arrays normalized to [0, 1].

    All images must have the same dimensions. EXIF orientation is applied
    automatically by OpenCV 4.x+.
    """
    images: list[np.ndarray] = []
    first_shape: tuple[int, ...] | None = None

    for p in paths:
        img = load(p)

        if first_shape is None:
            first_shape = img.shape
        elif img.shape != first_shape:
            raise ValueError(
                f"Image dimensions mismatch: {p} is {img.shape}, "
                f"expected {first_shape}"
            )

        images.append(img)

    return images


def to_pil(data: np.ndarray) -> Image.Image:
    """Convert a float32 [0, 1] RGB numpy array to a PIL Image."""
    clipped = np.clip(data, 0.0, 1.0)
    uint8 = (clipped * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def write_image(
    path: str,
    data: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    """Write a float32 [0, 1] image to disk. Optionally write weight map alongside."""
    clipped = np.clip(data, 0.0, 1.0)
    out = (clipped * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)

    if weights is not None:
        p = Path(path)
        weight_path = str(p.with_name(f"{p.stem}_weights{p.suffix}"))
        w_norm = np.clip(weights / (weights.max() + 1e-10), 0.0, 1.0)
        w_out = (w_norm * 255.0).astype(np.uint8)
        cv2.imwrite(weight_path, w_out)
