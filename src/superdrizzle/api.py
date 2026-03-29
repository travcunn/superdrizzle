from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from superdrizzle.pipeline import Pipeline


def drizzle(
    images: list[str | Path | np.ndarray | Image.Image],
    scale: int | None = None,
    pixfrac: float = 0.6,
    ref: int = 0,
) -> Image.Image:
    """Combine dithered images into a higher-resolution output.

    This is the simplest way to use superdrizzle. Pass a list of images
    (file paths, PIL Images, numpy arrays, or file objects), get back
    a PIL Image.

    Args:
        images: list of images in any supported format
        scale: output scale factor. If None, auto-estimated from dither pattern.
        pixfrac: drop shrink factor [0.0, 1.0], default 0.6
        ref: index of reference frame, default 0

    Returns:
        PIL Image of the drizzled result
    """
    pipe = Pipeline(images, ref=ref)
    result, _ = pipe.combine(scale=scale, pixfrac=pixfrac)
    return result
