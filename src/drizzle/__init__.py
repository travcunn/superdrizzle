"""Drizzle: combine dithered images into higher-resolution output.

Implements the Fruchter & Hook (2002) Variable-Pixel Linear Reconstruction
algorithm for combining multiple dithered, undersampled images.
"""

from drizzle.align import compute_transforms
from drizzle.drizzle import drizzle_combine
from drizzle.estimate import estimate_scale
from drizzle.io import read_images, write_image

__all__ = [
    "compute_transforms",
    "drizzle_combine",
    "estimate_scale",
    "read_images",
    "write_image",
]
