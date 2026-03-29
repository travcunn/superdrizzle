"""Drizzle: combine dithered images into higher-resolution output.

Implements the Fruchter & Hook (2002) Variable-Pixel Linear Reconstruction
algorithm for combining multiple dithered, undersampled images.
"""

from superdrizzle.align import compute_transforms
from superdrizzle.api import drizzle
from superdrizzle.drizzle import drizzle_combine
from superdrizzle.estimate import estimate_scale
from superdrizzle.io import read_images, write_image

__all__ = [
    "compute_transforms",
    "drizzle",
    "drizzle_combine",
    "estimate_scale",
    "read_images",
    "write_image",
]
