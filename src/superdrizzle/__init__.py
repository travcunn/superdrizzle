"""superdrizzle: combine dithered images into higher-resolution output.

Implements the Fruchter & Hook (2002) Variable-Pixel Linear Reconstruction
algorithm for combining multiple dithered, undersampled images.

Quick start:
    import superdrizzle
    result = superdrizzle.drizzle(["frame1.jpg", "frame2.jpg"])
    result.save("output.png")
"""

from superdrizzle.align import compute_transforms
from superdrizzle.api import drizzle
from superdrizzle.drizzle import drizzle_combine
from superdrizzle.estimate import estimate_scale
from superdrizzle.io import read_images, to_pil, write_image
from superdrizzle.load import load
from superdrizzle.pipeline import Pipeline

__all__ = [
    "drizzle",
    "Pipeline",
    "load",
    "to_pil",
    "compute_transforms",
    "drizzle_combine",
    "estimate_scale",
    "read_images",
    "write_image",
]
