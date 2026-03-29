from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from superdrizzle.align import compute_transforms
from superdrizzle.drizzle import drizzle_combine
from superdrizzle.estimate import estimate_scale
from superdrizzle.io import to_pil
from superdrizzle.load import load


class Pipeline:
    """Drizzle pipeline with eager image loading and lazy alignment.

    Images are loaded and validated on construction. Alignment is deferred
    until .transforms, .scale, .n_aligned, or .combine() is accessed.
    """

    def __init__(
        self,
        images: list[str | Path | np.ndarray | Image.Image],
        ref: int = 0,
    ) -> None:
        self._ref = ref
        self._images: list[np.ndarray] = [load(img) for img in images]
        self._transforms: list[np.ndarray | None] | None = None

        if len(self._images) > 1:
            first_shape = self._images[0].shape
            for i, img in enumerate(self._images[1:], 1):
                if img.shape != first_shape:
                    raise ValueError(
                        f"Image {i} has shape {img.shape}, expected {first_shape}"
                    )

    def add(self, image: str | Path | np.ndarray | Image.Image) -> None:
        """Add an image to the pipeline. Must be called before alignment runs."""
        if self._transforms is not None:
            raise RuntimeError(
                "Cannot add images after alignment has been computed. "
                "Create a new Pipeline instead."
            )
        img = load(image)
        if self._images and img.shape != self._images[0].shape:
            raise ValueError(
                f"Image has shape {img.shape}, expected {self._images[0].shape}"
            )
        self._images.append(img)

    def _ensure_aligned(self) -> None:
        if self._transforms is None:
            self._transforms = compute_transforms(self._images, ref=self._ref)

    @property
    def transforms(self) -> list[np.ndarray | None]:
        self._ensure_aligned()
        return self._transforms

    @property
    def scale(self) -> int:
        return estimate_scale(self.transforms)

    @property
    def n_aligned(self) -> int:
        return sum(1 for t in self.transforms if t is not None)

    def combine(
        self,
        scale: int | None = None,
        pixfrac: float = 0.6,
    ) -> tuple[Image.Image, np.ndarray]:
        self._ensure_aligned()
        transforms = self._transforms

        if scale is None:
            scale = self.scale

        aligned_images = [
            img for img, t in zip(self._images, transforms) if t is not None
        ]
        aligned_transforms = [t for t in transforms if t is not None]

        if not aligned_images:
            raise ValueError("No frames could be aligned")

        result, weights = drizzle_combine(
            aligned_images, aligned_transforms, scale=scale, pixfrac=pixfrac
        )
        return to_pil(result), weights
