import tempfile
import os

import cv2
import numpy as np
from PIL import Image

from superdrizzle.api import drizzle


def _make_textured_frames(directory: str, n: int = 4) -> list[str]:
    rng = np.random.RandomState(42)
    h, w = 100, 100
    base = rng.randint(0, 256, (h + 20, w + 20, 3), dtype=np.uint8)
    for _ in range(30):
        cx, cy = rng.randint(0, w + 20), rng.randint(0, h + 20)
        r = rng.randint(3, 15)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(base, (cx, cy), r, color, -1)
    paths = []
    for i in range(n):
        dx = rng.uniform(0, 1) + i * 0.3
        dy = rng.uniform(0, 1) + i * 0.2
        ix, iy = int(dx), int(dy)
        frame = base[iy : iy + h, ix : ix + w].copy()
        p = os.path.join(directory, f"frame_{i:02d}.png")
        cv2.imwrite(p, frame)
        paths.append(p)
    return paths


def test_drizzle_returns_pil_image():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        result = drizzle(paths)
        assert isinstance(result, Image.Image)


def test_drizzle_with_scale():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        result = drizzle(paths, scale=2)
        assert result.size == (200, 200)


def test_drizzle_accepts_pil_images():
    imgs = [
        Image.fromarray(np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8))
        for _ in range(3)
    ]
    result = drizzle(imgs, scale=1)
    assert isinstance(result, Image.Image)


def test_drizzle_accepts_numpy():
    arrays = [np.random.rand(50, 50, 3).astype(np.float32) for _ in range(3)]
    result = drizzle(arrays, scale=1)
    assert isinstance(result, Image.Image)
