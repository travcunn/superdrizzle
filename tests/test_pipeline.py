# tests/test_pipeline.py
import tempfile
import os

import cv2
import numpy as np
from PIL import Image

from superdrizzle.pipeline import Pipeline


def _make_textured_frames(directory: str, n: int = 4) -> list[str]:
    """Create test frames with structure and sub-pixel shifts."""
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


def test_images_loaded_eagerly():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        assert len(pipe._images) == 4


def test_alignment_is_lazy():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        assert pipe._transforms is None


def test_transforms_triggers_alignment():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        transforms = pipe.transforms
        assert len(transforms) == 4
        assert transforms[0] is not None


def test_scale_property():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        scale = pipe.scale
        assert isinstance(scale, int)
        assert scale >= 1


def test_n_aligned_property():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        assert pipe.n_aligned >= 1


def test_combine_returns_pil_and_weights():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        result, weights = pipe.combine(scale=2, pixfrac=0.6)
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (200, 200)


def test_combine_auto_scale():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        result, weights = pipe.combine()
        assert isinstance(result, Image.Image)


def test_add_before_alignment():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d, n=2)
        pipe = Pipeline(paths[:1])
        pipe.add(paths[1])
        assert len(pipe._images) == 2


def test_add_after_alignment_raises():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_textured_frames(d)
        pipe = Pipeline(paths)
        _ = pipe.transforms
        try:
            pipe.add(paths[0])
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


def test_accepts_pil_images():
    pil_imgs = [
        Image.fromarray(np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8))
        for _ in range(3)
    ]
    pipe = Pipeline(pil_imgs)
    assert len(pipe._images) == 3


def test_accepts_numpy_arrays():
    arrays = [np.random.rand(50, 50, 3).astype(np.float32) for _ in range(3)]
    pipe = Pipeline(arrays)
    assert len(pipe._images) == 3
