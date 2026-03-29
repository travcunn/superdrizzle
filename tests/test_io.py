import numpy as np
import cv2
import tempfile
import os
from superdrizzle.io import read_images, write_image


def _make_test_image(path: str, h: int, w: int, color: tuple[int, int, int]) -> None:
    """Write a solid-color test image to disk."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_read_images_returns_float32_normalized():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "img.png")
        _make_test_image(p, 10, 10, (0, 128, 255))
        images = read_images([p])
        assert len(images) == 1
        assert images[0].dtype == np.float32
        assert images[0].shape == (10, 10, 3)
        assert images[0].min() >= 0.0
        assert images[0].max() <= 1.0


def test_read_images_multiple():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(3):
            p = os.path.join(d, f"img{i}.png")
            _make_test_image(p, 8, 8, (i * 80, i * 80, i * 80))
            paths.append(p)
        images = read_images(paths)
        assert len(images) == 3


def test_read_images_mismatched_dimensions_raises():
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.png")
        p2 = os.path.join(d, "b.png")
        _make_test_image(p1, 10, 10, (0, 0, 0))
        _make_test_image(p2, 20, 20, (0, 0, 0))
        try:
            read_images([p1, p2])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "dimensions" in str(e).lower()


def test_write_image_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        data = np.random.rand(10, 10, 3).astype(np.float32)
        out = os.path.join(d, "out.png")
        write_image(out, data)
        assert os.path.exists(out)
        reloaded = cv2.imread(out)
        assert reloaded is not None
        assert reloaded.shape == (10, 10, 3)


def test_write_image_weight_map():
    with tempfile.TemporaryDirectory() as d:
        data = np.random.rand(10, 10, 3).astype(np.float32)
        weights = np.random.rand(10, 10).astype(np.float32)
        out = os.path.join(d, "out.png")
        write_image(out, data, weights=weights)
        weight_path = os.path.join(d, "out_weights.png")
        assert os.path.exists(weight_path)


from superdrizzle.io import to_pil
from PIL import Image


def test_to_pil_returns_pil_image():
    data = np.random.rand(10, 10, 3).astype(np.float32)
    result = to_pil(data)
    assert isinstance(result, Image.Image)
    assert result.size == (10, 10)
    assert result.mode == "RGB"


def test_to_pil_clamps_values():
    data = np.array([[[1.5, -0.1, 0.5]]], dtype=np.float32)
    result = to_pil(data)
    px = result.getpixel((0, 0))
    assert px[0] == 255  # clamped from 1.5
    assert px[1] == 0    # clamped from -0.1
    assert px[2] == 127  # 0.5 * 255 = 127.5, truncated to 127
