import io
import tempfile
import os

import cv2
import numpy as np
from PIL import Image

from superdrizzle.load import load


def _make_test_image_path(directory: str, h: int = 10, w: int = 10) -> str:
    img = np.full((h, w, 3), (0, 128, 255), dtype=np.uint8)
    p = os.path.join(directory, "test.png")
    cv2.imwrite(p, img)
    return p


def test_load_from_str_path():
    with tempfile.TemporaryDirectory() as d:
        p = _make_test_image_path(d)
        result = load(p)
        assert result.dtype == np.float32
        assert result.shape == (10, 10, 3)
        assert 0.0 <= result.min() and result.max() <= 1.0


def test_load_from_pathlib():
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = _make_test_image_path(d)
        result = load(Path(p))
        assert result.dtype == np.float32
        assert result.shape == (10, 10, 3)


def test_load_from_file_object():
    with tempfile.TemporaryDirectory() as d:
        p = _make_test_image_path(d)
        with open(p, "rb") as f:
            result = load(f)
        assert result.dtype == np.float32
        assert result.shape == (10, 10, 3)


def test_load_from_pil_image():
    pil_img = Image.fromarray(
        np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8), mode="RGB"
    )
    result = load(pil_img)
    assert result.dtype == np.float32
    assert result.shape == (10, 10, 3)
    np.testing.assert_allclose(result[0, 0], [100 / 255, 150 / 255, 200 / 255], atol=0.01)


def test_load_from_numpy_uint8():
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    result = load(arr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], [128 / 255] * 3, atol=0.01)


def test_load_from_numpy_float32():
    arr = np.full((10, 10, 3), 0.5, dtype=np.float32)
    result = load(arr)
    assert result.dtype == np.float32
    assert result is arr  # passthrough, no copy


def test_load_invalid_type_raises():
    try:
        load(12345)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_load_from_bytes_io():
    img = np.full((10, 10, 3), (50, 100, 200), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    bio = io.BytesIO(buf.tobytes())
    result = load(bio)
    assert result.dtype == np.float32
    assert result.shape == (10, 10, 3)


def test_load_bad_path_raises():
    try:
        load("/nonexistent/path/image.jpg")
        assert False
    except FileNotFoundError:
        pass


def test_load_numpy_float64_raises():
    arr = np.ones((10, 10, 3), dtype=np.float64)
    try:
        load(arr)
        assert False
    except TypeError:
        pass


def test_load_rgba_pil_image():
    rgba = Image.fromarray(np.zeros((10, 10, 4), dtype=np.uint8), mode="RGBA")
    result = load(rgba)
    assert result.shape == (10, 10, 3)


def test_load_grayscale_pil_image():
    gray = Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L")
    result = load(gray)
    assert result.shape == (10, 10, 3)


def test_load_corrupted_file_object_raises():
    bio = io.BytesIO(b"not an image at all")
    try:
        load(bio)
        assert False
    except ValueError:
        pass
