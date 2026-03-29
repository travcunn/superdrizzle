from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load(source: "str | Path | np.ndarray | Image.Image") -> np.ndarray:
    """Normalize any supported image source to a float32 RGB numpy array [0, 1].

    Supported inputs:
        - str or pathlib.Path: file path (EXIF orientation applied)
        - file-like with .read(): binary stream (JPEG/PNG bytes)
        - PIL.Image.Image: converted to RGB numpy
        - np.ndarray uint8: normalized to float32 / 255
        - np.ndarray float32: returned as-is

    Returns:
        float32 ndarray, shape (H, W, 3), range [0, 1]
    """
    if isinstance(source, (str, Path)):
        return _load_path(str(source))

    if hasattr(source, "read"):
        return _load_file_object(source)

    if isinstance(source, Image.Image):
        return _load_pil(source)

    if isinstance(source, np.ndarray):
        return _load_numpy(source)

    raise TypeError(
        f"Unsupported image type: {type(source).__name__}. "
        "Expected str, Path, file object, PIL Image, or numpy array."
    )


def _load_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _load_file_object(f: object) -> np.ndarray:
    data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from file object")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _load_pil(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    return np.array(rgb, dtype=np.float32) / 255.0


def _load_numpy(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    raise TypeError(f"Unsupported numpy dtype: {arr.dtype}. Expected float32 or uint8.")
