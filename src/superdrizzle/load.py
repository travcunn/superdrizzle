from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image

RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
    ".raf", ".srw", ".pef", ".raw", ".3fr", ".ari", ".bay",
    ".cap", ".iiq", ".erf", ".fff", ".mef", ".mos", ".mrw",
    ".nrw", ".ptx", ".r3d", ".rwl", ".rwz", ".x3f",
}


def load(source: "str | Path | np.ndarray | Image.Image") -> np.ndarray:
    """Normalize any supported image source to a float32 RGB numpy array [0, 1].

    Supported inputs:
        - str or pathlib.Path: file path (JPEG/PNG/TIFF + RAW formats)
        - file-like with .read(): binary stream (JPEG/PNG bytes)
        - PIL.Image.Image: converted to RGB numpy
        - np.ndarray uint8: normalized to float32 / 255
        - np.ndarray float32: returned as-is

    RAW formats supported: DNG, CR2, CR3, NEF, ARW, ORF, RW2, RAF, and others.

    Returns:
        float32 ndarray, shape (H, W, 3), range [0, 1]
    """
    if isinstance(source, (str, Path)):
        path = str(source)
        if Path(path).suffix.lower() in RAW_EXTENSIONS:
            return _load_raw(path)
        return _load_path(path)

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


def _load_raw(path: str) -> np.ndarray:
    """Load a RAW file via rawpy/LibRaw. Returns float32 RGB [0, 1]."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
        )
    return rgb.astype(np.float32) / 65535.0


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
