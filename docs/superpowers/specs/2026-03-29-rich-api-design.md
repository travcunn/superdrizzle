# Rich API Design Spec

Extend superdrizzle with a three-layer API: a one-liner for the common case, a Pipeline for inspection, and the existing bare functions for full control. Support flexible input types (str, Path, file objects, PIL Images, numpy arrays). Pillow is a required dependency.

## Layer 1: One-liner (`api.py`)

```python
import superdrizzle

result = superdrizzle.drizzle(["frame1.jpg", "frame2.jpg", "frame3.jpg"])
result.save("output.png")  # returns PIL.Image.Image

result = superdrizzle.drizzle(images, scale=3, pixfrac=0.4, ref=0)
```

**Signature:**

```python
def drizzle(
    images: list[str | Path | BinaryIO | Image.Image | np.ndarray],
    scale: int | None = None,
    pixfrac: float = 0.6,
    ref: int = 0,
) -> Image.Image:
```

- Normalizes all inputs via `load()`
- Aligns, estimates scale (if not provided), drizzles, returns PIL Image
- Skips frames that fail alignment, warns to stderr
- Raises `ValueError` if fewer than 1 frame aligns

## Layer 2: Pipeline (`pipeline.py`)

```python
from superdrizzle import Pipeline

pipe = Pipeline(["frame1.jpg", "frame2.jpg", "frame3.jpg"])

pipe.transforms    # list[ndarray | None], triggers alignment on first access
pipe.scale         # int, triggers alignment then estimates
pipe.n_aligned     # int, triggers alignment

result, weights = pipe.combine(scale=2, pixfrac=0.6)
# result: PIL Image, weights: numpy array
```

**Behavior:**

- Images are loaded eagerly on construction (validates they exist, have matching dimensions, catches errors early)
- Alignment is lazy (deferred until `.transforms`, `.scale`, `.n_aligned`, or `.combine()` is accessed)
- Once alignment runs, results are cached
- `pipe.add(img)` adds an image after construction, but only before alignment has run (raises `RuntimeError` if alignment already computed)

**Constructor signature:**

```python
class Pipeline:
    def __init__(
        self,
        images: list[str | Path | BinaryIO | Image.Image | np.ndarray],
        ref: int = 0,
    ) -> None:
```

**`combine()` signature:**

```python
def combine(
    self,
    scale: int | None = None,
    pixfrac: float = 0.6,
) -> tuple[Image.Image, np.ndarray]:
```

- If `scale` is None, uses `self.scale` (auto-estimated)
- Returns `(PIL Image, weight_map numpy array)`

## Layer 3: Bare functions (unchanged)

```python
from superdrizzle import compute_transforms, drizzle_combine, estimate_scale
```

These operate on float32 numpy arrays exactly as before. No changes.

## Input normalization (`load.py`)

```python
from superdrizzle import load

img = load("frame.jpg")          # str -> float32 RGB numpy [0, 1]
img = load(Path("frame.jpg"))    # pathlib.Path
img = load(open("f.jpg", "rb"))  # file-like with .read()
img = load(pil_image)            # PIL.Image.Image
img = load(numpy_uint8)          # uint8 ndarray -> float32 / 255
img = load(numpy_float32)        # float32 ndarray -> passthrough
```

**Type detection logic:**
1. `str` or `Path` -> `cv2.imread` (with EXIF orientation)
2. Has `.read()` method -> read bytes, decode with `cv2.imdecode`
3. `PIL.Image.Image` -> `np.array(img.convert("RGB"))`, normalize
4. `np.ndarray` with dtype uint8 -> `/ 255.0`
5. `np.ndarray` with dtype float32 -> passthrough
6. Otherwise -> `TypeError`

All outputs are float32 RGB (H, W, 3) in range [0, 1].

## Conversion helper (`io.py`)

Add `to_pil(data: np.ndarray) -> Image.Image` to convert float32 [0, 1] numpy back to PIL Image. Used internally by `drizzle()` and `Pipeline.combine()`.

## File changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `Pillow` to dependencies |
| `src/superdrizzle/load.py` | New: `load()` function |
| `src/superdrizzle/pipeline.py` | New: `Pipeline` class |
| `src/superdrizzle/api.py` | New: `drizzle()` one-liner |
| `src/superdrizzle/io.py` | Add `to_pil()`, refactor `read_images()` to use `load()` internally |
| `src/superdrizzle/__init__.py` | Re-export: `drizzle`, `Pipeline`, `load`, `to_pil` + existing exports |
| `README.md` | Update with new API examples |
| `tests/test_load.py` | New: tests for all input types |
| `tests/test_pipeline.py` | New: tests for Pipeline (lazy alignment, combine, add) |
| `tests/test_api.py` | New: tests for `drizzle()` one-liner |
| `tests/test_io.py` | Add test for `to_pil()` |

## Dependencies

- Add `Pillow` to `[project.dependencies]` in pyproject.toml
- Existing: `numpy`, `opencv-python`

## Public API summary

```python
# Layer 1
superdrizzle.drizzle(images, scale=None, pixfrac=0.6, ref=0) -> PIL.Image.Image

# Layer 2
superdrizzle.Pipeline(images, ref=0)
  .transforms -> list[ndarray | None]
  .scale -> int
  .n_aligned -> int
  .add(image) -> None
  .combine(scale=None, pixfrac=0.6) -> (PIL.Image.Image, ndarray)

# Layer 3 (unchanged)
superdrizzle.compute_transforms(images, ref=0) -> list[ndarray | None]
superdrizzle.estimate_scale(transforms) -> int
superdrizzle.drizzle_combine(images, transforms, scale, pixfrac) -> (ndarray, ndarray)

# Utilities
superdrizzle.load(source) -> ndarray
superdrizzle.read_images(paths) -> list[ndarray]
superdrizzle.write_image(path, data, weights=None) -> None
superdrizzle.to_pil(data) -> PIL.Image.Image
```
