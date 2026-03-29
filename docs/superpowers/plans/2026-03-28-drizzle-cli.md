# Drizzle CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that combines multiple dithered images into a higher-resolution output using the Fruchter & Hook drizzle algorithm.

**Architecture:** Four-module pipeline: alignment (ORB + RANSAC affine), scale estimation (sub-pixel coverage analysis), drizzle kernel (area-weighted drop accumulation), and I/O. All numpy-vectorized, no per-pixel Python loops.

**Tech Stack:** Python, numpy, opencv-python, argparse, uv + pyproject.toml

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, CLI entry point |
| `src/drizzle/__init__.py` | Package init (empty) |
| `src/drizzle/io.py` | Read images to float32 arrays, write output + weight maps |
| `src/drizzle/align.py` | ORB keypoint detection, BFMatcher, RANSAC affine estimation |
| `src/drizzle/estimate.py` | Auto-determine scale factor from sub-pixel dither coverage |
| `src/drizzle/drizzle.py` | Core drizzle kernel: drop mapping + weighted accumulation |
| `src/drizzle/cli.py` | argparse entry point, pipeline orchestration |
| `tests/test_io.py` | Tests for image read/write |
| `tests/test_align.py` | Tests for alignment |
| `tests/test_estimate.py` | Tests for scale estimation |
| `tests/test_drizzle.py` | Tests for drizzle kernel |
| `tests/test_cli.py` | Integration test for full pipeline |

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/drizzle/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "drizzle"
version = "0.1.0"
description = "Combine dithered images into higher-resolution output using the Fruchter & Hook drizzle algorithm"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "opencv-python",
]

[project.scripts]
drizzle = "drizzle.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/drizzle"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# src/drizzle/__init__.py
```

Empty file.

- [ ] **Step 3: Install in dev mode**

Run: `cd /Users/tcunningham/drizzle && uv sync`
Expected: dependencies installed, package editable-installed

- [ ] **Step 4: Verify import works**

Run: `cd /Users/tcunningham/drizzle && uv run python -c "import drizzle; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/drizzle/__init__.py
git commit -m "feat: project scaffold with pyproject.toml"
```

---

### Task 2: Image I/O (`io.py`)

**Files:**
- Create: `src/drizzle/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for read_images and write_image**

```python
# tests/test_io.py
import numpy as np
import cv2
import tempfile
import os
from drizzle.io import read_images, write_image


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement io.py**

```python
# src/drizzle/io.py
import sys
from pathlib import Path

import cv2
import numpy as np


def read_images(paths: list[str]) -> list[np.ndarray]:
    """Read images from paths, return as list of float32 arrays normalized to [0, 1].

    All images must have the same dimensions. Images are converted to RGB
    (OpenCV reads as BGR).
    """
    images: list[np.ndarray] = []
    first_shape: tuple[int, ...] | None = None

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if first_shape is None:
            first_shape = img.shape
        elif img.shape != first_shape:
            raise ValueError(
                f"Image dimensions mismatch: {p} is {img.shape}, "
                f"expected {first_shape}"
            )

        images.append(img.astype(np.float32) / 255.0)

    return images


def write_image(
    path: str,
    data: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    """Write a float32 [0, 1] image to disk. Optionally write weight map alongside."""
    clipped = np.clip(data, 0.0, 1.0)
    out = (clipped * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)

    if weights is not None:
        p = Path(path)
        weight_path = str(p.with_name(f"{p.stem}_weights{p.suffix}"))
        w_norm = np.clip(weights / (weights.max() + 1e-10), 0.0, 1.0)
        w_out = (w_norm * 255.0).astype(np.uint8)
        cv2.imwrite(weight_path, w_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_io.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/drizzle/io.py tests/test_io.py
git commit -m "feat: image I/O with float32 normalization and weight map output"
```

---

### Task 3: Alignment (`align.py`)

**Files:**
- Create: `src/drizzle/align.py`
- Create: `tests/test_align.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_align.py
import numpy as np
import cv2
from drizzle.align import compute_transforms


def _make_textured_image(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Create a textured image with features that ORB can detect."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    # Add some structure: circles and rectangles
    for _ in range(20):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(5, 30)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


def _apply_affine(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Shift an image by (tx, ty) pixels."""
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_identity_for_reference_frame():
    img = _make_textured_image(200, 200)
    frames = [img.astype(np.float32) / 255.0] * 2
    transforms = compute_transforms(frames, ref=0)
    assert len(transforms) == 2
    np.testing.assert_allclose(transforms[0], np.eye(2, 3), atol=1e-6)


def test_detects_translation():
    img = _make_textured_image(200, 200)
    shifted = _apply_affine(img, 5.5, -3.2)
    frames = [
        img.astype(np.float32) / 255.0,
        shifted.astype(np.float32) / 255.0,
    ]
    transforms = compute_transforms(frames, ref=0)
    # The transform for frame 1 should approximately recover the shift
    tx_recovered = transforms[1][0, 2]
    ty_recovered = transforms[1][1, 2]
    assert abs(tx_recovered - (-5.5)) < 2.0, f"tx={tx_recovered}, expected ~-5.5"
    assert abs(ty_recovered - (3.2)) < 2.0, f"ty={ty_recovered}, expected ~3.2"


def test_too_few_matches_returns_none():
    """A blank image should yield too few matches."""
    good = _make_textured_image(200, 200)
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    frames = [
        good.astype(np.float32) / 255.0,
        blank.astype(np.float32) / 255.0,
    ]
    transforms = compute_transforms(frames, ref=0)
    assert transforms[0] is not None  # reference is always identity
    assert transforms[1] is None  # blank has no features
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_align.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement align.py**

```python
# src/drizzle/align.py
import sys

import cv2
import numpy as np

MIN_MATCHES = 10


def compute_transforms(
    images: list[np.ndarray],
    ref: int = 0,
) -> list[np.ndarray | None]:
    """Compute affine transforms mapping each frame onto the reference frame.

    Args:
        images: list of float32 RGB images, shape (H, W, 3), range [0, 1]
        ref: index of the reference frame

    Returns:
        List of 2x3 affine matrices (np.float64). Identity for the reference
        frame. None for frames where alignment failed.
    """
    n = len(images)
    transforms: list[np.ndarray | None] = [None] * n
    transforms[ref] = np.eye(2, 3, dtype=np.float64)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    ref_gray = _to_gray_uint8(images[ref])
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)

    if desc_ref is None:
        return transforms

    for i in range(n):
        if i == ref:
            continue

        gray = _to_gray_uint8(images[i])
        kp_i, desc_i = orb.detectAndCompute(gray, None)

        if desc_i is None or len(kp_i) < MIN_MATCHES:
            print(f"Warning: frame {i} has too few features, skipping", file=sys.stderr)
            continue

        matches = bf.knnMatch(desc_i, desc_ref, k=2)

        # Ratio test
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n_match = pair
                if m.distance < 0.75 * n_match.distance:
                    good.append(m)

        if len(good) < MIN_MATCHES:
            print(
                f"Warning: frame {i} has only {len(good)} matches (need {MIN_MATCHES}), skipping",
                file=sys.stderr,
            )
            continue

        pts_i = np.array([kp_i[m.queryIdx].pt for m in good], dtype=np.float64)
        pts_ref = np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float64)

        # Estimate affine (maps frame i coords -> ref coords)
        M, inliers = cv2.estimateAffine2D(pts_i, pts_ref, method=cv2.RANSAC)

        if M is None or (inliers is not None and inliers.sum() < MIN_MATCHES):
            print(f"Warning: RANSAC failed for frame {i}, skipping", file=sys.stderr)
            continue

        transforms[i] = M

    return transforms


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] to uint8 grayscale."""
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return (gray * 255.0).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_align.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/drizzle/align.py tests/test_align.py
git commit -m "feat: ORB + RANSAC affine alignment"
```

---

### Task 4: Scale Estimation (`estimate.py`)

**Files:**
- Create: `src/drizzle/estimate.py`
- Create: `tests/test_estimate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_estimate.py
import numpy as np
from drizzle.estimate import estimate_scale


def test_half_pixel_dithers_gives_2x():
    # Four frames with half-pixel offsets in both axes -> 2x2 coverage -> scale 2
    transforms = [
        np.array([[1, 0, 0.0], [0, 1, 0.0]]),    # (0.0, 0.0)
        np.array([[1, 0, 0.5], [0, 1, 0.0]]),    # (0.5, 0.0)
        np.array([[1, 0, 0.0], [0, 1, 0.5]]),    # (0.0, 0.5)
        np.array([[1, 0, 0.5], [0, 1, 0.5]]),    # (0.5, 0.5)
    ]
    scale = estimate_scale(transforms)
    assert scale == 2


def test_integer_shifts_gives_1x():
    # All shifts are integer -> no sub-pixel diversity -> scale 1
    transforms = [
        np.array([[1, 0, 0.0], [0, 1, 0.0]]),
        np.array([[1, 0, 3.0], [0, 1, 5.0]]),
        np.array([[1, 0, 7.0], [0, 1, 2.0]]),
    ]
    scale = estimate_scale(transforms)
    assert scale == 1


def test_third_pixel_dithers_gives_3x():
    offsets = [(i / 3.0, j / 3.0) for i in range(3) for j in range(3)]
    transforms = [
        np.array([[1, 0, dx], [0, 1, dy]]) for dx, dy in offsets
    ]
    scale = estimate_scale(transforms)
    assert scale == 3


def test_single_frame_gives_1x():
    transforms = [np.eye(2, 3)]
    scale = estimate_scale(transforms)
    assert scale == 1


def test_none_transforms_are_skipped():
    transforms = [
        np.array([[1, 0, 0.0], [0, 1, 0.0]]),
        None,
        np.array([[1, 0, 0.5], [0, 1, 0.5]]),
    ]
    scale = estimate_scale(transforms)
    # Only 2 frames, partial coverage of 2x2 grid -> scale 1
    assert scale == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_estimate.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement estimate.py**

```python
# src/drizzle/estimate.py
import sys

import numpy as np

COVERAGE_THRESHOLD = 0.5


def estimate_scale(transforms: list[np.ndarray | None]) -> int:
    """Estimate the best output scale factor from sub-pixel dither coverage.

    Extracts the translation components from each affine transform, computes
    fractional pixel offsets, bins them into NxN grids for candidate scale
    factors, and picks the highest scale with sufficient coverage.

    Args:
        transforms: list of 2x3 affine matrices (None entries are skipped)

    Returns:
        Integer scale factor (1, 2, 3, or 4)
    """
    offsets = []
    for t in transforms:
        if t is None:
            continue
        tx, ty = t[0, 2], t[1, 2]
        offsets.append((tx, ty))

    if len(offsets) < 2:
        return 1

    best_scale = 1

    for candidate in [4, 3, 2]:
        # Bin fractional offsets into a candidate x candidate grid
        bins = set()
        for tx, ty in offsets:
            fx = tx % 1.0  # fractional part
            fy = ty % 1.0
            bx = int(fx * candidate) % candidate
            by = int(fy * candidate) % candidate
            bins.add((bx, by))

        coverage = len(bins) / (candidate * candidate)
        if coverage >= COVERAGE_THRESHOLD:
            best_scale = candidate
            break

    if best_scale == 1:
        print(
            "Warning: insufficient sub-pixel dither diversity, using scale=1",
            file=sys.stderr,
        )

    return best_scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_estimate.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/drizzle/estimate.py tests/test_estimate.py
git commit -m "feat: auto scale estimation from sub-pixel dither coverage"
```

---

### Task 5: Drizzle Kernel (`drizzle.py`)

**Files:**
- Create: `src/drizzle/drizzle.py`
- Create: `tests/test_drizzle.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_drizzle.py
import numpy as np
from drizzle.drizzle import drizzle_combine


def test_single_frame_identity_pixfrac_1():
    """Single frame with identity transform and pixfrac=1.0 (shift-and-add).
    At scale=1, the output should equal the input."""
    h, w = 4, 4
    img = np.random.rand(h, w, 3).astype(np.float32)
    transforms = [np.eye(2, 3, dtype=np.float64)]
    result, weights = drizzle_combine([img], transforms, scale=1, pixfrac=1.0)
    assert result.shape == (h, w, 3)
    np.testing.assert_allclose(result, img, atol=0.02)


def test_scale_2_doubles_dimensions():
    h, w = 4, 4
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    transforms = [np.eye(2, 3, dtype=np.float64)]
    result, weights = drizzle_combine([img], transforms, scale=2, pixfrac=0.6)
    assert result.shape == (h * 2, w * 2, 3)


def test_uniform_input_gives_uniform_output():
    """Two identical frames at identity should produce uniform output."""
    h, w = 8, 8
    img = np.ones((h, w, 3), dtype=np.float32) * 0.7
    transforms = [np.eye(2, 3, dtype=np.float64)] * 2
    result, weights = drizzle_combine([img, img], transforms, scale=1, pixfrac=1.0)
    np.testing.assert_allclose(result, 0.7, atol=0.02)


def test_flux_conservation():
    """Total flux in the output should be scale^2 times the input flux
    (surface intensity is conserved, but there are more pixels)."""
    h, w = 8, 8
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    transforms = [np.eye(2, 3, dtype=np.float64)]
    scale = 2
    result, weights = drizzle_combine([img], transforms, scale=scale, pixfrac=1.0)
    input_mean = img.mean()
    output_mean = result[weights > 0].mean() if (weights > 0).any() else 0.0
    # Surface intensity should be conserved (mean value ~ same)
    assert abs(output_mean - input_mean) < 0.05, (
        f"output_mean={output_mean}, input_mean={input_mean}"
    )


def test_weight_map_zero_where_no_data():
    """Pixels outside the footprint of all frames should have zero weight."""
    h, w = 4, 4
    img = np.ones((h, w, 3), dtype=np.float32)
    # Shift the image way off to the right so the left side has no data
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    transforms = [M]
    scale = 2
    result, weights = drizzle_combine([img], transforms, scale=scale, pixfrac=0.6)
    # The leftmost columns should have zero weight
    assert weights[:, 0].sum() == 0.0


def test_saturated_pixels_get_zero_weight():
    """Pixels at 1.0 (saturated) should be excluded."""
    h, w = 4, 4
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    img[2, 2, :] = 1.0  # saturated pixel
    transforms = [np.eye(2, 3, dtype=np.float64)]
    result, weights = drizzle_combine([img], transforms, scale=1, pixfrac=1.0)
    # The saturated pixel location should have lower weight
    # (zero contribution from that pixel, but edges of neighboring drops
    # might still contribute)
    assert weights[2, 2] < weights[0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_drizzle.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement drizzle.py**

```python
# src/drizzle/drizzle.py
import numpy as np


def drizzle_combine(
    images: list[np.ndarray],
    transforms: list[np.ndarray | None],
    scale: int,
    pixfrac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine dithered images using the Fruchter & Hook drizzle algorithm.

    Args:
        images: list of float32 RGB images, shape (H, W, 3), range [0, 1]
        transforms: list of 2x3 affine matrices mapping each frame to the
            reference. None entries are skipped.
        scale: superresolution factor (output is scale * input dimensions)
        pixfrac: drop shrink factor [0.0, 1.0]

    Returns:
        (output_image, weight_map) where output_image is float32 (H*scale, W*scale, 3)
        and weight_map is float32 (H*scale, W*scale).
    """
    h, w, c = images[0].shape
    out_h, out_w = h * scale, w * scale

    # Accumulated weighted data per channel and total weight
    numerator = np.zeros((out_h, out_w, c), dtype=np.float64)
    weight_map = np.zeros((out_h, out_w), dtype=np.float64)

    # Drop size in output pixel units
    drop_size = pixfrac * scale

    for img, M in zip(images, transforms):
        if M is None:
            continue
        _drizzle_one(img, M, scale, drop_size, numerator, weight_map)

    # Normalize
    safe_weight = np.where(weight_map > 0, weight_map, 1.0)
    output = (numerator / safe_weight[:, :, np.newaxis]).astype(np.float32)
    output = np.where(weight_map[:, :, np.newaxis] > 0, output, 0.0)

    return output, weight_map.astype(np.float32)


def _drizzle_one(
    img: np.ndarray,
    M: np.ndarray,
    scale: int,
    drop_size: float,
    numerator: np.ndarray,
    weight_map: np.ndarray,
) -> None:
    """Drizzle a single input image onto the output accumulation arrays.

    Vectorized: transforms all pixel centers at once, then iterates over
    the (small) grid of output pixels each drop can touch.
    """
    h, w, c = img.shape
    out_h, out_w = numerator.shape[:2]

    # Build grid of input pixel centers
    iy, ix = np.mgrid[0:h, 0:w]  # shape (H, W) each
    ix_flat = ix.ravel().astype(np.float64)
    iy_flat = iy.ravel().astype(np.float64)

    # Apply affine transform to get output coordinates
    # M is 2x3: [a b tx; c d ty]
    ox = M[0, 0] * ix_flat + M[0, 1] * iy_flat + M[0, 2]
    oy = M[1, 0] * ix_flat + M[1, 1] * iy_flat + M[1, 2]

    # Scale to output grid
    ox = ox * scale
    oy = oy * scale

    # Drop boundaries (centered on transformed position)
    half = drop_size / 2.0
    drop_l = ox - half
    drop_r = ox + half
    drop_b = oy - half
    drop_t = oy + half

    # Per-pixel input weights: 0 for saturated (all channels == 1.0)
    saturated = np.all(img >= 1.0, axis=2).ravel()
    pixel_weights = np.where(saturated, 0.0, 1.0)

    # Pixel data, shape (N, C)
    pixel_data = img.reshape(-1, c).astype(np.float64)

    # Range of output pixels each drop can touch
    ox_min = np.floor(drop_l).astype(np.int64)
    ox_max = np.floor(drop_r - 1e-10).astype(np.int64)
    oy_min = np.floor(drop_b).astype(np.int64)
    oy_max = np.floor(drop_t - 1e-10).astype(np.int64)

    # Surface intensity conservation factor
    s2 = float(scale * scale)

    # Iterate over the small offset grid (drop can touch at most
    # ceil(drop_size)+1 output pixels in each direction)
    max_span = int(np.ceil(drop_size)) + 1
    for dy in range(max_span):
        for dx in range(max_span):
            # Output pixel indices for this offset
            out_x = ox_min + dx
            out_y = oy_min + dy

            # Bounds check
            valid = (
                (out_x >= 0) & (out_x < out_w) &
                (out_y >= 0) & (out_y < out_h) &
                (pixel_weights > 0)
            )

            if not valid.any():
                continue

            # Output pixel boundaries
            opx_l = out_x.astype(np.float64)
            opx_r = opx_l + 1.0
            opx_b = out_y.astype(np.float64)
            opx_t = opx_b + 1.0

            # Overlap area
            overlap_x = np.maximum(0.0, np.minimum(drop_r, opx_r) - np.maximum(drop_l, opx_l))
            overlap_y = np.maximum(0.0, np.minimum(drop_t, opx_t) - np.maximum(drop_b, opx_b))
            area = overlap_x * overlap_y

            # Mask to only valid pixels with nonzero overlap
            mask = valid & (area > 0.0)
            if not mask.any():
                continue

            idx_x = out_x[mask]
            idx_y = out_y[mask]
            a = area[mask]
            w = pixel_weights[mask]
            d = pixel_data[mask]  # (K, C)

            # Accumulate: numerator += d * a * w * s^2, weight_map += a * w
            contrib_w = a * w
            contrib_d = d * (contrib_w * s2)[:, np.newaxis]

            # Use np.add.at for scatter-add (handles duplicate indices)
            np.add.at(weight_map, (idx_y, idx_x), contrib_w)
            np.add.at(numerator, (idx_y, idx_x), contrib_d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_drizzle.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/drizzle/drizzle.py tests/test_drizzle.py
git commit -m "feat: vectorized drizzle kernel with area-weighted drop accumulation"
```

---

### Task 6: CLI Entry Point (`cli.py`)

**Files:**
- Create: `src/drizzle/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_cli.py
import subprocess
import tempfile
import os

import cv2
import numpy as np


def _make_test_scene(directory: str, n_frames: int = 4) -> list[str]:
    """Create test frames: a textured image with sub-pixel shifts."""
    rng = np.random.RandomState(42)
    h, w = 100, 100
    base = rng.randint(0, 256, (h + 20, w + 20, 3), dtype=np.uint8)
    # Add structure
    for _ in range(30):
        cx, cy = rng.randint(0, w + 20), rng.randint(0, h + 20)
        r = rng.randint(3, 15)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(base, (cx, cy), r, color, -1)

    paths = []
    for i in range(n_frames):
        # Sub-pixel shifts
        dx = rng.uniform(0, 1) + i * 0.3
        dy = rng.uniform(0, 1) + i * 0.2
        ix, iy = int(dx), int(dy)
        frame = base[iy:iy + h, ix:ix + w].copy()
        p = os.path.join(directory, f"frame_{i:02d}.png")
        cv2.imwrite(p, frame)
        paths.append(p)

    return paths


def test_cli_produces_output():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_test_scene(d)
        out = os.path.join(d, "result.png")
        result = subprocess.run(
            ["uv", "run", "drizzle"] + paths + ["-o", out, "-s", "2", "--weights"],
            capture_output=True,
            text=True,
            cwd="/Users/tcunningham/drizzle",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(out), "Output image not created"
        img = cv2.imread(out)
        assert img is not None
        assert img.shape[0] == 200  # 100 * scale=2
        assert img.shape[1] == 200
        # Weight map should exist
        weight_path = os.path.join(d, "result_weights.png")
        assert os.path.exists(weight_path)


def test_cli_auto_scale():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_test_scene(d)
        out = os.path.join(d, "result.png")
        result = subprocess.run(
            ["uv", "run", "drizzle"] + paths + ["-o", out],
            capture_output=True,
            text=True,
            cwd="/Users/tcunningham/drizzle",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_cli.py -v`
Expected: FAIL (drizzle command not found or ImportError)

- [ ] **Step 3: Implement cli.py**

```python
# src/drizzle/cli.py
import argparse
import sys

from drizzle.align import compute_transforms
from drizzle.drizzle import drizzle_combine
from drizzle.estimate import estimate_scale
from drizzle.io import read_images, write_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine dithered images into a higher-resolution output using drizzle."
    )
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument(
        "-p", "--pixfrac", type=float, default=0.6,
        help="Drop shrink factor (0.0-1.0, default: 0.6)",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=None,
        help="Output scale factor (default: auto-estimate)",
    )
    parser.add_argument(
        "--weights", action="store_true",
        help="Emit weight map alongside output",
    )
    parser.add_argument(
        "--ref", type=int, default=0,
        help="Index of reference frame (default: 0)",
    )
    args = parser.parse_args()

    # 1. Read
    print(f"Reading {len(args.images)} images...", file=sys.stderr)
    images = read_images(args.images)
    print(f"  Image size: {images[0].shape[1]}x{images[0].shape[0]}", file=sys.stderr)

    # 2. Align
    print("Aligning frames...", file=sys.stderr)
    transforms = compute_transforms(images, ref=args.ref)
    n_aligned = sum(1 for t in transforms if t is not None)
    print(f"  {n_aligned}/{len(images)} frames aligned", file=sys.stderr)

    if n_aligned == 0:
        print("Error: no frames could be aligned", file=sys.stderr)
        sys.exit(1)

    # 3. Estimate scale
    if args.scale is not None:
        scale = args.scale
        print(f"Using user-specified scale: {scale}x", file=sys.stderr)
    else:
        scale = estimate_scale(transforms)
        print(f"Auto-estimated scale: {scale}x", file=sys.stderr)

    # 4. Drizzle
    print(f"Drizzling (pixfrac={args.pixfrac}, scale={scale}x)...", file=sys.stderr)
    aligned_images = [img for img, t in zip(images, transforms) if t is not None]
    aligned_transforms = [t for t in transforms if t is not None]
    result, weights = drizzle_combine(
        aligned_images, aligned_transforms, scale=scale, pixfrac=args.pixfrac,
    )

    # 5. Write
    print(f"Writing output to {args.output}", file=sys.stderr)
    write_image(args.output, result, weights=weights if args.weights else None)

    out_h, out_w = result.shape[:2]
    print(f"Done. Output size: {out_w}x{out_h}", file=sys.stderr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/test_cli.py -v`
Expected: all 2 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/tcunningham/drizzle && uv run pytest tests/ -v`
Expected: all tests PASS (16 total across all test files)

- [ ] **Step 6: Commit**

```bash
git add src/drizzle/cli.py tests/test_cli.py
git commit -m "feat: CLI entry point with full pipeline orchestration"
```

---

### Self-Review Checklist

**Spec coverage:**
- Alignment (ORB + RANSAC affine) -> Task 3
- Scale estimation (sub-pixel coverage) -> Task 4
- Drizzle kernel (area-weighted drops) -> Task 5
- I/O (float32, weight maps) -> Task 2
- CLI interface (all flags) -> Task 6
- Error handling (too few matches, single image, no diversity, saturated, mismatched dims) -> distributed across Tasks 2-5
- Color channels processed independently -> Task 5 (per-channel in numerator array)
- Scale convention (our scale = 1/paper's s) -> Task 5 implementation

**Placeholder scan:** No TBDs, TODOs, or vague steps. All code blocks are complete.

**Type consistency:** `compute_transforms` returns `list[np.ndarray | None]`, consumed correctly in cli.py and estimate.py. `drizzle_combine` signature matches usage in cli.py. `read_images`/`write_image` signatures consistent throughout.
