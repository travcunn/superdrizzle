# superdrizzle

Combine multiple dithered images into a higher-resolution output using the
Fruchter & Hook (2002) [Variable-Pixel Linear Reconstruction](https://arxiv.org/abs/astro-ph/9808087)
algorithm.

Works with handheld photo bursts, astrophotography, drone/satellite imagery,
or any set of images with sub-pixel offsets between frames.

## Install

From PyPI:

```bash
pip install superdrizzle
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add superdrizzle
```

From GitHub:

```bash
pip install git+https://github.com/travcunn/superdrizzle.git
# or
uv add git+https://github.com/travcunn/superdrizzle.git
```

For development:

```bash
git clone https://github.com/travcunn/superdrizzle.git
cd superdrizzle
uv sync
```

## Quick Start

```python
import superdrizzle

result = superdrizzle.drizzle(["frame1.jpg", "frame2.jpg", "frame3.jpg"])
result.save("output.png")
```

That's it. Frames are auto-aligned, the scale factor is estimated from the
dither pattern, and you get back a PIL Image.

## CLI

```bash
superdrizzle frame*.jpg -o output.png
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output image path | required |
| `-s, --scale` | Force output scale factor (e.g. 2, 3) | auto |
| `-p, --pixfrac` | Drop shrink factor (0.0-1.0) | 0.6 |
| `--weights` | Also write a weight map | off |
| `--ref` | Index of reference frame | 0 |
| `--no-progress` | Disable progress bar | off |

## API

### One-liner

```python
import superdrizzle

# From file paths
result = superdrizzle.drizzle(["frame1.jpg", "frame2.jpg"])
result.save("output.png")

# From PIL Images
result = superdrizzle.drizzle([pil_img1, pil_img2])
result.save("output.png")

# From numpy arrays
result = superdrizzle.drizzle(numpy_arrays)
result.save("output.png")

# With options
result = superdrizzle.drizzle(images, scale=3, pixfrac=0.4)
result.save("output_3x.png")
```

### Pipeline (when you want intermediate results)

```python
from superdrizzle import Pipeline

pipe = Pipeline(["frame1.jpg", "frame2.jpg", "frame3.jpg"])

pipe.transforms    # list of 2x3 affine matrices
pipe.scale         # auto-estimated scale factor
pipe.n_aligned     # how many frames aligned

result, weights = pipe.combine(scale=2, pixfrac=0.6)
result.save("output.png")  # PIL Image
```

### Bare functions (full control)

```python
from superdrizzle import (
    load, compute_transforms, estimate_scale, drizzle_combine, to_pil
)

images = [load(p) for p in paths]          # flexible input -> float32 numpy
transforms = compute_transforms(images)     # ORB + RANSAC affine
scale = estimate_scale(transforms)          # from sub-pixel coverage
result, weights = drizzle_combine(          # area-weighted drops
    images, transforms, scale=scale, pixfrac=0.6
)
pil_image = to_pil(result)                 # numpy -> PIL
pil_image.save("output.png")
```

### Input types

`superdrizzle.load()` and all high-level functions accept:

- `str` or `pathlib.Path` (file paths)
- File objects with `.read()` (binary streams)
- `PIL.Image.Image`
- `numpy.ndarray` (uint8 or float32)

## How it works

Take a burst of photos. Your hand shakes slightly between each shot, so each
frame captures the scene from a slightly different position. Most of these
shifts are smaller than a single pixel. Drizzle exploits these sub-pixel
shifts: by knowing exactly where each frame lands, it can piece together a
higher-resolution image than any single frame contains. It's the same
technique the Hubble Space Telescope uses to get sharp images from
undersampled detectors.

```mermaid
flowchart LR
    A["Input frames\n(JPEG, PNG, PIL, numpy)"] --> B["Align\nORB + RANSAC\naffine transforms"]
    B --> C["Estimate scale\nanalyze sub-pixel\ndither coverage"]
    C --> D["Drizzle\nshrink pixels into drops,\nmap onto finer grid,\narea-weighted accumulation"]
    D --> E["Output\nnormalize by\nweight map"]
```

**Align** - detect ORB keypoints in each frame, match against a reference
frame, and RANSAC-fit a full affine transform (translation, rotation, scale,
shear).

**Estimate scale** - analyze the sub-pixel fractional offsets across all
frames. If the dither pattern covers a 2x2 grid of sub-pixel phases, 2x
superresolution is justified. If 3x3, then 3x, and so on.

**Drizzle** - the core Fruchter & Hook algorithm. Each input pixel is shrunk
into a smaller "drop" (controlled by `pixfrac`) and mapped through its affine
transform onto the higher-resolution output grid. The contribution to each
output pixel is weighted by the overlap area between the drop and the output
pixel.

```mermaid
flowchart LR
    subgraph Input
        P["Input pixel"]
    end
    subgraph Shrink
        D["Drop\n(pixfrac = 0.6)"]
    end
    subgraph Output ["Output grid (2x finer)"]
        direction TB
        O1["pixel"] --- O2["pixel"]
        O3["pixel"] --- O4["pixel"]
    end
    P -- "shrink by\npixfrac" --> D
    D -- "affine\ntransform" --> Output
```

The drop lands across multiple output pixels. Each output pixel receives a
share of the input value proportional to the overlap area. By shrinking the
pixel before mapping, drizzle avoids reconvolving the image with the
detector's pixel response function, which is what preserves resolution.

The `pixfrac` parameter controls the trade-off:
- `pixfrac=1.0` (shift-and-add) - no shrinking, maximum S/N, but blurs
- `pixfrac=0.0` (interlacing) - point samples, maximum resolution, but
  requires perfectly placed dithers
- `pixfrac=0.6` (default) - a good balance for most real-world data

**Output** - normalize the accumulated image by the weight map and convert to
the requested format.

## Dependencies

- numpy
- opencv-python
- Pillow

## Reference

Fruchter, A. S. & Hook, R. N. (2002). "Drizzle: A Method for the Linear
Reconstruction of Undersampled Images." PASP, 114, 144-152.
[arXiv:astro-ph/9808087](https://arxiv.org/abs/astro-ph/9808087)
