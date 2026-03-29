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

## CLI

```bash
superdrizzle frame*.jpg -o output.png
```

All frames are automatically aligned (ORB + RANSAC), and the output scale
factor is estimated from the sub-pixel dither coverage.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output image path | required |
| `-s, --scale` | Force output scale factor (e.g. 2, 3) | auto |
| `-p, --pixfrac` | Drop shrink factor (0.0-1.0) | 0.6 |
| `--weights` | Also write a weight map | off |
| `--ref` | Index of reference frame | 0 |

### Examples

```bash
# Auto-detect scale from dither pattern
superdrizzle burst/*.jpg -o result.png

# Force 3x superresolution
superdrizzle burst/*.jpg -o result_3x.png -s 3

# Write weight map to see coverage
superdrizzle burst/*.jpg -o result.png --weights
# -> result.png + result_weights.png
```

## Library

```python
import superdrizzle

# Read images as float32 [0, 1] arrays
images = superdrizzle.read_images(["frame01.jpg", "frame02.jpg", "frame03.jpg"])

# Compute affine transforms (ORB + RANSAC)
transforms = superdrizzle.compute_transforms(images, ref=0)

# Auto-estimate scale from sub-pixel dither coverage
scale = superdrizzle.estimate_scale(transforms)

# Drizzle combine
result, weights = superdrizzle.drizzle_combine(images, transforms, scale=scale, pixfrac=0.6)

# Write output
superdrizzle.write_image("output.png", result, weights=weights)
```

### API

**`read_images(paths) -> list[ndarray]`**
Read images as float32 RGB arrays normalized to [0, 1]. EXIF orientation is
applied automatically. All images must have the same dimensions.

**`compute_transforms(images, ref=0) -> list[ndarray | None]`**
Compute 2x3 affine transforms mapping each frame onto the reference.
Returns `None` for frames that couldn't be aligned.

**`estimate_scale(transforms) -> int`**
Estimate the best integer scale factor (1-4) from sub-pixel dither coverage.

**`drizzle_combine(images, transforms, scale, pixfrac) -> (ndarray, ndarray)`**
Combine images using area-weighted drop accumulation. Returns `(output, weight_map)`.

**`write_image(path, data, weights=None)`**
Write a float32 image to disk. Pass `weights` to also write a weight map.

## How it works

Each input pixel is shrunk into a "drop" (controlled by `pixfrac`), mapped
through its affine transform onto a finer output grid, and accumulated with a
weight proportional to the overlap area. This preserves photometry and
resolution without the blurring introduced by shift-and-add (`pixfrac=1.0`)
or the strict requirements of interlacing (`pixfrac=0.0`).

```
Input Pixels          Output Grid
 ___________          _______________
|     |     |        |   |   |   |   |
|  *--+--*  |  --->  | * | * |   |   |
|  |  |  |  |        |   |   |   |   |
|--+--+--+--|        |---+---+---+---|
|  |  |  |  |        |   | * | * |   |
|  *--+--*  |        |   |   |   |   |
|_____|_____|        |___|___|___|___|

 Shrunken "drops"     Area-weighted
 mapped through        accumulation
 affine transform
```

## Dependencies

- numpy
- opencv-python

## Reference

Fruchter, A. S. & Hook, R. N. (2002). "Drizzle: A Method for the Linear
Reconstruction of Undersampled Images." PASP, 114, 144-152.
[arXiv:astro-ph/9808087](https://arxiv.org/abs/astro-ph/9808087)
