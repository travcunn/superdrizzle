# Drizzle CLI - Design Spec

A CLI tool that implements the Fruchter & Hook (2002) "Variable-Pixel Linear Reconstruction" (drizzle) algorithm for combining multiple dithered images into a higher-resolution output.

## Use Cases

- Astrophotography: combining dithered telescope frames
- Handheld photo bursts: exploiting natural hand shake for sub-pixel shifts
- Drone/satellite imagery: overlapping aerial frames with slight offsets

## Pipeline

Four stages, each a distinct module:

```
Input Images -> [1. Alignment] -> [2. Scale Estimation] -> [3. Drizzle] -> [4. Output]
```

### 1. Alignment (`align.py`)

For each input frame, estimate a full affine transform (6-parameter: translation, rotation, scale, shear) mapping it onto a reference frame.

- Detect ORB keypoints + descriptors in each frame
- Match against the reference frame using BFMatcher with ratio test
- RANSAC-fit an affine transform from the surviving matches
- Output: list of 2x3 affine matrices, one per frame (reference frame gets identity)

Reference frame is the first image by default, configurable via `--ref`.

### 2. Scale Estimation (`estimate.py`)

Automatically determine the output scale factor from the sub-pixel diversity of the dither pattern.

- For each matched keypoint pair, compute the fractional pixel offset (the sub-pixel residual after integer alignment)
- Bin fractional offsets into an NxN grid (e.g. 2x2, 3x3)
- If the bins are well-covered (e.g. >50% of bins occupied), that scale factor is justified
- Test scale factors 2, 3, 4 in order; pick the highest justified, fall back to 1 if none
- Output: integer scale factor `s`

### 3. Drizzle Kernel (`drizzle.py`)

The Fruchter & Hook core. Implements Equations 2-5 from the paper.

For each input image, for each pixel `(x_i, y_i)`:

1. Transform the pixel center through the affine matrix to get output-grid coordinates
2. Compute the "drop" boundary: a square of size `pixfrac / scale` in output pixel units, centered on the transformed position
3. For each output pixel `(x_o, y_o)` that the drop overlaps, compute the overlap area `a` via rectangle-rectangle intersection:
   `a = max(0, min(r1, r2) - max(l1, l2)) * max(0, min(t1, t2) - max(b1, b2))`
4. Accumulate:
   - `W[x_o, y_o] += a * w_i`
   - `I[x_o, y_o] = (d_i * a * w_i * s^2 + I[x_o, y_o] * W_prev) / W[x_o, y_o]`

Where:
- `w_i` = per-pixel input weight (1.0 for good pixels, 0.0 for saturated)
- `d_i` = pixel value
- `s^2` = surface intensity conservation factor

**Vectorization strategy:** No per-pixel Python loops. For each input image:
- Transform all pixel centers at once via matrix multiplication (numpy)
- Compute all drop boundaries as arrays
- Use integer floor/ceil to find output pixel ranges
- Vectorize the overlap area computation across all input pixels

**Color images:** Each channel (R, G, B) is drizzled independently with the same transforms and weight maps.

**Parameters:**
- `pixfrac` (p): drop size relative to input pixel, range [0.0, 1.0], default 0.6. At 0.0 this is interlacing, at 1.0 this is shift-and-add.
- `scale`: superresolution factor. `--scale 2` means the output is 2x the input resolution (output dimensions = input dimensions * scale). Note: this is the inverse of the paper's `s` parameter, where `s = 1/scale`. The drop size in output pixel units is `pixfrac * scale` (i.e. `pixfrac / s` in the paper's notation).

### 4. Output (`io.py`)

- Read input images as float32 arrays (normalize uint8 0-255 to 0.0-1.0)
- After drizzle: normalize accumulated image by weight map, convert back to uint8, write PNG/JPEG
- Optionally emit weight map as a separate grayscale image (useful for diagnosing coverage gaps)

## CLI Interface

```
drizzle INPUT_IMAGES... -o OUTPUT [-p PIXFRAC] [-s SCALE] [--weights] [--ref REF]
```

| Flag | Description | Default |
|------|-------------|---------|
| `INPUT_IMAGES` | Glob or list of image paths | required |
| `-o / --output` | Output image path | required |
| `-p / --pixfrac` | Drop shrink factor, 0.0-1.0 | 0.6 |
| `-s / --scale` | Force output scale factor (int). Omit for auto. | auto |
| `--weights` | Also emit weight map as `<output>_weights.png` | off |
| `--ref` | Index of reference frame (0-based) | 0 |

Entry point registered via `[project.scripts]` in pyproject.toml.

## Project Structure

```
drizzle/
  pyproject.toml
  src/
    drizzle/
      __init__.py
      cli.py          # argparse entry point
      align.py         # ORB + RANSAC affine estimation
      estimate.py      # auto scale factor from sub-pixel coverage
      drizzle.py       # core kernel (accumulate drops)
      io.py            # read/write images, weight maps
```

## Dependencies

- `opencv-python` - keypoint detection, feature matching, RANSAC
- `numpy` - array math, vectorized drizzle kernel

No other dependencies.

## Error Handling

- **Too few keypoint matches** (< 10 after RANSAC): skip frame, warn to stderr, continue
- **Single input image**: works, just upscales (no sub-pixel benefit). Warn user.
- **No sub-pixel diversity** (all frames integer-aligned): auto-scale picks 1x, warn user
- **Saturated pixels** (value at channel max): set weight to 0.0
- **Mismatched image dimensions**: error and exit

## References

- Fruchter, A. S. & Hook, R. N. (2002). "Drizzle: A Method for the Linear Reconstruction of Undersampled Images." PASP, 114, 144-152. arXiv: astro-ph/9808087
