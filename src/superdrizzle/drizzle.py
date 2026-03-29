from __future__ import annotations

import numpy as np
from tqdm import tqdm


def drizzle_combine(
    images: list[np.ndarray],
    transforms: list[np.ndarray | None],
    scale: int,
    pixfrac: float,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine dithered images using the Fruchter & Hook drizzle algorithm.

    Each input pixel is shrunk into a "drop" (controlled by pixfrac), mapped
    through its affine transform onto a higher-resolution output grid, and
    accumulated with area-weighted contributions. The area-weighted average
    naturally conserves surface intensity (mean pixel value).

    Args:
        images: list of float32 RGB images, shape (H, W, 3), range [0, 1]
        transforms: list of 2x3 affine matrices mapping each frame to the
            reference. None entries are skipped.
        scale: superresolution factor (output is scale * input dimensions)
        pixfrac: drop shrink factor [0.0, 1.0]
        progress: show a tqdm progress bar per frame

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

    pairs = zip(images, transforms)
    if progress:
        pairs = tqdm(list(pairs), desc="Drizzling", unit="frame", file=__import__("sys").stderr)

    for img, M in pairs:
        if M is None:
            continue
        _drizzle_one(img, M, scale, drop_size, numerator, weight_map)

    # Normalize: output = weighted average of input values
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

    Coordinate convention: input pixel (ix, iy) has its center at
    (ix + 0.5, iy + 0.5) in continuous coordinates. After affine transform
    and scaling, the drop center lands at the corresponding position in the
    output grid. This ensures identity transforms produce exact reproduction
    at scale=1 with pixfrac=1.
    """
    h, w, c = img.shape
    out_h, out_w = numerator.shape[:2]

    # Build grid of input pixel centers (center of pixel ix is at ix + 0.5)
    iy, ix = np.mgrid[0:h, 0:w]
    ix_center = ix.ravel().astype(np.float64) + 0.5
    iy_center = iy.ravel().astype(np.float64) + 0.5

    # Apply affine transform: M is 2x3 [a b tx; c d ty]
    # This maps input pixel centers to reference frame pixel centers
    ox = M[0, 0] * ix_center + M[0, 1] * iy_center + M[0, 2]
    oy = M[1, 0] * ix_center + M[1, 1] * iy_center + M[1, 2]

    # Scale to output grid coordinates
    ox = ox * scale
    oy = oy * scale

    # Drop boundaries (centered on transformed position)
    half = drop_size / 2.0
    drop_l = ox - half
    drop_r = ox + half
    drop_b = oy - half
    drop_t = oy + half

    # Per-pixel input weights: 0 for saturated (all channels >= 1.0)
    saturated = np.all(img >= 1.0, axis=2).ravel()
    pixel_weights = np.where(saturated, 0.0, 1.0)

    # Pixel data, shape (N, C)
    pixel_data = img.reshape(-1, c).astype(np.float64)

    # Range of output pixels each drop can touch
    ox_min = np.floor(drop_l).astype(np.int64)
    ox_max = np.floor(drop_r - 1e-10).astype(np.int64)
    oy_min = np.floor(drop_b).astype(np.int64)
    oy_max = np.floor(drop_t - 1e-10).astype(np.int64)

    # Iterate over the small offset grid. A drop can touch at most
    # ceil(drop_size) + 1 output pixels in each direction.
    max_span = int(np.ceil(drop_size)) + 1
    for dy in range(max_span):
        for dx in range(max_span):
            # Output pixel indices for this offset
            out_x = ox_min + dx
            out_y = oy_min + dy

            # Bounds check
            valid = (
                (out_x >= 0)
                & (out_x < out_w)
                & (out_y >= 0)
                & (out_y < out_h)
                & (pixel_weights > 0)
            )

            if not valid.any():
                continue

            # Output pixel boundaries: pixel at index k covers [k, k+1)
            opx_l = out_x.astype(np.float64)
            opx_r = opx_l + 1.0
            opx_b = out_y.astype(np.float64)
            opx_t = opx_b + 1.0

            # Overlap area between drop and output pixel
            overlap_x = np.maximum(
                0.0, np.minimum(drop_r, opx_r) - np.maximum(drop_l, opx_l)
            )
            overlap_y = np.maximum(
                0.0, np.minimum(drop_t, opx_t) - np.maximum(drop_b, opx_b)
            )
            area = overlap_x * overlap_y

            # Mask to valid pixels with nonzero overlap
            mask = valid & (area > 0.0)
            if not mask.any():
                continue

            idx_x = out_x[mask]
            idx_y = out_y[mask]
            a = area[mask]
            pw = pixel_weights[mask]
            d = pixel_data[mask]  # (K, C)

            # Accumulate: area-weighted average preserves surface intensity
            # numerator += value * overlap_area * pixel_weight
            # weight_map += overlap_area * pixel_weight
            contrib_w = a * pw
            contrib_d = d * contrib_w[:, np.newaxis]

            # np.add.at for scatter-add (handles duplicate output indices)
            np.add.at(weight_map, (idx_y, idx_x), contrib_w)
            np.add.at(numerator, (idx_y, idx_x), contrib_d)
