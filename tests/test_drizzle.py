import numpy as np
from superdrizzle.drizzle import drizzle_combine


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
    assert weights[2, 2] < weights[0, 0]


def test_subpixel_resolution_improvement():
    """Drizzling two half-pixel-shifted frames at 2x should recover high-res detail
    better than either single frame upscaled alone.

    Strategy: create a known 2x checkerboard pattern, downsample it into two
    frames with a 0.5-pixel offset, drizzle them back at 2x, and verify the
    reconstruction is closer to the original than naive upscaling of either frame.
    """
    # Build a known high-res pattern (alternating 0.3 and 0.7 columns)
    out_h, out_w = 16, 16
    hires = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for x in range(out_w):
        hires[:, x, :] = 0.3 if x % 2 == 0 else 0.7

    # Downsample to two "input" frames at half resolution with 0.5px offset.
    # Frame 0: sample at output pixels (0,0), (0,2), (0,4)... -> input (0,0),(0,1),(0,2)...
    # Frame 1: sample at output pixels (0,1), (0,3), (0,5)... -> shifted by 0.5 input pixels
    in_h, in_w = out_h // 2, out_w // 2
    frame0 = np.zeros((in_h, in_w, 3), dtype=np.float32)
    frame1 = np.zeros((in_h, in_w, 3), dtype=np.float32)
    for iy in range(in_h):
        for ix in range(in_w):
            # Frame 0 samples even columns of hires
            frame0[iy, ix] = hires[iy * 2, ix * 2]
            # Frame 1 samples odd columns of hires
            frame1[iy, ix] = hires[iy * 2, ix * 2 + 1]

    # Transforms: frame0 is identity, frame1 is shifted by 0.5 input pixels in x
    t0 = np.eye(2, 3, dtype=np.float64)
    t1 = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float64)

    result, weights = drizzle_combine([frame0, frame1], [t0, t1], scale=2, pixfrac=0.6)
    assert result.shape == (in_h * 2, in_w * 2, 3)

    # Where we have weight, the drizzled result should reconstruct the alternating
    # pattern better than a single frame upscaled (which would be flat).
    # Measure: standard deviation of column means in the drizzled output should be
    # higher than for a single frame naively upscaled.
    mask = weights > 0
    if mask.any():
        # Drizzled column variation
        col_means_drizzle = [
            result[:, x, 0][mask[:, x]].mean()
            for x in range(result.shape[1])
            if mask[:, x].any()
        ]
        std_drizzle = np.std(col_means_drizzle)

        # Single frame upscaled: just repeat frame0 pixels
        upscaled = np.repeat(np.repeat(frame0, 2, axis=0), 2, axis=1)
        col_means_single = [upscaled[:, x, 0].mean() for x in range(upscaled.shape[1])]
        std_single = np.std(col_means_single)

        # The drizzled output should have MORE variation (resolved the pattern)
        assert std_drizzle > std_single, (
            f"Drizzle std={std_drizzle:.4f} should exceed single-frame std={std_single:.4f}"
        )


def test_pixfrac_zero_gives_zero_weights():
    """With pixfrac=0.0, drop area is zero, so nothing should accumulate."""
    h, w = 8, 8
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    transforms = [np.eye(2, 3, dtype=np.float64)]
    result, weights = drizzle_combine([img], transforms, scale=2, pixfrac=0.0)
    assert weights.sum() == 0.0
    # Output should be all zeros since nothing was accumulated
    assert result.sum() == 0.0


def test_all_none_transforms():
    """When all transforms are None, output should be all zeros with zero weights."""
    h, w = 8, 8
    img = np.ones((h, w, 3), dtype=np.float32) * 0.5
    transforms = [None, None, None]
    result, weights = drizzle_combine([img, img, img], transforms, scale=2, pixfrac=0.6)
    assert result.shape == (h * 2, w * 2, 3)
    assert weights.sum() == 0.0
    assert result.sum() == 0.0


def test_multiple_frames_with_subpixel_shifts():
    """Drizzle multiple frames with known sub-pixel shifts at 2x and verify
    correct dimensions and non-zero weight coverage."""
    h, w = 16, 16
    rng = np.random.RandomState(99)
    frames = []
    transforms = []
    for dx, dy in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]:
        frame = rng.rand(h, w, 3).astype(np.float32) * 0.9  # avoid saturation
        frames.append(frame)
        transforms.append(np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64))

    result, weights = drizzle_combine(frames, transforms, scale=2, pixfrac=0.6)
    assert result.shape == (h * 2, w * 2, 3)
    assert weights.shape == (h * 2, w * 2)
    # Most of the interior should have non-zero weight from multiple frames
    interior = weights[4:-4, 4:-4]
    coverage = (interior > 0).sum() / interior.size
    assert coverage > 0.8, f"Expected >80% interior coverage, got {coverage:.1%}"
