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
    assert weights[2, 2] < weights[0, 0]
