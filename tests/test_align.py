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
