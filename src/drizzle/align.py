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
