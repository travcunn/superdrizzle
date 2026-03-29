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
    best_coverage = 0.0

    for candidate in [2, 3, 4]:
        # Bin fractional offsets into a candidate x candidate grid
        bins = set()
        for tx, ty in offsets:
            fx = tx % 1.0  # fractional part
            fy = ty % 1.0
            bx = int(fx * candidate) % candidate
            by = int(fy * candidate) % candidate
            bins.add((bx, by))

        coverage = len(bins) / (candidate * candidate)
        if coverage > COVERAGE_THRESHOLD and coverage >= best_coverage:
            best_scale = candidate
            best_coverage = coverage

    if best_scale == 1:
        print(
            "Warning: insufficient sub-pixel dither diversity, using scale=1",
            file=sys.stderr,
        )

    return best_scale
