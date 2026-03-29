import numpy as np
from superdrizzle.estimate import estimate_scale


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
