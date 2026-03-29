#!/usr/bin/env python3
"""Benchmark superdrizzle on real photos.

Since there's no ground truth for real bursts, this script:
1. Runs drizzle on all frames
2. Compares against single-frame bicubic upscale
3. Saves side-by-side comparison crops at multiple locations
4. Reports sharpness metrics (gradient magnitude as a proxy for resolution)

Usage:
    uv run python benchmarks/benchmark_real.py images/fridge/*.jpg
    uv run python benchmarks/benchmark_real.py images/fridge/*.jpg --scale 2 --crops 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from superdrizzle import (
    compute_transforms, drizzle_combine, estimate_scale, read_images
)

OUTPUT_DIR = Path(__file__).parent / "output" / "real"


def measure_sharpness(img: np.ndarray) -> float:
    """Measure image sharpness via mean gradient magnitude (Sobel)."""
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gx = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def save_comparison_crops(
    drizzled: np.ndarray,
    single: np.ndarray,
    shift_add: np.ndarray,
    n_crops: int,
    seed: int = 42,
) -> None:
    """Save side-by-side comparison crops at random locations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h, w = drizzled.shape[:2]
    crop_size = min(256, h // 3, w // 3)
    rng = np.random.RandomState(seed)

    # Pick crop locations avoiding borders
    margin = crop_size
    for i in range(n_crops):
        cy = rng.randint(margin, h - margin)
        cx = rng.randint(margin, w - margin)
        y1, y2 = cy - crop_size // 2, cy + crop_size // 2
        x1, x2 = cx - crop_size // 2, cx + crop_size // 2

        def to_uint8(arr: np.ndarray) -> np.ndarray:
            return np.clip(arr[y1:y2, x1:x2] * 255, 0, 255).astype(np.uint8)

        drz_crop = to_uint8(drizzled)
        single_crop = to_uint8(single)
        saa_crop = to_uint8(shift_add)

        drz_sharp = measure_sharpness(drizzled[y1:y2, x1:x2])
        single_sharp = measure_sharpness(single[y1:y2, x1:x2])
        saa_sharp = measure_sharpness(shift_add[y1:y2, x1:x2])

        label_h = 24

        def add_label(img: np.ndarray, text: str) -> np.ndarray:
            labeled = np.full((img.shape[0] + label_h, img.shape[1], 3), 255, dtype=np.uint8)
            labeled[label_h:, :] = img
            cv2.putText(labeled, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            return labeled

        drz_labeled = add_label(drz_crop, f"Drizzle (sharp={drz_sharp:.3f})")
        single_labeled = add_label(single_crop, f"Bicubic (sharp={single_sharp:.3f})")
        saa_labeled = add_label(saa_crop, f"Shift+Add (sharp={saa_sharp:.3f})")

        sep = np.full((drz_labeled.shape[0], 2, 3), 180, dtype=np.uint8)
        row = np.hstack([drz_labeled, sep, single_labeled, sep, saa_labeled])

        out_path = OUTPUT_DIR / f"crop_{i:02d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"  Saved {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark superdrizzle on real photos")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("-s", "--scale", type=int, default=None, help="Scale factor (default: auto)")
    parser.add_argument("-p", "--pixfrac", type=float, default=0.6, help="Pixfrac (default: 0.6)")
    parser.add_argument("--crops", type=int, default=5, help="Number of comparison crops (default: 5)")
    parser.add_argument("--ref", type=int, default=0, help="Reference frame index (default: 0)")
    args = parser.parse_args()

    # Read
    print(f"Reading {len(args.images)} images...", file=sys.stderr)
    images = read_images(args.images)
    h, w = images[0].shape[:2]
    print(f"  Size: {w}x{h}", file=sys.stderr)

    # Align
    print("Aligning frames...", file=sys.stderr)
    transforms = compute_transforms(images, ref=args.ref)
    n_aligned = sum(1 for t in transforms if t is not None)
    print(f"  {n_aligned}/{len(images)} frames aligned", file=sys.stderr)

    if n_aligned == 0:
        print("Error: no frames could be aligned", file=sys.stderr)
        sys.exit(1)

    # Scale
    if args.scale is not None:
        scale = args.scale
    else:
        scale = estimate_scale(transforms)
    print(f"  Scale: {scale}x", file=sys.stderr)

    aligned_images = [img for img, t in zip(images, transforms) if t is not None]
    aligned_transforms = [t for t in transforms if t is not None]

    # Drizzle
    print(f"Drizzling ({n_aligned} frames, {scale}x, pixfrac={args.pixfrac})...", file=sys.stderr)
    drizzled, weights = drizzle_combine(
        aligned_images, aligned_transforms, scale=scale, pixfrac=args.pixfrac,
        progress=True,
    )

    # Shift-and-add baseline
    print("Computing shift-and-add baseline...", file=sys.stderr)
    saa, _ = drizzle_combine(
        aligned_images, aligned_transforms, scale=scale, pixfrac=1.0,
    )

    # Single frame bicubic baseline
    single = cv2.resize(
        images[args.ref],
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    # Sharpness comparison (whole image)
    drz_sharp = measure_sharpness(drizzled)
    single_sharp = measure_sharpness(single)
    saa_sharp = measure_sharpness(saa)

    print(file=sys.stderr)
    print(f"{'Method':<15} {'Sharpness':>10} {'vs Single':>10}", file=sys.stderr)
    print("-" * 38, file=sys.stderr)
    print(f"{'Single bicubic':<15} {single_sharp:>10.4f} {'':>10}", file=sys.stderr)
    print(f"{'Shift+Add':<15} {saa_sharp:>10.4f} {saa_sharp / single_sharp:>9.2f}x", file=sys.stderr)
    print(f"{'Drizzle':<15} {drz_sharp:>10.4f} {drz_sharp / single_sharp:>9.2f}x", file=sys.stderr)

    # Coverage stats
    covered = (weights > 0).sum() / weights.size * 100
    print(f"\nWeight map coverage: {covered:.1f}%", file=sys.stderr)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(name: str, img: np.ndarray) -> None:
        out = np.clip(img * 255, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        path = OUTPUT_DIR / name
        cv2.imwrite(str(path), out_bgr)
        print(f"  Saved {path}", file=sys.stderr)

    print(file=sys.stderr)
    save("drizzled.png", drizzled)
    save("single_bicubic.png", single)
    save("shift_and_add.png", saa)

    # Save comparison crops
    print(file=sys.stderr)
    save_comparison_crops(drizzled, single, saa, n_crops=args.crops)


if __name__ == "__main__":
    main()
