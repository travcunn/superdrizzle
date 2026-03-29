"""CLI entry point: wire together read, align, estimate, drizzle, write."""

import argparse
import sys

from drizzle.align import compute_transforms
from drizzle.drizzle import drizzle_combine
from drizzle.estimate import estimate_scale
from drizzle.io import read_images, write_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine dithered images into a higher-resolution output using drizzle."
    )
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument(
        "-p", "--pixfrac", type=float, default=0.6,
        help="Drop shrink factor (0.0-1.0, default: 0.6)",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=None,
        help="Output scale factor (default: auto-estimate)",
    )
    parser.add_argument(
        "--weights", action="store_true",
        help="Emit weight map alongside output",
    )
    parser.add_argument(
        "--ref", type=int, default=0,
        help="Index of reference frame (default: 0)",
    )
    args = parser.parse_args()

    # 1. Read
    print(f"Reading {len(args.images)} images...", file=sys.stderr)
    images = read_images(args.images)
    print(f"  Image size: {images[0].shape[1]}x{images[0].shape[0]}", file=sys.stderr)

    # 2. Align
    print("Aligning frames...", file=sys.stderr)
    transforms = compute_transforms(images, ref=args.ref)
    n_aligned = sum(1 for t in transforms if t is not None)
    print(f"  {n_aligned}/{len(images)} frames aligned", file=sys.stderr)

    if n_aligned == 0:
        print("Error: no frames could be aligned", file=sys.stderr)
        sys.exit(1)

    # 3. Estimate scale
    if args.scale is not None:
        scale = args.scale
        print(f"Using user-specified scale: {scale}x", file=sys.stderr)
    else:
        scale = estimate_scale(transforms)
        print(f"Auto-estimated scale: {scale}x", file=sys.stderr)

    # 4. Drizzle
    print(f"Drizzling (pixfrac={args.pixfrac}, scale={scale}x)...", file=sys.stderr)
    aligned_images = [img for img, t in zip(images, transforms) if t is not None]
    aligned_transforms = [t for t in transforms if t is not None]
    result, weights = drizzle_combine(
        aligned_images, aligned_transforms, scale=scale, pixfrac=args.pixfrac,
    )

    # 5. Write
    print(f"Writing output to {args.output}", file=sys.stderr)
    write_image(args.output, result, weights=weights if args.weights else None)

    out_h, out_w = result.shape[:2]
    print(f"Done. Output size: {out_w}x{out_h}", file=sys.stderr)
