#!/usr/bin/env python3
"""Benchmark superdrizzle against synthetic bursts with known ground truth.

Generates high-res test images with fine detail, creates synthetic burst
frames with known sub-pixel shifts, runs drizzle, and measures PSNR/SSIM
against the original high-res ground truth.

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --images 5 --frames 16 --scale 3
    uv run python benchmarks/benchmark.py --auto-align
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from superdrizzle import compute_transforms, drizzle_combine, estimate_scale


def generate_test_scene(
    width: int,
    height: int,
    scene_type: str,
    seed: int = 42,
) -> np.ndarray:
    """Generate a high-res test scene with fine detail.

    Returns uint8 RGB image.
    """
    rng = np.random.RandomState(seed)

    if scene_type == "circles":
        # Random circles of varying size, good for testing edge sharpness
        img = np.full((height, width, 3), 32, dtype=np.uint8)
        for _ in range(200):
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            r = rng.randint(3, min(width, height) // 8)
            color = tuple(int(c) for c in rng.randint(60, 255, 3))
            cv2.circle(img, (cx, cy), r, color, -1)
            # Thin outline for high-frequency detail
            cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)
        return img

    elif scene_type == "text":
        # Synthetic text-like pattern, tests legibility at small scales
        img = np.full((height, width, 3), 240, dtype=np.uint8)
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
        words = ["superdrizzle", "benchmark", "resolution", "PSNR", "subpixel",
                 "Fruchter", "Hook", "2002", "drizzle", "affine"]
        y = 30
        while y < height - 30:
            x = 10
            while x < width - 100:
                word = words[rng.randint(0, len(words))]
                font = fonts[rng.randint(0, len(fonts))]
                scale = rng.uniform(0.3, 1.2)
                color = tuple(int(c) for c in rng.randint(0, 100, 3))
                thickness = rng.randint(1, 3)
                cv2.putText(img, word, (x, y), font, scale, color, thickness)
                x += rng.randint(80, 200)
            y += rng.randint(25, 50)
        return img

    elif scene_type == "grid":
        # Fine grid pattern, ideal for measuring resolution recovery
        img = np.full((height, width, 3), 200, dtype=np.uint8)
        # Horizontal and vertical lines at varying spacing
        for spacing in [4, 8, 16, 32]:
            region_w = width // 4
            x_offset = {4: 0, 8: region_w, 16: region_w * 2, 32: region_w * 3}[spacing]
            for y in range(0, height, spacing):
                cv2.line(img, (x_offset, y), (x_offset + region_w, y), (40, 40, 40), 1)
            for x in range(x_offset, x_offset + region_w, spacing):
                cv2.line(img, (x, 0), (x, height), (40, 40, 40), 1)
        return img

    elif scene_type == "natural":
        # Synthetic "natural" scene with gradients, noise, and edges
        img = np.zeros((height, width, 3), dtype=np.float64)
        # Smooth gradient background
        for c in range(3):
            freq_x = rng.uniform(0.5, 3.0)
            freq_y = rng.uniform(0.5, 3.0)
            phase = rng.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:height, 0:width]
            img[:, :, c] = 0.5 + 0.3 * np.sin(2 * np.pi * freq_x * xx / width + phase) * \
                np.cos(2 * np.pi * freq_y * yy / height)
        # Add sharp edges (rectangles)
        for _ in range(30):
            x1, y1 = rng.randint(0, width), rng.randint(0, height)
            x2, y2 = x1 + rng.randint(10, 100), y1 + rng.randint(10, 100)
            color = rng.uniform(0.2, 0.9, 3)
            img[y1:y2, x1:x2] = color
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    raise ValueError(f"Unknown scene type: {scene_type}")


def generate_synthetic_burst(
    hires: np.ndarray,
    scale: int,
    n_frames: int,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[tuple[float, float]], np.ndarray]:
    """Generate a synthetic burst by downsampling a high-res image with known shifts.

    Simulates an undersampled detector: shifts the high-res image, then
    point-samples (takes every Nth pixel) to create low-res frames. This
    preserves aliasing, which is exactly what drizzle is designed to recover.
    """
    rng = np.random.RandomState(seed)
    h, w = hires.shape[:2]

    # Crop to be evenly divisible by scale
    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale
    hires = hires[:h_crop, :w_crop].copy()

    hires_f = hires.astype(np.float32) / 255.0

    frames = []
    shifts = []

    for i in range(n_frames):
        # Random sub-pixel shift in [0, 1) for each axis (in low-res coordinates)
        dx = rng.uniform(0, 1)
        dy = rng.uniform(0, 1)
        shifts.append((dx, dy))

        # Apply the shift at high-res level
        shift_x_hr = dx * scale
        shift_y_hr = dy * scale

        M = np.array([[1.0, 0.0, shift_x_hr], [0.0, 1.0, shift_y_hr]], dtype=np.float64)
        shifted = cv2.warpAffine(
            hires_f, M, (w_crop, h_crop),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Point-sample: take every Nth pixel (simulates undersampled detector)
        frame = shifted[::scale, ::scale, :].copy()

        # Add realistic sensor noise (shot noise + read noise)
        noise_sigma = 0.02  # ~5 DN at 8-bit, typical for a phone camera
        noise = rng.normal(0, noise_sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame + noise, 0.0, 1.0)

        frames.append(frame.astype(np.float32))

    return frames, shifts, hires_f


def build_transforms_from_shifts(
    shifts: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Build affine transform matrices from known shifts.

    The first frame is the reference. Transforms map each frame to the reference.
    """
    ref_dx, ref_dy = shifts[0]
    transforms = []
    for dx, dy in shifts:
        M = np.array([
            [1.0, 0.0, ref_dx - dx],
            [0.0, 1.0, ref_dy - dy],
        ], dtype=np.float64)
        transforms.append(M)
    return transforms


def compute_metrics(
    result: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR and SSIM between drizzle result and ground truth."""
    # Crop to matching size
    h = min(result.shape[0], ground_truth.shape[0])
    w = min(result.shape[1], ground_truth.shape[1])

    r_y = (result.shape[0] - h) // 2
    r_x = (result.shape[1] - w) // 2
    g_y = (ground_truth.shape[0] - h) // 2
    g_x = (ground_truth.shape[1] - w) // 2

    r = result[r_y:r_y + h, r_x:r_x + w]
    g = ground_truth[g_y:g_y + h, g_x:g_x + w]

    # Trim border to avoid edge artifacts
    border = 10
    if h > 2 * border and w > 2 * border:
        r = r[border:-border, border:-border]
        g = g[border:-border, border:-border]

    return {
        "psnr": float(psnr(g, r, data_range=1.0)),
        "ssim": float(ssim(g, r, data_range=1.0, channel_axis=2)),
    }


def benchmark_single(
    scene_type: str,
    width: int,
    height: int,
    scale: int,
    n_frames: int,
    auto_align: bool,
    pixfrac: float,
    seed: int,
) -> dict:
    """Run benchmark on a single synthetic scene."""
    hires = generate_test_scene(width, height, scene_type, seed=seed)

    frames, shifts, ground_truth = generate_synthetic_burst(
        hires, scale=scale, n_frames=n_frames, seed=seed,
    )

    if auto_align:
        transforms = compute_transforms(frames, ref=0)
    else:
        transforms = build_transforms_from_shifts(shifts)

    n_aligned = sum(1 for t in transforms if t is not None)

    aligned_frames = [f for f, t in zip(frames, transforms) if t is not None]
    aligned_transforms = [t for t in transforms if t is not None]

    t0 = time.time()
    result, weights = drizzle_combine(
        aligned_frames, aligned_transforms, scale=scale, pixfrac=pixfrac,
    )
    drizzle_time = time.time() - t0

    # Baseline: single frame bicubic upscale
    baseline = cv2.resize(
        frames[0],
        (frames[0].shape[1] * scale, frames[0].shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    # Baseline 2: average all frames then bicubic upscale
    avg_frame = np.mean(frames, axis=0).astype(np.float32)
    avg_upscaled = cv2.resize(
        avg_frame,
        (avg_frame.shape[1] * scale, avg_frame.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    metrics_drizzle = compute_metrics(result, ground_truth)
    metrics_single = compute_metrics(baseline, ground_truth)
    metrics_avg = compute_metrics(avg_upscaled, ground_truth)

    return {
        "scene": scene_type,
        "n_frames": n_frames,
        "n_aligned": n_aligned,
        "drizzle_time_s": drizzle_time,
        "drizzle_psnr": metrics_drizzle["psnr"],
        "drizzle_ssim": metrics_drizzle["ssim"],
        "single_psnr": metrics_single["psnr"],
        "single_ssim": metrics_single["ssim"],
        "avg_psnr": metrics_avg["psnr"],
        "avg_ssim": metrics_avg["ssim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark superdrizzle against synthetic ground truth")
    parser.add_argument("--frames", type=int, default=8, help="Frames per burst (default: 8)")
    parser.add_argument("--scale", type=int, default=2, help="Superresolution scale (default: 2)")
    parser.add_argument("--pixfrac", type=float, default=0.6, help="Pixfrac (default: 0.6)")
    parser.add_argument("--size", type=int, default=512, help="High-res image size (default: 512)")
    parser.add_argument("--auto-align", action="store_true", help="Use ORB auto-alignment instead of known shifts")
    args = parser.parse_args()

    scenes = ["circles", "text", "grid", "natural"]

    print(f"Benchmark: {args.frames} frames/burst, {args.scale}x scale, "
          f"pixfrac={args.pixfrac}, {args.size}x{args.size} images", file=sys.stderr)
    print(f"Alignment: {'auto (ORB + RANSAC)' if args.auto_align else 'known ground truth shifts'}", file=sys.stderr)
    print(file=sys.stderr)

    # Header
    print(f"{'Scene':<10} {'Drizzle PSNR':>13} {'Single PSNR':>12} {'Avg PSNR':>9} "
          f"{'Drizzle SSIM':>13} {'Single SSIM':>12} {'Avg SSIM':>9} {'Time':>6}",
          file=sys.stderr)
    print("-" * 95, file=sys.stderr)

    results = []
    for i, scene in enumerate(scenes):
        r = benchmark_single(
            scene_type=scene,
            width=args.size, height=args.size,
            scale=args.scale, n_frames=args.frames,
            auto_align=args.auto_align, pixfrac=args.pixfrac,
            seed=42 + i,
        )
        results.append(r)

        print(f"{r['scene']:<10} {r['drizzle_psnr']:>10.2f} dB {r['single_psnr']:>9.2f} dB "
              f"{r['avg_psnr']:>6.2f} dB {r['drizzle_ssim']:>10.4f}    "
              f"{r['single_ssim']:>9.4f}    {r['avg_ssim']:>6.4f}    "
              f"{r['drizzle_time_s']:>4.1f}s", file=sys.stderr)

    # Summary
    print("-" * 95, file=sys.stderr)
    avg = lambda key: np.mean([r[key] for r in results])
    print(f"{'AVERAGE':<10} {avg('drizzle_psnr'):>10.2f} dB {avg('single_psnr'):>9.2f} dB "
          f"{avg('avg_psnr'):>6.2f} dB {avg('drizzle_ssim'):>10.4f}    "
          f"{avg('single_ssim'):>9.4f}    {avg('avg_ssim'):>6.4f}    ",
          file=sys.stderr)

    print(file=sys.stderr)
    d_psnr = avg('drizzle_psnr') - avg('single_psnr')
    d_ssim = avg('drizzle_ssim') - avg('single_ssim')
    print(f"Drizzle vs single-frame bicubic: +{d_psnr:.2f} dB PSNR, +{d_ssim:.4f} SSIM", file=sys.stderr)

    d_psnr2 = avg('drizzle_psnr') - avg('avg_psnr')
    d_ssim2 = avg('drizzle_ssim') - avg('avg_ssim')
    print(f"Drizzle vs averaged+bicubic:     +{d_psnr2:.2f} dB PSNR, +{d_ssim2:.4f} SSIM", file=sys.stderr)


if __name__ == "__main__":
    main()
