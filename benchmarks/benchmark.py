#!/usr/bin/env python3
"""Benchmark superdrizzle against synthetic bursts with known ground truth.

Modes:
    default     Run once with given params, print table
    --sweep     Sweep over frame counts (2, 4, 8, 16, 32)
    --pixsweep  Sweep over pixfrac (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    --noise     Sweep over noise levels (0.0, 0.01, 0.02, 0.05, 0.1)
    --visual    Save side-by-side comparison crops to benchmarks/output/

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --sweep
    uv run python benchmarks/benchmark.py --visual
    uv run python benchmarks/benchmark.py --noise --frames 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from superdrizzle import compute_transforms, drizzle_combine, estimate_scale

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_test_scene(
    width: int,
    height: int,
    scene_type: str,
    seed: int = 42,
) -> np.ndarray:
    """Generate a high-res test scene with fine detail. Returns uint8 RGB.

    Scenes are designed to have content at multiple spatial frequencies,
    especially near the Nyquist limit of a 2x-downsampled detector, since
    that's the band drizzle can recover.
    """
    rng = np.random.RandomState(seed)

    if scene_type == "starfield":
        # Point sources at sub-pixel positions with varying brightness.
        # The classic drizzle use case from astronomy. Stars are Gaussian
        # with FWHM of 2-4 pixels (well-sampled at high-res, undersampled
        # when downsampled by 2x).
        img = np.full((height, width, 3), 8, dtype=np.float64)
        n_stars = 300
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        for _ in range(n_stars):
            cx = rng.uniform(5, width - 5)
            cy = rng.uniform(5, height - 5)
            sigma = rng.uniform(1.0, 2.0)
            brightness = rng.uniform(0.3, 1.0)
            color = np.array([1.0, 1.0, 1.0]) * brightness
            # Tint some stars
            if rng.random() < 0.3:
                color = color * rng.uniform(0.5, 1.0, 3)
            gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            for c in range(3):
                img[:, :, c] += gauss * color[c]
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    elif scene_type == "siemens":
        # Siemens star: a radial frequency sweep. The center has the highest
        # spatial frequency (finest spokes), the edges have the lowest. This
        # tests resolution recovery at all frequencies in a single image.
        img = np.zeros((height, width), dtype=np.float64)
        cy, cx = height / 2, width / 2
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        angle = np.arctan2(yy - cy, xx - cx)
        n_spokes = 36
        img = 0.5 + 0.4 * np.sin(n_spokes * angle)
        # Add a second harmonic for more complexity
        img += 0.1 * np.sin(2 * n_spokes * angle)
        img = np.clip(img, 0, 1)
        rgb = np.stack([img, img, img], axis=2)
        return (rgb * 255).astype(np.uint8)

    elif scene_type == "resolution":
        # Resolution test chart: groups of bars at decreasing spacing.
        # Similar to a USAF 1951 target. Tests whether drizzle can resolve
        # bar groups that a single frame cannot.
        img = np.full((height, width, 3), 220, dtype=np.uint8)
        # Bar groups: 3 bars each, horizontal and vertical, at spacings
        # from 2px to 16px (at high-res). At 2x downsample, the 2px and 3px
        # groups are below Nyquist and should be recoverable by drizzle.
        spacings = [2, 3, 4, 6, 8, 12, 16]
        n_groups = len(spacings)
        group_h = height // (n_groups + 1)

        for gi, spacing in enumerate(spacings):
            y_start = (gi + 1) * group_h - group_h // 2
            bar_width = max(1, spacing // 2)
            # Vertical bars on left half
            x = width // 8
            for bar in range(3):
                x1 = x + bar * spacing
                x2 = x1 + bar_width
                cv2.rectangle(img, (x1, y_start), (x2, y_start + group_h // 2), (30, 30, 30), -1)
            # Horizontal bars on right half
            y = y_start
            for bar in range(3):
                y1 = y + bar * spacing
                y2 = y1 + bar_width
                x_start = width // 2 + width // 8
                cv2.rectangle(img, (x_start, y1), (x_start + group_h // 2, y2), (30, 30, 30), -1)
            # Label
            cv2.putText(img, f"{spacing}px", (4, y_start + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        return img

    elif scene_type == "fractal":
        # Multi-scale fractal noise with sharp edges. Simulates natural
        # textures (foliage, fabric, terrain) that have energy at all
        # spatial frequencies following a 1/f power law.
        img = np.zeros((height, width, 3), dtype=np.float64)
        for c in range(3):
            layer = np.zeros((height, width), dtype=np.float64)
            # Octaves of Perlin-like noise (simplified: sum of random
            # frequency sine waves at decreasing amplitude)
            for octave in range(8):
                freq = 2 ** octave
                amplitude = 1.0 / (freq ** 0.8)  # 1/f^0.8 spectrum
                n_waves = 3
                for _ in range(n_waves):
                    kx = rng.uniform(-freq, freq) * 2 * np.pi / width
                    ky = rng.uniform(-freq, freq) * 2 * np.pi / height
                    phase = rng.uniform(0, 2 * np.pi)
                    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
                    layer += amplitude * np.sin(kx * xx + ky * yy + phase) / n_waves
            img[:, :, c] = layer
        # Normalize to [0, 1]
        img = img - img.min()
        img = img / (img.max() + 1e-10)
        # Add sharp edges: random Voronoi-like boundaries
        n_regions = 20
        centers = np.column_stack([
            rng.randint(0, width, n_regions),
            rng.randint(0, height, n_regions),
        ])
        region_colors = rng.uniform(0.2, 0.8, (n_regions, 3))
        yy, xx = np.mgrid[0:height, 0:width]
        coords = np.stack([xx, yy], axis=2).reshape(-1, 2)
        dists = np.linalg.norm(
            coords[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        nearest = dists.argmin(axis=1).reshape(height, width)
        voronoi = region_colors[nearest]
        # Blend fractal noise with Voronoi regions
        img = 0.6 * img + 0.4 * voronoi
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    elif scene_type == "edges":
        # Dense edges at multiple orientations and scales. Diagonal lines,
        # curves, and corners that test anti-aliasing recovery. This is
        # where drizzle's sub-pixel positioning shines: reconstructing
        # edge profiles that were aliased in individual frames.
        img = np.full((height, width, 3), 200, dtype=np.uint8)
        # Concentric polygons at varying sizes
        cx, cy = width // 2, height // 2
        for n_sides in [3, 4, 5, 6, 8, 12]:
            for radius in range(20, min(width, height) // 2, 30):
                angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
                angles += rng.uniform(0, np.pi / n_sides)
                pts = np.array([
                    [int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))]
                    for a in angles
                ], dtype=np.int32)
                color = tuple(int(c) for c in rng.randint(20, 120, 3))
                thickness = rng.choice([1, 2])
                cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
        # Diagonal lines at various angles (anti-aliased)
        for _ in range(50):
            x1, y1 = rng.randint(0, width), rng.randint(0, height)
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.randint(30, 150)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            color = tuple(int(c) for c in rng.randint(20, 100, 3))
            cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        return img

    elif scene_type == "zoneplate":
        # Zone plate (Fresnel pattern): spatial frequency increases
        # radially from center. At some radius the frequency crosses the
        # Nyquist limit of the downsampled detector, creating visible
        # aliasing. Drizzle should push that boundary outward.
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        cy, cx = height / 2, width / 2
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        # Frequency increases with radius
        k = 0.5 / max(width, height)
        phase = k * r2
        img = 0.5 + 0.45 * np.cos(2 * np.pi * phase)
        rgb = np.stack([img, img * 0.95, img * 0.9], axis=2)  # slight warm tint
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown scene type: {scene_type}")


# ---------------------------------------------------------------------------
# Synthetic burst generation
# ---------------------------------------------------------------------------

def generate_synthetic_burst(
    hires: np.ndarray,
    scale: int,
    n_frames: int,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[tuple[float, float]], np.ndarray]:
    """Generate a synthetic burst by point-sampling a shifted high-res image.

    Simulates an undersampled detector with additive Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    h, w = hires.shape[:2]

    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale
    hires = hires[:h_crop, :w_crop].copy()
    hires_f = hires.astype(np.float32) / 255.0

    frames = []
    shifts = []

    for i in range(n_frames):
        dx = rng.uniform(0, 1)
        dy = rng.uniform(0, 1)
        shifts.append((dx, dy))

        shift_x_hr = dx * scale
        shift_y_hr = dy * scale

        M = np.array([[1.0, 0.0, shift_x_hr], [0.0, 1.0, shift_y_hr]], dtype=np.float64)
        shifted = cv2.warpAffine(
            hires_f, M, (w_crop, h_crop),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        frame = shifted[::scale, ::scale, :].copy()

        if noise_sigma > 0:
            noise = rng.normal(0, noise_sigma, frame.shape).astype(np.float32)
            frame = np.clip(frame + noise, 0.0, 1.0)

        frames.append(frame.astype(np.float32))

    return frames, shifts, hires_f


def build_transforms_from_shifts(
    shifts: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Build affine transforms from known shifts. First frame is reference."""
    ref_dx, ref_dy = shifts[0]
    return [
        np.array([[1.0, 0.0, ref_dx - dx], [0.0, 1.0, ref_dy - dy]], dtype=np.float64)
        for dx, dy in shifts
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

BORDER = 10


def crop_and_match(result: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop both arrays to matching size, trim border."""
    h = min(result.shape[0], ground_truth.shape[0])
    w = min(result.shape[1], ground_truth.shape[1])

    r_y, r_x = (result.shape[0] - h) // 2, (result.shape[1] - w) // 2
    g_y, g_x = (ground_truth.shape[0] - h) // 2, (ground_truth.shape[1] - w) // 2

    r = result[r_y:r_y + h, r_x:r_x + w]
    g = ground_truth[g_y:g_y + h, g_x:g_x + w]

    if h > 2 * BORDER and w > 2 * BORDER:
        r = r[BORDER:-BORDER, BORDER:-BORDER]
        g = g[BORDER:-BORDER, BORDER:-BORDER]

    return r, g


def compute_metrics(result: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    r, g = crop_and_match(result, ground_truth)
    return {
        "psnr": float(psnr(g, r, data_range=1.0)),
        "ssim": float(ssim(g, r, data_range=1.0, channel_axis=2)),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_single_bicubic(frames: list[np.ndarray], scale: int) -> np.ndarray:
    """Single frame upscaled with bicubic interpolation."""
    return cv2.resize(
        frames[0],
        (frames[0].shape[1] * scale, frames[0].shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )


def baseline_shift_and_add(
    frames: list[np.ndarray],
    transforms: list[np.ndarray | None],
    scale: int,
) -> np.ndarray:
    """Shift-and-add: drizzle with pixfrac=1.0 (no drop shrinking)."""
    aligned = [f for f, t in zip(frames, transforms) if t is not None]
    aligned_t = [t for t in transforms if t is not None]
    result, _ = drizzle_combine(aligned, aligned_t, scale=scale, pixfrac=1.0)
    return result


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_single(
    scene_type: str,
    width: int,
    height: int,
    scale: int,
    n_frames: int,
    auto_align: bool,
    pixfrac: float,
    noise_sigma: float,
    seed: int,
) -> dict:
    """Run benchmark on a single synthetic scene."""
    hires = generate_test_scene(width, height, scene_type, seed=seed)

    frames, shifts, ground_truth = generate_synthetic_burst(
        hires, scale=scale, n_frames=n_frames, noise_sigma=noise_sigma, seed=seed,
    )

    if auto_align:
        transforms = compute_transforms(frames, ref=0)
    else:
        transforms = build_transforms_from_shifts(shifts)

    n_aligned = sum(1 for t in transforms if t is not None)
    aligned_frames = [f for f, t in zip(frames, transforms) if t is not None]
    aligned_transforms = [t for t in transforms if t is not None]

    # Drizzle
    t0 = time.time()
    result, weights = drizzle_combine(
        aligned_frames, aligned_transforms, scale=scale, pixfrac=pixfrac,
    )
    drizzle_time = time.time() - t0

    # Baselines
    single = baseline_single_bicubic(frames, scale)
    saa = baseline_shift_and_add(frames, transforms, scale)

    return {
        "scene": scene_type,
        "n_frames": n_frames,
        "n_aligned": n_aligned,
        "noise_sigma": noise_sigma,
        "pixfrac": pixfrac,
        "drizzle_time_s": drizzle_time,
        "drizzle": compute_metrics(result, ground_truth),
        "single": compute_metrics(single, ground_truth),
        "shift_add": compute_metrics(saa, ground_truth),
        # Keep arrays for visual output
        "_result": result,
        "_single": single,
        "_saa": saa,
        "_ground_truth": ground_truth,
    }


SCENES = ["starfield", "siemens", "resolution", "fractal", "edges", "zoneplate"]


def run_benchmark(
    scenes: list[str],
    n_frames: int,
    scale: int,
    pixfrac: float,
    noise_sigma: float,
    size: int,
    auto_align: bool,
) -> list[dict]:
    results = []
    for i, scene in enumerate(scenes):
        r = benchmark_single(
            scene_type=scene, width=size, height=size,
            scale=scale, n_frames=n_frames,
            auto_align=auto_align, pixfrac=pixfrac,
            noise_sigma=noise_sigma, seed=42 + i,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Print results table comparing drizzle, single-frame, and shift-and-add."""
    hdr = (f"{'Scene':<10} {'Drizzle':>14} {'Single':>14} {'Shift+Add':>14} "
           f"{'Drizzle':>10} {'Single':>10} {'Shift+Add':>10} {'Time':>6}")
    sub = (f"{'':10} {'PSNR (dB)':>14} {'PSNR (dB)':>14} {'PSNR (dB)':>14} "
           f"{'SSIM':>10} {'SSIM':>10} {'SSIM':>10}")
    print(sub, file=sys.stderr)
    print(hdr, file=sys.stderr)
    print("-" * 100, file=sys.stderr)

    for r in results:
        print(f"{r['scene']:<10} "
              f"{r['drizzle']['psnr']:>11.2f} dB "
              f"{r['single']['psnr']:>11.2f} dB "
              f"{r['shift_add']['psnr']:>11.2f} dB "
              f"{r['drizzle']['ssim']:>10.4f} "
              f"{r['single']['ssim']:>10.4f} "
              f"{r['shift_add']['ssim']:>10.4f} "
              f"{r['drizzle_time_s']:>5.1f}s",
              file=sys.stderr)

    print("-" * 100, file=sys.stderr)
    avg = lambda m, k: np.mean([r[m][k] for r in results])
    print(f"{'AVERAGE':<10} "
          f"{avg('drizzle', 'psnr'):>11.2f} dB "
          f"{avg('single', 'psnr'):>11.2f} dB "
          f"{avg('shift_add', 'psnr'):>11.2f} dB "
          f"{avg('drizzle', 'ssim'):>10.4f} "
          f"{avg('single', 'ssim'):>10.4f} "
          f"{avg('shift_add', 'ssim'):>10.4f}",
          file=sys.stderr)

    print(file=sys.stderr)
    dp = avg('drizzle', 'psnr') - avg('single', 'psnr')
    ds = avg('drizzle', 'ssim') - avg('single', 'ssim')
    print(f"Drizzle vs single-frame:  {dp:+.2f} dB PSNR, {ds:+.4f} SSIM", file=sys.stderr)
    dp2 = avg('drizzle', 'psnr') - avg('shift_add', 'psnr')
    ds2 = avg('drizzle', 'ssim') - avg('shift_add', 'ssim')
    print(f"Drizzle vs shift-and-add: {dp2:+.2f} dB PSNR, {ds2:+.4f} SSIM", file=sys.stderr)


def save_visual_comparisons(results: list[dict], scale: int) -> None:
    """Save side-by-side comparison crops for each scene."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for r in results:
        gt = r["_ground_truth"]
        drz = r["_result"]
        single = r["_single"]
        saa = r["_saa"]

        # Pick a center crop (128x128 at output resolution)
        crop_size = 128
        cy, cx = gt.shape[0] // 2, gt.shape[1] // 2
        y1, y2 = cy - crop_size // 2, cy + crop_size // 2
        x1, x2 = cx - crop_size // 2, cx + crop_size // 2

        def to_uint8(arr: np.ndarray) -> np.ndarray:
            c = arr[y1:y2, x1:x2]
            return np.clip(c * 255, 0, 255).astype(np.uint8)

        gt_crop = to_uint8(gt)
        drz_crop = to_uint8(drz)
        single_crop = to_uint8(single)
        saa_crop = to_uint8(saa)

        # Add labels
        label_h = 24
        def add_label(img: np.ndarray, text: str) -> np.ndarray:
            labeled = np.full((img.shape[0] + label_h, img.shape[1], 3), 255, dtype=np.uint8)
            labeled[label_h:, :] = img
            cv2.putText(labeled, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            return labeled

        gt_labeled = add_label(gt_crop, "Ground Truth")
        drz_labeled = add_label(drz_crop, f"Drizzle ({r['drizzle']['psnr']:.1f} dB)")
        single_labeled = add_label(single_crop, f"Bicubic ({r['single']['psnr']:.1f} dB)")
        saa_labeled = add_label(saa_crop, f"Shift+Add ({r['shift_add']['psnr']:.1f} dB)")

        # Horizontal concat with 2px separator
        sep = np.full((gt_labeled.shape[0], 2, 3), 180, dtype=np.uint8)
        row = np.hstack([gt_labeled, sep, drz_labeled, sep, single_labeled, sep, saa_labeled])

        out_path = OUTPUT_DIR / f"{r['scene']}_comparison.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"  Saved {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sweep modes
# ---------------------------------------------------------------------------

def sweep_frames(args: argparse.Namespace) -> None:
    """Show how quality scales with frame count."""
    frame_counts = [2, 4, 8, 16, 32]
    print(f"Frame count sweep: scale={args.scale}x, pixfrac={args.pixfrac}, "
          f"noise={args.noise:.3f}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Frames':>7} {'Drizzle PSNR':>13} {'Single PSNR':>12} {'S+A PSNR':>9} "
          f"{'Drizzle SSIM':>13} {'Single SSIM':>12} {'S+A SSIM':>9}",
          file=sys.stderr)
    print("-" * 85, file=sys.stderr)

    for nf in frame_counts:
        results = run_benchmark(
            SCENES, n_frames=nf, scale=args.scale, pixfrac=args.pixfrac,
            noise_sigma=args.noise, size=args.size, auto_align=args.auto_align,
        )
        avg = lambda m, k: np.mean([r[m][k] for r in results])
        print(f"{nf:>7} {avg('drizzle', 'psnr'):>10.2f} dB "
              f"{avg('single', 'psnr'):>9.2f} dB "
              f"{avg('shift_add', 'psnr'):>6.2f} dB "
              f"{avg('drizzle', 'ssim'):>10.4f}    "
              f"{avg('single', 'ssim'):>9.4f}    "
              f"{avg('shift_add', 'ssim'):>6.4f}",
              file=sys.stderr)


def sweep_pixfrac(args: argparse.Namespace) -> None:
    """Show the pixfrac trade-off curve."""
    pixfracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"Pixfrac sweep: {args.frames} frames, scale={args.scale}x, "
          f"noise={args.noise:.3f}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Pixfrac':>8} {'PSNR (dB)':>10} {'SSIM':>8} {'Note':<20}",
          file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    for pf in pixfracs:
        results = run_benchmark(
            SCENES, n_frames=args.frames, scale=args.scale, pixfrac=pf,
            noise_sigma=args.noise, size=args.size, auto_align=args.auto_align,
        )
        avg_psnr = np.mean([r["drizzle"]["psnr"] for r in results])
        avg_ssim = np.mean([r["drizzle"]["ssim"] for r in results])
        note = {0.0: "(interlacing)", 0.6: "(default)", 1.0: "(shift-and-add)"}.get(pf, "")
        print(f"{pf:>8.1f} {avg_psnr:>7.2f} dB {avg_ssim:>8.4f} {note}",
              file=sys.stderr)


def sweep_noise(args: argparse.Namespace) -> None:
    """Show drizzle advantage at different noise levels."""
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    print(f"Noise sweep: {args.frames} frames, scale={args.scale}x, "
          f"pixfrac={args.pixfrac}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Noise':>7} {'Drizzle PSNR':>13} {'Single PSNR':>12} {'Gain':>8} "
          f"{'Drizzle SSIM':>13} {'Single SSIM':>12} {'Gain':>8}",
          file=sys.stderr)
    print("-" * 85, file=sys.stderr)

    for ns in noise_levels:
        results = run_benchmark(
            SCENES, n_frames=args.frames, scale=args.scale, pixfrac=args.pixfrac,
            noise_sigma=ns, size=args.size, auto_align=args.auto_align,
        )
        avg = lambda m, k: np.mean([r[m][k] for r in results])
        dp = avg('drizzle', 'psnr') - avg('single', 'psnr')
        ds = avg('drizzle', 'ssim') - avg('single', 'ssim')
        print(f"{ns:>7.3f} {avg('drizzle', 'psnr'):>10.2f} dB "
              f"{avg('single', 'psnr'):>9.2f} dB {dp:>+5.2f} dB "
              f"{avg('drizzle', 'ssim'):>10.4f}    "
              f"{avg('single', 'ssim'):>9.4f}    {ds:>+.4f}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark superdrizzle against synthetic ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
sweep modes:
  --sweep       frame count sweep (2, 4, 8, 16, 32)
  --pixsweep    pixfrac sweep (0.0 - 1.0)
  --noise       noise level sweep (0.0 - 0.1)
  --visual      save comparison crops to benchmarks/output/
""",
    )
    parser.add_argument("--frames", type=int, default=8, help="Frames per burst (default: 8)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default: 2)")
    parser.add_argument("--pixfrac", type=float, default=0.6, help="Pixfrac (default: 0.6)")
    parser.add_argument("--size", type=int, default=512, help="High-res image size (default: 512)")
    parser.add_argument("--noise", type=float, default=0.02, help="Noise sigma (default: 0.02)")
    parser.add_argument("--auto-align", action="store_true", help="Use ORB auto-alignment")
    parser.add_argument("--sweep", action="store_true", help="Sweep over frame counts")
    parser.add_argument("--pixsweep", action="store_true", help="Sweep over pixfrac values")
    parser.add_argument("--noise-sweep", action="store_true", dest="noise_sweep", help="Sweep over noise levels")
    parser.add_argument("--visual", action="store_true", help="Save comparison images")
    args = parser.parse_args()

    if args.sweep:
        sweep_frames(args)
        return
    if args.pixsweep:
        sweep_pixfrac(args)
        return
    if args.noise_sweep:
        sweep_noise(args)
        return

    # Default: single run
    print(f"Benchmark: {args.frames} frames, {args.scale}x, pixfrac={args.pixfrac}, "
          f"noise={args.noise:.3f}, {args.size}x{args.size}",
          file=sys.stderr)
    print(f"Alignment: {'auto (ORB)' if args.auto_align else 'known shifts'}",
          file=sys.stderr)
    print(file=sys.stderr)

    results = run_benchmark(
        SCENES, n_frames=args.frames, scale=args.scale, pixfrac=args.pixfrac,
        noise_sigma=args.noise, size=args.size, auto_align=args.auto_align,
    )
    print_table(results)

    if args.visual:
        print(file=sys.stderr)
        save_visual_comparisons(results, args.scale)


if __name__ == "__main__":
    main()
