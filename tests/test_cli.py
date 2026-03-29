import subprocess
import tempfile
import os

import cv2
import numpy as np


def _make_test_scene(directory: str, n_frames: int = 4) -> list[str]:
    """Create test frames: a textured image with sub-pixel shifts."""
    rng = np.random.RandomState(42)
    h, w = 100, 100
    base = rng.randint(0, 256, (h + 20, w + 20, 3), dtype=np.uint8)
    # Add structure
    for _ in range(30):
        cx, cy = rng.randint(0, w + 20), rng.randint(0, h + 20)
        r = rng.randint(3, 15)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(base, (cx, cy), r, color, -1)

    paths = []
    for i in range(n_frames):
        # Sub-pixel shifts
        dx = rng.uniform(0, 1) + i * 0.3
        dy = rng.uniform(0, 1) + i * 0.2
        ix, iy = int(dx), int(dy)
        frame = base[iy:iy + h, ix:ix + w].copy()
        p = os.path.join(directory, f"frame_{i:02d}.png")
        cv2.imwrite(p, frame)
        paths.append(p)

    return paths


def test_cli_produces_output():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_test_scene(d)
        out = os.path.join(d, "result.png")
        result = subprocess.run(
            ["uv", "run", "superdrizzle"] + paths + ["-o", out, "-s", "2", "--weights"],
            capture_output=True,
            text=True,
            cwd="/Users/tcunningham/drizzle",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(out), "Output image not created"
        img = cv2.imread(out)
        assert img is not None
        assert img.shape[0] == 200  # 100 * scale=2
        assert img.shape[1] == 200
        # Weight map should exist
        weight_path = os.path.join(d, "result_weights.png")
        assert os.path.exists(weight_path)


def test_cli_auto_scale():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_test_scene(d)
        out = os.path.join(d, "result.png")
        result = subprocess.run(
            ["uv", "run", "superdrizzle"] + paths + ["-o", out],
            capture_output=True,
            text=True,
            cwd="/Users/tcunningham/drizzle",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(out)


def test_cli_bad_file_path_exits_with_error():
    result = subprocess.run(
        ["uv", "run", "superdrizzle", "/nonexistent/frame.png", "-o", "/tmp/out.png"],
        capture_output=True,
        text=True,
        cwd="/Users/tcunningham/drizzle",
    )
    assert result.returncode != 0


def test_cli_single_image():
    with tempfile.TemporaryDirectory() as d:
        paths = _make_test_scene(d, n_frames=1)
        out = os.path.join(d, "result.png")
        result = subprocess.run(
            ["uv", "run", "superdrizzle"] + paths + ["-o", out],
            capture_output=True,
            text=True,
            cwd="/Users/tcunningham/drizzle",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(out)
