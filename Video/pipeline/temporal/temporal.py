#!/usr/bin/env python3
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Video Pipeline — Phase 3: Temporal Consistency
==========================================================
Smooth Gaussian positions and attributes across consecutive frames to
reduce temporal flicker, and suppress transient shadow blob artifacts.

Operates on the .ply files from Phase 2 (reconstruct), producing a new
set of smoothed .ply + .splat files.

Key insight: SHARP produces a fixed grid of 1,179,648 Gaussians per frame.
Gaussian i in frame N maps to the same pixel position as Gaussian i in
frame N±1. No optical flow or correspondence search is needed — the
array index IS the correspondence.

Steps (applied per-frame via sliding window):
    1. Smooth positions  — Gaussian-weighted rolling average on (x, y, z)
    2. Smooth attributes — same for color/opacity/scale/rotation
    3. Suppress shadows  — detect + clamp transient dark blobs

Usage:
    python temporal.py <reconstruct_output_dir> [options]

    # Process Phase 2 output for testclip2:
    python temporal.py ../reconstruct/output/testclip2

    # Custom window sizes:
    python temporal.py ../reconstruct/output/testclip2 --pos-window 4 --attr-window 3

Output:
    pipeline/temporal/output/<clip_name>/
    ├── splats/              ← smoothed .ply + .splat + manifest.json
    └── temporal_log.json    ← per-frame stats

VERSIONING NOTE:
    Phase 2 outputs in pipeline/reconstruct/output/ are NOT modified.
    v1 baseline in Video/output/ is also NOT modified.
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

# Add parent to path so steps/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from steps import read_ply, write_ply, GAUSSIAN_DTYPE
from steps.smooth_attributes import smooth_attributes_ema
from steps.suppress_shadows import suppress_shadows


# ─── PLY → .splat conversion (inlined from reconstruct.py) ───────────────────

C0 = 0.28209479177387814


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _convert_ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    """Convert a single .ply to .splat format. Returns Gaussian count."""
    vertices, _ = read_ply(ply_path)
    n = len(vertices)

    positions = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype("<f4")
    scales = np.column_stack([
        np.exp(vertices["scale_0"]),
        np.exp(vertices["scale_1"]),
        np.exp(vertices["scale_2"]),
    ]).astype("<f4")

    r = np.clip((0.5 + C0 * vertices["f_dc_0"]) * 255, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * vertices["f_dc_1"]) * 255, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * vertices["f_dc_2"]) * 255, 0, 255).astype(np.uint8)
    a = np.clip(_sigmoid(vertices["opacity"]) * 255, 0, 255).astype(np.uint8)
    rgba = np.column_stack([r, g, b, a])

    quats = np.column_stack([
        vertices["rot_0"], vertices["rot_1"],
        vertices["rot_2"], vertices["rot_3"],
    ])
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    quats = quats / norms
    quat_u8 = np.clip((quats * 128) + 128, 0, 255).astype(np.uint8)

    out_dtype = np.dtype([
        ("pos", "<f4", 3), ("scale", "<f4", 3),
        ("rgba", "u1", 4), ("quat", "u1", 4),
    ])
    out = np.zeros(n, dtype=out_dtype)
    out["pos"] = positions
    out["scale"] = scales
    out["rgba"] = rgba
    out["quat"] = quat_u8

    splat_path.write_bytes(out.tobytes())
    return n


# ─── Main pipeline ───────────────────────────────────────────────────────────

def temporal_smooth(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    color_alpha: float = 0.0,
    opacity_alpha: float = 0.0,
    opacity_threshold: float = 0.3,
    dark_threshold: float = 0.15,
    no_suppress: bool = False,
    convert: bool = True,
) -> dict:
    """
    Apply temporal smoothing to Phase 2 reconstruction output (v2).

    v2 approach: EMA on color + opacity ONLY. Geometry (positions, scale,
    rotation) is passed through untouched from SHARP's output.

    Parameters
    ----------
    reconstruct_dir : Path
        Phase 2 output directory (contains splats/*.ply).
    output_dir : Path or None
        Where to write smoothed files. Default: pipeline/temporal/output/<clip>/
    color_alpha : float
        EMA weight for color smoothing (0=none, 1=full). Default: 0.0 (disabled).
        Set to 0.2-0.3 for gentle temporal blending if needed.
    opacity_alpha : float
        EMA weight for opacity smoothing. Default: 0.0 (disabled).
        Set to 0.1-0.2 for gentle opacity blending if needed.
    opacity_threshold : float
        Shadow suppression opacity deviation threshold.
    dark_threshold : float
        Shadow suppression luminance threshold.
    no_suppress : bool
        Skip shadow blob suppression.
    convert : bool
        Whether to convert .ply → .splat after smoothing.

    Returns
    -------
    dict
        Processing summary / log.
    """
    reconstruct_dir = Path(reconstruct_dir).resolve()
    splats_dir = reconstruct_dir / "splats"
    clip_name = reconstruct_dir.name

    # Output directory
    _script_dir = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = _script_dir / "output" / clip_name
    else:
        output_dir = Path(output_dir).resolve()
    out_splats = output_dir / "splats"
    out_splats.mkdir(parents=True, exist_ok=True)

    # Find PLY files
    ply_files = sorted(splats_dir.glob("frame_*.ply"))
    if not ply_files:
        print(f"✗  No .ply files found in {splats_dir}", file=sys.stderr)
        sys.exit(1)

    n_frames = len(ply_files)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 3 — Temporal Consistency (v2)")
    print(f"  Clip:          {clip_name}")
    print(f"  Frames:        {n_frames}")
    print(f"  Color EMA α:   {color_alpha}")
    print(f"  Opacity EMA α: {opacity_alpha}")
    print(f"  Geometry:      FROZEN (pass-through from SHARP)")
    print(f"  Shadow suppress: {'OFF' if no_suppress else f'ON (opacity>{opacity_threshold}, dark<{dark_threshold})'}")
    print(f"  Output:        {out_splats}")
    print(f"{'='*60}")

    t_start = time.time()

    # ── Load all PLY files into memory ──
    print(f"\n▶  Loading {n_frames} PLY files...", end=" ", flush=True)
    t_load = time.time()
    all_vertices: list[np.ndarray] = []
    all_metadata: list[bytes] = []
    for ply_path in ply_files:
        v, m = read_ply(ply_path)
        all_vertices.append(v)
        all_metadata.append(m)
    print(f"✔ ({time.time() - t_load:.1f}s)")

    # ── Step 1: EMA smoothing on color + opacity ──
    print(f"\n▶  Applying EMA smoothing (color α={color_alpha}, opacity α={opacity_alpha})...")
    smoothed_frames = smooth_attributes_ema(all_vertices, color_alpha, opacity_alpha)
    print(f"   ✔ {n_frames} frames smoothed")

    # ── Step 2: Shadow blob suppression ──
    log_entries: list[dict] = []
    total_suppressed = 0
    sup_window = 3  # half-window for shadow detection context

    if not no_suppress:
        print(f"\n▶  Shadow blob suppression...")
        for i in range(n_frames):
            frame_name = ply_files[i].name

            # Use ORIGINAL frames for shadow detection context
            sup_lo = max(0, i - sup_window)
            sup_hi = min(n_frames, i + sup_window + 1)
            sup_frames = all_vertices[sup_lo:sup_hi]
            sup_center = i - sup_lo

            _, suppressed_count = suppress_shadows(
                sup_frames, sup_center, opacity_threshold, dark_threshold
            )

            # Apply suppression: if a blob was suppressed in the original,
            # clamp the smoothed frame's opacity to the temporal median too
            if suppressed_count > 0:
                suppressed_result, _ = suppress_shadows(
                    sup_frames, sup_center, opacity_threshold, dark_threshold
                )
                smoothed_frames[i]["opacity"] = suppressed_result["opacity"]

            total_suppressed += suppressed_count
            log_entries.append({
                "frame": frame_name,
                "suppressed_blobs": suppressed_count,
            })

            sup_str = f", {suppressed_count} blobs" if suppressed_count > 0 else ""
            print(f"  [{i+1}/{n_frames}] {frame_name} ✔{sup_str}")
    else:
        for i in range(n_frames):
            log_entries.append({"frame": ply_files[i].name, "suppressed_blobs": 0})

    # ── Write smoothed PLY files ──
    print(f"\n▶  Writing {n_frames} smoothed PLY files...")
    for i in range(n_frames):
        out_ply = out_splats / ply_files[i].name
        write_ply(out_ply, smoothed_frames[i], all_metadata[i])

    # ── Convert .ply → .splat ──
    if convert:
        print(f"\n{'─'*60}")
        print(f"▶  Converting {n_frames} .ply files → .splat")
        print(f"{'─'*60}")

        for j, ply_path in enumerate(sorted(out_splats.glob("frame_*.ply"))):
            splat_path = out_splats / f"{ply_path.stem}.splat"
            n = _convert_ply_to_splat(ply_path, splat_path)
            print(f"  [{j+1}/{n_frames}] {ply_path.name} → .splat ✔ ({n:,} gaussians)")

    # ── Write manifest ──
    splat_files = sorted(out_splats.glob("frame_*.splat"))
    manifest = [f.name for f in splat_files]
    (out_splats / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Summary ──
    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "reconstruct_dir": str(reconstruct_dir),
        "output_dir": str(output_dir),
        "settings": {
            "version": 2,
            "color_alpha": color_alpha,
            "opacity_alpha": opacity_alpha,
            "geometry": "frozen",
            "opacity_threshold": opacity_threshold,
            "dark_threshold": dark_threshold,
            "shadow_suppression": not no_suppress,
        },
        "results": {
            "total_frames": n_frames,
            "total_blobs_suppressed": total_suppressed,
            "splat_files": len(manifest),
        },
        "frames": log_entries,
        "total_seconds": total_time,
    }

    log_path = output_dir / "temporal_log.json"
    log_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 3 complete in {total_time}s")
    print(f"     {n_frames} frames smoothed → {out_splats}")
    print(f"     {total_suppressed} shadow blobs suppressed total")
    print(f"     {len(manifest)} .splat files → manifest.json")
    print(f"     Log → {log_path}")
    print(f"{'='*60}")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 3: Temporal consistency smoothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (contains splats/*.ply)")
    parser.add_argument("--color-alpha", type=float, default=0.0,
                        help="EMA weight for color smoothing (default: 0.0 = off)")
    parser.add_argument("--opacity-alpha", type=float, default=0.0,
                        help="EMA weight for opacity smoothing (default: 0.0 = off)")
    parser.add_argument("--opacity-threshold", type=float, default=0.3,
                        help="Shadow suppression opacity threshold (default: 0.3)")
    parser.add_argument("--dark-threshold", type=float, default=0.15,
                        help="Shadow suppression dark threshold (default: 0.15)")
    parser.add_argument("--no-suppress", action="store_true",
                        help="Disable shadow blob suppression")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip .ply → .splat conversion")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    temporal_smooth(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        color_alpha=args.color_alpha,
        opacity_alpha=args.opacity_alpha,
        opacity_threshold=args.opacity_threshold,
        dark_threshold=args.dark_threshold,
        no_suppress=args.no_suppress,
        convert=not args.no_convert,
    )


if __name__ == "__main__":
    main()
