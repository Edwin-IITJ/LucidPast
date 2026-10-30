#!/usr/bin/env python3
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Quality Validation
==========================================
Compare .splat4d polynomial reconstruction against original PLY frames.

For each original frame, evaluate the .splat4d at that timestamp and
measure per-Gaussian error across all attributes. This gives us a
ground-truth quality metric for the compression.

Usage:
    python validate_quality.py <reconstruct_output_dir>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TEMPORAL_DIR = _SCRIPT_DIR.parent / "temporal"
sys.path.insert(0, str(_TEMPORAL_DIR))
from steps import read_ply

from splat4d import read_splat4d, evaluate_at_time


C0 = 0.28209479177387814


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def validate_quality(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
) -> dict:
    """Compare .splat4d reconstruction vs original PLY frames."""
    reconstruct_dir = Path(reconstruct_dir).resolve()
    splats_dir = reconstruct_dir / "splats"
    clip_name = reconstruct_dir.name

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()

    splat4d_path = output_dir / f"{clip_name}.splat4d"
    if not splat4d_path.exists():
        print(f"✗  {splat4d_path} not found. Run compress.py first.", file=sys.stderr)
        sys.exit(1)

    ply_files = sorted(splats_dir.glob("frame_*.ply"))

    print(f"\n{'='*60}")
    print(f"  LucidPast — Quality Validation")
    print(f"  Clip:       {clip_name}")
    print(f"  .splat4d:   {splat4d_path.name}")
    print(f"  PLY frames: {len(ply_files)}")
    print(f"{'='*60}")

    t_start = time.time()

    # Load .splat4d
    print(f"\n▶  Loading .splat4d...", end=" ", flush=True)
    data = read_splat4d(splat4d_path)
    print(f"✔ ({data['n_gaussians']:,} gaussians, deg {data['degrees']})")

    fps = data["fps"]
    frame0_idx = int(ply_files[0].stem.split("_")[1])

    # Per-frame comparison
    frame_results = []

    print(f"\n▶  Comparing {len(ply_files)} frames...")
    for i, ply_path in enumerate(ply_files):
        frame_idx = int(ply_path.stem.split("_")[1])
        t = (frame_idx - frame0_idx) / fps  # time in seconds

        # Original
        original, _ = read_ply(ply_path)

        # Reconstructed from .splat4d
        reconstructed = evaluate_at_time(data, t)

        n = len(original)

        # Position error (L2)
        pos_err = np.sqrt(
            (reconstructed["x"] - original["x"])**2 +
            (reconstructed["y"] - original["y"])**2 +
            (reconstructed["z"] - original["z"])**2
        )

        # Color error (in linear RGB, after SH conversion)
        def _to_rgb(frame):
            r = 0.5 + C0 * frame["f_dc_0"]
            g = 0.5 + C0 * frame["f_dc_1"]
            b = 0.5 + C0 * frame["f_dc_2"]
            return np.column_stack([r, g, b])

        orig_rgb = _to_rgb(original)
        recon_rgb = _to_rgb(reconstructed)
        color_err = np.sqrt(np.sum((recon_rgb - orig_rgb)**2, axis=1))

        # Opacity error (sigmoid space)
        orig_opa = _sigmoid(original["opacity"])
        recon_opa = _sigmoid(reconstructed["opacity"])
        opa_err = np.abs(recon_opa - orig_opa)

        result = {
            "frame": ply_path.stem,
            "time": round(t, 3),
            "position_L2": {
                "median": float(np.median(pos_err)),
                "p95": float(np.percentile(pos_err, 95)),
                "max": float(np.max(pos_err)),
            },
            "color_RGB_L2": {
                "median": float(np.median(color_err)),
                "p95": float(np.percentile(color_err, 95)),
                "max": float(np.max(color_err)),
            },
            "opacity_abs": {
                "median": float(np.median(opa_err)),
                "p95": float(np.percentile(opa_err, 95)),
                "max": float(np.max(opa_err)),
            },
        }
        frame_results.append(result)

        pm = result["position_L2"]["median"]
        cm = result["color_RGB_L2"]["median"]
        om = result["opacity_abs"]["median"]
        print(f"  [{i+1}/{len(ply_files)}] {ply_path.stem} t={t:.2f}s  "
              f"pos={pm:.5f}  col={cm:.5f}  opa={om:.5f}")

    # Aggregate
    all_pos = [r["position_L2"]["median"] for r in frame_results]
    all_col = [r["color_RGB_L2"]["median"] for r in frame_results]
    all_opa = [r["opacity_abs"]["median"] for r in frame_results]

    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "n_frames": len(ply_files),
        "n_gaussians": data["n_gaussians"],
        "degrees": data["degrees"],
        "has_trbf": data["has_trbf"],
        "aggregate": {
            "position_L2_median": {
                "min": float(np.min(all_pos)),
                "median": float(np.median(all_pos)),
                "max": float(np.max(all_pos)),
            },
            "color_RGB_L2_median": {
                "min": float(np.min(all_col)),
                "median": float(np.median(all_col)),
                "max": float(np.max(all_col)),
            },
            "opacity_abs_median": {
                "min": float(np.min(all_opa)),
                "median": float(np.median(all_opa)),
                "max": float(np.max(all_opa)),
            },
        },
        "frames": frame_results,
        "total_seconds": total_time,
    }

    report_path = output_dir / "quality_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  Quality Summary — {clip_name}")
    print(f"{'='*60}")
    print(f"  Position L2 (median across frames): "
          f"min={np.min(all_pos):.5f}, median={np.median(all_pos):.5f}, max={np.max(all_pos):.5f}")
    print(f"  Color RGB L2 (median across frames): "
          f"min={np.min(all_col):.5f}, median={np.median(all_col):.5f}, max={np.max(all_col):.5f}")
    print(f"  Opacity (median across frames):      "
          f"min={np.min(all_opa):.5f}, median={np.median(all_opa):.5f}, max={np.max(all_opa):.5f}")
    print(f"\n  Report → {report_path}")
    print(f"  Time: {total_time}s")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate .splat4d quality against original PLY frames.",
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (must contain .splat4d)")
    args = parser.parse_args()

    validate_quality(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

