#!/usr/bin/env python3
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Step 1: Validate Index-Based Correspondence
=================================================================
SHARP produces a fixed grid of 1,179,648 Gaussians per frame. The
temporal pipeline assumes that Gaussian[i] in frame N corresponds to
Gaussian[i] in frame N±1 (same pixel grid position).

This script validates that assumption by measuring per-index deltas
across consecutive frame pairs for every Gaussian attribute.

If the assumption holds:
  - Static camera clips (testclip): tiny position deltas (scene unchanged)
  - Moving camera clips (testclip2): larger position deltas (camera moved),
    but CONSISTENT deltas within local neighborhoods (rigid motion)
  - Gaussian count: identical across all frames (1,179,648)

If the assumption is WRONG:
  - Random-looking deltas with no spatial coherence
  - We'd need Paper 1's tracking instead

Usage:
    python validate_correspondence.py <reconstruct_output_dir>

    # Validate testclip2:
    python validate_correspondence.py ../reconstruct/output/testclip2

Output:
    Prints per-pair statistics and a summary JSON to:
    pipeline/compress/output/<clip_name>/correspondence_report.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Reuse the shared PLY I/O from the temporal steps
_SCRIPT_DIR = Path(__file__).resolve().parent
_TEMPORAL_DIR = _SCRIPT_DIR.parent / "temporal"
sys.path.insert(0, str(_TEMPORAL_DIR))
from steps import read_ply, GAUSSIAN_DTYPE


C0 = 0.28209479177387814


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _compute_pair_stats(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> dict:
    """
    Compute per-index delta statistics between two consecutive frames.

    Returns a dict with statistics for each attribute group.
    """
    n = len(frame_a)
    assert len(frame_b) == n, f"Frame size mismatch: {len(frame_a)} vs {len(frame_b)}"

    # Position deltas
    dx = frame_b["x"] - frame_a["x"]
    dy = frame_b["y"] - frame_a["y"]
    dz = frame_b["z"] - frame_a["z"]
    pos_delta = np.sqrt(dx**2 + dy**2 + dz**2)

    # Color deltas (SH DC → linear RGB)
    dr = frame_b["f_dc_0"] - frame_a["f_dc_0"]
    dg = frame_b["f_dc_1"] - frame_a["f_dc_1"]
    db = frame_b["f_dc_2"] - frame_a["f_dc_2"]
    color_delta = np.sqrt(dr**2 + dg**2 + db**2)

    # Opacity deltas (sigmoid space)
    opa_a = _sigmoid(frame_a["opacity"])
    opa_b = _sigmoid(frame_b["opacity"])
    opacity_delta = np.abs(opa_b - opa_a)

    # Scale deltas (log space — raw values from SHARP)
    ds0 = frame_b["scale_0"] - frame_a["scale_0"]
    ds1 = frame_b["scale_1"] - frame_a["scale_1"]
    ds2 = frame_b["scale_2"] - frame_a["scale_2"]
    scale_delta = np.sqrt(ds0**2 + ds1**2 + ds2**2)

    # Rotation deltas (quaternion dot product → angular distance)
    q_a = np.column_stack([frame_a["rot_0"], frame_a["rot_1"],
                           frame_a["rot_2"], frame_a["rot_3"]])
    q_b = np.column_stack([frame_b["rot_0"], frame_b["rot_1"],
                           frame_b["rot_2"], frame_b["rot_3"]])
    # Normalize
    q_a = q_a / (np.linalg.norm(q_a, axis=1, keepdims=True) + 1e-8)
    q_b = q_b / (np.linalg.norm(q_b, axis=1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q_a * q_b, axis=1))
    dot = np.clip(dot, 0, 1)
    rot_delta_rad = 2 * np.arccos(dot)  # angular distance in radians

    def _stats(arr):
        return {
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "position_L2": _stats(pos_delta),
        "color_SH_L2": _stats(color_delta),
        "opacity_sigmoid": _stats(opacity_delta),
        "scale_log_L2": _stats(scale_delta),
        "rotation_rad": _stats(rot_delta_rad),
    }


def _spatial_coherence_check(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    n_samples: int = 10000,
    neighborhood: int = 50,
) -> dict:
    """
    Check whether nearby Gaussians have similar deltas (spatial coherence).

    If SHARP preserves grid correspondence, neighboring Gaussians should
    have similar position deltas (they're imaging nearby pixels).
    If the ordering is random, neighboring indices would have uncorrelated deltas.

    We sample random Gaussians, compute their delta, then compute the
    delta variance within a ±neighborhood window around them.
    Low intra-neighborhood variance = spatially coherent = correspondence holds.
    """
    n = len(frame_a)
    dx = frame_b["x"] - frame_a["x"]
    dy = frame_b["y"] - frame_a["y"]
    dz = frame_b["z"] - frame_a["z"]

    rng = np.random.default_rng(42)
    sample_indices = rng.integers(neighborhood, n - neighborhood, size=n_samples)

    local_variances = []
    for idx in sample_indices:
        lo = idx - neighborhood
        hi = idx + neighborhood + 1
        local_dx = dx[lo:hi]
        local_dy = dy[lo:hi]
        local_dz = dz[lo:hi]
        # Variance of the delta vectors in this local window
        var = np.var(local_dx) + np.var(local_dy) + np.var(local_dz)
        local_variances.append(var)

    local_var = np.array(local_variances)
    global_var = np.var(dx) + np.var(dy) + np.var(dz)

    # Ratio: if local variance << global variance, deltas are spatially coherent
    median_ratio = float(np.median(local_var) / (global_var + 1e-10))

    return {
        "global_position_variance": float(global_var),
        "median_local_variance": float(np.median(local_var)),
        "coherence_ratio": median_ratio,
        "interpretation": (
            "STRONG coherence (grid correspondence confirmed)"
            if median_ratio < 0.1
            else "MODERATE coherence (correspondence likely holds)"
            if median_ratio < 0.5
            else "WEAK coherence (correspondence may be unreliable)"
            if median_ratio < 0.9
            else "NO coherence (indices appear shuffled — need tracking)"
        ),
    }


def validate_correspondence(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    max_pairs: int | None = None,
) -> dict:
    """
    Validate index-based Gaussian correspondence across a frame sequence.
    """
    reconstruct_dir = Path(reconstruct_dir).resolve()
    splats_dir = reconstruct_dir / "splats"
    clip_name = reconstruct_dir.name

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(splats_dir.glob("frame_*.ply"))
    if len(ply_files) < 2:
        print(f"✗  Need at least 2 PLY files, found {len(ply_files)}", file=sys.stderr)
        sys.exit(1)

    n_pairs = len(ply_files) - 1
    if max_pairs and max_pairs < n_pairs:
        n_pairs = max_pairs

    print(f"\n{'='*60}")
    print(f"  LucidPast — Correspondence Validation")
    print(f"  Clip:      {clip_name}")
    print(f"  Frames:    {len(ply_files)}")
    print(f"  Pairs:     {n_pairs}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}")

    t_start = time.time()

    # Validate Gaussian counts are identical
    print(f"\n▶  Checking Gaussian counts...")
    counts = []
    for ply_path in ply_files:
        v, _ = read_ply(ply_path)
        counts.append(len(v))
    unique_counts = set(counts)
    count_ok = len(unique_counts) == 1
    print(f"   Unique counts: {unique_counts}")
    print(f"   All identical: {'✔ YES' if count_ok else '✗ NO — PROBLEM!'}")

    if not count_ok:
        print(f"\n   ⚠  Gaussian counts vary across frames. Index correspondence")
        print(f"      cannot work. Counts: {counts}")
        # Still continue to gather data

    # Pairwise analysis
    print(f"\n▶  Analyzing {n_pairs} consecutive frame pairs...")
    pair_results = []
    prev_frame = None
    prev_path = None

    for i, ply_path in enumerate(ply_files[:n_pairs + 1]):
        frame, _ = read_ply(ply_path)

        if prev_frame is not None:
            pair_name = f"{prev_path.stem} → {ply_path.stem}"
            print(f"  [{i}/{n_pairs}] {pair_name}...", end=" ", flush=True)

            stats = _compute_pair_stats(prev_frame, frame)
            coherence = _spatial_coherence_check(prev_frame, frame)

            pos_med = stats["position_L2"]["median"]
            col_med = stats["color_SH_L2"]["median"]
            coh = coherence["coherence_ratio"]

            pair_results.append({
                "pair": pair_name,
                "stats": stats,
                "coherence": coherence,
            })

            print(f"✔  pos_Δ={pos_med:.5f}  color_Δ={col_med:.4f}  coherence={coh:.4f}")

        prev_frame = frame
        prev_path = ply_path

    # Aggregate statistics across all pairs
    all_pos_medians = [p["stats"]["position_L2"]["median"] for p in pair_results]
    all_color_medians = [p["stats"]["color_SH_L2"]["median"] for p in pair_results]
    all_coherence = [p["coherence"]["coherence_ratio"] for p in pair_results]

    summary = {
        "clip_name": clip_name,
        "n_frames": len(ply_files),
        "n_pairs": n_pairs,
        "gaussian_count": counts[0] if count_ok else None,
        "counts_identical": count_ok,
        "all_counts": counts,

        "position_delta_medians": {
            "min": float(np.min(all_pos_medians)),
            "median": float(np.median(all_pos_medians)),
            "max": float(np.max(all_pos_medians)),
            "mean": float(np.mean(all_pos_medians)),
        },
        "color_delta_medians": {
            "min": float(np.min(all_color_medians)),
            "median": float(np.median(all_color_medians)),
            "max": float(np.max(all_color_medians)),
            "mean": float(np.mean(all_color_medians)),
        },
        "coherence_ratios": {
            "min": float(np.min(all_coherence)),
            "median": float(np.median(all_coherence)),
            "max": float(np.max(all_coherence)),
            "mean": float(np.mean(all_coherence)),
        },

        "verdict": (
            "CONFIRMED" if count_ok and np.median(all_coherence) < 0.5
            else "LIKELY" if count_ok and np.median(all_coherence) < 0.9
            else "REJECTED"
        ),

        "pairs": pair_results,
        "total_seconds": round(time.time() - t_start, 2),
    }

    # Print summary
    verdict = summary["verdict"]
    verdict_icon = "✅" if verdict == "CONFIRMED" else "⚠️" if verdict == "LIKELY" else "❌"

    print(f"\n{'='*60}")
    print(f"  {verdict_icon} Correspondence Verdict: {verdict}")
    print(f"{'='*60}")
    print(f"  Gaussian count:    {counts[0]:,} (identical: {count_ok})")
    print(f"  Position Δ median: {np.median(all_pos_medians):.6f} (range: {np.min(all_pos_medians):.6f} — {np.max(all_pos_medians):.6f})")
    print(f"  Color Δ median:    {np.median(all_color_medians):.6f}")
    print(f"  Coherence ratio:   {np.median(all_coherence):.4f} (lower = more coherent)")
    print(f"  Time:              {summary['total_seconds']}s")
    print(f"{'='*60}")

    if verdict == "CONFIRMED":
        print(f"\n  → Index-based correspondence holds. Polynomial fitting can proceed.")
        print(f"    Position deltas are spatially coherent — nearby Gaussians move")
        print(f"    similarly, consistent with a fixed image-grid mapping.")
    elif verdict == "LIKELY":
        print(f"\n  → Correspondence likely holds but coherence is moderate.")
        print(f"    Polynomial fitting should work but may need higher degree.")
    else:
        print(f"\n  → Correspondence REJECTED. SHARP appears to shuffle Gaussian order.")
        print(f"    Paper 1 (Dynamic 3DGS) tracking would be needed.")

    # Write report
    report_path = output_dir / "correspondence_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Report → {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate index-based Gaussian correspondence across frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (contains splats/*.ply)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs to analyze (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    validate_correspondence(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()

