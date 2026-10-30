#!/usr/bin/env python3
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Step 3: TRBF Temporal Opacity
=====================================================
Fit a Temporal Radial Basis Function (TRBF) to each Gaussian's opacity
trajectory across frames.

    σ_i(t) = σ_i^peak · exp(-τ_i · (t - μ_i)²)

Where:
    σ_i^peak  = the Gaussian's peak opacity (when fully visible)
    μ_i       = temporal center (when the Gaussian is most visible)
    τ_i       = temporal precision (1/width²; higher = more transient)

This replaces the heuristic shadow suppression from Phase 3 with a
principled, data-driven approach. Gaussians that exist stably across
all frames get τ ≈ 0 (always visible). Transient artifacts get high τ
(visible only briefly).

Reference: Paper 3 (Spacetime Gaussians, Li et al. CVPR 2024)
    σ_i(t) = σ_i^s · exp(-s_i^τ · |t - μ_i^τ|²)

Usage:
    python fit_trbf.py <reconstruct_output_dir> [options]

Output:
    pipeline/compress/output/<clip_name>/
    ├── trbf_result.npz    ← TRBF parameters per Gaussian
    └── trbf_report.json   ← classification stats
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_trbf(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    transient_threshold: float = 0.5,
    fps: float = 8.0,
) -> dict:
    """
    Fit TRBF temporal opacity to each Gaussian across a frame sequence.

    For each Gaussian index i, we observe its opacity across N frames:
        o_i(t_0), o_i(t_1), ..., o_i(t_{N-1})

    We fit three parameters:
        peak_i  = max opacity (in sigmoid space)
        mu_i    = temporal center (weighted mean time)
        tau_i   = temporal precision (measures how transient)

    Parameters
    ----------
    reconstruct_dir : Path
        Phase 2 output directory.
    output_dir : Path or None
        Output directory.
    transient_threshold : float
        τ threshold above which a Gaussian is classified as "transient".
        Default 0.5 corresponds to a Gaussian visible for ~2 seconds.
    fps : float
        Frame rate for time values.

    Returns
    -------
    dict : Summary with classification counts.
    """
    reconstruct_dir = Path(reconstruct_dir).resolve()
    splats_dir = reconstruct_dir / "splats"
    clip_name = reconstruct_dir.name

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(splats_dir.glob("frame_*.ply"))
    n_frames = len(ply_files)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 4 — TRBF Temporal Opacity Fitting")
    print(f"  Clip:      {clip_name}")
    print(f"  Frames:    {n_frames}")
    print(f"  Transient threshold: τ > {transient_threshold}")
    print(f"{'='*60}")

    t_start = time.time()

    # ── Load all opacity values ──
    print(f"\n▶  Loading opacity trajectories...", end=" ", flush=True)
    t_load = time.time()

    frame_indices = []
    opacity_frames = []

    for ply_path in ply_files:
        v, _ = read_ply(ply_path)
        idx = int(ply_path.stem.split("_")[1])
        frame_indices.append(idx)
        # Convert to sigmoid space for fitting (SHARP stores logits)
        opa_sigmoid = _sigmoid(v["opacity"]).astype(np.float32)
        opacity_frames.append(opa_sigmoid)

    n_gaussians = len(opacity_frames[0])
    print(f"✔ {n_frames} frames, {n_gaussians:,} gaussians ({time.time() - t_load:.1f}s)")

    # ── Time values ──
    t_values = (np.array(frame_indices, dtype=np.float32) - frame_indices[0]) / fps
    t_center_global = float(np.mean(t_values))
    t_duration = float(t_values[-1] - t_values[0])

    # Stack into (N_gaussians, N_frames)
    opa_matrix = np.column_stack(opacity_frames)  # each col is a frame → wrong shape
    # Actually: each opacity_frames[t] is (N_gaussians,), so stack along axis=1
    opa_matrix = np.stack(opacity_frames, axis=1)  # (N_gaussians, N_frames)

    # ── Fit TRBF parameters ──
    print(f"\n▶  Fitting TRBF parameters...", end=" ", flush=True)
    t_fit = time.time()

    # Peak opacity: maximum across all frames
    peak = np.max(opa_matrix, axis=1)  # (N_gaussians,)

    # Temporal center: opacity-weighted mean time
    # μ_i = Σ_t [ o_i(t) · t ] / Σ_t [ o_i(t) ]
    opa_sum = np.sum(opa_matrix, axis=1, keepdims=True)  # (N, 1)
    opa_sum = np.maximum(opa_sum, 1e-8)  # avoid division by zero
    t_broadcast = t_values[np.newaxis, :]  # (1, T)
    mu = np.sum(opa_matrix * t_broadcast, axis=1, keepdims=True) / opa_sum  # (N, 1)
    mu = mu.squeeze(1)  # (N,)

    # Temporal precision τ: fit from variance of opacity-weighted time
    # Var_i = Σ_t [ o_i(t) · (t - μ_i)² ] / Σ_t [ o_i(t) ]
    dt = t_broadcast - mu[:, np.newaxis]  # (N, T)
    weighted_var = np.sum(opa_matrix * dt**2, axis=1) / opa_sum.squeeze(1)  # (N,)

    # τ = 1 / (2 · variance)  [from Gaussian: exp(-τ·(t-μ)²) has variance 1/(2τ)]
    # Clamp variance to avoid division by zero and extreme tau
    weighted_var = np.maximum(weighted_var, 1e-6)
    tau = 1.0 / (2.0 * weighted_var)  # (N,)

    # Gaussians with nearly constant opacity across all frames should have τ ≈ 0
    # Detect: if opacity std is very small, force τ = 0 (always visible)
    opa_std = np.std(opa_matrix, axis=1)  # (N,)
    stable_mask = opa_std < 0.05  # opacity varies < 5% across frames
    tau[stable_mask] = 0.0
    # Reset mu to center for stable Gaussians (arbitrary — they're always visible)
    mu[stable_mask] = t_center_global

    # Clamp extreme tau (transient Gaussians that appear in only 1-2 frames)
    tau = np.clip(tau, 0.0, 100.0).astype(np.float32)
    mu = mu.astype(np.float32)
    peak = peak.astype(np.float32)

    fit_time = time.time() - t_fit
    print(f"✔ ({fit_time:.1f}s)")

    # ── Classify Gaussians ──
    n_stable = int(np.sum(tau == 0))
    n_persistent = int(np.sum((tau > 0) & (tau <= transient_threshold)))
    n_transient = int(np.sum(tau > transient_threshold))

    # Low-opacity Gaussians (effectively invisible regardless of TRBF)
    low_opacity_mask = peak < 0.05
    n_low_opacity = int(np.sum(low_opacity_mask))

    print(f"\n{'─'*60}")
    print(f"  Gaussian Classification:")
    print(f"  Stable (τ=0):       {n_stable:>10,}  ({100*n_stable/n_gaussians:.1f}%)")
    print(f"  Persistent (τ≤{transient_threshold}):  {n_persistent:>10,}  ({100*n_persistent/n_gaussians:.1f}%)")
    print(f"  Transient (τ>{transient_threshold}):   {n_transient:>10,}  ({100*n_transient/n_gaussians:.1f}%)")
    print(f"  Low opacity (<5%):  {n_low_opacity:>10,}  ({100*n_low_opacity/n_gaussians:.1f}%)")
    print(f"{'─'*60}")

    # ── Compute reconstruction quality ──
    # Reconstruct opacity at each frame and measure error vs original
    print(f"\n▶  Computing reconstruction quality...", end=" ", flush=True)
    dt_all = t_values[np.newaxis, :] - mu[:, np.newaxis]  # (N, T)
    reconstructed = peak[:, np.newaxis] * np.exp(-tau[:, np.newaxis] * dt_all**2)  # (N, T)
    error = np.abs(reconstructed - opa_matrix)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    print(f"✔")
    print(f"   Opacity RMSE — median: {np.median(rmse):.6f}, "
          f"p95: {np.percentile(rmse, 95):.6f}, "
          f"max: {np.max(rmse):.6f}")

    # ── Save ──
    print(f"\n▶  Saving TRBF results...", end=" ", flush=True)
    np.savez_compressed(
        output_dir / "trbf_result.npz",
        peak=peak,
        mu=mu,
        tau=tau,
        rmse=rmse,
        opa_std=opa_std,
    )
    npz_size = (output_dir / "trbf_result.npz").stat().st_size
    print(f"✔ ({npz_size / 1024 / 1024:.1f} MB)")

    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "n_frames": n_frames,
        "n_gaussians": n_gaussians,
        "t_duration_sec": t_duration,
        "classification": {
            "stable": n_stable,
            "persistent": n_persistent,
            "transient": n_transient,
            "low_opacity": n_low_opacity,
            "transient_threshold": transient_threshold,
        },
        "reconstruction": {
            "opacity_rmse_median": float(np.median(rmse)),
            "opacity_rmse_p95": float(np.percentile(rmse, 95)),
            "opacity_rmse_max": float(np.max(rmse)),
        },
        "tau_stats": {
            "min": float(np.min(tau)),
            "median": float(np.median(tau)),
            "mean": float(np.mean(tau)),
            "p95": float(np.percentile(tau, 95)),
            "max": float(np.max(tau)),
        },
        "total_seconds": total_time,
    }

    report_path = output_dir / "trbf_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ TRBF fitting complete in {total_time}s")
    print(f"     {n_stable:,} stable + {n_persistent:,} persistent + {n_transient:,} transient")
    print(f"     Report → {report_path}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 4: Fit TRBF temporal opacity.",
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (contains splats/*.ply)")
    parser.add_argument("--transient-threshold", type=float, default=0.5,
                        help="τ threshold for transient classification (default: 0.5)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Frame rate (default: 8.0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    fit_trbf(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        transient_threshold=args.transient_threshold,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

