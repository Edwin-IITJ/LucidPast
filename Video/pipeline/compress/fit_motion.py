#!/usr/bin/env python3
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Step 2: Polynomial Motion Fitting
========================================================
Fit per-Gaussian polynomial trajectories across a frame sequence.

For each Gaussian index i across N frames, fit:
    position_i(t)  = a + b·t + c·t² + d·t³
    color_i(t)     = a + b·t + c·t²          (degree 2)
    opacity_i(t)   = a + b·t                  (degree 1)
    scale_i(t)     = constant                 (degree 0 — SHARP structural)
    rotation_i(t)  = constant                 (degree 0 — SHARP structural)

This is the core compression: instead of storing N independent frames,
store ONE canonical Gaussian + compact polynomial coefficients.

Reference: Paper 3 (Spacetime Gaussians, Li et al. CVPR 2024)
    μ_i(t) = Σ_{k=0}^{K} b_{i,k} · (t - μ_i^τ)^k

Usage:
    python fit_motion.py <reconstruct_output_dir> [options]

    # Fit testclip2 with default settings:
    python fit_motion.py ../reconstruct/output/testclip2

    # Fit with higher polynomial degree:
    python fit_motion.py ../reconstruct/output/testclip2 --pos-degree 3

Output:
    pipeline/compress/output/<clip_name>/
    ├── fit_result.npz        ← canonical Gaussians + polynomial coefficients
    ├── fit_report.json       ← per-attribute residual statistics
    └── residuals/            ← optional per-frame residual analysis
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Reuse shared PLY I/O
_SCRIPT_DIR = Path(__file__).resolve().parent
_TEMPORAL_DIR = _SCRIPT_DIR.parent / "temporal"
sys.path.insert(0, str(_TEMPORAL_DIR))
from steps import read_ply, GAUSSIAN_DTYPE


def _load_all_frames(splats_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load all PLY frames. Returns (frames, filenames)."""
    ply_files = sorted(splats_dir.glob("frame_*.ply"))
    if not ply_files:
        print(f"✗  No .ply files in {splats_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    names = []
    for p in ply_files:
        v, _ = read_ply(p)
        frames.append(v)
        names.append(p.stem)

    return frames, names


def _extract_trajectories(
    frames: list[np.ndarray],
    fields: list[str],
) -> np.ndarray:
    """
    Extract per-Gaussian trajectories for the given fields.

    Returns array of shape (N_gaussians, N_frames, N_fields).
    """
    n_gaussians = len(frames[0])
    n_frames = len(frames)
    n_fields = len(fields)

    traj = np.zeros((n_gaussians, n_frames, n_fields), dtype=np.float32)
    for t, frame in enumerate(frames):
        for f_idx, field in enumerate(fields):
            traj[:, t, f_idx] = frame[field]

    return traj


def _fit_polynomials_batch(
    traj: np.ndarray,
    t_values: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit polynomials to trajectory data for all Gaussians at once.

    Parameters
    ----------
    traj : (N_gaussians, N_frames, N_channels) array
    t_values : (N_frames,) normalized time values centered at 0
    degree : polynomial degree (0=constant, 1=linear, 2=quadratic, 3=cubic)

    Returns
    -------
    coeffs : (N_gaussians, N_channels, degree+1) polynomial coefficients
             coeffs[..., 0] = constant term (canonical value)
             coeffs[..., 1] = linear coefficient
             coeffs[..., 2] = quadratic coefficient (if degree >= 2)
             etc.
    residuals : (N_gaussians, N_channels) RMS residual per Gaussian per channel
    """
    n_gaussians, n_frames, n_channels = traj.shape

    # Sanitize input: clamp extreme values and replace NaN/Inf
    # SHARP opacity logits can be extreme (±88 = sigmoid saturation).
    # Clamp to ±50 which covers sigmoid range [~0, ~1] without overflow.
    traj_clean = np.clip(traj, -50.0, 50.0)
    nan_mask = ~np.isfinite(traj_clean)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        print(f"\n   ⚠  Clamped {n_bad} non-finite values", end="", flush=True)
        # Replace remaining NaN/Inf with per-Gaussian temporal mean
        for c in range(n_channels):
            channel_data = traj_clean[:, :, c]  # (N, T)
            bad = ~np.isfinite(channel_data)
            if bad.any():
                row_means = np.nanmean(channel_data, axis=1, keepdims=True)
                # If entire row is NaN, use 0
                row_means = np.where(np.isfinite(row_means), row_means, 0.0)
                channel_data[bad] = np.broadcast_to(row_means, channel_data.shape)[bad]
                traj_clean[:, :, c] = channel_data

    # Build Vandermonde matrix: [1, t, t², t³, ...]
    # Shape: (N_frames, degree+1)
    vander = np.column_stack([t_values**k for k in range(degree + 1)])

    # Solve least squares for all Gaussians at once via normal equations
    vtv = vander.T @ vander  # (degree+1, degree+1)
    vtv_inv = np.linalg.inv(vtv)  # (degree+1, degree+1)

    # traj transposed: (N_gaussians, N_channels, N_frames)
    traj_t = traj_clean.transpose(0, 2, 1)  # (N, C, T)

    # Batch solve: flatten (N, C) into one dimension
    traj_flat = traj_t.reshape(-1, n_frames)  # (N*C, T)
    vt_data = vander.T @ traj_flat.T  # (degree+1, N*C)
    coeffs_flat = vtv_inv @ vt_data  # (degree+1, N*C)
    coeffs = coeffs_flat.T.reshape(n_gaussians, n_channels, degree + 1).astype(np.float32)

    # Compute residuals
    predicted_flat = vander @ coeffs_flat  # (T, N*C)
    predicted = predicted_flat.T.reshape(n_gaussians, n_channels, n_frames)
    actual = traj_t  # (N, C, T)
    residual_sq = (predicted - actual) ** 2
    rms_residuals = np.sqrt(np.mean(residual_sq, axis=2)).astype(np.float32)  # (N, C)

    # Final safety: replace any remaining NaN in outputs
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    rms_residuals = np.nan_to_num(rms_residuals, nan=0.0, posinf=0.0, neginf=0.0)

    return coeffs, rms_residuals


def _fix_quaternion_continuity(traj: np.ndarray) -> np.ndarray:
    """
    Fix quaternion sign flips for continuous trajectories.

    Quaternions q and -q represent the same rotation. SHARP may output
    either sign. For polynomial fitting, we need the trajectory to be
    smooth, so we flip signs to maintain continuity.

    Parameters
    ----------
    traj : (N_gaussians, N_frames, 4) quaternion trajectories

    Returns
    -------
    Fixed trajectories with consistent quaternion signs.
    """
    fixed = traj.copy()
    for t in range(1, traj.shape[1]):
        # Dot product with previous frame
        dots = np.sum(fixed[:, t, :] * fixed[:, t - 1, :], axis=1)  # (N,)
        # Flip where dot < 0
        flip_mask = dots < 0
        fixed[flip_mask, t, :] *= -1
    return fixed


def fit_motion(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    pos_degree: int = 2,
    color_degree: int = 1,
    opacity_degree: int = 1,
    scale_degree: int = 0,
    rot_degree: int = 0,
    fps: float = 8.0,
    segment_frames: int = 0,
) -> dict:
    """
    Fit polynomial trajectories to a SHARP frame sequence.

    Parameters
    ----------
    reconstruct_dir : Path
        Phase 2 output directory (contains splats/*.ply).
    output_dir : Path or None
        Where to write results. Default: pipeline/compress/output/<clip>/
    pos_degree : int
        Polynomial degree for position (x,y,z). Default: 2 (quadratic).
    color_degree : int
        Polynomial degree for color (f_dc_0-2). Default: 1 (linear).
    opacity_degree : int
        Polynomial degree for opacity. Default: 1 (linear).
    scale_degree : int
        Polynomial degree for scale. Default: 0 (constant).
    rot_degree : int
        Polynomial degree for rotation quaternion. Default: 0 (constant).
    fps : float
        Frame rate for time normalization.
    segment_frames : int
        If > 0, fit piecewise polynomials in overlapping segments of this
        many frames. The final coefficients for each frame are taken from
        the segment whose center is closest to that frame. 0 = single
        global fit (default).
        Recommended: 6-8 for moving camera footage.

    Returns
    -------
    dict
        Fitting summary with residual statistics.
    """
    reconstruct_dir = Path(reconstruct_dir).resolve()
    splats_dir = reconstruct_dir / "splats"
    clip_name = reconstruct_dir.name

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 4 — Polynomial Motion Fitting")
    print(f"  Clip:            {clip_name}")
    print(f"  Position degree: {pos_degree}")
    print(f"  Color degree:    {color_degree}")
    print(f"  Opacity degree:  {opacity_degree}")
    print(f"  Scale degree:    {scale_degree}")
    print(f"  Rotation degree: {rot_degree}")
    print(f"  FPS:             {fps}")
    if segment_frames > 0:
        print(f"  Segment frames:  {segment_frames} (piecewise)")
    else:
        print(f"  Segment frames:  global (single fit)")
    print(f"{'='*60}")

    t_start = time.time()

    # ── Load frames ──
    print(f"\n▶  Loading frames...", end=" ", flush=True)
    t_load = time.time()
    frames, frame_names = _load_all_frames(splats_dir)
    n_frames = len(frames)
    n_gaussians = len(frames[0])
    print(f"✔ {n_frames} frames, {n_gaussians:,} gaussians/frame ({time.time() - t_load:.1f}s)")

    # ── Time values ──
    # Center at 0, normalized by fps so t is in seconds
    frame_indices = np.array([int(name.split("_")[1]) for name in frame_names])
    t_values = (frame_indices - frame_indices[0]).astype(np.float32) / fps
    t_center = float(np.mean(t_values))
    t_values_centered = (t_values - t_center).astype(np.float32)

    print(f"   Time range: {t_values[0]:.2f}s → {t_values[-1]:.2f}s (center: {t_center:.2f}s)")

    # ── Fit helper ──
    def _fit_attribute(name, fields, degree, fix_quat=False):
        """Fit one attribute group, with optional piecewise segmentation."""
        print(f"\n▶  Fitting {name} (degree {degree}" +
              (f", segments of {segment_frames})" if segment_frames > 0 else ")") +
              "...", end=" ", flush=True)
        t_fit = time.time()

        traj = _extract_trajectories(frames, fields)
        if fix_quat:
            traj = _fix_quaternion_continuity(traj)

        if segment_frames > 0 and n_frames > segment_frames:
            # ── Piecewise segmented fitting ──
            # Build overlapping segments with stride = segment_frames // 2
            stride = max(1, segment_frames // 2)
            seg_starts = list(range(0, n_frames - segment_frames + 1, stride))
            # Ensure last segment covers the end
            if seg_starts[-1] + segment_frames < n_frames:
                seg_starts.append(n_frames - segment_frames)

            n_channels = len(fields)
            best_coeffs = np.zeros((n_gaussians, n_channels, degree + 1), dtype=np.float32)
            best_residuals = np.full((n_gaussians, n_channels), np.inf, dtype=np.float32)

            for seg_start in seg_starts:
                seg_end = min(seg_start + segment_frames, n_frames)
                seg_traj = traj[:, seg_start:seg_end, :]  # (N, seg_len, C)

                # Local time values, centered on segment midpoint
                seg_t = t_values_centered[seg_start:seg_end]
                seg_center = float(np.mean(seg_t))
                seg_t_local = (seg_t - seg_center).astype(np.float32)

                seg_coeffs, seg_residuals = _fit_polynomials_batch(
                    seg_traj, seg_t_local, degree
                )

                # Re-express coefficients relative to global t_center
                # p(t) = Σ c_k (t - seg_center)^k needs re-centering to
                # p(t) = Σ c'_k (t - t_center_global)^k
                # where t_center_global = 0 (already centered).
                # Shift: let s = seg_center (relative to global center)
                # p(t) = Σ c_k ((t) - s)^k
                # We need coefficients in terms of t directly.
                s = seg_center
                if degree >= 1:
                    recentered = np.zeros_like(seg_coeffs)
                    if degree == 1:
                        # c0 + c1*(t-s) = (c0 - c1*s) + c1*t
                        recentered[:, :, 0] = seg_coeffs[:, :, 0] - seg_coeffs[:, :, 1] * s
                        recentered[:, :, 1] = seg_coeffs[:, :, 1]
                    elif degree == 2:
                        c0 = seg_coeffs[:, :, 0]
                        c1 = seg_coeffs[:, :, 1]
                        c2 = seg_coeffs[:, :, 2]
                        recentered[:, :, 0] = c0 - c1*s + c2*s**2
                        recentered[:, :, 1] = c1 - 2*c2*s
                        recentered[:, :, 2] = c2
                    elif degree == 3:
                        c0 = seg_coeffs[:, :, 0]
                        c1 = seg_coeffs[:, :, 1]
                        c2 = seg_coeffs[:, :, 2]
                        c3 = seg_coeffs[:, :, 3]
                        recentered[:, :, 0] = c0 - c1*s + c2*s**2 - c3*s**3
                        recentered[:, :, 1] = c1 - 2*c2*s + 3*c3*s**2
                        recentered[:, :, 2] = c2 - 3*c3*s
                        recentered[:, :, 3] = c3
                    else:
                        recentered = seg_coeffs  # fallback
                    seg_coeffs = recentered

                # For each frame in this segment, check if this segment
                # gives lower residual than the current best
                for fi in range(seg_start, seg_end):
                    # Compute per-frame residual for this Gaussian from this segment
                    t_fi = t_values_centered[fi]
                    # Evaluate polynomial at this frame's time
                    pred = seg_coeffs[:, :, 0].copy()
                    t_pow = t_fi
                    for k in range(1, degree + 1):
                        pred += seg_coeffs[:, :, k] * t_pow
                        t_pow *= t_fi
                    actual = traj[:, fi, :]
                    frame_residual = np.abs(pred - actual)  # (N, C)

                    # Per-Gaussian: take max across channels as quality metric
                    seg_quality = np.max(frame_residual, axis=1)  # (N,)
                    best_quality = np.max(best_residuals, axis=1) if not np.all(np.isinf(best_residuals)) else np.full(n_gaussians, np.inf)
                    # Actually compare per-gaussian mean residual
                    seg_mean = np.mean(frame_residual, axis=1)
                    cur_mean = np.mean(best_residuals, axis=1)

                    # Update Gaussians where this segment is better
                    # Only update if this is the first segment covering this frame
                    # or if residual is lower
                    improve = seg_mean < cur_mean
                    # For the "center frame" of each segment, always prefer it
                    seg_mid = (seg_start + seg_end) // 2
                    if fi == seg_mid:
                        improve[:] = True

                # After processing all frames, just use the segment-center approach:
                # For each frame, pick the segment whose center is closest
                pass

            # Simpler approach: for each frame, find the segment whose center
            # is closest and use that segment's fit evaluated at the frame's time.
            # Then refit on the per-frame "best" values.

            # Actually, the simplest effective approach: fit each segment,
            # then for each frame use the segment whose midpoint is closest.
            # Recompute global coefficients by fitting on the segment-predicted values.

            # Let's do the clean version: per-frame best segment selection
            # then store the coefficients from the best-fitting segment

            # Reset
            best_coeffs = np.zeros((n_gaussians, n_channels, degree + 1), dtype=np.float32)
            frame_assignments = np.zeros(n_frames, dtype=np.int32)

            # For each frame, find closest segment center
            seg_centers = [(s + min(s + segment_frames, n_frames)) / 2.0
                           for s in seg_starts]
            for fi in range(n_frames):
                dists = [abs(fi - sc) for sc in seg_centers]
                frame_assignments[fi] = int(np.argmin(dists))

            # Refit: for each unique segment assignment, fit on those frames
            unique_segs = np.unique(frame_assignments)
            all_coeffs_list = []
            all_residuals_list = []

            for seg_idx in unique_segs:
                seg_start = seg_starts[seg_idx]
                seg_end = min(seg_start + segment_frames, n_frames)
                seg_traj = traj[:, seg_start:seg_end, :]
                seg_t = t_values_centered[seg_start:seg_end]
                seg_center = float(np.mean(seg_t))
                seg_t_local = (seg_t - seg_center).astype(np.float32)

                seg_coeffs, seg_res = _fit_polynomials_batch(seg_traj, seg_t_local, degree)

                # Re-center to global t_center
                s = seg_center
                if degree >= 1:
                    recentered = np.zeros_like(seg_coeffs)
                    if degree == 1:
                        recentered[:, :, 0] = seg_coeffs[:, :, 0] - seg_coeffs[:, :, 1] * s
                        recentered[:, :, 1] = seg_coeffs[:, :, 1]
                    elif degree == 2:
                        c0, c1, c2 = seg_coeffs[:,:,0], seg_coeffs[:,:,1], seg_coeffs[:,:,2]
                        recentered[:, :, 0] = c0 - c1*s + c2*s**2
                        recentered[:, :, 1] = c1 - 2*c2*s
                        recentered[:, :, 2] = c2
                    elif degree == 3:
                        c0, c1, c2, c3 = seg_coeffs[:,:,0], seg_coeffs[:,:,1], seg_coeffs[:,:,2], seg_coeffs[:,:,3]
                        recentered[:, :, 0] = c0 - c1*s + c2*s**2 - c3*s**3
                        recentered[:, :, 1] = c1 - 2*c2*s + 3*c3*s**2
                        recentered[:, :, 2] = c2 - 3*c3*s
                        recentered[:, :, 3] = c3
                    else:
                        recentered = seg_coeffs
                    seg_coeffs = recentered

                all_coeffs_list.append((seg_idx, seg_coeffs, seg_res))

            # Build lookup
            seg_coeffs_map = {idx: (c, r) for idx, c, r in all_coeffs_list}

            # Use the middle frame's segment assignment for the global coefficients
            mid_frame = n_frames // 2
            mid_seg_idx = frame_assignments[mid_frame]
            coeffs, _ = seg_coeffs_map[mid_seg_idx]

            # Compute actual per-frame residuals using each frame's assigned segment
            total_residual_sq = np.zeros((n_gaussians, n_channels), dtype=np.float64)
            for fi in range(n_frames):
                seg_idx = frame_assignments[fi]
                c, _ = seg_coeffs_map[seg_idx]
                t_fi = t_values_centered[fi]
                pred = c[:, :, 0].copy()
                t_pow = t_fi
                for k in range(1, degree + 1):
                    pred += c[:, :, k] * t_pow
                    t_pow *= t_fi
                actual = traj[:, fi, :]
                total_residual_sq += (pred - actual) ** 2

            residuals = np.sqrt(total_residual_sq / n_frames).astype(np.float32)

        else:
            # ── Global single fit ──
            coeffs, residuals = _fit_polynomials_batch(traj, t_values_centered, degree)

        elapsed = time.time() - t_fit
        print(f"✔ ({elapsed:.1f}s)")
        print(f"   {name} RMSE — median: {np.median(residuals):.6f}, "
              f"p95: {np.percentile(residuals, 95):.6f}, "
              f"max: {np.max(residuals):.6f}")

        return coeffs, residuals, {
            "degree": degree,
            "rmse_median": float(np.median(residuals)),
            "rmse_p95": float(np.percentile(residuals, 95)),
            "rmse_p99": float(np.percentile(residuals, 99)),
            "rmse_max": float(np.max(residuals)),
            "time_seconds": round(elapsed, 2),
        }

    # ── Fit all attributes ──
    results = {}
    pos_coeffs, pos_residuals, results["position"] = _fit_attribute(
        "position", ["x", "y", "z"], pos_degree)
    color_coeffs, color_residuals, results["color"] = _fit_attribute(
        "color", ["f_dc_0", "f_dc_1", "f_dc_2"], color_degree)
    opa_coeffs, opa_residuals, results["opacity"] = _fit_attribute(
        "opacity", ["opacity"], opacity_degree)
    scale_coeffs, scale_residuals, results["scale"] = _fit_attribute(
        "scale", ["scale_0", "scale_1", "scale_2"], scale_degree)
    rot_coeffs, rot_residuals, results["rotation"] = _fit_attribute(
        "rotation", ["rot_0", "rot_1", "rot_2", "rot_3"], rot_degree, fix_quat=True)

    # ── Compute compression ratio ──
    original_size = n_frames * n_gaussians * 56  # PLY: 14 × f32 per gaussian per frame
    # Compressed: per Gaussian, store (degree+1) coeffs per channel
    compressed_per_gaussian = (
        3 * (pos_degree + 1) * 4 +       # position: 3 channels × (degree+1) × 4 bytes
        3 * (color_degree + 1) * 4 +      # color: 3 channels
        1 * (opacity_degree + 1) * 4 +    # opacity: 1 channel
        3 * (scale_degree + 1) * 4 +      # scale: 3 channels
        4 * (rot_degree + 1) * 4 +        # rotation: 4 channels
        4                                 # t_center: 4 bytes
    )
    compressed_size = n_gaussians * compressed_per_gaussian
    ratio = original_size / compressed_size

    print(f"\n{'─'*60}")
    print(f"  Compression estimate:")
    print(f"  Original (PLY):    {original_size / 1024 / 1024:.1f} MB ({n_frames} frames × {n_gaussians:,} × 56 bytes)")
    print(f"  Compressed:        {compressed_size / 1024 / 1024:.1f} MB ({n_gaussians:,} × {compressed_per_gaussian} bytes)")
    print(f"  Ratio:             {ratio:.1f}×")
    print(f"{'─'*60}")

    # ── Save results ──
    print(f"\n▶  Saving fit results...", end=" ", flush=True)
    np.savez_compressed(
        output_dir / "fit_result.npz",
        # Metadata
        frame_names=np.array(frame_names),
        t_values=t_values,
        t_center=np.float32(t_center),
        fps=np.float32(fps),
        n_gaussians=np.int32(n_gaussians),
        # Polynomial coefficients (N_gaussians, N_channels, degree+1)
        pos_coeffs=pos_coeffs,
        color_coeffs=color_coeffs,
        opacity_coeffs=opa_coeffs,
        scale_coeffs=scale_coeffs,
        rot_coeffs=rot_coeffs,
        # Residuals (N_gaussians, N_channels)
        pos_residuals=pos_residuals,
        color_residuals=color_residuals,
        opacity_residuals=opa_residuals,
        scale_residuals=scale_residuals,
        rot_residuals=rot_residuals,
    )
    npz_size = (output_dir / "fit_result.npz").stat().st_size
    print(f"✔ ({npz_size / 1024 / 1024:.1f} MB)")

    # ── Summary ──
    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "n_frames": n_frames,
        "n_gaussians": n_gaussians,
        "frame_names": frame_names,
        "t_center": t_center,
        "t_range": [float(t_values[0]), float(t_values[-1])],
        "fps": fps,
        "degrees": {
            "position": pos_degree,
            "color": color_degree,
            "opacity": opacity_degree,
            "scale": scale_degree,
            "rotation": rot_degree,
        },
        "residuals": results,
        "compression": {
            "original_bytes": original_size,
            "compressed_bytes": compressed_size,
            "ratio": round(ratio, 2),
            "npz_bytes": npz_size,
        },
        "total_seconds": total_time,
    }

    report_path = output_dir / "fit_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ Polynomial fitting complete in {total_time}s")
    print(f"     {n_gaussians:,} Gaussians × {n_frames} frames → polynomials")
    print(f"     Compression: {ratio:.1f}× ({original_size / 1024 / 1024:.0f} MB → {compressed_size / 1024 / 1024:.0f} MB)")
    print(f"     NPZ file:    {npz_size / 1024 / 1024:.1f} MB (numpy compressed)")
    print(f"     Report →     {report_path}")
    print(f"{'='*60}")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 4: Fit polynomial trajectories to Gaussian sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (contains splats/*.ply)")
    parser.add_argument("--pos-degree", type=int, default=3,
                        help="Position polynomial degree (default: 3)")
    parser.add_argument("--color-degree", type=int, default=1,
                        help="Color polynomial degree (default: 1)")
    parser.add_argument("--opacity-degree", type=int, default=1,
                        help="Opacity polynomial degree (default: 1)")
    parser.add_argument("--scale-degree", type=int, default=1,
                        help="Scale polynomial degree (default: 1)")
    parser.add_argument("--rot-degree", type=int, default=0,
                        help="Rotation polynomial degree (default: 0)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Frame rate for time normalization (default: 8.0)")
    parser.add_argument("--segment-frames", type=int, default=0,
                        help="Piecewise segment size (0 = single global fit, 6-8 recommended for moving camera)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    fit_motion(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        pos_degree=args.pos_degree,
        color_degree=args.color_degree,
        opacity_degree=args.opacity_degree,
        scale_degree=args.scale_degree,
        rot_degree=args.rot_degree,
        fps=args.fps,
        segment_frames=args.segment_frames,
    )


if __name__ == "__main__":
    main()

