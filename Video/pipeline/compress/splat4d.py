#!/usr/bin/env python3
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Phase 4 — Step 4: .splat4d Format Writer & Reader
==============================================================
Encode polynomial coefficients + TRBF into a single binary file
that can be loaded by the web viewer or AR renderer.

Binary format (all little-endian):
─────────────────────────────────────────────────────────────────
HEADER (64 bytes)
  magic           char[4]     "SP4D"
  version         uint16      1
  n_gaussians     uint32      number of Gaussians
  n_frames        uint32      original frame count (for reference)
  fps             float32     playback frame rate
  t_start         float32     start time (seconds)
  t_end           float32     end time (seconds)
  t_center        float32     polynomial center time
  pos_degree      uint8       position polynomial degree (0-3)
  color_degree    uint8       color polynomial degree (0-3)
  opacity_degree  uint8       opacity polynomial degree (0-3)
  scale_degree    uint8       scale polynomial degree (0-3)
  rot_degree      uint8       rotation polynomial degree (0-3)
  flags           uint8       bit 0: has_trbf
  reserved        byte[26]    zero-padded

CANONICAL GAUSSIANS (n × 56 bytes) — same layout as SHARP PLY
  Per Gaussian:
    x, y, z          float32 × 3    position at t_center
    f_dc_0/1/2       float32 × 3    color SH DC at t_center
    opacity          float32        opacity logit at t_center
    scale_0/1/2      float32 × 3    log-scale at t_center
    rot_0/1/2/3      float32 × 4    quaternion at t_center

POSITION MOTION (n × 3 × pos_degree × 4 bytes)
  Per Gaussian: bx,by,bz [, cx,cy,cz [, dx,dy,dz]]
  Degree 0 → 0 bytes (constant — canonical only)
  Degree 1 → 36 bytes (linear velocity)
  Degree 2 → 72 bytes (+ acceleration)
  Degree 3 → 108 bytes (+ jerk)

COLOR MOTION (n × 3 × color_degree × 4 bytes)
  Same layout as position motion, for f_dc channels.

OPACITY MOTION (n × 1 × opacity_degree × 4 bytes)

SCALE MOTION (n × 3 × scale_degree × 4 bytes)

ROTATION MOTION (n × 4 × rot_degree × 4 bytes)

TRBF (n × 8 bytes, if flags.has_trbf)
  Per Gaussian:
    tau       float32    temporal precision (0 = always visible)
    mu        float32    temporal center (redundant with t_center but
                         allows per-Gaussian temporal offset)
─────────────────────────────────────────────────────────────────

Usage:
    python splat4d.py <reconstruct_output_dir> [options]

    # Build .splat4d from fit results:
    python splat4d.py ../reconstruct/output/testclip2

    # Build without TRBF:
    python splat4d.py ../reconstruct/output/testclip2 --no-trbf
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TEMPORAL_DIR = _SCRIPT_DIR.parent / "temporal"
sys.path.insert(0, str(_TEMPORAL_DIR))
from steps import read_ply, GAUSSIAN_DTYPE


MAGIC = b"SP4D"
VERSION = 1
HEADER_SIZE = 64


def _build_canonical_gaussians(
    fit_data: dict,
    splats_dir: Path,
) -> np.ndarray:
    """
    Build canonical Gaussians at t_center from polynomial coefficients.

    The constant term (coeffs[..., 0]) IS the canonical value at t=0
    (relative to t_center). So coefficients[..., 0] directly give us
    the Gaussian attributes at t_center.
    """
    n = int(fit_data["n_gaussians"])

    canonical = np.zeros(n, dtype=GAUSSIAN_DTYPE)

    # Position: constant term of position polynomial
    pos_c = fit_data["pos_coeffs"]  # (N, 3, degree+1)
    canonical["x"] = pos_c[:, 0, 0]
    canonical["y"] = pos_c[:, 1, 0]
    canonical["z"] = pos_c[:, 2, 0]

    # Color: constant term
    col_c = fit_data["color_coeffs"]  # (N, 3, degree+1)
    canonical["f_dc_0"] = col_c[:, 0, 0]
    canonical["f_dc_1"] = col_c[:, 1, 0]
    canonical["f_dc_2"] = col_c[:, 2, 0]

    # Opacity: constant term
    opa_c = fit_data["opacity_coeffs"]  # (N, 1, degree+1)
    canonical["opacity"] = opa_c[:, 0, 0]

    # Scale: constant term
    scl_c = fit_data["scale_coeffs"]  # (N, 3, degree+1)
    canonical["scale_0"] = scl_c[:, 0, 0]
    canonical["scale_1"] = scl_c[:, 1, 0]
    canonical["scale_2"] = scl_c[:, 2, 0]

    # Rotation: constant term
    rot_c = fit_data["rot_coeffs"]  # (N, 4, degree+1)
    canonical["rot_0"] = rot_c[:, 0, 0]
    canonical["rot_1"] = rot_c[:, 1, 0]
    canonical["rot_2"] = rot_c[:, 2, 0]
    canonical["rot_3"] = rot_c[:, 3, 0]

    return canonical


def _extract_motion_coeffs(coeffs: np.ndarray) -> np.ndarray | None:
    """
    Extract motion coefficients (all terms after the constant).

    coeffs: (N, C, degree+1)
    Returns: (N, C, degree) or None if degree == 0
    """
    degree = coeffs.shape[2] - 1
    if degree == 0:
        return None
    return coeffs[:, :, 1:].astype(np.float32)  # (N, C, degree)


def write_splat4d(
    output_path: Path,
    fit_data: dict,
    trbf_data: dict | None,
    splats_dir: Path,
) -> int:
    """
    Write a .splat4d binary file.

    Returns file size in bytes.
    """
    n = int(fit_data["n_gaussians"])
    t_values = fit_data["t_values"]
    fps = float(fit_data["fps"])
    t_center = float(fit_data["t_center"])

    pos_degree = fit_data["pos_coeffs"].shape[2] - 1
    color_degree = fit_data["color_coeffs"].shape[2] - 1
    opacity_degree = fit_data["opacity_coeffs"].shape[2] - 1
    scale_degree = fit_data["scale_coeffs"].shape[2] - 1
    rot_degree = fit_data["rot_coeffs"].shape[2] - 1

    has_trbf = trbf_data is not None
    flags = (1 if has_trbf else 0)

    # Build canonical
    canonical = _build_canonical_gaussians(fit_data, splats_dir)

    with open(output_path, "wb") as f:
        # ── Header (64 bytes) ──
        header = struct.pack(
            "<4sHIIffffBBBBBB",
            MAGIC,
            VERSION,
            n,
            len(fit_data["frame_names"]),
            fps,
            float(t_values[0]),
            float(t_values[-1]),
            t_center,
            pos_degree,
            color_degree,
            opacity_degree,
            scale_degree,
            rot_degree,
            flags,
        )
        header += b'\x00' * (HEADER_SIZE - len(header))  # pad to 64
        f.write(header)

        # ── Canonical Gaussians ──
        f.write(canonical.tobytes())

        # ── Motion coefficients ──
        # Position motion: (N, 3, pos_degree) — excludes constant term
        pos_motion = _extract_motion_coeffs(fit_data["pos_coeffs"])
        if pos_motion is not None:
            f.write(pos_motion.tobytes())

        # Color motion
        col_motion = _extract_motion_coeffs(fit_data["color_coeffs"])
        if col_motion is not None:
            f.write(col_motion.tobytes())

        # Opacity motion
        opa_motion = _extract_motion_coeffs(fit_data["opacity_coeffs"])
        if opa_motion is not None:
            f.write(opa_motion.tobytes())

        # Scale motion
        scl_motion = _extract_motion_coeffs(fit_data["scale_coeffs"])
        if scl_motion is not None:
            f.write(scl_motion.tobytes())

        # Rotation motion
        rot_motion = _extract_motion_coeffs(fit_data["rot_coeffs"])
        if rot_motion is not None:
            f.write(rot_motion.tobytes())

        # ── TRBF ──
        if has_trbf:
            trbf_packed = np.column_stack([
                trbf_data["tau"],
                trbf_data["mu"],
            ]).astype(np.float32)
            f.write(trbf_packed.tobytes())

    return output_path.stat().st_size


def read_splat4d(path: Path) -> dict:
    """
    Read a .splat4d file and return all data as a dict.

    Used by the viewer and for validation.
    """
    with open(path, "rb") as f:
        # ── Header ──
        raw_header = f.read(HEADER_SIZE)
        magic = raw_header[:4]
        assert magic == MAGIC, f"Invalid magic: {magic}"

        (version, n, n_frames, fps, t_start, t_end, t_center,
         pos_deg, col_deg, opa_deg, scl_deg, rot_deg, flags) = struct.unpack(
            "<HIIffffBBBBBB", raw_header[4:36]
        )

        has_trbf = bool(flags & 1)

        # ── Canonical Gaussians ──
        canonical = np.frombuffer(f.read(n * 56), dtype=GAUSSIAN_DTYPE, count=n).copy()

        # ── Motion coefficients ──
        def _read_motion(n_gaussians, n_channels, degree):
            if degree == 0:
                return None
            nbytes = n_gaussians * n_channels * degree * 4
            return np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(n_gaussians, n_channels, degree).copy()

        pos_motion = _read_motion(n, 3, pos_deg)
        col_motion = _read_motion(n, 3, col_deg)
        opa_motion = _read_motion(n, 1, opa_deg)
        scl_motion = _read_motion(n, 3, scl_deg)
        rot_motion = _read_motion(n, 4, rot_deg)

        # ── TRBF ──
        trbf = None
        if has_trbf:
            trbf_raw = np.frombuffer(f.read(n * 8), dtype=np.float32).reshape(n, 2).copy()
            trbf = {"tau": trbf_raw[:, 0], "mu": trbf_raw[:, 1]}

    return {
        "version": version,
        "n_gaussians": n,
        "n_frames": n_frames,
        "fps": fps,
        "t_start": t_start,
        "t_end": t_end,
        "t_center": t_center,
        "degrees": {
            "position": pos_deg,
            "color": col_deg,
            "opacity": opa_deg,
            "scale": scl_deg,
            "rotation": rot_deg,
        },
        "has_trbf": has_trbf,
        "canonical": canonical,
        "pos_motion": pos_motion,
        "col_motion": col_motion,
        "opa_motion": opa_motion,
        "scl_motion": scl_motion,
        "rot_motion": rot_motion,
        "trbf": trbf,
    }


def evaluate_at_time(data: dict, t: float) -> np.ndarray:
    """
    Evaluate the 4D representation at a specific time t.

    Returns a structured array of Gaussians with all attributes computed
    at time t via polynomial evaluation.
    """
    n = data["n_gaussians"]
    dt = t - data["t_center"]
    result = data["canonical"].copy()

    def _eval_poly(canonical_vals, motion, n_channels, dt):
        """Evaluate polynomial: canonical + motion[0]*dt + motion[1]*dt² + ..."""
        if motion is None:
            return canonical_vals
        degree = motion.shape[2]
        out = canonical_vals.copy()
        dt_power = dt
        for k in range(degree):
            out += motion[:, :, k] * dt_power
            dt_power *= dt
        return out

    # Position
    pos_canonical = np.column_stack([result["x"], result["y"], result["z"]])
    pos_eval = _eval_poly(pos_canonical, data["pos_motion"], 3, dt)
    result["x"] = pos_eval[:, 0]
    result["y"] = pos_eval[:, 1]
    result["z"] = pos_eval[:, 2]

    # Color
    col_canonical = np.column_stack([result["f_dc_0"], result["f_dc_1"], result["f_dc_2"]])
    col_eval = _eval_poly(col_canonical, data["col_motion"], 3, dt)
    result["f_dc_0"] = col_eval[:, 0]
    result["f_dc_1"] = col_eval[:, 1]
    result["f_dc_2"] = col_eval[:, 2]

    # Opacity
    opa_canonical = result["opacity"].reshape(-1, 1)
    opa_eval = _eval_poly(opa_canonical, data["opa_motion"], 1, dt)
    result["opacity"] = opa_eval[:, 0]

    # Scale
    scl_canonical = np.column_stack([result["scale_0"], result["scale_1"], result["scale_2"]])
    scl_eval = _eval_poly(scl_canonical, data["scl_motion"], 3, dt)
    result["scale_0"] = scl_eval[:, 0]
    result["scale_1"] = scl_eval[:, 1]
    result["scale_2"] = scl_eval[:, 2]

    # Rotation
    rot_canonical = np.column_stack([result["rot_0"], result["rot_1"], result["rot_2"], result["rot_3"]])
    rot_eval = _eval_poly(rot_canonical, data["rot_motion"], 4, dt)
    # Re-normalize quaternions
    norms = np.linalg.norm(rot_eval, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    rot_eval = rot_eval / norms
    result["rot_0"] = rot_eval[:, 0]
    result["rot_1"] = rot_eval[:, 1]
    result["rot_2"] = rot_eval[:, 2]
    result["rot_3"] = rot_eval[:, 3]

    # TRBF: modulate opacity
    if data["has_trbf"] and data["trbf"] is not None:
        tau = data["trbf"]["tau"]
        mu = data["trbf"]["mu"]
        dt_trbf = t - mu
        # In sigmoid space: multiply by temporal Gaussian
        # But opacity is stored as logit, so convert, modulate, convert back
        sigmoid_opa = 1.0 / (1.0 + np.exp(-np.clip(result["opacity"], -50, 50)))
        trbf_weight = np.exp(-tau * dt_trbf**2)
        modulated = sigmoid_opa * trbf_weight
        # Convert back to logit
        modulated = np.clip(modulated, 1e-7, 1 - 1e-7)
        result["opacity"] = np.log(modulated / (1 - modulated)).astype(np.float32)

    return result


# ─── CLI: Build .splat4d from existing fit results ────────────────────────────

def build_splat4d(
    reconstruct_dir: Path,
    output_dir: Path | None = None,
    no_trbf: bool = False,
) -> dict:
    """Build a .splat4d file from fit_result.npz and trbf_result.npz."""
    reconstruct_dir = Path(reconstruct_dir).resolve()
    clip_name = reconstruct_dir.name
    splats_dir = reconstruct_dir / "splats"

    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    output_dir = Path(output_dir).resolve()

    fit_path = output_dir / "fit_result.npz"
    trbf_path = output_dir / "trbf_result.npz"

    if not fit_path.exists():
        print(f"✗  fit_result.npz not found in {output_dir}", file=sys.stderr)
        print(f"   Run fit_motion.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 4 — Build .splat4d")
    print(f"  Clip:  {clip_name}")
    print(f"  TRBF:  {'OFF' if no_trbf else 'ON' if trbf_path.exists() else 'OFF (not found)'}")
    print(f"{'='*60}")

    t_start = time.time()

    # Load fit results
    print(f"\n▶  Loading fit results...", end=" ", flush=True)
    fit_npz = np.load(fit_path)
    fit_data = {
        "n_gaussians": int(fit_npz["n_gaussians"]),
        "frame_names": list(fit_npz["frame_names"]),
        "t_values": fit_npz["t_values"],
        "t_center": float(fit_npz["t_center"]),
        "fps": float(fit_npz["fps"]),
        "pos_coeffs": fit_npz["pos_coeffs"],
        "color_coeffs": fit_npz["color_coeffs"],
        "opacity_coeffs": fit_npz["opacity_coeffs"],
        "scale_coeffs": fit_npz["scale_coeffs"],
        "rot_coeffs": fit_npz["rot_coeffs"],
    }
    print(f"✔ ({fit_data['n_gaussians']:,} gaussians)")

    # Load TRBF
    trbf_data = None
    if not no_trbf and trbf_path.exists():
        print(f"▶  Loading TRBF results...", end=" ", flush=True)
        trbf_npz = np.load(trbf_path)
        trbf_data = {
            "tau": trbf_npz["tau"],
            "mu": trbf_npz["mu"],
        }
        print(f"✔")

    # Write .splat4d
    out_path = output_dir / f"{clip_name}.splat4d"
    print(f"\n▶  Writing {out_path.name}...", end=" ", flush=True)
    file_size = write_splat4d(out_path, fit_data, trbf_data, splats_dir)
    print(f"✔ ({file_size / 1024 / 1024:.1f} MB)")

    # ── Validate by reading it back ──
    print(f"\n▶  Validating (read-back test)...", end=" ", flush=True)
    readback = read_splat4d(out_path)
    assert readback["n_gaussians"] == fit_data["n_gaussians"]
    assert readback["degrees"]["position"] == fit_data["pos_coeffs"].shape[2] - 1
    # Test evaluation at t_center
    frame_at_center = evaluate_at_time(readback, readback["t_center"])
    assert len(frame_at_center) == fit_data["n_gaussians"]
    print(f"✔")

    # ── Compute original vs compressed size ──
    n_frames = len(fit_data["frame_names"])
    n_gaussians = fit_data["n_gaussians"]
    original_splat_size = n_frames * n_gaussians * 32  # .splat format: 32 bytes
    original_ply_size = n_frames * n_gaussians * 56

    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "splat4d_path": str(out_path),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / 1024 / 1024, 2),
        "n_gaussians": n_gaussians,
        "n_original_frames": n_frames,
        "degrees": {
            "position": fit_data["pos_coeffs"].shape[2] - 1,
            "color": fit_data["color_coeffs"].shape[2] - 1,
            "opacity": fit_data["opacity_coeffs"].shape[2] - 1,
            "scale": fit_data["scale_coeffs"].shape[2] - 1,
            "rotation": fit_data["rot_coeffs"].shape[2] - 1,
        },
        "has_trbf": trbf_data is not None,
        "compression_vs_splat": round(original_splat_size / file_size, 2),
        "compression_vs_ply": round(original_ply_size / file_size, 2),
        "total_seconds": total_time,
    }

    report_path = output_dir / "splat4d_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ .splat4d built in {total_time}s")
    print(f"     {out_path.name}: {file_size / 1024 / 1024:.1f} MB")
    print(f"     vs .splat flipbook: {original_splat_size / 1024 / 1024:.0f} MB → {summary['compression_vs_splat']}× compression")
    print(f"     vs .ply flipbook:   {original_ply_size / 1024 / 1024:.0f} MB → {summary['compression_vs_ply']}× compression")
    print(f"     Report → {report_path}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 4: Build .splat4d from polynomial fit results.",
    )
    parser.add_argument("reconstruct_dir", type=Path,
                        help="Phase 2 output directory (used for clip name)")
    parser.add_argument("--no-trbf", action="store_true",
                        help="Exclude TRBF data from output")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (must contain fit_result.npz)")
    args = parser.parse_args()

    build_splat4d(
        reconstruct_dir=args.reconstruct_dir,
        output_dir=args.output_dir,
        no_trbf=args.no_trbf,
    )


if __name__ == "__main__":
    main()
