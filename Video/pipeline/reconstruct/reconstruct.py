#!/usr/bin/env python3
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Video Pipeline — Phase 2: Reconstruct
==================================================
Production-grade SHARP batch wrapper. Takes the clean frames from Phase 1
and produces per-frame .ply Gaussian Splats, then converts them to the
compact .splat web format.

This is the evolved, production version of the spike's `run_sharp_batch.py`.

Key improvements over the spike:
  - Resume support: skips frames that already have a .ply output
  - Proper progress reporting with ETA
  - Anomaly detection: flags frames with unusual Gaussian counts
  - Configurable via config.yaml or CLI flags
  - Integrated .ply → .splat conversion (no separate step)
  - Detailed JSON log with per-frame timing and error data
  - Shot-aware: reads shots.json from Phase 1 to group outputs

Usage:
    python reconstruct.py <ingest_output_dir> [options]

    # Process Phase 1 output for testclip2:
    python reconstruct.py ../ingest/output/testclip2

    # Process every frame (not every-nth):
    python reconstruct.py ../ingest/output/testclip2 --every-nth 1

    # Resume a previously interrupted run:
    python reconstruct.py ../ingest/output/testclip2
    # (automatically skips frames that already have .ply files)

    # Skip .splat conversion (only produce .ply files):
    python reconstruct.py ../ingest/output/testclip2 --no-convert

Output:
    pipeline/reconstruct/output/<clip_name>/
    ├── splats/              ← per-frame .ply + .splat files
    │   ├── frame_00001.ply
    │   ├── frame_00001.splat
    │   ├── ...
    │   └── manifest.json    ← ordered list of .splat files for the viewer
    └── reconstruct_log.json ← per-frame timing, errors, anomaly flags

VERSIONING NOTE:
    Existing v1 outputs in Video/output/ are NOT touched.
    This pipeline writes to pipeline/reconstruct/output/.
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

# ─── SHARP configuration ─────────────────────────────────────────────────────
# Resolve paths relative to the LucidPast project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent  # Video/pipeline/reconstruct → LucidPast

_DEFAULT_SHARP_EXE = _PROJECT_ROOT / "env_sharp" / "Scripts" / "sharp.exe"
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "ml-sharp" / "sharp_model.pt"
_DEFAULT_TIMEOUT = 300  # seconds per frame


# ─── PLY → .splat conversion ─────────────────────────────────────────────────
# Inlined from phase0-spike/scripts/convert_to_splat.py (vectorized version)

C0 = 0.28209479177387814  # SH0 coefficient


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _read_sharp_ply(ply_path: Path) -> np.ndarray:
    """Read a SHARP PLY file, extracting only the vertex (Gaussian) data."""
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break

        # 14 float32 properties × 4 bytes = 56 bytes per vertex
        vertex_data = f.read(vertex_count * 56)

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    return np.frombuffer(vertex_data, dtype=dtype, count=vertex_count)


def _convert_ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    """Convert a single .ply to .splat format. Returns Gaussian count."""
    vertices = _read_sharp_ply(ply_path)
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


# ─── SHARP execution ─────────────────────────────────────────────────────────

def _build_sharp_cmd(
    input_path: Path,
    output_path: Path,
    sharp_exe: Path,
    checkpoint: Path,
) -> list[str]:
    """Build the SHARP CLI command."""
    return [
        str(sharp_exe), "predict",
        "-i", str(input_path),
        "-o", str(output_path),
        "-c", str(checkpoint),
    ]


def _run_sharp_frame(
    frame_path: Path,
    splats_dir: Path,
    sharp_exe: Path,
    checkpoint: Path,
    timeout: int,
) -> dict:
    """
    Run SHARP on a single frame. Returns a log entry dict.

    SHARP outputs to a subdirectory, so we run it into a temp subdir,
    then move the .ply to the final location and clean up.
    """
    ply_name = f"{frame_path.stem}.ply"
    ply_path = splats_dir / ply_name
    temp_dir = splats_dir / f"_temp_{frame_path.stem}"

    start = time.time()

    try:
        cmd = _build_sharp_cmd(frame_path, temp_dir, sharp_exe, checkpoint)
        subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = round(time.time() - start, 2)

        # Find and move the .ply output
        ply_files = list(temp_dir.glob("*.ply"))
        if ply_files:
            ply_files[0].rename(ply_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"frame": frame_path.name, "ply": ply_name, "ok": True, "seconds": elapsed}
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"frame": frame_path.name, "ok": False, "error": "no .ply output", "seconds": elapsed}

    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"frame": frame_path.name, "ok": False, "error": "timeout", "seconds": timeout}

    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        err = (e.stderr or "")[-500:]
        elapsed = round(time.time() - start, 2)
        return {"frame": frame_path.name, "ok": False, "error": err, "seconds": elapsed}

    except FileNotFoundError:
        return {"frame": frame_path.name, "ok": False,
                "error": f"SHARP executable not found: {sharp_exe}"}


# ─── Main pipeline ───────────────────────────────────────────────────────────

def reconstruct(
    ingest_dir: Path,
    output_dir: Path | None = None,
    every_nth: int = 1,
    sharp_exe: Path | None = None,
    checkpoint: Path | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    convert: bool = True,
) -> dict:
    """
    Run SHARP reconstruction on Phase 1 output frames.

    Parameters
    ----------
    ingest_dir : Path
        Phase 1 output directory (contains frames/ and shots.json).
    output_dir : Path or None
        Where to write .ply/.splat files. Default: pipeline/reconstruct/output/<clip>/
    every_nth : int
        Process every Nth frame (default 1 = all frames).
    sharp_exe : Path or None
        Path to sharp.exe. Default from config.yaml / project layout.
    checkpoint : Path or None
        Path to sharp_model.pt. Default from project layout.
    timeout : int
        Per-frame timeout in seconds.
    convert : bool
        Whether to convert .ply → .splat after reconstruction.

    Returns
    -------
    dict
        Processing summary / log.
    """
    ingest_dir = Path(ingest_dir).resolve()
    frames_dir = ingest_dir / "frames"
    shots_path = ingest_dir / "shots.json"
    clip_name = ingest_dir.name

    # Resolve SHARP paths
    sharp_exe = Path(sharp_exe) if sharp_exe else _DEFAULT_SHARP_EXE
    checkpoint = Path(checkpoint) if checkpoint else _DEFAULT_CHECKPOINT

    # Output directory
    if output_dir is None:
        output_dir = _SCRIPT_DIR / "output" / clip_name
    else:
        output_dir = Path(output_dir).resolve()
    splats_dir = output_dir / "splats"
    splats_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    if not frames_dir.exists():
        print(f"✗  Frames directory not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)
    if not sharp_exe.exists():
        print(f"✗  SHARP executable not found: {sharp_exe}", file=sys.stderr)
        print(f"   Expected at: {sharp_exe}")
        sys.exit(1)
    if not checkpoint.exists():
        print(f"✗  SHARP checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Load shots
    shots = []
    if shots_path.exists():
        shots = json.loads(shots_path.read_text())

    # Collect frames (every_nth)
    all_frames = sorted(frames_dir.glob("frame_*.png"))
    frames = all_frames[::every_nth]

    if not frames:
        print(f"✗  No frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 2 — Reconstruct (SHARP)")
    print(f"  Clip:       {clip_name}")
    print(f"  Frames:     {len(frames)}/{len(all_frames)} (every {every_nth})")
    print(f"  SHARP:      {sharp_exe}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output:     {splats_dir}")
    print(f"  Timeout:    {timeout}s per frame")
    print(f"{'='*60}")

    log_entries: list[dict] = []
    skipped = 0
    succeeded = 0
    failed = 0
    t_start = time.time()

    for i, frame_path in enumerate(frames):
        ply_path = splats_dir / f"{frame_path.stem}.ply"

        # Resume support: skip if .ply already exists
        if ply_path.exists():
            print(f"  [{i+1}/{len(frames)}] {frame_path.name} → already exists, skipping")
            log_entries.append({"frame": frame_path.name, "ply": ply_path.name,
                                "ok": True, "skipped": True})
            skipped += 1
            continue

        # ETA calculation
        if succeeded + failed > 0:
            elapsed_so_far = time.time() - t_start
            avg_per_frame = elapsed_so_far / (succeeded + failed)
            remaining = (len(frames) - i) * avg_per_frame
            eta_str = f" | ETA: {remaining/60:.0f}min"
        else:
            eta_str = ""

        print(f"  [{i+1}/{len(frames)}] {frame_path.name}{eta_str}...", end=" ", flush=True)

        entry = _run_sharp_frame(frame_path, splats_dir, sharp_exe, checkpoint, timeout)
        log_entries.append(entry)

        if entry["ok"]:
            succeeded += 1
            print(f"✔ ({entry['seconds']}s)")
        else:
            failed += 1
            print(f"✗ ({entry.get('error', 'unknown')})")

    # ── Convert .ply → .splat ──
    gaussian_counts: list[int] = []
    if convert:
        ply_files = sorted(splats_dir.glob("frame_*.ply"))
        if ply_files:
            print(f"\n{'─'*60}")
            print(f"▶  Converting {len(ply_files)} .ply files → .splat")
            print(f"{'─'*60}")

            for j, ply_path in enumerate(ply_files):
                splat_path = splats_dir / f"{ply_path.stem}.splat"
                if splat_path.exists():
                    # Count Gaussians from existing .splat file size
                    n = splat_path.stat().st_size // 32
                    gaussian_counts.append(n)
                    continue

                print(f"  [{j+1}/{len(ply_files)}] {ply_path.name} → .splat...", end=" ", flush=True)
                try:
                    n = _convert_ply_to_splat(ply_path, splat_path)
                    gaussian_counts.append(n)
                    print(f"✔ ({n:,} gaussians)")
                except Exception as e:
                    print(f"✗ ({e})")

    # ── Write manifest.json ──
    splat_files = sorted(splats_dir.glob("frame_*.splat"))
    manifest = [f.name for f in splat_files]
    (splats_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Anomaly detection ──
    anomalies = []
    if gaussian_counts:
        median_count = int(np.median(gaussian_counts))
        for j, count in enumerate(gaussian_counts):
            # Flag if Gaussian count deviates > 20% from median
            if abs(count - median_count) / median_count > 0.2:
                anomalies.append({
                    "frame": manifest[j] if j < len(manifest) else f"frame_{j}",
                    "gaussians": count,
                    "median": median_count,
                    "deviation": round((count - median_count) / median_count * 100, 1),
                })
        if anomalies:
            print(f"\n⚠  {len(anomalies)} anomalous frame(s) flagged (Gaussian count deviation > 20%)")
            for a in anomalies:
                print(f"   {a['frame']}: {a['gaussians']:,} gaussians ({a['deviation']:+.1f}% from median {a['median']:,})")

    # ── Summary ──
    total_time = round(time.time() - t_start, 2)

    summary = {
        "clip_name": clip_name,
        "ingest_dir": str(ingest_dir),
        "output_dir": str(output_dir),
        "settings": {
            "every_nth": every_nth,
            "timeout": timeout,
            "convert": convert,
        },
        "results": {
            "total_frames": len(frames),
            "succeeded": succeeded,
            "skipped": skipped,
            "failed": failed,
            "splat_files": len(manifest),
            "gaussian_count_median": int(np.median(gaussian_counts)) if gaussian_counts else None,
        },
        "anomalies": anomalies,
        "frames": log_entries,
        "total_seconds": total_time,
    }

    log_path = output_dir / "reconstruct_log.json"
    log_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 2 complete in {total_time}s")
    print(f"     {succeeded} reconstructed + {skipped} resumed + {failed} failed")
    print(f"     {len(manifest)} .splat files → {splats_dir}")
    if gaussian_counts:
        print(f"     Gaussians/frame: {int(np.median(gaussian_counts)):,} (median)")
    print(f"     Log → {log_path}")
    print(f"{'='*60}")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 2: SHARP reconstruction on Phase 1 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ingest_dir", type=Path,
                        help="Phase 1 output directory (contains frames/ and shots.json)")
    parser.add_argument("--every-nth", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT,
                        help=f"Per-frame timeout in seconds (default: {_DEFAULT_TIMEOUT})")
    parser.add_argument("--sharp-exe", type=Path, default=None,
                        help="Path to sharp.exe (default: auto-detect from project layout)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to sharp_model.pt (default: auto-detect)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip .ply → .splat conversion")
    args = parser.parse_args()

    reconstruct(
        ingest_dir=args.ingest_dir,
        output_dir=args.output_dir,
        every_nth=args.every_nth,
        sharp_exe=args.sharp_exe,
        checkpoint=args.checkpoint,
        timeout=args.timeout,
        convert=not args.no_convert,
    )


if __name__ == "__main__":
    main()
