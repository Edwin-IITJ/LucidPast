#!/usr/bin/env python3
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Video Pipeline — Phase 1: Ingest & Preprocess
=========================================================
Single entry point that takes a raw video file and produces clean,
stabilized, lossless PNG frames ready for SHARP reconstruction (Phase 2).

Usage:
    python ingest.py <video_file> [options]

    # Full pipeline with defaults:
    python ingest.py ../../Video/input/testclip2.mp4

    # Custom settings:
    python ingest.py ../../Video/input/testclip2.mp4 --fps 10 --max-dim 720

    # Skip stabilization (e.g. for intentional camera motion):
    python ingest.py ../../Video/input/testclip2.mp4 --no-stabilize

    # Re-run deflicker + stabilize on existing frames:
    python ingest.py ../../Video/input/testclip2.mp4 --skip-extract

Steps:
    1. Extract — ffmpeg frame extraction (configurable fps + resolution cap)
    2. Detect shots — histogram-based shot cut detection
    3. Deflicker — rolling luminance histogram matching per shot
    4. Stabilize — sparse optical flow jitter correction per shot

Output:
    pipeline/ingest/output/<clip_name>/
    ├── frames/          ← numbered lossless PNGs
    ├── shots.json       ← shot boundary manifest
    └── ingest_log.json  ← processing summary

VERSIONING NOTE:
    Existing outputs in Video/output/testclip/ and Video/output/testclip2/
    are the v1 baseline and must NOT be modified.  This pipeline writes to
    pipeline/ingest/output/ — a completely separate location.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path so steps/ is importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from steps.extract import extract_frames
from steps.detect_shots import detect_shots, save_shots
from steps.deflicker import deflicker
from steps.stabilize import stabilize


def main():
    parser = argparse.ArgumentParser(
        description="LucidPast Phase 1: Ingest & Preprocess a video for SHARP reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--fps", type=int, default=8, help="Extraction frame rate (default: 8)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Cap longer edge in pixels (default: 1024)")
    parser.add_argument("--shot-threshold", type=float, default=0.7,
                        help="Histogram correlation threshold for shot cuts (default: 0.7)")
    parser.add_argument("--deflicker-window", type=int, default=5,
                        help="Deflicker rolling window half-size (default: 5)")
    parser.add_argument("--max-motion", type=float, default=15.0,
                        help="Max motion in px before skipping stabilization (default: 15)")
    parser.add_argument("--no-stabilize", action="store_true",
                        help="Skip the stabilization step entirely")
    parser.add_argument("--no-deflicker", action="store_true",
                        help="Skip the deflicker step entirely")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction (reuse existing frames)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: pipeline/ingest/output/<clip_name>/)")
    args = parser.parse_args()

    video_path = args.video.resolve()
    clip_name = video_path.stem

    # ── Output directory ──
    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = Path(__file__).parent / "output" / clip_name
    frames_dir = out_dir / "frames"
    shots_path = out_dir / "shots.json"
    log_path = out_dir / "ingest_log.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  LucidPast Phase 1 — Ingest & Preprocess")
    print(f"  Video:     {video_path}")
    print(f"  Clip:      {clip_name}")
    print(f"  Output:    {out_dir}")
    print(f"  FPS: {args.fps}  |  Max dim: {args.max_dim}  |  Shot threshold: {args.shot_threshold}")
    print(f"{'='*60}")

    log = {
        "clip_name": clip_name,
        "video_path": str(video_path),
        "output_dir": str(out_dir),
        "settings": {
            "fps": args.fps,
            "max_dim": args.max_dim,
            "shot_threshold": args.shot_threshold,
            "deflicker_window": args.deflicker_window,
            "max_motion_px": args.max_motion,
        },
        "steps": {},
    }

    # ── Step 1: Extract frames ──
    if not args.skip_extract:
        t1 = time.time()
        frame_count = extract_frames(video_path, frames_dir, fps=args.fps, max_dim=args.max_dim)
        log["steps"]["extract"] = {
            "frame_count": frame_count,
            "seconds": round(time.time() - t1, 2),
        }
    else:
        existing = list(frames_dir.glob("frame_*.png"))
        frame_count = len(existing)
        print(f"\n⟳  Skipping extraction — using {frame_count} existing frames in {frames_dir}")
        log["steps"]["extract"] = {"skipped": True, "existing_frames": frame_count}

    # ── Step 2: Detect shots ──
    t2 = time.time()
    shots = detect_shots(frames_dir, threshold=args.shot_threshold)
    save_shots(shots, shots_path)
    log["steps"]["detect_shots"] = {
        "shot_count": len(shots),
        "shots": shots,
        "seconds": round(time.time() - t2, 2),
    }

    # ── Step 3: Deflicker ──
    if not args.no_deflicker:
        t3 = time.time()
        deflicker_count = deflicker(frames_dir, shots, window=args.deflicker_window)
        log["steps"]["deflicker"] = {
            "frames_processed": deflicker_count,
            "seconds": round(time.time() - t3, 2),
        }
    else:
        print(f"\n⟳  Skipping deflicker (--no-deflicker)")
        log["steps"]["deflicker"] = {"skipped": True}

    # ── Step 4: Stabilize ──
    if not args.no_stabilize:
        t4 = time.time()
        stab_stats = stabilize(frames_dir, shots, max_motion_px=args.max_motion)
        log["steps"]["stabilize"] = {
            **stab_stats,
            "seconds": round(time.time() - t4, 2),
        }
    else:
        print(f"\n⟳  Skipping stabilization (--no-stabilize)")
        log["steps"]["stabilize"] = {"skipped": True}

    # ── Summary ──
    total_time = round(time.time() - t_start, 2)
    log["total_seconds"] = total_time

    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 1 complete in {total_time}s")
    print(f"     {frame_count} frames → {frames_dir}")
    print(f"     {len(shots)} shot(s) → {shots_path}")
    print(f"     Log → {log_path}")
    print(f"{'='*60}")
    print(f"\n  Next: run Phase 2 (SHARP reconstruction) on {frames_dir}")


if __name__ == "__main__":
    main()
