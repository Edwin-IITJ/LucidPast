#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LucidPast Video-to-Splat Pipeline
==================================
One command to go from a raw video clip to a browser-viewable splat sequence.

Usage:
    python run_pipeline.py <clip_name> [options]

    <clip_name>   Name of the video file (without extension) inside Video/input/.
                  Example: if your clip is Video/input/testclip2.mp4, pass testclip2.

Options:
    --fps N          Frame rate to extract (default: 8). Lower = faster but choppier.
    --max-dim N      Cap longer edge in pixels (default: 1024). Lower = faster.
    --every-nth N    Run SHARP on every Nth frame (default: 3). 1 = every frame.
    --port N         HTTP server port for the viewer (default: 8001).
    --no-server      Skip launching the HTTP server at the end.
    --skip-extract   Skip frame extraction (reuse existing frames in output/<clip>/frames/).
    --skip-sharp     Skip SHARP reconstruction (reuse existing .ply files).
    --skip-convert   Skip .ply → .splat conversion (reuse existing .splat files).

Examples:
    # Full pipeline, default settings:
    python run_pipeline.py testclip2

    # Every frame (slow but highest quality):
    python run_pipeline.py testclip2 --every-nth 1

    # Cheap first-pass spike: every 5th frame at 480p:
    python run_pipeline.py testclip2 --every-nth 5 --max-dim 480

    # Re-convert splats from existing .ply files (e.g. after updating the converter):
    python run_pipeline.py testclip2 --skip-extract --skip-sharp
"""

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
VIDEO_DIR = Path(__file__).parent          # …/Video/
SCRIPTS_DIR = VIDEO_DIR / "phase0-spike" / "scripts"
VIEWER_HTML = VIDEO_DIR / "phase0-spike" / "viewer" / "view_sequence.html"
INPUT_DIR = VIDEO_DIR / "input"
OUTPUT_DIR = VIDEO_DIR / "output"
# ──────────────────────────────────────────────────────────────────────────────


def run(cmd, description, cwd=None):
    """Run a shell command, streaming output live. Exits on failure."""
    print(f"\n{'─'*60}")
    print(f"▶  {description}")
    print(f"   {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=cwd or VIDEO_DIR)
    if result.returncode != 0:
        print(f"\n✗  Step failed (exit code {result.returncode}). Aborting.", file=sys.stderr)
        sys.exit(result.returncode)


def find_video(clip_name: str) -> Path:
    """Find the clip in Video/input/, trying common video extensions."""
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]:
        p = INPUT_DIR / f"{clip_name}{ext}"
        if p.exists():
            return p
    print(f"✗  No video found for '{clip_name}' in {INPUT_DIR}", file=sys.stderr)
    print(f"   Tried: {', '.join(clip_name + e for e in ['.mp4','.mov','.avi','.mkv','.webm','.m4v'])}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LucidPast: convert a video clip to a Gaussian Splat sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("clip_name", help="Clip name without extension (must be in Video/input/)")
    parser.add_argument("--fps", type=int, default=8, help="Extraction frame rate (default: 8)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Cap longer edge in pixels (default: 1024)")
    parser.add_argument("--every-nth", type=int, default=3, help="Run SHARP on every Nth frame (default: 3)")
    parser.add_argument("--port", type=int, default=8001, help="Viewer HTTP server port (default: 8001)")
    parser.add_argument("--no-server", action="store_true", help="Don't launch the HTTP server at the end")
    parser.add_argument("--skip-extract", action="store_true", help="Skip frame extraction")
    parser.add_argument("--skip-sharp", action="store_true", help="Skip SHARP reconstruction")
    parser.add_argument("--skip-convert", action="store_true", help="Skip .ply → .splat conversion")
    args = parser.parse_args()

    clip_name = args.clip_name

    # ── Output directories ──
    frames_dir = OUTPUT_DIR / clip_name / "frames"
    splats_dir = OUTPUT_DIR / clip_name / "splats"
    frames_dir.mkdir(parents=True, exist_ok=True)
    splats_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LucidPast Video-to-Splat Pipeline")
    print(f"  Clip:      {clip_name}")
    print(f"  FPS:       {args.fps}  |  Max dim: {args.max_dim}  |  Every Nth: {args.every_nth}")
    print(f"{'='*60}")

    # ── Step 1: Extract frames ──
    if not args.skip_extract:
        video_path = find_video(clip_name)
        run(
            ["bash", str(SCRIPTS_DIR / "extract_frames.sh"),
             str(video_path), str(frames_dir),
             str(args.fps), str(args.max_dim)],
            f"Step 1/3 — Extracting frames at {args.fps} fps, max {args.max_dim}px",
        )
    else:
        existing = list(frames_dir.glob("frame_*.png"))
        print(f"\n⟳  Skipping extraction — using {len(existing)} existing frames in {frames_dir}")

    # ── Step 2: Run SHARP ──
    if not args.skip_sharp:
        run(
            [sys.executable, str(SCRIPTS_DIR / "run_sharp_batch.py"),
             str(frames_dir), str(splats_dir), str(args.every_nth)],
            f"Step 2/3 — Running SHARP (every {args.every_nth} frame(s))",
        )
    else:
        existing = list(splats_dir.glob("frame_*.ply"))
        print(f"\n⟳  Skipping SHARP — using {len(existing)} existing .ply files in {splats_dir}")

    # ── Step 3: Convert PLY → .splat ──
    if not args.skip_convert:
        run(
            [sys.executable, str(SCRIPTS_DIR / "convert_to_splat.py"),
             str(splats_dir), str(splats_dir)],
            "Step 3/3 — Converting .ply files to .splat format",
        )
    else:
        existing = list(splats_dir.glob("frame_*.splat"))
        print(f"\n⟳  Skipping conversion — using {len(existing)} existing .splat files")

    # ── Copy viewer HTML ──
    dest_html = splats_dir / "view_sequence.html"
    shutil.copy2(VIEWER_HTML, dest_html)
    print(f"\n✔  Copied viewer → {dest_html}")

    # ── Summary ──
    splat_count = len(list(splats_dir.glob("frame_*.splat")))
    url = f"http://localhost:{args.port}/view_sequence.html"

    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete!")
    print(f"     {splat_count} frames ready in:")
    print(f"     {splats_dir}")
    print(f"{'='*60}")

    if args.no_server:
        print(f"\n  To view manually:")
        print(f"    cd \"{splats_dir}\"")
        print(f"    python -m http.server {args.port}")
        print(f"    # Open {url}")
    else:
        print(f"\n  Starting viewer at {url}")
        print(f"  Press Ctrl+C to stop the server.\n")
        # Open browser first, then block on the server
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            subprocess.run(
                [sys.executable, "-m", "http.server", str(args.port)],
                cwd=splats_dir,
            )
        except KeyboardInterrupt:
            print("\n  Server stopped.")


if __name__ == "__main__":
    main()
