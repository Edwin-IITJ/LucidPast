#!/usr/bin/env python3
"""
Phase 0 Spike Orchestrator — single command to run the entire spike.

Usage:
    python phase0-spike/run_spike.py input/testclip.mp4 [--fps 8] [--max-dim 1024] [--every-nth 3] [--serve]

This will:
  1. Create output/<video_name>/ directory structure
  2. Extract frames via ffmpeg
  3. Run SHARP on each frame (or every Nth frame)
  4. Convert SHARP .ply output to web-friendly .splat format
  5. Copy the viewer into the output folder
  6. Optionally start a local HTTP server (--serve)
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = SCRIPT_DIR.parent  # Video/
VIEWER_HTML = SCRIPT_DIR / "viewer" / "view_sequence.html" if (SCRIPT_DIR / "viewer").exists() else SCRIPT_DIR / "phase0-spike" / "viewer" / "view_sequence.html"


def derive_video_name(video_path: Path) -> str:
    """Strip extension to get a clean folder name."""
    return video_path.stem


def run_step(description: str, cmd: list[str], cwd: Path = None):
    """Run a command, print its output, and abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if result.returncode != 0:
        print(f"\n[FAIL] Step failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 0 spike end-to-end.")
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument("--fps", type=int, default=8, help="Frame extraction rate (default: 8)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Cap longest edge in pixels (default: 1024)")
    parser.add_argument("--every-nth", type=int, default=3, help="Process every Nth frame (default: 3)")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server after processing")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    video_name = derive_video_name(video_path)
    output_base = VIDEO_ROOT / "output" / video_name

    # Directory structure
    frames_dir = output_base / "frames"
    splats_dir = output_base / "splats"
    splats_web_dir = output_base / "splats_web"

    frames_dir.mkdir(parents=True, exist_ok=True)
    splats_dir.mkdir(parents=True, exist_ok=True)
    splats_web_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLucidPast Video Spike — {video_name}")
    print(f"  Video:      {video_path}")
    print(f"  Output:     {output_base}")
    print(f"  FPS:        {args.fps}")
    print(f"  Max dim:    {args.max_dim}")
    print(f"  Every Nth:  {args.every_nth}")

    # ─── Step 1: Extract frames ───
    extract_script = SCRIPT_DIR / "scripts" / "extract_frames.sh"
    run_step(
        "Step 1/4: Extracting frames",
        ["bash", str(extract_script), str(video_path), str(frames_dir), str(args.fps), str(args.max_dim)],
        cwd=VIDEO_ROOT,
    )

    # ─── Step 2: Run SHARP ───
    batch_script = SCRIPT_DIR / "scripts" / "run_sharp_batch.py"
    run_step(
        "Step 2/4: Running SHARP on frames",
        [sys.executable, str(batch_script), str(frames_dir), str(splats_dir), str(args.every_nth)],
        cwd=VIDEO_ROOT,
    )

    # ─── Step 3: Convert to .splat ───
    convert_script = SCRIPT_DIR / "scripts" / "convert_to_splat.py"
    run_step(
        "Step 3/4: Converting PLY to web-friendly .splat",
        [sys.executable, str(convert_script), str(splats_dir), str(splats_web_dir)],
        cwd=VIDEO_ROOT,
    )

    # ─── Step 4: Copy viewer ───
    viewer_src = SCRIPT_DIR / "viewer" / "view_sequence.html"
    viewer_dst = splats_web_dir / "view_sequence.html"
    shutil.copy2(viewer_src, viewer_dst)
    print(f"\n[OK] Viewer copied to {viewer_dst}")

    # ─── Summary ───
    splat_count = len(list(splats_web_dir.glob("*.splat")))
    print(f"\n{'='*60}")
    print(f"  DONE — {video_name}")
    print(f"{'='*60}")
    print(f"  Frames extracted: {len(list(frames_dir.glob('frame_*.png')))}")
    print(f"  Splats generated: {len(list(splats_dir.glob('*.ply')))}")
    print(f"  Web splats:       {splat_count}")
    print(f"  Output:           {output_base}")
    print()

    if args.serve:
        print(f"Starting server at http://localhost:8000/view_sequence.html")
        print(f"Press Ctrl+C to stop.\n")
        subprocess.run(
            [sys.executable, "-m", "http.server", "8000"],
            cwd=str(splats_web_dir),
        )
    else:
        print(f"To view results:")
        print(f"  cd {splats_web_dir}")
        print(f"  python -m http.server 8000")
        print(f"  Open http://localhost:8000/view_sequence.html")


if __name__ == "__main__":
    main()
