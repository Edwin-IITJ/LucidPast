"""
Step 1 — Frame Extraction
==========================
Extract frames from a video file using ffmpeg.

Replaces the bash-only `phase0-spike/scripts/extract_frames.sh` with a
pure-Python wrapper that works on Windows without Git Bash / WSL.

Usage (standalone):
    python extract.py <input_video> <output_dir> [--fps 8] [--max-dim 1024]

As a module:
    from steps.extract import extract_frames
    count = extract_frames(video_path, output_dir, fps=8, max_dim=1024)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: int | None = 8,
    max_dim: int | None = 1024,
) -> int:
    """
    Extract video frames as lossless PNGs using ffmpeg.

    Parameters
    ----------
    video_path : Path
        Input video file.
    output_dir : Path
        Directory to write frame_NNNNN.png files into (created if needed).
    fps : int or None
        Target frame rate. None = keep source native fps.
    max_dim : int or None
        Cap the longer edge in pixels (aspect preserved). None = native res.

    Returns
    -------
    int
        Number of frames extracted.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"✗  Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # ── Build the ffmpeg filter chain ──
    filters: list[str] = []

    if fps is not None:
        filters.append(f"fps={fps}")

    if max_dim is not None:
        # Scale so the longer edge is capped at max_dim, shorter edge even
        # iw>ih → scale width to min(max_dim,iw), height auto (-2 = even)
        # iw<=ih → scale height to min(max_dim,ih), width auto
        scale = (
            f"scale="
            f"'if(gt(iw,ih),min({max_dim},iw),-2)'"
            f":"
            f"'if(gt(iw,ih),-2,min({max_dim},ih))'"
        )
        filters.append(scale)

    # ── Assemble the ffmpeg command ──
    cmd: list[str] = ["ffmpeg", "-y", "-i", str(video_path)]

    if filters:
        cmd += ["-vf", ",".join(filters)]

    output_pattern = str(output_dir / "frame_%05d.png")
    cmd.append(output_pattern)

    # ── Run ──
    print(f"\n{'─'*60}")
    print(f"▶  Extracting frames: fps={fps}, max_dim={max_dim}")
    print(f"   {' '.join(cmd)}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗  ffmpeg failed (exit code {result.returncode}):", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        sys.exit(result.returncode)

    # Count extracted frames
    count = len(list(output_dir.glob("frame_*.png")))
    print(f"✔  Extracted {count} frames → {output_dir}")
    return count


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames as PNGs.")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("output_dir", type=Path, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=8, help="Frame rate (default: 8)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Cap longer edge (default: 1024)")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, fps=args.fps, max_dim=args.max_dim)
