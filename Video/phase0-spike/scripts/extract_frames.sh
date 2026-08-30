#!/usr/bin/env bash
# Phase 0 spike: extract frames from a short archival-style clip for
# per-frame SHARP testing.
#
# Usage:
#   ./extract_frames.sh <input_video> <output_dir> [fps] [max_dim]
#
#   fps       optional. Downsample frame rate for the first spike run
#             (e.g. 8-10) so a 5-10s clip is cheap to batch through SHARP.
#             Leave blank to keep the source's native fps.
#   max_dim   optional. Caps the longer edge in pixels, aspect preserved.
#             On a 6GB mobile 3060 it's worth capping this explicitly
#             rather than assuming native video resolution is safe --
#             match whatever resolution has been working for SHARP on
#             your archival photos in LucidPast.
#
# Frames are saved as PNG (lossless) rather than JPG so no extra
# generational compression loss goes into the reconstruction step.

set -euo pipefail

INPUT="${1:?Usage: extract_frames.sh <input_video> <output_dir> [fps] [max_dim]}"
OUTDIR="${2:?Usage: extract_frames.sh <input_video> <output_dir> [fps] [max_dim]}"
FPS="${3:-}"
MAX_DIM="${4:-}"

mkdir -p "$OUTDIR"

VF=""
if [ -n "$FPS" ]; then
  VF="fps=$FPS"
fi
if [ -n "$MAX_DIM" ]; then
  SCALE="scale='if(gt(iw,ih),min(${MAX_DIM},iw),-2)':'if(gt(iw,ih),-2,min(${MAX_DIM},ih))'"
  if [ -n "$VF" ]; then VF="${VF},${SCALE}"; else VF="${SCALE}"; fi
fi

if [ -n "$VF" ]; then
  ffmpeg -y -i "$INPUT" -vf "$VF" -q:v 2 "$OUTDIR/frame_%05d.png"
else
  ffmpeg -y -i "$INPUT" -q:v 2 "$OUTDIR/frame_%05d.png"
fi

COUNT=$(ls "$OUTDIR"/frame_*.png 2>/dev/null | wc -l | tr -d ' ')
echo "Extracted $COUNT frames to $OUTDIR"
