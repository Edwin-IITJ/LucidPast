"""
Step 2 — Shot-Cut Detection
=============================
Classical histogram-based detection of hard cuts in an extracted frame sequence.

Each shot becomes its own independent reconstruction run in Phase 2, because a
hard cut is a scene discontinuity — SHARP has no context to bridge across cuts,
and temporal smoothing (Phase 3) must not interpolate across them either.

Approach:
    For each consecutive pair of frames, compute the per-channel histogram
    correlation.  When the correlation drops below a configurable threshold,
    mark a shot boundary.  This is a standard technique that works well on
    archival footage where tonal changes within a shot are gradual.

Usage (standalone):
    python detect_shots.py <frames_dir> [--threshold 0.7] [--output shots.json]

As a module:
    from steps.detect_shots import detect_shots
    shots = detect_shots(frames_dir, threshold=0.7)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _frame_histogram(img_path: Path) -> np.ndarray:
    """Compute a normalized per-channel histogram for a single frame."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Could not read image: {img_path}")

    hists = []
    for ch in range(3):  # B, G, R
        hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)

    return np.concatenate(hists, axis=0)


def detect_shots(
    frames_dir: Path,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Detect shot boundaries via histogram correlation.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame_NNNNN.png files (sorted by name).
    threshold : float
        Correlation below this value triggers a shot boundary.
        Lower = fewer cuts detected.  Default 0.7 is tuned for archival
        footage where within-shot tonal changes are slow.

    Returns
    -------
    list[dict]
        List of shots, each with:
        - shot_id (int)
        - start_frame (str): filename of first frame in shot
        - end_frame (str): filename of last frame in shot
        - start_index (int): 0-based index of first frame
        - end_index (int): 0-based index of last frame
        - frame_count (int)
    """
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob("frame_*.png"))

    if len(frames) == 0:
        print("✗  No frames found in", frames_dir, file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"▶  Detecting shot cuts across {len(frames)} frames (threshold={threshold})")
    print(f"{'─'*60}")

    # Compute histograms for all frames
    histograms: list[np.ndarray] = []
    for f in frames:
        histograms.append(_frame_histogram(f))

    # Find shot boundaries
    boundaries: list[int] = [0]  # First frame always starts shot 0
    for i in range(1, len(histograms)):
        # Correlation between consecutive frames
        corr = cv2.compareHist(
            histograms[i - 1], histograms[i], cv2.HISTCMP_CORREL
        )
        if corr < threshold:
            boundaries.append(i)
            print(f"   ✂  Shot cut at frame {i} ({frames[i].name}), correlation={corr:.3f}")

    # Build shot list
    shots: list[dict] = []
    for s, start_idx in enumerate(boundaries):
        end_idx = (boundaries[s + 1] - 1) if s + 1 < len(boundaries) else len(frames) - 1
        shots.append({
            "shot_id": s,
            "start_frame": frames[start_idx].name,
            "end_frame": frames[end_idx].name,
            "start_index": start_idx,
            "end_index": end_idx,
            "frame_count": end_idx - start_idx + 1,
        })

    if len(boundaries) == 1:
        print(f"   No shot cuts detected — single continuous shot ({len(frames)} frames)")
    else:
        print(f"   {len(shots)} shots detected")

    print(f"✔  Shot detection complete")
    return shots


def save_shots(shots: list[dict], output_path: Path) -> None:
    """Write the shot list to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(shots, indent=2))
    print(f"✔  Wrote {output_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect shot cuts in a frame sequence.")
    parser.add_argument("frames_dir", type=Path, help="Directory of frame_NNNNN.png files")
    parser.add_argument("--threshold", type=float, default=0.7, help="Correlation threshold (default: 0.7)")
    parser.add_argument("--output", type=Path, default=None, help="Output shots.json path")
    args = parser.parse_args()

    shots = detect_shots(args.frames_dir, threshold=args.threshold)
    out = args.output or (args.frames_dir.parent / "shots.json")
    save_shots(shots, out)
