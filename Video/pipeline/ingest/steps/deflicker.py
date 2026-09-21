"""
Step 3 — Deflicker
====================
Classical per-shot brightness/color normalization to smooth the temporal
flicker common in archival film footage.

Approach:
    For each frame within a shot, match its luminance histogram to the
    rolling mean histogram of its neighboring frames (window size
    configurable).  This is a classical technique — no generative models,
    no learned components.

    Works in LAB color space: only the L (luminance) channel is adjusted,
    preserving the original color hue and saturation.

Usage (standalone):
    python deflicker.py <frames_dir> <shots_json> [--window 5]

As a module:
    from steps.deflicker import deflicker
    deflicker(frames_dir, shots, window=5)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _match_histogram_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the histogram of `source` to `reference` for a single channel.
    Both are 2D arrays (single channel images).
    """
    src_vals, src_counts = np.unique(source, return_counts=True)
    ref_vals, ref_counts = np.unique(reference, return_counts=True)

    # Build CDFs
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]

    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Map source values → reference values via CDF matching
    # For each source CDF value, find the closest reference CDF value
    mapping = np.zeros(256, dtype=np.uint8)
    for s_val, s_cdf_val in zip(src_vals, src_cdf):
        idx = np.searchsorted(ref_cdf, s_cdf_val)
        idx = min(idx, len(ref_vals) - 1)
        mapping[s_val] = ref_vals[idx]

    return mapping[source]


def deflicker(
    frames_dir: Path,
    shots: list[dict],
    window: int = 5,
) -> int:
    """
    Apply rolling-window histogram matching to smooth brightness across frames.

    Operates per-shot (does not smooth across shot boundaries).
    Only adjusts the L channel in LAB space — hue and saturation are preserved.

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame_NNNNN.png files.
    shots : list[dict]
        Shot list from detect_shots (with start_index, end_index).
    window : int
        Number of neighboring frames on each side to average for the
        reference histogram.  Total reference window = 2*window + 1.

    Returns
    -------
    int
        Number of frames processed.
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("frame_*.png"))

    if not all_frames:
        print("✗  No frames found in", frames_dir, file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"▶  Deflickering {len(all_frames)} frames (window={window})")
    print(f"{'─'*60}")

    total_processed = 0

    for shot in shots:
        start = shot["start_index"]
        end = shot["end_index"]
        shot_frames = all_frames[start:end + 1]
        n = len(shot_frames)

        if n < 3:
            print(f"   Shot {shot['shot_id']}: {n} frames — too short, skipping")
            continue

        print(f"   Shot {shot['shot_id']}: deflickering {n} frames "
              f"({shot_frames[0].name} → {shot_frames[-1].name})")

        # Read all frames in this shot as LAB
        lab_frames: list[np.ndarray] = []
        for f in shot_frames:
            img = cv2.imread(str(f))
            if img is None:
                print(f"   ⚠  Could not read {f.name}, skipping", file=sys.stderr)
                lab_frames.append(None)
                continue
            lab_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

        # Compute L-channel histograms for each frame
        l_hists: list[np.ndarray | None] = []
        for lab in lab_frames:
            if lab is None:
                l_hists.append(None)
                continue
            l_channel = lab[:, :, 0]
            hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
            l_hists.append(hist)

        # For each frame, compute reference histogram as mean of neighbors
        for i in range(n):
            if lab_frames[i] is None:
                continue

            # Gather neighbor histograms within the window
            lo = max(0, i - window)
            hi = min(n - 1, i + window)
            neighbor_hists = [l_hists[j] for j in range(lo, hi + 1) if l_hists[j] is not None]

            if len(neighbor_hists) < 2:
                continue

            # Mean reference histogram
            ref_hist = np.mean(neighbor_hists, axis=0)

            # Build a synthetic reference L channel from the mean histogram
            # We need actual pixel values for histogram matching, so we build
            # a reference "image" with the right distribution
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            ref_cdf /= ref_cdf[-1]

            src_l = lab_frames[i][:, :, 0]
            src_hist = l_hists[i]
            src_cdf = np.cumsum(src_hist).astype(np.float64)
            src_cdf /= src_cdf[-1]

            # Map each source L value to the reference L value via CDF matching
            mapping = np.zeros(256, dtype=np.uint8)
            for src_val in range(256):
                src_cdf_val = src_cdf[src_val]
                ref_idx = np.searchsorted(ref_cdf, src_cdf_val)
                ref_idx = min(ref_idx, 255)
                mapping[src_val] = ref_idx

            # Apply the mapping
            lab_frames[i][:, :, 0] = mapping[src_l]

            # Convert back and save
            result = cv2.cvtColor(lab_frames[i], cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(shot_frames[i]), result)
            total_processed += 1

    print(f"✔  Deflickered {total_processed} frames")
    return total_processed


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deflicker a frame sequence per-shot.")
    parser.add_argument("frames_dir", type=Path, help="Directory of frame_NNNNN.png files")
    parser.add_argument("shots_json", type=Path, help="Path to shots.json from detect_shots")
    parser.add_argument("--window", type=int, default=5, help="Rolling window half-size (default: 5)")
    args = parser.parse_args()

    with open(args.shots_json) as f:
        shots = json.load(f)
    deflicker(args.frames_dir, shots, window=args.window)
