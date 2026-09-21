"""
Step 4 — Stabilize
====================
Feature-tracking-based stabilization to correct camera jitter in archival
footage, using sparse optical flow (Lucas-Kanade).

**Important design decision:**
    This step only corrects *jitter* (small involuntary camera shake on
    footage that is meant to be static).  It does NOT suppress intentional
    camera movement (pans, tilts, zooms, tracking shots).

    When the detected inter-frame motion magnitude exceeds `max_motion_px`
    (default 15 pixels), the frame is left untouched.  This means footage
    like testclip2 (dashcam/POV with continuous intentional movement) will
    be skipped automatically, while genuine hand-held jitter on a mostly-
    static shot will be corrected.

Usage (standalone):
    python stabilize.py <frames_dir> <shots_json> [--max-motion 15]

As a module:
    from steps.stabilize import stabilize
    stabilize(frames_dir, shots, max_motion_px=15)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def stabilize(
    frames_dir: Path,
    shots: list[dict],
    max_motion_px: float = 15.0,
) -> dict:
    """
    Stabilize frames within each shot using sparse optical flow.

    Only corrects jitter — frames where the median motion exceeds
    `max_motion_px` are left untouched (intentional camera movement).

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame_NNNNN.png files.
    shots : list[dict]
        Shot list from detect_shots.
    max_motion_px : float
        Maximum median motion (in pixels) to consider as jitter.
        Above this, the frame is classified as intentional motion and skipped.

    Returns
    -------
    dict
        Summary with keys: stabilized (int), skipped_intentional (int),
        skipped_short (int), total (int).
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("frame_*.png"))

    if not all_frames:
        print("✗  No frames found in", frames_dir, file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"▶  Stabilizing {len(all_frames)} frames (max_motion={max_motion_px}px)")
    print(f"{'─'*60}")

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # FAST feature detector for finding keypoints
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    stats = {"stabilized": 0, "skipped_intentional": 0, "skipped_short": 0, "total": 0}

    for shot in shots:
        start = shot["start_index"]
        end = shot["end_index"]
        shot_frames = all_frames[start:end + 1]
        n = len(shot_frames)
        stats["total"] += n

        if n < 3:
            print(f"   Shot {shot['shot_id']}: {n} frames — too short, skipping")
            stats["skipped_short"] += n
            continue

        print(f"   Shot {shot['shot_id']}: analyzing {n} frames "
              f"({shot_frames[0].name} → {shot_frames[-1].name})")

        # Read first frame
        prev_img = cv2.imread(str(shot_frames[0]))
        if prev_img is None:
            continue
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        # Accumulate transforms relative to first frame
        # transforms[i] = (dx, dy, da) from frame 0 to frame i
        transforms: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        intentional_motion_detected = False
        shot_intentional_count = 0

        for i in range(1, n):
            curr_img = cv2.imread(str(shot_frames[i]))
            if curr_img is None:
                transforms.append(transforms[-1])
                continue
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            # Detect features in previous frame
            kps = fast.detect(prev_gray, None)
            if len(kps) < 10:
                # Not enough features — skip
                transforms.append(transforms[-1])
                prev_gray = curr_gray
                continue

            prev_pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

            # Track features to current frame
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )

            # Filter to good matches
            good_mask = status.flatten() == 1
            if good_mask.sum() < 4:
                transforms.append(transforms[-1])
                prev_gray = curr_gray
                continue

            prev_good = prev_pts[good_mask]
            curr_good = curr_pts[good_mask]

            # Compute motion magnitude
            motion = np.sqrt(np.sum((curr_good - prev_good) ** 2, axis=2))
            median_motion = float(np.median(motion))

            if median_motion > max_motion_px:
                # Intentional camera movement — do not stabilize
                transforms.append(transforms[-1])
                shot_intentional_count += 1
                prev_gray = curr_gray
                continue

            # Estimate affine transform (translation + rotation)
            mat, _ = cv2.estimateAffinePartial2D(prev_good, curr_good)
            if mat is None:
                transforms.append(transforms[-1])
                prev_gray = curr_gray
                continue

            dx = mat[0, 2]
            dy = mat[1, 2]
            da = np.arctan2(mat[1, 0], mat[0, 0])

            # Accumulate
            prev_dx, prev_dy, prev_da = transforms[-1]
            transforms.append((prev_dx + dx, prev_dy + dy, prev_da + da))

            prev_gray = curr_gray

        # If most frames have intentional motion, skip the whole shot
        if shot_intentional_count > n * 0.5:
            print(f"   → Shot {shot['shot_id']}: {shot_intentional_count}/{n} frames "
                  f"exceed {max_motion_px}px — intentional motion, skipping stabilization")
            stats["skipped_intentional"] += n
            continue

        # Smooth the accumulated transforms with a rolling average
        smooth_window = min(15, n // 2)  # half-window
        smooth_transforms: list[tuple[float, float, float]] = []
        for i in range(n):
            lo = max(0, i - smooth_window)
            hi = min(n, i + smooth_window + 1)
            window = transforms[lo:hi]
            avg_dx = np.mean([t[0] for t in window])
            avg_dy = np.mean([t[1] for t in window])
            avg_da = np.mean([t[2] for t in window])
            smooth_transforms.append((avg_dx, avg_dy, avg_da))

        # Apply correction: for each frame, warp to cancel the difference
        # between its accumulated transform and the smoothed trajectory
        corrected = 0
        for i in range(n):
            dx = smooth_transforms[i][0] - transforms[i][0]
            dy = smooth_transforms[i][1] - transforms[i][1]
            da = smooth_transforms[i][2] - transforms[i][2]

            # Skip if correction is negligible
            if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(da) < 0.001:
                continue

            img = cv2.imread(str(shot_frames[i]))
            if img is None:
                continue

            h, w = img.shape[:2]
            cos_a = np.cos(da)
            sin_a = np.sin(da)
            mat = np.array([
                [cos_a, -sin_a, dx],
                [sin_a,  cos_a, dy],
            ], dtype=np.float64)

            stabilized = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(shot_frames[i]), stabilized)
            corrected += 1

        if corrected > 0:
            print(f"   → Shot {shot['shot_id']}: stabilized {corrected}/{n} frames")
        else:
            print(f"   → Shot {shot['shot_id']}: no correction needed (stable shot)")
        stats["stabilized"] += corrected

    print(f"✔  Stabilization complete: {stats['stabilized']} corrected, "
          f"{stats['skipped_intentional']} skipped (intentional motion)")
    return stats


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stabilize frames within shots.")
    parser.add_argument("frames_dir", type=Path, help="Directory of frame_NNNNN.png files")
    parser.add_argument("shots_json", type=Path, help="Path to shots.json from detect_shots")
    parser.add_argument("--max-motion", type=float, default=15.0,
                        help="Max median motion in pixels before skipping (default: 15)")
    args = parser.parse_args()

    with open(args.shots_json) as f:
        shots = json.load(f)
    stabilize(args.frames_dir, shots, max_motion_px=args.max_motion)
