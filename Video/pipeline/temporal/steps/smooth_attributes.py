"""
Step 2 — Attribute Smoothing (v2: EMA on Color + Opacity Only)
================================================================
v1 averaged ALL attributes (color, opacity, scale, rotation) across a
centered Gaussian window. This was too aggressive — it blurred scale
and rotation unnecessarily, and the centered window caused temporal
smearing.

v2: Exponential Moving Average (EMA) applied ONLY to:
  - Color (f_dc_0, f_dc_1, f_dc_2) — reduces color flicker
  - Opacity — reduces opacity popping

Scale and rotation are passed through UNTOUCHED — these are structural
properties that SHARP gets right per-frame.

EMA is forward-only (causal): each frame blends with its predecessor.
This is better for playback continuity than a centered window.

    smoothed[t] = (1 - α) * original[t] + α * smoothed[t-1]

α = 0 means no smoothing (pass-through).
α = 1 means complete replacement with previous frame.
"""

import numpy as np


def smooth_attributes_ema(
    frames: list[np.ndarray],
    color_alpha: float = 0.3,
    opacity_alpha: float = 0.2,
) -> list[np.ndarray]:
    """
    Apply forward EMA smoothing to color and opacity across a sequence.

    Unlike v1 (which operated on a sliding window per-frame), this processes
    the ENTIRE sequence in one forward pass. Much simpler and produces
    better temporal continuity.

    Parameters
    ----------
    frames : list[np.ndarray]
        Full sequence of structured Gaussian arrays (ORIGINAL, unmodified).
    color_alpha : float
        EMA blending weight for color SH coefficients. Higher = more
        smoothing. Default 0.3 is gentle enough to reduce flicker without
        ghosting.
    opacity_alpha : float
        EMA blending weight for opacity. Default 0.2 is conservative to
        avoid making semi-transparent objects sluggish.

    Returns
    -------
    list[np.ndarray]
        Sequence with smoothed color and opacity. All other attributes
        (positions, scale, rotation) are unchanged.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return []

    results = []

    # First frame: no predecessor to blend with → pass through
    results.append(frames[0].copy())

    # Forward EMA pass
    for t in range(1, n_frames):
        current = frames[t].copy()

        prev = results[t - 1]

        # Smooth color (SH DC coefficients)
        current["f_dc_0"] = (1 - color_alpha) * current["f_dc_0"] + color_alpha * prev["f_dc_0"]
        current["f_dc_1"] = (1 - color_alpha) * current["f_dc_1"] + color_alpha * prev["f_dc_1"]
        current["f_dc_2"] = (1 - color_alpha) * current["f_dc_2"] + color_alpha * prev["f_dc_2"]

        # Smooth opacity
        current["opacity"] = (1 - opacity_alpha) * current["opacity"] + opacity_alpha * prev["opacity"]

        # Scale, rotation, positions: UNTOUCHED
        results.append(current)

    return results


# ── Legacy API wrapper for backward compat with temporal.py ──────────────────

def smooth_attributes(
    frames: list[np.ndarray],
    center_idx: int,
    window: int = 2,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Legacy wrapper — NOT used by v2 pipeline.
    Kept for API compatibility. Returns center frame unchanged.
    """
    return frames[center_idx].copy()
