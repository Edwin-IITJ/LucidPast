"""
Step 1 — Position Pass-Through (v2)
=====================================
v1 averaged positions across frames, which destroyed geometry —
especially for moving-camera footage where each frame's Gaussians
are in different camera-space coordinates.

v2: Positions are passed through UNTOUCHED from SHARP's output.
This module exists as a no-op for pipeline consistency.
"""

import numpy as np


def smooth_positions(
    frames: list[np.ndarray],
    center_idx: int,
    window: int = 3,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Pass-through: return the center frame unchanged.

    v1 averaged (x, y, z) across frames, destroying geometry for
    moving-camera footage. v2 preserves SHARP's reconstruction exactly.
    """
    return frames[center_idx].copy()
