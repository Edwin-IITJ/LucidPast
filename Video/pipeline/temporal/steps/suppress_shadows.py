"""
Step 3 — Shadow Blob Suppression
==================================
Detect and suppress transient dark blobs — Gaussians whose opacity is
anomalously high AND color is anomalously dark compared to their temporal
neighbors.

These are the Tier 2 artifacts from the Phase 0 spike: dark shadow clusters
that appear/disappear on people and vehicles in color footage (testclip2).

The key insight is that *stable* dark regions (real shadows) will be
consistently dark across the temporal window, so their median opacity
and luminance will also be dark.  The *transient* blobs appear in only
a few frames, so the temporal median catches them.

SH coefficient (f_dc) convention:
    SHARP stores color as spherical harmonic DC coefficients.
    Luminance ≈ 0.5 + C0 * (0.2126*f_dc_0 + 0.7152*f_dc_1 + 0.0722*f_dc_2)
    where C0 = 0.28209479177387814
    
    But since we're comparing relative to a threshold, we can use the
    raw SH values directly (higher = brighter).
"""

import numpy as np

C0 = 0.28209479177387814


def _compute_luminance(frame: np.ndarray) -> np.ndarray:
    """Compute approximate luminance for each Gaussian from SH DC coefficients."""
    # Convert SH DC to linear RGB, then to luminance
    r = 0.5 + C0 * frame["f_dc_0"]
    g = 0.5 + C0 * frame["f_dc_1"]
    b = 0.5 + C0 * frame["f_dc_2"]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid to convert raw opacity logits to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


def suppress_shadows(
    frames: list[np.ndarray],
    center_idx: int,
    opacity_threshold: float = 0.3,
    dark_threshold: float = 0.15,
) -> tuple[np.ndarray, int]:
    """
    Suppress transient dark shadow blobs on the center frame.

    Parameters
    ----------
    frames : list[np.ndarray]
        Sliding window of structured Gaussian arrays.
    center_idx : int
        Index into `frames` of the frame to process.
    opacity_threshold : float
        A Gaussian is flagged if its sigmoid(opacity) exceeds the temporal
        median sigmoid(opacity) by more than this value.
    dark_threshold : float
        A Gaussian is flagged only if its luminance is below this threshold.
        This prevents suppression of legitimately opaque bright Gaussians.

    Returns
    -------
    result : np.ndarray
        The center frame with shadow blobs suppressed.
    suppressed_count : int
        Number of Gaussians whose opacity was clamped.
    """
    n_frames = len(frames)
    n_gaussians = len(frames[center_idx])

    # Collect opacities (raw logits) and luminances across the window
    opacities_raw = np.zeros((n_frames, n_gaussians), dtype=np.float32)
    luminances = np.zeros((n_frames, n_gaussians), dtype=np.float32)

    for i, frame in enumerate(frames):
        opacities_raw[i] = frame["opacity"]
        luminances[i] = _compute_luminance(frame)

    # Convert to sigmoid space for comparison
    opacities_sigmoid = _sigmoid(opacities_raw)

    # Temporal median in sigmoid space
    median_opacity = np.median(opacities_sigmoid, axis=0)
    median_luminance = np.median(luminances, axis=0)

    # Current frame values
    center_opacity = opacities_sigmoid[center_idx]
    center_luminance = luminances[center_idx]

    # Flag: opacity spikes above median AND luminance is dark
    is_opacity_spike = (center_opacity - median_opacity) > opacity_threshold
    is_dark = center_luminance < dark_threshold

    # Additional check: the Gaussian should NOT be consistently dark
    # (if the median luminance is also dark, it's a real shadow, not a blob)
    is_transient = median_luminance >= dark_threshold

    suppress_mask = is_opacity_spike & is_dark & is_transient
    suppressed_count = int(suppress_mask.sum())

    # Clamp: set suppressed Gaussians' opacity to the temporal median (in logit space)
    result = frames[center_idx].copy()
    if suppressed_count > 0:
        # Convert median opacity back to logit space: logit = ln(p / (1-p))
        median_clamped = np.clip(median_opacity[suppress_mask], 1e-6, 1 - 1e-6)
        result["opacity"][suppress_mask] = np.log(median_clamped / (1 - median_clamped))

    return result, suppressed_count
