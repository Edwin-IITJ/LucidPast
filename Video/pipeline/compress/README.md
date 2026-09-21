# Phase 4 — Compress

> **Status:** Not started. Depends on Phases 2 (and optionally 3).

## Scope

Compress the per-frame `.ply` sequence into a streamable format suitable for web delivery.

## Approach

1. **Study `.splat4d`'s format first** — understand its keyframe + delta encoding before designing anything custom.
2. **Keyframe + delta encoding** across the frame sequence.
3. **HTTP-range-streamable output** — the viewer should be able to start playback before the full sequence is downloaded. No full-sequence-download-before-play.

## Phase 0 Data (Informs Design)

- **Gaussians/frame:** Fixed at **1,179,648** per frame (both clips). SHARP always outputs this count regardless of scene complexity.
- **Per-frame .splat size:** ~36 MB uncompressed (32 bytes × 1,179,648 Gaussians). A 22-frame testclip2 sequence = ~792 MB raw — HTTP-range streaming and delta encoding are non-negotiable.
- **Inter-frame delta efficiency TBD** — requires Phase 3 output to quantify actual position/attribute changes between smoothed frames. The unsmoothed spike outputs show high variance (full-scene warp in moving-camera footage), suggesting raw delta encoding on unsmoothed frames would be inefficient.

> [!IMPORTANT]
> Do not start Phase 4 design until Phase 3 output exists. Study `.splat4d` format *before* designing anything custom.
