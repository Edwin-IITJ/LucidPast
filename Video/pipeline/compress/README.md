# Phase 4 — Compress

> **Status:** Not started. Depends on Phases 2 (and optionally 3).

## Scope

Compress the per-frame `.ply` sequence into a streamable format suitable for web delivery.

## Approach

1. **Study `.splat4d`'s format first** — understand its keyframe + delta encoding before designing anything custom.
2. **Keyframe + delta encoding** across the frame sequence.
3. **HTTP-range-streamable output** — the viewer should be able to start playback before the full sequence is downloaded. No full-sequence-download-before-play.

## Open Questions (To Be Answered by Phase 2 Data)

- How many Gaussians per frame? (Sets the per-frame budget.)
- How much do Gaussian positions change frame to frame? (Determines delta efficiency.)
