# Phase 1 — Ingest & Preprocess

> **Status:** Not started. Build only after Phase 0 spike is complete and evaluated.

## Scope

This module handles everything between "raw video file" and "clean, individual frames ready for SHARP."

## What Needs to Be Built

1. **Frame extraction** — Accept any common video container/codec via ffmpeg. The `phase0-spike/scripts/extract_frames.sh` script is a starting point but is not production-grade.

2. **Shot-cut detection** — Classical histogram or edge-based detection (no learned model needed). Each shot gets its own reconstruction run, because a hard cut is a scene discontinuity — SHARP has no context to bridge across cuts.

3. **Non-generative deflicker** — Temporal filtering to smooth brightness/color fluctuations common in old film. Must be classical (e.g., rolling average, histogram matching). No generative models.

4. **Stabilization** — Feature-tracking-based stabilization for shaky archival footage. Standard OpenCV-level approach.

## Constraints

- Output frames must be lossless PNG (no additional compression artifacts before SHARP).
- Must handle non-standard aspect ratios and resolutions common in archival material.
- No generative processing of any kind.
