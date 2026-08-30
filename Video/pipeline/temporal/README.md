# Phase 3 — Temporal Consistency

> **Status:** Not started. **Conditional — only build if Phase 0 shows real flicker.**

## Scope

Smooth Gaussian positions and attributes across consecutive frames to reduce temporal flicker.

## Approach (If Needed)

- Optical-flow-guided smoothing across frame depth/Gaussian positions
- Must be classical and deterministic — no diffusion-based video-depth models
- If Phase 0 looks acceptable without this, skip it and note as a stretch goal

## Decision Point

After Phase 0, record findings here:
- [ ] Flicker severity: _____
- [ ] Decision: Build / Skip / Stretch goal
