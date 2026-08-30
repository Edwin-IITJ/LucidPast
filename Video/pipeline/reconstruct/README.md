# Phase 2 — Reconstruct (SHARP Wrapper)

> **Status:** Not started. Depends on Phase 1.

## Scope

Production-grade batch wrapper around Apple SHARP. Frame directory in → per-frame `.ply` out.

## What Needs to Be Built

1. **Batch wrapper** — Evolved from `phase0-spike/scripts/run_sharp_batch.py`, but production-grade: proper error handling, resume support, progress reporting.

2. **Input normalization** — Handle SHARP's expected input resolution/aspect ratio explicitly. Archival footage often deviates from modern norms.

3. **Anomaly flagging** — Flag frames with depth-range outliers or unusual Gaussian counts for manual review, matching the pattern already seen with the image pipeline where a few exceptions needed attention.

## Constraints

- Must run on RTX 3060 (12 GB desktop / 6 GB mobile).
- Each frame is an independent reconstruction — no cross-frame state.
