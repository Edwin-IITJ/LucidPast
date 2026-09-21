# Phase 3 — Temporal Consistency

> **Status: CURRENT PHASE — next to be built.** CONFIRMED REQUIRED by Phase 0 findings.

## Scope

Smooth Gaussian positions and attributes across consecutive frames to reduce temporal flicker.

## Approach

- Optical-flow-guided smoothing across frame depth/Gaussian positions
- Must be classical and deterministic — no diffusion-based video-depth models
- Must include **shadow blob suppression** — new artifact class discovered in testclip2 (dark shadow blobs on people/vehicles in color footage)

## Phase 0 Findings (Decision Point — CLOSED)

Both test clips confirmed severe flicker. Phase 3 is **mandatory**, not optional.

- [x] **Flicker severity:** Three-tier instability observed in testclip2 (color, moving camera):
  - **Tier 1** — Edge scatter: consistent with testclip (B&W, static), present in both clips
  - **Tier 2** — Shadow instability: dark blobs on people/vehicles in color footage; unique to testclip2
  - **Tier 3** — Full-scene warp: entire geometry shifts when camera moves; worst category
- [x] **Decision: BUILD** — Full-scene scope required, not edge-only

## Key Findings from Spike

| Metric | testclip (B&W, static) | testclip2 (color, moving) |
|---|---|---|
| Flicker verdict | Yes — edges, subject "breathes" | Worse — edges + shadow blobs + full scene warp |
| Gaussians/frame | 1,179,648 (fixed by SHARP) | 1,179,648 (fixed by SHARP) |
| Failures | 0/35 | 0/22 |
| New artifacts | Edge scatter only | **Dark shadow blobs** (people/vehicles) |

> [!IMPORTANT]
> Phase 3 scope must be **full-scene**, not edge-only. Shadow suppression for dark blobs must be addressed.
> The unsmoothed outputs in `Video/output/testclip/` and `Video/output/testclip2/` are **preserved as-is** for A/B comparison after Phase 3 is built.
