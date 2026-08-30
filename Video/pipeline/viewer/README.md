# Phase 5 — Viewer (Production)

> **Status:** Not started. Depends on Phase 4.

## Scope

Production WebXR viewer for playing back compressed splat sequences in the browser and in VR.

## Approach

- **Do not write a WebGL/WebGPU renderer from scratch.** Extend GaussianSplats3D's dynamic-scene mode, or fork splaTV / `.splat4d`'s viewer.
- **WebXR sitting-VR session** — reuse the LucidPast WebXR setup from `../../LucidPast/web/` where it fits.
- **Cone boundary** — fade or soft-clamp at the edge of reconstructed geometry. The user should see into the void beyond the camera's original field of view, not a hard edge.

## Key Difference from Phase 0 Viewer

The Phase 0 viewer (`phase0-spike/viewer/view_sequence.html`) is throwaway — it loads raw `.ply` files with no disposal, no compression, no streaming. This viewer must handle compressed/streamed data and proper GPU resource management.
