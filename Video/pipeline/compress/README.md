# Phase 4 — Compress

> **Status:** Core pipeline COMPLETE. Static camera clips are near-lossless. Moving camera quality improvement (piecewise segmentation) is IN PROGRESS.

## What This Does

Compresses per-frame `.ply` Gaussian Splat sequences into a single `.splat4d` binary file using polynomial trajectory fitting and temporal opacity modeling, inspired by recent 4D Gaussian Splatting research.

Instead of storing N independent frames (each 36-63 MB), we fit polynomial curves through each Gaussian's trajectory over time and store only the coefficients. At playback, positions/colors/opacity are reconstructed by evaluating the polynomials.

## Paper Credits

This implementation draws from three papers collected in `AR/papers/`:

1. **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis**
   Jonathon Luiten, Georgios Kopanas, Bastian Leibe, Deva Ramanan
   *3DV 2024* | arXiv:2308.09713
   — Informed our understanding of inter-frame Gaussian correspondence and physically-based motion priors (local rigidity, isometry). Our validation step confirmed that SHARP's fixed grid provides free index-based correspondence, making Paper 1's tracking unnecessary for our use case.

2. **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**
   Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang
   *CVPR 2024* | arXiv:2310.08528
   — Demonstrated that compact deformation representations (HexPlane + MLP) can achieve dramatic compression of dynamic Gaussian Splat sequences. Informed our compression ratio targets.

3. **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis**
   Zhan Li, Zhang Chen, Zhong Li, Yi Xu
   *CVPR 2024* | arXiv:2312.16812
   — **Primary reference.** Our polynomial motion model (degree-3 position trajectories) and TRBF temporal opacity are directly adapted from this paper's approach. Key insight adopted: "Complex long motion is represented by multiple short STGs" — the basis for our piecewise segmentation strategy.

## Pipeline Scripts

```
compress/
├── compress.py                  ← CLI orchestrator (runs all steps)
├── validate_correspondence.py   ← Step 1: verify index-based correspondence
├── fit_motion.py                ← Step 2: polynomial trajectory fitting
├── fit_trbf.py                  ← Step 3: temporal opacity (TRBF)
├── splat4d.py                   ← Step 4: .splat4d format writer/reader
├── validate_quality.py          ← Quality validation vs original PLY
├── view_4d.html                 ← Step 5: Three.js web viewer for .splat4d
├── README.md                    ← This file
└── output/                      ← Generated outputs per clip
    ├── testclip/
    │   ├── correspondence_report.json
    │   ├── fit_result.npz
    │   ├── fit_report.json
    │   ├── trbf_result.npz
    │   ├── trbf_report.json
    │   ├── testclip.splat4d        ← 144 MB (vs 1260 MB flipbook)
    │   ├── quality_report.json
    │   └── view_4d.html
    └── testclip2/
        ├── (same structure)
        └── testclip2.splat4d       ← 144 MB (vs 792 MB flipbook)
```

## Usage

```bash
# Full pipeline (one command):
python compress.py ../reconstruct/output/testclip2

# Or step by step:
python validate_correspondence.py ../reconstruct/output/testclip2
python fit_motion.py ../reconstruct/output/testclip2 --pos-degree 3 --scale-degree 1
python fit_trbf.py ../reconstruct/output/testclip2
python splat4d.py ../reconstruct/output/testclip2

# Validate quality:
python validate_quality.py ../reconstruct/output/testclip2

# Serve viewer:
python compress.py ../reconstruct/output/testclip2 --skip-validate --serve --port 8002
```

## Quality Results (2026-10-31)

### testclip (B&W, static camera, 35 frames) — ✅ NEAR-LOSSLESS

| Metric | Value |
|---|---|
| Position L2 median | **0.045** (0.6% of scene scale) |
| Color RGB L2 median | **0.002** |
| Opacity error median | **0.001** |
| Compression vs .splat | **8.75×** (1260 → 144 MB) |
| Compression vs .ply | **15.31×** (2205 → 144 MB) |

### testclip2 (color dashcam, 22 frames) — ⚠️ LOSSY (needs piecewise segments)

| Metric | Value |
|---|---|
| Position L2 median | **2.23** (2.3% of scene diagonal) |
| Color RGB L2 median | **0.23** |
| Opacity error median | **0.25** |
| Compression vs .splat | **5.5×** (792 → 144 MB) |
| Compression vs .ply | **9.62×** (1386 → 144 MB) |

### Why the Quality Difference?

- **Static camera:** Gaussian[i] images the same world point across all frames. Trajectories are smooth. A single cubic polynomial fits perfectly.
- **Moving camera (dashcam):** Gaussian[i] images whatever is at pixel (r,c) in each frame — a *different* world point as the camera moves. The "trajectory" is non-smooth and scene-dependent. A single polynomial over 7.88 seconds can't capture this.

## .splat4d Binary Format

```
HEADER (64 bytes, little-endian)
  magic           char[4]     "SP4D"
  version         uint16      1
  n_gaussians     uint32
  n_frames        uint32      original frame count
  fps             float32
  t_start         float32     seconds
  t_end           float32     seconds
  t_center        float32     polynomial center time
  pos_degree      uint8       0-3
  color_degree    uint8       0-3
  opacity_degree  uint8       0-3
  scale_degree    uint8       0-3
  rot_degree      uint8       0-3
  flags           uint8       bit 0: has_trbf
  reserved        26 bytes

CANONICAL GAUSSIANS (n × 56 bytes)
  Same layout as SHARP PLY: x,y,z, f_dc_0-2, opacity, scale_0-2, rot_0-3

POSITION MOTION (n × 3 × pos_degree × 4 bytes)
  velocity, acceleration, jerk per axis

COLOR MOTION (n × 3 × color_degree × 4 bytes)
OPACITY MOTION (n × 1 × opacity_degree × 4 bytes)
SCALE MOTION (n × 3 × scale_degree × 4 bytes)
ROTATION MOTION (n × 4 × rot_degree × 4 bytes)

TRBF (n × 8 bytes, if flags.has_trbf)
  tau (float32) + mu (float32) per Gaussian
```

## Pending Work

### P0: Piecewise Segmentation for Moving Camera (IN PROGRESS)

The `--segment-frames` flag was being added to `fit_motion.py` when work was paused.

**Design (from Paper 3):**
- Divide clip into overlapping segments of N frames (recommended: 6-8)
- Fit independent polynomials per segment
- For each original frame, use coefficients from the segment whose center is closest
- This keeps the same .splat4d format (just better coefficients)
- Expected result: position error should drop from 2.23 to ~0.2 for testclip2

**Implementation status:**
- `fit_motion.py`: `--segment-frames` CLI arg added, function signature updated, but the segmented fitting logic in the function body is NOT yet implemented. The `segment_frames` parameter is accepted but not used.
- `splat4d.py`: No changes needed (format stays the same)
- `view_4d.html`: No changes needed (evaluates same polynomial)

**What remains:**
1. In `fit_motion.py`, add segmented fitting logic: loop over windows, fit each, pick best per-frame
2. Re-run on testclip2 with `--segment-frames 6`
3. Re-run quality validation to confirm improvement
4. Update compress.py to auto-detect moving camera and suggest segments

### P1: Web Viewer Testing

- `view_4d.html` is written but has NOT been tested in a browser yet
- Needs bypass-sandbox to serve files over HTTP
- The polynomial evaluation in JS matches the Python `evaluate_at_time()` logic

### P2: Integration with run_pipeline.py

- Wire Phase 4 into the top-level `run_pipeline.py`
- Add `--compress` flag that runs Phase 4 after Phase 3

### P3: AR Android Viewer

- Port polynomial evaluation to GLSL ES 3.1 compute shader in `SplatRenderer.kt`
- Load `.splat4d` instead of per-frame `.splat` files
- This is a separate milestone (AR M3 optimization)

## Key Technical Insights Discovered

1. **SHARP's fixed grid gives FREE correspondence.** Gaussian index IS the correspondence — no optical flow or tracking needed. Validated with coherence ratio 0.007-0.017 across both clips.

2. **Position smoothing (Phase 3 v1) failed because it averaged across unrelated Gaussians.** But polynomial fitting works because it fits EACH Gaussian's trajectory independently over time.

3. **TRBF replaces shadow suppression more principally.** Instead of heuristic thresholds, TRBF identifies transient Gaussians from their temporal opacity profile. testclip: 62.8% stable, 37.2% persistent, 0% transient. testclip2: 0.2% stable, 99.8% persistent.

4. **Static vs moving camera is the quality dividing line.** The same algorithm produces near-lossless quality for static camera and noticeable degradation for moving camera. Piecewise segmentation (pending) should bridge this gap.
