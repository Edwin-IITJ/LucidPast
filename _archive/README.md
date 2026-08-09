# LucidPast — Archived Files

These files are from earlier pipeline iterations (Nov–Dec 2025) that were superseded
by the **Apple SHARP Gaussian Splatting** pipeline. They are preserved here for
reference only and are **not part of the active codebase**.

## Why SHARP replaced these pipelines

After testing three approaches on the same 10-image dataset:

1. **Depth Anything V2 → Poisson Mesh** — Soft portrait edges, adequate but not archival-quality.
2. **Apple Depth Pro → Poisson Mesh** — Sharp boundaries, but 4-step pipeline with ~20 min/image.
3. **Apple SHARP → Gaussian Splat** — Photorealistic splats in <1 second, preserves film grain
   and tonal gradation. Naturally enforces the ethical constraint of limited parallax.

SHARP was selected as the final pipeline. See the project documentation for full rationale.

---

## Archived Scripts

### Pipeline A: Depth Anything V2 (abandoned)

| File | Version | Purpose |
|------|---------|---------|
| `scripts/batch_depth_estimation_v1.py` | v1.0 | First draft depth estimation. No error handling, saves redundant copies. |
| `scripts/batch_depth_estimation.py` | v3.0 | Mature Depth Anything V2 estimation. 16-bit output, VRAM-safe downscaling for RTX 3060. |
| `scripts/generate_3d_mesh_textured_gltf_v1.py` | v6.0 | Mesh generation with global centroid normal fix. Hardcoded to 4 test images. |
| `scripts/generate_3d_mesh_textured_gltf.py` | v7.2 | Final Depth Anything V2 mesh gen. KD-tree normal correction, full batch processing. |

### Pipeline B: Apple Depth Pro (abandoned)

| File | Version | Purpose |
|------|---------|---------|
| `scripts/batch_depth_estimation_pro.py` | v1.1 | Depth Pro batch estimation. Outputs 16-bit depth maps + focal length metadata. |
| `scripts/generate_mesh_single_depthpro.py` | — | Single-image test harness (hardcoded to `02_migrant`). Used to debug the depth inversion bug. |
| `scripts/generate_3d_mesh_depthpro.py` | v1.1 | Final Depth Pro mesh gen. Memory-safe with auto-downsampling for images >35 MP. |

---

## Legacy Output Directories

These directories may still exist at the project root. They contain **gigabytes** of
generated binary data and are safe to delete once you have verified you don't need them:

| Directory | Contents | Approx. Size |
|-----------|----------|--------------|
| `Depth-Maps/` | Depth Anything V2 depth maps (16-bit PNG + raw .npy) | ~300 MB |
| `Depth-Maps-Pro/` | Depth Pro depth maps (16-bit PNG + focal length .txt) | ~200 MB |
| `Textures/` | PNG textures converted from source TIFFs (for mesh pipeline) | ~200 MB |
| `Meshes/` | Depth Anything V2 output meshes (OBJ + PLY) | ~2.5 GB |
| `Meshes-DepthPro/` | Depth Pro output meshes (OBJ + PLY) | ~4.5 GB |

## Legacy Dependencies

These directories may also exist and are no longer needed:

| Directory | What it was |
|-----------|-------------|
| `Depth-Anything-V2/` | Cloned external repo for Depth Anything V2 model |
| `ml-depth-pro/` | Cloned external repo for Apple Depth Pro model |
| `env/` | Python virtual environment for Depth Anything V2 |
| `env_depthpro/` | Python virtual environment for Depth Pro |
