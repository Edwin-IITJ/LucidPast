# LucidPast

Gaze-driven volumetric navigation system for spatial exploration of archival photographs in extended reality.

LucidPast transforms 2D archival photographs into explorable 3D Gaussian Splat environments for VR/XR experiences. Users navigate historical photographs through natural gaze interaction, creating dream-like exploration of collective visual memory.

**Core principle:** Limited parallax maintains historical authenticity — what the photographer did not capture, the system does not invent.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Edwin-IITJ/LucidPast.git
cd LucidPast

# 2. Set up the SHARP environment (requires env_sharp with sharp CLI)
#    See "Environment Setup" below

# 3. Place source photographs in input/
#    Supported formats: TIFF, PNG, JPG, BMP

# 4. Run the pipeline
python scripts/batch_sharp_gaussian.py

# 5. Find your Gaussian Splats in output/gaussians/<image_name>/
```

---

## How It Works

```
Archival Photograph (TIFF/PNG/JPG)
        ↓
Apple SHARP Model (Gaussian Splatting)
        ↓
3D Gaussian Splat (.ply)
        ↓
XR Viewer / Unity Integration
```

The pipeline uses **Apple SHARP** to generate photorealistic Gaussian Splats from single photographs in under 1 second per image. SHARP preserves fine photographic texture — film grain, tonal gradation, edge softness — that traditional polygon mesh pipelines lose during reconstruction.

### Why SHARP?

After testing three approaches on a dataset of 10 diverse archival photographs:

| Approach | Quality | Speed | Ethical Constraint |
|----------|---------|-------|--------------------|
| Depth Anything V2 → Mesh | Soft portrait edges | ~2s depth + ~20min mesh | Manual |
| Apple Depth Pro → Mesh | Sharp boundaries | ~10s depth + ~20min mesh | Manual |
| **Apple SHARP → Splat** | **Photorealistic** | **<1s total** | **Architectural** |

SHARP's limited parallax range naturally prevents fabricating viewpoints the original camera never captured — enforcing the project's ethical constraint architecturally rather than by convention.

> See [`_archive/README.md`](_archive/README.md) for the full pipeline comparison history and archived scripts.

---

## Project Structure

```
LucidPast/
├── scripts/
│   └── batch_sharp_gaussian.py   # Batch SHARP processing pipeline
├── input/                        # Source archival photographs
├── output/
│   └── gaussians/                # Generated Gaussian Splats (one folder per image)
├── ml-sharp/                     # SHARP model repository + checkpoint
├── env_sharp/                    # Python environment with SHARP CLI
├── _archive/                     # Legacy pipeline scripts (Depth Anything V2, Depth Pro)
│   ├── README.md                 # What was archived and why
│   └── scripts/                  # 7 archived scripts from earlier iterations
├── config.yaml                   # Centralized configuration
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

Large binary files (source photos, depth maps, meshes, splats) are excluded via `.gitignore`.

---

## Environment Setup

### Prerequisites

- **Python**: 3.10+
- **GPU**: CUDA-compatible (tested on RTX 3060 6GB)
- **OS**: Windows 10/11

### SHARP Environment

The SHARP pipeline requires a dedicated Python environment with the `sharp` CLI tool:

```bash
# The env_sharp/ directory should contain the SHARP CLI executable at:
# env_sharp/Scripts/sharp.exe (Windows)

# The SHARP model checkpoint should be at:
# ml-sharp/sharp_model.pt
```

> **Note:** The SHARP model and environment are not included in this repository due to size. Contact the repository owner or refer to Apple's SHARP documentation for setup instructions.

---

## Configuration

All settings are centralized in [`config.yaml`](config.yaml):

```yaml
pipeline: sharp

paths:
  input_dir: input                          # Source photographs
  output_dir: output/gaussians              # Splat output
  checkpoint: ml-sharp/sharp_model.pt       # Model weights
  sharp_exe: env_sharp/Scripts/sharp.exe    # SHARP CLI

sharp:
  supported_extensions:
    - .jpg
    - .jpeg
    - .png
    - .tiff
    - .tif
    - .bmp
```

You can also specify an alternate config file:

```bash
python scripts/batch_sharp_gaussian.py --config path/to/custom_config.yaml
```

If `config.yaml` is missing or `pyyaml` is not installed, the script falls back to hardcoded defaults.

---

## Test Dataset

The pipeline has been validated across 10 diverse archival photographs:

| Image | Category | Source |
|-------|----------|--------|
| Migrant Mother (two crops) | Close-up portrait | Library of Congress |
| Helen Keller | Close-up portrait | Library of Congress |
| MLK gathering | Crowd scene | Library of Congress |
| Nikola Tesla | Unusual lighting (double exposure) | Wikimedia Commons |
| Buzz Aldrin | Extreme environment (lunar surface) | NASA |
| 1940s diner (two angles) | Interior architecture | Library of Congress |
| 1942 radio studio | Interior architecture | Library of Congress |
| Gandhi | Group of people | Wikimedia Commons |

All conditions generated convincing volumetric reconstructions with proper depth layering.

---

## Troubleshooting

### SHARP executable not found

Ensure `env_sharp/Scripts/sharp.exe` exists. The path can be customized in `config.yaml`.

### "Already processed" skipping

The script skips images whose output folder already contains files. To reprocess, delete the corresponding subfolder in `output/gaussians/`.

### CUDA out of memory

SHARP is designed to be memory-efficient, but very large images (>50 MP) may cause issues. Consider downsampling source images before processing.

---

## Design History

This project evolved through three technical iterations during a 2-week design sprint:

1. **Depth Anything V2 → Poisson Mesh** (Nov 2025) — Good generalization but soft portrait edges
2. **Apple Depth Pro → Poisson Mesh** (Nov 2025) — Sharp boundaries, but 4-step pipeline
3. **Apple SHARP → Gaussian Splat** (Dec 2025) — Final selection, photorealistic in <1 second

The interaction design (gaze navigation, progressive thematic narrowing, pathway-dependent interpretation) is documented in the project's design report and portfolio case study.

All legacy pipeline scripts are preserved in [`_archive/`](_archive/README.md).

---

## References

**SHARP**: Apple Machine Learning Research, 2025. Single-image to Gaussian Splat reconstruction.

**Depth Pro**: Bochkovskii, A., et al. "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." Apple Machine Learning Research, 2024.

**Depth Anything V2**: Yang, L., et al. "Depth Anything V2." arXiv:2406.09414, 2024.

---

## License

**Project License**: MIT License

Copyright (c) 2025 Edwin Meleth

**Recommended Citation**:
```
LucidPast: Gaze-Driven Volumetric Navigation Through Archival Memory
Edwin Meleth, IIT Jodhpur (2025)
https://github.com/Edwin-IITJ/LucidPast
```

**Archival Photographs**: Public domain sources (Library of Congress, Wikimedia Commons, NASA).

**Third-Party Dependencies**:
- Apple SHARP: Apple ML Research License
- Apple Depth Pro: Apache 2.0 License (archived pipeline)
- Open3D: MIT License (archived pipeline)