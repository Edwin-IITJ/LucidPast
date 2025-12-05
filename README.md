> ** Development Note**: This codebase was developed through AI-assisted prototyping using Claude (Anthropic) and Perplexity AI during a 2-week project sprint. Code is functional but follows rapid prototyping practices rather than production-grade architecture.

# LucidPast

Gaze-driven volumetric navigation system for spatial exploration of archival photographs in extended reality.

## Project Overview

LucidPast transforms 2D archival photographs into explorable 3D volumetric environments for VR experiences. Inspired by *Black Mirror: "Eulogy"* episode, this system enables users to navigate historical photographs through natural gaze, creating dream-like exploration of collective visual memory.

### Core Concept

**Input**: Single 2D archival photograph  
**Output**: 3D textured mesh with authentic depth preservation  
**Goal**: Natural attention-driven archival discovery  
**Principle**: Limited parallax maintains historical authenticity without fabricating invisible content

---

## Technical Pipeline

Archival Photograph (TIFF/PNG/JPG)
↓
Depth Estimation (Apple Depth Pro)
↓
3D Point Cloud Generation
↓
Poisson Surface Reconstruction (Open3D)
↓
Textured Mesh Export (OBJ/PLY)
↓
Unity XR Integration (Phase 2)

---

## Repository Structure

LucidPast/
│
├── Scripts/
│ ├── batch_depth_estimation_pro.py # Depth Pro batch processing
│ ├── generate_3d_mesh_depthpro.py # Mesh generation pipeline
│ ├── batch_depth_estimation.py        # DepthAnything V2 depth map generation
│ ├── generate_3d_mesh_textured_gltf.py # DepthAnything V2 Mesh creation
│ └── README.md
│
├── SourcePhotos/                         # Original archival images
├── Meshes/                               # Output 3D meshes
├── Meshes-DepthPro/                      # Output 3D meshes
├── Depth-Maps/                           # Generated depth maps
├── Depth-Maps-Pro/                       # Generated depth maps
├── Textures/                             # PNG texture files
├── Depth-Anything-V2/                    # External model repo (submodule)
├── ml-depth-pro/                         # External model repo (submodule)
│
├── .gitignore
├── requirements.txt
└── README.md

Large binary files (source photos, depth maps, meshes) are excluded via .gitignore.

---

## Prerequisites

**Python**: 3.8 to 3.10  
**GPU**: CUDA compatible (tested on RTX 3060 6GB)  
**RAM**: 16GB minimum  
**OS**: Windows 10/11, macOS, Linux

---

## Installation

### 1. Clone Repository

git clone https://github.com/Edwin-IITJ/LucidPast.git
cd LucidPast

### 2. Set Up Python Environment

python -m venv env
source env/bin/activate # On Windows: env\Scripts\activate
pip install -r requirements.txt

### 3. Install Apple Depth Pro

pip install git+https://github.com/apple/ml-depth-pro.git

### 4. Verify Installation

python -c "import torch; print(torch.cuda.is_available())"
python -c "import open3d; print(open3d.version)"

---

## Usage

### Step 1: Prepare Source Photographs

Place archival photographs in `SourcePhotos/` directory.  
Supported formats: TIFF, PNG, JPG

### Step 2: Generate Depth Maps

cd Scripts
python batch_depth_estimation_pro.py

Output: `DepthMaps-DepthPro/[filename]_depth.png`

### Step 3: Create 3D Meshes

python generate_3d_mesh_depthpro.py

Output:  
`Meshes-DepthPro/[filename].obj` (mesh with UV coordinates)  
`Meshes-DepthPro/[filename].ply` (mesh with vertex colors)

### Step 4: Import to Blender

1. File > Import > Wavefront (.obj)
2. Select mesh from Meshes-DepthPro/ folder
3. Edit Mode > Select All > Mesh > Normals > Recalculate Outside
4. Material Properties > Add Material > Image Texture
5. Link texture from SourcePhotos/ folder

---

## Dependencies

torch>=2.0.0 # PyTorch with CUDA support
torchvision>=0.15.0 # Vision utilities
opencv-python>=4.8.0 # Image processing
numpy>=1.24.0 # Array operations
Pillow>=10.0.0 # Image loading
open3d>=0.18.0 # 3D mesh processing
ml-depth-pro # Apple Depth Pro model

---

## Design Rationale

### Depth Model Selection

**Depth Pro** was selected over Depth Anything V2 after comparison testing.

| Aspect | Depth Anything V2 | Depth Pro |
|--------|-------------------|-----------|
| Facial detail | Soft edges | Sharp boundaries |
| Portrait quality | Adequate | Superior |
| Design priority | Speed | Quality |

**Decision**: Portrait fidelity prioritized over processing speed for human-centered archival documentation.

### Test Dataset Diversity

10 environments from 9 photographs tested across:

**Close-up portraits**: Migrant Mother (two crops), Helen Keller  
**Crowd scenes**: Martin Luther King Jr. gathering  
**Unusual lighting**: Nikola Tesla double exposure  
**Extreme conditions**: Buzz Aldrin lunar surface  
**Interior spaces**: 1940s diner, 1942 radio studio

All conditions generated convincing volumetric reconstructions with proper depth layering.

### Processing Performance

**Hardware**: RTX 3060 6GB, 16GB RAM

**Timing breakdown**:
- Total processing**: 8-9 hours

### Memory Optimization

Images exceeding 35 megapixels automatically downsample while preserving aspect ratio.  
Reduces memory requirements by 40 to 60 percent while maintaining perceptual quality for VR viewing.

---

## Configuration

### Key Parameters (in generate_3d_mesh_depthpro.py)

POISSON_DEPTH = 10 # Surface reconstruction detail level
MAX_MEGAPIXELS = 35 # Automatic downsampling threshold
DEPTH_INVERSION = True # Depth Pro requires inversion

### Ethical Constraints

Users can move head position to perceive depth through parallax, but movement is constrained.  
System prevents viewing angles the original camera never captured.  
No AI fabrication beyond depth inference from visible photograph content.

---

## Interaction Design (Phase 2 Implementation)

### Gaze Navigation

**Dwell time**: 1 second (research-based optimal duration)  
**Visual feedback**: Radial progress indicator  
**Transition**: 3 second dream-like sequence with Gaussian blur and vignette

### Progressive Thematic Narrowing

**Minutes 0 to 7**: Open exploration (algorithm observes patterns)  
**Minutes 7 to 13**: Emerging theme (subtle guidance begins)  
**Minutes 13 to 20**: Narrative clarity (focused sequence)

Algorithm operates invisibly. Users perceive continued free exploration while backend shapes thematic coherence.

### Spatial Voice Annotations

**Gesture**: Pinch near mouth activates recording  
**Duration**: Up to 60 seconds  
**Interface**: YouTube-style comment organization  
**Purpose**: Community interpretation layer without institutional hierarchy

---

## Troubleshooting

### CUDA out of memory

Add `torch.cuda.empty_cache()` after each image in batch processing script.

### Inverted normals in Blender

Edit Mode > Select All > Mesh > Normals > Recalculate Outside

### Depth Pro depth inversion

Ensure `DEPTH_INVERSION = True` in mesh generation script.  
Depth Pro uses inverted depth convention (black = near, white = far).

---

## References

**Depth Pro**: Bochkovskii, A., et al. "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." Apple Machine Learning Research, 2024.

**Depth Anything V2**: Yang, L., et al. "Depth Anything V2." arXiv:2406.09414, 2024.

**Open3D**: Zhou, Q., et al. "Open3D: A Modern Library for 3D Data Processing." arXiv:1801.09847, 2018.

---

## Phase 1 Achievements

**Complete interaction design**: User journey, gaze navigation patterns, voice annotation system, multi-user reflection spaces

**Research-based parameters**: 1 second gaze dwell time, 25 second environment familiarization, 3 second dream transitions

**Technical validation**: 10 diverse archival photographs successfully converted to explorable volumetric environments

**Design framework**: Pathway-dependent interpretation through contextual priming and Kuleshov Effect application

---

## License

**Project License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
- Use this project for any purpose (commercial or non-commercial)
- Modify and build upon this work
- Share and redistribute

**Under the following terms**:
- **Attribution required**: Credit Edwin Meleth and link to this repository

**Recommended Citation**:
LucidPast: Gaze-Driven Volumetric Navigation Through Archival Memory
Edwin Meleth, IIT Jodhpur (2025)
https://github.com/Edwin-IITJ/LucidPast

**Code Components**: MIT License (includes attribution clause)

**Archival Photographs**: Public domain sources (Library of Congress, Wikimedia Commons)  
Original photos not subject to copyright. Attribution to original photographers encouraged.

**Third-Party Dependencies**:
- Apple Depth Pro: Apache 2.0 License
- Open3D: MIT License