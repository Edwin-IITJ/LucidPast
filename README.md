# LucidPast - Volumetric Memory Navigation System

> ** Development Note**: This codebase was developed through AI-assisted prototyping using Claude (Anthropic) and Perplexity AI during a 2-week thesis project sprint. Code is functional but follows rapid prototyping practices rather than production-grade architecture.

## Project Overview

**LucidPast** transforms 2D archival photographs into explorable 3D "memory fragment" environments for VR experiences. Inspired by the *Black Mirror: "Eulogy"* episode aesthetic, this system converts historical photos into volumetric spaces that evoke the dreamlike quality of faded memories.

### Core Concept
- **Input**: Single 2D archival photograph (TIFF/PNG)
- **Output**: 3D textured mesh with depth-based geometry
- **Goal**: Immersive historical exploration, not photorealistic reconstruction
- **Aesthetic**: "Imperfection as authenticity" - mesh artifacts enhance memory fragmentation feel

---

## Technical Pipeline

```
Archival Photo (TIFF/PNG)
    ↓
Depth Estimation (Depth Anything V2 - Large Model)
    ↓
3D Point Cloud Generation
    ↓
Poisson Surface Reconstruction (Open3D)
    ↓
Textured Mesh Export (OBJ/PLY)
    ↓
Blender/Unity Integration
```

---

## Repository Structure

```
LucidPast/
│
├── Scripts/
│   ├── batch_depth_estimation.py        # Batch depth map generation
│   ├── generate_3d_mesh_textured_gltf.py # Mesh creation with textures (v6.0)
│   └── README.md                         # This file
│
├── SourcePhotos/                         # Original archival images (not in repo)
├── Depth-Maps/                           # Generated depth maps (not in repo)
├── Textures/                             # PNG texture files (not in repo)
├── Meshes/                               # Output 3D meshes (not in repo)
├── Depth-Anything-V2/                    # External model repo (submodule)
│
├── .gitignore
├── requirements.txt
└── README.md
```

**Note**: Large binary files (photos, meshes, depth maps) are excluded via `.gitignore`. Only processing scripts are versioned.

---

## Quick Start

### Prerequisites
- **Python**: 3.8 - 3.10 (tested on 3.10)
- **GPU**: CUDA-compatible (tested on RTX 3060 6GB)
- **OS**: Windows 10/11 (paths use Windows format)
- **Software**: Blender 3.6+ (for manual texture fixes), Unity 6 (optional)

### Installation

**1. Clone Repository**
```bash
git clone https://github.com/Edwin-IITJ/LucidPast.git
cd LucidPast
```

**2. Set Up Python Environment**
```bash
python -m venv env
env\Scripts\activate  # Windows
# source env/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

**3. Install Depth Anything V2**
```bash
cd ..
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2

# Download Large model checkpoint
# Visit: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
# Download: depth_anything_v2_vitl.pth
# Place in: Depth-Anything-V2/checkpoints/
```

**4. Verify Installation**
```bash
cd ../LucidPast/Scripts
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
python -c "import open3d; print(open3d.__version__)"        # Should print: 0.18.0+
```

---

## Usage

### Step 1: Prepare Source Photos
```bash
# Place archival photos in SourcePhotos/
# Supported formats: TIFF (8/16-bit), PNG, JPG
# Example: SourcePhotos/01_diner_1940.tif
```

### Step 2: Generate Depth Maps
```bash
cd Scripts
python batch_depth_estimation.py
```
**Output**: `Depth-Maps/[filename]_depth_highres.png`

### Step 3: Convert Textures (Manual)
```bash
# Open TIFF files in Photoshop/GIMP
# Export as: Textures/[filename]_texture.png
# Settings: 8-bit PNG, sRGB color space
```

### Step 4: Create 3D Meshes
```bash
python generate_3d_mesh_textured_gltf.py
```
**Output**: 
- `Meshes/[filename].obj` (OBJ mesh with UVs)
- `Meshes/[filename].ply` (PLY with vertex colors)

### Step 5: Import to Blender
```
1. Open Blender
2. File → Import → Wavefront (.obj)
3. Select mesh from Meshes/ folder
4. Edit Mode → Select All → Mesh → Normals → Recalculate Outside
5. Material Properties → Add Material → Image Texture
6. Link texture from Textures/ folder
7. Viewport Shading → Material Preview
```

---

## Dependencies

### Core Requirements
```
torch>=2.0.0              # PyTorch (CUDA 11.8+)
torchvision>=0.15.0       # Vision utilities
opencv-python>=4.8.0      # Image processing
numpy>=1.24.0             # Array operations
Pillow>=10.0.0            # Image loading
open3d>=0.18.0            # 3D mesh processing
```

### External Models
- **Depth Anything V2** (Large): [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- Model checkpoint: `depth_anything_v2_vitl.pth` (1.3GB)

---

## Design Philosophy

### Aesthetic Principles
1. **Imperfection as Feature**: Mesh artifacts represent memory decay
2. **Dark Void Aesthetic**: Black backgrounds create liminal spaces
3. **Soft Focus Acceptable**: Dreamlike blur over clinical precision
4. **Depth Ambiguity**: Background fade enhances mystery

### Technical Trade-offs
| Issue | Design Reframe |
|-------|----------------|
| Mesh artifacts | "Memory fragmentation" authenticity |
| Inverted normals | Hidden by darkness, eerie atmosphere |
| Vertex color blur | Soft focus memory effect |
| Depth wall ambiguity | Intentional liminal space |

### Interaction Model (Target VR Experience)
- **50cm head lean range**: Reveals parallax depth
- **Gaze-based navigation**: 4-second dwell time transitions
- **Corner studio paradigm**: Small enclosed environments
- **No photorealism**: Emotional resonance over accuracy

---

## 🔧 Configuration

### Key Parameters (in `generate_3d_mesh_textured_gltf.py`)

```python
# Poisson Surface Reconstruction
POISSON_DEPTH = 10        # Octree depth (8-10 recommended)
                          # Higher = more detail, but amplifies noise

# Camera Projection
FOCAL_LENGTH_MULTIPLIER = 1.2  # 1940s standard lens assumption
                               # Adjust if perspective feels wrong

# Normal Correction
NEIGHBOR_RADIUS = 0.05    # Centroid-based normal fix radius
NEIGHBOR_MAX = 30         # Max neighbors for normal averaging
```

### Hardware Optimization
```python
# For GPUs with <8GB VRAM (like RTX 3060 6GB):
# - Use depth_highres.png (not ultra_highres)
# - Poisson depth ≤ 10
# - Process 1 image at a time

# For GPUs with ≥8GB VRAM:
# - Can batch multiple images
# - Poisson depth up to 12
```

---

## 📊 Performance Benchmarks

**Test System**: RTX 3060 6GB, Intel i7-11700K, 32GB RAM

| Task | Time | Output Size |
|------|------|-------------|
| Depth estimation (4000×3000px) | 1-2 sec | 10-15 MB PNG |
| Mesh generation (Poisson depth=10) | 2-5 min | 800K-2M vertices |
| Blender import + texture | 10-30 sec | ~100-300 MB .blend |

**Known Limitations**:
- **High-res photos (>8000px)**: May cause CUDA OOM errors
- **Very dark/bright photos**: Depth estimation less accurate
- **Transparent objects**: Depth maps struggle with glass/reflections

---

## 🐛 Troubleshooting

### Issue: Purple textures in Blender
**Solution**: Texture file path broken. 
```
1. Switch to Shading workspace
2. Image Texture node → Open image
3. Browse to Textures/[filename]_texture.png
```

### Issue: CUDA out of memory
**Solution**: 
```python
# In batch_depth_estimation.py, reduce batch size:
torch.cuda.empty_cache()  # Add after each image
```

### Issue: Inverted normals (red faces)
**Solution**: 
```
Blender Edit Mode → Select All → Mesh → Normals → Recalculate Outside
```

### Issue: Mesh too "blobby"
**Solution**:
```python
# In generate_3d_mesh_textured_gltf.py, try:
POISSON_DEPTH = 9  # Lower depth = smoother (test vs. 10)
```

---

## 📚 References

### Research & Inspiration
- **Depth Anything V2**: [Paper](https://arxiv.org/abs/2406.09414) | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- **Black Mirror "Eulogy"**: Volumetric memory playback aesthetic

---

https://www.loc.gov/pictures/item/2017762905/
https://www.loc.gov/pictures/item/2017837877/

---

## Contributing

This is a project with a fixed 2-week deadline. Contributions are **not actively sought** during development, but post-thesis improvements are welcome:

- **Code optimization**: Performance improvements, GPU memory efficiency
- **Model upgrades**: Integration of newer depth models (Depth Pro, Marigold)
- **Blender automation**: Scripted texture relinking, batch processing
- **Unity XR integration**: Meta Quest 3, Apple Vision Pro support

**Pull Request Guidelines**:
1. Test on Windows + CUDA GPU
2. Maintain existing file structure
3. Document performance impact
4. Follow "imperfection as feature" aesthetic philosophy

---

## License

**Code**: MIT License (see LICENSE file)  
**Archival Photos**: Not included in repository. Original photos sourced from:
- Library of Congress (public domain)

**Attribution**: 
- Depth Anything V2 model © 2024 TikTok Ltd. (Apache 2.0)
- Open3D © Intel Corporation (MIT License)

---

## 🙏 Acknowledgments

- **AI Development**: Claude (Anthropic), Perplexity AI - code generation & debugging
- **Models**: Depth Anything V2 team (TikTok/ByteDance)
- **Libraries**: Open3D, PyTorch, OpenCV communities
- **Inspiration**: Charlie Brooker (*Black Mirror*), Dorothea Lange (photography)
- **Thesis Support**: Dr. Sajan Pillai, Assistant Professor, IIT Jodhpur

---

## Contact

**Developer**: Edwin Meleth  
**Email**: m24ldx008@iitj.ac.in
**Thesis Website**: https://edwinm.vercel.app/  
**Demo Video**: [YouTube/Vimeo link]  

---

*"Memories are not photographs, they are 3D spaces with missing walls, blurred faces, and darkness at the edges. LucidPast embraces this imperfection."*
