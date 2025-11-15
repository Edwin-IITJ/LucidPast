# batch_depth_estimation.py - VERSION 3.0 MAXIMUM RESOLUTION
# LucidPast - Batch depth map generation using Depth Anything V2
# AI-assisted development (Claude + Perplexity AI)
#
# v3.0 ENHANCEMENTS:
# - Processes at maximum safe resolution for RTX 3060 6GB
# - Intelligent downscaling only if needed (>2500px)
# - Better memory management
# - Ultra high-res depth maps for maximum detail

import cv2
import torch
import numpy as np
import sys
import os
from PIL import Image

# Add Depth-Anything-V2 to path
sys.path.append('../Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# === CONFIGURATION ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER = 'vitl'  # 'vitl' = Large model (best quality)

# GPU Memory limits (for RTX 3060 6GB)
MAX_RESOLUTION = 2500  # Max pixels on longest edge (safe for 6GB VRAM)
ENABLE_ULTRA_HIGHRES = True  # Process at max resolution (slower but better quality)

# Model configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# === LOAD MODEL ===
print("="*60)
print("LucidPast - Depth Map Generation v3.0")
print("="*60)
print(f"\nLoading Depth Anything V2 ({ENCODER.upper()}) model...")

model = DepthAnythingV2(**model_configs[ENCODER])
model.load_state_dict(
    torch.load(
        f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{ENCODER}.pth',
        map_location='cpu'
    )
)
model = model.to(DEVICE).eval()
print(f"✓ Model loaded on {DEVICE.upper()}")

if ENABLE_ULTRA_HIGHRES:
    print(f"✓ Ultra high-res mode enabled (max {MAX_RESOLUTION}px)")
else:
    print("✓ Standard resolution mode")

# === PROCESS PHOTOS ===
source_dir = '../SourcePhotos'
output_dir = '../Depth-Maps'
os.makedirs(output_dir, exist_ok=True)

# Find all image files
image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
photo_files = [
    f for f in os.listdir(source_dir)
    if os.path.splitext(f.lower())[1] in image_extensions
]

if not photo_files:
    print(f"\n❌ No images found in {source_dir}")
    print(f"   Supported formats: {', '.join(image_extensions)}")
    sys.exit(1)

print(f"\nFound {len(photo_files)} images to process\n")

# Process each image
for i, filename in enumerate(photo_files, 1):
    print(f"[{i}/{len(photo_files)}] Processing: {filename}")
    
    input_path = os.path.join(source_dir, filename)
    base_name = os.path.splitext(filename)[0]
    
    try:
        # === 1. LOAD IMAGE ===
        raw_image = cv2.imread(input_path)
        if raw_image is None:
            print(f"   ❌ Failed to load image")
            continue
        
        original_height, original_width = raw_image.shape[:2]
        print(f"   Original: {original_width} x {original_height} px")
        
        # === 2. PREPARE IMAGE FOR DEPTH ESTIMATION ===
        # Convert BGR to RGB (Depth Anything expects RGB)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        # Check if we need to downscale for GPU memory
        max_edge = max(original_height, original_width)
        needs_downscale = max_edge > MAX_RESOLUTION
        
        if needs_downscale and ENABLE_ULTRA_HIGHRES:
            # Calculate new dimensions maintaining aspect ratio
            scale = MAX_RESOLUTION / max_edge
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            print(f"   Downscaling to {new_width} x {new_height} for GPU memory")
            image_rgb = cv2.resize(
                image_rgb, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LANCZOS4  # High-quality downscaling
            )
            processing_width, processing_height = new_width, new_height
        else:
            processing_width, processing_height = original_width, original_height
            print(f"   Processing at native resolution")
        
        # === 3. ESTIMATE DEPTH ===
        print(f"   Estimating depth (this may take 2-5 seconds)...")
        
        with torch.no_grad():  # Reduce memory usage
            depth = model.infer_image(image_rgb)
        
        # === 4. UPSCALE DEPTH BACK TO ORIGINAL SIZE (if downscaled) ===
        if needs_downscale and ENABLE_ULTRA_HIGHRES:
            print(f"   Upscaling depth map to {original_width} x {original_height}")
            depth = cv2.resize(
                depth, 
                (original_width, original_height), 
                interpolation=cv2.INTER_CUBIC  # Smooth upscaling
            )
        
        # === 5. NORMALIZE AND SAVE ===
        # Normalize to 16-bit for maximum precision
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_16bit = (depth_norm * 65535).astype(np.uint16)
        
        # Save ultra high-res depth map
        output_highres = os.path.join(output_dir, f"{base_name}_depth_highres.png")
        cv2.imwrite(output_highres, depth_16bit)
        print(f"   ✓ Saved: {os.path.basename(output_highres)}")
        
        # Also save standard resolution preview (half size for quick viewing)
        depth_preview = cv2.resize(
            depth_16bit, 
            (original_width // 2, original_height // 2),
            interpolation=cv2.INTER_AREA  # Best for downsampling
        )
        output_preview = os.path.join(output_dir, f"{base_name}_depth.png")
        cv2.imwrite(output_preview, depth_preview)
        print(f"   ✓ Saved preview: {os.path.basename(output_preview)}")
        
        # === 6. CLEAN UP GPU MEMORY ===
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            
            # Print GPU memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to recover GPU memory
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        continue
    
    print()

# === FINAL SUMMARY ===
print("="*60)
print("DEPTH ESTIMATION COMPLETE")
print("="*60)
print(f"✅ Generated depth maps in: {output_dir}")
print(f"\nOutput files per image:")
print(f"  - *_depth_highres.png (full resolution, 16-bit)")
print(f"  - *_depth.png (preview, half resolution)")
print(f"\nNext steps:")
print("1. Convert TIFF photos to PNG: Photoshop → Save As PNG (8-bit)")
print("2. Place textures in: Textures/ folder")
print("3. Run: python generate_3d_mesh_textured_gltf.py")
print("="*60 + "\n")
