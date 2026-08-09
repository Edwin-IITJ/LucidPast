# batch_depth_estimation_pro.py - VERSION 1.1 FIXED PATHS
# LucidPast - High-quality depth map generation using Apple Depth Pro
# AI-assisted development (Claude + Perplexity AI)

import cv2
import torch
import numpy as np
import sys
import os
from PIL import Image

# === CONFIGURATION ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION = torch.float16  # Use half precision for RTX 3060

# Output settings
SAVE_16BIT = True          # Save as 16-bit PNG (maximum precision)
SAVE_PREVIEW = True        # Also save half-resolution preview
SAVE_FOCAL_LENGTH = True   # Save estimated focal length to text file

# === SETUP PATHS (FIXED FOR YOUR FOLDER STRUCTURE) ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # LucidPast folder

# ml-depth-pro is INSIDE LucidPast folder
ml_depth_pro_dir = os.path.join(project_root, 'ml-depth-pro')

print("="*60)
print("LucidPast - Depth Pro Batch Processing v1.1")
print("="*60)
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"ML Depth Pro directory: {ml_depth_pro_dir}")

# Verify ml-depth-pro exists
if not os.path.exists(ml_depth_pro_dir):
    print(f"\n❌ ERROR: ml-depth-pro folder not found at: {ml_depth_pro_dir}")
    print("\nExpected folder structure:")
    print("  LucidPast/")
    print("  ├── Scripts/")
    print("  ├── ml-depth-pro/  ← Should be here")
    print("  ├── SourcePhotos/")
    print("  └── ...")
    sys.exit(1)

# Change working directory to ml-depth-pro (required for checkpoint loading)
original_dir = os.getcwd()
os.chdir(ml_depth_pro_dir)
sys.path.insert(0, os.path.join(ml_depth_pro_dir, 'src'))

print(f"✓ Working directory changed to: {os.getcwd()}")

# === LOAD MODEL ===
print(f"\nLoading Depth Pro model...")

try:
    from depth_pro import create_model_and_transforms
    
    model, transform = create_model_and_transforms(
        device=torch.device(DEVICE),
        precision=PRECISION
    )
    model.eval()
    print(f"✓ Model loaded on {DEVICE.upper()}")
    
    if DEVICE == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {vram:.1f} GB")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    os.chdir(original_dir)
    sys.exit(1)

# === PROCESS PHOTOS ===
os.chdir(original_dir)  # Return to original directory

source_dir = os.path.join(project_root, 'SourcePhotos')
output_dir = os.path.join(project_root, 'Depth-Maps-Pro')
os.makedirs(output_dir, exist_ok=True)

# Verify source directory exists
if not os.path.exists(source_dir):
    print(f"\n❌ ERROR: SourcePhotos folder not found at: {source_dir}")
    sys.exit(1)

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

print(f"\nFound {len(photo_files)} images to process")
print(f"Output directory: {output_dir}\n")

# === PROCESS EACH IMAGE ===
processed = 0
failed = 0

for i, filename in enumerate(photo_files, 1):
    print(f"[{i}/{len(photo_files)}] Processing: {filename}")
    
    input_path = os.path.join(source_dir, filename)
    base_name = os.path.splitext(filename)[0]
    
    try:
        # === 1. LOAD IMAGE ===
        raw_image = cv2.imread(input_path)
        if raw_image is None:
            print(f"   ❌ Failed to load image")
            failed += 1
            continue
        
        # Convert BGR to RGB (Depth Pro expects RGB)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        height, width = raw_image.shape[:2]
        print(f"   Resolution: {width} x {height} px")
        
        # === 2. ESTIMATE DEPTH ===
        print(f"   Estimating depth with Depth Pro (5-10 seconds)...")
        
        # Transform image
        image_tensor = transform(pil_image)
        
        # Run inference
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=None)
        
        # Extract results
        depth = prediction["depth"].cpu().numpy()
        focal_length_px = prediction["focallength_px"]
        
        print(f"   ✓ Depth estimated (focal length: {focal_length_px:.1f}px)")
        print(f"   Depth range: {depth.min():.3f}m - {depth.max():.3f}m")
        
        # === 3. RESIZE DEPTH TO MATCH ORIGINAL IMAGE ===
        if depth.shape[0] != height or depth.shape[1] != width:
            depth_resized = cv2.resize(
                depth, 
                (width, height), 
                interpolation=cv2.INTER_CUBIC
            )
            print(f"   Upscaled depth to {width} x {height}")
        else:
            depth_resized = depth
        
        # === 4. NORMALIZE TO 16-BIT ===
        depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
        depth_16bit = (depth_norm * 65535).astype(np.uint16)
        
        # === 5. SAVE HIGH-RES DEPTH MAP ===
        if SAVE_16BIT:
            output_highres = os.path.join(output_dir, f"{base_name}_depthpro_highres.png")
            cv2.imwrite(output_highres, depth_16bit)
            print(f"   ✓ Saved: {os.path.basename(output_highres)}")
        
        # === 6. SAVE PREVIEW ===
        if SAVE_PREVIEW:
            depth_preview = cv2.resize(
                depth_16bit, 
                (width // 2, height // 2),
                interpolation=cv2.INTER_AREA
            )
            output_preview = os.path.join(output_dir, f"{base_name}_depthpro.png")
            cv2.imwrite(output_preview, depth_preview)
            print(f"   ✓ Saved preview: {os.path.basename(output_preview)}")
        
        # === 7. SAVE FOCAL LENGTH ===
        if SAVE_FOCAL_LENGTH:
            focal_txt = os.path.join(output_dir, f"{base_name}_focal_length.txt")
            with open(focal_txt, 'w') as f:
                f.write(f"Estimated Focal Length: {focal_length_px:.2f} pixels\n")
                f.write(f"Image Resolution: {width} x {height}\n")
                f.write(f"Depth Range: {depth_resized.min():.3f}m - {depth_resized.max():.3f}m\n")
            print(f"   ✓ Saved focal length: {os.path.basename(focal_txt)}")
        
        # === 8. CLEAN UP GPU MEMORY ===
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated")
        
        processed += 1
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        failed += 1
        continue
    
    print()

# === FINAL SUMMARY ===
print("="*60)
print("DEPTH PRO BATCH PROCESSING COMPLETE")
print("="*60)
print(f"✅ Processed: {processed}")
print(f"❌ Failed: {failed}")
print(f"📁 Output folder: {output_dir}")
print(f"\nOutput files per image:")
print(f"  - *_depthpro_highres.png (full resolution, 16-bit)")
print(f"  - *_depthpro.png (preview, half resolution)")
print(f"  - *_focal_length.txt (camera parameters)")
print(f"\nNext steps:")
print(f"1. Compare depth maps visually:")
print(f"   - Depth-Maps/ (Depth Anything V2)")
print(f"   - Depth-Maps-Pro/ (Depth Pro) ← NEW")
print(f"2. Choose better depth maps")
print(f"3. Regenerate meshes with chosen depth maps")
print("="*60 + "\n")
