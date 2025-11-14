# batch_depth_estimation.py
import cv2
import torch
import numpy as np
import sys
import os
from PIL import Image

sys.path.append('../Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config (CHANGE 'vits' to 'vitl' if you downloaded Large model)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder = 'vitl'  # CHANGE to 'vitl' if using Large model

# Load model
print("Loading Depth Anything V2 model...")
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}")

# Process all photos
source_dir = '../SourcePhotos'
output_dir = '../Depth-Maps'  # UPDATED to match your folder name
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(source_dir)):
    if filename.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print('='*60)
        
        # Load image
        image_path = os.path.join(source_dir, filename)
        raw_img = cv2.imread(image_path)
        
        if raw_img is None:
            print(f"ERROR: Could not load {filename}")
            continue
        
        print(f"Image size: {raw_img.shape[1]} x {raw_img.shape[0]} (W x H)")
        
        # Run depth estimation
        print("Estimating depth...")
        depth = model.infer_image(raw_img)
        
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")
        
        # Save THREE versions:
        base_name = filename.rsplit('.', 1)[0]  # Remove extension
        
        # 1. Standard depth map (normalized 0-255)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_uint8 = depth_normalized.astype(np.uint8)
        output_path_standard = os.path.join(output_dir, f"{base_name}_depth.png")
        cv2.imwrite(output_path_standard, depth_uint8)
        print(f"✓ Saved: {base_name}_depth.png")
        
        # 2. High-res version (same as standard for Depth Anything V2)
        output_path_highres = os.path.join(output_dir, f"{base_name}_depth_highres.png")
        cv2.imwrite(output_path_highres, depth_uint8)
        print(f"✓ Saved: {base_name}_depth_highres.png")
        
        # 3. Raw depth values (float32 for maximum precision)
        output_path_raw = os.path.join(output_dir, f"{base_name}_depth_raw.npy")
        np.save(output_path_raw, depth)
        print(f"✓ Saved: {base_name}_depth_raw.npy")

print(f"\n{'='*60}")
print("All photos processed successfully!")
print(f"Depth maps saved in: {output_dir}")
print('='*60)
