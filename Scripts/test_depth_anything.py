# Scripts/test_depth_anything.py
import cv2
import torch
import sys
sys.path.append('../Depth-Anything-V2')  # Add repo to path

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configs
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Use 'vits' (Small) or 'vitl' (Large)
encoder = 'vitl'  # Start with Large

# Load model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Test on one of your photos
raw_img = cv2.imread('../SourcePhotos/01_diner_1940.tif')
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# Save depth map
import numpy as np
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_uint8 = depth_normalized.astype(np.uint8)
cv2.imwrite('../Depth-Maps/test_depth.png', depth_uint8)

print(f"Depth map saved to Depth-Maps/test_depth.png")
print(f"Depth shape: {depth.shape}")
print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")
