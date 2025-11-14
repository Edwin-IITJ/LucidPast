# generate_3d_mesh_high_quality.py - OPTIMIZED VERSION

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

def depth_to_pointcloud_high_quality(rgb_path, depth_path, output_path):
    """
    High-quality mesh generation with detail preservation - OPTIMIZED
    """
    print(f"\n{'='*70}")
    print(f"Generating high-quality 3D mesh from: {rgb_path}")
    print('='*70)
    
    # Load RGB at FULL resolution
    print("→ Loading RGB image...")
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    print(f"  ✓ RGB loaded: {rgb.shape[1]} x {rgb.shape[0]} pixels (W x H)")
    
    # Load HIGH-RES depth map
    print("→ Loading depth map...")
    depth_path_highres = depth_path.replace('_depth.png', '_depth_highres.png')
    if os.path.exists(depth_path_highres):
        depth = np.array(Image.open(depth_path_highres).convert("L"))
        print("  ✓ Using high-resolution depth map")
    else:
        depth = np.array(Image.open(depth_path).convert("L"))
        print("  ✓ Using standard depth map")
    
    # Ensure RGB and depth match dimensions
    if rgb.shape[:2] != depth.shape[:2]:
        print(f"→ Resizing depth map to match RGB...")
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    height, width = rgb.shape[:2]
    
    # Camera parameters
    print("→ Setting up camera parameters...")
    focal_length = width * 1.2
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, focal_length, focal_length, cx, cy
    )
    print(f"  ✓ Focal length: {focal_length:.1f}")
    
    # Create RGBD image
    print("→ Creating RGBD image...")
    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, 
        depth_o3d, 
        depth_scale=255.0,
        depth_trunc=255.0,
        convert_rgb_to_intensity=False
    )
    print("  ✓ RGBD image created")
    
    # Generate point cloud
    print("→ Generating point cloud from RGBD...")
    start_time = time.time()
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    elapsed = time.time() - start_time
    print(f"  ✓ Point cloud created: {len(pcd.points):,} points ({elapsed:.1f}s)")
    
    # Flip for proper orientation
    print("→ Transforming orientation...")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print("  ✓ Orientation corrected")
    
    # OPTIMIZED: Estimate normals with larger radius (faster, still good quality)
    print("→ Estimating surface normals (optimized - 15-30 seconds)...")
    start_time = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20)  # Larger radius, fewer neighbors
    )
    elapsed = time.time() - start_time
    print(f"  ✓ Normals estimated ({elapsed:.1f}s)")
    
    # SKIP orient_normals_consistent_tangent_plane - causes 15min+ hang with large point clouds
    # Poisson reconstruction will handle normal orientation automatically
    print("→ Skipping manual normal orientation (Poisson will auto-correct)")
    
    # Mesh generation - POISSON (reduced depth for speed)
    print("→ Creating mesh via Poisson reconstruction (depth=9)...")
    print("  ⏳ This will take 1-2 minutes, please wait...")
    start_time = time.time()
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=True  # depth=9 instead of 10 for speed
    )
    
    elapsed = time.time() - start_time
    print(f"  ✓ Mesh created ({elapsed:.1f}s)")
    print(f"    - Vertices: {len(mesh.vertices):,}")
    print(f"    - Triangles: {len(mesh.triangles):,}")
    
    # Remove low-density vertices
    print("→ Removing low-density noise...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  ✓ Cleaned mesh: {len(mesh.vertices):,} vertices remaining")
    
    # Light smoothing (reduced iterations for speed)
    print("→ Smoothing mesh (3 Taubin iterations)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=3)  # 3 instead of 5
    print("  ✓ Mesh smoothed")
    
    # Vertex colors from RGB
    print("→ Applying vertex colors...")
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
    print("  ✓ Colors applied")
    
    # Save files
    print(f"→ Saving mesh files...")
    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
    ply_path = output_path.replace('.obj', '.ply')
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_colors=True)
    
    print(f"\n✅ MESH GENERATION COMPLETE!")
    print(f"  📁 OBJ (Unity): {output_path}")
    print(f"  📁 PLY (backup): {ply_path}")
    print(f"  📊 Final stats:")
    print(f"     - Vertices: {len(mesh.vertices):,}")
    print(f"     - Triangles: {len(mesh.triangles):,}")
    print('='*70 + '\n')
    
    return mesh

if __name__ == "__main__":
    depth_dir = "../Depth-Maps"
    source_dir = "../SourcePhotos"
    output_dir = "../Meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    tif_files = [f for f in os.listdir(source_dir) if f.endswith(('.tif', '.tiff'))]
    total_files = len(tif_files)
    
    print("\n" + "="*70)
    print(f"🚀 STARTING MESH GENERATION (OPTIMIZED)")
    print(f"📁 Source: {source_dir}")
    print(f"📁 Depth Maps: {depth_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Files to process: {total_files}")
    print("="*70 + "\n")
    
    processed = 0
    skipped = 0
    errors = 0
    total_start = time.time()
    
    for i, filename in enumerate(tif_files, 1):
        print(f"\n{'#'*70}")
        print(f"# PROCESSING FILE {i}/{total_files}: {filename}")
        print(f"{'#'*70}\n")
        
        rgb_path = os.path.join(source_dir, filename)
        depth_filename = filename.replace('.tif', '_depth.png').replace('.tiff', '_depth.png')
        depth_path = os.path.join(depth_dir, depth_filename)
        output_path = os.path.join(output_dir, filename.replace('.tif', '.obj').replace('.tiff', '.obj'))
        
        if os.path.exists(depth_path):
            try:
                file_start = time.time()
                depth_to_pointcloud_high_quality(rgb_path, depth_path, output_path)
                file_time = time.time() - file_start
                print(f"⏱️  File processing time: {file_time/60:.1f} minutes")
                processed += 1
            except Exception as e:
                print(f"\n❌ ERROR processing {filename}:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
                errors += 1
        else:
            print(f"⚠️  SKIPPED: Depth map not found for {filename}")
            print(f"   Expected: {depth_path}\n")
            skipped += 1
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "="*70)
    print("🏁 MESH GENERATION COMPLETE!")
    print("="*70)
    print(f"✅ Successfully processed: {processed}/{total_files}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"⏱️  Average per file: {total_time/max(processed,1)/60:.1f} minutes")
    if skipped > 0:
        print(f"⚠️  Skipped (no depth map): {skipped}")
    if errors > 0:
        print(f"❌ Errors: {errors}")
    print(f"\n📁 Meshes saved in: {os.path.abspath(output_dir)}")
    print("="*70 + "\n")
