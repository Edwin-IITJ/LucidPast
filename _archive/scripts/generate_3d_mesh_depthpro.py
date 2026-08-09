# generate_3d_mesh_batch_depthpro.py - VERSION 1.1 MEMORY-SAFE
# LucidPast - Batch 3D mesh generation with automatic downsampling for large images
# AI-assisted development (Claude + Perplexity AI)

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

# === CONFIGURATION ===
CONFIG = {
    # Poisson Surface Reconstruction
    'poisson_depth': 10,
    'poisson_scale': 1.05,
    'poisson_linear_fit': False,
    
    # Depth Map Preprocessing
    'enable_bilateral_filter': False,
    'invert_depth': True,             # CRITICAL: Depth Pro uses inverted convention
    
    # Memory Management (NEW)
    'max_megapixels': 35,             # Maximum image size (35 million pixels = ~32GB RAM safe)
    'downsample_huge_images': True,   # Auto-downsample if exceeded
    'point_cloud_stride': None,       # If set, skip pixels (e.g., 2 = every other pixel)
    
    # Camera Projection
    'use_depthpro_focal_length': True,
    'focal_length_multiplier': 1.0,
    
    # Normal Correction
    'normal_radius': 0.05,
    'normal_max_neighbors': 30,
    
    # Output
    'save_ply': True,
    'save_obj': True,
}

# === HELPER FUNCTIONS ===

def load_depthpro_focal_length(base_name, depthpro_dir):
    """Load focal length from Depth Pro output file"""
    focal_txt = os.path.join(depthpro_dir, f"{base_name}_focal_length.txt")
    if os.path.exists(focal_txt):
        with open(focal_txt, 'r') as f:
            first_line = f.readline()
            focal_px = float(first_line.split(':')[1].strip().split()[0])
            return focal_px
    return None


def calculate_safe_resolution(width, height, max_megapixels):
    """Calculate downsampled resolution that stays within memory limit"""
    current_megapixels = (width * height) / 1_000_000
    
    if current_megapixels <= max_megapixels:
        return width, height, 1.0  # No downsampling needed
    
    # Calculate scale factor to reach target megapixels
    scale = (max_megapixels / current_megapixels) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return new_width, new_height, scale


def add_camera_projection_uvs(mesh, width, height, focal_length):
    """Generate UV coordinates via camera projection"""
    vertices = np.asarray(mesh.vertices)
    cx, cy = width / 2, height / 2
    z_safe = np.maximum(-vertices[:, 2], 0.001)
    
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (vertices[:, 0] * focal_length / z_safe + cx) / width
    uvs[:, 1] = (vertices[:, 1] * focal_length / z_safe + cy) / height
    uvs = np.clip(uvs, 0.0, 1.0)
    
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = []
    for tri in triangles:
        triangle_uvs.extend([uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh


def fix_inverted_normals(mesh):
    """Fix normals pointing inward"""
    print("   - Fixing inverted normals...")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    corrected_normals = normals.copy()
    fixed_count = 0
    
    for i, vertex in enumerate(vertices):
        [k, idx, _] = tree.search_radius_vector_3d(vertex, CONFIG['normal_radius'])
        if k > 3:
            neighbor_indices = idx[1:min(k, CONFIG['normal_max_neighbors'] + 1)]
            neighbor_positions = vertices[neighbor_indices]
            centroid = np.mean(neighbor_positions, axis=0)
            outward_direction = vertex - centroid
            outward_direction = outward_direction / (np.linalg.norm(outward_direction) + 1e-8)
            
            if np.dot(normals[i], outward_direction) < 0:
                corrected_normals[i] = -normals[i]
                fixed_count += 1
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(corrected_normals)
    print(f"      Fixed {fixed_count}/{len(vertices)} inverted normals")
    return mesh


def cleanup_mesh(mesh):
    """Remove duplicate/degenerate geometry"""
    print("   - Cleaning up mesh...")
    original_verts = len(mesh.vertices)
    original_tris = len(mesh.triangles)
    
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    
    removed_verts = original_verts - len(mesh.vertices)
    removed_tris = original_tris - len(mesh.triangles)
    print(f"      Removed: {removed_verts} vertices, {removed_tris} triangles")
    return mesh


def find_texture_file(base_name, texture_dir):
    """Find texture file (PNG/JPG)"""
    for ext in ['.png', '.jpg', '.jpeg']:
        texture_path = os.path.join(texture_dir, f"{base_name}_texture{ext}")
        if os.path.exists(texture_path):
            try:
                texture_img = Image.open(texture_path)
                print(f"   ✓ Found texture: {os.path.basename(texture_path)}")
                return texture_path, texture_img
            except:
                continue
    return None, None


# === MAIN MESH GENERATION ===

def create_mesh_from_files(depth_path, texture_path, output_path, output_ply_path, base_name, depthpro_dir):
    """Generate 3D mesh with automatic memory management"""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")
    
    # === 1. LOAD DEPTH MAP ===
    print("📥 Loading Depth Pro depth map...")
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    
    original_height, original_width = depth_map.shape[:2]
    original_megapixels = (original_width * original_height) / 1_000_000
    
    print(f"   Original resolution: {original_width} x {original_height} ({original_megapixels:.1f} MP)")
    
    # === 2. CHECK IF DOWNSAMPLING NEEDED ===
    if CONFIG['downsample_huge_images']:
        safe_width, safe_height, scale = calculate_safe_resolution(
            original_width, original_height, CONFIG['max_megapixels']
        )
        
        if scale < 1.0:
            print(f"   ⚠️ Image too large for available RAM!")
            print(f"   📉 Downsampling to: {safe_width} x {safe_height} ({scale*100:.0f}% scale)")
            depth_map = cv2.resize(depth_map, (safe_width, safe_height), interpolation=cv2.INTER_AREA)
            width, height = safe_width, safe_height
        else:
            width, height = original_width, original_height
    else:
        width, height = original_width, original_height
    
    # === 3. NORMALIZE DEPTH ===
    if depth_map.dtype == np.uint16:
        depth_norm = depth_map.astype(np.float32) / 65535.0
    elif depth_map.dtype == np.uint8:
        depth_norm = depth_map.astype(np.float32) / 255.0
    else:
        depth_norm = depth_map.astype(np.float32)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
    
    # === 4. INVERT DEPTH ===
    if CONFIG['invert_depth']:
        print("   🔄 Inverting depth (Depth Pro fix)")
        depth_norm = 1.0 - depth_norm
    
    print(f"   Depth range: {depth_norm.min():.3f} - {depth_norm.max():.3f}")
    
    # === 5. LOAD TEXTURE ===
    print("📥 Loading texture...")
    texture_img = Image.open(texture_path)
    tex_width, tex_height = texture_img.size
    texture_resized = texture_img.resize((width, height), Image.Resampling.LANCZOS)
    texture_array = np.array(texture_resized)
    
    if len(texture_array.shape) == 2:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_GRAY2RGB)
    elif texture_array.shape[2] == 4:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGBA2RGB)
    
    print(f"   Texture: {tex_width} x {tex_height} → {width} x {height}")
    
    # === 6. FOCAL LENGTH ===
    if CONFIG['use_depthpro_focal_length']:
        focal_length_px = load_depthpro_focal_length(base_name, depthpro_dir)
        if focal_length_px:
            # Adjust focal length if we downsampled
            if scale < 1.0:
                focal_length = focal_length_px * scale * CONFIG['focal_length_multiplier']
                print(f"   ✓ Adjusted focal length: {focal_length:.1f}px (downsampled)")
            else:
                focal_length = focal_length_px * CONFIG['focal_length_multiplier']
                print(f"   ✓ Using Depth Pro focal length: {focal_length:.1f}px")
        else:
            focal_length = width * 1.2
            print(f"   Using fallback focal length: {focal_length:.1f}px")
    else:
        focal_length = width * 1.2
    
    # === 7. CREATE POINT CLOUD (MEMORY-EFFICIENT) ===
    print("🔨 Generating point cloud (memory-efficient)...")
    cx, cy = width / 2, height / 2
    
    # Determine stride for huge images
    stride = CONFIG['point_cloud_stride']
    if stride is None:
        # Auto-calculate stride based on size
        if original_megapixels > 50:
            stride = 2  # Skip every other pixel
            print(f"   ⚠️ Using stride={stride} (skipping pixels to save RAM)")
        else:
            stride = 1
    
    points = []
    colors = []
    
    for v in range(0, height, stride):
        for u in range(0, width, stride):
            z = -depth_norm[v, u]
            if z == 0:
                continue
            
            x = (u - cx) * (-z) / focal_length
            y = (v - cy) * (-z) / focal_length
            
            points.append([x, y, z])
            colors.append(texture_array[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"   Point cloud: {len(pcd.points):,} points")
    
    # === 8. ESTIMATE NORMALS ===
    print("   Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    
    # === 9. POISSON RECONSTRUCTION ===
    print(f"🔨 Poisson reconstruction (depth={CONFIG['poisson_depth']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=CONFIG['poisson_depth'],
        width=0,
        scale=CONFIG['poisson_scale'],
        linear_fit=CONFIG['poisson_linear_fit']
    )
    
    print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # === 10-14: CLEANUP, COLORS, NORMALS, UVS, SAVE ===
    mesh = cleanup_mesh(mesh)
    
    print("🎨 Transferring colors...")
    mesh_points = np.asarray(mesh.vertices)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((len(mesh_points), 3))
    
    for i, point in enumerate(mesh_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    print("📐 Computing normals...")
    mesh.compute_vertex_normals()
    mesh = fix_inverted_normals(mesh)
    
    print("📐 Generating UV coordinates...")
    mesh = add_camera_projection_uvs(mesh, width, height, focal_length)
    
    print("💾 Saving outputs...")
    if CONFIG['save_ply']:
        o3d.io.write_triangle_mesh(output_ply_path, mesh)
        print(f"   ✓ Saved PLY: {os.path.basename(output_ply_path)}")
    
    if CONFIG['save_obj']:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"   ✓ Saved OBJ: {os.path.basename(output_path)}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds")
    print(f"{'='*60}\n")


# === BATCH PROCESSING ===

def batch_process():
    """Process all Depth Pro depth maps with memory management"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    depth_dir = os.path.join(project_root, 'Depth-Maps-Pro')
    texture_dir = os.path.join(project_root, 'Textures')
    output_dir = os.path.join(project_root, 'Meshes-DepthPro')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LucidPast - Depth Pro Batch Mesh Generation v1.1")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Poisson Depth: {CONFIG['poisson_depth']}")
    print(f"  Max Image Size: {CONFIG['max_megapixels']} MP (memory limit)")
    print(f"  Auto-Downsample: {'Enabled' if CONFIG['downsample_huge_images'] else 'Disabled'}")
    print(f"  Depth Inversion: Enabled (Depth Pro fix)")
    
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('_depthpro_highres.png')]
    
    if not depth_files:
        print(f"\n❌ No depth maps found in {depth_dir}")
        return
    
    print(f"\nFound {len(depth_files)} depth maps to process\n")
    
    processed = 0
    skipped = 0
    total_start = time.time()
    
    for depth_file in depth_files:
        base_name = depth_file.replace('_depthpro_highres.png', '')
        depth_path = os.path.join(depth_dir, depth_file)
        output_obj = os.path.join(output_dir, f"{base_name}.obj")
        output_ply = os.path.join(output_dir, f"{base_name}.ply")
        
        texture_path, texture_img = find_texture_file(base_name, texture_dir)
        if texture_path is None:
            print(f"\n⚠️ SKIPPED: {base_name} (no texture)")
            skipped += 1
            continue
        
        try:
            create_mesh_from_files(depth_path, texture_path, output_obj, output_ply, base_name, depth_dir)
            processed += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    total_elapsed = time.time() - total_start
    print("\n" + "="*60)
    print("BATCH COMPLETE")
    print("="*60)
    print(f"✅ Processed: {processed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"⏱️ Total: {total_elapsed/60:.1f} min")
    print(f"📁 Output: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    batch_process()
