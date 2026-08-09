# generate_3d_mesh_textured_gltf.py - VERSION 7.2 FIXED
# LucidPast - Volumetric Memory Navigation System
# AI-assisted development (Claude + Perplexity AI)
#
# FIX v7.2: Compute normals BEFORE fix_inverted_normals()
# FIX v7.1: Removed invalid mesh.remove_statistical_outlier()

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import os
import time

# === CONFIGURATION ===
CONFIG = {
    # Poisson Surface Reconstruction
    'poisson_depth': 10,              # Octree depth (8=smooth, 9=balanced, 10=detailed)
    'poisson_scale': 1.1,            # Surface expansion (1.0-1.2)
    'poisson_linear_fit': False,      # Better for dense point clouds
    
    # Depth Map Preprocessing
    'enable_bilateral_filter': False,  # Raw depth, maximum detail
    'bilateral_diameter': 5,          # Filter size (5-9)
    'bilateral_sigma_color': 50,      # Color similarity (30-80)
    'bilateral_sigma_space': 50,      # Spatial proximity (30-80)
    
    # Camera Projection
    'focal_length_multiplier': 1.2,   # 1940s standard lens assumption
    
    # Normal Correction
    'normal_radius': 0.05,            # Neighbor search radius
    'normal_max_neighbors': 30,       # Max neighbors for averaging
    
    # Output
    'save_ply': True,                 # Save PLY (with vertex colors)
    'save_obj': True,                 # Save OBJ (with UVs for textures)
}

# === DEPTH MAP ENHANCEMENT ===
def enhance_depth_map(depth_array):
    """
    Enhance depth map quality with edge-preserving smoothing
    Returns: Enhanced depth array (normalized 0-1)
    """
    print("   - Enhancing depth map quality...")
    
    # Normalize to 0-1 range
    if depth_array.dtype == np.uint16:
        depth_norm = depth_array.astype(np.float32) / 65535.0
    elif depth_array.dtype == np.uint8:
        depth_norm = depth_array.astype(np.float32) / 255.0
    else:
        depth_norm = depth_array.astype(np.float32)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
    
    if not CONFIG['enable_bilateral_filter']:
        return depth_norm
    
    # Convert to uint8 for bilateral filter
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Bilateral filter - preserves edges while smoothing
    depth_filtered = cv2.bilateralFilter(
        depth_uint8,
        d=CONFIG['bilateral_diameter'],
        sigmaColor=CONFIG['bilateral_sigma_color'],
        sigmaSpace=CONFIG['bilateral_sigma_space']
    )
    
    # Median blur to remove spikes
    depth_filtered = cv2.medianBlur(depth_filtered, 5)
    
    # Back to float32 [0, 1]
    depth_enhanced = depth_filtered.astype(np.float32) / 255.0
    
    print(f"      Depth range: {depth_enhanced.min():.3f} - {depth_enhanced.max():.3f}")
    return depth_enhanced


# === UV GENERATION ===
def add_camera_projection_uvs(mesh, width, height, focal_length):
    """Generate UV coordinates via camera projection"""
    vertices = np.asarray(mesh.vertices)
    cx, cy = width / 2, height / 2
    
    # Prevent division by zero
    z_safe = np.maximum(-vertices[:, 2], 0.001)
    
    # Project 3D vertices to 2D UV space
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (vertices[:, 0] * focal_length / z_safe + cx) / width
    uvs[:, 1] = (vertices[:, 1] * focal_length / z_safe + cy) / height
    uvs = np.clip(uvs, 0.0, 1.0)
    
    # Assign UVs to triangle vertices
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = []
    for tri in triangles:
        triangle_uvs.extend([uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]])
    
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh


# === NORMAL CORRECTION ===
def fix_inverted_normals(mesh):
    """
    Fix normals pointing inward using centroid-based method
    IMPORTANT: Must call mesh.compute_vertex_normals() BEFORE this function
    """
    print("   - Fixing inverted normals...")
    
    # Check if normals exist
    if not mesh.has_vertex_normals():
        print("      ⚠️ No normals found, computing first...")
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # Build KD-tree for neighbor search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    corrected_normals = normals.copy()
    fixed_count = 0
    
    for i, vertex in enumerate(vertices):
        # Find neighbors
        [k, idx, _] = tree.search_radius_vector_3d(
            vertex, 
            CONFIG['normal_radius']
        )
        
        if k > 3:  # Need at least 3 neighbors
            # Limit neighbors for performance
            neighbor_indices = idx[1:min(k, CONFIG['normal_max_neighbors'] + 1)]
            neighbor_positions = vertices[neighbor_indices]
            
            # Calculate centroid
            centroid = np.mean(neighbor_positions, axis=0)
            
            # Vector from centroid to vertex (should point outward)
            outward_direction = vertex - centroid
            outward_direction = outward_direction / (np.linalg.norm(outward_direction) + 1e-8)
            
            # If normal points inward (dot product < 0), flip it
            if np.dot(normals[i], outward_direction) < 0:
                corrected_normals[i] = -normals[i]
                fixed_count += 1
    
    mesh.vertex_normals = o3d.utility.Vector3dVector(corrected_normals)
    print(f"      Fixed {fixed_count}/{len(vertices)} inverted normals")
    return mesh


# === MESH CLEANUP ===
def cleanup_mesh(mesh):
    """Remove duplicate/degenerate geometry"""
    print("   - Cleaning up mesh...")
    
    original_verts = len(mesh.vertices)
    original_tris = len(mesh.triangles)
    
    # Remove duplicate vertices
    mesh.remove_duplicated_vertices()
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    removed_verts = original_verts - len(mesh.vertices)
    removed_tris = original_tris - len(mesh.triangles)
    
    print(f"      Removed: {removed_verts} vertices, {removed_tris} triangles")
    return mesh


# === TEXTURE LOADING ===
def find_texture_file(base_name, texture_dir):
    """
    Find texture file with flexible format detection (PNG/JPG)
    Returns: (texture_path, texture_image) or (None, None)
    """
    for ext in ['.png', '.jpg', '.jpeg']:
        texture_path = os.path.join(texture_dir, f"{base_name}_texture{ext}")
        if os.path.exists(texture_path):
            try:
                texture_img = Image.open(texture_path)
                print(f"   ✓ Found texture: {os.path.basename(texture_path)}")
                return texture_path, texture_img
            except Exception as e:
                print(f"   ⚠️ Error loading {texture_path}: {e}")
                continue
    
    return None, None


# === MAIN MESH GENERATION ===
def create_mesh_from_files(depth_path, texture_path, output_path, output_ply_path):
    """
    Generate 3D mesh from depth map + texture
    """
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")
    
    # === 1. LOAD DEPTH MAP ===
    print("📥 Loading depth map...")
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    
    height, width = depth_map.shape[:2]
    print(f"   Resolution: {width} x {height}")
    
    # === 2. ENHANCE DEPTH MAP ===
    depth_enhanced = enhance_depth_map(depth_map)
    
    # === 3. LOAD TEXTURE ===
    print("📥 Loading texture...")
    texture_img = Image.open(texture_path)
    tex_width, tex_height = texture_img.size
    texture_resized = texture_img.resize((width, height), Image.Resampling.LANCZOS)
    texture_array = np.array(texture_resized)
    
    # Convert grayscale to RGB
    if len(texture_array.shape) == 2:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_GRAY2RGB)
    elif texture_array.shape[2] == 4:
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGBA2RGB)
    
    print(f"   Texture: {tex_width} x {tex_height} → {width} x {height}")
    
    # === 4. CREATE POINT CLOUD ===
    print("🔨 Generating point cloud...")
    focal_length = width * CONFIG['focal_length_multiplier']
    cx, cy = width / 2, height / 2
    
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            z = -depth_enhanced[v, u]  # Negative Z (camera looks down -Z axis)
            if z == 0:
                continue
            
            # Back-project to 3D
            x = (u - cx) * (-z) / focal_length
            y = (v - cy) * (-z) / focal_length
            
            points.append([x, y, z])
            colors.append(texture_array[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    
    print(f"   Point cloud: {len(pcd.points):,} points")
    
    # === 5. POISSON SURFACE RECONSTRUCTION ===
    print(f"🔨 Poisson reconstruction (depth={CONFIG['poisson_depth']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=CONFIG['poisson_depth'],
        width=0,
        scale=CONFIG['poisson_scale'],
        linear_fit=CONFIG['poisson_linear_fit']
    )
    
    print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # === 6. CLEAN UP MESH ===
    mesh = cleanup_mesh(mesh)
    
    # === 7. TRANSFER COLORS FROM POINT CLOUD ===
    print("🎨 Transferring colors...")
    mesh_points = np.asarray(mesh.vertices)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = np.zeros((len(mesh_points), 3))
    
    for i, point in enumerate(mesh_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # === 8. COMPUTE NORMALS (MUST BE BEFORE fix_inverted_normals) ===
    print("📐 Computing normals...")
    mesh.compute_vertex_normals()
    
    # === 9. FIX INVERTED NORMALS ===
    mesh = fix_inverted_normals(mesh)
    
    # === 10. ADD UV COORDINATES ===
    print("📐 Generating UV coordinates...")
    mesh = add_camera_projection_uvs(mesh, width, height, focal_length)
    
    # === 11. SAVE OUTPUTS ===
    print("💾 Saving outputs...")
    
    if CONFIG['save_ply']:
        o3d.io.write_triangle_mesh(output_ply_path, mesh)
        print(f"   ✓ Saved PLY: {os.path.basename(output_ply_path)}")
    
    if CONFIG['save_obj']:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"   ✓ Saved OBJ: {os.path.basename(output_path)}")
    
    # === SUMMARY ===
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds")
    print(f"   Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"{'='*60}\n")


# === BATCH PROCESSING ===
def batch_process():
    """
    Process all photos in SourcePhotos/ folder
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    depth_dir = os.path.join(project_root, 'Depth-Maps')
    texture_dir = os.path.join(project_root, 'Textures')
    output_dir = os.path.join(project_root, 'Meshes')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LucidPast - 3D Mesh Generation v7.2")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Poisson Depth: {CONFIG['poisson_depth']}")
    print(f"  Bilateral Filter: {'Enabled' if CONFIG['enable_bilateral_filter'] else 'Disabled'}")
    print(f"  Focal Length Multiplier: {CONFIG['focal_length_multiplier']}")
    print(f"  Output Formats: {'PLY ' if CONFIG['save_ply'] else ''}{'OBJ' if CONFIG['save_obj'] else ''}")
    
    # Find all high-res depth maps
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('_depth_highres.png')]
    
    if not depth_files:
        print(f"\n❌ No *_depth_highres.png files found in {depth_dir}")
        return
    
    print(f"\nFound {len(depth_files)} depth maps to process\n")
    
    processed = 0
    skipped = 0
    
    for depth_file in depth_files:
        # Extract base name (e.g., "01_diner_1940" from "01_diner_1940_depth_highres.png")
        base_name = depth_file.replace('_depth_highres.png', '')
        
        depth_path = os.path.join(depth_dir, depth_file)
        output_obj = os.path.join(output_dir, f"{base_name}.obj")
        output_ply = os.path.join(output_dir, f"{base_name}.ply")
        
        # Find texture file (PNG/JPG)
        texture_path, texture_img = find_texture_file(base_name, texture_dir)
        
        if texture_path is None:
            print(f"\n⚠️ SKIPPED: {base_name}")
            print(f"   Reason: No texture file found in {texture_dir}")
            print(f"   Looking for: {base_name}_texture.png or .jpg\n")
            skipped += 1
            continue
        
        try:
            create_mesh_from_files(depth_path, texture_path, output_obj, output_ply)
            processed += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {base_name}: {e}\n")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Processed: {processed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"📁 Output folder: {output_dir}")
    print("\nNext steps:")
    print("1. Open Blender")
    print("2. Import OBJ files from Meshes/ folder")
    print("3. Fix normals: Edit Mode → Mesh → Normals → Recalculate Outside")
    print("4. Link textures: Material Properties → Image Texture → Open texture PNG")
    print("="*60 + "\n")


# === ENTRY POINT ===
if __name__ == "__main__":
    batch_process()
